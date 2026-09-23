"""Fast held-out evaluation helpers for Stage-2 analysis (DataLoader-based)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage1_mimic_pretrain.metrics import multilabel_metrics, ranking_per_example
from stage2_nch.analysis import ANALYSIS
from stage2_nch.analysis.eval_protocol import (
    HORIZON_LABELS,
    HORIZONS_DAYS,
    build_model_from_ckpt,
    history_bin_span,
    patient_bootstrap_ci,
)
from stage2_nch.config import EVAL_KS, VOCAB_PATH, age_band_name
from stage2_nch.dataset import NCHForecastDataset, make_nch_collate
from stage2_nch.evaluate import evaluate_loader

BOOT_SEED = 0
N_BOOT = 500


def _apply_horizon_mask(batch: dict, horizon_days: float | None) -> dict:
    """Drop events older than horizon (relative to last event) and left-align.

    Stage-1/2 batches require a contiguous True prefix in ``attention_mask`` matching
    ``lengths``. Simply zeroing early positions would create holes / right-aligned masks.
    """
    if horizon_days is None:
        return batch
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    ts = out["timestamps_days"]
    mask = out["attention_mask"].bool()
    codes = out["code_indices"]
    ages = out["age_years"]
    b, lmax = ts.shape
    lengths = mask.sum(dim=1)
    device = ts.device

    new_mask = torch.zeros_like(mask)
    new_ts = torch.zeros_like(ts)
    new_codes = torch.zeros_like(codes)
    new_ages = torch.zeros_like(ages)
    new_demo = out["demographics"].clone() if "demographics" in out else None

    for i in range(b):
        n = int(lengths[i].item())
        if n <= 0:
            continue
        t = ts[i, :n]
        t_last = t[-1]
        keep = (t_last - t) <= float(horizon_days)
        keep[-1] = True
        idx = torch.where(keep)[0]
        n2 = int(idx.numel())
        new_mask[i, :n2] = True
        new_ts[i, :n2] = t[idx]
        # Re-base timestamps so first kept event is 0 (matches Stage-1 convention)
        new_ts[i, :n2] = new_ts[i, :n2] - new_ts[i, 0]
        new_codes[i, :n2] = codes[i, idx]
        new_ages[i, :n2] = ages[i, idx]
        if new_demo is not None:
            new_demo[i, :n2] = out["demographics"][i, idx]
            new_demo[i, n2:] = 0
            new_demo[i, :n2, 0] = new_ages[i, :n2]

    out["attention_mask"] = new_mask
    out["timestamps_days"] = new_ts
    out["code_indices"] = new_codes
    out["age_years"] = new_ages
    out["lengths"] = new_mask.sum(dim=1).long()
    if new_demo is not None:
        out["demographics"] = new_demo
    return out


def _permute_ages(batch: dict, rng: np.random.Generator) -> dict:
    out = dict(batch)
    ages = batch["age_years"].clone()
    mask = batch["attention_mask"].bool()
    last = batch["last_age_years"].cpu().numpy().copy()
    rng.shuffle(last)
    last_t = torch.tensor(last, device=ages.device, dtype=ages.dtype)
    # broadcast last age onto valid positions
    ages = torch.zeros_like(ages)
    ages = ages.masked_fill(mask, 0)  # placeholder
    ages = last_t.unsqueeze(1) * mask.float()
    out["age_years"] = ages
    # demographics channel 0 is age
    if "demographics" in out:
        demo = out["demographics"].clone()
        demo[..., 0] = ages
        out["demographics"] = demo
    out["last_age_years"] = last_t
    return out


def _constant_ages(batch: dict, age: float) -> dict:
    out = dict(batch)
    mask = batch["attention_mask"].bool()
    ages = mask.float() * float(age)
    out["age_years"] = ages
    if "demographics" in out:
        demo = out["demographics"].clone()
        demo[..., 0] = ages
        out["demographics"] = demo
    out["last_age_years"] = torch.full_like(batch["last_age_years"], float(age))
    return out


@torch.no_grad()
def eval_windows(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    horizon_days: float | None = None,
    age_mode: str = "natural",
    constant_age: float = 9.0,
    permute_seed: int = 0,
    store_logits: bool = False,
    logit_cap: int = 8000,
    max_batches: int = 0,
) -> tuple[pd.DataFrame, torch.Tensor | None, torch.Tensor | None]:
    model.eval()
    rng = np.random.default_rng(permute_seed)
    rows = []
    logit_chunks, target_chunks = [], []
    n_stored = 0
    for bi, batch in enumerate(loader, 1):
        if max_batches and bi > max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        natural_age = batch["last_age_years"].detach().cpu().numpy()
        if age_mode == "permute":
            batch = _permute_ages(batch, rng)
        elif age_mode == "constant":
            batch = _constant_ages(batch, constant_age)
        batch = _apply_horizon_mask(batch, horizon_days)

        out = model(batch)
        logits = out["code_logits"].float()
        targets = batch["target_codes"].float()
        per_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(-1)
        rank = ranking_per_example(logits, targets, ks=EVAL_KS)
        probs = torch.sigmoid(logits)
        brier = ((probs - targets) ** 2).mean(-1)

        # history span from original timestamps among kept keys
        ts = batch["timestamps_days"]
        mask = batch["attention_mask"].bool()
        for j in range(logits.shape[0]):
            m = mask[j]
            t = ts[j][m]
            span = float((t[-1] - t[0]).cpu()) if m.any() else 0.0
            age = float(natural_age[j])
            row = {
                "patient_id": int(batch["patient_id"][j].cpu()),
                "age_years_natural": age,
                "age_band": age_band_name(age),
                "n_input_events": int(m.sum().cpu()),
                "history_span_days": span,
                "history_bin": history_bin_span(span),
                "n_prior_visits": int(batch["n_prior_visits"][j].cpu()),
                "bce": float(per_bce[j].cpu()),
                "brier": float(brier[j].cpu()),
                "n_true": float(rank["n_true"][j]),
            }
            for k in EVAL_KS:
                row[f"recall@{k}"] = float(rank[f"recall@{k}"][j])
                row[f"precision@{k}"] = float(rank[f"precision@{k}"][j])
            rows.append(row)

        if store_logits and n_stored < logit_cap:
            take = min(logits.shape[0], logit_cap - n_stored)
            logit_chunks.append(logits[:take].detach().cpu())
            target_chunks.append(targets[:take].detach().cpu())
            n_stored += take

    df = pd.DataFrame(rows)
    df["example_idx"] = np.arange(len(df))
    logits_cat = torch.cat(logit_chunks) if logit_chunks else None
    targets_cat = torch.cat(target_chunks) if target_chunks else None
    return df, logits_cat, targets_cat


def run_fast_eval(
    *,
    ckpt: Path,
    tensorized: Path,
    device: str = "cuda:0",
    batch_size: int = 8,
    out_dir: Path = ANALYSIS,
    do_truncation: bool = True,
    do_ablations: bool = True,
    resume: bool = False,
) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device_t = torch.device(device)
    model = build_model_from_ckpt(ckpt, device_t)
    ds = NCHForecastDataset(tensorized / "test", VOCAB_PATH, max_seq_len=1024)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2,
        collate_fn=make_nch_collate(assert_horizon=False), pin_memory=True,
    )
    fig_dir, tab_dir, pred_dir, raw_dir = (out_dir / x for x in
                                          ("figures", "tables", "predictions", "raw"))
    for d in (fig_dir, tab_dir, pred_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[fast] natural eval on {device}  n={len(ds)}", flush=True)
    full_path = pred_dir / "adkm_windows_full.parquet"
    if resume and full_path.exists():
        print("[fast] resume: loading existing full-history predictions", flush=True)
        df = pd.read_parquet(full_path)
        logits = targets = None
        overall = {m: patient_bootstrap_ci(df, m) for m in
                   ("bce", "brier", "recall@5", "recall@20", "precision@5")}
        (raw_dir / "adkm_overall_fast.json").write_text(json.dumps(overall, indent=2) + "\n")
        print("[fast] natural metrics done (resumed)", flush=True)
    else:
        df, logits, targets = eval_windows(
            model, loader, device_t, store_logits=True, logit_cap=8000)
        df.to_parquet(full_path, index=False)

        overall = {m: patient_bootstrap_ci(df, m) for m in
                   ("bce", "brier", "recall@5", "recall@20", "precision@5")}
        # Point multilabel metrics on a capped set (full patient-bootstrap of AUROC/AUPRC is
        # O(n_boot · n · C) and too heavy for |V|=30k; ranking metrics carry the paper CI).
        if logits is not None:
            ml = multilabel_metrics(logits, targets, ks=EVAL_KS)
            overall["multilabel_point_on_cap"] = {
                "micro_auprc": ml["micro_auprc"],
                "micro_auroc": ml["micro_auroc"],
                "macro_auprc": ml.get("macro_auprc"),
                "macro_auroc": ml.get("macro_auroc"),
                "n_windows_in_cap": int(logits.shape[0]),
                "note": "Point estimate on logit cap; patient-bootstrap CIs reported for ranking/BCE",
            }
        (raw_dir / "adkm_overall_fast.json").write_text(json.dumps(overall, indent=2) + "\n")
        print("[fast] natural metrics done", flush=True)

    # age table + figure
    age_rows = []
    for band in ["<1", "1-5", "6-11", "12-17"]:
        sub = df[df["age_band"] == band]
        if sub.empty:
            continue
        rec = {"age_band": band, "n_windows": len(sub),
               "n_patients": int(sub["patient_id"].nunique())}
        for m in ("bce", "brier", "recall@5", "recall@20", "precision@5"):
            ci = patient_bootstrap_ci(sub, m)
            rec[m] = ci["point"]
            rec[f"{m}_ci_lo"] = ci["ci_lo"]
            rec[f"{m}_ci_hi"] = ci["ci_hi"]
        age_rows.append(rec)
    age_tab = pd.DataFrame(age_rows)
    age_tab.to_csv(tab_dir / "adkm_performance_by_age_boot.csv", index=False)

    def savefig(fig, stem):
        fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(age_tab["age_band"], age_tab["recall@5"], color="#1f4e79",
                yerr=[age_tab["recall@5"] - age_tab["recall@5_ci_lo"],
                      age_tab["recall@5_ci_hi"] - age_tab["recall@5"]], capsize=3)
    axes[0].set_title("Recall@5 by age (patient bootstrap CI)")
    axes[1].bar(age_tab["age_band"], age_tab["bce"], color="#1f4e79",
                yerr=[age_tab["bce"] - age_tab["bce_ci_lo"],
                      age_tab["bce_ci_hi"] - age_tab["bce"]], capsize=3)
    axes[1].set_title("BCE by age")
    for ax in axes:
        ax.set_xlabel("Age band")
    fig.tight_layout()
    savefig(fig, "fig_performance_by_age")

    # truncation
    trunc_rows = []
    if do_truncation:
        for horizon, label in zip(HORIZONS_DAYS, HORIZON_LABELS):
            print(f"[fast] truncation {label}", flush=True)
            dft, _, _ = eval_windows(model, loader, device_t, horizon_days=horizon)
            dft.to_parquet(pred_dir / f"adkm_windows_horizon_{label}.parquet", index=False)
            ci = patient_bootstrap_ci(dft, "recall@5")
            row = {"horizon": label, "horizon_days": horizon,
                   "recall@5": ci["point"], "recall@5_ci_lo": ci["ci_lo"],
                   "recall@5_ci_hi": ci["ci_hi"],
                   "bce": patient_bootstrap_ci(dft, "bce")["point"],
                   "n_patients": ci["n_patients"], "n_windows": ci["n_windows"]}
            for band in ["<1", "1-5", "6-11", "12-17"]:
                sub = dft[dft["age_band"] == band]
                row[f"recall@5_{band}"] = (
                    patient_bootstrap_ci(sub, "recall@5")["point"] if len(sub) else float("nan"))
                row[f"n_patients_{band}"] = int(sub["patient_id"].nunique()) if len(sub) else 0
            trunc_rows.append(row)
        trunc_df = pd.DataFrame(trunc_rows)
        trunc_df.to_csv(tab_dir / "adkm_performance_by_truncation.csv", index=False)

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.errorbar(range(len(trunc_df)), trunc_df["recall@5"],
                    yerr=[trunc_df["recall@5"] - trunc_df["recall@5_ci_lo"],
                          trunc_df["recall@5_ci_hi"] - trunc_df["recall@5"]],
                    fmt="o-", color="#1f4e79", capsize=3)
        ax.set_xticks(range(len(trunc_df)))
        ax.set_xticklabels(trunc_df["horizon"])
        ax.set_xlabel("Allowed history horizon")
        ax.set_ylabel("Recall@5")
        ax.set_title("Controlled history truncation")
        fig.tight_layout()
        savefig(fig, "fig_performance_by_history")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for band, color in zip(["<1", "1-5", "6-11", "12-17"],
                               ["#4c78a8", "#f58518", "#54a24b", "#e45756"]):
            ax.plot(range(len(trunc_df)), trunc_df[f"recall@5_{band}"], "o-",
                    color=color, label=band)
        ax.set_xticks(range(len(trunc_df)))
        ax.set_xticklabels(trunc_df["horizon"])
        ax.legend(title="Age")
        ax.set_ylabel("Recall@5")
        ax.set_title("Truncation × age")
        fig.tight_layout()
        savefig(fig, "fig_performance_age_history_curves")

        mat = np.array([[trunc_df.iloc[j][f"recall@5_{b}"] for j in range(len(trunc_df))]
                        for b in ["<1", "1-5", "6-11", "12-17"]], float)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_yticks(range(4)); ax.set_yticklabels(["<1", "1-5", "6-11", "12-17"])
        ax.set_xticks(range(len(trunc_df))); ax.set_xticklabels(trunc_df["horizon"])
        ax.set_title("Recall@5 age × history")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        savefig(fig, "fig_performance_age_history")

        delta = mat - mat[:, -1:]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        vmax = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1
        im = ax.imshow(delta, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(4)); ax.set_yticklabels(["<1", "1-5", "6-11", "12-17"])
        ax.set_xticks(range(len(trunc_df))); ax.set_xticklabels(trunc_df["horizon"])
        ax.set_title(r"Δ Recall@5 vs full history")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        savefig(fig, "fig_performance_age_history_delta_vs_full")

    abl = {}
    if do_ablations:
        print("[fast] ablations", flush=True)
        beta = float(model.temporal.beta.detach().cpu())
        with torch.no_grad():
            model.temporal.beta.zero_()
        df0, _, _ = eval_windows(model, loader, device_t)
        with torch.no_grad():
            model.temporal.beta.fill_(beta)
        perm_vals = []
        for s in range(3):
            dfp, _, _ = eval_windows(model, loader, device_t, age_mode="permute", permute_seed=s)
            perm_vals.append(patient_bootstrap_ci(dfp, "recall@5")["point"])
        dfc, _, _ = eval_windows(model, loader, device_t, age_mode="constant", constant_age=9.0)
        base = overall["recall@5"]["point"]
        abl = {
            "natural": overall["recall@5"],
            "beta0": patient_bootstrap_ci(df0, "recall@5"),
            "delta_beta0": patient_bootstrap_ci(df0, "recall@5")["point"] - base,
            "permute_mean": float(np.mean(perm_vals)),
            "permute_values": perm_vals,
            "delta_permute": float(np.mean(perm_vals) - base),
            "constant9": patient_bootstrap_ci(dfc, "recall@5"),
            "delta_constant9": patient_bootstrap_ci(dfc, "recall@5")["point"] - base,
        }
        (raw_dir / "adkm_interaction_ablations.json").write_text(json.dumps(abl, indent=2) + "\n")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(["natural", "β=0", "age perm", "age=9"],
               [base, abl["beta0"]["point"], abl["permute_mean"], abl["constant9"]["point"]],
               color=["#1f4e79", "#6b8fad", "#a0a0a0", "#c4a35a"])
        ax.set_ylabel("Recall@5")
        ax.set_title("Inference-only interaction ablations")
        fig.tight_layout()
        savefig(fig, "fig_interaction_ablation")

    # calibration brier by age
    cal = []
    for band in ["<1", "1-5", "6-11", "12-17"]:
        sub = df[df["age_band"] == band]
        if len(sub) < 50:
            continue
        ci = patient_bootstrap_ci(sub, "brier")
        cal.append({"age_band": band, **ci})
    pd.DataFrame(cal).to_csv(tab_dir / "adkm_calibration_brier_by_age.csv", index=False)
    if cal:
        cdf = pd.DataFrame(cal)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(cdf["age_band"], cdf["point"], color="#1f4e79",
               yerr=[cdf["point"] - cdf["ci_lo"], cdf["ci_hi"] - cdf["point"]], capsize=3)
        ax.set_ylabel("Brier"); ax.set_title("Calibration (Brier) by age")
        fig.tight_layout()
        savefig(fig, "fig_calibration_brier_by_age")

    return {"overall": overall, "ablations": abl, "age_tab": age_tab.to_dict(orient="records")}


if __name__ == "__main__":
    import argparse
    from stage2_nch.analysis import ADKM_DIR, PRIMARY_CKPT_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ckpt", default=PRIMARY_CKPT_NAME)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = json.loads((ADKM_DIR / "config.json").read_text())
    run_fast_eval(
        ckpt=ADKM_DIR / args.ckpt,
        tensorized=Path(cfg["data"]["paths"]["tensorized_dir"]),
        device=args.device,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    print("fast eval done")
