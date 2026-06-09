#!/usr/bin/env python3
"""PIC test-set evaluation stratified by developmental age bands."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.dataset import (
    TensorizedDiseaseClassificationDataset,
    _dataloader_worker_init,
    disease_collate,
)
from finetune.train import _compute_ece, _move_batch_to_device
from finetune.PIC.pic_age_eval_common import (
    BACKBONES,
    DEV_BANDS,
    MIN_BAND_N,
    PIC_CKPT_ROOT,
    PIC_TENSOR_ROOT,
    TASKS,
    assign_band,
    load_finetuned_classifier,
)


def _load_or_build_preds(
    run_dir: Path,
    backbone: str,
    task: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    preds_path = run_dir / "test_preds.parquet"
    legacy_path = run_dir / "test_predictions.parquet"

    if preds_path.exists():
        df = pd.read_parquet(preds_path)
        required = {"y_true", "y_score", "age_years_at_anchor", "seq_len"}
        if not required.issubset(df.columns):
            raise ValueError(f"{preds_path} missing columns: {required - set(df.columns)}")
        return df

    if legacy_path.exists():
        leg = pd.read_parquet(legacy_path)
        df = pd.DataFrame(
            {
                "subject_id": leg["subject_id"].astype(np.int64),
                "y_true": leg["label"].astype(np.int32),
                "y_score": leg["prob"].astype(np.float64),
                "age_years_at_anchor": np.clip(leg["age_at_landmark"].astype(np.float64), 0.0, None),
                "seq_len": leg["n_events_in_window"].astype(np.int64),
            }
        )
        df.to_parquet(preds_path, index=False)
        print(f"[{task}/{backbone}] mapped {legacy_path.name} -> {preds_path.name}", flush=True)
        return df

    print(f"[{task}/{backbone}] running test inference -> {preds_path}", flush=True)
    model = load_finetuned_classifier(run_dir, backbone, device)
    tensor_dir = PIC_TENSOR_ROOT / task / "test"
    ds = TensorizedDiseaseClassificationDataset(tensor_dir, max_seq_len=1024, shard_cache_size=4)
    loader_kw: dict = dict(
        batch_size=batch_size,
        shuffle=False,
        collate_fn=disease_collate,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        worker_init_fn=_dataloader_worker_init,
    )
    if num_workers > 0:
        loader_kw["multiprocessing_context"] = "spawn"
    loader = DataLoader(ds, **loader_kw)

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy().astype(np.int32)
            lengths = batch["attention_mask"].sum(dim=1).long()
            b = lengths.shape[0]
            ridx = torch.arange(b, device=device)
            age_anchor = batch["demographics"][ridx, lengths - 1, 0].detach().cpu().numpy()
            age_anchor = np.clip(age_anchor, 0.0, None)
            for i in range(b):
                rows.append(
                    {
                        "subject_id": int(batch["subject_id"][i]),
                        "y_true": int(labels[i]),
                        "y_score": float(probs[i]),
                        "age_years_at_anchor": float(age_anchor[i]),
                        "seq_len": int(lengths[i].item()),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_parquet(preds_path, index=False)
    return df


def _band_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    n = int(y_true.size)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    prevalence = float(n_pos / n) if n else float("nan")
    unreliable = n < MIN_BAND_N or n_pos == 0 or n_neg == 0
    out = {
        "N": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prevalence,
        "auroc": float("nan"),
        "auprc": float("nan"),
        "ece": float("nan"),
        "unreliable": unreliable,
    }
    if not unreliable:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
        out["ece"] = float(_compute_ece(y_true.astype(np.float64), y_score, n_bins=15))
    return out


def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    subject_id: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    if n < MIN_BAND_N or y_true.min() == y_true.max():
        return float("nan"), float("nan")
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if y_b.min() != y_b.max():
            vals.append(float(roc_auc_score(y_b, s_b)))
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _paired_delta_ci(
    merged: pd.DataFrame,
    band_name: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    sub = merged[merged["band"] == band_name]
    if sub.empty or sub["y_true"].nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    if len(sub) < MIN_BAND_N:
        return float("nan"), float("nan"), float("nan")

    y = sub["y_true"].to_numpy(dtype=np.int32)
    s_v = sub["y_score_vanilla"].to_numpy(dtype=np.float64)
    s_a = sub["y_score_age"].to_numpy(dtype=np.float64)
    if y.min() == y.max():
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(sub)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        a_v = float(roc_auc_score(y_b, s_v[idx]))
        a_a = float(roc_auc_score(y_b, s_a[idx]))
        deltas.append(a_a - a_v)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(deltas, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _plot_task_bars(task: str, table: pd.DataFrame, out_png: Path) -> None:
    bands = [b.name for b in DEV_BANDS]
    x = np.arange(len(bands))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, backbone, color in [(-width / 2, "vanilla", "#4C72B0"), (width / 2, "age", "#C44E52")]:
        aurocs = []
        err_lo = []
        err_hi = []
        colors = []
        for band in bands:
            row = table[table["band"] == band].iloc[0]
            unreliable = bool(row[f"unreliable_{backbone}"])
            a = row[f"auroc_{backbone}"]
            lo = row[f"auroc_ci_lo_{backbone}"]
            hi = row[f"auroc_ci_hi_{backbone}"]
            aurocs.append(a if np.isfinite(a) else 0.0)
            if unreliable or not np.isfinite(lo):
                err_lo.append(0.0)
                err_hi.append(0.0)
                colors.append("#BBBBBB")
            else:
                err_lo.append(a - lo)
                err_hi.append(hi - a)
                colors.append(color)
        bars = ax.bar(x + offset, aurocs, width=width, yerr=[err_lo, err_hi], capsize=3, color=colors, label=backbone)
        for bar, band in zip(bars, bands):
            row = table[table["band"] == band].iloc[0]
            if bool(row[f"unreliable_{backbone}"]):
                bar.set_alpha(0.45)

    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title(f"PIC test AUROC by developmental band — {task}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def evaluate_task(
    task: str,
    ckpt_root: Path,
    results_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    n_boot: int,
) -> pd.DataFrame:
    preds_by_bb: dict[str, pd.DataFrame] = {}
    for backbone in BACKBONES:
        run_dir = ckpt_root / f"{task}_{backbone}"
        preds_by_bb[backbone] = _load_or_build_preds(
            run_dir, backbone, task, device, batch_size, num_workers
        )

    merged = preds_by_bb["vanilla"][["subject_id", "y_true", "age_years_at_anchor"]].rename(
        columns={"age_years_at_anchor": "age_years"}
    )
    merged = merged.merge(
        preds_by_bb["vanilla"][["subject_id", "y_score"]].rename(columns={"y_score": "y_score_vanilla"}),
        on="subject_id",
    )
    merged = merged.merge(
        preds_by_bb["age"][["subject_id", "y_score"]].rename(columns={"y_score": "y_score_age"}),
        on="subject_id",
    )
    merged["band"] = assign_band(merged["age_years"].to_numpy())

    rows: list[dict] = []
    for band in DEV_BANDS:
        row: dict = {"task": task, "band": band.name, "band_lo_yr": band.lo_yr, "band_hi_yr": band.hi_yr}
        for backbone in BACKBONES:
            sub = preds_by_bb[backbone]
            sub = sub.assign(band=assign_band(sub["age_years_at_anchor"].to_numpy()))
            mask = sub["band"] == band.name
            y = sub.loc[mask, "y_true"].to_numpy(dtype=np.int32)
            s = sub.loc[mask, "y_score"].to_numpy(dtype=np.float64)
            sid = sub.loc[mask, "subject_id"].to_numpy(dtype=np.int64)
            m = _band_metrics(y, s)
            ci_lo, ci_hi = _bootstrap_auroc_ci(y, s, sid, n_boot=n_boot)
            for k, v in m.items():
                row[f"{k}_{backbone}"] = v
            row[f"auroc_ci_lo_{backbone}"] = ci_lo
            row[f"auroc_ci_hi_{backbone}"] = ci_hi

        delta_mean, delta_lo, delta_hi = _paired_delta_ci(merged, band.name, n_boot=n_boot)
        row["delta_auroc_age_minus_vanilla"] = delta_mean
        row["delta_auroc_ci_lo"] = delta_lo
        row["delta_auroc_ci_hi"] = delta_hi
        rows.append(row)

    table = pd.DataFrame(rows)
    out_csv = results_dir / f"{task}.csv"
    out_png = results_dir / f"{task}_auroc_by_band.png"
    table.to_csv(out_csv, index=False)
    _plot_task_bars(task, table, out_png)
    print(f"[{task}] wrote {out_csv} and {out_png}", flush=True)
    return table


def _print_summary(tables: dict[str, pd.DataFrame]) -> None:
    print("\n=== Summary: largest age gain (paired delta AUROC) per task ===", flush=True)
    for task, table in tables.items():
        ok = table[
            (~table["unreliable_vanilla"])
            & (~table["unreliable_age"])
            & table["delta_auroc_age_minus_vanilla"].notna()
        ].copy()
        if ok.empty:
            print(f"  {task}: no reliable band with paired delta", flush=True)
            continue
        ok = ok.sort_values("delta_auroc_age_minus_vanilla", ascending=False)
        best = ok.iloc[0]
        sig = ""
        if np.isfinite(best["delta_auroc_ci_lo"]) and best["delta_auroc_ci_lo"] > 0:
            sig = " (95% CI excludes 0)"
        print(
            f"  {task}: band={best['band']} delta={best['delta_auroc_age_minus_vanilla']:.4f} "
            f"CI=[{best['delta_auroc_ci_lo']:.4f}, {best['delta_auroc_ci_hi']:.4f}]{sig}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIC age-stratified test evaluation.")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--ckpt_root", type=Path, default=PIC_CKPT_ROOT)
    p.add_argument("--results_dir", type=Path, default=REPO_ROOT / "results" / "pic" / "age_stratified")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {}
    for task in args.tasks:
        tables[task] = evaluate_task(
            task,
            args.ckpt_root,
            args.results_dir,
            device,
            args.batch_size,
            args.num_workers,
            args.n_bootstrap,
        )
    _print_summary(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
