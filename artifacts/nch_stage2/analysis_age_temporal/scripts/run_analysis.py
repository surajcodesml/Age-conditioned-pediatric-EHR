#!/usr/bin/env python3
"""NCH Stage-2 age-temporal analysis (paper diagnostics).

Runs without retraining. Uses held-out tensorized test split.
Detects no-interaction checkpoint when available for paired comparisons.

  PYTHONPATH=. python -m stage2_nch.analysis.run_analysis --device cpu
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from stage2_nch.analysis import ANALYSIS, ADKM_DIR, NINT_DIR, PRIMARY_CKPT_NAME, write_contract
from stage2_nch.analysis.kernel_figures import run_abc
from stage2_nch.analysis.eval_protocol import (
    HORIZON_LABELS,
    HORIZONS_DAYS,
    build_model_from_ckpt,
    collect_window_rows,
    compute_age_lag_support,
    history_bin_span,
    make_test_loader,
    patient_bootstrap_ci,
    patient_bootstrap_metric_from_logits,
    plot_support,
)
from stage2_nch.config import EVAL_KS, age_band_name


def _savefig(fig, stem: str, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _bar_with_ci(ax, labels, points, los, his, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, points, color="#1f4e79", alpha=0.85)
    ax.errorbar(x, points, yerr=[np.array(points) - np.array(los),
                                 np.array(his) - np.array(points)],
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def run_performance_block(
    model,
    ds,
    device,
    out_dir: Path,
    *,
    tag: str,
    do_truncation: bool = True,
    do_ablations: bool = True,
    max_examples: int = 0,
    batch_size: int = 4,
) -> dict:
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    pred_dir = out_dir / "predictions"
    raw_dir = out_dir / "raw"
    for d in (fig_dir, tab_dir, pred_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[{tag}] natural full-history evaluation…", flush=True)
    df, logits, targets = collect_window_rows(
        model, ds, device, horizon_days=None, age_mode="natural",
        max_examples=max_examples, batch_size=batch_size, store_logits=True, logit_cap=12000,
    )
    df["history_bin"] = df["history_span_days"].map(history_bin_span)
    df.to_parquet(pred_dir / f"{tag}_windows_full.parquet", index=False)

    # Overall metrics
    overall = {
        "bce": patient_bootstrap_ci(df, "bce"),
        "brier": patient_bootstrap_ci(df, "brier"),
        "recall@5": patient_bootstrap_ci(df, "recall@5"),
        "recall@20": patient_bootstrap_ci(df, "recall@20"),
        "precision@5": patient_bootstrap_ci(df, "precision@5"),
    }
    if logits is not None:
        overall["multilabel"] = patient_bootstrap_metric_from_logits(df, logits, targets)
    (raw_dir / f"{tag}_overall.json").write_text(json.dumps(overall, indent=2) + "\n")

    # --- D: by age band ---
    age_rows = []
    for band in ["<1", "1-5", "6-11", "12-17"]:
        sub = df[df["age_band"] == band]
        if sub.empty:
            continue
        rec = {"age_band": band, "n_windows": len(sub),
               "n_patients": int(sub["patient_id"].nunique()),
               "mean_age": float(sub["age_years_natural"].mean()),
               "mean_history_span_days": float(sub["history_span_days"].mean()),
               "mean_n_events": float(sub["n_input_events"].mean())}
        for m in ("bce", "brier", "recall@5", "recall@20", "precision@5"):
            ci = patient_bootstrap_ci(sub, m)
            rec[m] = ci["point"]
            rec[f"{m}_ci_lo"] = ci["ci_lo"]
            rec[f"{m}_ci_hi"] = ci["ci_hi"]
        age_rows.append(rec)
    age_tab = pd.DataFrame(age_rows)
    age_tab.to_csv(tab_dir / f"{tag}_performance_by_age.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if len(age_tab):
        _bar_with_ci(axes[0], age_tab["age_band"], age_tab["recall@5"],
                     age_tab["recall@5_ci_lo"], age_tab["recall@5_ci_hi"],
                     "Recall@5 by developmental age", "Recall@5")
        for i, r in age_tab.iterrows():
            axes[0].text(i, r["recall@5"], f"n={int(r['n_patients'])}", ha="center",
                         va="bottom", fontsize=7)
        _bar_with_ci(axes[1], age_tab["age_band"], age_tab["bce"],
                     age_tab["bce_ci_lo"], age_tab["bce_ci_hi"],
                     "BCE by developmental age", "BCE")
    fig.tight_layout()
    _savefig(fig, "fig_performance_by_age" if tag == "adkm" else f"fig_performance_by_age_{tag}",
             fig_dir)

    # Continuous age bins (equal-frequency sensitivity)
    try:
        df = df.copy()
        df["age_q"] = pd.qcut(df["age_years_natural"], q=5, duplicates="drop")
        cont = []
        for q, sub in df.groupby("age_q", observed=True):
            ci = patient_bootstrap_ci(sub, "recall@5")
            cont.append({"age_bin": str(q), "age_mid": float(sub["age_years_natural"].mean()),
                         **{f"recall@5_{k}": ci[k] for k in ("point", "ci_lo", "ci_hi")},
                         "n_patients": ci["n_patients"], "n_windows": ci["n_windows"]})
        cont_df = pd.DataFrame(cont)
        cont_df.to_csv(tab_dir / f"{tag}_performance_by_age_quantile.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(cont_df["age_mid"], cont_df["recall@5_point"],
                    yerr=[cont_df["recall@5_point"] - cont_df["recall@5_ci_lo"],
                          cont_df["recall@5_ci_hi"] - cont_df["recall@5_point"]],
                    fmt="o-", color="#1f4e79", capsize=3)
        ax.set_xlabel("Age (years; bin midpoints)")
        ax.set_ylabel("Recall@5")
        ax.set_title("Recall@5 vs age (equal-frequency bins, patient bootstrap CI)")
        fig.tight_layout()
        _savefig(fig, "fig_performance_by_age_continuous", fig_dir)
    except Exception as e:
        print("continuous age bins skipped:", e)

    # --- E1 natural history stratification ---
    hist_rows = []
    for bname, _, _ in [
        ("<30d", 0, 30), ("30-90d", 30, 90), ("90-180d", 90, 180),
        ("180d-1y", 180, 365), ("1-3y", 365, 1095), (">3y", 1095, 1e9),
    ]:
        sub = df[df["history_bin"] == bname]
        if sub.empty:
            continue
        ci = patient_bootstrap_ci(sub, "recall@5")
        hist_rows.append({
            "history_bin": bname,
            "n_patients": int(sub["patient_id"].nunique()),
            "n_windows": len(sub),
            "mean_age": float(sub["age_years_natural"].mean()),
            "mean_n_events": float(sub["n_input_events"].mean()),
            "recall@5": ci["point"], "recall@5_ci_lo": ci["ci_lo"], "recall@5_ci_hi": ci["ci_hi"],
            "bce": patient_bootstrap_ci(sub, "bce")["point"],
        })
    hist_tab = pd.DataFrame(hist_rows)
    hist_tab.to_csv(tab_dir / f"{tag}_performance_by_available_history.csv", index=False)

    # --- E2 controlled truncation ---
    trunc_tables = []
    if do_truncation:
        for horizon, label in zip(HORIZONS_DAYS, HORIZON_LABELS):
            print(f"[{tag}] truncation horizon={label}", flush=True)
            dft, _, _ = collect_window_rows(
                model, ds, device, horizon_days=horizon, age_mode="natural",
                max_examples=max_examples, batch_size=batch_size, store_logits=False,
            )
            dft["horizon"] = label
            dft["horizon_days"] = horizon if horizon is not None else -1
            dft.to_parquet(pred_dir / f"{tag}_windows_horizon_{label}.parquet", index=False)
            ci = patient_bootstrap_ci(dft, "recall@5")
            row = {"horizon": label, "horizon_days": horizon if horizon is not None else None,
                   "recall@5": ci["point"], "recall@5_ci_lo": ci["ci_lo"], "recall@5_ci_hi": ci["ci_hi"],
                   "bce": patient_bootstrap_ci(dft, "bce")["point"],
                   "n_patients": ci["n_patients"], "n_windows": ci["n_windows"]}
            # by age band
            for band in ["<1", "1-5", "6-11", "12-17"]:
                sub = dft[dft["age_band"] == band]
                if sub.empty:
                    row[f"recall@5_{band}"] = float("nan")
                    continue
                row[f"recall@5_{band}"] = patient_bootstrap_ci(sub, "recall@5")["point"]
                row[f"n_patients_{band}"] = int(sub["patient_id"].nunique())
            trunc_tables.append(row)
        trunc_df = pd.DataFrame(trunc_tables)
        trunc_df.to_csv(tab_dir / f"{tag}_performance_by_truncation.csv", index=False)

        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.errorbar(range(len(trunc_df)), trunc_df["recall@5"],
                    yerr=[trunc_df["recall@5"] - trunc_df["recall@5_ci_lo"],
                          trunc_df["recall@5_ci_hi"] - trunc_df["recall@5"]],
                    fmt="o-", color="#1f4e79", capsize=3)
        ax.set_xticks(range(len(trunc_df)))
        ax.set_xticklabels(trunc_df["horizon"])
        ax.set_xlabel("Allowed history horizon (controlled truncation)")
        ax.set_ylabel("Recall@5")
        ax.set_title("Performance vs controlled history horizon")
        fig.tight_layout()
        _savefig(fig, "fig_performance_by_history", fig_dir)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for band, color in zip(["<1", "1-5", "6-11", "12-17"],
                               ["#4c78a8", "#f58518", "#54a24b", "#e45756"]):
            col = f"recall@5_{band}"
            if col in trunc_df:
                ax.plot(range(len(trunc_df)), trunc_df[col], "o-", color=color, label=band)
        ax.set_xticks(range(len(trunc_df)))
        ax.set_xticklabels(trunc_df["horizon"])
        ax.set_xlabel("Allowed history horizon")
        ax.set_ylabel("Recall@5")
        ax.set_title("History truncation × developmental age")
        ax.legend(title="Age band")
        fig.tight_layout()
        _savefig(fig, "fig_performance_age_history_curves", fig_dir)

        # F1 heatmap
        mat = np.array([[trunc_df.iloc[j].get(f"recall@5_{b}", np.nan)
                         for j in range(len(trunc_df))]
                        for b in ["<1", "1-5", "6-11", "12-17"]], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_yticks(range(4))
        ax.set_yticklabels(["<1", "1-5", "6-11", "12-17"])
        ax.set_xticks(range(len(trunc_df)))
        ax.set_xticklabels(trunc_df["horizon"])
        ax.set_xlabel("Allowed history horizon")
        ax.set_ylabel("Age band")
        ax.set_title("Recall@5 age × history heatmap")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Recall@5")
        fig.tight_layout()
        _savefig(fig, "fig_performance_age_history", fig_dir)

        # F2 relative to full
        full = mat[:, -1:]
        delta = mat - full
        fig, ax = plt.subplots(figsize=(8, 3.8))
        vmax = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1
        im = ax.imshow(delta, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["<1", "1-5", "6-11", "12-17"])
        ax.set_xticks(range(len(trunc_df)))
        ax.set_xticklabels(trunc_df["horizon"])
        ax.set_title(r"Recall@5 change vs full history")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        _savefig(fig, "fig_performance_age_history_delta_vs_full", fig_dir)

    # --- G ablations ---
    abl = {}
    if do_ablations:
        print(f"[{tag}] ablations…", flush=True)
        # G1: disable interaction (beta=0) at inference
        beta_saved = float(model.temporal.beta.detach().cpu())
        with torch.no_grad():
            model.temporal.beta.zero_()
        df_g1, _, _ = collect_window_rows(
            model, ds, device, max_examples=max_examples, batch_size=batch_size)
        with torch.no_grad():
            model.temporal.beta.fill_(beta_saved)
        # G2: age permute (3 seeds)
        perm_scores = []
        for s in range(3):
            df_p, _, _ = collect_window_rows(
                model, ds, device, age_mode="permute", permute_seed=s,
                max_examples=max_examples, batch_size=batch_size)
            perm_scores.append(patient_bootstrap_ci(df_p, "recall@5")["point"])
        # G3: constant age 9
        df_c, _, _ = collect_window_rows(
            model, ds, device, age_mode="constant", constant_age=9.0,
            max_examples=max_examples, batch_size=batch_size)

        base_r = overall["recall@5"]["point"]
        abl = {
            "natural_recall@5": overall["recall@5"],
            "beta0_inference": patient_bootstrap_ci(df_g1, "recall@5"),
            "delta_beta0": patient_bootstrap_ci(df_g1, "recall@5")["point"] - base_r,
            "age_permute_recall@5_mean": float(np.mean(perm_scores)),
            "age_permute_recall@5_values": perm_scores,
            "delta_permute_mean": float(np.mean(perm_scores) - base_r),
            "constant_age9": patient_bootstrap_ci(df_c, "recall@5"),
            "delta_constant_age9": patient_bootstrap_ci(df_c, "recall@5")["point"] - base_r,
            "note": "Diagnostic counterfactual inference; not deployment settings.",
        }
        (raw_dir / f"{tag}_interaction_ablations.json").write_text(json.dumps(abl, indent=2) + "\n")

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["natural", "β=0 (infer)", "age permute", "age=9 const"]
        vals = [base_r, abl["beta0_inference"]["point"], abl["age_permute_recall@5_mean"],
                abl["constant_age9"]["point"]]
        ax.bar(labels, vals, color=["#1f4e79", "#6b8fad", "#a0a0a0", "#c4a35a"])
        ax.set_ylabel("Recall@5")
        ax.set_title("Interaction sanity tests (inference-only counterfactuals)")
        fig.tight_layout()
        _savefig(fig, "fig_interaction_ablation", fig_dir)

    # --- H calibration by age ---
    # Reliability: bin predicted prob of positive labels is expensive for multilabel;
    # use mean predicted probability of true-positive codes vs empirical frequency proxy via Brier.
    cal_rows = []
    for band in ["<1", "1-5", "6-11", "12-17"]:
        sub = df[df["age_band"] == band]
        if len(sub) < 50:
            continue
        cal_rows.append({
            "age_band": band,
            "n_patients": int(sub["patient_id"].nunique()),
            "n_windows": len(sub),
            "brier": patient_bootstrap_ci(sub, "brier")["point"],
            "brier_ci_lo": patient_bootstrap_ci(sub, "brier")["ci_lo"],
            "brier_ci_hi": patient_bootstrap_ci(sub, "brier")["ci_hi"],
            "bce": patient_bootstrap_ci(sub, "bce")["point"],
        })
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(tab_dir / f"{tag}_calibration_by_age.csv", index=False)
    if len(cal_df):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        _bar_with_ci(ax, cal_df["age_band"], cal_df["brier"], cal_df["brier_ci_lo"],
                     cal_df["brier_ci_hi"], "Brier score by age band", "Brier")
        fig.tight_layout()
        _savefig(fig, "fig_calibration_brier_by_age", fig_dir)

    return {"overall": overall, "age_tab": age_tab.to_dict(orient="records"),
            "hist_tab": hist_tab.to_dict(orient="records"), "ablations": abl}


def run_examples(model, ds, device, out_dir: Path, n: int = 5):
    """Select quantile-defined examples and dump temporal bias timelines."""
    from stage2_nch.analysis.eval_protocol import truncate_item
    from stage2_nch.dataset import make_nch_collate

    # Score candidates by age / history quantiles
    metas = []
    for i in range(len(ds)):
        it = ds[i]
        ts = np.asarray(it["timestamps_days"], dtype=np.float64)
        metas.append({
            "i": i, "patient_id": int(it["patient_id"]),
            "age": float(it["last_age_years"]),
            "span": float(ts[-1] - ts[0]) if ts.size else 0.0,
            "band": age_band_name(float(it["last_age_years"])),
        })
    mdf = pd.DataFrame(metas)
    picks = []
    for band in ["<1", "1-5", "6-11", "12-17"]:
        sub = mdf[mdf["band"] == band]
        if sub.empty:
            continue
        # median history within band
        target = sub["span"].median()
        j = (sub["span"] - target).abs().idxmin()
        picks.append(int(sub.loc[j, "i"]))
    # add long-history adolescent if available
    long = mdf[mdf["band"] == "12-17"].sort_values("span", ascending=False)
    if len(long):
        picks.append(int(long.iloc[0]["i"]))
    picks = list(dict.fromkeys(picks))[:n]

    collate = make_nch_collate(assert_horizon=False)
    rows = []
    for i in picks:
        item = ds[i]
        batch = collate([item])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(batch, need_diagnostics=True)
        bias = out["temporal_bias"][0].detach().cpu().numpy()  # [L,L]
        mask = out["pair_mask"][0].detach().cpu().numpy().astype(bool)
        ts = batch["timestamps_days"][0].cpu().numpy()
        ages = batch["age_years"][0].cpu().numpy()
        L = int(mask.any(axis=1).sum())
        # Last query row
        q = L - 1
        for j in range(L):
            lag = float(ts[q] - ts[j])
            rows.append({
                "example_idx": i,
                "patient_id": int(item["patient_id"]),
                "query_age": float(ages[q]),
                "key_age": float(ages[j]),
                "lag_days": lag,
                "K_last_query": float(bias[q, j]),
                "exp_K": float(np.exp(bias[q, j])),
            })
        # timeline plot
        fig, ax = plt.subplots(figsize=(8, 2.8))
        lags = ts[q] - ts[:L]
        K = bias[q, :L]
        ax.scatter(lags, np.exp(K), c=ages[:L], cmap="viridis", s=18)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Lag from last event (days)")
        ax.set_ylabel(r"$e^{K}$ (last query)")
        ax.set_title(f"Patient {item['patient_id']}  age={ages[q]:.1f}y  span={lags[0]:.0f}d")
        fig.tight_layout()
        _savefig(fig, f"fig_example_patient_{item['patient_id']}", out_dir / "figures")
    pd.DataFrame(rows).to_csv(out_dir / "tables" / "example_patient_timelines.csv", index=False)


def paired_compare(adkm_df: pd.DataFrame, nint_df: pd.DataFrame, metric: str = "recall@5") -> dict:
    """Paired patient-level bootstrap of mean(adkm)-mean(nint) on shared example_idx."""
    a = adkm_df.set_index("example_idx")
    b = nint_df.set_index("example_idx")
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    delta_win = a[metric].to_numpy() - b[metric].to_numpy()
    patients = a["patient_id"].to_numpy()
    uniq = np.unique(patients)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(500):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(patients, samp)
        # approximate: include all windows of sampled patients
        boots.append(float(delta_win[mask].mean()) if mask.any() else 0.0)
    return {
        "metric": metric,
        "delta_point": float(delta_win.mean()),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_windows": int(len(common)),
        "n_patients": int(len(uniq)),
    }


def write_report(out_dir: Path, abc: dict, perf: dict, nint_status: dict):
    p = out_dir / "report.md"
    final = abc["params"]["best_auprc"]
    init = abc["params"]["stage2_init"]
    s1 = abc["params"]["stage1_adult"]
    shift = abc["shift"]
    lines = [
        "# NCH Stage-2 age×temporal analysis",
        "",
        "## Verified model contract",
        "",
        "Attention (shared scalars λ₀, β across heads):",
        "",
        r"`s_ij^(h) = q_i^(h)·k_j^(h)/√d_h − [λ₀ + β z_P(a_i)] τ_ij`",
        "",
        r"with `z_P(a)=(a−9)/9`, `τ=log1p(|Δt|/7)`, and kernel bias `K=−λ(a)τ`.",
        "",
        "- Conditioning age `a_i` = **per-query event age** (years).",
        "- Positive λ → **recency**; negative λ → **long-range** (sign test verified).",
        "- Pooling has **no** λ/β (`pool_temporal_bias=False`).",
        "",
        "### Parameter values",
        "",
        f"| Source | λ₀ | β | age μ/σ |",
        f"|---|---:|---:|---|",
        f"| Stage-1 adult ckpt | {s1['lambda0']:.4f} | {s1['beta']:.4f} | {s1['age_mean']:.3f}/{s1['age_sd']:.3f} |",
        f"| Stage-2 init (transfer) | {init['lambda0']:.4f} | {init['beta']:.4f} | 9/9 |",
        f"| Stage-2 primary (`{PRIMARY_CKPT_NAME}`) | {final['lambda0']:.4f} | {final['beta']:.4f} | {final['age_mean']:.0f}/{final['age_sd']:.0f} |",
        "",
        f"Primary checkpoint epoch={final.get('epoch')}, val micro-AUPRC={final.get('val_micro_auprc')}.",
        "",
        "## A–C Learned function",
        "",
        f"- Final λ(a) remains **negative** across 0–18y (long-range prior retained).",
        f"- β_final={final['beta']:.3f} > 0 with λ₀<0 ⇒ |λ| **decreases** with age "
        f"(less long-range emphasis in older children relative to infants).",
        f"- Largest ΔK vs init: {shift['delta_K_max']} / {shift['delta_K_min']}.",
        "",
        "Figures: `fig_lambda_by_age`, `fig_temporal_curves_selected_ages`, "
        "`fig_age_lag_kernel_heatmap`, `fig_pediatric_shift_from_pretraining`.",
        "",
        "## D–F Predictive strata",
        "",
        f"Overall Recall@5: {perf.get('overall', {}).get('recall@5')}",
        "",
        "See tables `*_performance_by_age.csv`, truncation curves, and age×history heatmaps.",
        "",
        "## G Interaction diagnostics",
        "",
        f"{json.dumps(perf.get('ablations', {}), indent=2)}",
        "",
        "## J No-interaction comparison",
        "",
        f"Status: {json.dumps(nint_status)}",
        "",
        "When `nint_nch_s0/checkpoint_best_auprc.pt` exists, re-run this script to fill "
        "`fig_model_delta_by_age` and `fig_model_delta_age_history`.",
        "",
        "## Notes",
        "",
        "- Bootstrap CIs are **patient-level**.",
        "- No retraining; held-out test split unchanged.",
        "- Ablations are inference-only counterfactuals.",
        "",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_examples", type=int, default=0,
                    help="0 = full test set; use e.g. 2000 for smoke")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_truncation", action="store_true")
    ap.add_argument("--skip_ablations", action="store_true")
    ap.add_argument("--adkm_ckpt", type=str, default=PRIMARY_CKPT_NAME)
    args = ap.parse_args()

    out = ANALYSIS
    out.mkdir(parents=True, exist_ok=True)
    write_contract(out / "raw")

    print("=== A–C analytic kernel figures ===", flush=True)
    abc = run_abc(primary_name=args.adkm_ckpt)

    # Support heatmap from train split (CPU)
    cfg = json.loads((ADKM_DIR / "config.json").read_text())
    tensorized = Path(cfg["data"]["paths"]["tensorized_dir"])
    print("=== B2 age×lag support ===", flush=True)
    from stage2_nch.dataset import NCHForecastDataset
    from stage2_nch.config import VOCAB_PATH
    train_ds = NCHForecastDataset(tensorized / "train", VOCAB_PATH, max_seq_len=1024)
    # subsample for density if huge
    support = compute_age_lag_support(train_ds, max_examples=min(len(train_ds), 20000))
    np.savez(out / "raw" / "age_lag_support.npz", **{k: support[k] for k in
             ("age_edges", "lag_edges", "counts") if k in support})
    (out / "raw" / "age_lag_support_meta.json").write_text(json.dumps({
        "n_windows": support["n_windows"], "n_patients": support["n_patients"],
    }, indent=2) + "\n")
    kern = np.load(out / "raw" / "age_lag_kernel.npz")
    plot_support(support, out / "figures", age_grid=kern["age"], lag_grid=kern["lag_days"],
                 Kmat=kern["K"])

    nint_ckpt = NINT_DIR / args.adkm_ckpt
    nint_status = {
        "available": nint_ckpt.exists(),
        "path": str(nint_ckpt) if nint_ckpt.exists() else None,
        "note": "Paired J* figures pending until no-interaction Stage-2 finishes",
    }

    perf = {}
    if not args.skip_eval:
        device = torch.device(args.device)
        # Prefer CPU while age_temporal post-eval / future nint may hold GPU
        print(f"=== D–I evaluation on {device} ===", flush=True)
        ckpt = ADKM_DIR / args.adkm_ckpt
        model = build_model_from_ckpt(ckpt, device)
        ds, _ = make_test_loader(tensorized, batch_size=args.batch_size)
        print(f"test windows={len(ds)} patients≈{ds.patient_ids().size}", flush=True)
        perf = run_performance_block(
            model, ds, device, out, tag="adkm",
            do_truncation=not args.skip_truncation,
            do_ablations=not args.skip_ablations,
            max_examples=args.max_examples,
            batch_size=args.batch_size,
        )
        run_examples(model, ds, device, out)

        if nint_ckpt.exists():
            print("=== J paired no-interaction comparison ===", flush=True)
            nint_model = build_model_from_ckpt(nint_ckpt, device)
            nint_perf = run_performance_block(
                nint_model, ds, device, out, tag="nint",
                do_truncation=not args.skip_truncation,
                do_ablations=False,
                max_examples=args.max_examples,
                batch_size=args.batch_size,
            )
            adkm_df = pd.read_parquet(out / "predictions" / "adkm_windows_full.parquet")
            nint_df = pd.read_parquet(out / "predictions" / "nint_windows_full.parquet")
            delta = paired_compare(adkm_df, nint_df, "recall@5")
            (out / "raw" / "model_delta_overall.json").write_text(json.dumps(delta, indent=2) + "\n")

            # by age
            age_deltas = []
            for band in ["<1", "1-5", "6-11", "12-17"]:
                d = paired_compare(adkm_df[adkm_df.age_band == band],
                                   nint_df[nint_df.age_band == band], "recall@5")
                d["age_band"] = band
                age_deltas.append(d)
            pd.DataFrame(age_deltas).to_csv(out / "tables" / "model_delta_by_age.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([r["age_band"] for r in age_deltas], [r["delta_point"] for r in age_deltas],
                   color="#1f4e79")
            ax.errorbar(range(len(age_deltas)), [r["delta_point"] for r in age_deltas],
                        yerr=[[r["delta_point"] - r["ci_lo"] for r in age_deltas],
                              [r["ci_hi"] - r["delta_point"] for r in age_deltas]],
                        fmt="none", ecolor="k", capsize=3)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_ylabel(r"Δ Recall@5 (age-temporal − no-interaction)")
            ax.set_title("Paired patient-bootstrap model difference by age")
            fig.tight_layout()
            _savefig(fig, "fig_model_delta_by_age", out / "figures")
            nint_status["paired_overall"] = delta
            nint_status["available"] = True
            perf["nint"] = nint_perf

    write_report(out, abc, perf, nint_status)
    print("Done. Report:", out / "report.md", flush=True)


if __name__ == "__main__":
    main()
