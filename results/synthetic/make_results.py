#!/usr/bin/env python
"""Extract synthetic-data downstream metrics and build per-band AUPRC bar charts.

- Point metrics (overall + per age band) are read directly from each
  checkpoints/synthea_compare/<disease>_<arm>/history.json.
- Per-band 95% AUPRC confidence intervals are NOT stored in history.json, so
  they are bootstrapped from test_predictions.parquet (the only use of parquet).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints" / "synthea_compare"
OUT = ROOT / "results" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

DISEASES = ["asthma", "obesity", "osa", "t2d"]
ARMS = ["age", "vanilla"]
BANDS = [("<1", 0, 1), ("1-5", 1, 6), ("6-11", 6, 12), ("12-17", 12, 18), ("18-25", 18, 26)]
SMALL_N = 500          # bands below this are greyed/hatched as untrustworthy
N_BOOT = 1000
RNG_SEED = 0

# ---------------------------------------------------------------------------
# 1. Metrics JSON (straight from history.json)
# ---------------------------------------------------------------------------
metrics = {}
for dis in DISEASES:
    metrics[dis] = {}
    for arm in ARMS:
        h = json.loads((CKPT / f"{dis}_{arm}" / "history.json").read_text())
        tm, te = h["test_metrics"], h["test_extended"]
        metrics[dis][arm] = {
            "accuracy": tm["accuracy"],
            "auroc": tm["auroc"],
            "auprc": tm["auprc"],
            "brier_raw": te["brier_raw"],
            "ece_raw": te["ece_raw"],
        }

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"wrote {OUT/'metrics.json'}")


# ---------------------------------------------------------------------------
# 2. Per-band AUPRC + bootstrap CI, then grouped bar chart per disease
# ---------------------------------------------------------------------------
def bootstrap_auprc_ci(y, p, n_boot=N_BOOT, seed=RNG_SEED):
    """Return (lo, hi) 95% CI; (nan, nan) if a band has only one class."""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(average_precision_score(yb, p[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


COLORS = {"age": "#1f77b4", "vanilla": "#ff7f0e"}
HATCH = {"age": None, "vanilla": None}

for dis in DISEASES:
    hists = {arm: json.loads((CKPT / f"{dis}_{arm}" / "history.json").read_text()) for arm in ARMS}
    preds = {arm: pd.read_parquet(CKPT / f"{dis}_{arm}" / "test_predictions.parquet") for arm in ARMS}

    band_auprc = {arm: [] for arm in ARMS}
    band_err = {arm: [[], []] for arm in ARMS}  # lower, upper error magnitudes
    band_n = []

    for name, lo, hi in BANDS:
        band_n.append(hists["age"]["test_age_stratified"][name]["n"])
        for arm in ARMS:
            ap = hists[arm]["test_age_stratified"][name]["auprc"]
            band_auprc[arm].append(ap)
            df = preds[arm]
            m = (df["age_at_landmark"] >= lo) & (df["age_at_landmark"] < hi)
            clo, chi = bootstrap_auprc_ci(df["label"][m].to_numpy(), df["prob"][m].to_numpy())
            if np.isnan(clo):
                band_err[arm][0].append(0.0)
                band_err[arm][1].append(0.0)
            else:
                band_err[arm][0].append(max(ap - clo, 0.0))
                band_err[arm][1].append(max(chi - ap, 0.0))

    # plot
    x = np.arange(len(BANDS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, arm in enumerate(ARMS):
        off = (-0.5 + i) * w
        for j in range(len(BANDS)):
            small = band_n[j] < SMALL_N
            color = COLORS[arm]
            alpha = 0.35 if small else 1.0
            hatch = "xx" if small else HATCH[arm]
            ax.bar(x[j] + off, band_auprc[arm][j], w, color=color, alpha=alpha,
                   hatch=hatch, edgecolor="black", linewidth=0.6, zorder=2)
        ax.errorbar(x + off, band_auprc[arm],
                    yerr=[band_err[arm][0], band_err[arm][1]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3)

    # shade untrustworthy band columns
    for j in range(len(BANDS)):
        if band_n[j] < SMALL_N:
            ax.axvspan(x[j] - 0.5, x[j] + 0.5, color="0.85", alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[0]}\nn={n}" for b, n in zip(BANDS, band_n)])
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dis} (synthetic) — per-band AUPRC: age vs vanilla")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    legend = [
        Patch(facecolor=COLORS["age"], edgecolor="black", label="age"),
        Patch(facecolor=COLORS["vanilla"], edgecolor="black", label="vanilla"),
        Patch(facecolor="0.85", edgecolor="black", hatch="xx", label=f"n<{SMALL_N} (low confidence)"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    p = OUT / f"auprc_by_band_{dis}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

print("done")
