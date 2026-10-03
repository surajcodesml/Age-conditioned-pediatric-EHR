#!/usr/bin/env python3
"""FIGURE A1: synthetic_lambda_curves — true vs learned λ(a) for S0–S3.

Data: final encounter-level DTR controlled runs (raw_additive aggregation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_ACCENT, REPO, apply_style, label_panel, save_fig  # noqa: E402

RUNS = {
    "S0": REPO
    / "synthetic_age_temporal/outputs/runs/dtr/controlled_S0_dtr_age_temporal_raw_additive_m0/metrics.json",
    "S1": REPO
    / "synthetic_age_temporal/outputs/runs/dtr/controlled_S1_dtr_age_temporal_raw_additive_m0/metrics.json",
    "S2": REPO
    / "synthetic_age_temporal/outputs/runs/dtr/controlled_S2_aggcmp_dtr_age_temporal_raw_additive_m0/metrics.json",
    "S3": REPO
    / "synthetic_age_temporal/outputs/runs/dtr/controlled_S3_dtr_age_temporal_raw_additive_m0/metrics.json",
}


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True, sharey=False)
    for ax, (sc, path), letter in zip(axes.ravel(), RUNS.items(), "ABCD"):
        rec = json.loads(path.read_text())["recovery"]
        ages = sorted(float(a) for a in rec["lambda_true_by_age"])
        # keys are "0.0", "1.0", ...
        lt = [rec["lambda_true_by_age"][f"{a:.1f}"] for a in ages]
        ll = [rec["lambda_learned_by_age"][f"{a:.1f}"] for a in ages]
        ax.plot(ages, lt, "k--", lw=1.8, label=r"$\lambda_{\mathrm{true}}$")
        ax.plot(ages, ll, color=C_ACCENT, lw=1.8, label=r"$\lambda_{\mathrm{learned}}$")
        corr = rec.get("corr_lambda")
        corr_s = "n/a" if corr is None or (isinstance(corr, float) and np.isnan(corr)) else f"{corr:.3f}"
        ax.set_title(f"{sc}  (β̂={rec['beta_hat']:.2f}, corr={corr_s})", fontsize=8.5)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(r"$\lambda(a)$")
        label_panel(ax, letter)
        if letter == "A":
            ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_fig(fig, "synthetic_lambda_curves")


if __name__ == "__main__":
    main()
