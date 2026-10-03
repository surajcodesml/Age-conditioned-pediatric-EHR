#!/usr/bin/env python3
"""FIGURE A3: additive_vs_softmax — M1 matched comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_ACCENT, C_BLUE, REPO, apply_style, save_fig  # noqa: E402

M1 = REPO / "synthetic_age_temporal/results/followup/M1_results.json"


def main() -> None:
    apply_style()
    m1 = json.loads(M1.read_text())
    arms = ["additive", "softmax"]
    labels = ["Additive", "Softmax"]

    metric_specs = [
        ("Surface RMSE", lambda a: m1[a]["age_temporal"]["recovery"]["RMSE_surface"]),
        (r"age-shuffle $\Delta$BCE", lambda a: m1[a]["gate"]["delta_bce_shuffle"]),
        (r"$\beta{=}0$ $\Delta$BCE", lambda a: m1[a]["gate"]["delta_bce_beta0"]),
        ("AUPRC", lambda a: m1[a]["age_temporal"]["test"]["micro_auprc"]),
        ("AUROC", lambda a: m1[a]["age_temporal"]["test"]["micro_auroc"]),
        (r"corr$(\lambda)$", lambda a: m1[a]["gate"]["corr_lambda"]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(metric_specs))
    w = 0.36
    add_vals = [fn("additive") for _, fn in metric_specs]
    soft_vals = [fn("softmax") for _, fn in metric_specs]
    ax.bar(x - w / 2, add_vals, w, label="Additive", color=C_BLUE)
    ax.bar(x + w / 2, soft_vals, w, label="Softmax", color=C_ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels([t for t, _ in metric_specs], rotation=20, ha="right")
    ax.axhline(0, color="#a0aec0", lw=0.8)
    ax.legend(frameon=False)
    ax.set_title("Additive vs softmax aggregation (M1, S2 controlled)")
    ax.set_ylabel("Metric value")
    # annotate corr sign failure
    ax.annotate(
        "softmax corr(λ)<0",
        xy=(5 + w / 2, soft_vals[5]),
        xytext=(4.2, -0.5),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="#718096", lw=0.8),
        color="#718096",
    )
    fig.tight_layout()
    save_fig(fig, "additive_vs_softmax")


if __name__ == "__main__":
    main()
