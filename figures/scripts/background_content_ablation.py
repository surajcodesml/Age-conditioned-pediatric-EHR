#!/usr/bin/env python3
"""FIGURE A5: background_content_ablation — full vs signal-only histories (S2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_ACCENT, C_BLUE, C_DTR, C_TEMPORAL, REPO, apply_style, label_panel, save_fig  # noqa: E402

ABL = REPO / "synthetic_age_temporal/results/followup/background_ablation_results.json"


def main() -> None:
    apply_style()
    data = json.loads(ABL.read_text())
    variants = [("S2/full", "Full Synthea"), ("S2/signal_only", "Signal-only")]

    # Panel metrics
    delta_auprc = []
    delta_auroc = []
    shuffle = []
    beta0 = []
    labels = []
    for key, lab in variants:
        at = data[key]["age_temporal"]
        to = data[key]["temporal_only"]
        labels.append(lab)
        delta_auprc.append(at["test"]["micro_auprc"] - to["test"]["micro_auprc"])
        delta_auroc.append(at["test"]["micro_auroc"] - to["test"]["micro_auroc"])
        shuffle.append(at["ablations"]["delta_bce_shuffle_age"])
        beta0.append(at["ablations"]["delta_bce_beta0"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Left: predictive gain DTR − temporal-only
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, delta_auprc, w, label=r"$\Delta$AUPRC (DTR − temp.)", color=C_DTR)
    ax.bar(x + w / 2, delta_auroc, w, label=r"$\Delta$AUROC (DTR − temp.)", color=C_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="#a0aec0", lw=0.8)
    ax.set_ylabel("Predictive gain")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("DTR vs Temporal-only")
    label_panel(ax, "A")

    # Right: mechanism reliance
    ax = axes[1]
    ax.bar(x - w / 2, shuffle, w, label=r"age-shuffle $\Delta$BCE", color=C_BLUE)
    ax.bar(x + w / 2, beta0, w, label=r"$\beta{=}0$ $\Delta$BCE", color=C_ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="#a0aec0", lw=0.8)
    ax.set_ylabel(r"$\Delta$BCE")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Mechanism reliance (DTR)")
    label_panel(ax, "B")

    fig.suptitle("Background-content ablation (S2)", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "background_content_ablation")


if __name__ == "__main__":
    main()
