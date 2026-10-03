#!/usr/bin/env python3
"""FIGURE A2: synthetic_architecture_ladder"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_ACCENT, C_BLUE, C_GRAY, REPO, apply_style, label_panel, save_fig  # noqa: E402

CSV = REPO / "synthetic_age_temporal/results/final_validation/architecture_comparison.csv"

# Display names aligned with paper draft
DISPLAY = {
    "Original Transformer": "Original Transformer",
    "GLM interaction": "GLM interaction",
    "M1 additive": "M1 kernel-only additive",
    "M2": "M2 content × gate",
    "event-M3": "M3 encounter encoder + DTR",
    "Final encounter-level DTR": "Final DTR (encounter)",
}


def main() -> None:
    apply_style()
    rows = list(csv.DictReader(CSV.open()))
    # Prefer the validated ladder path; drop GLM if all-NaN ablations clutter
    keep = [
        "Original Transformer",
        "M1 additive",
        "M2",
        "event-M3",
        "Final encounter-level DTR",
    ]
    rows = [r for r in rows if r["model"] in keep]
    names = [DISPLAY[r["model"]] for r in rows]
    x = np.arange(len(names))

    metrics = [
        ("delta_auroc", r"$\Delta$AUROC", C_BLUE),
        ("delta_auprc", r"$\Delta$AUPRC", C_ACCENT),
        ("delta_bce_shuffle", r"age-shuffle $\Delta$BCE", "#2C7A7B"),
        ("delta_bce_beta0", r"$\beta{=}0$ $\Delta$BCE", "#9B2C2C"),
        ("RMSE_surface", "Surface RMSE ↓", C_GRAY),
        ("corr_lambda", r"corr$(\lambda)$", "#553C9A"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6))
    for ax, (key, title, color), letter in zip(axes.ravel(), metrics, "ABCDEF"):
        vals = []
        for r in rows:
            v = r[key]
            vals.append(float("nan") if v in ("", "nan", "None") else float(v))
        colors = [C_GRAY if "Transformer" in n else color for n in names]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=6.5)
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="#a0aec0", lw=0.7)
        label_panel(ax, letter, x=-0.12, y=1.12)
    fig.suptitle("Architecture ladder (S2 controlled)", fontsize=10, y=1.01)
    fig.tight_layout()
    save_fig(fig, "synthetic_architecture_ladder")


if __name__ == "__main__":
    main()
