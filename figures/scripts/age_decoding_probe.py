#!/usr/bin/env python3
"""FIGURE A4: age_decoding_probe — developmental-age leakage from frozen reps."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_ACCENT, C_BLUE, C_GRAY, REPO, apply_style, label_panel, save_fig  # noqa: E402

PROBE = REPO / "synthetic_age_temporal/results/followup/followup_age_probe.json"

ARM_ORDER = ["no_age", "temporal_only", "age_temporal"]
ARM_LABEL = {
    "no_age": "No-age",
    "temporal_only": "Temporal-only",
    "age_temporal": "DTR (age×temporal)",
}
SITE_ORDER = ["pre_pool", "pooled", "head_mean"]
SITE_LABEL = {"pre_pool": "Pre-pool", "pooled": "Pooled", "head_mean": "Head mean"}


def main() -> None:
    apply_style()
    data = json.loads(PROBE.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    x = np.arange(len(SITE_ORDER))
    w = 0.25
    colors = [C_GRAY, C_ACCENT, C_BLUE]

    for ax, metric, title, letter in zip(
        axes,
        ["r2", "band_acc"],
        [r"Age probe $R^2$", "Age-band accuracy"],
        "AB",
    ):
        for i, arm in enumerate(ARM_ORDER):
            vals = [data[arm][site][metric] for site in SITE_ORDER]
            ax.bar(x + (i - 1) * w, vals, w, label=ARM_LABEL[arm], color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels([SITE_LABEL[s] for s in SITE_ORDER])
        ax.set_ylim(0.9, 1.0)
        ax.set_ylabel(title)
        ax.set_title(title)
        label_panel(ax, letter)
        if letter == "A":
            ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    # Annotate key no-age pooled R2
    r2 = data["no_age"]["pooled"]["r2"]
    ba = data["no_age"]["pooled"]["band_acc"]
    fig.text(
        0.5,
        -0.02,
        f"No-age pooled: $R^2$={r2:.3f}, band acc={ba:.3f} (age recoverable without explicit age input)",
        ha="center",
        fontsize=7,
        color="#4A5568",
    )
    fig.tight_layout()
    save_fig(fig, "age_decoding_probe")


if __name__ == "__main__":
    main()
