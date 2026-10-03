#!/usr/bin/env python3
"""FIGURE A9: persistence_distribution

NCH/MIMIC DTR checkpoints do not learn content-dependent persistence θ_m.
This figure reports synthetic S5 learned persistence offsets (acute / intermediate / chronic)
from the Content-Persistence mechanism summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_ACCENT, C_BLUE, C_DTR, REPO, apply_style, save_fig  # noqa: E402

SUMMARY = REPO / "results/paper_synthetic/dtr_mechanism_summary.json"


def main() -> None:
    apply_style()
    s5 = json.loads(SUMMARY.read_text())["S5"]
    offsets = s5["learned_persistence_offset"]
    # Higher offset → higher λ → faster decay → less persistence.
    # Implied relative persistence rank: chronic > intermediate > acute when offsets decrease.
    types = ["acute", "intermediate", "chronic"]
    vals = [offsets[t] for t in types]
    colors = [C_ACCENT, C_BLUE, C_DTR]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    x = np.arange(len(types))
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylabel(r"Learned persistence offset $\theta_m$")
    ax.axhline(0, color="#a0aec0", lw=0.8)
    ax.set_title("Synthetic S5 content-dependent offsets")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.04 if v >= 0 else -0.08), f"{v:.2f}", ha="center", fontsize=7)

    ax = axes[1]
    ax.axis("off")
    order_ok = s5["learned_offset_order_acute_gt_intermediate_gt_chronic"]
    lines = [
        "Interpretation",
        "─────────────",
        "Larger θ_m → larger λ → faster decay",
        "→ lower temporal persistence.",
        "",
        f"Order acute>inter>chronic: {order_ok}",
        f"β̂ = {s5['beta_hat']:.3f} (true {s5['beta_true']})",
        f"Gate surface RMSE = {s5['gate_surface_RMSE']:.3f}",
        f"ΔAUPRC vs global = {s5['delta_AUPRC_vs_global']:.4f}",
        "",
        "Note: NCH/MIMIC DTR checkpoints",
        "have no θ_m parameters; clinical",
        "persistence histograms unavailable.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=7.5, family="DejaVu Sans", transform=ax.transAxes)

    fig.tight_layout()
    save_fig(fig, "persistence_distribution")


if __name__ == "__main__":
    main()
