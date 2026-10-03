#!/usr/bin/env python3
"""FIGURE A6: synthetic_multiseed_summary

Multi-seed replications were not run. This figure reports single-seed point
estimates across S0–S3 from final_validation/controlled_metrics.csv, and
overlays patient-bootstrap mean±SD for S2 functional metrics where available.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_ACCENT, C_BLUE, REPO, apply_style, label_panel, save_fig  # noqa: E402

CSV = REPO / "synthetic_age_temporal/results/final_validation/controlled_metrics.csv"
BOOT = REPO / "synthetic_age_temporal/results/final_validation/bootstrap_results.json"


def main() -> None:
    apply_style()
    rows = {r["scenario"]: r for r in csv.DictReader(CSV.open())}
    boot = json.loads(BOOT.read_text())["controlled_S2"]
    scenarios = ["S0", "S1", "S2", "S3"]
    x = np.arange(len(scenarios))

    specs = [
        ("beta_hat", r"$\hat\beta$", None, None),
        ("shuffle_delta_bce", r"age-shuffle $\Delta$BCE", "delta_bce_shuffle", C_BLUE),
        ("beta0_delta_bce", r"$\beta{=}0$ $\Delta$BCE", "delta_bce_beta0", C_ACCENT),
        ("RMSE_surface", "Surface RMSE", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, (key, title, boot_key, color), letter in zip(axes.ravel(), specs, "ABCD"):
        vals = [float(rows[s][key]) for s in scenarios]
        yerr = [0.0, 0.0, 0.0, 0.0]
        if boot_key is not None:
            yerr[2] = float(boot[boot_key]["std"])
        ax.errorbar(
            x,
            vals,
            yerr=yerr,
            fmt="o-",
            color=color or C_BLUE,
            lw=1.5,
            ms=6,
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_title(title)
        ax.axhline(0, color="#a0aec0", lw=0.7)
        label_panel(ax, letter)
        if letter == "A":
            ax.axhline(-2.5, color="#a0aec0", ls="--", lw=0.8, label=r"$\beta_{\mathrm{true}}=-2.5$ (S2)")
            ax.axhline(2.5, color="#cbd5e0", ls=":", lw=0.8, label=r"$\beta_{\mathrm{true}}=+2.5$ (S3)")
            ax.legend(frameon=False, fontsize=6)

    fig.suptitle(
        "S0–S3 mechanism summary (seed 0; S2 error bars = patient-bootstrap SD)",
        fontsize=9.5,
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "synthetic_multiseed_summary")


if __name__ == "__main__":
    main()
