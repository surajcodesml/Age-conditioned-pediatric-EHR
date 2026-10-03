#!/usr/bin/env python3
"""FIGURE 2: synthetic_mechanism_recovery

Panels A–C: oracle / DTR / Original Transformer age×lag relevance surfaces (S2).
Panel D: functional interventions (age-shuffle ΔBCE, β=0 ΔBCE) across S0–S3.

Data sources (no invented values):
- Final encounter-level DTR S2: controlled_S2_aggcmp_dtr_age_temporal_raw_additive_m0/metrics.json
- Original Transformer S2: arch_S2_age_temporal_d20260922_m0_interonly/metrics.json
- Functional ablations S0–S3: final_validation/controlled_metrics.csv
- Patient-bootstrap CIs (S2 only): final_validation/bootstrap_results.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (  # noqa: E402
    C_ACCENT,
    C_BLUE,
    REPO,
    apply_style,
    label_panel,
    save_fig,
)

# Import synthetic relevance helpers from the synthetic package.
sys.path.insert(0, str(REPO / "synthetic_age_temporal"))
from config import (  # noqa: E402
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    relevance,
    tau_from_days,
)

DTR_S2 = (
    REPO
    / "synthetic_age_temporal/outputs/runs/dtr/"
    "controlled_S2_aggcmp_dtr_age_temporal_raw_additive_m0/metrics.json"
)
TRANSFORMER_S2 = (
    REPO
    / "synthetic_age_temporal/outputs/runs/controlled/"
    "arch_S2_age_temporal_d20260922_m0_interonly/metrics.json"
)
CONTROLLED_CSV = REPO / "synthetic_age_temporal/results/final_validation/controlled_metrics.csv"
BOOTSTRAP = REPO / "synthetic_age_temporal/results/final_validation/bootstrap_results.json"


def _surface(theta0: float, beta: float, ages, lags) -> np.ndarray:
    """R(a, τ) with shape (n_ages, n_lags) for imshow (age on y, lag on x)."""
    R = np.zeros((len(ages), len(lags)), dtype=np.float64)
    for i, a in enumerate(ages):
        for j, d in enumerate(lags):
            t = float(tau_from_days(d))
            R[i, j] = float(relevance(a, t, theta0, beta))
    return R


def _imshow_surface(ax, R, ages, lags, vmin, vmax, title: str):
    # extent: [lag_min, lag_max, age_min, age_max] with origin lower
    # Use categorical lag ticks for interpretability.
    im = ax.imshow(
        R,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f"{int(d)}" if d != 0 else "0" for d in lags], fontsize=6.5)
    ax.set_yticks(range(0, len(ages), 3))
    ax.set_yticklabels([f"{int(ages[i])}" for i in range(0, len(ages), 3)], fontsize=6.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Age (years)")
    ax.set_title(title, fontsize=9)
    return im


def main() -> None:
    apply_style()
    dtr = json.loads(DTR_S2.read_text())["recovery"]
    tr = json.loads(TRANSFORMER_S2.read_text())["recovery"]

    ages = list(SURFACE_AGES)
    lags = list(SURFACE_LAGS_DAYS)

    R_oracle = _surface(dtr["theta0_true"], dtr["beta_true"], ages, lags)
    R_dtr = _surface(dtr["theta0_hat"], dtr["beta_hat"], ages, lags)
    R_base = _surface(tr["theta0_hat"], tr["beta_hat"], ages, lags)
    vmax = float(max(R_oracle.max(), R_dtr.max(), R_base.max()))

    # Panel D data
    rows = {r["scenario"]: r for r in csv.DictReader(CONTROLLED_CSV.open())}
    scenarios = ["S0", "S1", "S2", "S3"]
    shuffle = [float(rows[s]["shuffle_delta_bce"]) for s in scenarios]
    beta0 = [float(rows[s]["beta0_delta_bce"]) for s in scenarios]

    # Bootstrap error bars available for S2 only (patient bootstrap, one seed)
    boot = json.loads(BOOTSTRAP.read_text())["controlled_S2"]
    shuf_err = [0.0, 0.0, boot["delta_bce_shuffle"]["std"], 0.0]
    beta0_err = [0.0, 0.0, boot["delta_bce_beta0"]["std"], 0.0]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    # A Oracle
    im0 = _imshow_surface(axes[0, 0], R_oracle, ages, lags, 0, vmax, "Oracle")
    label_panel(axes[0, 0], "A")

    # B DTR
    im1 = _imshow_surface(axes[0, 1], R_dtr, ages, lags, 0, vmax, "DTR")
    label_panel(axes[0, 1], "B")
    # annotate recovery quality
    axes[0, 1].text(
        0.02,
        0.98,
        f"Surface RMSE={dtr['RMSE_surface']:.3f}\n"
        f"corr(λ)={dtr['corr_lambda']:.3f}",
        transform=axes[0, 1].transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="none"),
    )

    # C Original Transformer (parametric non-factorized baseline with reconstructible surface)
    # CEHR-BERT has lower CF Surface RMSE among black-box baselines but no saved gate surface.
    im2 = _imshow_surface(
        axes[1, 0],
        R_base,
        ages,
        lags,
        0,
        vmax,
        "Best baseline: Original Transformer",
    )
    label_panel(axes[1, 0], "C")
    axes[1, 0].text(
        0.02,
        0.98,
        f"Surface RMSE={tr['RMSE_surface']:.3f}\n"
        f"|β̂|={abs(tr['beta_hat']):.2f} (true 2.5)",
        transform=axes[1, 0].transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35, edgecolor="none"),
    )

    cax = fig.add_axes([0.92, 0.55, 0.018, 0.35])
    fig.colorbar(im1, cax=cax, label=r"Relevance $R(a,\tau)$")

    # D functional interventions
    ax = axes[1, 1]
    x = np.arange(len(scenarios))
    w = 0.36
    ax.bar(
        x - w / 2,
        shuffle,
        w,
        yerr=shuf_err,
        capsize=2.5,
        label=r"age-shuffle $\Delta$BCE",
        color=C_BLUE,
        error_kw=dict(ecolor="#2d3748", lw=0.8),
    )
    ax.bar(
        x + w / 2,
        beta0,
        w,
        yerr=beta0_err,
        capsize=2.5,
        label=r"$\beta{=}0$ $\Delta$BCE",
        color=C_ACCENT,
        error_kw=dict(ecolor="#2d3748", lw=0.8),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(r"$\Delta$BCE")
    ax.set_xlabel("Scenario")
    ax.axhline(0, color="#a0aec0", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Functional interventions")
    label_panel(ax, "D")
    ax.text(
        0.98,
        0.02,
        "Error bars: patient-bootstrap SD (S2 only)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#718096",
    )

    fig.subplots_adjust(left=0.08, right=0.90, top=0.94, bottom=0.08, wspace=0.35, hspace=0.40)
    save_fig(fig, "synthetic_mechanism_recovery")


if __name__ == "__main__":
    main()
