#!/usr/bin/env python3
"""FIGURE 3: nch_developmental_behavior

Panel A: learned developmental relevance g(a,τ)=exp[-λ(a)τ] from NCH DTR checkpoint.
Panel B: micro-AUPRC by developmental age (DTR vs Temporal-only DTR) with bootstrap CIs.
Panel C: Recall@5 by controlled history truncation (DTR vs Temporal-only) with CIs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (  # noqa: E402
    C_DTR,
    C_TEMPORAL,
    REPO,
    apply_style,
    label_panel,
    save_fig,
)

NPZ = REPO / "figures/results/raw/nch_age_lag_kernel.npz"
SUBGROUP = REPO / "figures/results/nch_subgroup_performance.csv"
TRUNC = REPO / "artifacts/nch_stage2/analysis_age_temporal/tables/adkm_performance_by_truncation.csv"
DELTA_H = REPO / "artifacts/nch_stage2/analysis_age_temporal/tables/model_delta_by_horizon.csv"

AGE_ORDER = ["<1", "1-5", "6-11", "12-17"]
AGE_TICK = ["<1", "1–5", "6–11", "12–17"]


def main() -> None:
    apply_style()
    z = np.load(NPZ)
    ages = z["age_years"]
    lags = z["lag_days"]
    # Clinical NCH λ(a)<0, so K=−λτ>0. Report stores relevance e^K (grows with lag).
    # Plot log10(e^K)=K/ln(10) structure via K itself (age×lag temporal bias),
    # plus a side panel of row-normalized e^K (relative lag mass).
    K = z["K_adkm"]
    E = np.exp(K - K.max(axis=1, keepdims=True))
    W = E / E.sum(axis=1, keepdims=True)

    # Subgroup micro-AUPRC by age
    age_rows = []
    with SUBGROUP.open() as f:
        for r in csv.DictReader(f):
            if r["stratum_type"] == "age":
                age_rows.append(r)
    age_rows = sorted(age_rows, key=lambda r: AGE_ORDER.index(r["stratum"]))

    # Truncation curves: ADKM absolute + NINT = ADKM - delta
    trunc = list(csv.DictReader(TRUNC.open()))
    deltas = {
        r["horizon"]: float(r["delta_point"])
        for r in csv.DictReader(DELTA_H.open())
        if r["metric"] == "recall@5"
    }

    fig = plt.figure(figsize=(7.4, 7.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 1.0], hspace=0.45)
    gs0 = gs[0].subgridspec(1, 2, wspace=0.28)

    # ---- Panel A: two mini-heatmaps (K and row-normalized relevance) ----
    ax0a = fig.add_subplot(gs0[0, 0])
    ax0b = fig.add_subplot(gs0[0, 1])
    lag_tick_days = [1, 7, 30, 90, 180, 365, 1095, 3650]
    lag_tick_labels = ["1", "7", "30", "90", "180", "1y", "3y", "10y"]
    lag_tick_pos = [int(np.argmin(np.abs(lags - d))) for d in lag_tick_days]
    age_tick_yrs = [0, 3, 6, 9, 12, 15, 18]
    age_tick_pos = [int(np.argmin(np.abs(ages - a))) for a in age_tick_yrs]

    vmax_k = float(np.nanmax(np.abs(K)))
    im_k = ax0a.imshow(
        K,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=vmax_k,
        interpolation="nearest",
    )
    ax0a.set_xticks(lag_tick_pos)
    ax0a.set_xticklabels(lag_tick_labels, fontsize=6.5)
    ax0a.set_yticks(age_tick_pos)
    ax0a.set_yticklabels([str(a) for a in age_tick_yrs])
    ax0a.set_xlabel("Lag (days)")
    ax0a.set_ylabel("Age (years)")
    ax0a.set_title(r"$K(a,\tau)=-\lambda(a)\,\tau$")
    cbar_k = fig.colorbar(im_k, ax=ax0a, fraction=0.046, pad=0.04)
    cbar_k.set_label(r"$K$")

    im_w = ax0b.imshow(
        W,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    ax0b.set_xticks(lag_tick_pos)
    ax0b.set_xticklabels(lag_tick_labels, fontsize=6.5)
    ax0b.set_yticks(age_tick_pos)
    ax0b.set_yticklabels([str(a) for a in age_tick_yrs])
    ax0b.set_xlabel("Lag (days)")
    ax0b.set_ylabel("Age (years)")
    ax0b.set_title(r"Row-norm $e^{K}$  ($\propto e^{-\lambda\tau}$)")
    cbar_w = fig.colorbar(im_w, ax=ax0b, fraction=0.046, pad=0.04)
    cbar_w.set_label("Relative mass")
    label_panel(ax0a, "A", x=-0.12, y=1.14)

    # ---- Panel B age performance ----
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(len(age_rows))
    w = 0.36
    adkm = [float(r["adkm_micro_auprc"]) for r in age_rows]
    nint = [float(r["nint_micro_auprc"]) for r in age_rows]
    adkm_err = np.array(
        [
            [float(r["adkm_micro_auprc"]) - float(r["adkm_ci_lo"]), float(r["adkm_ci_hi"]) - float(r["adkm_micro_auprc"])]
            for r in age_rows
        ]
    ).T
    nint_err = np.array(
        [
            [float(r["nint_micro_auprc"]) - float(r["nint_ci_lo"]), float(r["nint_ci_hi"]) - float(r["nint_micro_auprc"])]
            for r in age_rows
        ]
    ).T
    ax1.bar(
        x - w / 2,
        adkm,
        w,
        yerr=adkm_err,
        capsize=2.5,
        label="DTR",
        color=C_DTR,
        error_kw=dict(ecolor="#2d3748", lw=0.8),
    )
    ax1.bar(
        x + w / 2,
        nint,
        w,
        yerr=nint_err,
        capsize=2.5,
        label="Temporal-only DTR",
        color=C_TEMPORAL,
        error_kw=dict(ecolor="#2d3748", lw=0.8),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(AGE_TICK)
    ax1.set_ylabel("Micro-AUPRC")
    ax1.set_xlabel("Developmental age (years)")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_title("NCH performance by developmental age")
    label_panel(ax1, "B", x=-0.06, y=1.12)
    # n patients annotation
    for i, r in enumerate(age_rows):
        ax1.text(i, 0.01, f"n={r['n_patients']}", ha="center", va="bottom", fontsize=5.5, color="#718096")

    # ---- Panel C history horizon ----
    ax2 = fig.add_subplot(gs[2])
    # Use available truncation horizons (no 730d in artifacts; use 30/90/180/365/1095/full)
    labels = []
    dtr_y, dtr_lo, dtr_hi = [], [], []
    to_y = []
    for r in trunc:
        h = r["horizon"]
        labels.append("full" if h == "full" else h)
        dtr_y.append(float(r["recall@5"]))
        dtr_lo.append(float(r["recall@5_ci_lo"]))
        dtr_hi.append(float(r["recall@5_ci_hi"]))
        to_y.append(float(r["recall@5"]) - deltas[h])

    xs = np.arange(len(labels))
    dtr_y = np.asarray(dtr_y)
    dtr_lo = np.asarray(dtr_lo)
    dtr_hi = np.asarray(dtr_hi)
    to_y = np.asarray(to_y)
    ax2.errorbar(
        xs,
        dtr_y,
        yerr=[dtr_y - dtr_lo, dtr_hi - dtr_y],
        fmt="o-",
        color=C_DTR,
        lw=1.6,
        ms=5,
        capsize=2.5,
        label="DTR",
    )
    ax2.plot(xs, to_y, "s--", color=C_TEMPORAL, lw=1.6, ms=5, label="Temporal-only DTR")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Available history horizon")
    ax2.set_ylabel("Recall@5")
    ax2.legend(frameon=False, loc="lower right")
    ax2.set_title("NCH performance by history truncation")
    label_panel(ax2, "C", x=-0.06, y=1.12)
    ax2.set_ylim(0.30, 0.45)

    fig.subplots_adjust(left=0.10, right=0.96, top=0.96, bottom=0.06)
    save_fig(fig, "nch_developmental_behavior")


if __name__ == "__main__":
    main()
