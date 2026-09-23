"""Analytic age×temporal kernel: lambda(a), K(a,lag), Stage-1/init comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch

from stage2_nch.analysis import ANALYSIS, ADKM_DIR, STAGE1_BEST, PRIMARY_CKPT_NAME, write_contract
from stage2_nch.config import (
    PEDIATRIC_AGE_CENTER_YEARS,
    PEDIATRIC_AGE_SCALE_YEARS,
    WEEK_DAYS,
)

AGE_BANDS = [
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.0),
]
REP_AGES = (0.5, 2.0, 5.0, 8.0, 12.0, 16.0)
# Lag grid in days (log-spaced + clinical anchors)
LAG_ANCHORS_DAYS = np.array([0, 1, 7, 30, 90, 180, 365, 730, 1825, 3650], dtype=np.float64)


def lag_to_tau(delta_t_days: np.ndarray) -> np.ndarray:
    """Exact Stage-2/1 transform: tau = log1p(|dt| / 7)."""
    return np.log1p(np.abs(delta_t_days) / WEEK_DAYS)


def z_p(age_years: np.ndarray, center: float = 9.0, scale: float = 9.0) -> np.ndarray:
    return (age_years - center) / scale


def z_adult(age_years: np.ndarray, mean: float, sd: float) -> np.ndarray:
    return (age_years - mean) / sd


def lambda_of(age_years: np.ndarray, lambda0: float, beta: float,
              center: float, scale: float) -> np.ndarray:
    return lambda0 + beta * z_p(age_years, center, scale)


def K_of(age_years: np.ndarray, lag_days: np.ndarray, lambda0: float, beta: float,
         center: float, scale: float) -> np.ndarray:
    """K(a, lag) = -lambda(a) * tau(lag). Broadcast ages [A] x lags [T] -> [A,T]."""
    a = np.asarray(age_years, dtype=np.float64).reshape(-1)
    t = np.asarray(lag_days, dtype=np.float64).reshape(-1)
    lam = lambda_of(a, lambda0, beta, center, scale)[:, None]
    tau = lag_to_tau(t)[None, :]
    return -lam * tau


def load_temporal_params(ckpt_path: Path) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    return {
        "path": str(ckpt_path),
        "kind": ck.get("kind"),
        "epoch": ck.get("epoch"),
        "val_bce": ck.get("val_bce"),
        "val_micro_auprc": ck.get("val_micro_auprc"),
        "lambda0": float(sd["temporal.lambda0"].reshape(-1)[0]),
        "beta": float(sd["temporal.beta"].reshape(-1)[0]),
        "age_mean": float(sd["temporal.age_mean"].reshape(-1)[0]),
        "age_sd": float(sd["temporal.age_sd"].reshape(-1)[0]),
    }


def _shade_age_bands(ax):
    colors = ["#f0f0f0", "#e8e8e8", "#f0f0f0", "#e8e8e8"]
    for (name, lo, hi), c in zip(AGE_BANDS, colors):
        ax.axvspan(lo, min(hi, 18), color=c, zorder=0)
        ax.text((lo + min(hi, 18)) / 2, ax.get_ylim()[1], name, ha="center", va="bottom",
                fontsize=8, color="0.35")


def _savefig(fig, stem: str, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_abc(
    *,
    adkm_dir: Path = ADKM_DIR,
    out_dir: Path = ANALYSIS,
    primary_name: str = PRIMARY_CKPT_NAME,
    max_lag_days: float = 3650.0,
) -> dict:
    write_contract(out_dir / "raw")
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    raw_dir = out_dir / "raw"
    for d in (fig_dir, tab_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    final = load_temporal_params(adkm_dir / primary_name)
    # Stage-2 initialization (documented): lambda0 from Stage-1, beta=0, z_P(9,9)
    s1 = load_temporal_params(STAGE1_BEST)
    init = {
        "lambda0": s1["lambda0"],
        "beta": 0.0,
        "age_mean": PEDIATRIC_AGE_CENTER_YEARS,
        "age_sd": PEDIATRIC_AGE_SCALE_YEARS,
        "note": "Stage-2 init after transfer: keep lambda0, reset beta→0, age→z_P(9,9)",
    }
    # Also load final epoch for comparison table
    finals = {
        "best_auprc": final,
        "best_bce": load_temporal_params(adkm_dir / "checkpoint_best_bce.pt"),
        "checkpoint_final": load_temporal_params(adkm_dir / "checkpoint_final.pt"),
        "stage1_adult": s1,
        "stage2_init": init,
    }
    (raw_dir / "temporal_params.json").write_text(json.dumps(finals, indent=2) + "\n")

    ages = np.arange(0.0, 18.0 + 1e-9, 0.1)
    lam_f = lambda_of(ages, final["lambda0"], final["beta"], final["age_mean"], final["age_sd"])
    lam_i = lambda_of(ages, init["lambda0"], init["beta"], init["age_mean"], init["age_sd"])
    # Adult Stage-1 function evaluated on pediatric ages (extrapolation reference only)
    lam_s1 = s1["lambda0"] + s1["beta"] * z_adult(ages, s1["age_mean"], s1["age_sd"])

    df_lam = pd.DataFrame({
        "age": ages,
        "lambda_final": lam_f,
        "lambda_init": lam_i,
        "delta_lambda": lam_f - lam_i,
        "lambda_stage1_adult_extrapolated": lam_s1,
        "z_P": z_p(ages, final["age_mean"], final["age_sd"]),
    })
    # Scalar shared across heads — no head-specific columns
    df_lam["note"] = "lambda0/beta are single scalars shared across heads (not per-head)"
    df_lam.to_csv(tab_dir / "lambda_by_age.csv", index=False)

    # --- Figure A1 ---
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2]})
    ax = axes[0]
    ax.plot(ages, lam_i, ls="--", color="0.45", lw=1.8,
            label=f"Stage-2 init (λ₀={init['lambda0']:.3f}, β=0)")
    ax.plot(ages, lam_f, color="#1f4e79", lw=2.2,
            label=f"Stage-2 final ({primary_name}; λ₀={final['lambda0']:.3f}, β={final['beta']:.3f})")
    ax.plot(ages, lam_s1, ls=":", color="#a64d00", lw=1.4,
            label="Stage-1 adult λ(a) on pediatric ages (extrapolation)")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_ylabel(r"$\lambda(a)=\lambda_0+\beta\,z_P(a)$")
    ax.set_title("Learned age-conditioned temporal parameter (scalar, shared across heads)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, 18)
    _shade_age_bands(ax)
    ax = axes[1]
    ax.plot(ages, lam_f - lam_i, color="#1f4e79", lw=2.0)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\Delta\lambda(a)=\lambda_{\mathrm{final}}-\lambda_{\mathrm{init}}$")
    ax.set_xlim(0, 18)
    _shade_age_bands(ax)
    fig.tight_layout()
    _savefig(fig, "fig_lambda_by_age", fig_dir)

    # --- Figure A2: temporal relevance curves ---
    lags = np.unique(np.concatenate([
        LAG_ANCHORS_DAYS,
        np.geomspace(1.0, max_lag_days, 200),
        np.array([0.0]),
    ]))
    lags.sort()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(REP_AGES)))
    for age, color in zip(REP_AGES, cmap):
        K = K_of(np.array([age]), lags, final["lambda0"], final["beta"],
                 final["age_mean"], final["age_sd"])[0]
        axes[0].plot(lags, K, color=color, lw=1.8, label=f"{age:g} y")
        axes[1].plot(lags, np.exp(K), color=color, lw=1.8, label=f"{age:g} y")
    for ax, ylab, title in zip(
        axes,
        [r"Raw kernel bias $K(a,\Delta t)=-\lambda(a)\,\tau(\Delta t)$",
         r"Multiplicative relevance $\mathrm{e}^{K(a,\Delta t)}$"],
        ["Temporal logit bias vs lag", "Positive temporal weight vs lag"],
    ):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Lag Δt (days; symlog)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8, title="Query age")
        ax.axvline(7, color="0.8", ls=":", lw=0.8)
        ax.axvline(365, color="0.8", ls=":", lw=0.8)
    fig.suptitle(
        r"At fixed content logits: larger $e^{K}$ raises pre-softmax score for that lag "
        r"($\tau=\log(1+|\Delta t|/7)$)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, "fig_temporal_curves_selected_ages", fig_dir)

    curves = []
    for age in REP_AGES:
        K = K_of(np.array([age]), lags, final["lambda0"], final["beta"],
                 final["age_mean"], final["age_sd"])[0]
        for lag, k in zip(lags, K):
            curves.append({"age": age, "lag_days": lag, "tau": lag_to_tau(np.array([lag]))[0],
                           "K": k, "exp_K": float(np.exp(k))})
    pd.DataFrame(curves).to_csv(tab_dir / "temporal_curves_selected_ages.csv", index=False)

    # --- Figure B1 heatmaps ---
    age_grid = np.arange(0.0, 18.0 + 1e-9, 0.25)
    lag_grid = np.unique(np.concatenate([
        LAG_ANCHORS_DAYS[LAG_ANCHORS_DAYS <= max_lag_days],
        np.geomspace(1.0, max_lag_days, 80),
    ]))
    lag_grid.sort()
    Kmat = K_of(age_grid, lag_grid, final["lambda0"], final["beta"],
                final["age_mean"], final["age_sd"])
    Emat = np.exp(Kmat)
    # Row-normalize exp(K) shape (relative across lags at fixed age)
    row = Emat / np.maximum(Emat.sum(axis=1, keepdims=True), 1e-12)

    def _heat(mat, stem, cbar_label, cmap="coolwarm", center0=False):
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        if center0:
            vmax = np.nanmax(np.abs(mat))
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap,
                           vmin=-vmax, vmax=vmax,
                           extent=[0, len(lag_grid) - 1, age_grid[0], age_grid[-1]])
        else:
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap,
                           extent=[0, len(lag_grid) - 1, age_grid[0], age_grid[-1]])
        # tick anchors
        tick_lags = [1, 7, 30, 90, 180, 365, 730, 1825, 3650]
        tick_pos, tick_lab = [], []
        for tl in tick_lags:
            if tl > lag_grid[-1]:
                continue
            j = int(np.argmin(np.abs(lag_grid - tl)))
            tick_pos.append(j)
            tick_lab.append({1: "1d", 7: "7d", 30: "30d", 90: "90d", 180: "180d",
                             365: "1y", 730: "2y", 1825: "5y", 3650: "10y"}[tl])
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.set_ylabel("Age (years)")
        ax.set_xlabel("Historical lag")
        fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
        for _, lo, hi in AGE_BANDS:
            ax.axhline(lo, color="k", lw=0.3, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, stem, fig_dir)

    _heat(Kmat, "fig_age_lag_kernel_heatmap", r"$K(a,\Delta t)=-\lambda(a)\tau$", center0=True)
    _heat(Emat, "fig_age_lag_expK_heatmap", r"$e^{K(a,\Delta t)}$", cmap="viridis")
    _heat(row, "fig_age_lag_rownorm_heatmap", "row-normalized $e^{K}$", cmap="viridis")

    # save matrices
    np.savez(raw_dir / "age_lag_kernel.npz", age=age_grid, lag_days=lag_grid,
             K=Kmat, exp_K=Emat, row_norm_exp_K=row)
    pd.DataFrame(Kmat, index=np.round(age_grid, 4), columns=np.round(lag_grid, 4)).to_csv(
        tab_dir / "age_lag_K.csv")

    # --- Figure C: pediatric shift from init ---
    K_init = K_of(age_grid, lag_grid, init["lambda0"], init["beta"], init["age_mean"], init["age_sd"])
    dK = Kmat - K_init
    dlam = lam_f - lam_i
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(ages, dlam, color="#1f4e79", lw=2)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\lambda_{\mathrm{final}}-\lambda_{\mathrm{init}}$")
    ax.set_title("Change in λ(a) from Stage-2 initialization")
    ax.set_xlim(0, 18)
    _shade_age_bands(ax)
    fig.tight_layout()
    _savefig(fig, "fig_pediatric_shift_lambda", fig_dir)

    _heat(dK, "fig_pediatric_shift_from_pretraining",
          r"$K_{\mathrm{final}}-K_{\mathrm{init}}$", center0=True)

    # Exclude lag=0 (tau=0 ⇒ K≡0 ⇒ ΔK≡0) when locating extrema
    lag_pos = lag_grid > 0
    dK_pos = dK[:, lag_pos]
    lags_pos = lag_grid[lag_pos]
    ai_max, li_max = np.unravel_index(int(np.nanargmax(dK_pos)), dK_pos.shape)
    ai_min, li_min = np.unravel_index(int(np.nanargmin(dK_pos)), dK_pos.shape)
    shift_summary = {
        "primary_checkpoint": primary_name,
        "final_params": final,
        "init_params": init,
        "delta_lambda_range": [float(dlam.min()), float(dlam.max())],
        "delta_K_max": {
            "value": float(dK_pos[ai_max, li_max]),
            "age": float(age_grid[ai_max]),
            "lag_days": float(lags_pos[li_max]),
        },
        "delta_K_min": {
            "value": float(dK_pos[ai_min, li_min]),
            "age": float(age_grid[ai_min]),
            "lag_days": float(lags_pos[li_min]),
        },
        "interpretation_note": (
            "Init has beta=0 so K_init is age-independent. Final beta>0 with lambda0<0 "
            "makes |lambda| decrease with age (lambda remains negative), so long-range "
            "bias weakens toward adolescence relative to Stage-2 initialization."
        ),
    }
    (raw_dir / "pediatric_shift_summary.json").write_text(
        json.dumps(shift_summary, indent=2) + "\n")

    return {"params": finals, "shift": shift_summary, "ages": ages.tolist()}
