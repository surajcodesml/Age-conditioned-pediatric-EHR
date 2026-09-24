"""Paper-ready figures for final DTR validation (PNG + SVG)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import SURFACE_AGES, SURFACE_LAGS_DAYS, lambda_true, relevance, tau_from_days


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def fig_a_mechanism(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.2, "Synthea\npediatric history"),
        (2.3, 1.2, "Historical\nencounter"),
        (4.3, 1.2, "Lag from\nprediction cutoff"),
        (6.3, 1.8, "Current\ndevelopmental age"),
        (8.0, 1.2, "Age-conditioned\ndecay → outcome"),
    ]
    for x, y, t in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), 1.6, 0.9, fill=True, facecolor="#edf2f7", edgecolor="#2d3748", lw=1.2)
        )
        ax.text(x + 0.8, y + 0.45, t, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(2.3, 1.65), xytext=(1.9, 1.65), arrowprops=dict(arrowstyle="->", color="#2d3748"))
    ax.annotate("", xy=(4.3, 1.65), xytext=(3.9, 1.65), arrowprops=dict(arrowstyle="->", color="#2d3748"))
    ax.annotate("", xy=(8.0, 1.65), xytext=(5.9, 1.65), arrowprops=dict(arrowstyle="->", color="#2d3748"))
    ax.annotate("", xy=(8.0, 1.9), xytext=(7.1, 2.25), arrowprops=dict(arrowstyle="->", color="#c05621"))
    ax.set_title("Figure A — Benchmark developmental age × temporal mechanism")
    _save(fig, fig_dir / "figA_benchmark_mechanism")


def fig_b_ladder(ladder: dict[str, dict], fig_dir: Path) -> None:
    names = list(ladder.keys())
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    x = np.arange(len(names))
    for ax, key, title in zip(
        axes,
        ["delta_auroc", "delta_bce_shuffle", "delta_bce_beta0"],
        [r"$\Delta$AUROC vs temporal control", r"$\Delta$BCE age-shuffle", r"$\Delta$BCE $\beta{=}0$"],
    ):
        vals = [ladder[n].get(key, np.nan) for n in names]
        colors = ["#718096" if "Transformer" in n else "#2b6cb0" for n in names]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="#a0aec0", lw=0.8)
    fig.suptitle("Figure B — Architecture evolution / mechanism recovery", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "figB_architecture_ladder")


def fig_c_lambda(rec_s2: dict, rec_s3: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, rec, title, beta in zip(
        axes, [rec_s2, rec_s3], ["S2 (developmental)", "S3 (reversed)"], [-2.5, 2.5]
    ):
        ages = sorted(float(a) for a in rec.get("lambda_true_by_age", {}).keys())
        if not ages:
            ages = list(SURFACE_AGES)
        lt = [rec["lambda_true_by_age"][str(a)] for a in ages]
        ll = [rec["lambda_learned_by_age"][str(a)] for a in ages]
        ax.plot(ages, lt, "k--", lw=2, label=r"$\lambda_{\mathrm{true}}$")
        ax.plot(ages, ll, color="#c05621", lw=2, label=r"$\lambda_{\mathrm{learned}}$")
        ax.set_xlabel("Age (years)")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$\lambda(a)$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure C — True vs learned developmental decay", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "figC_lambda_curves")


def fig_d_surface(rec: dict, fig_dir: Path) -> None:
    ages = sorted(float(a) for a in rec.get("lambda_true_by_age", {}).keys()) or list(SURFACE_AGES)
    beta_t, theta_t = rec["beta_true"], rec["theta0_true"]
    beta_h, theta_h = rec["beta_hat"], rec["theta0_hat"]
    lags = list(SURFACE_LAGS_DAYS)
    R_t = np.zeros((len(lags), len(ages)))
    R_l = np.zeros_like(R_t)
    for i, d in enumerate(lags):
        t = float(tau_from_days(d))
        for j, a in enumerate(ages):
            R_t[i, j] = relevance(a, t, theta_t, beta_t)
            lam = float(lambda_true(a, theta_h, beta_h))
            R_l[i, j] = np.exp(-lam * t)
    err = np.abs(R_l - R_t)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    vmax = max(float(R_t.max()), float(R_l.max()), 1e-6)
    for ax, M, title, cmap, vm in zip(
        axes,
        [R_t, R_l, err],
        ["Ground truth", "Learned", "Absolute error"],
        ["viridis", "viridis", "magma"],
        [vmax, vmax, None],
    ):
        im = ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            extent=[ages[0], ages[-1], 0, len(lags) - 1],
            cmap=cmap,
            vmin=0,
            vmax=vm,
        )
        ax.set_yticks(range(len(lags)))
        ax.set_yticklabels([f"{int(d)}d" for d in lags], fontsize=7)
        ax.set_xlabel("Age")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Figure D — Age × lag relevance surface (DTR, S2)", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "figD_age_lag_surface")


def fig_e_ablations(by_scenario: dict[str, dict], fig_dir: Path) -> None:
    scenarios = ["S0", "S1", "S2", "S3"]
    x = np.arange(len(scenarios))
    w = 0.35
    shuf = [by_scenario.get(s, {}).get("delta_bce_shuffle", np.nan) for s in scenarios]
    b0 = [by_scenario.get(s, {}).get("delta_bce_beta0", np.nan) for s in scenarios]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - w / 2, shuf, w, label=r"age-shuffle $\Delta$BCE", color="#2b6cb0")
    ax.bar(x + w / 2, b0, w, label=r"$\beta{=}0$ $\Delta$BCE", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(r"$\Delta$BCE")
    ax.axhline(0, color="#a0aec0", lw=0.8)
    ax.legend(frameon=False)
    ax.set_title("Figure E — Functional ablations by scenario (final DTR)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, fig_dir / "figE_ablations_by_scenario")


def fig_f_controlled_vs_full(controlled: dict, full: dict, fig_dir: Path) -> None:
    metrics = ["delta_auroc", "delta_bce_shuffle", "delta_bce_beta0", "RMSE_surface"]
    labels = [r"$\Delta$AUROC", r"shuffle $\Delta$BCE", r"$\beta{=}0$ $\Delta$BCE", "surface RMSE"]
    x = np.arange(len(metrics))
    w = 0.35
    c_vals = [controlled.get(k, np.nan) for k in metrics]
    f_vals = [full.get(k, np.nan) for k in metrics]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2, c_vals, w, label="controlled", color="#2b6cb0")
    ax.bar(x + w / 2, f_vals, w, label="full-realism", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.set_title("Figure F — Controlled vs full-realism (DTR, S2)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, fig_dir / "figF_controlled_vs_full")


def fig_g_strength(rows: list[dict], fig_dir: Path) -> None:
    if not rows:
        return
    xs = [abs(r["beta_true"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].plot(xs, [abs(r["beta_hat"]) for r in rows], "o-", color="#2b6cb0")
    axes[0].plot(xs, xs, "k--", lw=1, alpha=0.5, label="identity")
    axes[0].set_xlabel(r"$|\beta_{\mathrm{true}}|$")
    axes[0].set_ylabel(r"$|\hat\beta|$")
    axes[0].set_title("Learned interaction strength")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(xs, [r["delta_bce_shuffle"] for r in rows], "o-", color="#2b6cb0", label="shuffle")
    axes[1].plot(xs, [r["delta_bce_beta0"] for r in rows], "s-", color="#c05621", label=r"$\beta{=}0$")
    axes[1].set_xlabel(r"$|\beta_{\mathrm{true}}|$")
    axes[1].set_ylabel(r"$\Delta$BCE")
    axes[1].set_title("Functional ablation strength")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Figure G — Interaction-strength sensitivity (DTR, S2)", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "figG_strength_sensitivity")


def fig_h_counterfactual(cf: dict, fig_dir: Path) -> None:
    ages = cf.get("ages", [2, 5, 9, 13, 17])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, style in [
        ("oracle_history", ("k--", "Oracle history contrib.")),
        ("dtr_temporal_only_history", ("#718096", "DTR temporal-only history")),
        ("dtr_age_temporal_history", ("#c05621", "DTR age-temporal history")),
    ]:
        if key not in cf:
            continue
        ax.plot(ages, cf[key], style[0] if isinstance(style[0], str) and style[0].startswith("#") else style[0],
                lw=2, label=style[1], linestyle="--" if "oracle" in key else "-")
    # fix styles
    ax.clear()
    if "oracle_history" in cf:
        ax.plot(ages, cf["oracle_history"], "k--", lw=2, label="Oracle history contrib.")
    if "dtr_temporal_only_history" in cf:
        ax.plot(ages, cf["dtr_temporal_only_history"], color="#718096", lw=2, label="DTR temporal-only history")
    if "dtr_age_temporal_history" in cf:
        ax.plot(ages, cf["dtr_age_temporal_history"], color="#c05621", lw=2, label="DTR age-temporal history")
    ax.set_xlabel(r"Counterfactual cutoff age $a_*$")
    ax.set_ylabel("Mean history logit (interaction labels)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure H — Counterfactual developmental retrieval")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, fig_dir / "figH_counterfactual_retrieval")
