#!/usr/bin/env python3
"""Publication-quality figures for the synthetic age × temporal benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    DEFAULT_RESULTS_DIR,
    PROBE_AGES,
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    lambda_true,
    relevance,
    tau_from_days,
)

FIG_DPI = 200


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def fig1_mechanism(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_axis_off()
    steps = [
        "Synthea pediatric longitudinal history",
        "historical signal event",
        "lag Δt from prediction cutoff t*",
        "τ = log(1 + Δt/7)",
        "developmental age a* at cutoff",
        "z(a*) = (a* − 9)/9",
        "λ_true(a*) = softplus(θ₀ + β z(a*))",
        "R(a*, τ) = exp(−λ_true τ)",
        "future multi-label target Y",
    ]
    y = 0.92
    for i, s in enumerate(steps):
        ax.text(0.5, y, s, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f1fa", edgecolor="#2c5282"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.09), xytext=(0.5, y - 0.02),
                        arrowprops=dict(arrowstyle="->", color="#2c5282", lw=1.5))
        y -= 0.10
    ax.set_title("Figure 1 — Benchmark mechanism", fontsize=14, pad=12)
    _save(fig, out_dir / "fig1_benchmark_mechanism")


def fig2_decay(
    out_dir: Path,
    *,
    theta0: float,
    beta_true: float,
    learned_curves: list[dict[str, float]] | None = None,
    title_suffix: str = "S2",
) -> None:
    ages = np.linspace(0, 18, 200)
    lam_t = lambda_true(ages, theta0, beta_true)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ages, lam_t, color="#1a365d", lw=2.5, label=r"$\lambda_{\mathrm{true}}(a)$")
    if learned_curves:
        mat = []
        for c in learned_curves:
            mat.append([c[str(a)] for a in SURFACE_AGES])
        mat = np.asarray(mat, dtype=np.float64)
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0)
        ax.plot(SURFACE_AGES, mu, color="#c05621", lw=2, label=r"$\lambda_{\mathrm{learned}}(a)$")
        ax.fill_between(SURFACE_AGES, mu - sd, mu + sd, color="#c05621", alpha=0.25)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\lambda(a)=\mathrm{softplus}(\theta_0+\beta z(a))$")
    ax.set_title(f"Figure 2 — True vs learned decay ({title_suffix})")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir / f"fig2_decay_{title_suffix}")


def fig3_surfaces(
    out_dir: Path,
    *,
    theta0: float,
    beta_true: float,
    theta0_hat: float,
    beta_hat: float,
    title_suffix: str = "S2",
) -> None:
    ages = np.asarray(SURFACE_AGES, dtype=np.float64)
    lags = np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64)
    taus = tau_from_days(lags)
    A, T = np.meshgrid(ages, taus, indexing="xy")
    # Note meshgrid ages along columns.
    R_true = np.zeros((len(lags), len(ages)))
    R_learn = np.zeros_like(R_true)
    for i, t in enumerate(taus):
        for j, a in enumerate(ages):
            R_true[i, j] = relevance(a, t, theta0, beta_true)
            lam = float(lambda_true(a, theta0_hat, beta_hat))  # softplus form with hats as raw params?
            # Learned λ uses softplus(θ̂ + β̂ z); lambda_true does that.
            R_learn[i, j] = float(np.exp(-lam * t))
    err = np.abs(R_learn - R_true)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, M, title, cmap in zip(
        axes,
        [R_true, R_learn, err],
        ["Ground truth", "Learned", "Absolute error"],
        ["viridis", "viridis", "magma"],
    ):
        im = ax.imshow(
            M,
            aspect="auto",
            origin="lower",
            extent=[ages.min(), ages.max(), 0, len(lags) - 1],
            cmap=cmap,
        )
        ax.set_yticks(range(len(lags)))
        ax.set_yticklabels([f"{int(d)}d" for d in lags])
        ax.set_xlabel("Age (years)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("Lag")
    fig.suptitle(f"Figure 3 — Age × lag relevance surfaces ({title_suffix})", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / f"fig3_surfaces_{title_suffix}")


def fig4_performance(out_dir: Path, results: dict[str, dict[str, float]], metric: str = "micro_auroc") -> None:
    scenarios = ["S0", "S1", "S2", "S3"]
    arms = ["no_age", "age_only", "temporal_only", "age_temporal"]
    x = np.arange(len(scenarios))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, arm in enumerate(arms):
        vals = [results.get(f"{s}/{arm}", {}).get(metric, np.nan) for s in scenarios]
        ax.bar(x + (i - 1.5) * width, vals, width, label=arm)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(metric)
    ax.set_title("Figure 4 — Model performance by scenario")
    ax.legend(frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir / "fig4_performance_by_scenario")


def fig5_ablations(out_dir: Path, ablations: dict[str, dict[str, Any]]) -> None:
    scenarios = [s for s in ("S2", "S3") if s in ablations]
    if not scenarios:
        return
    modes = ["normal", "shuffle_age", "beta0", "constant_age"]
    labels = ["normal", "shuffled age", "β=0", "constant age"]
    x = np.arange(len(modes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, s in enumerate(scenarios):
        vals = [ablations[s].get(m, {}).get("micro_auroc", np.nan) for m in modes]
        ax.bar(x + (i - 0.5) * width, vals, width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("micro AUROC")
    ax.set_title("Figure 5 — Mechanism ablations (age_temporal)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir / "fig5_mechanism_ablations")


def fig6_strength_sweep(out_dir: Path, sweep: list[dict[str, Any]]) -> None:
    if not sweep:
        return
    strengths = [r["strength"] for r in sweep]
    gain = [r.get("auroc_gain", np.nan) for r in sweep]
    rmse = [r.get("RMSE_surface", np.nan) for r in sweep]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    ax1.plot(strengths, gain, "o-", color="#2b6cb0", label="AUROC gain vs temporal_only")
    ax2.plot(strengths, rmse, "s--", color="#c05621", label="Surface RMSE")
    ax1.set_xlabel(r"Interaction strength $|\beta_{\mathrm{true}}|$")
    ax1.set_ylabel("AUROC gain")
    ax2.set_ylabel("Surface RMSE")
    ax1.set_title("Figure 6 — Interaction-strength sensitivity (S2)")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, frameon=False)
    _save(fig, out_dir / "fig6_interaction_strength")


def generate_all(
    out_dir: Path,
    *,
    metrics_root: Path | None = None,
    theta0: float = 0.0,
    beta_s2: float = -2.0,
    beta_s3: float = 2.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1_mechanism(out_dir)
    fig2_decay(out_dir, theta0=theta0, beta_true=beta_s2, title_suffix="S2")
    fig2_decay(out_dir, theta0=theta0, beta_true=beta_s3, title_suffix="S3")
    fig3_surfaces(
        out_dir,
        theta0=theta0,
        beta_true=beta_s2,
        theta0_hat=theta0,
        beta_hat=beta_s2,
        title_suffix="S2_oracle",
    )

    results: dict[str, dict[str, float]] = {}
    ablations: dict[str, dict[str, Any]] = {}
    sweep: list[dict[str, Any]] = []
    if metrics_root and metrics_root.exists():
        for path in metrics_root.rglob("metrics.json"):
            with path.open() as f:
                m = json.load(f)
            key = f"{m['scenario']}/{m['arm']}"
            results[key] = m.get("test", {})
            if m["arm"] == "age_temporal" and m["scenario"] in ("S2", "S3"):
                ablations[m["scenario"]] = m.get("ablations", {})
                if m["scenario"] == "S2":
                    # Fill learned curve into fig2 when available.
                    rec = m.get("recovery", {})
                    if rec.get("lambda_learned_by_age"):
                        fig2_decay(
                            out_dir,
                            theta0=theta0,
                            beta_true=m.get("beta_true", beta_s2),
                            learned_curves=[rec["lambda_learned_by_age"]],
                            title_suffix=f"{m['scenario']}_learned",
                        )
                        fig3_surfaces(
                            out_dir,
                            theta0=m.get("theta0_true", theta0),
                            beta_true=m.get("beta_true", beta_s2),
                            theta0_hat=rec.get("theta0_hat", 0.0),
                            beta_hat=rec.get("beta_hat", 0.0),
                            title_suffix=m["scenario"],
                        )
            if m["scenario"] == "S2" and m["arm"] == "age_temporal":
                temp = results.get("S2/temporal_only", {})
                sweep.append(
                    {
                        "strength": m.get("strength", "medium"),
                        "auroc_gain": m.get("test", {}).get("micro_auroc", np.nan)
                        - temp.get("micro_auroc", np.nan),
                        "RMSE_surface": m.get("recovery", {}).get("RMSE_surface", np.nan),
                    }
                )

    fig4_performance(out_dir, results)
    fig5_ablations(out_dir, ablations)
    fig6_strength_sweep(out_dir, sweep)
    print("Figures written to", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR / "figures")
    ap.add_argument("--metrics-root", type=Path, default=None)
    args = ap.parse_args()
    generate_all(args.out_dir, metrics_root=args.metrics_root)


if __name__ == "__main__":
    main()
