#!/usr/bin/env python3
"""Figures 7–12 for the follow-up mechanism investigation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_DIR, MAX_SEQ_LEN

FIG_DPI = 200


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def fig8_full_vs_visible(follow_dir: Path, fig_dir: Path) -> None:
    path = follow_dir / f"visible_oracle_summary_L{MAX_SEQ_LEN}.json"
    if not path.exists():
        return
    data = _load_json(path)
    scenarios = [s for s in ("S0", "S1", "S2", "S3") if s in data]
    metrics = []
    # Also try to pull neural deltas from followup runs.
    neural = {}
    run_root = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    if run_root.exists():
        for p in run_root.glob("followup_S*_age_temporal_*_interonly/metrics.json"):
            m = _load_json(p)
            neural[m["scenario"]] = m["ablations"].get("delta_bce_shuffle_age", np.nan)

    x = np.arange(len(scenarios))
    width = 0.25
    full = [data[s]["full_oracle"]["delta_bce_shuffle_age_interaction"] for s in scenarios]
    vis = [data[s]["visible_oracle"]["delta_bce_shuffle_age_interaction"] for s in scenarios]
    neu = [neural.get(s, np.nan) for s in scenarios]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width, full, width, label="full oracle", color="#2b6cb0")
    ax.bar(x, vis, width, label="visible oracle", color="#38a169")
    ax.bar(x + width, neu, width, label="neural (inter-only)", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(r"$\Delta$BCE age-shuffle (interaction labels)")
    ax.set_title("Figure 8 — Full vs visible oracle vs neural")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, fig_dir / "fig8_full_vs_visible_oracle")


def fig9_dilution(fig_dir: Path) -> None:
    run_root = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    if not run_root.exists():
        return
    rows = {"all": {}, "inter": {}}
    # Prior all-label runs (no interonly) and followup interonly.
    for p in run_root.rglob("metrics.json"):
        m = _load_json(p)
        if m.get("arm") != "age_temporal" or m.get("scenario") not in ("S2", "S3"):
            continue
        key = "inter" if m.get("interaction_only") else "all"
        # Prefer followup / longer runs when multiple exist.
        prev = rows[key].get(m["scenario"])
        if prev is None or len(m.get("history", [])) >= len(prev.get("history", [])):
            rows[key][m["scenario"]] = m

    scenarios = [s for s in ("S2", "S3") if s in rows["all"] or s in rows["inter"]]
    if not scenarios:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    titles = [r"$|\hat\beta|$", r"$\Delta$BCE $\beta{=}0$", "AUROC gain vs temporal"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = np.arange(len(scenarios))
    width = 0.35
    for i, key, label, color in (
        (0, "all", "all 32 labels", "#2b6cb0"),
        (1, "inter", "interaction only", "#c05621"),
    ):
        betas, d0, gains = [], [], []
        for s in scenarios:
            m = rows[key].get(s)
            if m is None:
                betas.append(np.nan)
                d0.append(np.nan)
                gains.append(np.nan)
                continue
            betas.append(abs(m.get("beta_hat", np.nan)))
            d0.append(m.get("ablations", {}).get("delta_bce_beta0", np.nan))
            # Gain needs temporal_only sibling — approximate from recovery AUROC alone.
            gains.append(m.get("test", {}).get("micro_auroc", np.nan))
        axes[0].bar(x + (i - 0.5) * width, betas, width, label=label, color=color)
        axes[1].bar(x + (i - 0.5) * width, d0, width, label=label, color=color)
        axes[2].bar(x + (i - 0.5) * width, gains, width, label=label, color=color)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 9 — Global model dilution (all labels vs interaction-only)", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig9_global_dilution")


def fig10_convergence(fig_dir: Path) -> None:
    run_root = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    if not run_root.exists():
        return
    cands = list(run_root.glob("followup_S2_age_temporal_*_interonly/metrics.json"))
    if not cands:
        return
    m = _load_json(cands[0])
    hist = m.get("history", [])
    if not hist or "mech_delta_bce_shuffle_age" not in hist[0]:
        return
    epochs = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.ravel()
    axes[0].plot(epochs, [h["beta_hat"] for h in hist], color="#2b6cb0")
    axes[0].axhline(m["beta_true"], color="#c05621", ls="--", label=r"$\beta_{\mathrm{true}}$")
    axes[0].set_ylabel(r"$\hat\beta$")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [h["mech_delta_bce_shuffle_age"] for h in hist], color="#2b6cb0")
    axes[1].set_ylabel(r"$\Delta$BCE shuffle")
    axes[2].plot(epochs, [h["mech_delta_bce_beta0"] for h in hist], color="#2b6cb0")
    axes[2].set_ylabel(r"$\Delta$BCE $\beta{=}0$")
    axes[3].plot(epochs, [h.get("mech_val_auprc", h.get("val_micro_auprc")) for h in hist], color="#2b6cb0")
    axes[3].set_ylabel("val AUPRC (interaction)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Figure 10 — Convergence (S2 interaction-only age_temporal)")
    fig.tight_layout()
    _save(fig, fig_dir / "fig10_convergence")


def fig11_global_vs_perhead(fig_dir: Path) -> None:
    run_root = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    if not run_root.exists():
        return
    arms = ["temporal_only", "age_temporal", "temporal_only_per_head", "age_temporal_per_head"]
    scenarios = ["S0", "S1", "S2", "S3"]
    data: dict[str, dict[str, float]] = {a: {} for a in arms}
    for p in run_root.glob("arch_*/metrics.json"):
        m = _load_json(p)
        if m.get("arm") in arms and m.get("scenario") in scenarios:
            data[m["arm"]][m["scenario"]] = m["test"]["micro_auroc"]

    x = np.arange(len(scenarios))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["#718096", "#2b6cb0", "#38a169", "#c05621"]
    for i, (arm, color) in enumerate(zip(arms, colors)):
        vals = [data[arm].get(s, np.nan) for s in scenarios]
        ax.bar(x + (i - 1.5) * width, vals, width, label=arm, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("micro AUROC")
    ax.set_title("Figure 11 — Global vs per-head recovery (arch runs)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, fig_dir / "fig11_global_vs_perhead")


def fig12_per_head_lambda(fig_dir: Path) -> None:
    run_root = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    if not run_root.exists():
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, scen in zip(axes, ("S2", "S3")):
        cands = list(run_root.glob(f"arch_{scen}_age_temporal_per_head_*/metrics.json"))
        if not cands:
            ax.set_title(scen)
            continue
        m = _load_json(cands[0])
        ph = m.get("recovery", {}).get("lambda_per_head", {})
        ages = [0, 1, 5, 10, 15, 18]
        for h, curve in ph.items():
            ys = [curve.get(str(a), np.nan) for a in ages]
            ax.plot(ages, ys, "-o", label=h)
        true = m.get("recovery", {}).get("lambda_true_by_age", {})
        if true:
            ax.plot(
                ages,
                [true.get(str(a), np.nan) for a in ages],
                "k--",
                lw=2,
                label=r"$\lambda_{\mathrm{true}}$",
            )
        ax.set_title(scen)
        ax.set_xlabel("Age (years)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"$\lambda_h(a)$")
    fig.suptitle("Figure 12 — Per-head learned decay curves")
    fig.tight_layout()
    _save(fig, fig_dir / "fig12_per_head_lambda")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow-dir", type=Path, default=DEFAULT_RESULTS_DIR / "followup")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_RESULTS_DIR / "figures")
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    # fig7 is written by audit_visibility; keep regenerating others.
    fig8_full_vs_visible(args.follow_dir, args.fig_dir)
    fig9_dilution(args.fig_dir)
    fig10_convergence(args.fig_dir)
    fig11_global_vs_perhead(args.fig_dir)
    fig12_per_head_lambda(args.fig_dir)
    print("Follow-up figures written to", args.fig_dir)


if __name__ == "__main__":
    main()
