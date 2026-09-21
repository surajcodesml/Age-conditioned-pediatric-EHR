"""Plots required for every Stage-1 run."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from stage1_mimic_pretrain.config import PROBE_AGES_YEARS


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_run_plots(run_dir: Path, history: list[dict], age_tests: dict | None) -> list[Path]:
    plt = _setup_mpl()
    run_dir = Path(run_dir)
    fig_dir = run_dir / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not history:
        return written
    epochs = [int(r["epoch"]) for r in history]

    def _save(fig, name: str) -> Path:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
        return path

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r["train_bce"] for r in history], marker="o", label="train BCE")
    ax.plot(epochs, [r["val_bce"] for r in history], marker="o", label="val BCE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE")
    ax.set_title("Training / validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r["beta"] for r in history], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(r"$\beta$ across epochs")
    ax.grid(True, alpha=0.3)
    _save(fig, "beta.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r["lambda0"] for r in history], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\lambda_0$")
    ax.set_title(r"$\lambda_0$ across epochs")
    ax.grid(True, alpha=0.3)
    _save(fig, "lambda0.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ages = list(PROBE_AGES_YEARS)
    n_show = min(6, len(history))
    idxs = np.unique(np.linspace(0, len(history) - 1, n_show, dtype=int))
    for i in idxs:
        rec = history[int(i)]
        lam = rec.get("lambda_at_ages") or {}
        ys = [lam.get(str(a), lam.get(a, float("nan"))) for a in ages]
        ax.plot(ages, ys, marker="o", label=f"epoch {rec['epoch']}")
    ax.set_xlabel("age (years)")
    ax.set_ylabel(r"$\lambda(a)=\lambda_0+\beta z(a)$")
    ax.set_title(r"$\lambda(a)$ curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, "lambda_a.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    tb = [r.get("temporal_bias_abs_mean", float("nan")) for r in history]
    ct = [r.get("content_abs_mean", float("nan")) for r in history]
    ax.plot(epochs, tb, marker="o", label=r"mean $|-\lambda(a)\tau|$")
    ax.plot(epochs, ct, marker="o", label=r"mean $|q^\top k / \sqrt{d}|$")
    ax.set_xlabel("epoch")
    ax.set_ylabel("magnitude")
    ax.set_title("Temporal-bias vs content-logit magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "bias_vs_content.png")

    if age_tests:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        labels = ["correct", "shuffle mean", "const. mean age", "const. median age"]
        vals = [
            age_tests.get("L_correct", float("nan")),
            age_tests.get("L_shuffle_mean", float("nan")),
            age_tests.get("L_constant_mean_age", float("nan")),
            age_tests.get("L_constant_median_age", float("nan")),
        ]
        ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])
        ax.set_ylabel("validation BCE")
        ax.set_title("Correct-age vs shuffled-age vs constant-age")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, "age_shuffle.png")

    return written
