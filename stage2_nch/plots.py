"""Plots required for every Stage-2 run."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from stage2_nch.config import PROBE_AGES_YEARS


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_run_plots(run_dir: Path, history: list[dict], age_tests: dict | None,
                   age_stratified: dict | None = None,
                   history_stratified: dict | None = None) -> list[Path]:
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
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        return path

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r["train_bce"] for r in history], marker="o", label="train BCE")
    ax.plot(epochs, [r["val_bce"] for r in history], marker="o", label="val BCE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE")
    ax.set_title("Train / validation BCE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r.get("val_micro_auprc", r.get("micro_auprc", float("nan")))
                     for r in history], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("micro AUPRC")
    ax.set_title("Validation micro AUPRC")
    ax.grid(True, alpha=0.3)
    _save(fig, "micro_auprc.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r.get("train_positive_bce", float("nan")) for r in history],
            marker="o", label="train positive BCE")
    ax.plot(epochs, [r.get("val_positive_bce", float("nan")) for r in history],
            marker="o", label="val positive BCE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("positive-label BCE")
    ax.set_title("Positive-label BCE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "positive_bce.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r["lambda0"] for r in history], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\lambda_0$")
    ax.set_title(r"$\lambda_0$ trajectory")
    ax.grid(True, alpha=0.3)
    _save(fig, "lambda0.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r["beta"] for r in history], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\beta_P$")
    ax.set_title(r"$\beta_P$ trajectory")
    ax.grid(True, alpha=0.3)
    _save(fig, "beta.png")

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
    ax.set_ylabel(r"$\lambda_P(a)=\lambda_0+\beta_P z_P(a)$")
    ax.set_title(r"$\lambda_P(a)$ at pediatric probe ages")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, "lambda_a.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, [r.get("ratio_temporal_abs_to_content_abs", float("nan")) for r in history],
            marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("R_bias = E|(lambda+beta z) tau| / E|qk/sqrt(d)|")
    ax.set_title("Temporal-bias / content-logit ratio")
    ax.grid(True, alpha=0.3)
    _save(fig, "bias_vs_content.png")

    if age_tests:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        labels = ["correct", "shuffle mean", "const. age 9"]
        vals = [
            age_tests.get("L_correct", float("nan")),
            age_tests.get("L_shuffle_mean", float("nan")),
            age_tests.get("L_constant_mean_age", float("nan")),
        ]
        ax.bar(labels, vals)
        ax.set_ylabel("validation BCE")
        ax.set_title("Correct vs shuffled vs constant attention age")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, "age_shuffle.png")

    def _bar(metric_map: dict | None, title: str, fname: str, metric: str = "micro_auprc"):
        if not metric_map:
            return
        names = list(metric_map)
        vals = [metric_map[n].get(metric, float("nan")) for n in names]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.bar(names, vals)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, fname)

    _bar(age_stratified, "Micro AUPRC by pediatric age band", "age_groups.png")
    _bar(history_stratified, "Micro AUPRC by history-length bin", "history_groups.png")
    return written
