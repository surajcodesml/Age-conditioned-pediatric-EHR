"""Figures for the age × temporal interaction experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BETA_TRUE, KERNEL_AGES, LAMBDA0_TRUE, PROBE_AGES


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_lambda_recovery(
    rows: list[dict[str, Any]],
    age_mean: float,
    age_std: float,
    path: Path,
) -> None:
    ages = np.linspace(0.0, 18.0, 400)
    z = (ages - age_mean) / max(age_std, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, task in zip(axes, ("T0", "T1", "T2")):
        beta_t = BETA_TRUE[task]
        ax.plot(ages, LAMBDA0_TRUE + beta_t * z, color="black", lw=2.2, label="true λ*(a)")
        task_rows = [r for r in rows if r["task"] == task and r["arm"] == "age_temporal"]
        for i, r in enumerate(task_rows):
            lam = r["lambda0_hat"] + r["beta_hat"] * z
            ax.plot(
                ages,
                lam,
                color="#1f4e79",
                alpha=0.55 if len(task_rows) > 1 else 1.0,
                lw=1.6,
                label="learned λ̂(a)" if i == 0 else None,
            )
        marks = np.array(PROBE_AGES)
        ax.scatter(marks, LAMBDA0_TRUE + beta_t * (marks - age_mean) / max(age_std, 1e-6),
                   c="black", s=18, zorder=5)
        ax.set_title(f"{task}  β*={beta_t:+.0f}")
        ax.set_xlabel("Age (years)")
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"$\lambda(a)$")
    fig.suptitle("Age-dependent slope recovery", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_kernel_recovery(
    rows: list[dict[str, Any]],
    age_mean: float,
    age_std: float,
    path: Path,
) -> None:
    tau = np.linspace(0.0, 1.0, 200)
    fig, axes = plt.subplots(3, 5, figsize=(14.0, 8.2), sharex=True, sharey=True)
    for r_i, task in enumerate(("T0", "T1", "T2")):
        beta_t = BETA_TRUE[task]
        task_rows = [r for r in rows if r["task"] == task and r["arm"] == "age_temporal"]
        for c_i, age in enumerate(KERNEL_AGES):
            ax = axes[r_i, c_i]
            z_a = (age - age_mean) / max(age_std, 1e-6)
            true_b = -(LAMBDA0_TRUE + beta_t * z_a) * tau
            ax.plot(tau, true_b, color="black", lw=2.0, label="true")
            for j, rec in enumerate(task_rows):
                hat_b = -(rec["lambda0_hat"] + rec["beta_hat"] * z_a) * tau
                ax.plot(
                    tau,
                    hat_b,
                    color="#1f4e79",
                    alpha=0.55 if len(task_rows) > 1 else 1.0,
                    lw=1.5,
                    label="learned" if j == 0 else None,
                )
            if r_i == 0:
                ax.set_title(f"age={age:.0f}")
            if c_i == 0:
                ax.set_ylabel(f"{task}\nb(a,τ)")
            if r_i == 2:
                ax.set_xlabel(r"$\tau$")
            _style(ax)
            if r_i == 0 and c_i == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Temporal kernel recovery  b(a,τ)=−(λ0+β z(a)) τ", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_beta_recovery(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    colors = {"T0": "#4c4c4c", "T1": "#1f4e79", "T2": "#b35c1e"}
    rng = np.random.default_rng(0)
    for task in ("T0", "T1", "T2"):
        vals = [r["beta_hat"] for r in rows if r["task"] == task and r["arm"] == "age_temporal"]
        if not vals:
            continue
        x = np.full(len(vals), {"T0": 0, "T1": 1, "T2": 2}[task], dtype=float)
        x = x + rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(x, vals, color=colors[task], s=42, zorder=3, label=f"{task} β̂")
        ax.hlines(BETA_TRUE[task], {"T0": 0, "T1": 1, "T2": 2}[task] - 0.25,
                  {"T0": 0, "T1": 1, "T2": 2}[task] + 0.25, colors=colors[task], lw=2)
    ax.axhline(0.0, color="0.7", lw=1)
    ax.set_xticks([0, 1, 2], ["T0 (β*=0)", "T1 (β*=+1)", "T2 (β*=−1)"])
    ax.set_ylabel(r"learned $\hat\beta$")
    ax.set_title("Interaction coefficient recovery")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_task_comparison(df: pd.DataFrame, path: Path, metric: str = "accuracy") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    arms = ["no_age", "temporal_only", "late_age", "age_temporal"]
    colors = ["#7a7a7a", "#5b8fa8", "#c47b3b", "#1f4e79"]
    for ax, task in zip(axes, ("T0", "T1", "T2")):
        sub = df[df["task"] == task]
        means, stds = [], []
        for arm in arms:
            v = sub.loc[sub["model"] == arm, metric]
            means.append(float(v.mean()) if len(v) else np.nan)
            stds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        ax.bar(np.arange(len(arms)), means, yerr=stds, color=colors, capsize=3)
        ax.set_xticks(np.arange(len(arms)), ["no age", "temporal", "late age", "age×time"], rotation=25, ha="right")
        ax.set_title(task)
        _style(ax)
    axes[0].set_ylabel(metric)
    fig.suptitle(f"Test {metric} by task and model", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_intervention(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, col, title in (
        (axes[0], "delta_constant_mean", "ΔBCE constant age (worse if >0)"),
        (axes[1], "delta_shuffle_mean", "ΔBCE shuffled age (worse if >0)"),
    ):
        sub = df[df["model"] == "age_temporal"]
        for i, task in enumerate(("T0", "T1", "T2")):
            v = sub.loc[sub["task"] == task, col]
            ax.bar(i, float(v.mean()) if len(v) else np.nan,
                   yerr=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                   color=["#4c4c4c", "#1f4e79", "#b35c1e"][i], capsize=3)
        ax.axhline(0.0, color="0.5", lw=1)
        ax.set_xticks([0, 1, 2], ["T0", "T1", "T2"])
        ax.set_title(title)
        _style(ax)
    fig.suptitle("Age-temporal arm: cost of destroying the true age", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_smoke_history(history: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ep = [h["epoch"] for h in history]
    axes[0].plot(ep, [h["train_bce"] for h in history], color="#1f4e79")
    axes[0].set_title("Train BCE")
    axes[1].plot(ep, [h["train_accuracy"] for h in history], color="#1f4e79")
    axes[1].set_title("Train accuracy")
    axes[2].plot(ep, [h["recovered"]["beta_hat"] for h in history], color="#1f4e79")
    axes[2].axhline(1.0, color="0.5", ls="--")
    axes[2].set_title(r"learned $\hat\beta$ (true +1)")
    for ax in axes:
        ax.set_xlabel("epoch")
        _style(ax)
    fig.suptitle("Tiny overfit smoke test (T1, age_temporal)", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
