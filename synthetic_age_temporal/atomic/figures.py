"""Figures from saved npz/csv artifacts. Does not reload models."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atomic.aggregate import iter_runs
from atomic.config import EXPERIMENT_ORDER

_ORDER = list(EXPERIMENT_ORDER)


def _summary_index(root: Path) -> dict[tuple, float]:
    path = root / "atomic_followup_summary.csv"
    index = {}
    if not path.exists():
        return index
    with path.open() as handle:
        for row in csv.DictReader(handle):
            index[(row["experiment"], row["scenario"], row["arm"], row["metric"])] = float(row["mean"])
    return index


def _bar(path: Path, title: str, labels: list[str], values: list[float | None]) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ys = [np.nan if value is None else value for value in values]
    ax.bar(range(len(labels)), ys)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _mean_surface(root: Path, experiment: str, scenario: str, arm: str, key: str) -> np.ndarray | None:
    grids = []
    for exp, scen, arm_name, _seed, seed_dir in iter_runs(root):
        if exp == experiment and scen == scenario and arm_name == arm:
            with np.load(seed_dir / "mechanism_outputs.npz", allow_pickle=False) as blob:
                if key not in blob.files:
                    continue
                grids.append(np.asarray(blob[key], dtype=np.float64))
    if not grids:
        return None
    return np.nanmean(np.stack(grids, axis=0), axis=0)


def plot_prediction_surfaces(root: Path) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scenario in ("S2", "S3"):
        panels = [("oracle", _mean_surface(root, "C00_current_dtr", scenario, "age_temporal", "surface_oracle"))]
        for experiment in _ORDER:
            panels.append((experiment, _mean_surface(root, experiment, scenario, "age_temporal", "surface_model")))
        arrays = [np.nanmean(grid, axis=-1) if grid is not None and grid.ndim == 3 else grid for _, grid in panels]
        finite = [grid for grid in arrays if grid is not None]
        if not finite:
            continue
        vmin = min(float(np.nanmin(grid)) for grid in finite)
        vmax = max(float(np.nanmax(grid)) for grid in finite)
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4), squeeze=False)
        for ax, (name, grid) in zip(axes[0], panels):
            show = None if grid is None else (np.nanmean(grid, axis=-1) if grid.ndim == 3 else grid)
            if show is None:
                ax.set_axis_off()
                continue
            image = ax.imshow(show, aspect="auto", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(name.replace("_", "\n"), fontsize=8)
            ax.set_xlabel("lag index")
            ax.set_ylabel("age index")
        fig.colorbar(image, ax=axes[0].tolist(), fraction=0.02)
        fig.suptitle(f"{scenario} prediction surface, shared scale, mean over targets and seeds")
        fig.savefig(fig_dir / f"prediction_surface_{scenario}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def plot_gate_and_controls(root: Path) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    index = _summary_index(root)
    labels = [name.split("_", 1)[0] for name in _ORDER]
    _bar(
        fig_dir / "gate_signal_rmse_s2.png",
        "S2 age-temporal signal-encounter gate RMSE",
        labels,
        [index.get((exp, "S2", "age_temporal", "gate_signal_rmse")) for exp in _ORDER],
    )
    _bar(
        fig_dir / "negative_controls_s0.png",
        "S0 age-temporal gate-age-shuffle ΔBCE",
        labels,
        [index.get((exp, "S0", "age_temporal", "delta_bce_gate_age_shuffle")) for exp in _ORDER],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, scenario in zip(axes, ("S2", "S3")):
        curves = []
        for experiment in ("C00_current_dtr", "C04_no_content_persistence", "C05_shared_beta_mixture", "C06_component_beta_mixture"):
            grids = []
            key = "content_free_lambda_age" if experiment.startswith("C00") or experiment.startswith("C04") else "lambda_k"
            for exp, scen, arm, _seed, seed_dir in iter_runs(root):
                if exp == experiment and scen == scenario and arm == "age_temporal":
                    with np.load(seed_dir / "mechanism_outputs.npz", allow_pickle=False) as blob:
                        if key not in blob.files:
                            continue
                        grids.append(np.asarray(blob[key], dtype=np.float64))
            if not grids:
                continue
            stacked = np.stack(grids, axis=0)
            mean = np.nanmean(stacked, axis=0)
            if mean.ndim == 2:
                for k, row in enumerate(mean):
                    ax.plot(row, label=f"{experiment[-12:]} k={k}")
            else:
                ax.plot(mean, label=experiment.split("_")[0])
        oracle = None
        for exp, scen, arm, _seed, seed_dir in iter_runs(root):
            if scen == scenario and arm == "age_temporal":
                with np.load(seed_dir / "mechanism_outputs.npz", allow_pickle=False) as blob:
                    oracle = np.asarray(blob["lambda_true_age"], dtype=np.float64)
                break
        if oracle is not None:
            ax.plot(oracle, label="oracle", color="black", linewidth=2)
        ax.set_title(f"{scenario} lambda curves")
        ax.set_xlabel("age index 0–18")
        handles, labs = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "lambda_curves.png", dpi=140)
    plt.close(fig)


def make_all_figures(root: Path) -> None:
    plot_prediction_surfaces(root)
    plot_gate_and_controls(root)
