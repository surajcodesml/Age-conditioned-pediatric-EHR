"""Figures from saved high-impact artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from high_impact.aggregate import iter_runs
from high_impact.config import EXPERIMENT_ORDER

_ORDER = ["C01_staged_current"] + list(EXPERIMENT_ORDER)


def _summary_index(root: Path) -> dict[tuple, float]:
    path = root / "high_impact_followup_summary.csv"
    index = {}
    if not path.exists():
        return index
    with path.open() as handle:
        for row in csv.DictReader(handle):
            index[(row["experiment"], row["scenario"], row["arm"], row["metric"])] = float(row["mean"])
    return index


def _mean_array(root: Path, experiment: str, scenario: str, arm: str, key: str):
    grids = []
    search_roots = [root]
    if experiment == "C01_staged_current":
        from high_impact import C01_ARTIFACT_ROOT
        search_roots = [C01_ARTIFACT_ROOT]
    for base in search_roots:
        for exp, scen, arm_name, _seed, seed_dir in iter_runs(base):
            if exp == experiment and scen == scenario and arm_name == arm:
                with np.load(seed_dir / "mechanism_outputs.npz", allow_pickle=False) as blob:
                    if key in blob.files:
                        grids.append(np.asarray(blob[key], dtype=np.float64))
    if not grids:
        return None
    return np.nanmean(np.stack(grids, axis=0), axis=0)


def plot_surfaces(root: Path) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scenario in ("S2", "S3"):
        panels = []
        for experiment in _ORDER:
            key = "surface_model" if experiment != "D00_oracle_gate" else "surface_model"
            panels.append((experiment, _mean_array(root, experiment, scenario, "age_temporal", key)))
        oracle = _mean_array(root, "D00_oracle_gate", scenario, "age_temporal", "surface_oracle")
        if oracle is None:
            oracle = _mean_array(root, "C01_staged_current", scenario, "age_temporal", "surface_oracle")
        panels = [("oracle", oracle)] + panels
        arrays = [np.nanmean(g, axis=-1) if g is not None and g.ndim == 3 else g for _, g in panels]
        finite = [g for g in arrays if g is not None]
        if not finite:
            continue
        vmin = min(float(np.nanmin(g)) for g in finite)
        vmax = max(float(np.nanmax(g)) for g in finite)
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.8), squeeze=False)
        image = None
        for ax, (name, grid) in zip(axes[0], panels):
            show = None if grid is None else (np.nanmean(grid, axis=-1) if grid.ndim == 3 else grid)
            if show is None:
                ax.set_axis_off()
                continue
            image = ax.imshow(show, aspect="auto", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(name.replace("_", "\n"), fontsize=7)
        if image is not None:
            fig.colorbar(image, ax=axes[0].tolist(), fraction=0.02)
        fig.suptitle(f"{scenario} prediction surfaces")
        fig.savefig(fig_dir / f"prediction_surface_{scenario}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def plot_specialization(root: Path) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for experiment in ("D01_multihead_shared", "D02_multihead_dev"):
        cosines, corrs = [], []
        for exp, scen, arm, _seed, seed_dir in iter_runs(root):
            if exp != experiment or scen != "S2" or arm != "age_temporal":
                continue
            mech = json.loads((seed_dir / "mechanism_metrics.json").read_text())
            if "query_cosine" in mech:
                cosines.append(np.asarray(mech["query_cosine"], dtype=np.float64))
            if "content_score_corr" in mech:
                corrs.append(np.asarray(mech["content_score_corr"], dtype=np.float64))
        if cosines:
            mat = np.nanmean(np.stack(cosines, axis=0), axis=0)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            image = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_title(f"{experiment} query cosine")
            fig.colorbar(image, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{experiment}_query_cosine.png", dpi=140)
            plt.close(fig)
        if corrs:
            mat = np.nanmean(np.stack(corrs, axis=0), axis=0)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            image = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_title(f"{experiment} content-score corr")
            fig.colorbar(image, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{experiment}_content_score_corr.png", dpi=140)
            plt.close(fig)

    # D02 lambda_h curves
    curves = []
    for exp, scen, arm, _seed, seed_dir in iter_runs(root):
        if exp != "D02_multihead_dev" or scen != "S2" or arm != "age_temporal":
            continue
        with np.load(seed_dir / "mechanism_outputs.npz", allow_pickle=False) as blob:
            if "lambda_h" in blob.files:
                curves.append(np.asarray(blob["lambda_h"], dtype=np.float64))
                truth = np.asarray(blob["lambda_true_age"], dtype=np.float64)
    if curves:
        mean = np.nanmean(np.stack(curves, axis=0), axis=0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for h, row in enumerate(mean):
            ax.plot(row, label=f"head {h}")
        ax.plot(truth, color="black", linewidth=2, label="oracle")
        ax.set_title("D02 S2 lambda_h(a)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "d02_lambda_h.png", dpi=140)
        plt.close(fig)

    index = _summary_index(root)
    labels = [e.split("_", 1)[0] for e in _ORDER]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    vals = [index.get((e, "S2", "age_temporal", "surface_rmse")) for e in _ORDER]
    ax.bar(range(len(labels)), [np.nan if v is None else v for v in vals])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("S2 surface RMSE")
    fig.tight_layout()
    fig.savefig(fig_dir / "surface_rmse_s2.png", dpi=140)
    plt.close(fig)


def make_all_figures(root: Path) -> None:
    plot_surfaces(root)
    plot_specialization(root)
