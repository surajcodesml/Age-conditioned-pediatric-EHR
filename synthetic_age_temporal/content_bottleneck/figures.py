"""Figures for content bottleneck final test."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from high_impact.aggregate import iter_runs


def _mean_matrix(root: Path, scenario: str, key: str):
    grids = []
    for _e, scen, arm, _s, seed_dir in iter_runs(root):
        if scen != scenario or arm != "age_temporal":
            continue
        path = seed_dir / "mechanism_outputs.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as blob:
            if key in blob.files:
                grids.append(np.asarray(blob[key], dtype=np.float64))
    if not grids:
        return None
    return np.nanmean(np.stack(grids, axis=0), axis=0)


def plot_content_heatmaps(root: Path) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scenario in ("S2", "S3"):
        W = _mean_matrix(root, scenario, "W_true")
        learned = _mean_matrix(root, scenario, "learned_evidence")
        residual = _mean_matrix(root, scenario, "evidence_residual")
        if W is None or learned is None:
            continue
        if residual is None:
            residual = learned - W
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for ax, mat, title, cmap in (
            (axes[0], W, "true W[target, signal]", "coolwarm"),
            (axes[1], learned, "learned evidence", "coolwarm"),
            (axes[2], residual, "residual", "coolwarm"),
        ):
            vmax = float(np.nanmax(np.abs(mat))) or 1.0
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("signal")
            ax.set_ylabel("target")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{scenario} target×signal content")
        fig.savefig(fig_dir / f"true_vs_learned_content_{scenario}.png", dpi=150)
        plt.close(fig)


def make_all_figures(root: Path) -> None:
    plot_content_heatmaps(Path(root))
