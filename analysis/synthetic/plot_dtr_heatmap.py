"""Synthetic age × lag heatmaps: Oracle, DTR, temporal-only, and residual.

Color on the first three panels is the mean predicted probability
P̂(Y | current age, lag). The residual is

    E(a, τ) = P̂_DTR(a, τ) − P_oracle(a, τ).

Probability panels share one color scale with the CEHR-BERT figure
(oracle, DTR, temporal-only, and CEHR-BERT together). Residual panels on
both figures share one diverging scale centered at 0. Neither panel is
normalized to its own range.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.figures_style import apply_style, save_figure  # noqa: E402
from analysis.synthetic.surfaces import (  # noqa: E402
    get_surfaces,
    probability_limits,
    residual_limit,
    residuals,
)

STEM = "synthetic_s2_dtr_age_lag_heatmap"


def _lag_labels(lags: np.ndarray) -> list[str]:
    labels = []
    for lag in lags:
        days = float(lag)
        if days == 0:
            labels.append("0")
        elif abs(days - 365) < 1:
            labels.append("1y")
        elif abs(days - 730) < 1:
            labels.append("2y")
        else:
            labels.append(str(int(days)))
    return labels


def _draw_panel(ax, grid: np.ndarray, *, vmin: float, vmax: float, cmap: str, title: str):
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=8)
    return im


def _decorate(axes, ages: np.ndarray, lags: np.ndarray) -> None:
    lag_labs = _lag_labels(lags)
    age_idx = list(range(0, len(ages), 3))
    for j, ax in enumerate(axes):
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels(lag_labs, fontsize=6.5, rotation=45)
        ax.set_yticks(age_idx)
        ax.set_yticklabels([str(int(ages[i])) for i in age_idx])
        ax.set_xlim(-0.5, len(lags) - 0.5)
        ax.set_ylim(-0.5, len(ages) - 0.5)
        ax.set_xlabel("Lag")
        if j == 0:
            ax.set_ylabel("Current age (years)")


def plot(*, scenario: str = "S2", recompute: bool = False, device: str = "cpu"):
    pack = get_surfaces(scenario, recompute=recompute, device=device)
    ages = pack["ages"]
    lags = pack["lags"]
    surf = pack["surfaces"]
    vmin, vmax = probability_limits(pack)
    rmax = residual_limit(pack)
    resid = residuals(pack)["dtr"]

    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 3.15), constrained_layout=True)
    panels = [
        (surf["oracle"], vmin, vmax, "viridis", "Oracle"),
        (surf["dtr_age_temporal_new"], vmin, vmax, "viridis", "DTR"),
        (surf["dtr_temporal_only_new"], vmin, vmax, "viridis", "Temporal-only"),
        (resid, -rmax, rmax, "RdBu_r", r"DTR $-$ oracle"),
    ]
    images = []
    for ax, (grid, lo, hi, cmap, title) in zip(axes, panels):
        images.append(_draw_panel(ax, grid, vmin=lo, vmax=hi, cmap=cmap, title=title))
    _decorate(axes, ages, lags)
    cbar_p = fig.colorbar(images[0], ax=list(axes[:3]), fraction=0.046, pad=0.02)
    cbar_p.set_label(r"mean $\hat{P}(Y\mid a,\tau)$")
    cbar_r = fig.colorbar(images[3], ax=axes[3], fraction=0.046, pad=0.04)
    cbar_r.set_label(r"$\hat{P}_{\mathrm{DTR}}-P_{\mathrm{oracle}}$")
    fig.suptitle(f"{scenario} age × lag  (shared probability scale)", fontsize=9)
    png, svg = save_figure(fig, STEM if scenario == "S2" else f"synthetic_{scenario.lower()}_dtr_age_lag_heatmap")
    return {
        "png": png,
        "svg": svg,
        "vmin": vmin,
        "vmax": vmax,
        "residual_limit": rmax,
        "pack": pack,
    }


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S2")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    return plot(scenario=args.scenario, recompute=args.recompute, device=args.device)


if __name__ == "__main__":
    main()
