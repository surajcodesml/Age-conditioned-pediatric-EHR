"""Synthetic age × lag heatmaps for CEHR-BERT against the oracle.

Panels: Oracle, CEHR-BERT, and

    E(a, τ) = P̂_CEHR-BERT(a, τ) − P_oracle(a, τ).

Axis limits match the DTR figure. The probability color scale is the shared
scale over oracle, DTR, temporal-only, and CEHR-BERT. The residual color
scale is the shared symmetric scale used for the DTR residual, so the two
figures are not normalized separately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.figures_style import apply_style, save_figure  # noqa: E402
from analysis.synthetic.plot_dtr_heatmap import _decorate, _draw_panel  # noqa: E402
from analysis.synthetic.surfaces import (  # noqa: E402
    get_surfaces,
    probability_limits,
    residual_limit,
    residuals,
)

import matplotlib.pyplot as plt  # noqa: E402

STEM = "synthetic_s2_cehrbert_age_lag_heatmap"


def plot(*, scenario: str = "S2", recompute: bool = False, device: str = "cpu"):
    pack = get_surfaces(scenario, recompute=recompute, device=device)
    ages = pack["ages"]
    lags = pack["lags"]
    surf = pack["surfaces"]
    vmin, vmax = probability_limits(pack)
    rmax = residual_limit(pack)
    resid = residuals(pack)["cehrbert"]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.15), constrained_layout=True)
    panels = [
        (surf["oracle"], vmin, vmax, "viridis", "Oracle"),
        (surf["cehrbert"], vmin, vmax, "viridis", "CEHR-BERT"),
        (resid, -rmax, rmax, "RdBu_r", r"CEHR-BERT $-$ oracle"),
    ]
    images = []
    for ax, (grid, lo, hi, cmap, title) in zip(axes, panels):
        images.append(_draw_panel(ax, grid, vmin=lo, vmax=hi, cmap=cmap, title=title))
    _decorate(axes, ages, lags)
    cbar_p = fig.colorbar(images[0], ax=list(axes[:2]), fraction=0.046, pad=0.02)
    cbar_p.set_label(r"mean $\hat{P}(Y\mid a,\tau)$")
    cbar_r = fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.04)
    cbar_r.set_label(r"$\hat{P}_{\mathrm{CEHR}}-P_{\mathrm{oracle}}$")
    fig.suptitle(f"{scenario} age × lag  (same scales as the DTR figure)", fontsize=9)
    png, svg = save_figure(
        fig,
        STEM if scenario == "S2" else f"synthetic_{scenario.lower()}_cehrbert_age_lag_heatmap",
    )
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
