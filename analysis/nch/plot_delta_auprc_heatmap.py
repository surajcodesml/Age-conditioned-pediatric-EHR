"""NCH heatmap of ΔAUPRC by developmental age band and history horizon.

Cell value:
    AUPRC_DTR − AUPRC_temporal-only
on ``dtr_age_temporal_new`` and ``dtr_temporal_only_new``.
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
from analysis.nch.evaluate import (  # noqa: E402
    AGE_BAND_NAMES,
    HEATMAP_HORIZONS,
    delta_matrix,
    get_metrics,
)

STEM = "nch_delta_auprc_heatmap"
DISPLAY_BANDS = ["<1", "1–5", "6–11", "12–17"]


def _fmt(val: float) -> str:
    if not np.isfinite(val):
        return "—"
    if abs(val) >= 0.01:
        return f"{val:+.3f}"
    return f"{val:+.4f}"


def plot(payload: dict | None = None, *, recompute: bool = False, max_examples: int = 0, batch_size: int = 128, device: str = "cpu"):
    if payload is None:
        payload = get_metrics(
            recompute=recompute,
            max_examples=max_examples,
            batch_size=batch_size,
            device=device,
        )
    mat = delta_matrix(payload, HEATMAP_HORIZONS)
    finite = mat[np.isfinite(mat)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1e-6

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    im = ax.imshow(
        mat,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(HEATMAP_HORIZONS)))
    ax.set_xticklabels(list(HEATMAP_HORIZONS))
    ylabels = []
    for i, disp in enumerate(DISPLAY_BANDS):
        n = payload["by_horizon"][HEATMAP_HORIZONS[0]]["cells"][AGE_BAND_NAMES[i]]["n_windows"]
        ylabels.append(f"{disp}\n(n={n})")
    ax.set_yticks(range(len(DISPLAY_BANDS)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("History horizon")
    ax.set_ylabel("Age band (years)")
    ax.set_title(r"NCH  $\Delta$AUPRC  $=$  AUPRC$_{DTR}$ $-$ AUPRC$_{temporal\text{-}only}$")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if np.isfinite(val) and abs(val) > 0.55 * vmax else "black"
            ax.text(j, i, _fmt(val), ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\Delta$AUPRC")
    fig.tight_layout()
    png, svg = save_figure(fig, STEM)
    return {"png": png, "svg": svg, "matrix": mat, "payload": payload}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    return plot(
        recompute=args.recompute,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
