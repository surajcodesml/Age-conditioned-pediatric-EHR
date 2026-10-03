"""Shared matplotlib style and output paths for paper figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "results" / "figures"

C_DTR = "#0B6E4F"
C_TEMPORAL = "#C45C26"
C_ZERO = "#4A5568"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str, fig_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ``stem.png`` and ``stem.svg`` under ``results/figures``."""
    out = FIG_DIR if fig_dir is None else Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{stem}.png"
    svg = out / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")
    return png, svg
