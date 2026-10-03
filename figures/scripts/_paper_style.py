"""Shared publication style for DTR paper figures (matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
FIG_PNG = REPO / "figures" / "png"
FIG_SVG = REPO / "figures" / "svg"

# Paper colors
C_DTR = "#0B6E4F"
C_TEMPORAL = "#C45C26"
C_ORACLE = "#1A202C"
C_BASELINE = "#4A5568"
C_BLUE = "#2B6CB0"
C_ACCENT = "#C05621"
C_GRAY = "#718096"

PANEL_KW = dict(fontsize=11, fontweight="bold", loc="left")


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


def save_fig(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIG_PNG.mkdir(parents=True, exist_ok=True)
    FIG_SVG.mkdir(parents=True, exist_ok=True)
    png = FIG_PNG / f"{stem}.png"
    svg = FIG_SVG / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {svg}")
    return png, svg


def label_panel(ax: plt.Axes, letter: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="right",
    )
