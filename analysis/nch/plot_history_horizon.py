"""NCH ΔAUPRC versus available history horizon.

Horizons: 30d, 90d, 180d, 1y, 3y, 5y, Full.
ΔAUPRC = micro-AUPRC(dtr_age_temporal_new) − micro-AUPRC(dtr_temporal_only_new)
on the held-out test split, with input history truncated to each horizon.
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

from analysis.figures_style import C_DTR, C_ZERO, apply_style, save_figure  # noqa: E402
from analysis.nch.evaluate import get_metrics, overall_curve  # noqa: E402

STEM = "nch_history_horizon_delta_auprc"


def plot(payload: dict | None = None, *, recompute: bool = False, max_examples: int = 0, batch_size: int = 128, device: str = "cpu"):
    if payload is None:
        payload = get_metrics(
            recompute=recompute,
            max_examples=max_examples,
            batch_size=batch_size,
            device=device,
        )
    curve = overall_curve(payload)
    x = np.arange(len(curve["horizon"]))
    y = curve["delta_auprc"]

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.axhline(0.0, color=C_ZERO, lw=0.8, zorder=0)
    ax.plot(x, y, "o-", color=C_DTR, lw=1.6, ms=5.5, label=r"DTR $-$ temporal-only")
    for i, val in enumerate(y):
        if np.isfinite(val):
            above = float(val) >= 0.0
            ax.annotate(
                f"{val:+.5f}",
                (x[i], val),
                textcoords="offset points",
                xytext=(0, 8 if above else -10),
                ha="center",
                va="bottom" if above else "top",
                fontsize=6.5,
                color="#1A202C",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(list(curve["horizon"]))
    ax.set_xlabel("History horizon")
    ax.set_ylabel(r"$\Delta$AUPRC  (DTR $-$ temporal-only)")
    n = int(curve["n_windows"][0]) if len(curve["n_windows"]) else 0
    ax.set_title(f"NCH test  (n={n} windows)")
    # Leave room for the value labels.
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max(0.45 * (ymax - ymin), 1.5e-4)
    ax.set_ylim(min(0.0, ymin) - pad, max(0.0, ymax) + pad)
    fig.tight_layout()
    png, svg = save_figure(fig, STEM)
    return {"png": png, "svg": svg, "curve": curve, "payload": payload}


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
