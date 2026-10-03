#!/usr/bin/env python3
"""FIGURE A7: mimic_learning_curves — Stage-1 MIMIC pretraining."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_DTR, C_TEMPORAL, REPO, apply_style, label_panel, save_fig  # noqa: E402

ADKM = REPO / "stage1_mimic_pretrain/run/adkm_s0/history.json"
NINT = REPO / "stage1_mimic_pretrain/run/nint_s0/history.json"


def main() -> None:
    apply_style()
    adkm = json.loads(ADKM.read_text())
    nint = json.loads(NINT.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    # Loss
    ax = axes[0]
    ax.plot([r["epoch"] for r in adkm], [r["train_bce"] for r in adkm], "-", color=C_DTR, lw=1.5, label="DTR train")
    ax.plot([r["epoch"] for r in adkm], [r["val_bce"] for r in adkm], "--", color=C_DTR, lw=1.5, label="DTR val")
    ax.plot([r["epoch"] for r in nint], [r["train_bce"] for r in nint], "-", color=C_TEMPORAL, lw=1.5, label="Temporal-only train")
    ax.plot([r["epoch"] for r in nint], [r["val_bce"] for r in nint], "--", color=C_TEMPORAL, lw=1.5, label="Temporal-only val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE")
    ax.set_title("Training / validation loss")
    ax.legend(frameon=False, fontsize=6)
    label_panel(ax, "A")

    # Primary metric
    ax = axes[1]
    ax.plot([r["epoch"] for r in adkm], [r["micro_auprc"] for r in adkm], "o-", color=C_DTR, lw=1.5, ms=4, label="DTR")
    ax.plot([r["epoch"] for r in nint], [r["micro_auprc"] for r in nint], "s-", color=C_TEMPORAL, lw=1.5, ms=4, label="Temporal-only DTR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val micro-AUPRC")
    ax.set_title("Validation micro-AUPRC")
    ax.legend(frameon=False, fontsize=7)
    label_panel(ax, "B")
    # mark selected epoch 5 (checkpoint used for Stage-2 init)
    ax.axvline(5, color="#a0aec0", ls=":", lw=0.9)
    ymin = min(min(r["micro_auprc"] for r in adkm), min(r["micro_auprc"] for r in nint))
    ax.text(5.15, ymin, "selected ep.5", fontsize=6, color="#718096", rotation=90, va="bottom")

    fig.suptitle("MIMIC Stage-1 pretraining", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, "mimic_learning_curves")


if __name__ == "__main__":
    main()
