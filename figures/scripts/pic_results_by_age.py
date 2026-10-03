#!/usr/bin/env python3
"""FIGURE A8: pic_results_by_age

PIC results use developmental bands neonate/infant/toddler/preschool/school/adolescent
and arms vanilla vs age-kernel (not NCH-style DTR vs temporal-only).

We remap to paper bins <1, 1–5, 6–11, 12–17 by pooling compatible bands with
positive sample support, and report AUPRC with available CIs where possible.
Sparse cells are flagged; mortality adolescent is especially sparse.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import C_DTR, C_TEMPORAL, REPO, apply_style, label_panel, save_fig  # noqa: E402

PIC_DIR = REPO / "results/pic/age_stratified"
TASKS = ["mortality", "pneumonia", "los_gt7", "heart_malformations"]
TASK_LABEL = {
    "mortality": "Mortality",
    "pneumonia": "Pneumonia",
    "los_gt7": "LOS > 7d",
    "heart_malformations": "Heart malformations",
}

# Remap PIC bands → paper developmental bins
PAPER_BINS = {
    "<1": ["neonate", "infant"],
    "1-5": ["toddler", "preschool"],
    "6-11": ["school"],
    "12-17": ["adolescent"],
}
BIN_TICK = ["<1", "1–5", "6–11", "12–17"]


def _pool_auprc(rows_by_band: dict, bands: list[str], arm: str):
    """Precision-weighted pool of AUPRC by n_pos (approximation for display)."""
    nums, dens, ns, npos = [], [], 0, 0
    ci_spans = []
    for b in bands:
        if b not in rows_by_band:
            continue
        r = rows_by_band[b]
        n = int(float(r[f"N_{arm}"]))
        pos = int(float(r[f"n_pos_{arm}"]))
        if n == 0 or pos == 0:
            continue
        auprc = float(r[f"auprc_{arm}"])
        nums.append(auprc * pos)
        dens.append(pos)
        ns += n
        npos += pos
        # AUROC CI available; use as qualitative uncertainty proxy only if present for auprc — not available
    if dens:
        return sum(nums) / sum(dens), ns, npos
    return float("nan"), 0, 0


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), sharey=False)
    omitted_notes = []

    for ax, task, letter in zip(axes.ravel(), TASKS, "ABCD"):
        path = PIC_DIR / f"{task}.csv"
        rows = list(csv.DictReader(path.open()))
        by_band = {r["band"]: r for r in rows}

        xs = np.arange(len(BIN_TICK))
        w = 0.36
        age_vals, van_vals, npos_ann = [], [], []
        for paper_bin, bands in PAPER_BINS.items():
            a_age, n_age, pos_age = _pool_auprc(by_band, bands, "age")
            a_van, n_van, pos_van = _pool_auprc(by_band, bands, "vanilla")
            age_vals.append(a_age)
            van_vals.append(a_van)
            npos_ann.append(pos_age)
            if pos_age < 10:
                omitted_notes.append(f"{task}/{paper_bin}: n_pos={pos_age}")

        ax.bar(xs - w / 2, age_vals, w, label="Age-kernel", color=C_DTR)
        ax.bar(xs + w / 2, van_vals, w, label="Vanilla", color=C_TEMPORAL)
        ax.set_xticks(xs)
        ax.set_xticklabels(BIN_TICK)
        ax.set_ylabel("AUPRC")
        ax.set_title(TASK_LABEL[task])
        label_panel(ax, letter)
        for i, p in enumerate(npos_ann):
            if not np.isnan(age_vals[i]):
                ax.text(i, 0.02, f"pos={p}", ha="center", fontsize=5, color="#718096")
        if letter == "A":
            ax.legend(frameon=False, fontsize=6.5)

    fig.suptitle(
        "PIC age-stratified AUPRC (remapped bins; age-kernel vs vanilla)",
        fontsize=9.5,
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "pic_results_by_age")
    if omitted_notes:
        print("Sparse cells:", "; ".join(omitted_notes))


if __name__ == "__main__":
    main()
