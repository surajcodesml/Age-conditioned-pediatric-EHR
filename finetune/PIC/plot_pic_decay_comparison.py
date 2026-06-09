#!/usr/bin/env python3
"""Plot PIC fine-tuned temporal decay curves: TALE-EHR vs age-conditioned."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.PIC.age_kernel_viz import (
    _dt_grid,
    _w_curve_age,
    _w_curve_vanilla,
)
from finetune.PIC.pic_age_eval_common import (
    PIC_CKPT_ROOT,
    REPRESENTATIVE_AGES_YR,
    load_finetuned_classifier,
)
from finetune.train import _resolve_age_conditioned_backbone

TASKS = ("pneumonia", "heart_malformations")
TASK_LABELS = {
    "pneumonia": "Pneumonia",
    "heart_malformations": "Heart malformations",
}
# Single age for the 4-way combined panel (pediatric mid-childhood).
COMBINED_AGE_YR = 2.0


def _run_dir(ckpt_root: Path, task: str, backbone: str) -> Path:
    return ckpt_root / f"{task}_{backbone}_lr_10eph"


def _load_curves(
    ckpt_root: Path,
    task: str,
    backbone: str,
    dt_days: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray] | np.ndarray:
    run_dir = _run_dir(ckpt_root, task, backbone)
    if not (run_dir / "best.pt").exists():
        raise FileNotFoundError(f"Missing fine-tuned checkpoint: {run_dir / 'best.pt'}")
    model = load_finetuned_classifier(run_dir, backbone, device)
    bb = model.backbone
    age_bb = _resolve_age_conditioned_backbone(model)
    if age_bb is None:
        tw = bb.time_aware_attention.temporal_weight
        return _w_curve_vanilla(tw, dt_days)
    tw = age_bb.time_aware_attention.temporal_weight
    emb = age_bb.time_aware_attention.age_emb
    return {float(a): _w_curve_age(tw, emb, dt_days, float(a)) for a in REPRESENTATIVE_AGES_YR}


def _plot_tale_ehr(
    curves: dict[str, np.ndarray],
    dt_days: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"pneumonia": "-", "heart_malformations": "--"}
    colors = {"pneumonia": "#4C72B0", "heart_malformations": "#DD8452"}
    for task, w in curves.items():
        ax.plot(
            dt_days,
            w,
            linestyle=styles[task],
            color=colors[task],
            linewidth=2.2,
            label=TASK_LABELS[task],
        )
    ax.set_xlabel("Δt (days since landmark event)")
    ax.set_ylabel("w(Δt)")
    ax.set_title("PIC fine-tuned · TALE-EHR · attention temporal decay")
    ax.set_xlim(0, 730)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def _plot_age_conditioned(
    curves_by_task: dict[str, dict[float, np.ndarray]],
    dt_days: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(REPRESENTATIVE_AGES_YR)))
    for ax, task in zip(axes, TASKS):
        fam = curves_by_task[task]
        for i, age in enumerate(REPRESENTATIVE_AGES_YR):
            ax.plot(dt_days, fam[float(age)], color=cmap[i], linewidth=1.8, label=f"{age:g} yr")
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Δt (days)")
        ax.set_xlim(0, 730)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, title="Patient age", loc="lower left")
    axes[0].set_ylabel("w(Δt, age)")
    fig.suptitle("PIC fine-tuned · Age-conditioned · attention temporal decay", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def _plot_combined(
    vanilla: dict[str, np.ndarray],
    age_at_2: dict[str, np.ndarray],
    dt_days: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    specs = [
        ("pneumonia", "TALE-EHR", vanilla["pneumonia"], "-", "#4C72B0", 2.4),
        ("heart_malformations", "TALE-EHR", vanilla["heart_malformations"], "--", "#DD8452", 2.4),
        ("pneumonia", f"Age-conditioned (a={COMBINED_AGE_YR:g} yr)", age_at_2["pneumonia"], "-", "#C44E52", 2.0),
        (
            "heart_malformations",
            f"Age-conditioned (a={COMBINED_AGE_YR:g} yr)",
            age_at_2["heart_malformations"],
            "--",
            "#55A868",
            2.0,
        ),
    ]
    for task, model_lbl, w, ls, color, lw in specs:
        ax.plot(
            dt_days,
            w,
            linestyle=ls,
            color=color,
            linewidth=lw,
            label=f"{TASK_LABELS[task]} · {model_lbl}",
        )
    ax.set_xlabel("Δt (days since landmark event)")
    ax.set_ylabel("w(Δt)")
    ax.set_title("PIC fine-tuned decay comparison (attention module)")
    ax.set_xlim(0, 730)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIC decay comparison plots.")
    p.add_argument("--ckpt_root", type=Path, default=PIC_CKPT_ROOT)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "figures" / "pic_decay",
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    dt_days = _dt_grid()

    vanilla_curves: dict[str, np.ndarray] = {}
    age_families: dict[str, dict[float, np.ndarray]] = {}
    age_at_2: dict[str, np.ndarray] = {}

    for task in TASKS:
        print(f"Loading TALE-EHR · {task} ...", flush=True)
        w_v = _load_curves(args.ckpt_root, task, "vanilla", dt_days, device)
        if isinstance(w_v, dict):
            raise RuntimeError(f"Expected scalar decay curve for {task}/vanilla")
        vanilla_curves[task] = w_v
        print(f"Loading age-conditioned · {task} ...", flush=True)
        fam = _load_curves(args.ckpt_root, task, "age", dt_days, device)
        if not isinstance(fam, dict):
            raise RuntimeError(f"Expected age family for {task}/age")
        age_families[task] = fam
        age_at_2[task] = fam[float(COMBINED_AGE_YR)]

    out = args.out_dir
    _plot_tale_ehr(vanilla_curves, dt_days, out / "pic_decay_tale_ehr.png")
    _plot_age_conditioned(age_families, dt_days, out / "pic_decay_age_conditioned.png")
    _plot_combined(vanilla_curves, age_at_2, dt_days, out / "pic_decay_combined.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
