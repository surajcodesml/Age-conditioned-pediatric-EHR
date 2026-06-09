#!/usr/bin/env python3
"""Visualize age-conditioned temporal decay kernels from PIC fine-tuned checkpoints."""

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

from model.age_embedding import FourierAgeEmbedding
from model.time_aware_attention import PolynomialTemporalWeight
from model.time_aware_attention_age import AgeConditionedPolynomialWeight

from finetune.PIC.pic_age_eval_common import (
    DEV_BANDS,
    PIC_CKPT_ROOT,
    REPRESENTATIVE_AGES_YR,
    TASKS,
    detect_variant_from_state_dict,
    kernel_deviation_at_band_center,
    load_finetuned_classifier,
)


def _dt_grid() -> np.ndarray:
    """0-730 days: log-spaced positive values plus 0."""
    pos = np.geomspace(0.5, 730.0, num=200)
    return np.unique(np.concatenate([[0.0], pos]))


def _log_dt_days(dt_days: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.log1p(dt_days / 7.0), dtype=torch.float32)


@torch.no_grad()
def _w_curve_age(
    temporal_weight: AgeConditionedPolynomialWeight,
    age_emb: FourierAgeEmbedding,
    dt_days: np.ndarray,
    age_yr: float,
) -> np.ndarray:
    log_dt = _log_dt_days(dt_days).view(1, 1, -1)
    age_t = torch.tensor([float(age_yr)], dtype=torch.float32)
    age_feat = age_emb(age_t.clamp(min=0.0)).unsqueeze(1)
    w = temporal_weight(log_dt, age_feat).squeeze().cpu().numpy()
    return w.astype(np.float64)


@torch.no_grad()
def _w_curve_vanilla(temporal_weight: PolynomialTemporalWeight, dt_days: np.ndarray) -> np.ndarray:
    log_dt = _log_dt_days(dt_days).view(1, 1, -1)
    w = temporal_weight(log_dt).squeeze().cpu().numpy()
    return w.astype(np.float64)


@torch.no_grad()
def _alpha_at_age(
    temporal_weight: AgeConditionedPolynomialWeight,
    age_emb: FourierAgeEmbedding,
    ages_yr: np.ndarray,
) -> np.ndarray:
    age_t = torch.as_tensor(ages_yr, dtype=torch.float32)
    age_feat = age_emb(age_t.clamp(min=0.0))
    delta = temporal_weight.age_coeff_gen(age_feat)
    alpha = temporal_weight.coefficients.unsqueeze(0) + delta
    return alpha.cpu().numpy().astype(np.float64)


@torch.no_grad()
def _alpha_sensitivity(
    temporal_weight: AgeConditionedPolynomialWeight,
    age_emb: FourierAgeEmbedding,
    ages_yr: np.ndarray,
) -> np.ndarray:
    alpha = _alpha_at_age(temporal_weight, age_emb, ages_yr)
    base = temporal_weight.coefficients.detach().cpu().numpy()
    return np.linalg.norm(alpha - base.reshape(1, -1), axis=1)


def _half_life_days(dt_days: np.ndarray, w: np.ndarray) -> float:
    w0 = float(w[0])
    if not np.isfinite(w0) or w0 <= 0:
        return float("nan")
    target = 0.5 * w0
    below = np.where(w <= target)[0]
    if below.size == 0:
        return float("nan")
    return float(dt_days[int(below[0])])


def _auc_w(dt_days: np.ndarray, w: np.ndarray) -> float:
    return float(np.trapezoid(w, dt_days))


def _plot_kernel_family(
    task: str,
    dt_days: np.ndarray,
    curves_age: dict[str, dict[float, np.ndarray]],
    vanilla_w: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(REPRESENTATIVE_AGES_YR)))
    for ax, module in zip(axes, ("attention", "aggregation")):
        v = vanilla_w[module]
        ax.plot(dt_days, v, color="black", linewidth=2.5, linestyle="--", label="vanilla (age-independent)")
        for i, age in enumerate(REPRESENTATIVE_AGES_YR):
            ax.plot(dt_days, curves_age[module][age], color=colors[i], label=f"{age:g} yr")
        ax.set_ylabel("w(dt)")
        ax.set_title(f"{module}: age-indexed kernel family — {task}")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("dt_days (lag from current event)")
    fig.suptitle(f"NOVEL: temporal kernel family vs vanilla — {task}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_coeff_trajectories(
    task: str,
    ages: np.ndarray,
    alpha_age: dict[str, np.ndarray],
    vanilla_coeffs: dict[str, np.ndarray],
    inactive: bool,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    k_range = range(vanilla_coeffs["attention"].shape[0])
    for ax, module in zip(axes, ("attention", "aggregation")):
        base = vanilla_coeffs[module]
        alpha = alpha_age[module]
        for k in k_range:
            ax.plot(ages, alpha[:, k], label=f"alpha_{k}(a)")
            ax.axhline(float(base[k]), color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_title(f"{module}: coefficient trajectories — {task}")
        ax.set_ylabel("alpha_k")
        ax.grid(alpha=0.3)
        if module == "attention":
            ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("age (years)")
    if inactive:
        fig.text(0.5, 0.01, "AGE CONDITIONING INACTIVE (alpha_k(a) ~ flat / vanilla)", ha="center", color="red", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sensitivity(task: str, ages: np.ndarray, sens: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for module, style in [("attention", "-"), ("aggregation", "--")]:
        ax.plot(ages, sens[module], style, label=module)
    ax.set_xlabel("age (years)")
    ax.set_ylabel(r"$||\alpha(a) - \alpha_{base}||_2$")
    ax.set_title(f"Age sensitivity of polynomial coefficients — {task}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_horizon(task: str, ages: np.ndarray, half_life: dict[str, np.ndarray], auc: dict[str, np.ndarray], out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(ages, half_life["attention"], "o-", color="#C44E52", label="t_50% attention")
    ax1.plot(ages, half_life["aggregation"], "s--", color="#4C72B0", label="t_50% aggregation")
    ax2.plot(ages, auc["attention"], "^-", color="#55A868", label="AUC attention")
    ax2.plot(ages, auc["aggregation"], "v--", color="#8172B2", label="AUC aggregation")
    ax1.set_xlabel("age (years)")
    ax1.set_ylabel("dt_days at 50% of w(0)  [effective memory horizon]")
    ax2.set_ylabel("integral of w(dt) over 0-730d")
    ax1.set_title(f"NOVEL: effective temporal relevance window by developmental stage — {task}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _check_inactive(alpha_age: dict[str, np.ndarray], vanilla_coeffs: dict[str, np.ndarray], ages: np.ndarray) -> bool:
    inactive = True
    for module in ("attention", "aggregation"):
        alpha = alpha_age[module]
        base = vanilla_coeffs[module]
        spread = float(np.max(alpha) - np.min(alpha))
        base_spread = float(np.max(base) - np.min(base))
        if spread > 1e-4 or np.max(np.abs(alpha - base.reshape(1, -1))) > 1e-3:
            inactive = False
    return inactive


def process_task(task: str, ckpt_root: Path, out_dir: Path) -> Path:
    age_dir = ckpt_root / f"{task}_age"
    van_dir = ckpt_root / f"{task}_vanilla"
    age_model = load_finetuned_classifier(age_dir, "age", torch.device("cpu"))
    van_model = load_finetuned_classifier(van_dir, "vanilla", torch.device("cpu"))

    sd_age = torch.load(age_dir / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    sd_van = torch.load(van_dir / "best.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    print(f"[sanity] {task} age ckpt variant: {detect_variant_from_state_dict(sd_age)}", flush=True)
    print(f"[sanity] {task} vanilla ckpt variant: {detect_variant_from_state_dict(sd_van)}", flush=True)

    bb_age = age_model.backbone
    bb_van = van_model.backbone

    dt_days = _dt_grid()
    ages_dense = np.linspace(0.0, 18.0, num=181)

    modules = ("attention", "aggregation")
    curves_age: dict[str, dict[float, np.ndarray]] = {m: {} for m in modules}
    vanilla_w: dict[str, np.ndarray] = {}
    alpha_age: dict[str, np.ndarray] = {}
    vanilla_coeffs: dict[str, np.ndarray] = {}
    sens: dict[str, np.ndarray] = {}
    half_life: dict[str, np.ndarray] = {}
    auc_w: dict[str, np.ndarray] = {}

    for module in modules:
        if module == "attention":
            tw_a = bb_age.time_aware_attention.temporal_weight
            emb_a = bb_age.time_aware_attention.age_emb
            tw_v = bb_van.time_aware_attention.temporal_weight
        else:
            tw_a = bb_age.temporal_aggregation.temporal_weight
            emb_a = bb_age.temporal_aggregation.age_emb
            tw_v = bb_van.temporal_aggregation.temporal_weight

        vanilla_coeffs[module] = tw_v.coefficients.detach().cpu().numpy()
        vanilla_w[module] = _w_curve_vanilla(tw_v, dt_days)
        alpha_age[module] = _alpha_at_age(tw_a, emb_a, ages_dense)
        sens[module] = _alpha_sensitivity(tw_a, emb_a, ages_dense)
        hl = []
        au = []
        for age in ages_dense:
            w = _w_curve_age(tw_a, emb_a, dt_days, float(age))
            hl.append(_half_life_days(dt_days, w))
            au.append(_auc_w(dt_days, w))
        half_life[module] = np.asarray(hl, dtype=np.float64)
        auc_w[module] = np.asarray(au, dtype=np.float64)
        for age in REPRESENTATIVE_AGES_YR:
            curves_age[module][age] = _w_curve_age(tw_a, emb_a, dt_days, age)

    d_attn_01 = float(
        torch.norm(
            bb_age.time_aware_attention.temporal_weight.age_coeff_gen(
                bb_age.time_aware_attention.age_emb(torch.tensor([[0.1]]))
            ).squeeze()
        ).item()
    )
    d_attn_15 = float(
        torch.norm(
            bb_age.time_aware_attention.temporal_weight.age_coeff_gen(
                bb_age.time_aware_attention.age_emb(torch.tensor([[15.0]]))
            ).squeeze()
        ).item()
    )
    print(f"[sanity] {task} ||delta_alpha|| attention: a=0.1yr -> {d_attn_01:.6f}, a=15yr -> {d_attn_15:.6f}", flush=True)
    if d_attn_01 < 1e-5 and d_attn_15 < 1e-5:
        print(f"[sanity] WARNING {task}: age conditioning appears DEAD in attention AgeCoefficientGenerator", flush=True)

    inactive = _check_inactive(alpha_age, vanilla_coeffs, ages_dense)
    if inactive:
        print(f"[{task}] AGE CONDITIONING INACTIVE (alpha_k(a) ~ vanilla for all ages)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_kernel_family(task, dt_days, curves_age, vanilla_w, out_dir / f"{task}_family.png")
    _plot_coeff_trajectories(task, ages_dense, alpha_age, vanilla_coeffs, inactive, out_dir / f"{task}_coeffs.png")
    _plot_sensitivity(task, ages_dense, sens, out_dir / f"{task}_sensitivity.png")
    _plot_horizon(task, ages_dense, half_life, auc_w, out_dir / f"{task}_horizon.png")

    band_deviation = {
        band.name: {
            "center_yr": band.center_yr,
            "deviation_attention": kernel_deviation_at_band_center(bb_age, band, "attention"),
            "deviation_aggregation": kernel_deviation_at_band_center(bb_age, band, "aggregation"),
        }
        for band in DEV_BANDS
    }

    npz_path = out_dir / f"{task}_kernel.npz"
    np.savez(
        npz_path,
        dt_days=dt_days,
        ages_dense=ages_dense,
        representative_ages=np.asarray(REPRESENTATIVE_AGES_YR, dtype=np.float64),
        vanilla_w_attention=vanilla_w["attention"],
        vanilla_w_aggregation=vanilla_w["aggregation"],
        vanilla_coeffs_attention=vanilla_coeffs["attention"],
        vanilla_coeffs_aggregation=vanilla_coeffs["aggregation"],
        alpha_age_attention=alpha_age["attention"],
        alpha_age_aggregation=alpha_age["aggregation"],
        sensitivity_attention=sens["attention"],
        sensitivity_aggregation=sens["aggregation"],
        half_life_days_attention=half_life["attention"],
        half_life_days_aggregation=half_life["aggregation"],
        auc_w_attention=auc_w["attention"],
        auc_w_aggregation=auc_w["aggregation"],
        band_names=np.asarray([b.name for b in DEV_BANDS], dtype=object),
        band_center_yr=np.asarray([b.center_yr for b in DEV_BANDS], dtype=np.float64),
        band_deviation_attention=np.asarray([band_deviation[b.name]["deviation_attention"] for b in DEV_BANDS]),
        band_deviation_aggregation=np.asarray([band_deviation[b.name]["deviation_aggregation"] for b in DEV_BANDS]),
        **{f"w_age_attention_{a:g}yr": curves_age["attention"][a] for a in REPRESENTATIVE_AGES_YR},
        **{f"w_age_aggregation_{a:g}yr": curves_age["aggregation"][a] for a in REPRESENTATIVE_AGES_YR},
    )
    print(f"[{task}] wrote plots + {npz_path}", flush=True)
    return npz_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIC age-kernel mechanism visualization.")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--ckpt_root", type=Path, default=PIC_CKPT_ROOT)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "results" / "pic" / "age_kernel")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        process_task(task, args.ckpt_root, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
