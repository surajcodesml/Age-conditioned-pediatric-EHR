#!/usr/bin/env python3
"""Link age-stratified AUROC gains to kernel deformation magnitude (mechanistic evidence)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.PIC.pic_age_eval_common import DEV_BANDS, PIC_CKPT_ROOT, TASKS


def link_task(
    task: str,
    stratified_dir: Path,
    kernel_dir: Path,
    out_dir: Path,
) -> dict:
    csv_path = stratified_dir / f"{task}.csv"
    npz_path = kernel_dir / f"{task}_kernel.npz"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run age_stratified_eval first: {csv_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Run age_kernel_viz first: {npz_path}")

    table = pd.read_csv(csv_path)
    kern = np.load(npz_path, allow_pickle=True)

    band_names = list(kern["band_names"])
    dev_attn = kern["band_deviation_attention"].astype(np.float64)
    dev_agg = kern["band_deviation_aggregation"].astype(np.float64)

    rows: list[dict] = []
    for i, band_name in enumerate(band_names):
        trow = table[table["band"] == band_name]
        if trow.empty:
            continue
        trow = trow.iloc[0]
        delta = float(trow["delta_auroc_age_minus_vanilla"])
        unreliable = bool(trow["unreliable_vanilla"]) or bool(trow["unreliable_age"])
        rows.append(
            {
                "band": band_name,
                "delta_auroc": delta,
                "kernel_dev_attention": float(dev_attn[i]),
                "kernel_dev_aggregation": float(dev_agg[i]),
                "kernel_dev_mean": float(0.5 * (dev_attn[i] + dev_agg[i])),
                "unreliable": unreliable,
            }
        )

    df = pd.DataFrame(rows)
    reliable = df[~df["unreliable"]].copy()

    def _rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan"), float("nan")
        return spearmanr(x[mask], y[mask])

    rho_attn, p_attn = _rho(reliable["kernel_dev_attention"].to_numpy(), reliable["delta_auroc"].to_numpy())
    rho_mean, p_mean = _rho(reliable["kernel_dev_mean"].to_numpy(), reliable["delta_auroc"].to_numpy())

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        color = "#BBBBBB" if r["unreliable"] else "#C44E52"
        ax.scatter(r["kernel_dev_attention"], r["delta_auroc"], s=80, color=color, edgecolors="k", linewidths=0.5)
        ax.annotate(r["band"], (r["kernel_dev_attention"], r["delta_auroc"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    if reliable.shape[0] >= 2:
        x = reliable["kernel_dev_attention"].to_numpy()
        y = reliable["delta_auroc"].to_numpy()
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min(), x.max(), num=50)
        ax.plot(xs, np.polyval(coef, xs), "k--", alpha=0.6, label="linear fit (reliable bands)")

    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel(r"$||\Delta\alpha(a_{band})||_2$  (attention AgeCoefficientGenerator)")
    ax.set_ylabel("paired ΔAUROC (age − vanilla)")
    ax.set_title(
        f"NOVEL: outcome gain vs kernel deformation — {task}\n"
        f"Spearman ρ={rho_attn:.3f} (p={p_attn:.3g}); mean-module ρ={rho_mean:.3f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{task}_mechanism_link.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_csv = out_dir / f"{task}_mechanism_link.csv"
    df.to_csv(out_csv, index=False)
    print(
        f"[{task}] mechanism link: Spearman rho(attention_dev, delta_auroc)={rho_attn:.4f} p={p_attn:.4g} "
        f"| reliable bands={len(reliable)}/{len(df)} -> {out_png}",
        flush=True,
    )
    return {
        "task": task,
        "spearman_rho_attention": float(rho_attn),
        "spearman_p_attention": float(p_attn),
        "spearman_rho_mean_modules": float(rho_mean),
        "spearman_p_mean_modules": float(p_mean),
        "n_bands": len(df),
        "n_reliable": len(reliable),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Link PIC age AUROC gains to kernel deformation.")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--stratified_dir", type=Path, default=REPO_ROOT / "results" / "pic" / "age_stratified")
    p.add_argument("--kernel_dir", type=Path, default=REPO_ROOT / "results" / "pic" / "age_kernel")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "results" / "pic" / "age_mechanism_link")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summaries: list[dict] = []
    for task in args.tasks:
        summaries.append(
            link_task(task, args.stratified_dir, args.kernel_dir, args.out_dir)
        )
    print("\n=== Mechanism-link Spearman summary ===", flush=True)
    for s in summaries:
        print(
            f"  {s['task']}: rho={s['spearman_rho_attention']:.3f} (p={s['spearman_p_attention']:.3g}) "
            f"[{s['n_reliable']}/{s['n_bands']} reliable bands]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
