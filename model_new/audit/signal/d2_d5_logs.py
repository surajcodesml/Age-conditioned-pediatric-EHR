"""D2 / D5 — Free diagnostics from train.json (no forward pass).

D2: correlate centered Δα(a) curves across epochs 3→8, per site.
D5: re-select checkpoints on recall@10; rebuild the T1 arm table from train.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model_new import diagnostics as D
from model_new.audit.common import ARMS, REPO_ROOT, discover_runs, read_json
from model_new.audit.signal.common import (
    SIGNAL_SEED,
    add_common_args,
    write_json_atomic,
)


def _centered_delta_alpha(site_curve: list[list[float]]) -> np.ndarray:
    """``site_curve`` is [n_ages, s]; center over ages."""
    arr = np.asarray(site_curve, dtype=np.float64)
    return arr - arr.mean(axis=0, keepdims=True)


def run_d2(run_dirs: dict[str, Path], *, epochs: tuple[int, ...] = tuple(range(3, 9))
           ) -> dict:
    """Pearson correlation of centered Δα across successive epochs, per site (kernel)."""
    kernel_dir = run_dirs["kernel"]
    train = read_json(kernel_dir / "train.json")
    by_ep = {int(e["epoch"]): e for e in train}
    missing = [ep for ep in epochs if ep not in by_ep]
    if missing:
        raise FileNotFoundError(
            f"[HARD] kernel train.json missing epochs {missing} under {kernel_dir}")

    sites = list(by_ep[epochs[0]]["delta_alpha_grid"]["sites"].keys())
    per_site = {}
    for site in sites:
        curves = []
        for ep in epochs:
            curves.append(_centered_delta_alpha(
                by_ep[ep]["delta_alpha_grid"]["sites"][site]))
        # Flatten each epoch's centered curve; correlate consecutive epochs.
        flat = [c.reshape(-1) for c in curves]
        pair_corr = []
        for i in range(len(flat) - 1):
            a, b = flat[i], flat[i + 1]
            if a.std() < 1e-12 or b.std() < 1e-12:
                r = float("nan")
            else:
                r = float(np.corrcoef(a, b)[0, 1])
            pair_corr.append({
                "from_epoch": int(epochs[i]),
                "to_epoch": int(epochs[i + 1]),
                "pearson_r": r,
            })
        rs = [p["pearson_r"] for p in pair_corr if np.isfinite(p["pearson_r"])]
        mean_r = float(np.mean(rs)) if rs else float("nan")
        per_site[site] = {
            "pairwise": pair_corr,
            "mean_consecutive_r": mean_r,
            "unidentified": bool(np.isfinite(mean_r) and mean_r < 0.3),
        }

    mean_across = float(np.nanmean(
        [per_site[s]["mean_consecutive_r"] for s in sites]))
    return {
        "seed": SIGNAL_SEED,
        "arm": "kernel",
        "run_dir": str(kernel_dir),
        "epochs": list(epochs),
        "sites": per_site,
        "mean_consecutive_r_across_sites": mean_across,
        "verdict": {
            "route": (
                "age function unidentified, not merely unrewarded"
                if (np.isfinite(mean_across) and mean_across < 0.3)
                else "Δα trajectory stable across epochs 3→8"
            ),
        },
    }


def _select_by(train: list[dict], key: str, maximize: bool) -> dict:
    best = max(train, key=lambda e: float(e[key])) if maximize \
        else min(train, key=lambda e: float(e[key]))
    return {
        "epoch": int(best["epoch"]),
        "val_loss": float(best["val_loss"]),
        "recall@5": float(best.get("recall@5", float("nan"))),
        "recall@10": float(best.get("recall@10", float("nan"))),
        "recall@20": float(best.get("recall@20", float("nan"))),
    }


def run_d5(run_dirs: dict[str, Path], configs: dict[str, dict]) -> dict:
    """Re-select on recall@10; report both selections and primary_endpoint."""
    by_arm = {}
    for arm in ARMS:
        train = read_json(run_dirs[arm] / "train.json")
        by_bce = _select_by(train, "val_loss", maximize=False)
        by_r10 = _select_by(train, "recall@10", maximize=True)
        by_arm[arm] = {
            "selection_val_bce": by_bce,
            "selection_recall@10": by_r10,
            "same_epoch": by_bce["epoch"] == by_r10["epoch"],
        }

    # T1-style tables at each selection rule (numbers from train.json, not re-eval).
    def table(rule: str) -> dict:
        rows = {}
        for arm in ARMS:
            rows[arm] = by_arm[arm][rule]
        return rows

    primary = configs["vanilla"].get("primary_endpoint")
    return {
        "seed": SIGNAL_SEED,
        "primary_endpoint": primary,
        "per_arm": by_arm,
        "t1_table_val_bce_selection": table("selection_val_bce"),
        "t1_table_recall@10_selection": table("selection_recall@10"),
        "any_epoch_disagreement": any(not by_arm[a]["same_epoch"] for a in ARMS),
    }


def main(argv: list[str] | None = None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.run_root if args.run_root.is_absolute() else REPO_ROOT / args.run_root
    run_dirs = discover_runs(run_root)
    configs = {arm: read_json(run_dirs[arm] / "config.json") for arm in ARMS}

    # Hash of fixed batches if present (for JSON invariant); optional for log-only tests.
    batch_hash = None
    meta_path = out_dir / "fixed_batches_meta.json"
    if meta_path.is_file():
        batch_hash = read_json(meta_path).get("batch_list_hash")

    D.print_block("D2 Δα epoch correlation", [f"out={out_dir}"])
    d2 = run_d2(run_dirs)
    d2["batch_list_hash"] = batch_hash
    write_json_atomic(out_dir / "d2_d5_logs.json", {"d2": d2})  # placeholder then merge

    D.print_block("D5 recall@10 re-selection", [f"out={out_dir}"])
    d5 = run_d5(run_dirs, configs)
    d5["batch_list_hash"] = batch_hash

    result = {
        "batch_list_hash": batch_hash,
        "seed": SIGNAL_SEED,
        "smoke": bool(args.smoke),
        "d2": d2,
        "d5": d5,
    }
    write_json_atomic(out_dir / "d2_d5_logs.json", result)

    lines = [
        f"D2 mean consecutive r={d2['mean_consecutive_r_across_sites']:.4f}",
        f"D2 verdict: {d2['verdict']['route']}",
        f"D5 any epoch disagreement={d5['any_epoch_disagreement']}",
        f"primary_endpoint.metric={((primary := d5.get('primary_endpoint')) or {}).get('metric')}",
    ]
    for arm in ARMS:
        a = d5["per_arm"][arm]
        lines.append(
            f"{arm}: bce_ep={a['selection_val_bce']['epoch']} "
            f"r10_ep={a['selection_recall@10']['epoch']} "
            f"r@10(bce_sel)={a['selection_val_bce']['recall@10']:.4f} "
            f"r@10(r10_sel)={a['selection_recall@10']['recall@10']:.4f}"
        )
    D.print_block("D2/D5 results", lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
