"""D8 — Is the objective horizon-marginalized?

D8.1 (CPU): histogram of days between window end and target visit.
D8.2 (GPU): vanilla recall@10 binned by that gap (needs D8.1 JSON + D1-style forward).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model_new import diagnostics as D
from model_new.audit.common import REPO_ROOT
from model_new.audit.signal.common import (
    add_common_args,
    assert_batch_hash,
    base_result_meta,
    ensure_batches,
    eval_model_on_store,
    load_arm_model,
    probe_precision,
    require_cuda,
    resolve_device,
    write_json_atomic,
)


GAP_BINS: tuple[tuple[str, float, float], ...] = (
    ("<0", float("-inf"), 0.0),
    ("0-1d", 0.0, 1.0),
    ("1-7d", 1.0, 7.0),
    ("7-30d", 7.0, 30.0),
    ("30-90d", 30.0, 90.0),
    ("90-365d", 90.0, 365.0),
    (">365d", 365.0, float("inf")),
)


def _gap_array(store: dict) -> np.ndarray:
    return store["target_gap_days"].numpy().astype(np.float64)


def run_d8_1(ctx: dict, store: dict) -> dict:
    gaps = _gap_array(store)
    finite = gaps[np.isfinite(gaps)]
    horizon = np.maximum(finite, 0.0)

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else float("nan")

    hist = []
    for name, lo, hi in GAP_BINS:
        if np.isneginf(lo):
            sel = finite < hi
        elif np.isinf(hi):
            sel = finite >= lo
        else:
            sel = (finite >= lo) & (finite < hi)
        hist.append({"bin": name, "n": int(sel.sum()),
                     "frac": float(sel.mean()) if finite.size else float("nan")})

    out = {
        **base_result_meta(ctx, store),
        "n": int(finite.size),
        "frac_negative": float((finite < 0).mean()) if finite.size else float("nan"),
        "signed": {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "iqr": [pct(finite, 25), pct(finite, 75)],
            "p10": pct(finite, 10),
            "p90": pct(finite, 90),
        },
        "horizon_clipped_at_0": {
            "median": float(np.median(horizon)) if horizon.size else float("nan"),
            "iqr": [pct(horizon, 25), pct(horizon, 75)],
            "p10": pct(horizon, 10),
            "p90": pct(horizon, 90),
        },
        "histogram": hist,
    }
    return out


def run_d8_2(ctx: dict, store: dict, device: torch.device,
             d81: dict | None = None) -> dict:
    assert_batch_hash(store, ctx["batch_meta"]["batch_list_hash"])
    require_cuda(device, batch_size=int(store["batch_size"]), label="D8.2")
    model, meta = load_arm_model(ctx, "vanilla", device)
    prec = probe_precision(model, store, device)
    dtype = torch.bfloat16 if prec["dtype"] == "bf16" else torch.float32
    res = eval_model_on_store(model, store, device, dtype=dtype)
    if res["batch_list_hash"] != store["batch_list_hash"]:
        raise AssertionError("[HARD] D8.2 batch hash mismatch")
    gaps = res["target_gap_days"]
    r10 = res["recall_per_example"]["recall@10"]

    by_bin = []
    recalls = []
    for name, lo, hi in GAP_BINS:
        if np.isneginf(lo):
            sel = gaps < hi
        elif np.isinf(hi):
            sel = gaps >= lo
        else:
            sel = (gaps >= lo) & (gaps < hi)
        vals = r10[sel]
        vals = vals[np.isfinite(vals)]
        entry = {
            "bin": name, "n": int(vals.size),
            "recall@10": float(vals.mean()) if vals.size else float("nan"),
        }
        by_bin.append(entry)
        if name != "<0" and vals.size >= 5:
            recalls.append(entry["recall@10"])

    # Flat vs steep: coefficient of variation across positive-horizon bins.
    arr = np.asarray(recalls, dtype=np.float64)
    if arr.size >= 2 and np.nanmean(arr) != 0:
        cv = float(np.nanstd(arr) / abs(np.nanmean(arr)))
        # Also check drop from shortest to longest positive bin.
        drop = float(arr[0] - arr[-1]) if arr.size >= 2 else float("nan")
    else:
        cv, drop = float("nan"), float("nan")
    steep = bool(np.isfinite(drop) and drop > 0.03) or bool(np.isfinite(cv) and cv > 0.15)
    flat = not steep if np.isfinite(cv) else None

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        **base_result_meta(ctx, store),
        "arm": "vanilla",
        "checkpoint": meta,
        "precision": prec,
        "bce_mean": res["bce_mean"],
        "recall": res["recall"],
        "by_gap_bin": by_bin,
        "cv_across_bins": cv,
        "recall_drop_short_to_long": drop,
        "d8_1_ref": {
            "median_horizon": (d81 or {}).get("horizon_clipped_at_0", {}).get("median"),
        },
        "verdict": {
            "flat": flat,
            "route": (
                "horizon-marginalized → TTE justified" if flat
                else "horizon already used → TTE unnecessary" if flat is False
                else "insufficient bins to judge"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--part", choices=("1", "2", "all"), default="all")
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ctx, store = ensure_batches(
        out_dir, smoke=args.smoke, batch_size=args.batch_size,
        force=args.force, run_root=args.run_root,
    )
    ctx["seed"] = args.seed

    d81 = None
    if args.part in ("1", "all"):
        D.print_block("D8.1 gap histogram", [f"out={out_dir}  smoke={args.smoke}"])
        d81 = run_d8_1(ctx, store)
        write_json_atomic(out_dir / "d8_horizon_hist.json", d81)
        s = d81["signed"]
        D.print_block("D8.1 results", [
            f"n={d81['n']}  frac_neg={d81['frac_negative']:.3f}",
            f"signed median={s['median']:.3f}  IQR={s['iqr']}  "
            f"p10/p90={s['p10']:.3f}/{s['p90']:.3f}",
            f"horizon median={d81['horizon_clipped_at_0']['median']:.3f}",
        ])

    if args.part in ("2", "all"):
        if d81 is None:
            p81 = out_dir / "d8_horizon_hist.json"
            if p81.is_file():
                import json
                d81 = json.loads(p81.read_text())
        device = resolve_device(args.device)
        D.print_block("D8.2 recall vs gap", [
            f"out={out_dir}  smoke={args.smoke}  device={device}",
        ])
        d82 = run_d8_2(ctx, store, device, d81=d81)
        write_json_atomic(out_dir / "d8_horizon_recall.json", d82)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [e["bin"] for e in d82["by_gap_bin"] if e["bin"] != "<0"]
            vals = [e["recall@10"] for e in d82["by_gap_bin"] if e["bin"] != "<0"]
            ns = [e["n"] for e in d82["by_gap_bin"] if e["bin"] != "<0"]
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            ax.bar(range(len(labels)), vals, color="#1f4e79")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("vanilla recall@10")
            ax.set_title("D8: recall@10 vs target gap")
            for i, n in enumerate(ns):
                ax.text(i, (vals[i] if np.isfinite(vals[i]) else 0) + 0.005,
                        f"n={n}", ha="center", fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "d8_recall_vs_gap.png", dpi=140)
            plt.close(fig)
        except Exception as e:
            D.print_block("D8 figure", [f"skipped: {e}"])

        D.print_block("D8.2 results", [
            f"batch_list_hash={d82['batch_list_hash']}",
            f"cv={d82['cv_across_bins']:.4g}  drop={d82['recall_drop_short_to_long']:.4g}",
            f"verdict: {d82['verdict']['route']}",
        ] + [f"{e['bin']}: r@10={e['recall@10']:.4f} n={e['n']}" for e in d82["by_gap_bin"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
