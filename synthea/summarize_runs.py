#!/usr/bin/env python3
"""STEP 5/6: collect every fine-tune run's results and print the comparison table.

Reads, for each (task, arm in {vanilla, age}) run directory under --runs_root:
  - history.json           : overall + per-band test AUROC/AUPRC (written by
                             train_synthea.py via age_stratified_metrics).
  - test_predictions.parquet : per-subject (label, prob, age_at_landmark) used to
                             compute the PAIRED per-band age-minus-vanilla AUROC
                             delta with a bootstrap 95% CI (the real result).

Prints a task x arm x band table and writes a consolidated summary JSON.
Aggregate AUROC is secondary; the per-band (age - vanilla) delta is the headline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BANDS = ["<1", "1-5", "6-11", "12-17", "18-25"]
BAND_EDGES = {"<1": (0.0, 1.0), "1-5": (1.0, 6.0), "6-11": (6.0, 12.0),
              "12-17": (12.0, 18.0), "18-25": (18.0, 26.0)}
ARMS = ["vanilla", "age"]
MIN_BAND_N = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Synthea finetune comparison runs.")
    p.add_argument("--runs_root", type=Path, required=True,
                   help="Root holding <task>_<arm>/ run directories.")
    p.add_argument("--tasks", nargs="*", required=True)
    p.add_argument("--out_json", type=Path, default=None)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _band_of(age: np.ndarray) -> np.ndarray:
    out = np.full(age.shape, "", dtype=object)
    for name, (lo, hi) in BAND_EDGES.items():
        out[(age >= lo) & (age < hi)] = name
    return out


def _paired_delta(merged: pd.DataFrame, band: str, n_boot: int, seed: int):
    sub = merged[merged["band"] == band]
    if len(sub) < MIN_BAND_N or sub["label"].nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    y = sub["label"].to_numpy(np.int32)
    sv = sub["prob_vanilla"].to_numpy(np.float64)
    sa = sub["prob_age"].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    n = len(sub)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        deltas.append(roc_auc_score(yb, sa[idx]) - roc_auc_score(yb, sv[idx]))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(deltas)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> int:
    args = parse_args()
    summary: dict = {"tasks": {}}

    header = f"{'task':<10}{'band':>7}{'n':>7}{'AUROC_van':>11}{'AUROC_age':>11}{'delta':>9}{'delta_CI95':>20}"
    print("=" * len(header))
    print("Synthea: age-conditioned vs vanilla TALE-EHR  (per developmental band)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for task in args.tasks:
        preds = {}
        hist = {}
        for arm in ARMS:
            run_dir = args.runs_root / f"{task}_{arm}"
            hp = run_dir / "history.json"
            pp = run_dir / "test_predictions.parquet"
            if hp.exists():
                hist[arm] = json.loads(hp.read_text())
            if pp.exists():
                df = pd.read_parquet(pp, columns=["subject_id", "label", "prob", "age_at_landmark"])
                df["age_at_landmark"] = np.clip(df["age_at_landmark"].astype(float), 0.0, None)
                preds[arm] = df

        task_rec: dict = {"overall": {}, "bands": {}}
        for arm in ARMS:
            if arm in hist:
                m = hist[arm].get("test_metrics", {})
                task_rec["overall"][arm] = {"auroc": m.get("auroc"), "auprc": m.get("auprc"),
                                            "test_loss": hist[arm].get("test_loss")}

        merged = None
        if "vanilla" in preds and "age" in preds:
            v = preds["vanilla"].rename(columns={"prob": "prob_vanilla"})
            a = preds["age"][["subject_id", "prob"]].rename(columns={"prob": "prob_age"})
            merged = v.merge(a, on="subject_id")
            merged["band"] = _band_of(merged["age_at_landmark"].to_numpy())

        for band in BANDS:
            van = (hist.get("vanilla", {}).get("test_age_stratified", {}) or {}).get(band, {})
            age = (hist.get("age", {}).get("test_age_stratified", {}) or {}).get(band, {})
            n_band = int(van.get("n", 0) or age.get("n", 0) or 0)
            au_v = van.get("auroc", float("nan"))
            au_a = age.get("auroc", float("nan"))
            dmean = dlo = dhi = float("nan")
            if merged is not None:
                dmean, dlo, dhi = _paired_delta(merged, band, args.n_boot, args.seed)
            task_rec["bands"][band] = {
                "n": n_band, "auroc_vanilla": au_v, "auroc_age": au_a,
                "delta_age_minus_vanilla": dmean, "delta_ci": [dlo, dhi],
            }
            def f(x):
                return f"{x:.4f}" if isinstance(x, (int, float)) and np.isfinite(x) else "  n/a "
            ci = f"[{dlo:+.3f},{dhi:+.3f}]" if np.isfinite(dlo) else "        n/a        "
            print(f"{task:<10}{band:>7}{n_band:>7}{f(au_v):>11}{f(au_a):>11}{f(dmean):>9}{ci:>20}")
        # overall row
        ov = task_rec["overall"]
        au_v = ov.get("vanilla", {}).get("auroc", float("nan"))
        au_a = ov.get("age", {}).get("auroc", float("nan"))
        def f(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None and np.isfinite(x) else "  n/a "
        print(f"{task:<10}{'ALL':>7}{'':>7}{f(au_v):>11}{f(au_a):>11}{'':>9}{'':>20}")
        print("-" * len(header))
        summary["tasks"][task] = task_rec

    out_json = args.out_json or (args.runs_root / "summary.json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
