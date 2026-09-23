#!/usr/bin/env python3
"""A3: background-content ablation — signal-only and randomized-background variants.

Does **not** regenerate Y. Same patients/targets; only history inputs change.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_SEED, DEFAULT_OUTPUT_DIR, all_signal_codes


def _make_signal_only(examples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in examples.itertuples(index=False):
        codes, types, lags, taus = [], [], [], []
        hist_tau = list(row.history_tau) if isinstance(row.history_tau, list) else []
        for i, t in enumerate(row.history_types):
            if t == "signal":
                codes.append(row.history_codes[i])
                types.append(t)
                lags.append(float(row.history_lag_days[i]))
                taus.append(float(hist_tau[i]) if hist_tau else float("nan"))
        from config import tau_from_days

        if any(np.isnan(taus)):
            taus = tau_from_days(lags).tolist()
        d = row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())
        # itertuples namedtuple
        d = {
            "example_id": int(row.example_id),
            "patient_id": str(row.patient_id),
            "split": str(row.split),
            "cutoff_time": str(row.cutoff_time),
            "age_at_cutoff": float(row.age_at_cutoff),
            "z_age": float(row.z_age),
            "history_codes": codes,
            "history_types": types,
            "history_lag_days": lags,
            "history_tau": taus,
            "labels": row.labels,
        }
        rows.append(d)
    return pd.DataFrame(rows)


def _make_bg_randomized(examples: pd.DataFrame, seed: int = 20260923) -> pd.DataFrame:
    """Permute background CODE identities across the cohort; keep timestamps/signals."""
    rng = np.random.default_rng(seed)
    # Collect all background codes pool
    bg_pool = []
    for row in examples.itertuples(index=False):
        for i, t in enumerate(row.history_types):
            if t != "signal":
                bg_pool.append(str(row.history_codes[i]))
    bg_pool = np.array(bg_pool, dtype=object)
    rng.shuffle(bg_pool)
    ptr = 0
    rows = []
    sig_set = set(all_signal_codes())
    for row in examples.itertuples(index=False):
        codes, types, lags = [], [], []
        hist_tau = list(row.history_tau) if isinstance(row.history_tau, list) else []
        taus = []
        for i, t in enumerate(row.history_types):
            types.append(t)
            lags.append(float(row.history_lag_days[i]))
            taus.append(float(hist_tau[i]) if hist_tau else float("nan"))
            if t == "signal" or str(row.history_codes[i]) in sig_set:
                codes.append(row.history_codes[i])
            else:
                codes.append(str(bg_pool[ptr % len(bg_pool)]))
                ptr += 1
        from config import tau_from_days

        if any(np.isnan(taus)):
            taus = tau_from_days(lags).tolist()
        rows.append(
            {
                "example_id": int(row.example_id),
                "patient_id": str(row.patient_id),
                "split": str(row.split),
                "cutoff_time": str(row.cutoff_time),
                "age_at_cutoff": float(row.age_at_cutoff),
                "z_age": float(row.z_age),
                "history_codes": codes,
                "history_types": types,
                "history_lag_days": lags,
                "history_tau": taus,
                "labels": row.labels,
            }
        )
    return pd.DataFrame(rows)


def build_variants(scenario_dir: Path, out_root: Path) -> dict[str, str]:
    examples = pd.read_parquet(scenario_dir / "examples.parquet")
    out_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in (
        ("full", lambda e: e),
        ("signal_only", _make_signal_only),
        ("bg_randomized", lambda e: _make_bg_randomized(e)),
    ):
        dest = out_root / name
        dest.mkdir(parents=True, exist_ok=True)
        # Copy static artifacts
        for fname in (
            "labels.npz",
            "target_specs.json",
            "meta.json",
            "splits.json",
            "calibration.json",
            "oracle_metrics.json",
            "READY",
            "ground_truth.parquet",
        ):
            src = scenario_dir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)
        ex = fn(examples)
        ex.to_parquet(dest / "examples.parquet", index=False)
        paths[name] = str(dest)
        print(f"Wrote {name}: n={len(ex)} -> {dest}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--data-seed", type=int, default=DATA_SEED)
    ap.add_argument("--cohort", default="controlled")
    args = ap.parse_args()
    sdir = (
        DEFAULT_OUTPUT_DIR
        / "data"
        / f"seed{args.data_seed}"
        / args.cohort
        / args.scenario
    )
    out = (
        DEFAULT_OUTPUT_DIR
        / "data"
        / f"seed{args.data_seed}"
        / args.cohort
        / f"{args.scenario}_ablation"
    )
    paths = build_variants(sdir, out)
    (out / "variants.json").write_text(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
