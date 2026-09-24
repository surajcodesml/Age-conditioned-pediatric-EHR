#!/usr/bin/env python3
"""Build multi-horizon labels from an existing benchmark scenario.

Adds labels for horizons: 30, 90, 180, 365 days.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_SEED, DAYS_PER_YEAR, SCENARIO_SPECS, tau_from_days, z_age
from ground_truth import ExampleSignals, compute_target_logits, sample_targets

HORIZONS_DAYS = (30.0, 90.0, 180.0, 365.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=DATA_SEED)
    args = ap.parse_args()

    scenario_dir = args.scenario_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (scenario_dir / "meta.json").open() as f:
        meta = json.load(f)
    with (scenario_dir / "target_specs.json").open() as f:
        specs = json.load(f)

    df = pd.read_parquet(scenario_dir / "examples.parquet")
    n = len(df)
    n_t = len(specs)

    rng = np.random.default_rng(args.seed + 10101)

    # For each horizon, we will have a separate label matrix
    # Shape: [n, n_horizons, n_t]
    Y_all = np.zeros((n, len(HORIZONS_DAYS), n_t), dtype=np.float32)
    logits_all = np.zeros((n, len(HORIZONS_DAYS), n_t), dtype=np.float64)

    scenario = meta["scenario"]
    theta0 = meta["theta0"]
    beta = meta["beta_true"]
    from config import NOISE_STD

    # Pre-generate noise for consistency (one noise vector per patient/horizon)
    noise = rng.normal(0.0, NOISE_STD, size=(n, len(HORIZONS_DAYS), n_t))

    for i, row in enumerate(df.itertuples(index=False)):
        age_base = float(row.age_at_cutoff)
        
        codes = np.array(row.history_codes, dtype=object)
        lags = np.array(row.history_lag_days, dtype=np.float64)
        
        # Keep only signals for the mechanism
        mask = [str(c).startswith("SYN_SIGNAL") for c in codes]
        codes = codes[mask]
        lags = lags[mask]
        
        for h_idx, H in enumerate(HORIZONS_DAYS):
            future_age = age_base + H / DAYS_PER_YEAR
            future_lags = lags + H
            future_tau = tau_from_days(future_lags)
            
            sig = ExampleSignals(
                codes=codes,
                lag_days=future_lags,
                tau=future_tau,
                times=np.array([]), # times not used in compute_target_logits
            )
            
            logits, probs, _, _ = compute_target_logits(
                age=future_age,
                signals=sig,
                specs=specs,
                scenario=scenario,
                theta0=theta0,
                beta=beta,
                noise=noise[i, h_idx],
            )
            y = sample_targets(probs, rng)
            
            Y_all[i, h_idx] = y
            logits_all[i, h_idx] = logits

    np.savez_compressed(
        out_dir / "labels_forecast.npz",
        Y=Y_all,
        logits=logits_all,
        horizons=np.array(HORIZONS_DAYS),
    )
    print(f"Wrote {out_dir / 'labels_forecast.npz'} for horizons {HORIZONS_DAYS}")

if __name__ == "__main__":
    main()
