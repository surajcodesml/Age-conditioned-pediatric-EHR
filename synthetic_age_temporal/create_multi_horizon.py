import numpy as np
import pandas as pd
from pathlib import Path
from ground_truth import compute_target_logits, sample_targets, ExampleSignals
from config import SCENARIO_SPECS
import json
import shutil
import sys
import os

os.chdir('/home/suraj/Git/Age-conditioned-pediatric-EHR/synthetic_age_temporal')
sys.path.append('.')

def create_horizon(base_dir, horizon_days):
    out_dir = Path(str(base_dir) + f"_h{horizon_days}")
    if out_dir.exists():
        print(f"Skipping {out_dir}, exists.")
        return
    print(f"Creating {out_dir} ...")
    shutil.copytree(base_dir, out_dir)
    
    df = pd.read_parquet(base_dir / "examples.parquet")
    specs = json.loads((base_dir / "target_specs.json").read_text())
    meta = json.loads((base_dir / "meta.json").read_text())
    
    Y = np.zeros((len(df), len(specs)), dtype=np.float32)
    rng = np.random.default_rng(meta["data_seed"] + horizon_days)
    noise = rng.normal(0.0, 0.35, size=(len(df), len(specs)))
    
    for i, row in enumerate(df.itertuples(index=False)):
        sig_idx = [j for j, t in enumerate(row.history_types) if t == "signal"]
        codes = np.array([row.history_codes[j] for j in sig_idx], dtype=object)
        lag_days = np.array([row.history_lag_days[j] for j in sig_idx]) + horizon_days
        tau = np.log1p(lag_days / 7.0)
        times = np.array([np.datetime64('2026-01-01')] * len(codes))
        sig = ExampleSignals(codes=codes, lag_days=lag_days, tau=tau, times=times)
        
        logits, probs, _, _ = compute_target_logits(
            age=float(row.age_at_cutoff),
            signals=sig,
            specs=specs,
            scenario="S2",
            theta0=0.0,
            beta=-2.5,
            noise=noise[i]
        )
        y = sample_targets(probs, rng)
        Y[i] = y
        # We need to update df with new labels, but dataframe columns cannot be updated easily if it's object. 
        # We'll build a new labels column.
        
    df["labels"] = list(Y)
    df.to_parquet(out_dir / "examples.parquet", index=False)
    # update labels.npz
    d = dict(np.load(base_dir / "labels.npz"))
    d["Y"] = Y
    np.savez_compressed(out_dir / "labels.npz", **d)

if __name__ == "__main__":
    base_s2 = Path("outputs/data/seed20260922/controlled/S2")
    for h in [30, 90, 180, 365]:
        create_horizon(base_s2, h)
