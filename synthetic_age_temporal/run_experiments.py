import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/home/suraj/Git/Age-conditioned-pediatric-EHR/synthetic_age_temporal')
sys.path.append('.')

from model_dtr import DevelopmentalTemporalRetrieval, DTRContentPersistence, DTRMultiQuery
from train_dtr import train_dtr

def train_eval(name, model_class, scenario_dir, beta_true):
    run_dir = Path(f"outputs/runs/experiment_{name}")
    print(f"Training {name} on {scenario_dir}...")
    return train_dtr(
        age_temporal=True,
        scenario_dir=Path(scenario_dir),
        run_dir=run_dir,
        beta_true=beta_true,
        aggregation="weighted_mean_plus_log_mass",
        interaction_only=True,
        max_epochs=25,
        patience=5,
        batch_size=32,
        d_model=256,
        lr=3e-4,
        seed=0,
        device="cuda",
        model_class=model_class
    )

def main():
    results = {}
    
    # 1. Baseline DTR
    print("--- BASELINE ---")
    results["Baseline_S2"] = train_eval("Baseline_S2", DevelopmentalTemporalRetrieval, "outputs/data/seed20260922/controlled/S2", -2.5)
    results["Baseline_S5"] = train_eval("Baseline_S5", DevelopmentalTemporalRetrieval, "outputs/data/seed20260922/controlled/S5", -2.5)
    results["Baseline_S6"] = train_eval("Baseline_S6", DevelopmentalTemporalRetrieval, "outputs/data/seed20260922/controlled/S6", -2.5)

    # 2. Content Persistence
    print("--- CONTENT PERSISTENCE ---")
    results["Content_S2"] = train_eval("Content_S2", DTRContentPersistence, "outputs/data/seed20260922/controlled/S2", -2.5)
    results["Content_S5"] = train_eval("Content_S5", DTRContentPersistence, "outputs/data/seed20260922/controlled/S5", -2.5)

    # 3. Multi-Query
    print("--- MULTI QUERY ---")
    results["MultiQuery_S2"] = train_eval("MultiQuery_S2", DTRMultiQuery, "outputs/data/seed20260922/controlled/S2", -2.5)
    results["MultiQuery_S6"] = train_eval("MultiQuery_S6", DTRMultiQuery, "outputs/data/seed20260922/controlled/S6", -2.5)

    # 4. Multi Horizon
    print("--- MULTI HORIZON ---")
    results["SingleHorizon_180"] = train_eval("SingleHorizon_180", DevelopmentalTemporalRetrieval, "outputs/data/seed20260922/controlled/S2_h180", -2.5)
    results["MultiHorizon"] = train_eval("MultiHorizon", DevelopmentalTemporalRetrieval, "outputs/data/seed20260922/controlled/S2_multi_horizon", -2.5)

    # Save metrics
    Path("results/model_improvement").mkdir(parents=True, exist_ok=True)
    with open("results/model_improvement/raw_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
