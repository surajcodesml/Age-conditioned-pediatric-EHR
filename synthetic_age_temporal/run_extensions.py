#!/usr/bin/env python3
import subprocess
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, executable="/bin/bash")
    if res.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)

def main():
    base_dir = Path("/home/suraj/Git/Age-conditioned-pediatric-EHR/synthetic_age_temporal")
    data_dir = base_dir / "outputs/data/seed20260922/controlled"
    runs_dir = base_dir / "outputs/runs/controlled"
    res_dir = base_dir / "results/model_improvement"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Build Multi Horizon data
    if not (data_dir / "S2/labels_forecast.npz").exists():
        run_cmd(f"conda run -n ehr python build_multi_horizon.py --scenario-dir {data_dir}/S2 --out-dir {data_dir}/S2")

    # 2. Train baseline DTR
    for s in ["S2", "S5", "S6"]:
        d = runs_dir / f"{s}_DTR_baseline"
        if not (d / "metrics.json").exists():
            run_cmd(f"conda run -n ehr python train_dtr.py --age-temporal --scenario-dir {data_dir}/{s} --run-dir {d}")

    # 3. Train Content Persistence
    for s in ["S2", "S5", "S6"]:
        d = runs_dir / f"{s}_DTR_content_persistence"
        if not (d / "metrics.json").exists():
            run_cmd(f"conda run -n ehr python train_dtr.py --age-temporal --content-persistence --scenario-dir {data_dir}/{s} --run-dir {d}")

    # 4. Train Multi Query K=4
    for s in ["S2", "S5", "S6"]:
        d = runs_dir / f"{s}_DTR_multi_query_K4"
        if not (d / "metrics.json").exists():
            run_cmd(f"conda run -n ehr python train_dtr.py --age-temporal --multi-query-k 4 --scenario-dir {data_dir}/{s} --run-dir {d}")

    # 5. Train Multi Horizon
    d_sh = runs_dir / "S2_DTR_single_horizon"
    if not (d_sh / "metrics.json").exists():
        run_cmd(f"conda run -n ehr python train_multi_horizon.py --single-horizon --scenario-dir {data_dir}/S2 --run-dir {d_sh}")
    
    d_mh = runs_dir / "S2_DTR_multi_horizon"
    if not (d_mh / "metrics.json").exists():
        run_cmd(f"conda run -n ehr python train_multi_horizon.py --scenario-dir {data_dir}/S2 --run-dir {d_mh}")

    # Load and collect metrics
    def load_m(name):
        p = runs_dir / name / "metrics.json"
        if p.exists():
            with p.open() as f: return json.load(f)
        return None

    def get_row(name_base, model_name):
        s2 = load_m(f"S2_{name_base}")
        s5 = load_m(f"S5_{name_base}")
        s6 = load_m(f"S6_{name_base}")
        
        row = {"Model": model_name}
        if s2:
            row["S2 AUROC"] = s2["test"]["micro_auroc"]
            row["S2 AUPRC"] = s2["test"]["micro_auprc"]
            row["S2 shuffle ΔBCE"] = s2["ablations"]["delta_bce_shuffle_age"]
            row["S2 β=0 ΔBCE"] = s2["ablations"]["delta_bce_beta0"]
            row["S2 β_hat"] = s2["beta_hat"]
            row["parameter count"] = s2["n_params"]
        if s5:
            row["S5 AUROC"] = s5["test"]["micro_auroc"]
            row["S5 AUPRC"] = s5["test"]["micro_auprc"]
        if s6:
            row["S6 AUROC"] = s6["test"]["micro_auroc"]
            row["S6 AUPRC"] = s6["test"]["micro_auprc"]
        return row

    baseline = get_row("DTR_baseline", "Baseline DTR")
    content = get_row("DTR_content_persistence", "Content-persistence DTR")
    mquery = get_row("DTR_multi_query_K4", "Multi-query DTR")
    
    sh = load_m("S2_DTR_single_horizon")
    mh = load_m("S2_DTR_multi_horizon")
    
    mh_row = {"Model": "Multi-horizon DTR"}
    if mh:
        mh_row["multi-horizon mean AUPRC"] = np.mean([mh["test"][f"horizon_{h}"]["micro_auprc"] for h in [30,90,180,365]])
        mh_row["S2 β_hat"] = mh["beta_hat"]
    if sh:
        baseline["multi-horizon mean AUPRC"] = np.mean([sh["test"][f"horizon_{h}"]["micro_auprc"] for h in [30,90,180,365]])
        
    # Decisions
    content_supported = False
    if content.get("S5 AUPRC", 0) > baseline.get("S5 AUPRC", 0):
        content_supported = True
    content["selected?"] = content_supported
    content["reason"] = "Improved S5 AUPRC" if content_supported else "Failed to improve S5 AUPRC"

    mquery_supported = False
    if mquery.get("S6 AUPRC", 0) > baseline.get("S6 AUPRC", 0):
        mquery_supported = True
    mquery["selected?"] = mquery_supported
    mquery["reason"] = "Improved S6 AUPRC" if mquery_supported else "Failed to improve S6 AUPRC"

    mhorizon_supported = False
    if mh and sh and mh_row.get("multi-horizon mean AUPRC", 0) > baseline.get("multi-horizon mean AUPRC", 0):
        mhorizon_supported = True
    mh_row["selected?"] = mhorizon_supported
    mh_row["reason"] = "Improved mean forecasting AUPRC" if mhorizon_supported else "Failed to improve forecasting AUPRC"

    baseline["selected?"] = True
    baseline["reason"] = "Baseline"

    print("Content Supported:", content_supported)
    print("Multi-query Supported:", mquery_supported)
    print("Multi-horizon Supported:", mhorizon_supported)

    # Train combined model
    flags = []
    if content_supported: flags.append("--content-persistence")
    if mquery_supported: flags.append("--multi-query-k 4")
    
    c_runs = ["S0", "S1", "S2", "S3", "S5", "S6"]
    for s in c_runs:
        d = runs_dir / f"{s}_DTR_combined"
        if not (d / "metrics.json").exists():
            run_cmd(f"conda run -n ehr python train_dtr.py --age-temporal {' '.join(flags)} --scenario-dir {data_dir}/{s} --run-dir {d}")

    comb = get_row("DTR_combined", "Final supported combination")
    comb["selected?"] = True
    comb["reason"] = "Final combination"
    
    rows = [baseline, content, mquery, mh_row, comb]
    df = pd.DataFrame(rows)
    cols = ["Model", "S2 AUROC", "S2 AUPRC", "S2 shuffle ΔBCE", "S2 β=0 ΔBCE", "S2 β_hat", 
            "S5 AUROC", "S5 AUPRC", "S6 AUROC", "S6 AUPRC", "multi-horizon mean AUPRC", 
            "parameter count", "selected?", "reason"]
    df = df.reindex(columns=cols)
    df.to_csv(res_dir / "model_selection.csv", index=False)
    
    print("Completed model selection. CSV written.")
    
if __name__ == "__main__":
    main()
