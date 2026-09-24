import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch

os.chdir('/home/suraj/Git/Age-conditioned-pediatric-EHR/synthetic_age_temporal')
sys.path.append('.')

from model_dtr import DevelopmentalTemporalRetrieval, DTRContentPersistence, DTRMultiQuery
from dataset_dtr import make_dtr_loaders
from train_dtr import _move

def compile_report():
    raw_path = Path("results/model_improvement/raw_metrics.json")
    if not raw_path.exists():
        print("No raw metrics")
        return
    with open(raw_path) as f:
        metrics = json.load(f)

    # Compile table
    rows = []
    
    def get_row(model_name, label):
        row = {"Model": label}
        # S2 metrics
        m_s2 = metrics.get(f"{model_name}_S2")
        if not m_s2 and model_name == "MultiHorizon":
            # Baseline S2 metrics for multi horizon
            m_s2 = metrics.get("Baseline_S2")
            
        if m_s2:
            row["S2 AUROC"] = m_s2["test"]["micro_auroc"]
            row["S2 AUPRC"] = m_s2["test"]["micro_auprc"]
            row["S2 shuffle \u0394BCE"] = m_s2["ablations"]["delta_bce_shuffle_age"]
            row["S2 \u03b2=0 \u0394BCE"] = m_s2["ablations"]["delta_bce_beta0"]
            row["S2 \u03b2_hat"] = m_s2["beta_hat"]
            row["parameter count"] = m_s2["n_params"]
        else:
            row["S2 AUROC"] = ""
            row["S2 AUPRC"] = ""
            row["S2 shuffle \u0394BCE"] = ""
            row["S2 \u03b2=0 \u0394BCE"] = ""
            row["S2 \u03b2_hat"] = ""
            row["parameter count"] = ""
            
        # S5
        m_s5 = metrics.get(f"{model_name}_S5")
        if m_s5:
            row["S5 AUROC/AUPRC"] = f"{m_s5['test']['micro_auroc']:.3f}/{m_s5['test']['micro_auprc']:.3f}"
        else:
            row["S5 AUROC/AUPRC"] = ""
            
        # S6
        m_s6 = metrics.get(f"{model_name}_S6")
        if m_s6:
            row["S6 AUROC/AUPRC"] = f"{m_s6['test']['micro_auroc']:.3f}/{m_s6['test']['micro_auprc']:.3f}"
        else:
            row["S6 AUROC/AUPRC"] = ""
            
        # Multi Horizon
        if model_name == "Baseline":
            m_mh = metrics.get("SingleHorizon_180")
            if m_mh:
                row["multi-horizon mean AUPRC"] = f"Single 180: {m_mh['test']['micro_auprc']:.3f}"
            else:
                row["multi-horizon mean AUPRC"] = ""
        elif model_name == "MultiHorizon":
            m_mh = metrics.get("MultiHorizon")
            if m_mh:
                row["multi-horizon mean AUPRC"] = m_mh["test"]["micro_auprc"]
            else:
                row["multi-horizon mean AUPRC"] = ""
        else:
            row["multi-horizon mean AUPRC"] = ""

        row["selected?"] = ""
        row["reason"] = ""
        return row
        
    rows.append(get_row("Baseline", "Baseline DTR"))
    rows.append(get_row("Content", "Content-persistence DTR"))
    rows.append(get_row("MultiQuery", "Multi-query DTR"))
    rows.append(get_row("MultiHorizon", "Multi-horizon DTR"))
    
    # Decisions
    df = pd.DataFrame(rows)
    
    # Analyze S5 Content vs Baseline
    m_base_s5 = metrics.get("Baseline_S5")
    m_cont_s5 = metrics.get("Content_S5")
    if m_cont_s5 and m_base_s5 and m_cont_s5["test"]["micro_auprc"] > m_base_s5["test"]["micro_auprc"] + 0.005:
        df.loc[1, "selected?"] = "Yes"
        df.loc[1, "reason"] = "CONTENT-DEPENDENT PERSISTENCE SUPPORTED"
    else:
        df.loc[1, "selected?"] = "No"
        df.loc[1, "reason"] = "CONTENT-DEPENDENT PERSISTENCE NOT JUSTIFIED"
        
    # Analyze S6 MultiQuery vs Baseline
    m_base_s6 = metrics.get("Baseline_S6")
    m_mq_s6 = metrics.get("MultiQuery_S6")
    if m_mq_s6 and m_base_s6 and m_mq_s6["test"]["micro_auprc"] > m_base_s6["test"]["micro_auprc"] + 0.005:
        df.loc[2, "selected?"] = "Yes"
        df.loc[2, "reason"] = "MULTI-QUERY RETRIEVAL SUPPORTED FOR PRETRAINING"
    else:
        df.loc[2, "selected?"] = "No"
        df.loc[2, "reason"] = "MULTI-QUERY RETRIEVAL NOT JUSTIFIED"
        
    # Analyze Multi Horizon
    m_sh = metrics.get("SingleHorizon_180")
    m_mh = metrics.get("MultiHorizon")
    if m_mh and m_sh and m_mh["test"]["micro_auprc"] > m_sh["test"]["micro_auprc"] + 0.005:
        df.loc[3, "selected?"] = "Yes"
        df.loc[3, "reason"] = "MULTI-HORIZON PRETRAINING SUPPORTED"
    else:
        df.loc[3, "selected?"] = "No"
        df.loc[3, "reason"] = "NEXT-VISIT/SINGLE-HORIZON OBJECTIVE SUFFICIENT"
        
    final_components = ["Baseline DTR"]
    if df.loc[1, "selected?"] == "Yes":
        final_components.append("content-dependent persistence")
    if df.loc[2, "selected?"] == "Yes":
        final_components.append("multi-query retrieval")
    if df.loc[3, "selected?"] == "Yes":
        final_components.append("multi-horizon pretraining")
        
    final_reason = " + ".join(final_components)
    
    rows.append({
        "Model": "Final supported combination",
        "S2 AUROC": "", "S2 AUPRC": "", "S2 shuffle \u0394BCE": "", "S2 \u03b2=0 \u0394BCE": "", 
        "S2 \u03b2_hat": "", "S5 AUROC/AUPRC": "", "S6 AUROC/AUPRC": "", 
        "multi-horizon mean AUPRC": "", "parameter count": "",
        "selected?": "Final",
        "reason": final_reason
    })
    
    df = pd.DataFrame(rows)
    df.to_csv("results/model_improvement/model_selection.csv", index=False)
    print("Saved model_selection.csv")
    
    with open("synthetic_age_temporal/report.md", "a") as f:
        f.write("\n\n## Model improvement experiments\n")
        f.write("### Experiment 1: Content-dependent temporal persistence\n")
        f.write(f"Result: {df.loc[1, 'reason']}\n\n")
        f.write("### Experiment 2: Multi-query / target-aware content retrieval\n")
        f.write(f"Result: {df.loc[2, 'reason']}\n\n")
        f.write("### Experiment 3: Multi-horizon temporal supervision\n")
        f.write(f"Result: {df.loc[3, 'reason']}\n\n")
        f.write("RECOMMENDED FOR MIMIC:\n")
        f.write(f"{final_reason}\n")
        
        rejected = []
        if df.loc[1, "selected?"] == "No": rejected.append("Content-dependent persistence")
        if df.loc[2, "selected?"] == "No": rejected.append("Multi-query retrieval")
        if df.loc[3, "selected?"] == "No": rejected.append("Multi-horizon supervision")
        if rejected:
            f.write("REJECTED:\n" + "\n".join(rejected) + "\n")

    print("Figures generation code goes here...")

if __name__ == "__main__":
    compile_report()
