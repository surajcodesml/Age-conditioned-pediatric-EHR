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

from dataset_dtr import make_dtr_loaders
from train_dtr import load_dtr_model, evaluate, _move
from evaluate import classification_metrics
from config import all_signal_codes

def generate_figures():
    Path("figures").mkdir(exist_ok=True)
    
    # We will load metrics directly from the directories since run_extensions.py doesn't produce raw_metrics.json
    runs_dir = Path("outputs/runs/controlled")

    # Fig I: Heterogeneous persistence recovery
    # Load Content_S5
    run_dir = runs_dir / "S5_DTR_content_persistence"
    if run_dir.exists() and (run_dir / "model.pt").exists():
        model, ckpt = load_dtr_model(run_dir, torch.device("cpu"))
        vocab = ckpt["vocab_stoi"]
        
        groups = {
            "Acute": ["SYN_SIGNAL_A", "SYN_SIGNAL_B", "SYN_SIGNAL_C", "SYN_SIGNAL_D"],
            "Intermediate": ["SYN_SIGNAL_E", "SYN_SIGNAL_F", "SYN_SIGNAL_G", "SYN_SIGNAL_H"],
            "Chronic": ["SYN_SIGNAL_I", "SYN_SIGNAL_J", "SYN_SIGNAL_K", "SYN_SIGNAL_L"]
        }
        
        plt.figure(figsize=(8, 5))
        ages = np.linspace(0, 18, 50)
        z_ages = (ages - 9.0) / 9.0
        
        for name, codes in groups.items():
            ids = [vocab.get(c, 0) for c in codes]
            ids_t = torch.tensor(ids).unsqueeze(0).unsqueeze(-1)
            mask_t = torch.ones_like(ids_t, dtype=torch.bool)
            with torch.no_grad():
                v = model.encode_encounters(ids_t, mask_t)
                theta = model.persistence_projection(v).squeeze(-1)
                theta_mean = theta.mean().item()
            
            beta = float(model.beta.detach())
            lam = np.log1p(np.exp(theta_mean + beta * z_ages))
            if name == "Acute": true_th = 1.0
            elif name == "Intermediate": true_th = 0.0
            else: true_th = -1.0
            true_lam = np.log1p(np.exp(true_th - 2.5 * z_ages))
            
            p = plt.plot(ages, lam, label=f"Learned ({name})")
            plt.plot(ages, true_lam, '--', color=p[0].get_color(), label=f"True ({name})")
            
        plt.xlabel("Age")
        plt.ylabel("Lambda")
        plt.legend()
        plt.title("Heterogeneous Persistence Recovery")
        plt.savefig("figures/fig_s5_persistence_curves.png")
        plt.savefig("figures/fig_s5_persistence_curves.svg")
        plt.close()

    # Fig J: Performance gain S5
    b_s5_path = runs_dir / "S5_DTR_baseline" / "metrics.json"
    c_s5_path = runs_dir / "S5_DTR_content_persistence" / "metrics.json"
    if b_s5_path.exists() and c_s5_path.exists():
        with open(b_s5_path) as f: b_s5 = json.load(f)["test"]
        with open(c_s5_path) as f: c_s5 = json.load(f)["test"]
        df_p = pd.DataFrame({
            "Model": ["Baseline", "Baseline", "Content-Persistence", "Content-Persistence"],
            "Metric": ["AUROC", "AUPRC", "AUROC", "AUPRC"],
            "Score": [b_s5["micro_auroc"], b_s5["micro_auprc"], c_s5["micro_auroc"], c_s5["micro_auprc"]]
        })
        plt.figure(figsize=(6, 5))
        sns.barplot(data=df_p, x="Metric", y="Score", hue="Model")
        plt.ylim(0.5, 1.0)
        plt.title("Performance on S5 (Heterogeneous Persistence)")
        plt.savefig("figures/fig_s5_performance.png")
        plt.savefig("figures/fig_s5_performance.svg")
        plt.close()

    # Fig K: Multi-query specialization heatmap
    run_dir = runs_dir / "S6_DTR_multi_query_K4"
    if run_dir.exists() and (run_dir / "model.pt").exists():
        model, ckpt = load_dtr_model(run_dir, torch.device("cpu"))
        vocab = ckpt["vocab_stoi"]
        groups = {
            "Group A": ["SYN_SIGNAL_A", "SYN_SIGNAL_B", "SYN_SIGNAL_C"],
            "Group B": ["SYN_SIGNAL_D", "SYN_SIGNAL_E", "SYN_SIGNAL_F"],
            "Group C": ["SYN_SIGNAL_G", "SYN_SIGNAL_H", "SYN_SIGNAL_I"],
            "Group D": ["SYN_SIGNAL_J", "SYN_SIGNAL_K", "SYN_SIGNAL_L"],
        }
        heatmap = np.zeros((4, 4))
        for i, (name, codes) in enumerate(groups.items()):
            ids = [vocab.get(c, 0) for c in codes]
            ids_t = torch.tensor(ids).unsqueeze(0).unsqueeze(-1)
            mask_t = torch.ones_like(ids_t, dtype=torch.bool)
            with torch.no_grad():
                v = model.encode_encounters(ids_t, mask_t)
                k = model.W_k(v)
                u = torch.matmul(k, model.q.T) / np.sqrt(k.size(-1))
                heatmap[i] = u.mean(dim=(0, 1)).numpy()

        plt.figure(figsize=(6, 5))
        sns.heatmap(heatmap.T, annot=True, xticklabels=list(groups.keys()), yticklabels=[f"Query {j}" for j in range(4)])
        plt.xlabel("Signal Group")
        plt.ylabel("Retrieval Query")
        plt.title("Multi-Query Specialization")
        plt.savefig("figures/fig_s6_multi_query_heatmap.png")
        plt.savefig("figures/fig_s6_multi_query_heatmap.svg")
        plt.close()

    # Fig L: Multi-query performance S6
    b_s6_path = runs_dir / "S6_DTR_baseline" / "metrics.json"
    mq_s6_path = runs_dir / "S6_DTR_multi_query_K4" / "metrics.json"
    if b_s6_path.exists() and mq_s6_path.exists():
        with open(b_s6_path) as f: b_s6 = json.load(f)["test"]
        with open(mq_s6_path) as f: mq_s6 = json.load(f)["test"]
        df_p = pd.DataFrame({
            "Model": ["Single-Query", "Single-Query", "Multi-Query (K=4)", "Multi-Query (K=4)"],
            "Metric": ["AUROC", "AUPRC", "AUROC", "AUPRC"],
            "Score": [b_s6["micro_auroc"], b_s6["micro_auprc"], mq_s6["micro_auroc"], mq_s6["micro_auprc"]]
        })
        plt.figure(figsize=(6, 5))
        sns.barplot(data=df_p, x="Metric", y="Score", hue="Model")
        plt.ylim(0.5, 1.0)
        plt.title("Performance on S6 (Target-Selective History)")
        plt.savefig("figures/fig_l_multi_query_perf.png")
        plt.savefig("figures/fig_l_multi_query_perf.svg")
        plt.close()

    # Fig M: Multi-horizon performance
    mh_path = runs_dir / "S2_DTR_multi_horizon" / "metrics.json"
    sh_path = runs_dir / "S2_DTR_single_horizon" / "metrics.json"
    if mh_path.exists() and sh_path.exists():
        with open(mh_path) as f: mh_metrics = json.load(f)["test"]
        with open(sh_path) as f: sh_metrics = json.load(f)["test"]
        
        horizons = [30, 90, 180, 365]
        mh_auprc = [mh_metrics[f"horizon_{h}"]["micro_auprc"] for h in horizons]
        sh_auprc = [sh_metrics[f"horizon_{h}"]["micro_auprc"] for h in horizons]
            
        df_mh = pd.DataFrame({
            "Horizon": horizons * 2,
            "Model": ["Single-Horizon (180)"] * 4 + ["Multi-Horizon"] * 4,
            "AUPRC": sh_auprc + mh_auprc
        })
        plt.figure(figsize=(6, 5))
        sns.barplot(data=df_mh, x="Horizon", y="AUPRC", hue="Model")
        plt.title("Multi-Horizon vs Single-Horizon Performance")
        plt.savefig("figures/fig_m_multi_horizon_perf.png")
        plt.savefig("figures/fig_m_multi_horizon_perf.svg")
        plt.close()

    # Append to report.md
    df = pd.read_csv("results/model_improvement/model_selection.csv")
    with open("report.md", "a") as f:
        f.write("\n\n## Model improvement experiments\n")
        f.write("### Experiment 1: Content-dependent temporal persistence\n")
        f.write(f"Result: {df.loc[1, 'reason']}\n\n")
        f.write("### Experiment 2: Multi-query / target-aware content retrieval\n")
        f.write(f"Result: {df.loc[2, 'reason']}\n\n")
        f.write("### Experiment 3: Multi-horizon temporal supervision\n")
        f.write(f"Result: {df.loc[3, 'reason']}\n\n")
        f.write("RECOMMENDED FOR MIMIC:\n")
        f.write(f"{df.loc[4, 'reason']}\n")
        
        rejected = []
        if not df.loc[1, "selected?"]: rejected.append("Content-dependent persistence")
        if not df.loc[2, "selected?"]: rejected.append("Multi-query retrieval")
        if not df.loc[3, "selected?"]: rejected.append("Multi-horizon supervision")
        if rejected:
            f.write("\nREJECTED:\n" + "\n".join(rejected) + "\n")

if __name__ == "__main__":
    generate_figures()
