#!/usr/bin/env python3
"""Generate the final markdown tables for the synthetic benchmark (ICLR paper).

Parses the JSON outputs from runner.py and counterfactual_eval.py to produce
the report at baselines/report_synthea.md.
"""
import json
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "baselines" / "synthetic"
REPORT_PATH = REPO_ROOT / "baselines" / "report_synthea.md"

SCENARIOS = ["S0", "S1", "S2", "S3"]
MODELS = [
    ("count_lightgbm", "Count + LightGBM"),
    ("retain", "RETAIN"),
    ("ehr_bert", "EHR-BERT"),
    ("medbert", "Med-BERT"),
    ("behrt", "BEHRT"),
    ("cehrbert", "CEHR-BERT"),
    ("dtr_no_age", "DTR (Temporal)"),
    ("dtr_age_only", "DTR (Age)"),
    ("dtr_age_temporal", "DTR (Age×Temporal)"),
]

def format_metric(val: float | None) -> str:
    if val is None or isinstance(val, str):
        return "---"
    return f"{val:.3f}"

def format_rmse(val: float | None) -> str:
    if val is None or isinstance(val, str):
        return "---"
    return f"{val:.4f}"

def main():
    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR}")
        return

    # Load performance data (AUROC, AUPRC)
    perf_data = {}
    if (RESULTS_DIR / "all_results.json").exists():
        with (RESULTS_DIR / "all_results.json").open() as f:
            perf_data = json.load(f)

    # Note: DTR models save their results individually in model_dir/scenario/result.json
    # but also the runner scripts dumps them to all_results.json if implemented correctly.
    # We will fall back to reading individually if needed.
    
    # Load counterfactual data
    cf_data = {}
    for s in SCENARIOS:
        cf_file = RESULTS_DIR / f"cf_summary_{s}.json"
        if cf_file.exists():
            with cf_file.open() as f:
                cf_data[s] = json.load(f)

    # Gather data into a structured format
    table_data = {}
    for m_id, m_name in MODELS:
        table_data[m_id] = {s: {} for s in SCENARIOS}
        
        for s in SCENARIOS:
            # 1. Performance
            auroc = None
            key = f"{m_id}_{s}"
            # Check all_results
            if key in perf_data and "test_metrics" in perf_data[key]:
                auroc = perf_data[key]["test_metrics"].get("micro_auroc")
            else:
                # Check individual result file
                r_file = RESULTS_DIR / m_id / s / "result.json"
                if r_file.exists():
                    with r_file.open() as f:
                        rd = json.load(f)
                        if "test_metrics" in rd:
                            auroc = rd["test_metrics"].get("micro_auroc")
            
            table_data[m_id][s]["auroc"] = auroc
            
            # 2. Mechanism
            if s in cf_data and m_id in cf_data[s]:
                cf_s = cf_data[s][m_id]
                table_data[m_id][s]["cf_rmse_age"] = cf_s.get("cf_rmse_age")
                table_data[m_id][s]["surface_rmse"] = cf_s.get("surface_rmse")
                table_data[m_id][s]["mechanism_classification"] = cf_s.get("mechanism_classification")

    # Generate Markdown Report
    lines = [
        "# Synthetic Benchmark Results",
        "",
        "This document contains the automated evaluation results for all baselines",
        "across the Synthea S0–S3 scenarios.",
        "",
        "## 1. Predictive Performance (Test AUROC)",
        "",
        "| Model | S0 (No Interaction) | S1 (Age-only) | S2 (Age×Temporal) | S3 (Non-linear) |",
        "|-------|---------------------|---------------|-------------------|-----------------|",
    ]
    
    for m_id, m_name in MODELS:
        row = [m_name]
        for s in SCENARIOS:
            row.append(format_metric(table_data[m_id][s].get("auroc")))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## 2. Mechanism Recovery (S2: Age×Temporal)",
        "",
        "Evaluation of how accurately models capture the true data-generating mechanism",
        "(the temporal interaction surface λ(a) over age).",
        "",
        "| Model | CF-RMSE (Age) | Surface RMSE | Recovery Classification |",
        "|-------|---------------|--------------|--------------------------|",
    ])
    
    for m_id, m_name in MODELS:
        s2_data = table_data[m_id]["S2"]
        row = [
            m_name,
            format_rmse(s2_data.get("cf_rmse_age")),
            format_rmse(s2_data.get("surface_rmse")),
            s2_data.get("mechanism_classification", "---").replace("_", " "),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Summary",
        "",
        "The DTR (Age×Temporal) model should uniquely demonstrate `FUNCTIONAL RECOVERY`",
        "of the true mechanism, while maintaining top-tier predictive performance."
    ])
    
    with REPORT_PATH.open("w") as f:
        f.write("\n".join(lines))
        
    print(f"Report generated successfully at {REPORT_PATH}")

if __name__ == "__main__":
    main()
