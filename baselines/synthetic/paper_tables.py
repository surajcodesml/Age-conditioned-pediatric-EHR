#!/usr/bin/env python3
"""Generate markdown tables for the synthetic baseline benchmark (ICLR paper).

Parses JSON outputs from runner.py and counterfactual_eval.py.

Prepared outputs (call after training; not generated in this implementation task):
  1. Consolidated S0–S5 baseline results table
  2. S0–S3 age×lag mechanism comparison
  3. Hook pointing at S5 persistence-curve figure (see s5_figures.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.synthetic.data_adapter import BENCHMARK_SCENARIOS, CORE_SCENARIOS
from baselines.synthetic.result_schema import consolidate_records, from_train_and_cf

RESULTS_DIR = REPO_ROOT / "results" / "baselines" / "synthetic"
REPORT_PATH = REPO_ROOT / "baselines" / "report_synthea.md"

SCENARIOS = list(BENCHMARK_SCENARIOS)  # S0–S3 + S5
CORE = list(CORE_SCENARIOS)

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


def format_metric(val) -> str:
    if val is None or isinstance(val, str):
        return "---"
    return f"{val:.3f}"


def format_rmse(val) -> str:
    if val is None or isinstance(val, str):
        return "---"
    return f"{val:.4f}"


def format_bool(val) -> str:
    if val is None:
        return "---"
    return "yes" if val else "no"


def _load_perf_and_cf():
    perf_data = {}
    if (RESULTS_DIR / "all_results.json").exists():
        with (RESULTS_DIR / "all_results.json").open() as f:
            perf_data = json.load(f)

    cf_data = {}
    for s in SCENARIOS:
        cf_file = RESULTS_DIR / f"cf_summary_{s}.json"
        if cf_file.exists():
            with cf_file.open() as f:
                cf_data[s] = json.load(f)
    return perf_data, cf_data


def gather_table_data(perf_data, cf_data):
    table_data = {}
    flat_records = []
    for m_id, m_name in MODELS:
        table_data[m_id] = {s: {} for s in SCENARIOS}
        for s in SCENARIOS:
            auroc = auprc = bce = None
            key = f"{m_id}_{s}"
            if key in perf_data and "test_metrics" in perf_data[key]:
                tm = perf_data[key]["test_metrics"]
                auroc = tm.get("micro_auroc")
                auprc = tm.get("micro_auprc")
                bce = tm.get("bce")
            else:
                r_file = RESULTS_DIR / m_id / s / "result.json"
                if r_file.exists():
                    with r_file.open() as f:
                        rd = json.load(f)
                    if "test_metrics" in rd:
                        auroc = rd["test_metrics"].get("micro_auroc")
                        auprc = rd["test_metrics"].get("micro_auprc")
                        bce = rd["test_metrics"].get("bce")

            table_data[m_id][s]["auroc"] = auroc
            table_data[m_id][s]["auprc"] = auprc
            table_data[m_id][s]["bce"] = bce

            cf_s = {}
            if s in cf_data and m_id in cf_data[s]:
                cf_s = cf_data[s][m_id]
                table_data[m_id][s]["cf_rmse_age"] = cf_s.get("cf_rmse_age")
                table_data[m_id][s]["cf_rmse_lag"] = cf_s.get("cf_rmse_lag")
                table_data[m_id][s]["surface_rmse"] = cf_s.get("surface_rmse")
                table_data[m_id][s]["mechanism_classification"] = cf_s.get(
                    "mechanism_classification"
                )
                for fld in (
                    "S5_Surface_RMSE_acute",
                    "S5_Surface_RMSE_intermediate",
                    "S5_Surface_RMSE_chronic",
                    "S5_Surface_RMSE_mean",
                    "persistence_order_correct",
                ):
                    table_data[m_id][s][fld] = cf_s.get(fld)

            flat_records.append(
                from_train_and_cf(
                    scenario=s,
                    model=m_id,
                    test_metrics={
                        "micro_auroc": auroc,
                        "micro_auprc": auprc,
                        "bce": bce,
                    },
                    cf_report=cf_s or None,
                )
            )
    return table_data, consolidate_records(flat_records)


def build_consolidated_s0_s5_table(table_data) -> list[str]:
    """(1) One consolidated S0–S5 baseline results table."""
    lines = [
        "## 1. Consolidated S0–S5 baseline results",
        "",
        "Core age×temporal scenarios (S0–S3) plus heterogeneous persistence (S5).",
        "",
        "| Model | Metric | S0 | S1 | S2 | S3 | S5 |",
        "|-------|--------|----|----|----|----|----|",
    ]
    for m_id, m_name in MODELS:
        for metric, key, fmt in (
            ("AUROC", "auroc", format_metric),
            ("AUPRC", "auprc", format_metric),
            ("BCE", "bce", format_metric),
            ("Surface RMSE", "surface_rmse", format_rmse),
        ):
            row = [m_name if metric == "AUROC" else "", metric]
            for s in SCENARIOS:
                row.append(fmt(table_data[m_id][s].get(key)))
            lines.append("| " + " | ".join(row) + " |")
    return lines


def build_s0_s3_mechanism_table(table_data) -> list[str]:
    """(2) S0–S3 age×lag mechanism comparison."""
    lines = [
        "## 2. S0–S3 age × lag mechanism comparison",
        "",
        "Evaluation of how accurately models capture the true age×temporal mechanism.",
        "S0–S3 mechanism-recovery thresholds are unchanged "
        "(Surface RMSE < 0.10 functional, < 0.25 partial).",
        "",
        "| Model | S2 CF-RMSE (Age) | S2 Surface RMSE | S2 Classification |"
        " S3 Surface RMSE | S3 Classification |",
        "|-------|------------------|-----------------|-------------------|"
        "-----------------|-------------------|",
    ]
    for m_id, m_name in MODELS:
        s2 = table_data[m_id]["S2"]
        s3 = table_data[m_id]["S3"]
        row = [
            m_name,
            format_rmse(s2.get("cf_rmse_age")),
            format_rmse(s2.get("surface_rmse")),
            (s2.get("mechanism_classification") or "---").replace("_", " "),
            format_rmse(s3.get("surface_rmse")),
            (s3.get("mechanism_classification") or "---").replace("_", " "),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_s5_persistence_table(table_data) -> list[str]:
    """S5 heterogeneous-persistence metrics (+ figure hook)."""
    lines = [
        "## 3. S5 heterogeneous persistence",
        "",
        "Can a model learn that different kinds of historical clinical information",
        "persist for different lengths of time while also recovering the",
        "developmental age×temporal relationship?",
        "",
        "| Model | AUROC | Surface RMSE | Acute | Intermediate | Chronic |"
        " Mean | Order correct | Classification |",
        "|-------|-------|--------------|-------|--------------|---------|"
        "------|---------------|----------------|",
    ]
    for m_id, m_name in MODELS:
        s5 = table_data[m_id]["S5"]
        row = [
            m_name,
            format_metric(s5.get("auroc")),
            format_rmse(s5.get("surface_rmse")),
            format_rmse(s5.get("S5_Surface_RMSE_acute")),
            format_rmse(s5.get("S5_Surface_RMSE_intermediate")),
            format_rmse(s5.get("S5_Surface_RMSE_chronic")),
            format_rmse(s5.get("S5_Surface_RMSE_mean")),
            format_bool(s5.get("persistence_order_correct")),
            (s5.get("mechanism_classification") or "---").replace("_", " "),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "### S5 persistence-curve figure (future)",
        "",
        "True vs predicted persistence curves for acute / intermediate / chronic",
        "history are produced by `baselines.synthetic.s5_figures` after CF eval:",
        "",
        "```text",
        "results/baselines/synthetic/figures/fig_s5_persistence_curves.{png,svg}",
        "```",
        "",
        "Do not generate that figure from this table script during implementation-only runs.",
    ])
    return lines


def main():
    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR}")
        return

    perf_data, cf_data = _load_perf_and_cf()
    table_data, flat_records = gather_table_data(perf_data, cf_data)

    # Persist consolidated JSON for downstream paper tooling
    consolidated_path = RESULTS_DIR / "consolidated_s0_s5_records.json"
    with consolidated_path.open("w") as f:
        json.dump(flat_records, f, indent=2, default=str)

    lines = [
        "# Synthetic Benchmark Results",
        "",
        "Automated evaluation for all baselines across S0–S3 (core age × temporal)",
        "and S5 (heterogeneous temporal persistence). S6 / multi-horizon are excluded.",
        "",
    ]
    lines.extend(build_consolidated_s0_s5_table(table_data))
    lines.append("")
    lines.extend(build_s0_s3_mechanism_table(table_data))
    lines.append("")
    lines.extend(build_s5_persistence_table(table_data))
    lines.extend([
        "",
        "## Summary",
        "",
        "- S0–S3: mechanism classification uses Functional / Partial / None thresholds.",
        "- S5: classification uses HETEROGENEOUS_PERSISTENCE_RECOVERED /",
        "  PARTIAL_HETEROGENEOUS_PERSISTENCE_RECOVERY / NO_HETEROGENEOUS_PERSISTENCE_RECOVERY",
        "  (mean Surface RMSE thresholds + correct persistence ordering for full recovery).",
    ])

    with REPORT_PATH.open("w") as f:
        f.write("\n".join(lines))

    print(f"Report generated at {REPORT_PATH}")
    print(f"Consolidated records: {consolidated_path}")


if __name__ == "__main__":
    main()
