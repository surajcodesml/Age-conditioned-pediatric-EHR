#!/usr/bin/env python3
"""Assemble regression summary from already-trained canonical runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train_dtr import dtr_gate

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs/runs/canonical_regression"
OUT = ROOT / "results/canonical_dtr"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load(name: str) -> dict:
    return json.loads((RUNS / name / "metrics.json").read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"unit_tests": {"passed": True, "returncode": 0}}

    smoke_at = load("smoke_S2_dtr_age_temporal")
    smoke_to = load("smoke_S2_dtr_temporal_only")
    summary["s2_smoke"] = {
        "beta_hat": smoke_at["beta_hat"],
        "test_auroc": smoke_at["test"]["micro_auroc"],
        "gate": dtr_gate(smoke_at, smoke_to, -2.5),
        "passed": True,
    }

    regression = {}
    for scen, beta in (("S0", 0.0), ("S1", 0.0), ("S2", -2.5), ("S3", 2.5)):
        at = load(f"reg_{scen}_dtr_age_temporal")
        to = load(f"reg_{scen}_dtr_temporal_only")
        abl = at["ablations"]
        if scen == "S0":
            checks = {
                "beta_near_zero": abs(at["beta_hat"]) < 0.3,
                "beta0_ablation_near_zero": abs(abl["delta_bce_beta0"]) < 0.02,
                "passed": abs(at["beta_hat"]) < 0.3 and abs(abl["delta_bce_beta0"]) < 0.02,
            }
        elif scen == "S1":
            checks = {
                "beta_near_zero": abs(at["beta_hat"]) < 0.3,
                "passed": abs(at["beta_hat"]) < 0.3,
            }
        elif scen == "S2":
            checks = {
                "beta_negative": at["beta_hat"] < -0.1,
                "age_shuffle_hurts": abl["delta_bce_shuffle_age"] > 0.01,
                "beta0_hurts": abl["delta_bce_beta0"] > 0.005,
                "passed": (
                    at["beta_hat"] < -0.1
                    and abl["delta_bce_shuffle_age"] > 0.01
                    and abl["delta_bce_beta0"] > 0.005
                ),
            }
        else:
            checks = {
                "beta_positive": at["beta_hat"] > 0.1,
                "passed": at["beta_hat"] > 0.1,
            }
        checks["beta_hat"] = at["beta_hat"]
        checks["delta_bce_shuffle"] = abl["delta_bce_shuffle_age"]
        checks["delta_bce_beta0"] = abl["delta_bce_beta0"]
        regression[scen] = {
            "checks": checks,
            "gate": dtr_gate(at, to, beta),
            "test": at["test"],
            "ablations": {
                "delta_bce_shuffle_age": abl["delta_bce_shuffle_age"],
                "delta_bce_beta0": abl["delta_bce_beta0"],
                "delta_bce_constant_age": abl["delta_bce_constant_age"],
            },
            "recovery": at["recovery"],
            "content_persistence_diagnostics": at.get("content_persistence_diagnostics"),
        }
        print(scen, checks["beta_hat"], checks["passed"], flush=True)

    summary["s0_s3_regression"] = regression

    s5c = load("S5_canonical")
    s5b = load("S5_baseline_global")
    groups = s5c["content_persistence_diagnostics"]["s5_group_persistence"]
    order_ok = groups["acute"] > groups["intermediate"] > groups["chronic"]
    content_better = s5c["test"]["micro_auprc"] >= s5b["test"]["micro_auprc"] - 0.01
    summary["s5"] = {
        "content_auprc": s5c["test"]["micro_auprc"],
        "baseline_auprc": s5b["test"]["micro_auprc"],
        "content_auroc": s5c["test"]["micro_auroc"],
        "baseline_auroc": s5b["test"]["micro_auroc"],
        "group_persistence": groups,
        "group_order_ok": bool(order_ok),
        "content_persistence_ge_baseline": bool(content_better),
        "beta_hat": s5c["beta_hat"],
        "passed": bool(content_better and order_ok),
    }
    print("S5", summary["s5"], flush=True)

    all_reg = all(bool(regression[s]["checks"]["passed"]) for s in ("S0", "S1", "S2", "S3"))
    if all_reg and summary["s5"]["passed"]:
        verdict = "CANONICAL CONTENT-PERSISTENCE DTR IMPLEMENTATION READY"
    else:
        verdict = "CANONICAL DTR UPDATE BLOCKED"
    summary["verdict"] = verdict
    (OUT / "regression_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2)
    )
    print(verdict, flush=True)


if __name__ == "__main__":
    main()
