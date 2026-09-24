#!/usr/bin/env python3
"""Lightweight regression validation for canonical Content-Persistence DTR.

Runs: unit tests → S2 smoke → S0–S3 regression → S5 group-ordering check.
Does NOT launch multi-seed or expensive sweeps.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DATA_SEED, DEFAULT_OUTPUT_DIR
from train_dtr import dtr_gate, s5_group_mean_persistence, train_dtr

DATA = DEFAULT_OUTPUT_DIR / "data" / f"seed{DATA_SEED}" / "controlled"
RUNS = ROOT / "outputs" / "runs" / "canonical_regression"
OUT = ROOT / "results" / "canonical_dtr"


def run_unit_tests() -> dict:
    print("=== Unit tests ===", flush=True)
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_dtr.py")],
        cwd=str(ROOT),
    )
    return {"passed": r.returncode == 0, "returncode": r.returncode}


def train_pair(scenario: str, beta: float, epochs: int, tag: str) -> dict:
    sdir = DATA / scenario
    pair = {}
    for age_temporal, arm in ((True, "dtr_age_temporal"), (False, "dtr_temporal_only")):
        rdir = RUNS / f"{tag}_{scenario}_{arm}"
        pair[arm] = train_dtr(
            age_temporal=age_temporal,
            scenario_dir=sdir,
            run_dir=rdir,
            beta_true=beta,
            aggregation="raw_additive",
            content_persistence=True,
            interaction_only=True,
            max_epochs=epochs,
            patience=max(5, epochs // 3),
            batch_size=64,
            d_model=64,
            lr=1e-3,
            seed=0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    gate = dtr_gate(pair["dtr_age_temporal"], pair["dtr_temporal_only"], beta)
    return {"pair": pair, "gate": gate}


def check_scenario_expectations(scenario: str, result: dict) -> dict:
    at = result["pair"]["dtr_age_temporal"]
    abl = at["ablations"]
    beta_hat = at["beta_hat"]
    checks = {}
    if scenario == "S0":
        checks["beta_near_zero"] = abs(beta_hat) < 0.3
        checks["beta0_ablation_near_zero"] = abs(abl["delta_bce_beta0"]) < 0.02
        checks["passed"] = checks["beta_near_zero"] and checks["beta0_ablation_near_zero"]
    elif scenario == "S1":
        # Ordinary age effect allowed; age×time interaction remains inert
        checks["beta_near_zero"] = abs(beta_hat) < 0.3
        checks["beta0_ablation_near_zero"] = abs(abl["delta_bce_beta0"]) < 0.05
        checks["passed"] = checks["beta_near_zero"]
    elif scenario == "S2":
        checks["beta_negative"] = beta_hat < -0.1
        checks["age_shuffle_hurts"] = abl["delta_bce_shuffle_age"] > 0.01
        checks["beta0_hurts"] = abl["delta_bce_beta0"] > 0.005
        checks["passed"] = all(
            checks[k] for k in ("beta_negative", "age_shuffle_hurts", "beta0_hurts")
        )
    elif scenario == "S3":
        checks["beta_positive"] = beta_hat > 0.1
        checks["passed"] = checks["beta_positive"]
    else:
        checks["passed"] = True
    checks["beta_hat"] = beta_hat
    checks["delta_bce_shuffle"] = abl["delta_bce_shuffle_age"]
    checks["delta_bce_beta0"] = abl["delta_bce_beta0"]
    return checks


def s5_compatibility() -> dict:
    print("=== S5 compatibility ===", flush=True)
    rdir = RUNS / "S5_canonical"
    at = train_dtr(
        age_temporal=True,
        scenario_dir=DATA / "S5",
        run_dir=rdir,
        beta_true=-2.5,
        aggregation="raw_additive",
        content_persistence=True,
        interaction_only=True,
        max_epochs=25,
        patience=8,
        seed=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Baseline without content persistence for comparison
    base = train_dtr(
        age_temporal=True,
        scenario_dir=DATA / "S5",
        run_dir=RUNS / "S5_baseline_global",
        beta_true=-2.5,
        aggregation="raw_additive",
        content_persistence=False,
        interaction_only=True,
        max_epochs=25,
        patience=8,
        seed=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    groups = at["content_persistence_diagnostics"].get("s5_group_persistence", {})
    # Expected ordering: acute (fast decay / high λ offset) > intermediate > chronic
    # Ground truth θ: acute=1, intermediate=0, chronic=-1
    # Learned persistence offset should follow same order if recovery works.
    order_ok = False
    if groups:
        order_ok = groups.get("acute", 0) > groups.get("intermediate", 0) > groups.get(
            "chronic", 0
        )
    content_better = at["test"]["micro_auprc"] >= base["test"]["micro_auprc"] - 0.01
    return {
        "content_auprc": at["test"]["micro_auprc"],
        "baseline_auprc": base["test"]["micro_auprc"],
        "content_auroc": at["test"]["micro_auroc"],
        "baseline_auroc": base["test"]["micro_auroc"],
        "group_persistence": groups,
        "group_order_ok": order_ok,
        "content_persistence_ge_baseline": content_better,
        "beta_hat": at["beta_hat"],
        "passed": bool(content_better and (order_ok or abs(at["beta_hat"] + 2.5) < 1.5)),
    }


def _jsonable(obj):
    """Convert numpy / torch scalars for JSON dump."""
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
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    unit = run_unit_tests()
    summary["unit_tests"] = unit
    if not unit["passed"]:
        summary["verdict"] = "CANONICAL DTR UPDATE BLOCKED: unit tests failed"
        (OUT / "regression_summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2)
        )
        print(summary["verdict"])
        raise SystemExit(1)

    print("=== S2 smoke ===", flush=True)
    smoke = train_pair("S2", -2.5, epochs=8, tag="smoke")
    smoke_checks = check_scenario_expectations("S2", smoke)
    summary["s2_smoke"] = {
        "beta_hat": smoke["pair"]["dtr_age_temporal"]["beta_hat"],
        "test_auroc": smoke["pair"]["dtr_age_temporal"]["test"]["micro_auroc"],
        "gate": smoke["gate"],
        "checks": smoke_checks,
        # Smoke is short — only require it trains and produces finite beta
        "passed": bool(np.isfinite(smoke["pair"]["dtr_age_temporal"]["beta_hat"])),
    }
    print(
        f"S2 smoke beta_hat={summary['s2_smoke']['beta_hat']:.3f} "
        f"auroc={summary['s2_smoke']['test_auroc']:.3f}",
        flush=True,
    )

    print("=== S0–S3 regression ===", flush=True)
    regression = {}
    for scen, beta in (("S0", 0.0), ("S1", 0.0), ("S2", -2.5), ("S3", 2.5)):
        print(f"--- {scen} ---", flush=True)
        res = train_pair(scen, beta, epochs=30, tag="reg")
        checks = check_scenario_expectations(scen, res)
        regression[scen] = {
            "checks": checks,
            "gate": res["gate"],
            "test": res["pair"]["dtr_age_temporal"]["test"],
            "ablations": {
                "delta_bce_shuffle_age": res["pair"]["dtr_age_temporal"]["ablations"][
                    "delta_bce_shuffle_age"
                ],
                "delta_bce_beta0": res["pair"]["dtr_age_temporal"]["ablations"][
                    "delta_bce_beta0"
                ],
                "delta_bce_constant_age": res["pair"]["dtr_age_temporal"]["ablations"][
                    "delta_bce_constant_age"
                ],
            },
            "recovery": res["pair"]["dtr_age_temporal"]["recovery"],
            "content_persistence_diagnostics": res["pair"]["dtr_age_temporal"][
                "content_persistence_diagnostics"
            ],
        }
        print(
            f"{scen}: beta_hat={checks['beta_hat']:.3f} passed={checks['passed']}",
            flush=True,
        )
    summary["s0_s3_regression"] = regression

    s5 = s5_compatibility()
    summary["s5"] = s5
    print(
        f"S5: content_auprc={s5['content_auprc']:.3f} "
        f"baseline={s5['baseline_auprc']:.3f} "
        f"groups={s5['group_persistence']} order_ok={s5['group_order_ok']}",
        flush=True,
    )

    all_reg = all(bool(regression[s]["checks"]["passed"]) for s in ("S0", "S1", "S2", "S3"))
    if unit["passed"] and summary["s2_smoke"]["passed"] and all_reg and s5["passed"]:
        verdict = "CANONICAL CONTENT-PERSISTENCE DTR IMPLEMENTATION READY"
    else:
        failed = []
        if not unit["passed"]:
            failed.append("unit_tests")
        if not summary["s2_smoke"]["passed"]:
            failed.append("s2_smoke")
        for s in ("S0", "S1", "S2", "S3"):
            if not regression[s]["checks"]["passed"]:
                failed.append(f"regression_{s}")
        if not s5["passed"]:
            failed.append("s5")
        verdict = "CANONICAL DTR UPDATE BLOCKED: " + ", ".join(failed)

    summary["verdict"] = verdict
    (OUT / "regression_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    print(verdict, flush=True)
    if "BLOCKED" in verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
