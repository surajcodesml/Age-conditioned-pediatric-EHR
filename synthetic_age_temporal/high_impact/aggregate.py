"""Aggregate high-impact metrics and paired deltas versus C01."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from atomic.aggregate import ARM_METRICS, MATCHED_METRICS, _finite, _matched_rows, _summary, _write_csv
from high_impact import C01_ARTIFACT_ROOT
from ladder.artifacts import write_json

EXTRA_METRICS = (
    "effective_n_heads",
)


def iter_runs(root: Path):
    root = Path(root)
    if not root.exists():
        return
    for experiment_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != "figures"):
        for scenario_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for arm_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
                for seed_dir in sorted(arm_dir.glob("seed_*")):
                    if (seed_dir / "metrics.json").exists() and (seed_dir / "mechanism_metrics.json").exists():
                        yield experiment_dir.name, scenario_dir.name.upper(), arm_dir.name, int(seed_dir.name.split("_")[1]), seed_dir


def _rows_for_run(experiment, scenario, arm, seed, seed_dir: Path) -> list[dict[str, Any]]:
    metrics = json.loads((seed_dir / "metrics.json").read_text())
    mechanism = json.loads((seed_dir / "mechanism_metrics.json").read_text())
    blob = {**metrics, **mechanism}
    rows = []
    for metric in list(ARM_METRICS) + list(EXTRA_METRICS):
        value = _finite(blob.get(metric))
        if value is None:
            continue
        rows.append({
            "experiment": experiment, "scenario": scenario, "arm": arm,
            "seed": seed, "metric": metric, "value": value,
        })
    ablation = mechanism.get("head_ablation_delta_bce") or {}
    for head, payload in ablation.items():
        value = _finite(payload.get("delta_bce"))
        if value is None:
            continue
        rows.append({
            "experiment": experiment, "scenario": scenario, "arm": arm,
            "seed": seed, "metric": f"head_ablation_delta_bce_{head}", "value": value,
        })
    return rows


def _load_c01_rows() -> list[dict[str, Any]]:
    rows = []
    root = C01_ARTIFACT_ROOT
    for experiment, scenario, arm, seed, seed_dir in iter_runs(root):
        if experiment != "C01_staged_current":
            continue
        for row in _rows_for_run("C01_staged_current", scenario, arm, seed, seed_dir):
            rows.append(row)
    rows.extend(_matched_rows(rows))
    return rows


def _paired(rows: list[dict[str, Any]], reference: str = "C01_staged_current") -> list[dict[str, Any]]:
    index = {(r["experiment"], r["scenario"], r["arm"], r["seed"], r["metric"]): r["value"] for r in rows}
    out = []
    for (experiment, scenario, arm, seed, metric), value in sorted(index.items()):
        if experiment == reference:
            continue
        ref = index.get((reference, scenario, arm, seed, metric))
        if ref is None:
            continue
        out.append({
            "experiment": experiment,
            "scenario": scenario,
            "arm": arm,
            "seed": seed,
            "metric": metric,
            "candidate": value,
            "c01": ref,
            "delta": float(value) - float(ref),
        })
    return out


def aggregate(root: Path) -> dict[str, Any]:
    root = Path(root)
    arm_rows: list[dict[str, Any]] = []
    for experiment, scenario, arm, seed, seed_dir in iter_runs(root):
        arm_rows.extend(_rows_for_run(experiment, scenario, arm, seed, seed_dir))
    arm_rows.extend(_matched_rows(arm_rows))
    c01_rows = _load_c01_rows()
    all_rows = c01_rows + arm_rows
    summary = _summary(all_rows)
    paired = _paired(all_rows, reference="C01_staged_current")
    _write_csv(root / "high_impact_followup_metrics.csv", all_rows, ["experiment", "scenario", "arm", "seed", "metric", "value"])
    _write_csv(
        root / "high_impact_followup_summary.csv",
        summary,
        ["experiment", "scenario", "arm", "metric", "n", "mean", "std", "seed_ci95_lo", "seed_ci95_hi"],
    )
    _write_csv(
        root / "paired_delta_vs_C01.csv",
        paired,
        ["experiment", "scenario", "arm", "seed", "metric", "candidate", "c01", "delta"],
    )
    payload = {
        "n_metric_rows": len(all_rows),
        "n_paired_rows": len(paired),
        "metrics_csv": str(root / "high_impact_followup_metrics.csv"),
        "summary_csv": str(root / "high_impact_followup_summary.csv"),
        "paired_csv": str(root / "paired_delta_vs_C01.csv"),
        "c01_source": str(C01_ARTIFACT_ROOT / "C01_staged_current"),
    }
    write_json(root / "high_impact_followup_summary.json", payload)
    return payload
