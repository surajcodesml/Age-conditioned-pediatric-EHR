"""Aggregate saved atomic-follow-up metrics. Does not load models."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ladder.artifacts import write_json

ARM_METRICS = (
    "bce", "auroc", "auprc",
    "surface_rmse", "cf_rmse_age", "cf_rmse_lag",
    "delta_bce_beta0", "delta_bce_full_age_shuffle", "delta_bce_gate_age_shuffle",
    "gate_signal_rmse", "gate_signal_mae", "gate_signal_correlation",
    "gate_surface_rmse", "gate_surface_mae", "gate_surface_correlation",
    "beta_mean", "abs_beta_mean", "content_free_lambda_rmse", "content_free_lambda_corr",
)
MATCHED_METRICS = (
    "delta_bce_to_minus_at",
    "delta_auroc_at_minus_to",
    "delta_auprc_at_minus_to",
)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def iter_runs(root: Path):
    root = Path(root)
    if not root.exists():
        return
    for experiment_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
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
    for metric in ARM_METRICS:
        value = _finite(blob.get(metric))
        if value is None:
            continue
        rows.append({
            "experiment": experiment,
            "scenario": scenario,
            "arm": arm,
            "seed": seed,
            "metric": metric,
            "value": value,
        })
    return rows


def _matched_rows(arm_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["experiment"], row["scenario"], row["arm"], row["seed"], row["metric"]): row["value"] for row in arm_rows}
    keys = {(row["experiment"], row["scenario"], row["seed"]) for row in arm_rows}
    out = []
    for experiment, scenario, seed in sorted(keys):
        def get(arm: str, metric: str) -> float | None:
            return index.get((experiment, scenario, arm, seed, metric))
        pairs = (
            ("delta_bce_to_minus_at", (get("temporal_only", "bce"), get("age_temporal", "bce")), lambda a, b: a - b),
            ("delta_auroc_at_minus_to", (get("age_temporal", "auroc"), get("temporal_only", "auroc")), lambda a, b: a - b),
            ("delta_auprc_at_minus_to", (get("age_temporal", "auprc"), get("temporal_only", "auprc")), lambda a, b: a - b),
        )
        for metric, (left, right), op in pairs:
            if left is None or right is None:
                continue
            out.append({
                "experiment": experiment,
                "scenario": scenario,
                "arm": "matched",
                "seed": seed,
                "metric": metric,
                "value": float(op(left, right)),
            })
    return out


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        groups.setdefault((row["experiment"], row["scenario"], row["arm"], row["metric"]), []).append(float(row["value"]))
    out = []
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        half = 1.96 * std / math.sqrt(n) if n > 1 else None
        out.append({
            "experiment": key[0],
            "scenario": key[1],
            "arm": key[2],
            "metric": key[3],
            "n": n,
            "mean": mean,
            "std": std,
            "seed_ci95_lo": None if half is None else mean - half,
            "seed_ci95_hi": None if half is None else mean + half,
        })
    return out


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["experiment"], row["scenario"], row["arm"], row["seed"], row["metric"]): row["value"] for row in rows}
    out = []
    for (experiment, scenario, arm, seed, metric), value in sorted(index.items()):
        if experiment == "C00_current_dtr":
            continue
        c00 = index.get(("C00_current_dtr", scenario, arm, seed, metric))
        if c00 is None:
            continue
        out.append({
            "experiment": experiment,
            "scenario": scenario,
            "arm": arm,
            "seed": seed,
            "metric": metric,
            "candidate": value,
            "c00": c00,
            "delta": float(value) - float(c00),
        })
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root: Path) -> dict[str, Any]:
    root = Path(root)
    arm_rows: list[dict[str, Any]] = []
    for experiment, scenario, arm, seed, seed_dir in iter_runs(root):
        arm_rows.extend(_rows_for_run(experiment, scenario, arm, seed, seed_dir))
    rows = arm_rows + _matched_rows(arm_rows)
    summary = _summary(rows)
    paired = _paired(rows)
    _write_csv(root / "atomic_followup_metrics.csv", rows, ["experiment", "scenario", "arm", "seed", "metric", "value"])
    _write_csv(
        root / "atomic_followup_summary.csv",
        summary,
        ["experiment", "scenario", "arm", "metric", "n", "mean", "std", "seed_ci95_lo", "seed_ci95_hi"],
    )
    _write_csv(
        root / "paired_delta_vs_C00.csv",
        paired,
        ["experiment", "scenario", "arm", "seed", "metric", "candidate", "c00", "delta"],
    )
    payload = {
        "n_metric_rows": len(rows),
        "n_paired_rows": len(paired),
        "metrics_csv": str(root / "atomic_followup_metrics.csv"),
        "summary_csv": str(root / "atomic_followup_summary.csv"),
        "paired_csv": str(root / "paired_delta_vs_C00.csv"),
    }
    write_json(root / "atomic_followup_summary.json", payload)
    return payload
