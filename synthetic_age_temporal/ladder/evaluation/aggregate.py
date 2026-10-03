"""Aggregate saved run metrics. Does not load models or rerun inference."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ladder import REPO_ROOT

LONG_FIELDS = ["experiment", "scenario", "arm", "seed", "metric", "value"]
SUMMARY_FIELDS = [
    "experiment", "scenario", "arm", "metric", "n", "mean", "std",
    "seed_ci95_lo", "seed_ci95_hi",
]

REFERENCE_MODELS = (
    ("E00_dtr_age_temporal_new", "age_temporal", "dtr_age_temporal_new"),
    ("E00_dtr_temporal_only_new", "temporal_only", "dtr_temporal_only_new"),
    ("E00_legacy_dtr_age_temporal", "age_temporal", "dtr_age_temporal"),
    ("E00_legacy_dtr_temporal_only", "temporal_only", "dtr_temporal_only"),
    ("CEHR-BERT", "cehrbert", "cehrbert"),
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


def _add(rows: list[dict[str, Any]], experiment: str, scenario: str, arm: str, seed: int, metric: str, value: Any) -> None:
    number = _finite(value)
    if number is None:
        return
    rows.append({
        "experiment": experiment,
        "scenario": scenario,
        "arm": arm,
        "seed": int(seed),
        "metric": metric,
        "value": number,
    })


def rows_from_run(run_dir: Path, *, experiment: str, scenario: str, arm: str, seed: int) -> list[dict[str, Any]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    mechanism = json.loads((run_dir / "mechanism_metrics.json").read_text())
    rows: list[dict[str, Any]] = []
    for metric in (
        "bce", "auroc", "auprc", "delta_bce_beta0", "delta_bce_age_shuffle",
    ):
        _add(rows, experiment, scenario, arm, seed, metric, metrics.get(metric))
    for metric in (
        "surface_rmse", "cf_rmse_age", "cf_rmse_lag", "lambda_rmse", "lambda_corr",
        "abs_beta_mean",
    ):
        _add(rows, experiment, scenario, arm, seed, metric, mechanism.get(metric))
    beta = mechanism.get("beta") or []
    if len(beta) == 1:
        _add(rows, experiment, scenario, arm, seed, "beta", beta[0])
    else:
        for i, value in enumerate(beta):
            _add(rows, experiment, scenario, arm, seed, f"beta_{i}", value)
    theta = mechanism.get("theta") or []
    if len(theta) == 1:
        _add(rows, experiment, scenario, arm, seed, "theta0", theta[0])
    else:
        for i, value in enumerate(theta):
            _add(rows, experiment, scenario, arm, seed, f"theta_{i}", value)
    return rows


def iter_ladder_runs(root: Path):
    root = Path(root)
    if not root.exists():
        return
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if exp_dir.name in {"figures"}:
            continue
        for scen_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            scenario = scen_dir.name.upper()
            for arm in ("age_temporal", "temporal_only"):
                arm_dir = scen_dir / arm
                if not arm_dir.is_dir():
                    continue
                for seed_dir in sorted(arm_dir.glob("seed_*")):
                    if not (seed_dir / "metrics.json").exists():
                        continue
                    if not (seed_dir / "mechanism_metrics.json").exists():
                        continue
                    seed = int(seed_dir.name.split("_", 1)[1])
                    yield exp_dir.name, scenario, arm, seed, seed_dir


def reference_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = REPO_ROOT / "results" / "baselines" / "synthetic"
    for experiment, arm, dirname in REFERENCE_MODELS:
        model_dir = base / dirname
        if not model_dir.is_dir():
            continue
        for scen_dir in sorted(model_dir.iterdir()):
            result_path = scen_dir / "result.json"
            if not scen_dir.is_dir() or not result_path.exists():
                continue
            if scen_dir.name not in {"S0", "S1", "S2", "S3"}:
                continue
            payload = json.loads(result_path.read_text())
            seed = int(payload.get("seed", 0))
            _add(rows, experiment, scen_dir.name, arm, seed, "bce", payload.get("BCE"))
            _add(rows, experiment, scen_dir.name, arm, seed, "auroc", payload.get("AUROC"))
            _add(rows, experiment, scen_dir.name, arm, seed, "auprc", payload.get("AUPRC"))
            _add(rows, experiment, scen_dir.name, arm, seed, "surface_rmse", payload.get("Surface_RMSE"))
            _add(rows, experiment, scen_dir.name, arm, seed, "cf_rmse_age", payload.get("CF_RMSE_age"))
            _add(rows, experiment, scen_dir.name, arm, seed, "cf_rmse_lag", payload.get("CF_RMSE_lag"))
            _add(rows, experiment, scen_dir.name, arm, seed, "delta_bce_beta0", payload.get("delta_BCE_beta0"))
            _add(rows, experiment, scen_dir.name, arm, seed, "delta_bce_age_shuffle", payload.get("delta_BCE_age_shuffle"))
            _add(rows, experiment, scen_dir.name, arm, seed, "beta", payload.get("beta_hat"))
            _add(rows, experiment, scen_dir.name, arm, seed, "lambda_rmse", payload.get("lambda_RMSE"))
            _add(rows, experiment, scen_dir.name, arm, seed, "lambda_corr", payload.get("lambda_corr"))
    small = base / "cehrbert_small"
    if small.is_dir():
        for seed_dir in sorted(small.glob("seed*")):
            result_path = seed_dir / "S2" / "result.json"
            if not result_path.exists():
                continue
            payload = json.loads(result_path.read_text())
            seed = int(payload.get("seed", seed_dir.name.replace("seed", "")))
            _add(rows, "CEHR-BERT-small", "S2", "cehrbert_small", seed, "bce", payload.get("BCE"))
            _add(rows, "CEHR-BERT-small", "S2", "cehrbert_small", seed, "auroc", payload.get("AUROC"))
            _add(rows, "CEHR-BERT-small", "S2", "cehrbert_small", seed, "auprc", payload.get("AUPRC"))
            _add(rows, "CEHR-BERT-small", "S2", "cehrbert_small", seed, "surface_rmse", payload.get("Surface_RMSE"))
            _add(rows, "CEHR-BERT-small", "S2", "cehrbert_small", seed, "cf_rmse_age", payload.get("CF_RMSE_age"))
            _add(rows, "CEHR-BERT-small", "S2", "cehrbert_small", seed, "cf_rmse_lag", payload.get("CF_RMSE_lag"))
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        key = (row["experiment"], row["scenario"], row["arm"], row["metric"])
        groups.setdefault(key, []).append(float(row["value"]))
    summary = []
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            half = 1.96 * std / math.sqrt(n)
            lo, hi = mean - half, mean + half
        else:
            lo = hi = ""
        summary.append({
            "experiment": key[0],
            "scenario": key[1],
            "arm": key[2],
            "metric": key[3],
            "n": n,
            "mean": mean,
            "std": std,
            "seed_ci95_lo": lo,
            "seed_ci95_hi": hi,
        })
    return summary


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def aggregate(root: Path) -> dict[str, Any]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for experiment, scenario, arm, seed, run_dir in iter_ladder_runs(root):
        rows.extend(rows_from_run(
            run_dir, experiment=experiment, scenario=scenario, arm=arm, seed=seed,
        ))
    rows.extend(reference_rows())
    summary = summarize(rows)
    _write_csv(root / "architecture_ladder_metrics.csv", LONG_FIELDS, rows)
    _write_csv(root / "architecture_ladder_summary.csv", SUMMARY_FIELDS, summary)
    payload = {
        "n_long_rows": len(rows),
        "n_summary_rows": len(summary),
        "metrics_csv": str(root / "architecture_ladder_metrics.csv"),
        "summary_csv": str(root / "architecture_ladder_summary.csv"),
    }
    (root / "architecture_ladder_summary.json").write_text(json.dumps(payload, indent=2))
    return payload
