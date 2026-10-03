"""Aggregate E01 vs C01 and write content-recovery tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from atomic.aggregate import ARM_METRICS, MATCHED_METRICS, _finite, _matched_rows, _summary, _write_csv
from content_bottleneck import ARTIFACT_ROOT, C01_ARTIFACT_ROOT
from high_impact.aggregate import iter_runs
from ladder.artifacts import write_json


def _rows_for_run(experiment, scenario, arm, seed, seed_dir: Path) -> list[dict[str, Any]]:
    metrics = json.loads((seed_dir / "metrics.json").read_text())
    mechanism = json.loads((seed_dir / "mechanism_metrics.json").read_text())
    blob = {**metrics, **mechanism}
    rows = []
    for metric in list(ARM_METRICS):
        value = _finite(blob.get(metric))
        if value is None:
            continue
        rows.append({
            "experiment": experiment, "scenario": scenario, "arm": arm,
            "seed": seed, "metric": metric, "value": value,
        })
    rec = mechanism.get("content_recovery") or {}
    for key in ("matrix_rmse", "pearson", "spearman", "sign_agreement", "background_abs_evidence_mean"):
        value = _finite(rec.get(key))
        if value is None:
            continue
        rows.append({
            "experiment": experiment, "scenario": scenario, "arm": arm,
            "seed": seed, "metric": f"content_{key}", "value": value,
        })
    return rows


def _load_c01():
    rows = []
    for experiment, scenario, arm, seed, seed_dir in iter_runs(C01_ARTIFACT_ROOT):
        if experiment != "C01_staged_current":
            continue
        rows.extend(_rows_for_run(experiment, scenario, arm, seed, seed_dir))
    rows.extend(_matched_rows(rows))
    return rows


def _paired(rows, reference="C01_staged_current"):
    index = {(r["experiment"], r["scenario"], r["arm"], r["seed"], r["metric"]): r["value"] for r in rows}
    out = []
    for (experiment, scenario, arm, seed, metric), value in sorted(index.items()):
        if experiment == reference:
            continue
        ref = index.get((reference, scenario, arm, seed, metric))
        if ref is None:
            continue
        out.append({
            "experiment": experiment, "scenario": scenario, "arm": arm, "seed": seed,
            "metric": metric, "candidate": value, "c01": ref, "delta": float(value) - float(ref),
        })
    return out


def _content_recovery_table(root: Path) -> list[dict[str, Any]]:
    rows = []
    for experiment, scenario, arm, seed, seed_dir in iter_runs(root):
        if arm != "age_temporal":
            continue
        rec_path = seed_dir / "content_recovery.json"
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text())
        rows.append({
            "experiment": experiment, "scenario": scenario, "seed": seed,
            "matrix_rmse": rec.get("matrix_rmse"),
            "pearson": rec.get("pearson"),
            "spearman": rec.get("spearman"),
            "sign_agreement": rec.get("sign_agreement"),
            "background_abs_evidence_mean": rec.get("background_abs_evidence_mean"),
        })
    return rows


def aggregate(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or ARTIFACT_ROOT)
    arm_rows = []
    for experiment, scenario, arm, seed, seed_dir in iter_runs(root):
        arm_rows.extend(_rows_for_run(experiment, scenario, arm, seed, seed_dir))
    arm_rows.extend(_matched_rows(arm_rows))
    all_rows = _load_c01() + arm_rows
    summary = _summary(all_rows)
    paired = _paired(all_rows)
    content_rows = _content_recovery_table(root)
    _write_csv(root / "e01_vs_c01_metrics.csv", all_rows, ["experiment", "scenario", "arm", "seed", "metric", "value"])
    _write_csv(
        root / "e01_vs_c01_summary.csv", summary,
        ["experiment", "scenario", "arm", "metric", "n", "mean", "std", "seed_ci95_lo", "seed_ci95_hi"],
    )
    _write_csv(
        root / "paired_delta_vs_C01.csv", paired,
        ["experiment", "scenario", "arm", "seed", "metric", "candidate", "c01", "delta"],
    )
    _write_csv(
        root / "target_signal_content_recovery.csv", content_rows,
        ["experiment", "scenario", "seed", "matrix_rmse", "pearson", "spearman", "sign_agreement", "background_abs_evidence_mean"],
    )
    payload = {
        "n_metric_rows": len(all_rows),
        "n_paired": len(paired),
        "n_content_rows": len(content_rows),
    }
    write_json(root / "aggregate_summary.json", payload)
    return payload
