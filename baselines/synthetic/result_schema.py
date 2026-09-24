"""Canonical result-record schema for the synthetic baseline benchmark.

Supports consolidated S0–S5 tables. S5-specific fields are null for S0–S3.
"""
from __future__ import annotations

from typing import Any


RESULT_FIELD_ORDER: tuple[str, ...] = (
    "scenario",
    "model",
    "AUROC",
    "AUPRC",
    "BCE",
    "CF_RMSE_age",
    "CF_RMSE_lag",
    "Surface_RMSE",
    "S5_Surface_RMSE_acute",
    "S5_Surface_RMSE_intermediate",
    "S5_Surface_RMSE_chronic",
    "S5_Surface_RMSE_mean",
    "persistence_order_correct",
    "mechanism_classification",
)

S5_NULL_FIELDS: tuple[str, ...] = (
    "S5_Surface_RMSE_acute",
    "S5_Surface_RMSE_intermediate",
    "S5_Surface_RMSE_chronic",
    "S5_Surface_RMSE_mean",
    "persistence_order_correct",
)


def empty_result_record(
    *,
    scenario: str,
    model: str,
) -> dict[str, Any]:
    rec = {k: None for k in RESULT_FIELD_ORDER}
    rec["scenario"] = scenario
    rec["model"] = model
    return rec


def from_train_and_cf(
    *,
    scenario: str,
    model: str,
    test_metrics: dict[str, Any] | None = None,
    cf_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat result record from training metrics + CF report."""
    rec = empty_result_record(scenario=scenario, model=model)
    if test_metrics:
        # Prefer micro metrics used throughout the suite
        rec["AUROC"] = test_metrics.get("micro_auroc", test_metrics.get("auroc"))
        rec["AUPRC"] = test_metrics.get("micro_auprc", test_metrics.get("auprc"))
        rec["BCE"] = test_metrics.get("bce")

    if cf_report:
        rec["CF_RMSE_age"] = cf_report.get("cf_rmse_age", cf_report.get("CF_RMSE_age"))
        rec["CF_RMSE_lag"] = cf_report.get("cf_rmse_lag", cf_report.get("CF_RMSE_lag"))
        rec["Surface_RMSE"] = cf_report.get(
            "surface_rmse", cf_report.get("Surface_RMSE"),
        )
        rec["mechanism_classification"] = cf_report.get("mechanism_classification")

        for field in S5_NULL_FIELDS:
            if field in cf_report:
                rec[field] = cf_report[field]

    # Explicit nulls for non-S5
    if scenario != "S5":
        for field in S5_NULL_FIELDS:
            rec[field] = None

    return rec


def consolidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a list of result dicts to the canonical field order."""
    out = []
    for r in records:
        row = empty_result_record(
            scenario=str(r.get("scenario", "")),
            model=str(r.get("model", "")),
        )
        for k in RESULT_FIELD_ORDER:
            if k in r:
                row[k] = r[k]
        if row["scenario"] != "S5":
            for field in S5_NULL_FIELDS:
                row[field] = None
        out.append(row)
    return out
