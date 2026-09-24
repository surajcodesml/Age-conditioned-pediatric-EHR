"""Universal counterfactual mechanism evaluation.

These tests work for ANY model regardless of internal parameterization.
They measure whether a model's predictions respond correctly to
controlled changes in age and temporal lag.

CF-RMSE_age:   hold history fixed, vary prediction age
CF-RMSE_lag:   hold age + content fixed, vary historical lags
Surface RMSE:  full age × lag grid comparison with oracle

For models without explicit age/time inputs, the counterfactuals
expose their inability to recover age/lag-dependent behavior.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


# Standard grids from config — duplicated here to avoid circular import.
CF_AGES = (2.0, 5.0, 9.0, 13.0, 17.0)
CF_LAGS_DAYS = (7.0, 30.0, 90.0, 180.0, 365.0, 730.0)
SURFACE_AGES = tuple(float(x) for x in range(0, 19))
SURFACE_LAGS_DAYS = (0.0, 7.0, 30.0, 90.0, 180.0, 365.0, 730.0)

# Mechanism classification thresholds (documented before running).
FUNCTIONAL_SURFACE_RMSE = 0.10
PARTIAL_SURFACE_RMSE = 0.25
S0_MAX_CF_AGE_RMSE = 0.05  # must not show false age interaction in S0


def cf_rmse_age(
    predict_fn: Callable[[float], np.ndarray],
    oracle_fn: Callable[[float], np.ndarray],
    ages: tuple[float, ...] = CF_AGES,
) -> float:
    """CF-RMSE_age: RMS error of P̂(Y|a*, H) vs oracle across age grid.

    Args:
        predict_fn: age_years → predicted probabilities [n_targets]
        oracle_fn:  age_years → oracle probabilities [n_targets]
    """
    errors = []
    for a in ages:
        p_hat = np.asarray(predict_fn(a), dtype=np.float64)
        p_true = np.asarray(oracle_fn(a), dtype=np.float64)
        errors.append(np.mean((p_hat - p_true) ** 2))
    return float(np.sqrt(np.mean(errors)))


def cf_rmse_lag(
    predict_fn: Callable[[float], np.ndarray],
    oracle_fn: Callable[[float], np.ndarray],
    lags_days: tuple[float, ...] = CF_LAGS_DAYS,
) -> float:
    """CF-RMSE_lag: RMS error of P̂(Y|a, τ) vs oracle across lag grid.

    Args:
        predict_fn: lag_days → predicted probabilities [n_targets]
        oracle_fn:  lag_days → oracle probabilities [n_targets]
    """
    errors = []
    for lag in lags_days:
        p_hat = np.asarray(predict_fn(lag), dtype=np.float64)
        p_true = np.asarray(oracle_fn(lag), dtype=np.float64)
        errors.append(np.mean((p_hat - p_true) ** 2))
    return float(np.sqrt(np.mean(errors)))


def surface_rmse(
    predict_fn: Callable[[float, float], np.ndarray],
    oracle_fn: Callable[[float, float], np.ndarray],
    ages: tuple[float, ...] = SURFACE_AGES,
    lags_days: tuple[float, ...] = SURFACE_LAGS_DAYS,
) -> float:
    """Surface RMSE over the full age × lag grid.

    Args:
        predict_fn: (age_years, lag_days) → predicted probs [n_targets]
        oracle_fn:  (age_years, lag_days) → oracle probs [n_targets]
    """
    errors = []
    for a in ages:
        for lag in lags_days:
            p_hat = np.asarray(predict_fn(a, lag), dtype=np.float64)
            p_true = np.asarray(oracle_fn(a, lag), dtype=np.float64)
            errors.append(np.mean((p_hat - p_true) ** 2))
    return float(np.sqrt(np.mean(errors)))


def classify_mechanism(
    surface_rmse_val: float,
    cf_age_rmse_s0: float | None = None,
) -> str:
    """Classify mechanism recovery quality.

    Returns one of:
        'FUNCTIONAL_RECOVERY'
        'PARTIAL_RECOVERY'
        'NO_MECHANISM_RECOVERY'

    Args:
        surface_rmse_val: Surface RMSE on S2
        cf_age_rmse_s0:   CF-RMSE_age on S0 (if available; checks false interaction)
    """
    # Reject if model shows false interaction in S0
    if cf_age_rmse_s0 is not None and cf_age_rmse_s0 > S0_MAX_CF_AGE_RMSE:
        return "NO_MECHANISM_RECOVERY"
    if surface_rmse_val < FUNCTIONAL_SURFACE_RMSE:
        return "FUNCTIONAL_RECOVERY"
    if surface_rmse_val < PARTIAL_SURFACE_RMSE:
        return "PARTIAL_RECOVERY"
    return "NO_MECHANISM_RECOVERY"


def build_surface_grid(
    predict_fn: Callable[[float, float], np.ndarray],
    ages: tuple[float, ...] = SURFACE_AGES,
    lags_days: tuple[float, ...] = SURFACE_LAGS_DAYS,
) -> np.ndarray:
    """Build the full prediction surface [n_ages, n_lags, n_targets]."""
    rows = []
    for a in ages:
        row = []
        for lag in lags_days:
            row.append(np.asarray(predict_fn(a, lag), dtype=np.float64))
        rows.append(np.stack(row, axis=0))
    return np.stack(rows, axis=0)


def full_counterfactual_report(
    predict_age_fn: Callable[[float], np.ndarray],
    predict_lag_fn: Callable[[float], np.ndarray],
    predict_surface_fn: Callable[[float, float], np.ndarray],
    oracle_age_fn: Callable[[float], np.ndarray],
    oracle_lag_fn: Callable[[float], np.ndarray],
    oracle_surface_fn: Callable[[float, float], np.ndarray],
    cf_age_rmse_s0: float | None = None,
) -> dict[str, Any]:
    """Compute all counterfactual metrics for one model (S0–S3 style).

    S5 heterogeneous-persistence metrics are added separately via
    ``baselines.synthetic.s5_eval.full_s5_counterfactual_report``.
    """
    cf_age = cf_rmse_age(predict_age_fn, oracle_age_fn)
    cf_lag = cf_rmse_lag(predict_lag_fn, oracle_lag_fn)
    srmse = surface_rmse(predict_surface_fn, oracle_surface_fn)
    classification = classify_mechanism(srmse, cf_age_rmse_s0)

    return {
        "cf_rmse_age": cf_age,
        "cf_rmse_lag": cf_lag,
        "surface_rmse": srmse,
        "CF_RMSE_age": cf_age,
        "CF_RMSE_lag": cf_lag,
        "Surface_RMSE": srmse,
        "mechanism_classification": classification,
        # S5-specific fields (null for S0–S3)
        "S5_Surface_RMSE_acute": None,
        "S5_Surface_RMSE_intermediate": None,
        "S5_Surface_RMSE_chronic": None,
        "S5_Surface_RMSE_mean": None,
        "persistence_order_correct": None,
        "thresholds": {
            "functional": FUNCTIONAL_SURFACE_RMSE,
            "partial": PARTIAL_SURFACE_RMSE,
            "s0_max_cf_age": S0_MAX_CF_AGE_RMSE,
        },
    }
