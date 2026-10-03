"""Metrics from saved predictions and mechanism arrays. Does not load models."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from baselines.common.counterfactual import cf_rmse_age, cf_rmse_lag, surface_rmse
from evaluate import classification_metrics

from ladder.artifacts import read_predictions, write_json

N_BOOT = 200


def _finite(value: float) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or np.std(left) <= 1e-8 or np.std(right) <= 1e-8:
        return None
    return _finite(float(np.corrcoef(left, right)[0, 1]))


def _rmse(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or left.size == 0:
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _indexed(grid: np.ndarray, coords_a: np.ndarray, coords_b: np.ndarray | None = None):
    if coords_b is None:
        index = {float(value): i for i, value in enumerate(coords_a)}

        def fetch(x: float) -> np.ndarray:
            return grid[index[float(x)]]

        return fetch
    index_a = {float(value): i for i, value in enumerate(coords_a)}
    index_b = {float(value): i for i, value in enumerate(coords_b)}

    def fetch2(a: float, b: float) -> np.ndarray:
        return grid[index_a[float(a)], index_b[float(b)]]

    return fetch2


def mechanism_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as blob:
        arrays = {key: blob[key] for key in blob.files}
    ages = arrays["surface_ages"]
    lags = arrays["surface_lags"]
    cf_ages = arrays["cf_ages"]
    cf_lags = arrays["cf_lags"]
    surface = surface_rmse(
        _indexed(arrays["surface_model"], ages, lags),
        _indexed(arrays["surface_oracle"], ages, lags),
        ages=tuple(float(a) for a in ages),
        lags_days=tuple(float(x) for x in lags),
    )
    cf_age = cf_rmse_age(
        _indexed(arrays["cf_age_model"], cf_ages),
        _indexed(arrays["cf_age_oracle"], cf_ages),
        ages=tuple(float(a) for a in cf_ages),
    )
    cf_lag = cf_rmse_lag(
        _indexed(arrays["cf_lag_model"], cf_lags),
        _indexed(arrays["cf_lag_oracle"], cf_lags),
        lags_days=tuple(float(x) for x in cf_lags),
    )
    beta = np.asarray(arrays["beta"], dtype=np.float64).reshape(-1)
    theta = np.asarray(arrays["theta"], dtype=np.float64).reshape(-1)
    arch = arrays["architecture"]
    architecture = str(arch.item()) if getattr(arch, "shape", None) == () else str(arch)
    out: dict[str, Any] = {
        "architecture": architecture,
        "surface_rmse": surface,
        "cf_rmse_age": cf_age,
        "cf_rmse_lag": cf_lag,
        "beta": [float(x) for x in beta],
        "theta": [float(x) for x in theta],
        "abs_beta_mean": float(np.mean(np.abs(beta))) if beta.size else None,
        "beta_true": float(np.asarray(arrays["beta_true"])),
        "theta0_true": float(np.asarray(arrays["theta0_true"])),
    }
    # Unweighted mean of component curves when a model has several lambdas.
    # Per-component values are retained; the mean is not chosen after fitting.
    if "lambda_age" in arrays and "lambda_true_age" in arrays:
        learned = np.asarray(arrays["lambda_age"], dtype=np.float64).reshape(-1)
        truth = np.asarray(arrays["lambda_true_age"], dtype=np.float64).reshape(-1)
        out["lambda_rmse"] = _rmse(learned, truth)
        out["lambda_corr"] = _corr(learned, truth)
    else:
        out["lambda_rmse"] = None
        out["lambda_corr"] = None
        out["lambda_note"] = (
            "lambda(a)=softplus(theta+beta z) is not the parameterization; "
            "rho(a) is stored instead"
        )
    if "lambda_k" in arrays and "lambda_true_age" in arrays:
        truth = np.asarray(arrays["lambda_true_age"], dtype=np.float64).reshape(-1)
        per = []
        for curve in np.asarray(arrays["lambda_k"], dtype=np.float64):
            per.append({
                "rmse": _rmse(curve.reshape(-1), truth),
                "corr": _corr(curve.reshape(-1), truth),
            })
        out["lambda_k_vs_oracle"] = per
    if "rho_age" in arrays:
        out["rho_age"] = [float(x) for x in np.asarray(arrays["rho_age"]).reshape(-1)]
    for key in ("mean_gate", "mean_mass", "mean_channel_score", "mean_mixture"):
        if key in arrays:
            out[key] = np.asarray(arrays[key], dtype=np.float64).reshape(-1).tolist()
    return out


def _patient_bootstrap(
    patient_ids: list[str],
    y: np.ndarray,
    logits: np.ndarray,
    logits_beta0: np.ndarray,
    logits_shuffle: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pids = np.asarray(patient_ids)
    unique = np.unique(pids)
    groups = {pid: np.where(pids == pid)[0] for pid in unique}

    def pack(idx: np.ndarray) -> dict[str, float]:
        base = classification_metrics(y[idx], logits[idx])["bce"]
        beta0 = classification_metrics(y[idx], logits_beta0[idx])["bce"]
        shuffled = classification_metrics(y[idx], logits_shuffle[idx])["bce"]
        return {
            "bce": base,
            "delta_bce_beta0": beta0 - base,
            "delta_bce_age_shuffle": shuffled - base,
        }

    samples = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[pid] for pid in drawn])
        samples.append(pack(idx))
    summary = {}
    for key in samples[0]:
        values = np.asarray([row[key] for row in samples], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci95_lo": float(np.percentile(values, 2.5)),
            "ci95_hi": float(np.percentile(values, 97.5)),
        }
    return summary


def compute_run_metrics(run_dir: Path, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Recompute predictive and mechanism metrics from saved artifacts."""
    run_dir = Path(run_dir)
    predictions = read_predictions(run_dir / "predictions.parquet")
    y = predictions["labels"]
    logits = predictions["logits"]
    predictive = classification_metrics(y, logits)
    beta0 = classification_metrics(y, predictions["logits_beta0"])
    shuffled = classification_metrics(y, predictions["logits_age_shuffle"])
    mechanism = mechanism_from_npz(run_dir / "mechanism_outputs.npz")
    bootstrap = _patient_bootstrap(
        predictions["patient_id"],
        y,
        logits,
        predictions["logits_beta0"],
        predictions["logits_age_shuffle"],
        n_boot=n_boot,
    )
    metrics = {
        "bce": predictive["bce"],
        "auroc": predictive["micro_auroc"],
        "auprc": predictive["micro_auprc"],
        "macro_auroc": predictive["macro_auroc"],
        "macro_auprc": predictive["macro_auprc"],
        "delta_bce_beta0": beta0["bce"] - predictive["bce"],
        "delta_bce_age_shuffle": shuffled["bce"] - predictive["bce"],
        "patient_bootstrap": bootstrap,
        "n_rows": int(y.shape[0]),
        "n_targets": int(y.shape[1]),
    }
    # Contract: the saved table reproduces the predictive numbers just written.
    again = classification_metrics(y, logits)
    if abs(again["bce"] - metrics["bce"]) > 1e-12:
        raise RuntimeError("Saved predictions did not reproduce BCE")
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "mechanism_metrics.json", mechanism)
    return {"metrics": metrics, "mechanism": mechanism}
