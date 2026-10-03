"""Metrics from saved predictions and mechanism arrays. Does not load models."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from baselines.common.counterfactual import cf_rmse_age, cf_rmse_lag, surface_rmse
from evaluate import classification_metrics
from ladder.artifacts import write_json
from ladder.evaluation.metrics import _corr, _indexed, _rmse

from atomic.io import read_predictions

N_BOOT = 200


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _gate_summary(hat: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    left = np.asarray(hat, dtype=np.float64).reshape(-1)
    right = np.asarray(truth, dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size == 0:
        return {"rmse": None, "mae": None, "correlation": None, "n": 0}
    err = left - right
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "correlation": _corr(left, right),
        "n": int(left.size),
    }


def mechanism_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as blob:
        arrays = {key: blob[key] for key in blob.files}
    ages = arrays["surface_ages"]
    lags = arrays["surface_lags"]
    cf_ages = arrays["cf_ages"]
    cf_lags = arrays["cf_lags"]
    out: dict[str, Any] = {
        "architecture": str(np.asarray(arrays["architecture"]).item()),
        "surface_rmse": surface_rmse(
            _indexed(arrays["surface_model"], ages, lags),
            _indexed(arrays["surface_oracle"], ages, lags),
            ages=tuple(float(a) for a in ages),
            lags_days=tuple(float(x) for x in lags),
        ),
        "cf_rmse_age": cf_rmse_age(
            _indexed(arrays["cf_age_model"], cf_ages),
            _indexed(arrays["cf_age_oracle"], cf_ages),
            ages=tuple(float(a) for a in cf_ages),
        ),
        "cf_rmse_lag": cf_rmse_lag(
            _indexed(arrays["cf_lag_model"], cf_lags),
            _indexed(arrays["cf_lag_oracle"], cf_lags),
            lags_days=tuple(float(x) for x in cf_lags),
        ),
        "beta_true": float(np.asarray(arrays["beta_true"])),
        "theta0_true": float(np.asarray(arrays["theta0_true"])),
    }
    beta = np.asarray(arrays["beta"], dtype=np.float64).reshape(-1)
    out["beta"] = [float(x) for x in beta]
    out["beta_mean"] = float(np.mean(beta)) if beta.size else None
    out["abs_beta_mean"] = float(np.mean(np.abs(beta))) if beta.size else None
    out["theta"] = [float(x) for x in np.asarray(arrays["theta"], dtype=np.float64).reshape(-1)]
    signal = _gate_summary(arrays["gate_signal_hat"], arrays["gate_signal_true"])
    out["gate_signal_rmse"] = signal["rmse"]
    out["gate_signal_mae"] = signal["mae"]
    out["gate_signal_correlation"] = signal["correlation"]
    out["gate_signal_n"] = signal["n"]
    if "gate_signal_code" in arrays and signal["n"]:
        codes = np.asarray(arrays["gate_signal_code"]).reshape(-1)
        hat = np.asarray(arrays["gate_signal_hat"], dtype=np.float64).reshape(-1)
        truth = np.asarray(arrays["gate_signal_true"], dtype=np.float64).reshape(-1)
        per_code = {}
        for code in sorted(set(codes.tolist())):
            mask = codes == code
            stats = _gate_summary(hat[mask], truth[mask])
            per_code[str(code)] = {"rmse": stats["rmse"], "n": stats["n"]}
        out["gate_signal_rmse_by_code"] = per_code
    if int(np.asarray(arrays.get("has_global_gate", 0))) == 1 and "gate_surface_model" in arrays:
        grid = _gate_summary(arrays["gate_surface_model"], arrays["gate_surface_oracle"])
        out["gate_surface_rmse"] = grid["rmse"]
        out["gate_surface_mae"] = grid["mae"]
        out["gate_surface_correlation"] = grid["correlation"]
    else:
        out["gate_surface_rmse"] = None
        out["gate_surface_mae"] = None
        out["gate_surface_correlation"] = None
        out["gate_surface_note"] = "content-dependent gate; signal-encounter g_eff is the gate metric"
    if "content_free_lambda_age" in arrays and "lambda_true_age" in arrays:
        learned = np.asarray(arrays["content_free_lambda_age"], dtype=np.float64).reshape(-1)
        truth = np.asarray(arrays["lambda_true_age"], dtype=np.float64).reshape(-1)
        out["content_free_lambda_rmse"] = _rmse(learned, truth)
        out["content_free_lambda_corr"] = _corr(learned, truth)
    if "lambda_k" in arrays and "lambda_true_age" in arrays:
        truth = np.asarray(arrays["lambda_true_age"], dtype=np.float64).reshape(-1)
        per = []
        for curve in np.asarray(arrays["lambda_k"], dtype=np.float64):
            per.append({"rmse": _rmse(curve.reshape(-1), truth), "corr": _corr(curve.reshape(-1), truth)})
        out["lambda_k_vs_oracle"] = per
    if "pi_signal" in arrays:
        pi = np.asarray(arrays["pi_signal"], dtype=np.float64)
        if pi.size:
            mean_pi = pi.mean(axis=0)
            safe = np.clip(mean_pi, 1e-12, 1.0)
            out["mixture_mean_pi"] = [float(x) for x in mean_pi]
            out["mixture_entropy"] = float(-(safe * np.log(safe)).sum())
            counts = np.bincount(pi.argmax(axis=1), minlength=pi.shape[1]).astype(np.float64)
            out["mixture_assignment_fraction"] = [float(x) for x in counts / counts.sum()]
    return out


def _bootstrap(patient_ids, y, logits, logits_beta0, logits_full, logits_gate, *, n_boot: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    pids = np.asarray(patient_ids)
    unique = np.unique(pids)
    groups = {pid: np.where(pids == pid)[0] for pid in unique}

    def pack(idx: np.ndarray) -> dict[str, float]:
        base = classification_metrics(y[idx], logits[idx])["bce"]
        return {
            "bce": base,
            "delta_bce_beta0": classification_metrics(y[idx], logits_beta0[idx])["bce"] - base,
            "delta_bce_full_age_shuffle": classification_metrics(y[idx], logits_full[idx])["bce"] - base,
            "delta_bce_gate_age_shuffle": classification_metrics(y[idx], logits_gate[idx])["bce"] - base,
        }

    samples = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        samples.append(pack(np.concatenate([groups[pid] for pid in drawn])))
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
    run_dir = Path(run_dir)
    predictions = read_predictions(run_dir / "predictions.parquet")
    y = predictions["labels"]
    logits = predictions["logits"]
    predictive = classification_metrics(y, logits)
    beta0 = classification_metrics(y, predictions["logits_beta0"])
    full = classification_metrics(y, predictions["logits_full_age_shuffle"])
    gate = classification_metrics(y, predictions["logits_gate_age_shuffle"])
    mechanism = mechanism_from_npz(run_dir / "mechanism_outputs.npz")
    metrics = {
        "bce": predictive["bce"],
        "auroc": predictive["micro_auroc"],
        "auprc": predictive["micro_auprc"],
        "delta_bce_beta0": beta0["bce"] - predictive["bce"],
        "delta_bce_full_age_shuffle": full["bce"] - predictive["bce"],
        "delta_bce_gate_age_shuffle": gate["bce"] - predictive["bce"],
        "patient_bootstrap": _bootstrap(
            predictions["patient_id"], y, logits,
            predictions["logits_beta0"],
            predictions["logits_full_age_shuffle"],
            predictions["logits_gate_age_shuffle"],
            n_boot=n_boot,
        ),
        "n_rows": int(y.shape[0]),
        "n_targets": int(y.shape[1]),
    }
    again = classification_metrics(y, logits)
    if abs(again["bce"] - metrics["bce"]) > 1e-12:
        raise RuntimeError("Saved predictions did not reproduce BCE")
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "mechanism_metrics.json", mechanism)
    return {"metrics": metrics, "mechanism": mechanism}
