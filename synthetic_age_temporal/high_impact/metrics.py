"""Metrics from saved high-impact artifacts. Does not load models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from atomic.io import read_predictions
from atomic.metrics import _bootstrap, _gate_summary, mechanism_from_npz as _atomic_mechanism
from evaluate import classification_metrics
from ladder.artifacts import write_json


def _head_specialization(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "query_vectors" in arrays:
        q = np.asarray(arrays["query_vectors"], dtype=np.float64)
        norms = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
        cos = (q @ q.T) / (norms @ norms.T)
        out["query_cosine"] = cos.tolist()
    if "u_valid" in arrays:
        u = np.asarray(arrays["u_valid"], dtype=np.float64)
        if u.ndim == 2 and u.shape[0] > 1:
            corr = np.corrcoef(u.T)
            out["content_score_corr"] = corr.tolist()
            out["mean_retrieval_mass"] = np.mean(np.exp(np.clip(u, None, 20)), axis=0).tolist()
            # Effective number of active heads from mean softmass.
            mass = np.mean(np.exp(np.clip(u - u.max(axis=1, keepdims=True), -40, 0)), axis=0)
            mass = mass / max(float(mass.sum()), 1e-12)
            out["effective_n_heads"] = float(np.exp(-(mass * np.log(mass.clip(1e-12))).sum()))
    if "h_head_norms" in arrays:
        norms = np.asarray(arrays["h_head_norms"], dtype=np.float64)
        out["mean_head_contribution_norm"] = norms.mean(axis=0).tolist()
    if "target_head_weight_norm" in arrays:
        out["target_head_weight_norm_mean"] = np.asarray(
            arrays["target_head_weight_norm"], dtype=np.float64
        ).mean(axis=0).tolist()
    if "gate_signal_heads" in arrays and "gate_signal_true" in arrays:
        heads = np.asarray(arrays["gate_signal_heads"], dtype=np.float64)
        truth = np.asarray(arrays["gate_signal_true"], dtype=np.float64)
        per = []
        for h in range(heads.shape[1]):
            per.append(_gate_summary(heads[:, h], truth))
        out["gate_rmse_per_head"] = [row["rmse"] for row in per]
    if "beta" in arrays:
        out["beta"] = np.asarray(arrays["beta"], dtype=np.float64).reshape(-1).tolist()
        out["beta_mean"] = float(np.mean(arrays["beta"]))
        out["abs_beta_mean"] = float(np.mean(np.abs(arrays["beta"])))
    for key in ("beta_global", "delta_h", "lambda_h"):
        if key in arrays:
            out[key] = np.asarray(arrays[key], dtype=np.float64).reshape(-1).tolist() if np.asarray(arrays[key]).ndim == 1 else np.asarray(arrays[key], dtype=np.float64).tolist()
    return out


def mechanism_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as blob:
        arrays = {key: blob[key] for key in blob.files}
    # Reuse atomic surface/CF/gate-signal metrics by writing a temporary-compatible view.
    # Call the atomic helper on the same file: it tolerates missing mixture fields.
    base = _atomic_mechanism(path)
    base.update(_head_specialization(arrays))
    return base


def compute_run_metrics(run_dir: Path, *, n_boot: int = 200) -> dict[str, Any]:
    run_dir = Path(run_dir)
    predictions = read_predictions(run_dir / "predictions.parquet")
    y = predictions["labels"]
    logits = predictions["logits"]
    predictive = classification_metrics(y, logits)
    beta0 = classification_metrics(y, predictions["logits_beta0"])
    full = classification_metrics(y, predictions["logits_full_age_shuffle"])
    gate = classification_metrics(y, predictions["logits_gate_age_shuffle"])
    mechanism = mechanism_from_npz(run_dir / "mechanism_outputs.npz")

    with np.load(run_dir / "mechanism_outputs.npz", allow_pickle=False) as blob:
        arrays = {key: blob[key] for key in blob.files}
    ablation = {}
    for key in arrays:
        if key.startswith("logits_ablate_head_"):
            h = int(key.rsplit("_", 1)[-1])
            ablated = classification_metrics(y, arrays[key])
            ablation[str(h)] = {
                "bce": ablated["bce"],
                "delta_bce": ablated["bce"] - predictive["bce"],
            }
    if ablation:
        mechanism["head_ablation_delta_bce"] = ablation

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
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "mechanism_metrics.json", mechanism)
    return {"metrics": metrics, "mechanism": mechanism}
