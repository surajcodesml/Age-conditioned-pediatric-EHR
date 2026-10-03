"""Checkpoint to immutable predictions and mechanism arrays. No metrics here."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.common.counterfactual import (
    CF_AGES,
    CF_LAGS_DAYS,
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    build_surface_grid,
)
from baselines.synthetic.counterfactual_eval import _build_oracle_fns, make_predict_fns
from config import lambda_true, tau_from_days
from ladder.artifacts import write_json
from ladder.cached_data import cached_dtr_loaders
from ladder.inference.predict import _EncounterPredictor, select_templates

from atomic.models import AtomicDTR, build_atomic


def load_trained_model(path: Path, device: torch.device) -> tuple[AtomicDTR, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_atomic(
        ckpt["config"],
        int(ckpt["n_codes"]),
        int(ckpt["n_targets"]),
        age_temporal=bool(ckpt["age_temporal"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.configure_arm_()
    model.to(device)
    model.eval()
    return model, ckpt


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _logits(out: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    return out


def _signal_ids(vocab) -> dict[int, str]:
    return {
        int(idx): str(token)
        for idx, token in vocab.itos.items()
        if str(token).startswith("SYN_SIGNAL_")
    }


@torch.no_grad()
def _collect(
    model: AtomicDTR,
    loader,
    device: torch.device,
    *,
    shuffle_seed: int,
    signal_ids: dict[int, str],
    theta0_true: float,
    beta_true: float,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    batches = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        batches.append(batch)
    if not batches:
        raise RuntimeError("Empty loader")
    ages = np.concatenate([batch["age"].numpy() for batch in batches])
    shuffled = np.random.default_rng(int(shuffle_seed)).permutation(ages)

    labels, logits, logits_b0, logits_full, logits_gate = [], [], [], [], []
    example_ids, patient_ids, age_rows = [], [], []
    gate_hat, gate_true, gate_code = [], [], []
    pi_rows = []
    pointer = 0
    for batch in batches:
        moved = _move(batch, device)
        parts = model(
            enc_code_ids=moved["enc_code_ids"],
            enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"],
            enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"],
            enc_lag_days=moved.get("enc_lag_days"),
            return_parts=True,
        )
        labels.append(moved["labels"].cpu().numpy())
        logits.append(_logits(parts).cpu().numpy())
        age_rows.append(moved["age"].cpu().numpy())
        example_ids.append(batch["example_ids"].numpy())
        patient_ids.extend(batch["patient_ids"])
        _gather_signal_gates(
            parts, batch, signal_ids, theta0_true, beta_true,
            gate_hat, gate_true, gate_code, pi_rows,
        )
        pointer += int(moved["age"].shape[0])

    saved = model.zero_all_betas_()
    for batch in batches:
        moved = _move(batch, device)
        out = model(
            enc_code_ids=moved["enc_code_ids"],
            enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"],
            enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"],
            enc_lag_days=moved.get("enc_lag_days"),
        )
        logits_b0.append(_logits(out).cpu().numpy())
    model.restore_betas_(saved)

    pointer = 0
    for batch in batches:
        moved = _move(batch, device)
        width = int(moved["age"].shape[0])
        age_full = torch.tensor(shuffled[pointer:pointer + width], dtype=torch.float32, device=device)
        pointer += width
        out = model(
            enc_code_ids=moved["enc_code_ids"],
            enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"],
            enc_padding_mask=moved["enc_padding_mask"],
            age=age_full,
            enc_lag_days=moved.get("enc_lag_days"),
        )
        logits_full.append(_logits(out).cpu().numpy())

    pointer = 0
    for batch in batches:
        moved = _move(batch, device)
        width = int(moved["age"].shape[0])
        age_gate = torch.tensor(shuffled[pointer:pointer + width], dtype=torch.float32, device=device)
        pointer += width
        out = model(
            enc_code_ids=moved["enc_code_ids"],
            enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"],
            enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"],
            gate_age=age_gate,
            enc_lag_days=moved.get("enc_lag_days"),
        )
        logits_gate.append(_logits(out).cpu().numpy())

    payload: dict[str, Any] = {
        "example_id": np.concatenate(example_ids),
        "patient_id": patient_ids,
        "age": np.concatenate(age_rows),
        "labels": np.concatenate(labels),
        "logits": np.concatenate(logits),
        "logits_beta0": np.concatenate(logits_b0),
        "logits_full_age_shuffle": np.concatenate(logits_full),
        "logits_gate_age_shuffle": np.concatenate(logits_gate),
        "gate_signal_hat": np.asarray(gate_hat, dtype=np.float64),
        "gate_signal_true": np.asarray(gate_true, dtype=np.float64),
        "gate_signal_code": np.asarray(gate_code, dtype="U32"),
    }
    if pi_rows:
        payload["pi_signal"] = np.asarray(pi_rows, dtype=np.float64)
    return payload


def _gather_signal_gates(parts, batch, signal_ids, theta0_true, beta_true, gate_hat, gate_true, gate_code, pi_rows) -> None:
    g = parts["g"].detach().cpu().numpy()
    tau = batch["enc_tau"].numpy()
    pad = batch["enc_padding_mask"].numpy()
    n_sig = batch["enc_n_signal"].numpy()
    code_ids = batch["enc_code_ids"].numpy()
    code_mask = batch["enc_code_mask"].numpy()
    ages = batch["age"].numpy()
    pi = parts["pi"].detach().cpu().numpy() if "pi" in parts else None
    for b in range(g.shape[0]):
        lam = float(lambda_true(float(ages[b]), theta0_true, beta_true))
        for m in range(g.shape[1]):
            if bool(pad[b, m]) or int(n_sig[b, m]) <= 0:
                continue
            names = []
            for k in range(code_ids.shape[2]):
                if not bool(code_mask[b, m, k]):
                    continue
                name = signal_ids.get(int(code_ids[b, m, k]))
                if name is not None:
                    names.append(name)
            if not names:
                continue
            hat = float(g[b, m])
            truth = float(np.exp(-lam * float(tau[b, m])))
            for name in names:
                gate_hat.append(hat)
                gate_true.append(truth)
                gate_code.append(name)
            if pi is not None:
                pi_rows.append(pi[b, m])


def _parameter_arrays(model: AtomicDTR, device: torch.device, theta0_true: float, beta_true: float) -> dict[str, np.ndarray]:
    ages_t = torch.tensor(SURFACE_AGES, dtype=torch.float32, device=device)
    beta = model._active_beta().detach().float().cpu().numpy().reshape(-1)
    if model.variant in ("shared_beta_mixture", "component_beta_mixture"):
        theta = model.theta_k.detach().float().cpu().numpy().reshape(-1)
    else:
        theta = model.base.theta0.detach().float().cpu().numpy().reshape(-1)
    out: dict[str, np.ndarray] = {
        "beta": beta.astype(np.float64),
        "theta": theta.astype(np.float64),
        "surface_ages": np.asarray(SURFACE_AGES, dtype=np.float64),
        "surface_lags": np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64),
        "cf_ages": np.asarray(CF_AGES, dtype=np.float64),
        "cf_lags": np.asarray(CF_LAGS_DAYS, dtype=np.float64),
        "beta_true": np.asarray(beta_true, dtype=np.float64),
        "theta0_true": np.asarray(theta0_true, dtype=np.float64),
        "architecture": np.asarray(model.variant),
        "has_global_gate": np.asarray(1 if model.has_global_gate() else 0, dtype=np.int64),
        "lambda_true_age": np.asarray(lambda_true(np.asarray(SURFACE_AGES, dtype=np.float64), theta0_true, beta_true), dtype=np.float64),
    }
    lam_k = model.component_lambda(ages_t)
    if lam_k is not None:
        curve = lam_k.detach().cpu().numpy().astype(np.float64)
        out["lambda_k"] = curve.T if curve.ndim == 2 else curve.reshape(1, -1)
    free = model.content_free_lambda(ages_t)
    if free is not None and model.variant not in ("shared_beta_mixture", "component_beta_mixture"):
        out["content_free_lambda_age"] = free.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if model.has_global_gate():
        lam = model.content_free_lambda(ages_t).detach().cpu().numpy().reshape(-1)
        tau = np.asarray(tau_from_days(np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64)), dtype=np.float64)
        out["gate_surface_model"] = np.exp(-lam.reshape(-1, 1) * tau.reshape(1, -1))
        truth = np.asarray(out["lambda_true_age"], dtype=np.float64).reshape(-1)
        out["gate_surface_oracle"] = np.exp(-truth.reshape(-1, 1) * tau.reshape(1, -1))
    return out


def run_inference(
    *,
    checkpoint: Path,
    scenario: str,
    run_dir: Path,
    data_seed: int,
    device: torch.device,
    shuffle_seed: int,
    batch_size: int,
    max_test_batches: int | None = None,
) -> Path:
    pred_path = run_dir / "predictions.parquet"
    if pred_path.exists():
        raise FileExistsError(pred_path)
    model, _ckpt = load_trained_model(checkpoint, device)
    _train, _val, test_loader, vocab, _info = cached_dtr_loaders(
        scenario, data_seed=data_seed, batch_size=batch_size,
    )
    template, dtr_template, _vocab, _info2, specs, meta, _sdir = select_templates(scenario, data_seed)
    theta0_true = float(meta["theta0"])
    beta_true = float(meta["beta_true"])
    collected = _collect(
        model, test_loader, device,
        shuffle_seed=shuffle_seed,
        signal_ids=_signal_ids(vocab),
        theta0_true=theta0_true,
        beta_true=beta_true,
        max_batches=max_test_batches,
    )
    oracle_age, oracle_lag, oracle_surface = _build_oracle_fns(
        template, dict(_vocab.itos), specs, scenario, theta0_true, beta_true,
    )
    predict_age, predict_lag, predict_surface, _predict_batch = make_predict_fns(
        _EncounterPredictor(model), dtr_template, device, int(_ckpt["n_codes"]),
    )
    mechanism = _parameter_arrays(model, device, theta0_true, beta_true)
    mechanism.update({
        "surface_model": np.asarray(build_surface_grid(predict_surface, SURFACE_AGES, SURFACE_LAGS_DAYS), dtype=np.float64),
        "surface_oracle": np.asarray(build_surface_grid(oracle_surface, SURFACE_AGES, SURFACE_LAGS_DAYS), dtype=np.float64),
        "cf_age_model": np.asarray(np.stack([predict_age(a) for a in CF_AGES], axis=0), dtype=np.float64),
        "cf_age_oracle": np.asarray(np.stack([oracle_age(a) for a in CF_AGES], axis=0), dtype=np.float64),
        "cf_lag_model": np.asarray(np.stack([predict_lag(lag) for lag in CF_LAGS_DAYS], axis=0), dtype=np.float64),
        "cf_lag_oracle": np.asarray(np.stack([oracle_lag(lag) for lag in CF_LAGS_DAYS], axis=0), dtype=np.float64),
        "gate_signal_hat": collected["gate_signal_hat"],
        "gate_signal_true": collected["gate_signal_true"],
        "gate_signal_code": collected["gate_signal_code"],
    })
    if "pi_signal" in collected:
        mechanism["pi_signal"] = collected["pi_signal"]
    np.savez_compressed(run_dir / "mechanism_outputs.npz", **mechanism)
    from atomic.io import write_predictions
    write_predictions(
        pred_path,
        example_id=collected["example_id"],
        patient_id=collected["patient_id"],
        age=collected["age"],
        labels=collected["labels"],
        logits=collected["logits"],
        logits_beta0=collected["logits_beta0"],
        logits_full_age_shuffle=collected["logits_full_age_shuffle"],
        logits_gate_age_shuffle=collected["logits_gate_age_shuffle"],
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_json(run_dir / "inference_done.json", {"predictions": str(pred_path)})
    return pred_path
