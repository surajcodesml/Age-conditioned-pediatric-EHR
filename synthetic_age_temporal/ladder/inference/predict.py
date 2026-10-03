"""Run a trained ladder model and store immutable predictions plus mechanism arrays.

Metrics are not computed here.
"""
from __future__ import annotations

import json
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
from baselines.common.interface import ModelOutput
from baselines.synthetic.counterfactual_eval import _build_oracle_fns
from baselines.synthetic.data_adapter import make_dtr_baseline_loaders, scenario_dir

from ladder.cached_data import cached_dtr_loaders
from config import lambda_true
from dataset import make_loaders

from ladder.artifacts import write_predictions
from ladder.models.factory import build_model


def select_templates(scenario: str, data_seed: int):
    """Same template rule as analysis.final_results.build_final_paper_artifacts._select_templates."""
    sdir = scenario_dir(scenario, data_seed)
    _, _, test_loader, vocab, info = make_loaders(sdir, batch_size=1)
    template = None
    for batch in test_loader:
        if batch["is_signal"].any():
            template = {k: v for k, v in batch.items()}
            break
    if template is None:
        raise RuntimeError(f"No signal template in {scenario}")
    _, _, dtr_test, _, _ = make_dtr_baseline_loaders(
        scenario, data_seed=data_seed, batch_size=1
    )
    dtr_template = None
    for batch in dtr_test:
        if (~batch["enc_padding_mask"]).any():
            dtr_template = {k: v for k, v in batch.items()}
            break
    if dtr_template is None:
        raise RuntimeError(f"No encounter template in {scenario}")
    specs = json.loads((sdir / "target_specs.json").read_text())
    meta = json.loads((sdir / "meta.json").read_text())
    return template, dtr_template, vocab, info, specs, meta, sdir


class _EncounterPredictor:
    """Adapter so the existing counterfactual probe can call this module."""

    uses_encounter_batch = True
    name = "ladder"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def predict(self, batch: dict[str, Any]) -> ModelOutput:
        self.model.eval()
        logits = self.model(
            enc_code_ids=batch["enc_code_ids"],
            enc_code_mask=batch["enc_code_mask"],
            enc_tau=batch["enc_tau"],
            enc_padding_mask=batch["enc_padding_mask"],
            age=batch["age"],
            enc_lag_days=batch.get("enc_lag_days"),
        )
        if isinstance(logits, dict):
            logits = logits["logits"]
        return ModelOutput(
            logits=logits,
            patient_repr=logits.new_zeros(logits.size(0), 1),
        )


def load_trained_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
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


def _mechanism_arrays(model: torch.nn.Module, device: torch.device) -> dict[str, np.ndarray]:
    ages = torch.tensor(SURFACE_AGES, dtype=torch.float32, device=device)
    beta = model.beta_param().detach().float().cpu().numpy().reshape(-1)
    theta = model.theta_param().detach().float().cpu().numpy().reshape(-1)
    out: dict[str, np.ndarray] = {
        "beta": beta.astype(np.float64),
        "theta": theta.astype(np.float64),
        "surface_ages": np.asarray(SURFACE_AGES, dtype=np.float64),
        "surface_lags": np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64),
        "cf_ages": np.asarray(CF_AGES, dtype=np.float64),
        "cf_lags": np.asarray(CF_LAGS_DAYS, dtype=np.float64),
    }
    architecture = getattr(model, "architecture", "")
    if architecture == "mixture":
        lam = model.lambda_components(ages).detach().cpu().numpy()  # [A, K]
        out["lambda_k"] = np.asarray(lam, dtype=np.float64).T
        out["lambda_age"] = np.asarray(lam.mean(axis=1), dtype=np.float64)
    elif architecture == "integrated_hazard":
        rho = model.rho_of(ages).detach().cpu().numpy()
        out["rho_age"] = np.asarray(rho, dtype=np.float64)
        out["rho_knots_age"] = model.knots.detach().cpu().numpy().astype(np.float64)
        out["rho_knots"] = model.rho_knots().detach().cpu().numpy().astype(np.float64)
        count = ages.numel()
        age_event = ages.view(count, 1).expand(count, count)
        age_current = ages.view(1, count).expand(count, count)
        integral = model.integrate(age_event, age_current)
        gate = torch.exp(-integral.clamp(0, model.integral_clip))
        gate = torch.where(age_current >= age_event, gate, torch.full_like(gate, float("nan")))
        out["persistence_surface"] = gate.detach().cpu().numpy().astype(np.float64)
    else:
        lam = model.developmental_lambda(ages).detach().cpu().numpy()
        out["lambda_age"] = np.asarray(lam, dtype=np.float64).reshape(-1)
    return out


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


@torch.no_grad()
def _collect_split(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    shuffle_seed: int,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    batches = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batches.append(batch)
    if not batches:
        raise RuntimeError("Empty loader")

    ages = np.concatenate([b["age"].numpy() for b in batches])
    shuffled = np.random.default_rng(int(shuffle_seed)).permutation(ages)

    labels, logits, logits_b0, logits_sh = [], [], [], []
    example_ids, patient_ids, age_rows = [], [], []
    gate_sum = 0.0
    gate_count = 0.0
    mass_sum = 0.0
    mass_count = 0.0
    channel_sum = None
    channel_count = 0.0
    pi_sum = None
    pi_count = 0.0

    def accumulate(parts, batch):
        nonlocal gate_sum, gate_count, mass_sum, mass_count
        nonlocal channel_sum, channel_count, pi_sum, pi_count
        hist = ~batch["enc_padding_mask"]
        if "g" in parts:
            gate_sum += float(parts["g"][hist].sum().cpu())
            gate_count += float(hist.sum().cpu())
        if "M" in parts:
            mass_sum += float(parts["M"].sum().cpu())
            mass_count += float(parts["M"].shape[0])
        if "c" in parts:
            valid = parts["c"][hist]
            if valid.numel():
                summed = valid.sum(dim=0)
                channel_sum = summed if channel_sum is None else channel_sum + summed
                channel_count += float(valid.shape[0])
        if "pi" in parts:
            valid = parts["pi"][hist]
            if valid.numel():
                summed = valid.sum(dim=0)
                pi_sum = summed if pi_sum is None else pi_sum + summed
                pi_count += float(valid.shape[0])

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
        accumulate(parts, moved)
        labels.append(moved["labels"].cpu().numpy())
        logits.append(parts["logits"].cpu().numpy())
        age_rows.append(moved["age"].cpu().numpy())
        example_ids.append(batch["example_ids"].numpy())
        patient_ids.extend(batch["patient_ids"])
        pointer += moved["age"].shape[0]

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
        logits_b0.append(out.cpu().numpy() if torch.is_tensor(out) else out["logits"].cpu().numpy())
    model.restore_betas_(saved)

    pointer = 0
    for batch in batches:
        moved = _move(batch, device)
        width = moved["age"].shape[0]
        age = torch.tensor(shuffled[pointer : pointer + width], dtype=torch.float32, device=device)
        pointer += width
        out = model(
            enc_code_ids=moved["enc_code_ids"],
            enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"],
            enc_padding_mask=moved["enc_padding_mask"],
            age=age,
            enc_lag_days=moved.get("enc_lag_days"),
        )
        logits_sh.append(out.cpu().numpy() if torch.is_tensor(out) else out["logits"].cpu().numpy())

    summaries: dict[str, np.ndarray] = {}
    if gate_count > 0:
        summaries["mean_gate"] = np.asarray(gate_sum / gate_count, dtype=np.float64)
    if mass_count > 0:
        summaries["mean_mass"] = np.asarray(mass_sum / mass_count, dtype=np.float64)
    if channel_sum is not None and channel_count > 0:
        summaries["mean_channel_score"] = (channel_sum / channel_count).detach().cpu().numpy()
    if pi_sum is not None and pi_count > 0:
        summaries["mean_mixture"] = (pi_sum / pi_count).detach().cpu().numpy()

    return {
        "example_id": np.concatenate(example_ids),
        "patient_id": patient_ids,
        "age": np.concatenate(age_rows),
        "labels": np.concatenate(labels),
        "logits": np.concatenate(logits),
        "logits_beta0": np.concatenate(logits_b0),
        "logits_age_shuffle": np.concatenate(logits_sh),
        "summaries": summaries,
    }


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
    """Write predictions.parquet and mechanism_outputs.npz. Does not write metrics."""
    pred_path = run_dir / "predictions.parquet"
    if pred_path.exists():
        raise FileExistsError(pred_path)
    model, ckpt = load_trained_model(checkpoint, device)
    _, _, test_loader, _, _ = cached_dtr_loaders(
        scenario, data_seed=data_seed, batch_size=batch_size
    )
    collected = _collect_split(
        model, test_loader, device, shuffle_seed=shuffle_seed, max_batches=max_test_batches
    )
    template, dtr_template, vocab, _info, specs, meta, _sdir = select_templates(scenario, data_seed)
    theta0_true = float(meta["theta0"])
    beta_true = float(meta["beta_true"])
    oracle_age, oracle_lag, oracle_surface = _build_oracle_fns(
        template, dict(vocab.itos), specs, scenario, theta0_true, beta_true
    )
    from baselines.synthetic.counterfactual_eval import make_predict_fns

    predict_age, predict_lag, predict_surface, _predict_batch = make_predict_fns(
        _EncounterPredictor(model), dtr_template, device, int(ckpt["n_codes"])
    )
    surface_model = build_surface_grid(predict_surface, SURFACE_AGES, SURFACE_LAGS_DAYS)
    surface_oracle = build_surface_grid(oracle_surface, SURFACE_AGES, SURFACE_LAGS_DAYS)
    cf_age_model = np.stack([predict_age(a) for a in CF_AGES], axis=0)
    cf_age_oracle = np.stack([oracle_age(a) for a in CF_AGES], axis=0)
    cf_lag_model = np.stack([predict_lag(lag) for lag in CF_LAGS_DAYS], axis=0)
    cf_lag_oracle = np.stack([oracle_lag(lag) for lag in CF_LAGS_DAYS], axis=0)
    mechanism = _mechanism_arrays(model, device)
    lam_true = lambda_true(np.asarray(SURFACE_AGES, dtype=np.float64), theta0_true, beta_true)
    payload = {
        **mechanism,
        **collected["summaries"],
        "surface_model": np.asarray(surface_model, dtype=np.float64),
        "surface_oracle": np.asarray(surface_oracle, dtype=np.float64),
        "cf_age_model": np.asarray(cf_age_model, dtype=np.float64),
        "cf_age_oracle": np.asarray(cf_age_oracle, dtype=np.float64),
        "cf_lag_model": np.asarray(cf_lag_model, dtype=np.float64),
        "cf_lag_oracle": np.asarray(cf_lag_oracle, dtype=np.float64),
        "lambda_true_age": np.asarray(lam_true, dtype=np.float64),
        "beta_true": np.asarray(beta_true, dtype=np.float64),
        "theta0_true": np.asarray(theta0_true, dtype=np.float64),
        "architecture": np.asarray(getattr(model, "architecture", "")),
    }
    np.savez_compressed(run_dir / "mechanism_outputs.npz", **payload)
    write_predictions(
        pred_path,
        example_id=collected["example_id"],
        patient_id=collected["patient_id"],
        age=collected["age"],
        labels=collected["labels"],
        logits=collected["logits"],
        logits_beta0=collected["logits_beta0"],
        logits_age_shuffle=collected["logits_age_shuffle"],
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_path
