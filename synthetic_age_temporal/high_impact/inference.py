"""Immutable predictions and mechanism arrays for high-impact experiments."""
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

from atomic.io import write_predictions
from high_impact.models import HighImpactDTR, build_high_impact


def load_trained_model(path: Path, device: torch.device) -> tuple[HighImpactDTR, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_high_impact(
        cfg,
        int(ckpt["n_codes"]),
        int(ckpt["n_targets"]),
        age_temporal=bool(ckpt["age_temporal"]),
        oracle_theta0=float(ckpt.get("oracle_theta0", cfg.get("oracle_theta0", 0.0))),
        oracle_beta=float(ckpt.get("oracle_beta", cfg.get("oracle_beta", 0.0))),
    )
    model.load_state_dict(ckpt["state_dict"])
    if "oracle_theta0" in ckpt:
        model.set_oracle_(float(ckpt["oracle_theta0"]), float(ckpt["oracle_beta"]))
    model.configure_arm_()
    model.to(device)
    model.eval()
    return model, ckpt


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _logits(out):
    return out["logits"] if isinstance(out, dict) else out


def _signal_ids(vocab) -> dict[int, str]:
    return {int(i): str(t) for i, t in vocab.itos.items() if str(t).startswith("SYN_SIGNAL_")}


def _gather_signal_gates(parts, batch, signal_ids, theta0_true, beta_true, gate_hat, gate_true, gate_code, head_gates):
    g = parts["g"].detach().cpu().numpy()
    g_heads = parts["g_heads"].detach().cpu().numpy() if "g_heads" in parts else None
    tau = batch["enc_tau"].numpy()
    pad = batch["enc_padding_mask"].numpy()
    n_sig = batch["enc_n_signal"].numpy()
    code_ids = batch["enc_code_ids"].numpy()
    code_mask = batch["enc_code_mask"].numpy()
    ages = batch["age"].numpy()
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
                if g_heads is not None:
                    head_gates.append(g_heads[b, m])


@torch.no_grad()
def _collect(model, loader, device, *, shuffle_seed, signal_ids, theta0_true, beta_true, max_batches):
    batches = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        batches.append(batch)
    if not batches:
        raise RuntimeError("Empty loader")
    ages = np.concatenate([b["age"].numpy() for b in batches])
    shuffled = np.random.default_rng(int(shuffle_seed)).permutation(ages)

    labels, logits, logits_b0, logits_full, logits_gate = [], [], [], [], []
    example_ids, patient_ids, age_rows = [], [], []
    gate_hat, gate_true, gate_code, head_gates = [], [], [], []
    u_rows, h_norms = [], []
    head_ablation = {h: [] for h in range(getattr(model, "n_heads", 0))}

    for batch in batches:
        moved = _move(batch, device)
        parts = model(
            enc_code_ids=moved["enc_code_ids"], enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"], enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"], return_parts=True,
        )
        labels.append(moved["labels"].cpu().numpy())
        logits.append(_logits(parts).cpu().numpy())
        age_rows.append(moved["age"].cpu().numpy())
        example_ids.append(batch["example_ids"].numpy())
        patient_ids.extend(batch["patient_ids"])
        _gather_signal_gates(parts, batch, signal_ids, theta0_true, beta_true, gate_hat, gate_true, gate_code, head_gates)
        if "u" in parts and parts["u"].ndim == 3:
            hist = ~moved["enc_padding_mask"]
            u_rows.append(parts["u"][hist].detach().cpu().numpy())
            h_norms.append(parts["h_heads"].norm(dim=-1).detach().cpu().numpy())
            for h in range(model.n_heads):
                ablated = model(
                    enc_code_ids=moved["enc_code_ids"], enc_code_mask=moved["enc_code_mask"],
                    enc_tau=moved["enc_tau"], enc_padding_mask=moved["enc_padding_mask"],
                    age=moved["age"], ablate_head=h,
                )
                head_ablation[h].append(_logits(ablated).cpu().numpy())

    saved = model.zero_all_betas_()
    for batch in batches:
        moved = _move(batch, device)
        out = model(
            enc_code_ids=moved["enc_code_ids"], enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"], enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"],
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
            enc_code_ids=moved["enc_code_ids"], enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"], enc_padding_mask=moved["enc_padding_mask"],
            age=age_full,
        )
        logits_full.append(_logits(out).cpu().numpy())

    pointer = 0
    for batch in batches:
        moved = _move(batch, device)
        width = int(moved["age"].shape[0])
        age_gate = torch.tensor(shuffled[pointer:pointer + width], dtype=torch.float32, device=device)
        pointer += width
        out = model(
            enc_code_ids=moved["enc_code_ids"], enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"], enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"], gate_age=age_gate,
        )
        logits_gate.append(_logits(out).cpu().numpy())

    payload = {
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
    if head_gates:
        payload["gate_signal_heads"] = np.asarray(head_gates, dtype=np.float64)
    if u_rows:
        payload["u_valid"] = np.concatenate(u_rows, axis=0)
        payload["h_head_norms"] = np.concatenate(h_norms, axis=0)
        for h, rows in head_ablation.items():
            payload[f"logits_ablate_head_{h}"] = np.concatenate(rows, axis=0)
    return payload


def _parameter_arrays(model: HighImpactDTR, device, theta0_true, beta_true) -> dict[str, np.ndarray]:
    ages = torch.tensor(SURFACE_AGES, dtype=torch.float32, device=device)
    tau = np.asarray(tau_from_days(np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64)), dtype=np.float64)
    truth = np.asarray(lambda_true(np.asarray(SURFACE_AGES, dtype=np.float64), theta0_true, beta_true), dtype=np.float64)
    out: dict[str, np.ndarray] = {
        "surface_ages": np.asarray(SURFACE_AGES, dtype=np.float64),
        "surface_lags": np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64),
        "cf_ages": np.asarray(CF_AGES, dtype=np.float64),
        "cf_lags": np.asarray(CF_LAGS_DAYS, dtype=np.float64),
        "beta_true": np.asarray(beta_true, dtype=np.float64),
        "theta0_true": np.asarray(theta0_true, dtype=np.float64),
        "architecture": np.asarray(model.variant),
        "lambda_true_age": truth,
        "has_global_gate": np.asarray(1 if model.variant in ("oracle_gate", "multihead_shared") else 0, dtype=np.int64),
    }
    if model.variant == "oracle_gate":
        lam = model.oracle_lambda(ages).detach().cpu().numpy().reshape(-1)
        out["beta"] = np.asarray([model._oracle_beta_value()], dtype=np.float64)
        out["theta"] = model.oracle_theta0.detach().cpu().numpy().astype(np.float64)
        out["content_free_lambda_age"] = lam.astype(np.float64)
        out["gate_surface_model"] = np.exp(-lam.reshape(-1, 1) * tau.reshape(1, -1))
        out["gate_surface_oracle"] = np.exp(-truth.reshape(-1, 1) * tau.reshape(1, -1))
    elif model.variant == "multihead_shared":
        beta = model.base.beta.detach().cpu().numpy().reshape(-1)
        theta = model.base.theta0.detach().cpu().numpy().reshape(-1)
        out["beta"] = beta.astype(np.float64)
        out["theta"] = theta.astype(np.float64)
        if model.age_temporal:
            lam = torch.nn.functional.softplus(model.base.theta0 + model.base.beta * model.base.z_of(ages))
        else:
            lam = torch.nn.functional.softplus(model.base.theta0).expand_as(ages)
        lam_np = lam.detach().cpu().numpy().reshape(-1)
        out["content_free_lambda_age"] = lam_np.astype(np.float64)
        out["gate_surface_model"] = np.exp(-lam_np.reshape(-1, 1) * tau.reshape(1, -1))
        out["gate_surface_oracle"] = np.exp(-truth.reshape(-1, 1) * tau.reshape(1, -1))
        q = model.content_queries.detach().cpu().numpy()
        out["query_vectors"] = q.astype(np.float64)
    else:
        betas = model.head_betas().detach().cpu().numpy().reshape(-1)
        out["beta"] = betas.astype(np.float64)
        out["beta_global"] = model.beta_global.detach().cpu().numpy().astype(np.float64)
        out["delta_h"] = model.delta.detach().cpu().numpy().astype(np.float64)
        out["theta"] = model.base.theta0.detach().cpu().numpy().astype(np.float64)
        z = model.base.z_of(ages).detach().cpu().numpy().reshape(-1)
        theta0 = float(model.base.theta0.detach().cpu())
        lam_h = np.stack([
            np.log1p(np.exp(np.clip(theta0 + float(b) * z, -40, 40))) for b in betas
        ], axis=0)
        out["lambda_h"] = lam_h.astype(np.float64)
        out["gate_surface_heads"] = np.stack([
            np.exp(-lam_h[h].reshape(-1, 1) * tau.reshape(1, -1)) for h in range(len(betas))
        ], axis=0)
        out["gate_surface_oracle"] = np.exp(-truth.reshape(-1, 1) * tau.reshape(1, -1))
        out["query_vectors"] = model.content_queries.detach().cpu().numpy().astype(np.float64)
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
    model, ckpt = load_trained_model(checkpoint, device)
    _, _, test_loader, vocab, _ = cached_dtr_loaders(scenario, data_seed=data_seed, batch_size=batch_size)
    template, dtr_template, vocab2, _info, specs, meta, _sdir = select_templates(scenario, data_seed)
    theta0_true = float(meta["theta0"])
    beta_true = float(meta["beta_true"])
    collected = _collect(
        model, test_loader, device,
        shuffle_seed=shuffle_seed, signal_ids=_signal_ids(vocab),
        theta0_true=theta0_true, beta_true=beta_true, max_batches=max_test_batches,
    )
    oracle_age, oracle_lag, oracle_surface = _build_oracle_fns(
        template, dict(vocab2.itos), specs, scenario, theta0_true, beta_true,
    )
    predict_age, predict_lag, predict_surface, _ = make_predict_fns(
        _EncounterPredictor(model), dtr_template, device, int(ckpt["n_codes"]),
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
    for key in ("gate_signal_heads", "u_valid", "h_head_norms"):
        if key in collected:
            mechanism[key] = collected[key]
    for h in range(getattr(model, "n_heads", 0)):
        key = f"logits_ablate_head_{h}"
        if key in collected:
            mechanism[key] = collected[key]
    # History-head contribution matrix: targets x heads from last linear layer blocks.
    if model.variant.startswith("multihead"):
        w_out = model.base.history_head[2].weight.detach().cpu().numpy()  # [T, 64]
        blocks = w_out.reshape(w_out.shape[0], model.n_heads, model.d_head)
        mechanism["target_head_weight_norm"] = np.linalg.norm(blocks, axis=-1).astype(np.float64)
    np.savez_compressed(run_dir / "mechanism_outputs.npz", **mechanism)
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
