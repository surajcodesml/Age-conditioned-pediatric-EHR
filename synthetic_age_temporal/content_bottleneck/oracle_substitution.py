"""Evaluation-only oracle content substitution (C1/C2) vs C01/D00."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from atomic.inference import load_trained_model
from baselines.common.counterfactual import SURFACE_AGES, SURFACE_LAGS_DAYS, build_surface_grid
from baselines.synthetic.counterfactual_eval import _build_oracle_fns, make_predict_fns
from content_bottleneck.extract import c01_checkpoint
from content_bottleneck.generator_content import (
    build_oracle_logits_with_gates,
    encounter_signal_membership,
    load_weight_matrix,
    signal_id_map,
)
from evaluate import classification_metrics
from ladder.cached_data import cached_dtr_loaders
from ladder.inference.predict import _EncounterPredictor, select_templates


@torch.no_grad()
def _c01_gates_and_membership(model, loader, device, sig_map, n_signals, max_batches):
    ages, labels, gates, memberships, pads, taus, example_ids = [], [], [], [], [], [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        parts = model(
            enc_code_ids=moved["enc_code_ids"],
            enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"],
            enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"],
            return_parts=True,
        )
        g = parts["g"].detach().cpu().numpy()
        pad = batch["enc_padding_mask"].numpy()
        code_ids = batch["enc_code_ids"].numpy()
        code_mask = batch["enc_code_mask"].numpy()
        bsz, n_enc = pad.shape
        mem = np.zeros((bsz, n_enc, n_signals), dtype=np.float64)
        for b in range(bsz):
            for m in range(n_enc):
                if bool(pad[b, m]):
                    continue
                mem[b, m] = encounter_signal_membership(
                    code_ids[b, m], code_mask[b, m], sig_map, n_signals,
                )
        ages.append(batch["age"].numpy())
        labels.append(batch["labels"].numpy())
        gates.append(g)
        memberships.append(mem)
        pads.append(pad)
        taus.append(batch["enc_tau"].numpy())
        example_ids.append(batch["example_ids"].numpy())
    return {
        "age": np.concatenate(ages),
        "labels": np.concatenate(labels),
        "gates": np.concatenate(gates),
        "membership": np.concatenate(memberships),
        "pad": np.concatenate(pads),
        "tau": np.concatenate(taus),
        "example_id": np.concatenate(example_ids),
    }


def _surface_rmse_from_logits_fn(predict_surface, oracle_surface) -> float:
    model_grid = np.asarray(
        build_surface_grid(predict_surface, SURFACE_AGES, SURFACE_LAGS_DAYS), dtype=np.float64,
    )
    oracle_grid = np.asarray(
        build_surface_grid(oracle_surface, SURFACE_AGES, SURFACE_LAGS_DAYS), dtype=np.float64,
    )
    return float(np.sqrt(np.mean((model_grid - oracle_grid) ** 2)))


def run_oracle_substitution(
    *,
    scenario: str,
    seed: int,
    data_seed: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """C1 = oracle content + C01 gate; C2 = oracle content + oracle gate."""
    model, _ = load_trained_model(c01_checkpoint(scenario, seed), device)
    _, _, test_loader, vocab, info = cached_dtr_loaders(
        scenario, data_seed=data_seed, batch_size=32,
    )
    specs = info["specs"]
    meta = info["meta"]
    W, signal_names, _ = load_weight_matrix(specs)
    sig_map = signal_id_map(vocab)
    packed = _c01_gates_and_membership(
        model, test_loader, device, sig_map, len(signal_names), max_batches,
    )
    theta0 = float(meta["theta0"])
    beta_true = float(meta["beta_true"])

    logits_c1 = build_oracle_logits_with_gates(
        ages=packed["age"], enc_tau=packed["tau"], enc_pad=packed["pad"],
        enc_membership=packed["membership"], gates=packed["gates"],
        specs=specs, W=W, scenario=scenario, use_model_gates=True,
        theta0=theta0, beta_true=beta_true,
    )
    logits_c2 = build_oracle_logits_with_gates(
        ages=packed["age"], enc_tau=packed["tau"], enc_pad=packed["pad"],
        enc_membership=packed["membership"], gates=packed["gates"],
        specs=specs, W=W, scenario=scenario, use_model_gates=False,
        theta0=theta0, beta_true=beta_true,
    )
    y = packed["labels"]
    m1 = classification_metrics(y, logits_c1)
    m2 = classification_metrics(y, logits_c2)

    # Surface RMSE: build predict_fns that inject oracle content with chosen gate
    # on the counterfactual template path. For surface, use C2-style exact oracle
    # surface from existing helper as ground truth; approximate C1/C2 surface via
    # template substitution using generator content + gates at (age, lag).
    template, dtr_template, vocab2, _info, specs2, meta2, _sdir = select_templates(scenario, data_seed)
    oracle_age, oracle_lag, oracle_surface = _build_oracle_fns(
        template, dict(vocab2.itos), specs2, scenario, theta0, beta_true,
    )
    # C2 surface should match generator oracle surface nearly exactly.
    surface_c2 = float(np.sqrt(np.mean((
        np.asarray(build_surface_grid(oracle_surface, SURFACE_AGES, SURFACE_LAGS_DAYS))
        - np.asarray(build_surface_grid(oracle_surface, SURFACE_AGES, SURFACE_LAGS_DAYS))
    ) ** 2)))  # identically 0; replaced below by diagnostic grids

    # Build surface by evaluating oracle logits on a fixed history template while
    # sweeping age/lag: reuse dtr_template membership + vary age/tau.
    def _surface_predict(use_model_gates: bool):
        # Extract template tensors once
        pad = dtr_template["enc_padding_mask"].numpy()
        code_ids = dtr_template["enc_code_ids"].numpy()
        code_mask = dtr_template["enc_code_mask"].numpy()
        bsz, n_enc = pad.shape
        assert bsz == 1
        mem = np.zeros((1, n_enc, len(signal_names)), dtype=np.float64)
        for m in range(n_enc):
            if bool(pad[0, m]):
                continue
            mem[0, m] = encounter_signal_membership(
                code_ids[0, m], code_mask[0, m], signal_id_map(vocab2), len(signal_names),
            )
        # Precompute model gates at each surface age with template taus, if needed
        def predict_surface(age: float, lag_days: float) -> np.ndarray:
            from config import tau_from_days
            age_arr = np.asarray([age], dtype=np.float64)
            tau = dtr_template["enc_tau"].numpy().copy()
            # Replace non-pad taus? Counterfactual lag typically shifts signal events.
            # Match existing CF probe: set all historical taus from lag for signal encs.
            lag_tau = float(tau_from_days(np.asarray(lag_days)))
            for m in range(n_enc):
                if bool(pad[0, m]):
                    continue
                if mem[0, m].sum() > 0:
                    tau[0, m] = lag_tau
            if use_model_gates:
                moved = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in dtr_template.items()
                }
                moved["age"] = torch.tensor([age], dtype=torch.float32, device=device)
                # override tau
                moved["enc_tau"] = torch.tensor(tau, dtype=torch.float32, device=device)
                parts = model(
                    enc_code_ids=moved["enc_code_ids"],
                    enc_code_mask=moved["enc_code_mask"],
                    enc_tau=moved["enc_tau"],
                    enc_padding_mask=moved["enc_padding_mask"],
                    age=moved["age"],
                    return_parts=True,
                )
                gates = parts["g"].detach().cpu().numpy()
            else:
                gates = np.ones_like(tau)
            logits = build_oracle_logits_with_gates(
                ages=age_arr, enc_tau=tau, enc_pad=pad, enc_membership=mem,
                gates=gates, specs=specs, W=W, scenario=scenario,
                use_model_gates=use_model_gates, theta0=theta0, beta_true=beta_true,
            )
            # Return probabilities for surface comparison (oracle_surface returns probs)
            x = np.clip(logits[0], -30, 30)
            return 1.0 / (1.0 + np.exp(-x))
        return predict_surface

    pred_c1 = _surface_predict(True)
    pred_c2 = _surface_predict(False)
    surface_rmse_c1 = _surface_rmse_from_logits_fn(pred_c1, oracle_surface)
    surface_rmse_c2 = _surface_rmse_from_logits_fn(pred_c2, oracle_surface)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "scenario": scenario,
        "seed": int(seed),
        "C1": {
            "bce": m1["bce"], "auroc": m1["micro_auroc"], "auprc": m1["micro_auprc"],
            "surface_rmse": surface_rmse_c1,
        },
        "C2": {
            "bce": m2["bce"], "auroc": m2["micro_auroc"], "auprc": m2["micro_auprc"],
            "surface_rmse": surface_rmse_c2,
        },
        "delta_C1_minus_C2": {
            "bce": m1["bce"] - m2["bce"],
            "auroc": m1["micro_auroc"] - m2["micro_auroc"],
            "auprc": m1["micro_auprc"] - m2["micro_auprc"],
            "surface_rmse": surface_rmse_c1 - surface_rmse_c2,
        },
        "n_rows": int(y.shape[0]),
    }
