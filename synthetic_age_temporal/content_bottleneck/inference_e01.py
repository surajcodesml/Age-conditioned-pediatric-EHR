"""E01 inference + target×signal content recovery diagnostics."""
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
from content_bottleneck.generator_content import (
    encounter_signal_membership,
    load_weight_matrix,
    signal_id_map,
)
from content_bottleneck.models_e01 import TargetConditionedDTR, build_e01
from ladder.artifacts import write_json
from ladder.cached_data import cached_dtr_loaders
from ladder.inference.predict import _EncounterPredictor, select_templates
from atomic.io import write_predictions


def load_trained_e01(path: Path, device: torch.device) -> tuple[TargetConditionedDTR, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_e01(
        ckpt["config"], int(ckpt["n_codes"]), int(ckpt["n_targets"]),
        age_temporal=bool(ckpt["age_temporal"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.configure_arm_()
    model.to(device)
    model.eval()
    return model, ckpt


def _move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _logits(out):
    return out["logits"] if isinstance(out, dict) else out


@torch.no_grad()
def _collect(model, loader, device, *, shuffle_seed, signal_ids, sig_map, n_signals, theta0, beta, max_batches):
    batches = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        batches.append(batch)
    ages = np.concatenate([b["age"].numpy() for b in batches])
    shuffled = np.random.default_rng(int(shuffle_seed)).permutation(ages)

    labels, logits, logits_b0, logits_full, logits_gate = [], [], [], [], []
    example_ids, patient_ids, age_rows = [], [], []
    gate_hat, gate_true, gate_code = [], [], []
    # learned evidence accumulators: sum and count per (target, signal)
    n_t = model.n_targets
    evid_sum = np.zeros((n_t, n_signals), dtype=np.float64)
    evid_count = np.zeros((n_signals,), dtype=np.float64)
    bg_abs = []

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

        g = parts["g"].cpu().numpy()
        ae = (parts["a"] * parts["e"]).cpu().numpy()  # [B,M,T] pre-gate evidence
        pad = batch["enc_padding_mask"].numpy()
        code_ids = batch["enc_code_ids"].numpy()
        code_mask = batch["enc_code_mask"].numpy()
        tau = batch["enc_tau"].numpy()
        ages_b = batch["age"].numpy()
        n_sig = batch["enc_n_signal"].numpy()
        for b in range(g.shape[0]):
            lam = float(lambda_true(float(ages_b[b]), theta0, beta))
            for m in range(g.shape[1]):
                if bool(pad[b, m]):
                    continue
                mem = encounter_signal_membership(code_ids[b, m], code_mask[b, m], sig_map, n_signals)
                if mem.sum() == 0:
                    bg_abs.append(float(np.mean(np.abs(ae[b, m]))))
                    continue
                for j in range(n_signals):
                    if mem[j] <= 0:
                        continue
                    evid_sum[:, j] += ae[b, m]
                    evid_count[j] += 1.0
                if int(n_sig[b, m]) > 0:
                    # signal gate RMSE using mean over codes present
                    names = []
                    for k in range(code_ids.shape[2]):
                        if not bool(code_mask[b, m, k]):
                            continue
                        tok = signal_ids.get(int(code_ids[b, m, k]))
                        if tok is not None:
                            names.append(tok)
                    if names:
                        truth = float(np.exp(-lam * float(tau[b, m])))
                        hat = float(g[b, m])
                        for name in names:
                            gate_hat.append(hat)
                            gate_true.append(truth)
                            gate_code.append(name)

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

    evid_mean = evid_sum / np.maximum(evid_count.reshape(1, -1), 1.0)
    return {
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
        "learned_evidence": evid_mean,
        "evidence_count": evid_count,
        "background_abs_evidence_mean": float(np.mean(bg_abs)) if bg_abs else 0.0,
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
    pred_path = run_dir / "predictions.parquet"
    if pred_path.exists():
        raise FileExistsError(pred_path)
    model, ckpt = load_trained_e01(checkpoint, device)
    _, _, test_loader, vocab, info = cached_dtr_loaders(scenario, data_seed=data_seed, batch_size=batch_size)
    template, dtr_template, vocab2, _info, specs, meta, _sdir = select_templates(scenario, data_seed)
    theta0 = float(meta["theta0"])
    beta_true = float(meta["beta_true"])
    W, signal_names, mechs = load_weight_matrix(specs)
    sig_map = signal_id_map(vocab)
    signal_ids = {int(i): str(t) for i, t in vocab.itos.items() if str(t).startswith("SYN_SIGNAL_")}
    collected = _collect(
        model, test_loader, device, shuffle_seed=shuffle_seed, signal_ids=signal_ids,
        sig_map=sig_map, n_signals=len(signal_names), theta0=theta0, beta=beta_true,
        max_batches=max_test_batches,
    )
    oracle_age, oracle_lag, oracle_surface = _build_oracle_fns(
        template, dict(vocab2.itos), specs, scenario, theta0, beta_true,
    )
    predict_age, predict_lag, predict_surface, _ = make_predict_fns(
        _EncounterPredictor(model), dtr_template, device, int(ckpt["n_codes"]),
    )
    ages = torch.tensor(SURFACE_AGES, dtype=torch.float32, device=device)
    if model.age_temporal:
        lam = torch.nn.functional.softplus(model.base.theta0 + model.base.beta * model.base.z_of(ages))
    else:
        lam = torch.nn.functional.softplus(model.base.theta0).expand(len(SURFACE_AGES))
    lam_np = lam.detach().cpu().numpy().reshape(-1)
    tau = np.asarray(tau_from_days(np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64)))
    truth = np.asarray(lambda_true(np.asarray(SURFACE_AGES, dtype=np.float64), theta0, beta_true))

    learned = collected["learned_evidence"]
    residual = learned - W
    flat_l, flat_w = learned.ravel(), W.ravel()
    pearson = float(np.corrcoef(flat_l, flat_w)[0, 1]) if flat_w.std() > 0 and flat_l.std() > 0 else None
    from scipy.stats import spearmanr
    spearman = float(spearmanr(flat_l, flat_w).correlation) if flat_w.std() > 0 else None
    per_target_corr = []
    for t in range(W.shape[0]):
        if W[t].std() == 0 or learned[t].std() == 0:
            per_target_corr.append(None)
        else:
            per_target_corr.append(float(np.corrcoef(learned[t], W[t])[0, 1]))
    sign_agree = float(np.mean(np.sign(learned) == np.sign(W))) if ((W > 0).any() and (W < 0).any()) else None

    mechanism = {
        "surface_ages": np.asarray(SURFACE_AGES, dtype=np.float64),
        "surface_lags": np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64),
        "cf_ages": np.asarray(CF_AGES, dtype=np.float64),
        "cf_lags": np.asarray(CF_LAGS_DAYS, dtype=np.float64),
        "beta": model.base.beta.detach().cpu().numpy().astype(np.float64),
        "theta": model.base.theta0.detach().cpu().numpy().astype(np.float64),
        "beta_true": np.asarray(beta_true, dtype=np.float64),
        "theta0_true": np.asarray(theta0, dtype=np.float64),
        "content_free_lambda_age": lam_np.astype(np.float64),
        "lambda_true_age": truth.astype(np.float64),
        "gate_surface_model": np.exp(-lam_np.reshape(-1, 1) * tau.reshape(1, -1)),
        "gate_surface_oracle": np.exp(-truth.reshape(-1, 1) * tau.reshape(1, -1)),
        "surface_model": np.asarray(build_surface_grid(predict_surface, SURFACE_AGES, SURFACE_LAGS_DAYS), dtype=np.float64),
        "surface_oracle": np.asarray(build_surface_grid(oracle_surface, SURFACE_AGES, SURFACE_LAGS_DAYS), dtype=np.float64),
        "cf_age_model": np.asarray(np.stack([predict_age(a) for a in CF_AGES], axis=0), dtype=np.float64),
        "cf_age_oracle": np.asarray(np.stack([oracle_age(a) for a in CF_AGES], axis=0), dtype=np.float64),
        "cf_lag_model": np.asarray(np.stack([predict_lag(lag) for lag in CF_LAGS_DAYS], axis=0), dtype=np.float64),
        "cf_lag_oracle": np.asarray(np.stack([oracle_lag(lag) for lag in CF_LAGS_DAYS], axis=0), dtype=np.float64),
        "gate_signal_hat": collected["gate_signal_hat"],
        "gate_signal_true": collected["gate_signal_true"],
        "gate_signal_code": collected["gate_signal_code"],
        "W_true": W.astype(np.float64),
        "learned_evidence": learned.astype(np.float64),
        "evidence_residual": residual.astype(np.float64),
        "signal_names": np.asarray(signal_names),
        "mechanisms": np.asarray(mechs),
        "content_matrix_rmse": np.asarray(np.sqrt(np.mean(residual ** 2)), dtype=np.float64),
        "content_pearson": np.asarray(pearson if pearson is not None else np.nan),
        "content_spearman": np.asarray(spearman if spearman is not None else np.nan),
        "content_sign_agreement": np.asarray(sign_agree if sign_agree is not None else np.nan),
        "content_per_target_corr": np.asarray([c if c is not None else np.nan for c in per_target_corr]),
        "background_abs_evidence_mean": np.asarray(collected["background_abs_evidence_mean"]),
        "architecture": np.asarray("target_conditioned"),
        "has_global_gate": np.asarray(1, dtype=np.int64),
    }
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
    write_json(run_dir / "content_recovery.json", {
        "matrix_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "pearson": pearson,
        "spearman": spearman,
        "sign_agreement": sign_agree,
        "per_target_corr": per_target_corr,
        "background_abs_evidence_mean": collected["background_abs_evidence_mean"],
    })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_json(run_dir / "inference_done.json", {"predictions": str(pred_path)})
    return pred_path
