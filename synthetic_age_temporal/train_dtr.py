"""Train / evaluate Developmental Temporal Retrieval (DTR)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from config import (
    DATA_SEED,
    PROBE_AGES,
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    lambda_true,
    relevance,
    tau_from_days,
)
from dataset_dtr import make_dtr_loaders
from evaluate import classification_metrics
from model_dtr import build_dtr, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def predict(model, loader, device, return_parts: bool = False):
    model.eval()
    ys, logits, parts = [], [], []
    ages, pids = [], []
    for batch in loader:
        batch = _move(batch, device)
        out = model(
            enc_code_ids=batch["enc_code_ids"],
            enc_code_mask=batch["enc_code_mask"],
            enc_tau=batch["enc_tau"],
            enc_padding_mask=batch["enc_padding_mask"],
            age=batch["age"],
            return_parts=return_parts,
        )
        if return_parts:
            logits.append(out["logits"].cpu().numpy())
            parts.append(
                {
                    "history_logit": out["history_logit"].cpu().numpy(),
                    "age_logit": out["age_logit"].cpu().numpy(),
                }
            )
        else:
            logits.append(out.cpu().numpy())
        ys.append(batch["labels"].cpu().numpy())
        ages.append(batch["age"].cpu().numpy())
        pids.extend(batch["patient_ids"])
    result = {
        "y": np.concatenate(ys),
        "logits": np.concatenate(logits),
        "age": np.concatenate(ages),
        "patient_ids": pids,
    }
    if return_parts:
        result["history_logit"] = np.concatenate([p["history_logit"] for p in parts])
        result["age_logit"] = np.concatenate([p["age_logit"] for p in parts])
    return result


def evaluate(model, loader, device) -> dict[str, float]:
    pred = predict(model, loader, device)
    return classification_metrics(pred["y"], pred["logits"])


@torch.no_grad()
def ablations(model, loader, device) -> dict[str, Any]:
    base_pred = predict(model, loader, device)
    base = classification_metrics(base_pred["y"], base_pred["logits"])
    ages = base_pred["age"]

    def run(age_arr=None, beta0=False, lag_shuffle=False):
        model.eval()
        ys, logits = [], []
        saved = None
        if beta0:
            saved = model.zero_all_betas_()
        # For lag shuffle we permute enc_tau within each example
        for batch in loader:
            batch = _move(batch, device)
            age = batch["age"]
            if age_arr is not None:
                # Map by iterating — use shuffled ages matching batch size from global pool
                pass
            tau = batch["enc_tau"]
            if lag_shuffle:
                tau = tau.clone()
                for i in range(tau.size(0)):
                    valid = ~batch["enc_padding_mask"][i]
                    idx = torch.where(valid)[0]
                    if len(idx) > 1:
                        perm = idx[torch.randperm(len(idx), device=device)]
                        tau[i, idx] = batch["enc_tau"][i, perm]
            age_use = age
            out = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=tau,
                enc_padding_mask=batch["enc_padding_mask"],
                age=age_use,
            )
            ys.append(batch["labels"].cpu().numpy())
            logits.append(out.cpu().numpy())
        if saved is not None:
            model.restore_betas_(saved)
        return classification_metrics(np.concatenate(ys), np.concatenate(logits))

    # Age shuffle: permute ages across the loader in one pass
    def run_age_shuffle():
        model.eval()
        # collect all ages then shuffle assignment by patient-example order
        all_ages = []
        batches = []
        for batch in loader:
            batches.append(_move(batch, device))
            all_ages.append(batch["age"].cpu().numpy())
        flat = np.concatenate(all_ages)
        rng = np.random.default_rng(0)
        shuf = rng.permutation(flat)
        ptr = 0
        ys, logits = [], []
        for batch in batches:
            bsz = batch["age"].size(0)
            age = torch.tensor(shuf[ptr : ptr + bsz], dtype=torch.float32, device=device)
            ptr += bsz
            out = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=age,
            )
            ys.append(batch["labels"].cpu().numpy())
            logits.append(out.cpu().numpy())
        return classification_metrics(np.concatenate(ys), np.concatenate(logits))

    def run_constant_age(a=9.0):
        model.eval()
        ys, logits = [], []
        for batch in loader:
            batch = _move(batch, device)
            age = torch.full_like(batch["age"], float(a))
            out = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=age,
            )
            ys.append(batch["labels"].cpu().numpy())
            logits.append(out.cpu().numpy())
        return classification_metrics(np.concatenate(ys), np.concatenate(logits))

    sh = run_age_shuffle()
    b0 = run(beta0=True)
    const = run_constant_age(9.0)
    lag = run(lag_shuffle=True)
    return {
        "normal": base,
        "shuffle_age": sh,
        "beta0": b0,
        "constant_age": const,
        "lag_shuffle": lag,
        "delta_bce_shuffle_age": sh["bce"] - base["bce"],
        "delta_bce_beta0": b0["bce"] - base["bce"],
        "delta_bce_constant_age": const["bce"] - base["bce"],
        "delta_bce_lag_shuffle": lag["bce"] - base["bce"],
        "delta_auroc_shuffle": base["micro_auroc"] - sh["micro_auroc"],
        "delta_auroc_beta0": base["micro_auroc"] - b0["micro_auroc"],
    }


def recovery(model, beta_true: float, theta0_true: float = 0.0) -> dict[str, Any]:
    gate = model.gate
    device = gate.theta0.device
    ages = np.asarray(SURFACE_AGES, dtype=np.float64)
    lam_true = lambda_true(ages, theta0_true, beta_true)
    lam_learned = np.array(
        [
            float(gate.lambda_of(torch.tensor([float(a)], device=device))[0].item())
            for a in ages
        ]
    )
    rmse = float(np.sqrt(np.mean((lam_learned - lam_true) ** 2)))
    corr = (
        float(np.corrcoef(lam_learned, lam_true)[0, 1])
        if np.std(lam_learned) > 1e-8 and np.std(lam_true) > 1e-8
        else float("nan")
    )
    errs = []
    for a, lam_l in zip(ages, lam_learned):
        for d in SURFACE_LAGS_DAYS:
            t = float(tau_from_days(d))
            errs.append((np.exp(-lam_l * t) - relevance(a, t, theta0_true, beta_true)) ** 2)
    beta_hat = float(gate.beta.detach().cpu())
    theta0_hat = float(gate.theta0.detach().cpu())
    probe = {
        str(a): float(gate.lambda_of(torch.tensor([float(a)], device=device))[0].item())
        for a in PROBE_AGES
    }
    return {
        "beta_hat": beta_hat,
        "beta_true": beta_true,
        "theta0_hat": theta0_hat,
        "theta0_true": theta0_true,
        "sign_match": (
            bool(np.sign(beta_hat) == np.sign(beta_true) and abs(beta_hat) > 1e-3)
            if abs(beta_true) > 0
            else abs(beta_hat) < 0.2
        ),
        "RMSE_lambda": rmse,
        "corr_lambda": corr,
        "RMSE_surface": float(np.sqrt(np.mean(errs))),
        "lambda_true_by_age": {str(a): float(v) for a, v in zip(ages, lam_true)},
        "lambda_learned_by_age": {str(a): float(v) for a, v in zip(ages, lam_learned)},
        "probe_lambda_learned": probe,
    }


def patient_bootstrap_deltas(
    y_at: np.ndarray,
    logits_at: np.ndarray,
    y_to: np.ndarray,
    logits_to: np.ndarray,
    patient_ids: list[str],
    abl_logits: dict[str, np.ndarray],
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Patient-level bootstrap CIs for primary deltas (not model-seed uncertainty)."""
    rng = np.random.default_rng(seed)
    # unique patients
    pids = np.asarray(patient_ids)
    uniq = np.unique(pids)
    # map patient -> row indices
    groups = {p: np.where(pids == p)[0] for p in uniq}

    def metrics_on(idx):
        return classification_metrics(y_at[idx], logits_at[idx]), classification_metrics(
            y_to[idx], logits_to[idx]
        )

    def bce(y, logits):
        return classification_metrics(y, logits)["bce"]

    samples = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[p] for p in drawn])
        at_m, to_m = metrics_on(idx)
        base_bce = bce(y_at[idx], logits_at[idx])
        sh_bce = bce(y_at[idx], abl_logits["shuffle_age"][idx])
        b0_bce = bce(y_at[idx], abl_logits["beta0"][idx])
        samples.append(
            {
                "delta_auroc": at_m["micro_auroc"] - to_m["micro_auroc"],
                "delta_auprc": at_m["micro_auprc"] - to_m["micro_auprc"],
                "delta_bce_shuffle": sh_bce - base_bce,
                "delta_bce_beta0": b0_bce - base_bce,
            }
        )

    def summarize(key):
        vals = np.array([s[key] for s in samples], dtype=np.float64)
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci95_lo": float(np.percentile(vals, 2.5)),
            "ci95_hi": float(np.percentile(vals, 97.5)),
        }

    return {k: summarize(k) for k in samples[0].keys()}


@torch.no_grad()
def collect_ablation_logits(model, loader, device) -> dict[str, np.ndarray]:
    """Logits under shuffle-age and beta0 for bootstrap pairing."""
    # shuffle
    batches = []
    all_ages = []
    for batch in loader:
        b = _move(batch, device)
        batches.append(b)
        all_ages.append(b["age"].cpu().numpy())
    flat = np.concatenate(all_ages)
    shuf = np.random.default_rng(0).permutation(flat)
    ptr = 0
    sh_logits = []
    for batch in batches:
        bsz = batch["age"].size(0)
        age = torch.tensor(shuf[ptr : ptr + bsz], dtype=torch.float32, device=device)
        ptr += bsz
        out = model(
            enc_code_ids=batch["enc_code_ids"],
            enc_code_mask=batch["enc_code_mask"],
            enc_tau=batch["enc_tau"],
            enc_padding_mask=batch["enc_padding_mask"],
            age=age,
        )
        sh_logits.append(out.cpu().numpy())
    saved = model.zero_all_betas_()
    b0_logits = []
    for batch in batches:
        out = model(
            enc_code_ids=batch["enc_code_ids"],
            enc_code_mask=batch["enc_code_mask"],
            enc_tau=batch["enc_tau"],
            enc_padding_mask=batch["enc_padding_mask"],
            age=batch["age"],
        )
        b0_logits.append(out.cpu().numpy())
    model.restore_betas_(saved)
    return {
        "shuffle_age": np.concatenate(sh_logits),
        "beta0": np.concatenate(b0_logits),
    }


def dtr_gate(at: dict, to: dict, beta_true: float) -> dict[str, Any]:
    d_auroc = at["test"]["micro_auroc"] - to["test"]["micro_auroc"]
    d_auprc = at["test"]["micro_auprc"] - to["test"]["micro_auprc"]
    d_shuf = at["ablations"]["delta_bce_shuffle_age"]
    d_b0 = at["ablations"]["delta_bce_beta0"]
    if abs(beta_true) < 1e-8:
        # S0/S1: expect inert interaction
        passed = abs(at["beta_hat"]) < 0.3 and abs(d_b0) < 0.02 and abs(d_shuf) < 0.03
        sign_ok = abs(at["beta_hat"]) < 0.3
    else:
        sign_ok = bool(np.sign(at["beta_hat"]) == np.sign(beta_true) and abs(at["beta_hat"]) > 0.1)
        pred_ok = (d_auroc >= 0.01) or (d_auprc >= 0.01)
        abl_ok = (d_shuf > 0.01) and (d_b0 > 0.005)
        lam_ok = (at["recovery"].get("corr_lambda") or 0) > 0.5
        passed = bool(sign_ok and pred_ok and abl_ok and lam_ok)
    return {
        "passed": passed,
        "sign_ok": sign_ok if abs(beta_true) > 0 else abs(at["beta_hat"]) < 0.3,
        "delta_auroc": d_auroc,
        "delta_auprc": d_auprc,
        "delta_bce_shuffle": d_shuf,
        "delta_bce_beta0": d_b0,
        "corr_lambda": at["recovery"].get("corr_lambda"),
        "RMSE_surface": at["recovery"].get("RMSE_surface"),
        "beta_hat": at["beta_hat"],
    }


def train_dtr(
    *,
    age_temporal: bool,
    scenario_dir: Path,
    run_dir: Path,
    beta_true: float,
    theta0_true: float = 0.0,
    aggregation: str = "weighted_mean_plus_log_mass",
    interaction_only: bool = True,
    content_persistence: bool = False,
    multi_query_k: int = 1,
    max_epochs: int = 40,
    patience: int = 10,
    batch_size: int = 64,
    d_model: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cuda",
    model_class = None,
) -> dict[str, Any]:
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    specs = json.loads((scenario_dir / "target_specs.json").read_text())
    target_idx = (
        [i for i, s in enumerate(specs) if s["mechanism"] == "interaction"]
        if interaction_only
        else None
    )
    train_loader, val_loader, test_loader, vocab, info = make_dtr_loaders(
        scenario_dir, batch_size=batch_size, target_idx=target_idx
    )
    if model_class is None:
        from model_dtr import DevelopmentalTemporalRetrieval
        model_class = DevelopmentalTemporalRetrieval
    
    model = model_class(
        n_codes=info["n_codes"],
        n_targets=info["n_targets"],
        d_model=d_model,
        age_temporal=age_temporal,
        aggregation=aggregation,
        content_persistence=content_persistence,
        multi_query_K=multi_query_k,
    ).to(dev)

    temporal_ids = {id(p) for p in model.age_parameters()}
    decay, temporal = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (temporal if id(p) in temporal_ids else decay).append(p)
    opt = AdamW(
        [
            {"params": decay, "lr": lr, "weight_decay": 1e-2},
            {"params": temporal, "lr": 10 * lr, "weight_decay": 0.0},
        ]
    )
    loss_fn = BCEWithLogitsLoss()
    run_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    best_state = None
    patience_left = patience
    history = []
    arm = "dtr_age_temporal" if age_temporal else "dtr_temporal_only"
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        grad_theta = grad_beta = None
        for batch in train_loader:
            batch = _move(batch, dev)
            opt.zero_grad(set_to_none=True)
            logits = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=batch["age"],
            )
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            if model.gate.theta0.grad is not None:
                grad_theta = float(model.gate.theta0.grad.norm().cpu())
            if model.gate.beta.requires_grad and model.gate.beta.grad is not None:
                grad_beta = float(model.gate.beta.grad.norm().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * batch["labels"].size(0)
            n += batch["labels"].size(0)
        val = evaluate(model, val_loader, dev)
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_bce": val["bce"],
            "val_auroc": val["micro_auroc"],
            "val_auprc": val["micro_auprc"],
            "beta": float(model.gate.beta.detach().cpu()),
            "theta0": float(model.gate.theta0.detach().cpu()),
            "grad_theta0": grad_theta,
            "grad_beta": grad_beta,
        }
        history.append(row)
        print(
            f"{arm}/{aggregation} ep{epoch} "
            f"train={row['train_loss']:.4f} val_auroc={val['micro_auroc']:.3f} "
            f"beta={row['beta']:.3f}",
            flush=True,
        )
        if val["bce"] < best - 1e-5:
            best = val["bce"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state:
        model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_codes": info["n_codes"],
            "n_targets": info["n_targets"],
            "d_model": d_model,
            "aggregation": aggregation,
            "age_temporal": age_temporal,
            "content_persistence": content_persistence,
            "multi_query_K": multi_query_k,
            "vocab_stoi": vocab.stoi,
        },
        run_dir / "model.pt",
    )

    test = evaluate(model, test_loader, dev)
    abl = ablations(model, test_loader, dev)
    rec = recovery(model, beta_true, theta0_true)
    # weight magnitude diagnostics
    mag = {"u_abs_mean": None, "g_mean": None, "w_mean": None, "M_mean": None}
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = _move(batch, dev)
            _ = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=batch["age"],
            )
            c = model._cache
            mag = {
                "u_abs_mean": float(c["u"].abs().mean().cpu()),
                "exp_u_mean": float(torch.exp(c["u"].clamp(max=20)).mean().cpu()),
                "g_mean": float(c["g"].mean().cpu()),
                "w_mean": float(c["w"].mean().cpu()),
                "M_mean": float(c["M"].mean().cpu()),
            }
            break

    result = {
        "arm": arm,
        "age_temporal": age_temporal,
        "aggregation": aggregation,
        "interaction_only": interaction_only,
        "content_persistence": content_persistence,
        "multi_query_k": multi_query_k,
        "n_params": count_parameters(model),
        "test": test,
        "ablations": abl,
        "recovery": rec,
        "history": history,
        "beta_hat": rec["beta_hat"],
        "theta0_hat": rec["theta0_hat"],
        "beta_true": beta_true,
        "weight_magnitudes": mag,
        "scenario_dir": str(scenario_dir),
        "data_seed": DATA_SEED,
        "model_seed": seed,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def load_dtr_model(run_dir: Path, device: torch.device):
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    model = build_dtr(
        age_temporal=ckpt["age_temporal"],
        n_codes=ckpt["n_codes"],
        n_targets=ckpt["n_targets"],
        d_model=ckpt["d_model"],
        aggregation=ckpt["aggregation"],
        content_persistence=ckpt.get("content_persistence", False),
        multi_query_K=ckpt.get("multi_query_K", 1),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--age-temporal", action="store_true")
    ap.add_argument("--temporal-only", action="store_true")
    ap.add_argument("--aggregation", default="weighted_mean_plus_log_mass")
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--beta-true", type=float, default=-2.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--interaction-only", action="store_true", default=True)
    ap.add_argument("--all-labels", action="store_true")
    ap.add_argument("--content-persistence", action="store_true")
    ap.add_argument("--multi-query-k", type=int, default=1)
    args = ap.parse_args()
    age_temporal = True
    if args.temporal_only:
        age_temporal = False
    if args.age_temporal:
        age_temporal = True
    train_dtr(
        age_temporal=age_temporal,
        scenario_dir=args.scenario_dir,
        run_dir=args.run_dir,
        beta_true=args.beta_true,
        aggregation=args.aggregation,
        interaction_only=not args.all_labels,
        content_persistence=args.content_persistence,
        multi_query_k=args.multi_query_k,
        max_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        d_model=args.d_model,
    )


if __name__ == "__main__":
    main()
