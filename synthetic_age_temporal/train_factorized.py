#!/usr/bin/env python3
"""Train / evaluate factorized M1–M3 models with mechanism diagnostics."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from config import (
    DEFAULT_OUTPUT_DIR,
    PROBE_AGES,
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    lambda_true,
    relevance,
    tau_from_days,
)
from dataset import make_loaders
from evaluate import classification_metrics, predict as _unused  # noqa: F401
from model_factorized import build_factorized, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, logits = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(
            code_ids=batch["code_ids"],
            type_ids=batch["type_ids"],
            tau=batch["tau"],
            padding_mask=batch["padding_mask"],
            is_query=batch["is_query"],
            age=batch["age"],
            lag_days=batch["lag_days"],
        )
        ys.append(batch["labels"].cpu().numpy())
        logits.append(out.cpu().numpy())
    return np.concatenate(ys), np.concatenate(logits)


def evaluate(model, loader, device):
    y, logits = predict(model, loader, device)
    return classification_metrics(y, logits)


@torch.no_grad()
def ablations(model, loader, device) -> dict[str, Any]:
    y, logits = predict(model, loader, device)
    base = classification_metrics(y, logits)
    ages = np.concatenate([b["age"].numpy() for b in loader])
    shuffled = np.random.default_rng(0).permutation(ages)

    def run(age_arr=None, beta0=False):
        saved = None
        if beta0:
            saved = model.zero_all_betas_()
        ys, outs = [], []
        off = 0
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            bsz = batch["age"].size(0)
            if age_arr is not None:
                batch["age"] = torch.tensor(
                    age_arr[off : off + bsz], dtype=torch.float32, device=device
                )
                off += bsz
            out = model(
                code_ids=batch["code_ids"],
                type_ids=batch["type_ids"],
                tau=batch["tau"],
                padding_mask=batch["padding_mask"],
                is_query=batch["is_query"],
                age=batch["age"],
                lag_days=batch["lag_days"],
            )
            ys.append(batch["labels"].cpu().numpy())
            outs.append(out.cpu().numpy())
        if saved is not None:
            model.restore_betas_(saved)
        return classification_metrics(np.concatenate(ys), np.concatenate(outs))

    sh = run(shuffled)
    b0 = run(None, beta0=True)
    const = run(np.full_like(ages, 9.0))
    return {
        "normal": base,
        "shuffle_age": sh,
        "beta0": b0,
        "constant_age": const,
        "delta_bce_shuffle_age": sh["bce"] - base["bce"],
        "delta_bce_beta0": b0["bce"] - base["bce"],
        "delta_bce_constant_age": const["bce"] - base["bce"],
        "delta_auroc_shuffle": base["micro_auroc"] - sh["micro_auroc"],
        "delta_auroc_beta0": base["micro_auroc"] - b0["micro_auroc"],
    }


def recovery(model, beta_true: float, theta0_true: float) -> dict[str, Any]:
    gate = model.gate if hasattr(model, "gate") else model.inner.gate
    device = gate.theta0.device
    ages = np.asarray(SURFACE_AGES, dtype=np.float64)
    lam_true = lambda_true(ages, theta0_true, beta_true)
    lam_learned = []
    for a in ages:
        lam_learned.append(
            float(gate.lambda_of(torch.tensor([float(a)], device=device))[0].item())
        )
    lam_learned = np.asarray(lam_learned)
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
    return {
        "beta_hat": beta_hat,
        "beta_true": beta_true,
        "theta0_hat": theta0_hat,
        "theta0_true": theta0_true,
        "sign_match": bool(np.sign(beta_hat) == np.sign(beta_true) and abs(beta_hat) > 1e-3)
        if abs(beta_true) > 0
        else abs(beta_hat) < 0.2,
        "RMSE_lambda": rmse,
        "corr_lambda": corr,
        "RMSE_surface": float(np.sqrt(np.mean(errs))),
        "lambda_true_by_age": {str(a): float(v) for a, v in zip(ages, lam_true)},
        "lambda_learned_by_age": {str(a): float(v) for a, v in zip(ages, lam_learned)},
        "probe_lambda_learned": {
            str(a): float(gate.lambda_of(torch.tensor([float(a)], device=device))[0])
            for a in PROBE_AGES
        },
    }


def m1_gate(
    age_temporal_metrics: dict, temporal_only_metrics: dict, beta_true: float
) -> dict[str, Any]:
    """M1 decision gate from the user spec."""
    at = age_temporal_metrics
    to = temporal_only_metrics
    d_auroc = at["test"]["micro_auroc"] - to["test"]["micro_auroc"]
    d_auprc = at["test"]["micro_auprc"] - to["test"]["micro_auprc"]
    d_shuf = at["ablations"]["delta_bce_shuffle_age"]
    d_b0 = at["ablations"]["delta_bce_beta0"]
    sign_ok = bool(at["recovery"].get("sign_match"))
    pred_ok = (d_auroc >= 0.01) or (d_auprc >= 0.01)
    abl_ok = (d_shuf > 0.01) and (d_b0 > 0.005)
    lam_ok = (at["recovery"].get("corr_lambda") or 0) > 0.5
    passed = bool(sign_ok and pred_ok and abl_ok and lam_ok)
    return {
        "passed": passed,
        "sign_ok": sign_ok,
        "pred_ok": pred_ok,
        "ablation_ok": abl_ok,
        "lambda_ok": lam_ok,
        "delta_auroc": d_auroc,
        "delta_auprc": d_auprc,
        "delta_bce_shuffle": d_shuf,
        "delta_bce_beta0": d_b0,
        "corr_lambda": at["recovery"].get("corr_lambda"),
    }


def train_factorized(
    *,
    family: str,
    age_temporal: bool,
    scenario_dir: Path,
    run_dir: Path,
    beta_true: float,
    theta0_true: float = 0.0,
    aggregation: str = "additive",
    interaction_only: bool = True,
    max_epochs: int = 40,
    patience: int = 10,
    batch_size: int = 64,
    d_model: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    specs = json.loads((scenario_dir / "target_specs.json").read_text())
    target_idx = (
        [i for i, s in enumerate(specs) if s["mechanism"] == "interaction"]
        if interaction_only
        else None
    )
    train_loader, val_loader, test_loader, vocab, info = make_loaders(
        scenario_dir, batch_size=batch_size, target_idx=target_idx
    )
    model = build_factorized(
        family,
        age_temporal=age_temporal,
        n_codes=info["n_codes"],
        n_types=info["n_types"],
        n_targets=info["n_targets"],
        d_model=d_model,
        aggregation=aggregation,
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
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        grad_theta = grad_beta = None
        for batch in train_loader:
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            logits = model(
                code_ids=batch["code_ids"],
                type_ids=batch["type_ids"],
                tau=batch["tau"],
                padding_mask=batch["padding_mask"],
                is_query=batch["is_query"],
                age=batch["age"],
                lag_days=batch["lag_days"],
            )
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            gate = model.gate if hasattr(model, "gate") else model.inner.gate
            if gate.theta0.grad is not None:
                grad_theta = float(gate.theta0.grad.norm().cpu())
            if gate.beta.requires_grad and gate.beta.grad is not None:
                grad_beta = float(gate.beta.grad.norm().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * batch["labels"].size(0)
            n += batch["labels"].size(0)
        val = evaluate(model, val_loader, dev)
        gate = model.gate if hasattr(model, "gate") else model.inner.gate
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_bce": val["bce"],
            "val_auroc": val["micro_auroc"],
            "val_auprc": val["micro_auprc"],
            "beta": float(gate.beta.detach().cpu()),
            "theta0": float(gate.theta0.detach().cpu()),
            "grad_theta0": grad_theta,
            "grad_beta": grad_beta,
        }
        history.append(row)
        arm = "age_temporal" if age_temporal else "temporal_only"
        print(
            f"{family}/{arm}/{aggregation} ep{epoch} "
            f"train={row['train_loss']:.4f} val_auroc={val['micro_auroc']:.3f} "
            f"beta={row['beta']:.3f}"
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
    torch.save(model.state_dict(), run_dir / "model.pt")

    test = evaluate(model, test_loader, dev)
    abl = ablations(model, test_loader, dev)
    rec = recovery(model, beta_true, theta0_true)
    result = {
        "family": family,
        "age_temporal": age_temporal,
        "aggregation": aggregation,
        "interaction_only": interaction_only,
        "n_params": count_parameters(model),
        "test": test,
        "ablations": abl,
        "recovery": rec,
        "history": history,
        "beta_hat": rec["beta_hat"],
        "beta_true": beta_true,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["M1", "M2", "M3"], default="M1")
    ap.add_argument("--age-temporal", action="store_true")
    ap.add_argument("--temporal-only", action="store_true")
    ap.add_argument("--aggregation", default="additive")
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--beta-true", type=float, default=-2.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    args = ap.parse_args()
    age_temporal = args.age_temporal or (not args.temporal_only)
    if args.temporal_only:
        age_temporal = False
    train_factorized(
        family=args.family,
        age_temporal=age_temporal,
        scenario_dir=args.scenario_dir,
        run_dir=args.run_dir,
        beta_true=args.beta_true,
        aggregation=args.aggregation,
        max_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        d_model=args.d_model,
    )


if __name__ == "__main__":
    main()
