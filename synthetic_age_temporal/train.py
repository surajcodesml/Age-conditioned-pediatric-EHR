#!/usr/bin/env python3
"""Train matched Transformer arms on the synthetic age × temporal benchmark."""
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

from config import ARMS, Config
from dataset import make_loaders
from evaluate import (
    evaluate_model,
    functional_ablations,
    parameter_recovery,
    quick_ablation_deltas,
)
from model import BenchmarkModel, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: BenchmarkModel, cfg: Config) -> AdamW:
    temporal_ids = {id(p) for p in model.age_parameters()}
    decay, temporal = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in temporal_ids:
            temporal.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr}]
    if temporal:
        groups.append({"params": temporal, "weight_decay": 0.0, "lr": 10.0 * cfg.lr})
    return AdamW(groups, lr=cfg.lr)


def _beta_snapshot(model: BenchmarkModel) -> dict[str, Any]:
    bv = model.temporal.beta_vector().cpu().tolist()
    tv = model.temporal.theta0_vector().cpu().tolist()
    return {
        "beta_hat": float(np.mean(bv)),
        "beta_vec": bv,
        "theta0_hat": float(np.mean(tv)),
        "theta0_vec": tv,
    }


def train_one(
    cfg: Config,
    scenario_dir: Path | None = None,
) -> dict[str, Any]:
    cfg.resolve_gt()
    set_seed(cfg.model_seed)
    device = get_device(cfg.device)
    scenario_dir = Path(scenario_dir or cfg.scenario_dir())
    if not (scenario_dir / "READY").exists() and (scenario_dir / "NOT_READY").exists():
        raise RuntimeError(
            f"Scenario not oracle-ready: {scenario_dir}. Fix generator before training."
        )

    # Resolve interaction-only target indices from specs.
    target_idx = None
    with (scenario_dir / "target_specs.json").open() as f:
        specs = json.load(f)
    if cfg.interaction_only:
        target_idx = [i for i, sp in enumerate(specs) if sp["mechanism"] == "interaction"]
        if not target_idx:
            raise RuntimeError("interaction_only requested but no interaction targets")

    train_loader, val_loader, test_loader, vocab, info = make_loaders(
        scenario_dir,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        target_idx=target_idx,
    )
    model = BenchmarkModel(
        arm=cfg.arm,
        n_codes=info["n_codes"],
        n_types=info["n_types"],
        n_targets=info["n_targets"],
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).to(device)
    opt = build_optimizer(model, cfg)
    loss_fn = BCEWithLogitsLoss()

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        last_grad = {}
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
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
            if any(p.requires_grad for p in (model.temporal.theta0,)):
                if model.temporal.theta0.grad is None:
                    raise RuntimeError("theta0 received no gradient")
            if model.temporal.beta.requires_grad and model.temporal.beta.grad is None:
                raise RuntimeError("beta received no gradient")
            if model.temporal.theta0.grad is not None:
                last_grad["theta0"] = float(model.temporal.theta0.grad.detach().norm().cpu())
            if model.temporal.beta.grad is not None:
                last_grad["beta"] = float(model.temporal.beta.grad.detach().norm().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.item()) * batch["labels"].size(0)
            n += batch["labels"].size(0)
        train_loss = total / max(n, 1)
        val_metrics = evaluate_model(model, val_loader, device)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **_beta_snapshot(model),
            "grad_norms": last_grad,
        }
        if cfg.track_mechanism_each_epoch:
            # Full ablations each epoch (slower; used for convergence plots).
            mech = quick_ablation_deltas(model, val_loader, device)
            row.update({f"mech_{k}": v for k, v in mech.items()})
        history.append(row)
        print(
            f"{cfg.scenario}/{cfg.arm}"
            f"{'_inter' if cfg.interaction_only else ''} "
            f"ep{epoch} train={train_loss:.4f} val_bce={val_metrics['bce']:.4f} "
            f"val_auroc={val_metrics['micro_auroc']:.3f} "
            f"beta={row['beta_hat']:.3f}"
        )
        if val_metrics["bce"] < best_val - 1e-5:
            best_val = val_metrics["bce"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "model.pt")

    test_metrics = evaluate_model(model, test_loader, device)
    ablations = functional_ablations(model, test_loader, device)
    recovery = parameter_recovery(
        model,
        beta_true=cfg.beta_true,
        theta0_true=cfg.theta0,
        scenario=cfg.scenario,
    )
    if last_grad:
        recovery["grad_norms"] = last_grad

    # Predictive gain vs a temporal_only sibling is left to the follow-up runner.
    result = {
        "arm": cfg.arm,
        "scenario": cfg.scenario,
        "strength": cfg.strength,
        "cohort": cfg.cohort,
        "data_seed": cfg.data_seed,
        "model_seed": cfg.model_seed,
        "interaction_only": cfg.interaction_only,
        "max_seq_len": cfg.max_seq_len,
        "n_params": count_parameters(model),
        "n_temporal_params": sum(p.numel() for p in model.age_parameters()),
        "best_val_bce": best_val,
        "test": test_metrics,
        "ablations": ablations,
        "recovery": recovery,
        "history": history,
        "theta0_hat": recovery["theta0_hat"],
        "beta_hat": recovery["beta_hat"],
        "beta_true": cfg.beta_true,
        "theta0_true": cfg.theta0,
        "target_idx": target_idx,
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--arm", default="age_temporal", choices=ARMS)
    ap.add_argument("--strength", default="medium")
    ap.add_argument("--cohort", default="controlled")
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--model-seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenario-dir", type=Path, default=None)
    ap.add_argument("--interaction-only", action="store_true")
    ap.add_argument("--track-mechanism", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()

    cfg = Config(
        scenario=args.scenario,
        arm=args.arm,
        strength=args.strength,
        cohort=args.cohort,
        device=args.device,
        interaction_only=args.interaction_only,
        track_mechanism_each_epoch=args.track_mechanism,
        run_tag=args.run_tag,
    )
    if args.data_seed is not None:
        cfg.data_seed = args.data_seed
    if args.model_seed is not None:
        cfg.model_seed = args.model_seed
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.patience is not None:
        cfg.patience = args.patience
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    result = train_one(cfg, scenario_dir=args.scenario_dir)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "test",
                    "recovery",
                    "beta_hat",
                    "beta_true",
                    "ablations",
                    "interaction_only",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
