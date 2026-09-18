"""Training loop for the minimal age × temporal interaction experiment."""
from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from config import BETA_TRUE, Config, LAMBDA0_TRUE, NEG_CODE, POS_CODE, QUERY_CODE
from evaluate import (
    counterfactual_age,
    evaluate,
    signal_attention_recovery,
)
from model import InteractionModel, count_parameters, temporal_param_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _optimizer(model: InteractionModel, cfg: Config) -> AdamW:
    temporal_ids = temporal_param_ids(model)
    decay, nodecay, temporal = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in temporal_ids:
            temporal.append(p)
        else:
            decay.append(p)
    return AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
            {"params": temporal, "weight_decay": 0.0, "lr": 10.0 * cfg.lr},
        ],
        lr=cfg.lr,
    )


def recovered_params(model: InteractionModel) -> dict[str, float]:
    return {
        "lambda0_hat": float(model.lambda0.detach().cpu()),
        "beta_hat": float(model.beta.detach().cpu()),
        "lambda0_true": float(LAMBDA0_TRUE),
        "beta_true": float(BETA_TRUE.get(getattr(model, "_task", "T1"), 0.0)),
    }


def train_run(
    cfg: Config,
    train_loader,
    val_loader,
    test_loader,
    n_codes: int,
    n_types: int,
    age_mean: float,
    age_std: float,
    extra_meta: dict[str, Any] | None = None,
    z_test: np.ndarray | None = None,
    collect_attention: bool = True,
    early_stop: bool = True,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device()
    model = InteractionModel(
        arm=cfg.arm,
        n_codes=n_codes,
        n_types=n_types,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        head_hidden=cfg.head_hidden,
        pos_id=extra_meta.get("pos_id") if extra_meta else None,
        neg_id=extra_meta.get("neg_id") if extra_meta else None,
        query_id=extra_meta.get("query_id") if extra_meta else None,
    ).to(device)
    model._task = cfg.task
    n_params = count_parameters(model)
    opt = _optimizer(model, cfg)
    loss_fn = BCEWithLogitsLoss()

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = run_dir / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    last_epoch = 0
    stale = 0

    def _payload(epoch: int, val_bce: float, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "model_state_dict": model.state_dict(),
            "arm": cfg.arm,
            "task": cfg.task,
            "seed": cfg.seed,
            "epoch": epoch,
            "val_bce": val_bce,
            "n_params": n_params,
            "n_codes": n_codes,
            "n_types": n_types,
            "config": cfg.to_dict(),
            "recovered": recovered_params(model),
        }

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        n_correct = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["code_ids"].to(device),
                batch["type_ids"].to(device),
                batch["time_norm"].to(device),
                batch["days_before"].to(device),
                batch["padding_mask"].to(device),
                batch["z_age"].to(device),
                batch["is_query"].to(device),
            )
            labels = batch["labels"].to(device)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            bs = int(labels.size(0))
            running += float(loss.item()) * bs
            n_seen += bs
            n_correct += int(((logits.detach() >= 0).float() == labels).sum().item())
        train_bce = running / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {
            "bce": train_bce,
            "accuracy": train_acc,
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
        rec = recovered_params(model)
        rec["lambda_at_ages"] = model.lambda_at_ages(
            (1, 3, 5, 8, 11, 14, 17), age_mean, age_std
        )
        row = {
            "epoch": epoch,
            "train_bce": train_bce,
            "train_accuracy": train_acc,
            "val_bce": val_metrics["bce"],
            "val_accuracy": val_metrics.get("accuracy"),
            "val_auroc": val_metrics.get("auroc"),
            "val_auprc": val_metrics.get("auprc"),
            "recovered": rec,
        }
        history.append(row)
        last_epoch = epoch
        print(
            f"[{cfg.task} {cfg.arm} seed={cfg.seed}] epoch {epoch:02d} "
            f"train_bce={train_bce:.4f} train_acc={train_acc:.3f} "
            f"val_bce={val_metrics['bce']:.4f} "
            f"λ0={rec['lambda0_hat']:+.3f} β={rec['beta_hat']:+.3f}",
            flush=True,
        )
        is_best = float(val_metrics["bce"]) < best_val - 1e-6
        if is_best:
            best_val = float(val_metrics["bce"])
            best_epoch = epoch
            stale = 0
            torch.save(_payload(epoch, best_val, "best"), ckpt_best)
        else:
            stale += 1
            if early_stop and stale >= cfg.patience:
                print(
                    f"early stop at epoch {epoch} (best val BCE {best_val:.4f} @ {best_epoch})",
                    flush=True,
                )
                break

    if ckpt_best.exists():
        blob = torch.load(ckpt_best, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state_dict"])
    else:
        torch.save(_payload(last_epoch, best_val, "best"), ckpt_best)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device) if val_loader is not None else train_metrics
    test_metrics = evaluate(model, test_loader, device) if test_loader is not None else train_metrics
    rec = recovered_params(model)
    rec["lambda_at_ages"] = model.lambda_at_ages(
        (0, 1, 2, 3, 5, 6, 8, 10, 11, 14, 17, 18), age_mean, age_std
    )

    result: dict[str, Any] = {
        "task": cfg.task,
        "arm": cfg.arm,
        "seed": cfg.seed,
        "hyperparameters": cfg.to_dict(),
        "n_codes": n_codes,
        "n_types": n_types,
        "n_params": n_params,
        "device": str(device),
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "recovered": rec,
        "history": [
            {k: v for k, v in h.items() if k != "recovered"}
            | {"lambda0_hat": h["recovered"]["lambda0_hat"], "beta_hat": h["recovered"]["beta_hat"]}
            for h in history
        ],
        "age_mean_train": age_mean,
        "age_std_train": age_std,
    }

    if collect_attention and test_loader is not None:
        result["attention_recovery"] = signal_attention_recovery(model, test_loader, device)
        if z_test is not None:
            result["intervention"] = counterfactual_age(model, test_loader, device, z_test, seed=cfg.seed)

    if extra_meta:
        result.update(extra_meta)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2, default=str) + "\n")
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    return result
