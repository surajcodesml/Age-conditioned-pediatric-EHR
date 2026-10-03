"""Training loops. Standard runs call the existing baseline trainer unchanged."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines.common.training import set_seed, train_neural_baseline

from ladder.artifacts import save_checkpoint, write_config_yaml, write_json
from ladder.models.factory import build_model


class LimitedLoader:
    """First-N batch view used only by smoke tests."""

    def __init__(self, loader, n_batches: int) -> None:
        self.loader = loader
        self.n_batches = int(n_batches)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n_batches:
                break
            yield batch

    def __len__(self) -> int:
        return min(self.n_batches, len(self.loader))


def forward_logits(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    out = model(
        enc_code_ids=batch["enc_code_ids"],
        enc_code_mask=batch["enc_code_mask"],
        enc_tau=batch["enc_tau"],
        enc_padding_mask=batch["enc_padding_mask"],
        age=batch["age"],
        enc_lag_days=batch.get("enc_lag_days"),
    )
    if isinstance(out, dict):
        return out["logits"]
    return out


def _loaders(train_loader, val_loader, limit_batches: int | None):
    if limit_batches is None:
        return train_loader, val_loader
    return LimitedLoader(train_loader, limit_batches), LimitedLoader(val_loader, limit_batches)


def train_standard(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    seed: int,
    device: str,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    """One AdamW group over every trainable parameter. Matches dtr_*_new."""
    model.configure_arm_()
    train_loader, val_loader = _loaders(train_loader, val_loader, limit_batches)

    def train_fn(batch):
        logits = forward_logits(model, batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        return {"loss": loss}

    def predict_fn(batch):
        return forward_logits(model, batch)

    return train_neural_baseline(
        model=model,
        train_fn=train_fn,
        predict_fn=predict_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        max_epochs=int(cfg["max_epochs"]),
        patience=int(cfg["patience"]),
        min_epochs=int(cfg["min_epochs"]),
        grad_clip=float(cfg["grad_clip"]),
        device=device,
        run_dir=run_dir,
        seed=int(seed),
    )


def train_temporal_only_group(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    seed: int,
    device: str,
    epochs: int,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    """Stage B: encoder and readout frozen. theta/beta use the temporal learning rate."""
    model.set_backbone_requires_grad(False)
    params = [p for p in model.temporal_parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Stage B has no trainable temporal parameters")
    groups = [{
        "params": params,
        "lr": float(cfg["lr"]) * float(cfg["temporal_lr_mult"]),
        "weight_decay": float(cfg["temporal_weight_decay"]),
    }]
    train_loader, val_loader = _loaders(train_loader, val_loader, limit_batches)

    def train_fn(batch):
        logits = forward_logits(model, batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        return {"loss": loss}

    def predict_fn(batch):
        return forward_logits(model, batch)

    return train_neural_baseline(
        model=model,
        train_fn=train_fn,
        predict_fn=predict_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        max_epochs=int(epochs),
        patience=int(epochs),
        min_epochs=int(epochs),
        grad_clip=float(cfg["grad_clip"]),
        device=device,
        run_dir=run_dir,
        seed=int(seed),
        optimizer_groups=groups,
    )


def train_joint_temporal_group(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    seed: int,
    device: str,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    """Stage C: unfreeze all parameters. Temporal group keeps its own learning rate."""
    model.set_backbone_requires_grad(True)
    temporal_ids = {id(p) for p in model.temporal_parameters()}
    backbone, temporal = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in temporal_ids:
            temporal.append(param)
        else:
            backbone.append(param)
    groups = []
    if backbone:
        groups.append({
            "params": backbone,
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg["weight_decay"]),
        })
    if temporal:
        groups.append({
            "params": temporal,
            "lr": float(cfg["lr"]) * float(cfg["temporal_lr_mult"]),
            "weight_decay": float(cfg["temporal_weight_decay"]),
        })
    train_loader, val_loader = _loaders(train_loader, val_loader, limit_batches)

    def train_fn(batch):
        logits = forward_logits(model, batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        return {"loss": loss}

    def predict_fn(batch):
        return forward_logits(model, batch)

    return train_neural_baseline(
        model=model,
        train_fn=train_fn,
        predict_fn=predict_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        max_epochs=int(cfg["max_epochs"]),
        patience=int(cfg["patience"]),
        min_epochs=int(cfg["min_epochs"]),
        grad_clip=float(cfg["grad_clip"]),
        device=device,
        run_dir=run_dir,
        seed=int(seed),
        optimizer_groups=groups,
    )


def fork_matched_arms(
    state: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    n_codes: int,
    n_targets: int,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Clone one checkpoint into temporal_only (beta frozen) and age_temporal (beta trainable)."""
    age_temporal = build_model(cfg, n_codes, n_targets, age_temporal=True)
    temporal_only = build_model(cfg, n_codes, n_targets, age_temporal=False)
    age_temporal.load_state_dict(state)
    temporal_only.load_state_dict(state)
    age_temporal.configure_arm_()
    temporal_only.configure_arm_()
    return age_temporal, temporal_only


def max_state_diff(left: torch.nn.Module, right: torch.nn.Module) -> float:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise KeyError("Forked arms do not have the same state keys")
    worst = 0.0
    for key in left_state:
        worst = max(worst, float((left_state[key] - right_state[key]).abs().max().item()))
    return worst


def flatten_history(stages: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, result in stages:
        for row in result.get("history", []):
            tagged = dict(row)
            tagged["stage"] = stage
            rows.append(tagged)
    return rows


def finalize_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    cfg: dict[str, Any],
    *,
    n_codes: int,
    n_targets: int,
    age_temporal: bool,
    history_rows: list[dict[str, Any]],
) -> Path:
    path = run_dir / "checkpoint_best.pt"
    save_checkpoint(
        path,
        model,
        cfg=cfg,
        n_codes=n_codes,
        n_targets=n_targets,
        age_temporal=age_temporal,
    )
    write_json(run_dir / "history.json", history_rows)
    write_config_yaml(run_dir / "config.yaml", cfg)
    return path
