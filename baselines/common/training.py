"""Common training loop for neural baselines on the synthetic benchmark.

Handles:
  - Seed setting
  - Training loop with early stopping
  - Validation metrics per epoch
  - Checkpoint saving
  - History recording
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.metrics import multilabel_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "cuda") -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    predict_fn,
    loader: DataLoader,
    device: torch.device,
    ks: tuple[int, ...] = (5,),
) -> dict[str, float]:
    """Run predictions over a loader and compute multilabel metrics."""
    model.eval()
    all_y, all_logits = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = predict_fn(batch)
        all_y.append(batch["labels"].cpu().numpy())
        if isinstance(out, ModelOutput):
            all_logits.append(out.logits.cpu().numpy())
        elif isinstance(out, dict) and "logits" in out:
            all_logits.append(out["logits"].cpu().numpy())
        elif isinstance(out, torch.Tensor):
            all_logits.append(out.cpu().numpy())
        else:
            raise TypeError(f"predict_fn returned unexpected type {type(out)}")
    y = np.concatenate(all_y, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    return multilabel_metrics(y, logits, ks=ks)


def train_neural_baseline(
    model: nn.Module,
    train_fn,
    predict_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    max_epochs: int = 25,
    patience: int = 5,
    grad_clip: float = 1.0,
    device: str = "cuda",
    run_dir: Path | None = None,
    seed: int = 0,
    optimizer_groups: list[dict] | None = None,
) -> dict[str, Any]:
    """Common training loop for neural baselines.

    Args:
        model: nn.Module to train
        train_fn: (batch) -> dict with 'loss' key
        predict_fn: (batch) -> ModelOutput or tensor
        train_loader, val_loader: dataloaders
        optimizer_groups: if None, uses all trainable params
    """
    set_seed(seed)
    dev = get_device(device)
    model.to(dev)

    if optimizer_groups is None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer_groups = [{"params": params, "lr": lr, "weight_decay": weight_decay}]
    optimizer = AdamW(optimizer_groups, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    best_val_bce = float("inf")
    best_state = None
    patience_left = patience
    history: list[dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(dev) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = train_fn(batch)
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bsz = batch["labels"].size(0)
            total_loss += loss.item() * bsz
            n_samples += bsz

        train_loss = total_loss / max(n_samples, 1)
        val_metrics = evaluate_loader(model, predict_fn, val_loader, dev)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "time_s": time.time() - t0,
        }
        history.append(row)

        val_bce = val_metrics["bce"]
        if val_bce < best_val_bce:
            best_val_bce = val_bce
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    if run_dir is not None:
        torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")
        with (run_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2, default=str)

    return {
        "best_val_bce": best_val_bce,
        "epochs_trained": len(history),
        "history": history,
        "total_time_s": time.time() - t0,
    }
