"""Generic training loop for binary sequence classification."""
from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from config import Config
from evaluate import evaluate
from model import AgeIncorporationModel, count_parameters


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


def train_run(
    cfg: Config,
    train_loader,
    val_loader,
    test_loader,
    n_codes: int,
    n_types: int,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device()
    model = AgeIncorporationModel(
        arm=cfg.arm,
        n_codes=n_codes,
        n_types=n_types,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        age_hidden=cfg.age_hidden,
        head_hidden=cfg.head_hidden,
    ).to(device)
    n_params = count_parameters(model)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = BCEWithLogitsLoss()

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = run_dir / "checkpoint_best.pt"
    ckpt_last = run_dir / "checkpoint_last.pt"
    history_path = run_dir / "history.json"

    def _payload(epoch: int, val_auprc: float, kind: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "model_state_dict": model.state_dict(),
            "arm": cfg.arm,
            "task": cfg.task,
            "seed": cfg.seed,
            "epoch": epoch,
            "val_auprc": val_auprc,
            "n_params": n_params,
            "n_codes": n_codes,
            "n_types": n_types,
            "config": cfg.to_dict(),
        }

    history: list[dict[str, Any]] = []
    best_val = -1.0
    best_epoch = 0
    last_epoch = 0
    stale = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["code_ids"].to(device),
                batch["type_ids"].to(device),
                batch["time_norm"].to(device),
                batch["age_event_norm"].to(device),
                batch["padding_mask"].to(device),
                batch["index_age_norm"].to(device),
            )
            labels = batch["labels"].to(device)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            bs = int(labels.size(0))
            running += float(loss.item()) * bs
            n_seen += bs
        train_loss = running / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        is_best = float(val_metrics["auprc"]) > best_val
        row = {
            "epoch": epoch,
            "train_bce": train_loss,
            "val_auprc": val_metrics["auprc"],
            "val_auroc": val_metrics["auroc"],
            "val_bce": val_metrics["bce"],
            "val_by_age_group": val_metrics["by_age_group"],
            "is_best": is_best,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2, default=str) + "\n")
        last_epoch = epoch
        torch.save(_payload(epoch, float(val_metrics["auprc"]), "last"), ckpt_last)
        print(
            f"[{cfg.task} {cfg.arm} seed={cfg.seed}] epoch {epoch:02d} "
            f"train_bce={train_loss:.4f} val_auprc={val_metrics['auprc']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}",
            flush=True,
        )
        if is_best:
            best_val = float(val_metrics["auprc"])
            best_epoch = epoch
            stale = 0
            torch.save(_payload(epoch, best_val, "best"), ckpt_best)
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stop at epoch {epoch} (best val AUPRC {best_val:.4f} @ {best_epoch})")
                break

    blob = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    val_best = evaluate(model, val_loader, device)

    result = {
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
        "val_auprc": val_best["auprc"],
        "val_auroc": val_best["auroc"],
        "val_bce": val_best["bce"],
        "test_auprc": test_metrics["auprc"],
        "test_auroc": test_metrics["auroc"],
        "test_bce": test_metrics["bce"],
        "test_prevalence": test_metrics["prevalence"],
        "test_n": test_metrics["n"],
        "test_by_age_group": test_metrics["by_age_group"],
        "history": history,
        "checkpoint_best": str(ckpt_best),
        "checkpoint_last": str(ckpt_last),
        "history_path": str(history_path),
    }
    if extra_meta:
        result.update(extra_meta)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    history_path.write_text(json.dumps(history, indent=2, default=str) + "\n")
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    return result
