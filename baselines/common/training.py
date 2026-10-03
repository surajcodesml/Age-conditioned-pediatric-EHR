"""Common training loop for neural baselines.

Handles:
  - Seed setting
  - Training loop with early stopping (after ``min_epochs``)
  - Validation metrics per epoch
  - Checkpoint saving: best + last
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
    max_batches: int | None = None,
    metrics_mode: str = "full",
) -> dict[str, float]:
    """Run predictions over a loader and compute multilabel metrics.

    ``max_batches`` caps how many batches are scored. Required for MIMIC-scale
    next-visit heads (~30k codes) where concatenating the full val/test
    logit tensor OOMs (tens of GB).

    ``metrics_mode``: ``"bce"`` (early-stopping), ``"fast"``, or ``"full"``.
    Training validation should use ``"bce"`` — full micro/macro AUROC on a
    30k-code head can take >30 minutes and thrash host RAM.
    """
    model.eval()
    # Online BCE avoids building a giant logit tensor when only BCE is needed.
    bce_sum, n_elem = 0.0, 0
    all_y, all_logits = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= int(max_batches):
            break
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = predict_fn(batch)
        y_t = batch["labels"]
        if isinstance(out, ModelOutput):
            logits_t = out.logits
        elif isinstance(out, dict) and "logits" in out:
            logits_t = out["logits"]
        elif isinstance(out, torch.Tensor):
            logits_t = out
        else:
            raise TypeError(f"predict_fn returned unexpected type {type(out)}")

        # Running BCE in float32 on device (matches training objective).
        bce_sum += float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits_t.float(), y_t.float(), reduction="sum"
            ).item()
        )
        n_elem += int(y_t.numel())

        if metrics_mode != "bce":
            all_y.append(y_t.detach().cpu().numpy())
            all_logits.append(logits_t.detach().cpu().numpy())

    if n_elem == 0:
        return {"bce": float("nan"), "micro_auroc": float("nan"),
                "macro_auroc": float("nan"), "micro_auprc": float("nan"),
                "macro_auprc": float("nan")}

    if metrics_mode == "bce":
        return {
            "bce": bce_sum / max(n_elem, 1),
            "micro_auroc": float("nan"),
            "macro_auroc": float("nan"),
            "micro_auprc": float("nan"),
            "macro_auprc": float("nan"),
        }

    y = np.concatenate(all_y, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    return multilabel_metrics(y, logits, ks=ks, mode=metrics_mode)


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
    min_epochs: int | None = None,
    grad_clip: float = 1.0,
    device: str = "cuda",
    run_dir: Path | None = None,
    seed: int = 0,
    optimizer_groups: list[dict] | None = None,
    val_max_batches: int | None = None,
    grad_accum_steps: int = 1,
    amp: str = "fp32",
) -> dict[str, Any]:
    """Common training loop for neural baselines.

    Early stopping (``patience``) is enabled but cannot fire before
    ``min_epochs`` have completed. Default ``min_epochs`` is
    ``max(patience, max_epochs // 2)`` so short budgets still train
    a meaningful number of epochs.

    ``grad_accum_steps`` > 1 accumulates gradients over micro-batches before
    each optimizer step (effective batch ≈ micro-batch × accum). Default 1
    preserves prior behavior.

    ``amp``: ``"fp32"`` (default), ``"bf16"``, or ``"fp16"``. BF16 on ROCm/CUDA
    materially raises samples/sec for Transformer baselines; BCEWithLogits stays
    in float32 under autocast.

    When ``run_dir`` is set, writes:
      - ``last_checkpoint.pt``  — weights after the final trained epoch
      - ``best_checkpoint.pt``  — best validation-BCE weights (also loaded into model)
      - ``history.json``        — per-epoch train/val metrics
    """
    set_seed(seed)
    dev = get_device(device)
    model.to(dev)
    grad_accum_steps = max(1, int(grad_accum_steps))
    amp = str(amp or "fp32").lower()
    if amp not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"amp must be fp32|bf16|fp16, got {amp!r}")
    use_amp = amp in ("bf16", "fp16") and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    # GradScaler only required for fp16; bf16 is numerically stable without it.
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp == "fp16"))

    if min_epochs is None:
        min_epochs = max(int(patience), int(max_epochs) // 2)
    min_epochs = int(max(1, min(min_epochs, max_epochs)))

    if optimizer_groups is None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer_groups = [{"params": params, "lr": lr, "weight_decay": weight_decay}]
    optimizer = AdamW(optimizer_groups, lr=lr)

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    best_val_bce = float("inf")
    best_state = None
    best_epoch = 0
    patience_left = patience
    history: list[dict[str, Any]] = []
    t0 = time.time()
    stopped_early = False

    n_train_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_samples, n_batches = 0.0, 0, 0
        epoch_t0 = time.time()
        print(
            f"  epoch {epoch}/{max_epochs} start"
            + (f" ({n_train_batches} train batches)" if n_train_batches else "")
            + (f"  accum={grad_accum_steps}" if grad_accum_steps > 1 else "")
            + (f"  amp={amp}" if use_amp else ""),
            flush=True,
        )
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast(
                device_type=dev.type, dtype=amp_dtype, enabled=use_amp,
            ):
                result = train_fn(batch)
                loss = result["loss"] / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            bsz = batch["labels"].size(0)
            total_loss += loss.item() * grad_accum_steps * bsz
            n_samples += bsz
            n_batches += 1
            if (n_batches % grad_accum_steps) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # Heartbeat so long MIMIC epochs aren't silent for hours.
            if n_batches == 1 or n_batches % 200 == 0:
                print(
                    f"    batch {n_batches}"
                    + (f"/{n_train_batches}" if n_train_batches else "")
                    + f"  loss={loss.item() * grad_accum_steps:.4f}  "
                    f"elapsed={time.time() - epoch_t0:.0f}s",
                    flush=True,
                )
        # Flush leftover micro-batches that didn't fill an accum window.
        if (n_batches % grad_accum_steps) != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = total_loss / max(n_samples, 1)
        print(f"    train done ({n_batches} batches); starting val...", flush=True)
        # BCE-only val: early stopping key. Full AUROC on |V|≈30k is prohibitively slow.
        val_metrics = evaluate_loader(
            model, predict_fn, val_loader, dev,
            max_batches=val_max_batches, metrics_mode="bce",
        )
        print(f"    val done  val_bce={val_metrics['bce']:.6f}", flush=True)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "time_s": time.time() - t0,
        }
        history.append(row)

        val_bce = val_metrics["bce"]
        is_best = val_bce < best_val_bce
        if is_best:
            best_val_bce = val_bce
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1

        # Persist after every epoch so a multi-hour first epoch still leaves artifacts.
        if run_dir is not None:
            torch.save(model.state_dict(), run_dir / "last_checkpoint.pt")
            if is_best and best_state is not None:
                torch.save(best_state, run_dir / "best_checkpoint.pt")
                torch.save(best_state, run_dir / "checkpoint.pt")
            with (run_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2, default=str)

        print(
            f"  epoch {epoch}/{max_epochs} done  train_loss={train_loss:.4f}  "
            f"val_bce={val_bce:.4f}  best_val_bce={best_val_bce:.4f}  "
            f"best_epoch={best_epoch}  epoch_s={time.time() - epoch_t0:.0f}",
            flush=True,
        )

        # Early stop only after the mandatory minimum epoch budget.
        if epoch >= min_epochs and patience_left <= 0:
            stopped_early = True
            break

    last_epoch = len(history)

    if best_state is not None:
        model.load_state_dict(best_state)

    if run_dir is not None:
        # Final best/canonical write (idempotent if already saved mid-loop).
        torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")
        torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        with (run_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2, default=str)

    return {
        "best_val_bce": best_val_bce,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "epochs_trained": last_epoch,
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "patience": patience,
        "val_max_batches": val_max_batches,
        "stopped_early": stopped_early,
        "amp": amp,
        "history": history,
        "total_time_s": time.time() - t0,
        "checkpoints": {
            "best": str(run_dir / "best_checkpoint.pt") if run_dir else None,
            "last": str(run_dir / "last_checkpoint.pt") if run_dir else None,
            "canonical": str(run_dir / "checkpoint.pt") if run_dir else None,
        },
    }
