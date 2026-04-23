#!/usr/bin/env python3
"""TALE-EHR pretraining loop (Algorithm 1 pretraining phase)."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    from model.dataset import EHRDataset, ehr_collate
    from model.tale_ehr import TALEEHR
except ModuleNotFoundError:
    from dataset import EHRDataset, ehr_collate
    from tale_ehr import TALEEHR


def focal_code_loss(
    code_logits: torch.Tensor,
    target_codes: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    smoothed = target_codes * 0.95 + (1.0 - target_codes) * 0.05
    p = torch.sigmoid(code_logits)
    p_t = p * smoothed + (1.0 - p) * (1.0 - smoothed)
    focal_weight = alpha * (1.0 - p_t).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(code_logits, smoothed, reduction="none")
    loss = focal_weight * bce
    return loss.mean()


def temporal_point_process_loss(
    intensity: torch.Tensor,
    target_time_gap: torch.Tensor,
    T: float,
    n_mc_samples: int = 20,
) -> torch.Tensor:
    # target_time_gap, T, and n_mc_samples are kept for interface parity with paper notation.
    _ = target_time_gap, T, n_mc_samples
    norm_sq = (intensity**2).mean()
    fit_term = 2.0 * intensity.mean()
    return norm_sq - fit_term


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    gamma_loss: float = 1.0,
    device: str = "cuda",
    save_dir: str | Path = "checkpoints/",
    dry_run: bool = False,
) -> None:
    device_t = torch.device(device)
    model.to(device_t)
    optimizer = Adam(model.parameters(), lr=lr)
    use_amp = device_t.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        train_total = 0.0
        train_code = 0.0
        train_time = 0.0
        n_train = 0

        max_train_steps = 3 if dry_run else None
        max_val_steps = 1 if dry_run else None

        for step, batch in enumerate(train_loader, start=1):
            if max_train_steps is not None and step > max_train_steps:
                break
            batch = _move_batch_to_device(batch, device_t)
            optimizer.zero_grad(set_to_none=True)

            ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(batch)
                loss_code = focal_code_loss(out["code_logits"], batch["target_codes"])
                # Use max observed gap in batch as horizon proxy T.
                T = float(torch.clamp(batch["target_time_gap"].max(), min=1.0).item())
                loss_time = temporal_point_process_loss(out["intensity"], batch["target_time_gap"], T=T)
                loss = loss_time + gamma_loss * loss_code

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_total += float(loss.detach().item())
            train_code += float(loss_code.detach().item())
            train_time += float(loss_time.detach().item())
            n_train += 1

        model.eval()
        val_total = 0.0
        val_code = 0.0
        val_time = 0.0
        n_val = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader, start=1):
                if max_val_steps is not None and step > max_val_steps:
                    break
                batch = _move_batch_to_device(batch, device_t)
                ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    out = model(batch)
                    loss_code = focal_code_loss(out["code_logits"], batch["target_codes"])
                    T = float(torch.clamp(batch["target_time_gap"].max(), min=1.0).item())
                    loss_time = temporal_point_process_loss(out["intensity"], batch["target_time_gap"], T=T)
                    loss = loss_time + gamma_loss * loss_code
                val_total += float(loss.detach().item())
                val_code += float(loss_code.detach().item())
                val_time += float(loss_time.detach().item())
                n_val += 1

        train_loss = train_total / max(n_train, 1)
        train_code_loss = train_code / max(n_train, 1)
        train_time_loss = train_time / max(n_train, 1)
        val_loss = val_total / max(n_val, 1)
        val_code_loss = val_code / max(n_val, 1)
        val_time_loss = val_time / max(n_val, 1)
        elapsed = time.perf_counter() - ep_start

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} (code={train_code_loss:.6f}, time={train_time_loss:.6f}) | "
            f"val={val_loss:.6f} (code={val_code_loss:.6f}, time={val_time_loss:.6f}) | "
            f"time={elapsed:.1f}s"
        )

        ckpt_path = save_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

        if dry_run:
            print("Dry run complete (3 train steps, 1 val step).")
            break


def _resolve_split_path(data_dir: Path, split_name: str) -> Path:
    path = data_dir / f"{split_name}_events.parquet"
    if path.exists():
        return path
    if split_name == "val":
        fallback = data_dir / "test_events.parquet"
        if fallback.exists():
            print(f"Validation split not found; using fallback {fallback}")
            return fallback
    raise FileNotFoundError(f"Missing split file: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain TALE-EHR model")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/"))
    parser.add_argument("--embedding_path", type=Path, default=Path("data/processed/bge_embeddings.pt"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--poly_degree", type=int, default=5)
    parser.add_argument("--gamma_loss", type=float, default=1.0)
    parser.add_argument("--max_rows", type=int, default=0, help="0 for full dataset, >0 for quick debug")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints/"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run exactly 3 training steps and 1 validation step, then exit.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    with args.vocab_path.open("r", encoding="utf-8") as f:
        code_vocab = json.load(f)
    num_codes = len(code_vocab)

    train_path = _resolve_split_path(args.data_dir, "train")
    val_path = _resolve_split_path(args.data_dir, "val")
    row_limit = None if args.max_rows == 0 else args.max_rows

    train_ds = EHRDataset(train_path, args.vocab_path, max_rows=row_limit)
    val_ds = EHRDataset(val_path, args.vocab_path, max_rows=row_limit)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ehr_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ehr_collate,
    )

    model = TALEEHR(
        embedding_path=args.embedding_path,
        num_codes=num_codes,
        d_model=args.d_model,
        poly_degree=args.poly_degree,
    )
    pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        gamma_loss=args.gamma_loss,
        device=args.device,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
    )
    total_time = time.perf_counter() - start
    print(f"Total training time: {total_time:.1f}s")
