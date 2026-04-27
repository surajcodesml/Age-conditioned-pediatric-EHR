#!/usr/bin/env python3
"""TALE-EHR pretraining loop (Algorithm 1 pretraining phase)."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    from model.dataset import EHRDataset, TensorizedEHRDataset, ehr_collate
    from model.tale_ehr import TALEEHR
except ModuleNotFoundError:
    from dataset import EHRDataset, TensorizedEHRDataset, ehr_collate
    from tale_ehr import TALEEHR


def _dataloader_worker_init(_worker_id: int) -> None:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def bce_code_loss(
    code_logits: torch.Tensor,
    target_codes: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plain BCE on multi-label code targets without label smoothing.
    Optional pos_weight scales positive-class loss by (n_neg / n_pos) per
    class to combat extreme positive sparsity.
    """
    return F.binary_cross_entropy_with_logits(
        code_logits,
        target_codes,
        pos_weight=pos_weight,
        reduction="mean",
    )


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
    intensity: torch.Tensor,       # [B] predicted intensity at the event
    target_time_gap: torch.Tensor, # [B] observed Δt to next event (days or weeks)
    T: float | None = None,
    n_mc_samples: int = 20,
) -> torch.Tensor:
    eps = 1e-6
    lam = F.softplus(intensity) + eps
    tau_weeks = (target_time_gap / 7.0).clamp(min=eps, max=520.0).to(lam.dtype)
    nll = lam * tau_weeks - torch.log(lam)
    return nll.mean()


def compute_metrics(code_logits: torch.Tensor, target_codes: torch.Tensor, ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    with torch.no_grad():
        probs = torch.sigmoid(code_logits)
        out: dict[str, float] = {}
        for k in ks:
            topk = probs.topk(min(k, probs.shape[-1]), dim=-1).indices
            hits = target_codes.gather(1, topk).sum(dim=-1)
            n_pos = target_codes.sum(dim=-1).clamp(min=1)
            out[f"recall@{k}"] = (hits / n_pos).mean().item()
        try:
            from sklearn.metrics import roc_auc_score

            y = target_codes.cpu().numpy().ravel()
            p = probs.float().cpu().numpy().ravel()
            if 0 < y.sum() < len(y):
                out["auroc"] = float(roc_auc_score(y, p))
        except Exception:
            pass
        return out


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


class _TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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
    code_loss_name: str = "bce",
    bce_pos_weight: float = 0.0,
    resume_from: str | Path | None = None,
) -> None:
    device_t = torch.device(device)
    model.to(device_t)
    optimizer = Adam(model.parameters(), lr=lr)
    use_amp = device_t.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1

    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device_t)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] loaded checkpoint: {resume_path}", flush=True)
        print(f"[resume] continuing from epoch {start_epoch}", flush=True)

    if code_loss_name == "focal":
        code_loss_fn = focal_code_loss
        pos_weight_tensor = None
    else:
        if bce_pos_weight > 0:
            pos_weight_tensor = torch.full((1,), float(bce_pos_weight), device=device_t)
        else:
            pos_weight_tensor = None

        def code_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return bce_code_loss(logits, targets, pos_weight=pos_weight_tensor)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "config.json", "w") as f:
        json.dump({"epochs": epochs, "lr": lr, "gamma_loss": gamma_loss,
               "device": device, "dry_run": dry_run, "code_loss": code_loss_name,
               "bce_pos_weight": bce_pos_weight, "resume_from": str(resume_from) if resume_from else None}, f, indent=2)

    log_file = save_dir / "train.log"
    log_mode = "a" if resume_from is not None else "w"
    log_fh = open(log_file, log_mode, buffering=1)  # line-buffered

    if start_epoch > epochs:
        print(
            f"[resume] checkpoint epoch already >= target epochs ({start_epoch - 1} >= {epochs}). Nothing to do.",
            flush=True,
        )
        log_fh.close()
        return

    for epoch in range(start_epoch, epochs + 1):
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
                loss_code = code_loss_fn(out["code_logits"], batch["target_codes"])
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
            if step % 50 == 0 or step == 1:
                print(f"  ep{epoch:03d} step {step:6d} "
                    f"loss={float(loss.detach().item()):.4f} "
                    f"(code={float(loss_code.detach().item()):.4f}, "
                    f"time={float(loss_time.detach().item()):.4f})",
                    flush=True)

        model.eval()
        val_total = 0.0
        val_code = 0.0
        val_time = 0.0
        val_metrics_sum: dict[str, float] = {}
        n_metric_batches = 0
        n_val = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader, start=1):
                if max_val_steps is not None and step > max_val_steps:
                    break
                batch = _move_batch_to_device(batch, device_t)
                ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    out = model(batch)
                    loss_code = code_loss_fn(out["code_logits"], batch["target_codes"])
                    T = float(torch.clamp(batch["target_time_gap"].max(), min=1.0).item())
                    loss_time = temporal_point_process_loss(out["intensity"], batch["target_time_gap"], T=T)
                    loss = loss_time + gamma_loss * loss_code
                val_total += float(loss.detach().item())
                val_code += float(loss_code.detach().item())
                val_time += float(loss_time.detach().item())
                m = compute_metrics(out["code_logits"], batch["target_codes"])
                for k, v in m.items():
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + v
                n_metric_batches += 1
                n_val += 1
                if step % 50 == 0 or step == 1:
                    print(f"  val epoch {epoch:03d} step {step:6d} loss={float(loss.detach().item()):.4f}", flush=True)

        train_loss = train_total / max(n_train, 1)
        train_code_loss = train_code / max(n_train, 1)
        train_time_loss = train_time / max(n_train, 1)
        val_loss = val_total / max(n_val, 1)
        val_code_loss = val_code / max(n_val, 1)
        val_time_loss = val_time / max(n_val, 1)
        elapsed = time.perf_counter() - ep_start
        metric_str = " ".join(
            f"{k}={v / max(n_metric_batches, 1):.4f}"
            for k, v in val_metrics_sum.items()
        )

        msg = (f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} (code={train_code_loss:.6f}, time={train_time_loss:.6f}) | "
                f"val={val_loss:.6f} (code={val_code_loss:.6f}, time={val_time_loss:.6f}) | "
                f"time={elapsed:.1f}s")
        if metric_str:
            msg += " | " + metric_str
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        
        ckpt_path = save_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}", flush=True)
        log_fh.write(f"Saved checkpoint: {ckpt_path}\n")

        if dry_run:
            print("Dry run complete (3 train steps, 1 val step).")
            break

    log_fh.close()
    print(f"Training log saved to {log_file}", flush=True)
        
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
    parser.add_argument("--gamma_loss", type=float, default=500.0)
    parser.add_argument(
        "--code_loss",
        choices=["bce", "focal"],
        default="bce",
        help="Code prediction loss. 'bce' is the default (plain BCE without "
             "smoothing or focal weighting). 'focal' uses paper's focal loss.",
    )
    parser.add_argument(
        "--bce_pos_weight",
        type=float,
        default=0.0,
        help="If >0, apply pos_weight=this scalar to BCE loss. Common values "
             "10-100. Default 0 = no pos_weight.",
    )
    parser.add_argument("--max_rows", type=int, default=0, help="0 for full dataset, >0 for quick debug")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints/"))
    parser.add_argument("--run_name", type=str, default=None,help="Subdir name under save_dir. Default: timestamp.")
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Path to checkpoint .pt file to resume from. Restores model/optimizer and continues from next epoch.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--use_tensorized",
        action="store_true",
        help="Load from data/tensorized/ shards instead of raw parquet. "
        "Requires preprocessing/tensorize.py to have been run first.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run exactly 3 training steps and 1 validation step, then exit.",
    )
    args = parser.parse_args()
    
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    args.save_dir = args.save_dir / run_name
    args.save_dir.mkdir(parents=True, exist_ok=True)
    console_log_path = args.save_dir / "console.log"
    console_mode = "a" if args.resume_from is not None else "w"
    console_fh = open(console_log_path, console_mode, buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, console_fh)
    sys.stderr = _TeeStream(orig_stderr, console_fh)
    try:
        print(f"[run] checkpoints -> {args.save_dir}", flush=True)
        print(f"[run] console log -> {console_log_path}", flush=True)

        start = time.perf_counter()
        with args.vocab_path.open("r", encoding="utf-8") as f:
            code_vocab = json.load(f)
        num_codes = len(code_vocab)

        train_path = _resolve_split_path(args.data_dir, "train")
        val_path = _resolve_split_path(args.data_dir, "val")
        row_limit = None if args.max_rows == 0 else args.max_rows

        if args.use_tensorized:
            tensorized_root = args.data_dir / "tensorized"
            train_ds = TensorizedEHRDataset(tensorized_root / "train", args.vocab_path)
            val_ds = TensorizedEHRDataset(tensorized_root / "val", args.vocab_path)
        else:
            train_ds = EHRDataset(train_path, args.vocab_path, max_rows=row_limit)
            val_ds = EHRDataset(val_path, args.vocab_path, max_rows=row_limit)

        _loader_kw = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=ehr_collate,
            pin_memory=(args.device == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(4 if args.num_workers > 0 else None),
            worker_init_fn=_dataloader_worker_init,
            multiprocessing_context="spawn"
        )
        train_loader = DataLoader(train_ds, shuffle=True, **_loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **_loader_kw)

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
            code_loss_name=args.code_loss,
            bce_pos_weight=args.bce_pos_weight,
            resume_from=args.resume_from,
        )
        total_time = time.perf_counter() - start
        print(f"Total training time: {total_time:.1f}s", flush=True)
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        console_fh.close()
