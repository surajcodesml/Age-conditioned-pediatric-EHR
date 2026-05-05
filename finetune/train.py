#!/usr/bin/env python3
"""Fine-tune TALE-EHR pretrained backbone for binary disease prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import duckdb
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.dataset import (
    DiseaseClassificationDataset,
    TensorizedDiseaseClassificationDataset,
    _dataloader_worker_init,
    disease_collate,
)
from finetune.model import TALEEHRClassifier


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TALE-EHR for disease classification.")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=Path, required=True)
    parser.add_argument("--cohort_dir", type=Path, required=True)
    parser.add_argument(
        "--tensorized_dir",
        type=Path,
        default=None,
        help="If set, use tensorized shards under <tensorized_dir>/{train,val,test}/",
    )
    parser.add_argument("--events_parquet", type=Path, default=Path("data/processed/patient_events_rolled_full.parquet"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_workers", type=int, default=max(4, min(12, (os.cpu_count() or 16) - 2)))
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--dry_run_one_epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cohort_stats(cohort_path: Path) -> tuple[int, int, float]:
    con = duckdb.connect()
    try:
        n_pos, n_neg = con.execute(
            """
            SELECT
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS n_pos,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS n_neg
            FROM read_parquet(?)
            """,
            [str(cohort_path.resolve())],
        ).fetchone()
    finally:
        con.close()
    n_pos = int(n_pos or 0)
    n_neg = int(n_neg or 0)
    pos_weight = float(n_neg / max(n_pos, 1))
    return n_pos, n_neg, pos_weight


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)
    acc = float((y_pred == y_true).mean())
    out = {"accuracy": acc, "auroc": float("nan"), "auprc": float("nan")}
    if y_true.min() != y_true.max():
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])
        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        labels = batch["labels"].detach().cpu().numpy().astype(np.int32)
        probs_all.append(probs)
        labels_all.append(labels)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_prob = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float64)
    y_true = np.concatenate(labels_all, axis=0) if labels_all else np.array([], dtype=np.int32)
    metrics = _compute_metrics(y_true, y_prob) if y_true.size > 0 else {"accuracy": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
    return mean_loss, metrics


def _check_gradients(model: TALEEHRClassifier) -> tuple[bool, bool]:
    has_classifier_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0 for p in model.classifier.parameters())
    has_backbone_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0 for p in model.backbone.parameters() if p.requires_grad)
    return has_classifier_grad, has_backbone_grad


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    run_name = f"{args.disease}_run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.run_dir or (Path("checkpoints/finetune") / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    console_log_path = run_dir / "console.log"
    console_fh = open(console_log_path, "w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, console_fh)
    sys.stderr = _TeeStream(orig_stderr, console_fh)

    try:
        run_t0 = time.perf_counter()

        print(f"[run] checkpoints -> {run_dir}", flush=True)
        print(f"[run] console log -> {console_log_path}", flush=True)

        train_cohort = args.cohort_dir / "train_cohort.parquet"
        val_cohort = args.cohort_dir / "val_cohort.parquet"
        test_cohort = args.cohort_dir / "test_cohort.parquet"
        for p in (train_cohort, val_cohort, test_cohort, args.events_parquet, args.vocab_path, args.pretrained_ckpt):
            if not p.exists():
                raise FileNotFoundError(f"Missing required path: {p}")

        if args.tensorized_dir is not None:
            train_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "train", max_seq_len=args.max_seq_len, shard_cache_size=4)
            val_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "val", max_seq_len=args.max_seq_len, shard_cache_size=4)
            test_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "test", max_seq_len=args.max_seq_len, shard_cache_size=4)
            print(f"[data] tensorized_dir={args.tensorized_dir}", flush=True)
        else:
            train_ds = DiseaseClassificationDataset(train_cohort, args.events_parquet, args.vocab_path, args.max_seq_len)
            val_ds = DiseaseClassificationDataset(val_cohort, args.events_parquet, args.vocab_path, args.max_seq_len)
            test_ds = DiseaseClassificationDataset(test_cohort, args.events_parquet, args.vocab_path, args.max_seq_len)
            print("[data] mode=on_the_fly_duckdb", flush=True)

        loader_kw = dict(
            batch_size=args.batch_size,
            collate_fn=disease_collate,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
            worker_init_fn=_dataloader_worker_init,
        )
        if args.num_workers > 0:
            loader_kw["multiprocessing_context"] = "spawn"
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

        n_pos, n_neg, pos_weight = _cohort_stats(train_cohort)
        print(f"[train cohort] positives={n_pos:,} negatives={n_neg:,} pos_weight={pos_weight:.6f}", flush=True)

        device = torch.device(args.device)
        model = TALEEHRClassifier(args.pretrained_ckpt, freeze_backbone=False).to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.classifier.parameters(), "lr": args.lr_head},
            ]
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Required pre-train sanity: one forward/loss/backward check.
        first_batch = next(iter(train_loader))
        first_batch = _move_batch_to_device(first_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            logits = model(first_batch)
            loss = criterion(logits, first_batch["labels"])
        if logits.ndim != 1 or logits.shape[0] != first_batch["labels"].shape[0]:
            raise RuntimeError(f"Expected logits [B], got {tuple(logits.shape)}")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in sanity check")
        scaler.scale(loss).backward()
        has_cls_grad, has_backbone_grad = _check_gradients(model)
        sanity_elapsed = time.perf_counter() - run_t0
        print(
            f"[sanity] t+{sanity_elapsed:.1f}s logits_shape={tuple(logits.shape)} "
            f"loss={float(loss.item()):.6f} classifier_grad={has_cls_grad} backbone_grad={has_backbone_grad}",
            flush=True,
        )
        if not has_cls_grad or not has_backbone_grad:
            raise RuntimeError("Gradient check failed for classifier/backbone")
        optimizer.zero_grad(set_to_none=True)

        best_val_auroc = -float("inf")
        best_epoch = 0
        best_ckpt_path = run_dir / "best.pt"
        history: list[dict[str, float | int]] = []

        total_epochs = 1 if args.dry_run_one_epoch else args.epochs
        for epoch in range(1, total_epochs + 1):
            ep_start = time.perf_counter()
            model.train()
            train_losses: list[float] = []

            for step, batch in enumerate(train_loader, start=1):
                batch = _move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    logits = model(batch)
                    loss = criterion(logits, batch["labels"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                if step % 200 == 0 or step == 1:
                    step_elapsed = time.perf_counter() - run_t0
                    print(
                        f"  ep{epoch:03d} step {step:6d} t+{step_elapsed:.1f}s loss={float(loss.item()):.6f}",
                        flush=True,
                    )

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            elapsed = time.perf_counter() - ep_start

            msg = (
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | val_AUROC={val_metrics['auroc']:.6f} | "
                f"val_AUPRC={val_metrics['auprc']:.6f} | val_acc={val_metrics['accuracy']:.6f} | "
                f"time={elapsed:.1f}s"
            )
            print(msg, flush=True)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auroc": float(val_metrics["auroc"]),
                    "val_auprc": float(val_metrics["auprc"]),
                    "val_acc": float(val_metrics["accuracy"]),
                }
            )

            val_auroc = float(val_metrics["auroc"])
            rank_auroc = val_auroc if np.isfinite(val_auroc) else -float("inf")
            if rank_auroc > best_val_auroc:
                best_val_auroc = rank_auroc
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "val_loss": val_loss,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )
                print(f"[best] Saved checkpoint: {best_ckpt_path}", flush=True)

        if not best_ckpt_path.exists():
            raise RuntimeError("No best checkpoint was saved.")
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"[test @ best epoch {best_epoch}] loss={test_loss:.6f} "
            f"AUROC={test_metrics['auroc']:.6f} AUPRC={test_metrics['auprc']:.6f} "
            f"acc={test_metrics['accuracy']:.6f}",
            flush=True,
        )

        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_val_auroc": best_val_auroc,
                    "history": history,
                    "test_loss": test_loss,
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
            )
        return 0
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        console_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
