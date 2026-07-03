#!/usr/bin/env python3
"""Arm-specific fine-tune on the CHD (heart-malformation) task.

Only ``--arm`` varies training behavior; data splits, seed, optimizer, hyper-
parameters, early-stop criterion, and eval are identical across the four arms
(enforced by shared defaults + a single code path). Locked: additive-logspace
kernel, no time/Weibull loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_ablation.arms import ARMS
from model_ablation.dataset_finetune import (
    DiseaseClassificationDataset,
    TensorizedDiseaseClassificationDataset,
    _dataloader_worker_init,
    disease_collate,
)
from model_ablation.model_finetune import TALEEHRAblationClassifier

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = average_precision_score = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Four-arm ablation fine-tune (CHD).")
    p.add_argument("--arm", type=str, required=True, choices=ARMS)
    p.add_argument("--pretrained_ckpt", type=Path, required=True)
    p.add_argument("--cohort_dir", type=Path, default=None)
    p.add_argument("--tensorized_dir", type=Path, default=None)
    p.add_argument("--events_parquet", type=Path, default=REPO_ROOT / "data/processed/patient_events_rolled_full.parquet")
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)  # SHARED across arms
    p.add_argument("--num_workers", type=int, default=max(2, min(8, (os.cpu_count() or 8) - 2)))
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--run_dir", type=Path, default=None)
    p.add_argument("--dry_run_one_epoch", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _pos_weight(train_ds) -> float:
    labels = np.array([r["label"] for r in getattr(train_ds, "_rows", [])], dtype=np.float64) \
        if hasattr(train_ds, "_rows") else None
    if labels is None or labels.size == 0:
        return 1.0
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    return n_neg / max(n_pos, 1.0)


DEV_BANDS = (("<1", 0.0, 1.0), ("1-5", 1.0, 6.0), ("6-11", 6.0, 12.0),
             ("12-17", 12.0, 18.0), ("18-25", 18.0, 26.0))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses, probs_all, labels_all, age_all = [], [], [], []
    for batch in loader:
        batch = _move(batch, device)
        logits = model(batch)
        losses.append(float(criterion(logits, batch["labels"]).item()))
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(batch["labels"].cpu().numpy())
        # Age-at-landmark from the SEPARATE age_years field (not demographics).
        lengths = batch["attention_mask"].sum(dim=1).long().clamp(min=1)
        ridx = torch.arange(lengths.shape[0], device=device)
        age_all.append(batch["age_years"][ridx, lengths - 1].cpu().numpy())
    y_prob = np.concatenate(probs_all) if probs_all else np.array([])
    y_true = np.concatenate(labels_all) if labels_all else np.array([])
    age = np.concatenate(age_all) if age_all else np.array([])
    m = {"auroc": float("nan"), "auprc": float("nan")}
    if roc_auc_score is not None and y_true.size and y_true.min() != y_true.max():
        m["auroc"] = float(roc_auc_score(y_true, y_prob))
        m["auprc"] = float(average_precision_score(y_true, y_prob))
    return float(np.mean(losses)) if losses else float("nan"), m, y_true, y_prob, age


def age_stratified(y_true, y_prob, age, min_n=20) -> dict[str, dict[str, float]]:
    out = {}
    for name, lo, hi in DEV_BANDS:
        mask = (age >= lo) & (age < hi)
        yt, yp = y_true[mask], y_prob[mask]
        rec = {"n": int(yt.size), "auroc": float("nan"), "auprc": float("nan")}
        if yt.size >= min_n and yt.min() != yt.max() and roc_auc_score is not None:
            rec["auroc"] = float(roc_auc_score(yt, yp))
            rec["auprc"] = float(average_precision_score(yt, yp))
        out[name] = rec
    return out


def build_loaders(args):
    kw = dict(batch_size=args.batch_size, collate_fn=disease_collate, num_workers=args.num_workers,
              pin_memory=(args.device == "cuda"), persistent_workers=(args.num_workers > 0),
              prefetch_factor=(4 if args.num_workers > 0 else None), worker_init_fn=_dataloader_worker_init)
    if args.num_workers > 0:
        kw["multiprocessing_context"] = "spawn"
    if args.tensorized_dir is not None:
        mk = lambda split: TensorizedDiseaseClassificationDataset(args.tensorized_dir / split, args.max_seq_len)
        train_ds, val_ds, test_ds = mk("train"), mk("val"), mk("test")
    else:
        if args.cohort_dir is None:
            raise ValueError("Provide --tensorized_dir or --cohort_dir")
        mk = lambda name: DiseaseClassificationDataset(args.cohort_dir / f"{name}_cohort.parquet",
                                                       args.events_parquet, args.vocab_path, args.max_seq_len)
        train_ds, val_ds, test_ds = mk("train"), mk("val"), mk("test")
    return (train_ds,
            DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = args.run_dir or (REPO_ROOT / "checkpoints/finetune/ablation" / f"chd_{args.arm}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] arm={args.arm} -> {run_dir}", flush=True)

    train_ds, train_loader, val_loader, test_loader = build_loaders(args)
    pos_weight = _pos_weight(train_ds)
    print(f"[cohort] pos_weight={pos_weight:.4f}", flush=True)

    model = TALEEHRAblationClassifier(args.arm, args.pretrained_ckpt, freeze_backbone=False).to(device)
    print(f"[arm={args.arm}] age_pathway_params={model.backbone.age_pathway_param_count()}", flush=True)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auroc, best_epoch = -float("inf"), 0
    best_ckpt = run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    total_epochs = 1 if args.dry_run_one_epoch else args.epochs
    for epoch in range(1, total_epochs + 1):
        model.train()
        tl = []
        for step, batch in enumerate(train_loader, 1):
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                loss = criterion(model(batch), batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            tl.append(float(loss.item()))
        val_loss, vm, *_ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train_loss={np.mean(tl):.6f} | val_loss={val_loss:.6f} | "
              f"val_AUROC={vm['auroc']:.6f} | val_AUPRC={vm['auprc']:.6f}", flush=True)
        history.append({"epoch": epoch, "train_loss": float(np.mean(tl)), "val_auroc": vm["auroc"]})
        rank = vm["auroc"] if np.isfinite(vm["auroc"]) else -float("inf")
        if rank > best_auroc:
            best_auroc, best_epoch = rank, epoch
            torch.save({"epoch": epoch, "arm": args.arm, "model_state_dict": model.state_dict()}, best_ckpt)

    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
    test_loss, tm, yt, yp, age = evaluate(model, test_loader, criterion, device)
    strat = age_stratified(yt, yp, np.clip(age, 0.0, None))
    print(f"[test @ best epoch {best_epoch}] AUROC={tm['auroc']:.6f} AUPRC={tm['auprc']:.6f}", flush=True)
    print("[age_stratified test]", flush=True)
    for band, rec in strat.items():
        print(f"  {band:>6}: n={rec['n']:5d} AUROC={rec['auroc']:.4f} AUPRC={rec['auprc']:.4f}", flush=True)
    with (run_dir / "history.json").open("w") as f:
        json.dump({"arm": args.arm, "best_epoch": best_epoch, "history": history,
                   "test_metrics": tm, "test_age_stratified": strat}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
