#!/usr/bin/env python3
"""STEP 5/6: fine-tune a TALE-EHR pretrained backbone on a Synthea disease task.

Why a thin wrapper instead of calling ``finetune/train.py`` directly: that script
constructs ``finetune.model.TALEEHRClassifier``, which (a) hardcodes the MIMIC
``data/processed/bge_embeddings.pt`` table and (b) sizes ``num_codes`` from the
MIMIC checkpoint head -- both incompatible with a FRESH Synthea BGE table/vocab
(the embedding-table row count would mismatch the MIMIC checkpoint and crash the
load). This is the same reason the endorsed PIC path uses ``train_pic.py`` +
``PICTALEEHRClassifier``.

This wrapper REUSES, unchanged:
  * ``finetune.PIC.model_pic.PICTALEEHRClassifier`` -- warm-starts the pretrained
    backbone (auto-detecting age-conditioned vs vanilla) onto a fresh embedding
    table; the classifier head trains from scratch.
  * ``finetune.train`` evaluation / calibration / diagnostics helpers, including
    the newly-extended ``age_stratified_metrics`` (per developmental band).

The classifier class is identical for both arms (it auto-detects the backbone
from the checkpoint), so the only difference between the age and vanilla runs is
``--pretrained_ckpt``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.dataset import (
    TensorizedDiseaseClassificationDataset,
    _dataloader_worker_init,
    disease_collate,
)
from finetune.train import (
    _TeeStream,
    _bootstrap_metric_cis,
    _check_gradients,
    _cohort_stats,
    _compute_decay_grid,
    _compute_ece,
    _fit_temperature,
    _move_batch_to_device,
    _resolve_age_conditioned_backbone,
    _sigmoid_np,
    _write_predictions_parquet,
    age_stratified_metrics,
    evaluate,
    evaluate_full,
    set_seed,
)
from finetune.PIC.model_pic import PICTALEEHRClassifier, _num_codes_from_vocab


def _is_kernel_param(name: str) -> bool:
    return ("age_coeff_gen" in name) or ("temporal_weight.coefficients" in name)


def _detect_arm_type(model: PICTALEEHRClassifier) -> str:
    return "age" if _resolve_age_conditioned_backbone(model) is not None else "vanilla"


def _build_optimizer_param_groups(model, lr_kernel, lr_backbone, lr_head):
    kernel_params, slow_params = [], []
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        (kernel_params if _is_kernel_param(name) else slow_params).append(param)
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    groups = [
        {"params": kernel_params, "lr": lr_kernel},
        {"params": slow_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    trainable = [p for _, p in model.named_parameters() if p.requires_grad]
    assigned = {id(p) for g in groups for p in g["params"]}
    if len(assigned) != len(trainable):
        raise RuntimeError(f"optimizer groups cover {len(assigned)} != {len(trainable)} trainable params")
    return groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune TALE-EHR on a Synthea disease task.")
    p.add_argument("--disease", type=str, required=True)
    p.add_argument("--pretrained_ckpt", type=Path, required=True)
    p.add_argument("--cohort_dir", type=Path, required=True)
    p.add_argument("--tensorized_dir", type=Path, required=True)
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data" / "synthea" / "processed" / "code_vocab.json")
    p.add_argument("--embedding_path", type=Path, default=REPO_ROOT / "data" / "synthea" / "processed" / "bge_embeddings.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_kernel", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--num_workers", type=int, default=max(2, min(8, (os.cpu_count() or 8) - 2)))
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--dry_run_one_epoch", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    console_fh = open(run_dir / "console.log", "w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, console_fh)
    sys.stderr = _TeeStream(orig_stderr, console_fh)

    try:
        run_t0 = time.perf_counter()
        print(f"[run] disease={args.disease} run_dir={run_dir}", flush=True)
        print(f"[run] pretrained_ckpt={args.pretrained_ckpt}", flush=True)

        train_cohort = args.cohort_dir / "train_cohort.parquet"
        for p in (train_cohort, args.vocab_path, args.embedding_path, args.pretrained_ckpt):
            if not p.exists():
                raise FileNotFoundError(f"Missing required path: {p}")

        train_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "train", max_seq_len=args.max_seq_len, shard_cache_size=4)
        val_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "val", max_seq_len=args.max_seq_len, shard_cache_size=4)
        test_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "test", max_seq_len=args.max_seq_len, shard_cache_size=4)

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

        num_codes = _num_codes_from_vocab(args.vocab_path)
        device = torch.device(args.device)
        model = PICTALEEHRClassifier(
            pretrained_ckpt_path=args.pretrained_ckpt,
            pic_embedding_path=args.embedding_path,
            pic_num_codes=num_codes,
            freeze_backbone=False,
        ).to(device)
        arm_type = _detect_arm_type(model)
        print(f"[model] arm_type={arm_type} num_codes={num_codes}", flush=True)

        optimizer = torch.optim.AdamW(
            _build_optimizer_param_groups(model, args.lr_kernel, args.lr_backbone, args.lr_head)
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Pre-train sanity: one forward/loss/backward check (finite loss + grads).
        first_batch = _move_batch_to_device(next(iter(train_loader)), device)
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
        print(f"[sanity] loss={float(loss.item()):.6f} classifier_grad={has_cls_grad} backbone_grad={has_backbone_grad}", flush=True)
        if not has_cls_grad or not has_backbone_grad:
            raise RuntimeError("Gradient check failed for classifier/backbone")
        optimizer.zero_grad(set_to_none=True)

        best_val_auroc = -float("inf")
        best_epoch = 0
        best_ckpt_path = run_dir / "best.pt"
        history: list[dict] = []
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

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_AUROC={val_metrics['auroc']:.6f} val_AUPRC={val_metrics['auprc']:.6f} | "
                f"time={time.perf_counter() - ep_start:.1f}s",
                flush=True,
            )
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_auroc": float(val_metrics["auroc"]), "val_auprc": float(val_metrics["auprc"]),
                "val_acc": float(val_metrics["accuracy"]),
            })

            # Model selection on val loss/NLL (per the spec: keep val-loss for selection,
            # not AUROC). Tie-break to lowest val_loss; track best val_auroc for reporting.
            select_metric = -val_loss if np.isfinite(val_loss) else -float("inf")
            if epoch == 1 or select_metric > getattr(main, "_best_select", -float("inf")):
                main._best_select = select_metric
                best_epoch = epoch
                best_val_auroc = float(val_metrics["auroc"]) if np.isfinite(val_metrics["auroc"]) else best_val_auroc
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "val_metrics": val_metrics, "val_loss": val_loss, "args": vars(args)}, best_ckpt_path)
                print(f"[best] epoch={epoch} val_loss={val_loss:.6f} -> {best_ckpt_path}", flush=True)

        if not best_ckpt_path.exists():
            raise RuntimeError("No best checkpoint was saved.")
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

        val_full = evaluate_full(model, val_loader, criterion, device)
        test_full = evaluate_full(model, test_loader, criterion, device)
        test_loss = float(test_full["mean_loss"])
        test_metrics = test_full["metrics"]
        _write_predictions_parquet(run_dir / "test_predictions.parquet", test_full)

        y_test = test_full["y_true"].astype(np.int32)
        p_test = test_full["y_prob"].astype(np.float64)
        z_test = test_full["y_logit"].astype(np.float64)
        n_events_test = test_full["n_events_in_window"].astype(np.float64)
        length_only_auroc = float("nan")
        if y_test.size > 0 and y_test.min() != y_test.max():
            length_only_auroc = float(roc_auc_score(y_test, n_events_test))
        leakage_gap = (float(test_metrics["auroc"]) - length_only_auroc
                       if np.isfinite(float(test_metrics["auroc"])) and np.isfinite(length_only_auroc) else float("nan"))

        temperature = _fit_temperature(val_full["y_logit"].astype(np.float64), val_full["y_true"].astype(np.float64), device=device)
        p_test_ts = _sigmoid_np(z_test / max(temperature, 1e-6))
        brier_raw = float(brier_score_loss(y_test, p_test)) if y_test.size > 0 else float("nan")
        brier_ts = float(brier_score_loss(y_test, p_test_ts)) if y_test.size > 0 else float("nan")
        ece_raw = _compute_ece(y_test.astype(np.float64), p_test, n_bins=15)
        ece_ts = _compute_ece(y_test.astype(np.float64), p_test_ts, n_bins=15)
        bootstrap = _bootstrap_metric_cis(y_test, p_test, n_bootstrap=1000, seed=42)

        test_extended = {
            "length_only_auroc": float(length_only_auroc),
            "leakage_gap": float(leakage_gap),
            "auroc_ci": [float(bootstrap["auroc_ci"][0]), float(bootstrap["auroc_ci"][1])],
            "auprc_ci": [float(bootstrap["auprc_ci"][0]), float(bootstrap["auprc_ci"][1])],
            "brier_raw": float(brier_raw), "brier_ts": float(brier_ts),
            "ece_raw": float(ece_raw), "ece_ts": float(ece_ts), "temperature": float(temperature),
        }

        test_age_stratified = age_stratified_metrics(
            test_full["y_true"], test_full["y_prob"],
            np.clip(test_full["age_at_landmark"].astype(np.float64), 0.0, None),
        )

        print(f"[test @ best epoch {best_epoch}] loss={test_loss:.6f} "
              f"AUROC={test_metrics['auroc']:.6f} AUPRC={test_metrics['auprc']:.6f}", flush=True)
        print(f"[test_extended] length_only_AUROC={length_only_auroc:.4f} leakage_gap={leakage_gap:.4f}", flush=True)
        print("[age_stratified test] AUROC by developmental band:", flush=True)
        for band, rec in test_age_stratified.items():
            flag = " (unreliable)" if rec["unreliable"] else ""
            print(f"  {band:>6}: n={int(rec['n']):5d} prev={rec['prevalence']:.3f} "
                  f"AUROC={rec['auroc']:.4f} AUPRC={rec['auprc']:.4f}{flag}", flush=True)

        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump({
                "disease": args.disease,
                "arm_type": arm_type,
                "pretrained_ckpt": str(args.pretrained_ckpt),
                "best_epoch": best_epoch,
                "best_val_auroc": best_val_auroc,
                "history": history,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "test_extended": test_extended,
                "test_age_stratified": test_age_stratified,
            }, f, indent=2)
        print(f"[done] wrote {run_dir / 'history.json'}", flush=True)
        return 0
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        console_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
