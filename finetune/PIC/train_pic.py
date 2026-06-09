#!/usr/bin/env python3
"""Fine-tune a TALE-EHR pretrained backbone on PIC binary disease tasks.

Mirrors ``finetune/train.py`` but uses :class:`PICTALEEHRClassifier`, which
warm-starts the pretrained backbone while retaining PIC's own BGE embedding
table and vocabulary. All evaluation / calibration / diagnostic helpers are
imported from ``finetune.train`` (read-only reuse).
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

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    _compute_alpha_band_spread,
    _compute_decay_grid,
    _compute_ece,
    _fit_temperature,
    _move_batch_to_device,
    _resolve_age_conditioned_backbone,
    _sigmoid_np,
    _write_decay_grid_csv,
    _write_predictions_parquet,
    evaluate,
    evaluate_full,
    set_seed,
)
from finetune.PIC.model_pic import PICTALEEHRClassifier, _num_codes_from_vocab


def _is_kernel_param(name: str) -> bool:
    return ("age_coeff_gen" in name) or ("temporal_weight.coefficients" in name)


def _detect_arm_type(model: PICTALEEHRClassifier) -> str:
    return "age" if _resolve_age_conditioned_backbone(model) is not None else "vanilla"


def _build_optimizer_param_groups(
    model: PICTALEEHRClassifier,
    lr_kernel: float,
    lr_backbone: float,
    lr_head: float,
) -> tuple[list[dict], dict[str, list[str]]]:
    kernel_params: list[torch.nn.Parameter] = []
    slow_params: list[torch.nn.Parameter] = []
    kernel_names: list[str] = []
    slow_names: list[str] = []

    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if _is_kernel_param(name):
            kernel_params.append(param)
            kernel_names.append(name)
        else:
            slow_params.append(param)
            slow_names.append(name)

    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    head_names = [f"classifier.{n}" for n, _ in model.classifier.named_parameters() if _.requires_grad]

    groups = [
        {"params": kernel_params, "lr": lr_kernel},
        {"params": slow_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    name_lists = {"kernel": kernel_names, "backbone_slow": slow_names, "head": head_names}

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assigned = set(id(p) for g in groups for p in g["params"])
    if len(assigned) != len(trainable):
        raise RuntimeError(
            f"Optimizer groups do not cover all trainable params: "
            f"assigned={len(assigned)} trainable={len(trainable)}"
        )
    if len(assigned) != len(kernel_params) + len(slow_params) + len(head_params):
        raise RuntimeError("Double-counted parameters across optimizer groups")

    overlap = {id(p) for p in kernel_params} & {id(p) for p in slow_params}
    if overlap:
        raise RuntimeError("Kernel and slow backbone groups overlap")

    return groups, name_lists


def _print_optimizer_groups(
    arm_type: str,
    name_lists: dict[str, list[str]],
    lr_kernel: float,
    lr_backbone: float,
    lr_head: float,
) -> None:
    print(f"[optimizer] arm_type={arm_type}", flush=True)
    for group_name, lr in (
        ("kernel", lr_kernel),
        ("backbone_slow", lr_backbone),
        ("head", lr_head),
    ):
        names = name_lists[group_name]
        print(
            f"[optimizer] {group_name}: n_params={len(names)} lr={lr:g}",
            flush=True,
        )
        for n in names:
            print(f"    {n}", flush=True)


@torch.no_grad()
def _print_postrun_age_kernel_norms(model: PICTALEEHRClassifier) -> None:
    """||alpha(a) - coefficients||_2 on attention temporal weights (age arm only)."""
    backbone = _resolve_age_conditioned_backbone(model)
    if backbone is None:
        return
    device = next(model.parameters()).device
    tw = backbone.time_aware_attention.temporal_weight
    age_emb = backbone.time_aware_attention.age_emb
    base = tw.coefficients.detach()
    for age_yr in (0.1, 15.0):
        age_feat = age_emb(
            torch.tensor([float(age_yr)], device=device, dtype=torch.float32).clamp(min=0.0)
        )
        delta = tw.age_coeff_gen(age_feat).squeeze(0)
        alpha = base + delta
        norm = float(torch.norm(alpha - base, p=2).item())
        print(
            f"[post_run] ||alpha(a)-coefficients||_2 attention @ a={age_yr:g}yr: {norm:.6f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TALE-EHR on PIC disease tasks.")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=Path, required=True)
    parser.add_argument("--cohort_dir", type=Path, required=True)
    parser.add_argument("--tensorized_dir", type=Path, required=True)
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("data/processed/pic/code_vocab_pic.json"),
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        default=Path("data/processed/pic/bge_embeddings_pic.pt"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_kernel", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_workers", type=int, default=max(4, min(8, (os.cpu_count() or 16) - 2)))
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dry_run_one_epoch", action="store_true")
    return parser.parse_args()


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

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    console_log_path = run_dir / "console.log"
    console_fh = open(console_log_path, "w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, console_fh)
    sys.stderr = _TeeStream(orig_stderr, console_fh)

    try:
        run_t0 = time.perf_counter()
        print(f"[run] disease={args.disease} checkpoints -> {run_dir}", flush=True)
        print(f"[run] pretrained_ckpt={args.pretrained_ckpt}", flush=True)
        print(f"[run] console log -> {console_log_path}", flush=True)

        train_cohort = args.cohort_dir / "train_cohort.parquet"
        val_cohort = args.cohort_dir / "val_cohort.parquet"
        test_cohort = args.cohort_dir / "test_cohort.parquet"
        for p in (
            train_cohort,
            val_cohort,
            test_cohort,
            args.vocab_path,
            args.embedding_path,
            args.pretrained_ckpt,
        ):
            if not p.exists():
                raise FileNotFoundError(f"Missing required path: {p}")

        train_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "train", max_seq_len=args.max_seq_len, shard_cache_size=4)
        val_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "val", max_seq_len=args.max_seq_len, shard_cache_size=4)
        test_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "test", max_seq_len=args.max_seq_len, shard_cache_size=4)
        print(f"[data] tensorized_dir={args.tensorized_dir}", flush=True)

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

        pic_num_codes = _num_codes_from_vocab(args.vocab_path)
        device = torch.device(args.device)
        model = PICTALEEHRClassifier(
            pretrained_ckpt_path=args.pretrained_ckpt,
            pic_embedding_path=args.embedding_path,
            pic_num_codes=pic_num_codes,
            freeze_backbone=False,
        ).to(device)
        arm_type = _detect_arm_type(model)
        opt_groups, group_names = _build_optimizer_param_groups(
            model,
            lr_kernel=args.lr_kernel,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
        )
        _print_optimizer_groups(arm_type, group_names, args.lr_kernel, args.lr_backbone, args.lr_head)
        optimizer = torch.optim.AdamW(opt_groups)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Pre-train sanity: one forward/loss/backward check.
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
                if step % 100 == 0 or step == 1:
                    step_elapsed = time.perf_counter() - run_t0
                    print(
                        f"  ep{epoch:03d} step {step:6d} t+{step_elapsed:.1f}s loss={float(loss.item()):.6f}",
                        flush=True,
                    )

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            elapsed = time.perf_counter() - ep_start
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | val_AUROC={val_metrics['auroc']:.6f} | "
                f"val_AUPRC={val_metrics['auprc']:.6f} | val_acc={val_metrics['accuracy']:.6f} | "
                f"time={elapsed:.1f}s",
                flush=True,
            )
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
        _print_postrun_age_kernel_norms(model)
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
        leakage_gap = (
            float(test_metrics["auroc"]) - length_only_auroc
            if np.isfinite(float(test_metrics["auroc"])) and np.isfinite(length_only_auroc)
            else float("nan")
        )

        temperature = _fit_temperature(
            val_full["y_logit"].astype(np.float64),
            val_full["y_true"].astype(np.float64),
            device=device,
        )
        p_test_ts = _sigmoid_np(z_test / max(temperature, 1e-6))
        brier_raw = float(brier_score_loss(y_test, p_test)) if y_test.size > 0 else float("nan")
        brier_ts = float(brier_score_loss(y_test, p_test_ts)) if y_test.size > 0 else float("nan")
        ece_raw = _compute_ece(y_test.astype(np.float64), p_test, n_bins=15)
        ece_ts = _compute_ece(y_test.astype(np.float64), p_test_ts, n_bins=15)
        bootstrap = _bootstrap_metric_cis(y_test, p_test, n_bootstrap=1000, seed=42)

        decay_json_path = run_dir / "decay_alpha.json"
        decay_grid_csv_path = run_dir / "decay_kernel_grid.csv"
        decay_json: dict = {}
        decay_rows: list[dict[str, float]] = []
        age_backbone = _resolve_age_conditioned_backbone(model)
        if age_backbone is None:
            print("[decay] vanilla backbone, skipping", flush=True)
            decay_json = {"status": "vanilla_backbone_skipped"}
        else:
            from model.age_diagnostics import compute_alpha_delta_stats

            one_test_batch = next(iter(test_loader))
            one_test_batch = _move_batch_to_device(one_test_batch, device)
            alpha_stats = compute_alpha_delta_stats(age_backbone, one_test_batch)
            alpha_spread = _compute_alpha_band_spread(age_backbone, one_test_batch)
            decay_rows, decay_grid_stats = _compute_decay_grid(age_backbone, device)
            decay_json = {"status": "ok", "alpha_delta_stats": alpha_stats, "alpha_spread": alpha_spread, **decay_grid_stats}
        _write_decay_grid_csv(decay_grid_csv_path, decay_rows)
        with decay_json_path.open("w", encoding="utf-8") as f:
            json.dump(decay_json, f, indent=2)

        test_extended = {
            "length_only_auroc": float(length_only_auroc),
            "leakage_gap": float(leakage_gap),
            "auroc_ci": [float(bootstrap["auroc_ci"][0]), float(bootstrap["auroc_ci"][1])],
            "auprc_ci": [float(bootstrap["auprc_ci"][0]), float(bootstrap["auprc_ci"][1])],
            "brier_raw": float(brier_raw),
            "brier_ts": float(brier_ts),
            "ece_raw": float(ece_raw),
            "ece_ts": float(ece_ts),
            "temperature": float(temperature),
        }

        print(
            f"[test @ best epoch {best_epoch}] loss={test_loss:.6f} "
            f"AUROC={test_metrics['auroc']:.6f} AUPRC={test_metrics['auprc']:.6f} "
            f"acc={test_metrics['accuracy']:.6f}",
            flush=True,
        )
        print(
            "[test_extended] "
            f"length_only_AUROC={test_extended['length_only_auroc']:.6f} "
            f"leakage_gap={test_extended['leakage_gap']:.6f} "
            f"AUROC_CI95=({test_extended['auroc_ci'][0]:.6f},{test_extended['auroc_ci'][1]:.6f}) "
            f"brier_raw={test_extended['brier_raw']:.6f} brier_ts={test_extended['brier_ts']:.6f} "
            f"ece_raw={test_extended['ece_raw']:.6f} ece_ts={test_extended['ece_ts']:.6f} "
            f"T={test_extended['temperature']:.6f}",
            flush=True,
        )

        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "disease": args.disease,
                    "pretrained_ckpt": str(args.pretrained_ckpt),
                    "best_epoch": best_epoch,
                    "best_val_auroc": best_val_auroc,
                    "history": history,
                    "test_loss": test_loss,
                    "test_metrics": test_metrics,
                    "test_extended": test_extended,
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
