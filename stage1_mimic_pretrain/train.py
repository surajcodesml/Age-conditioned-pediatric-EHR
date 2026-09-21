#!/usr/bin/env python3
"""Stage-1 MIMIC-IV pretraining: no_interaction vs age-conditioned self-attention.

    python -m stage1_mimic_pretrain.train --arm no_interaction --run_name nint_s0
    python -m stage1_mimic_pretrain.train --arm age_temporal   --run_name adkm_s0

The two arms share seed, data, architecture, optimizer, splits, and demographic
age. The only difference is whether β is trained in Transformer self-attention
or frozen at 0. Pooling does not use λ0/β.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_new.data import (
    TensorizedPretrainDataset,
    corpus_stats_cached,
    dataloader_worker_init,
    demo_layout,
    make_collate,
)
from model_new.diagnostics import write_json
from stage1_mimic_pretrain.config import (
    ARM_CHOICES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_D_MODEL,
    DEFAULT_DEMO_HIDDEN,
    DEFAULT_EPOCHS,
    DEFAULT_FFN_MULT,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR_AGE,
    DEFAULT_LR_BACKBONE,
    DEFAULT_LR_HEAD,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    EMBEDDING_PATH,
    EVAL_KS,
    MIMIC_AGE_MEAN_YEARS,
    MIMIC_AGE_STD_YEARS,
    N_SHUFFLE,
    PROBE_AGES_YEARS,
    REPO_ROOT,
    SHUFFLE_SEED,
    TENSORIZED_DIR,
    VOCAB_PATH,
    age_transform_spec,
    target_spec,
    tau_transform_spec,
    resolve_arm,
)
from stage1_mimic_pretrain.evaluate import (
    collect_batches,
    epoch_diagnostics,
    evaluate_loader,
    maybe_subset,
    run_age_tests,
    save_split_ids,
)
from stage1_mimic_pretrain.metrics import class_imbalance_report
from stage1_mimic_pretrain.model import MinimalDKMModel, build_param_groups
from stage1_mimic_pretrain.plots import save_run_plots

LOSS = torch.nn.BCEWithLogitsLoss()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARM_CHOICES, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_root", type=Path, default=REPO_ROOT / "stage1_mimic_pretrain/run")
    p.add_argument("--tensorized_dir", type=Path, default=TENSORIZED_DIR)
    p.add_argument("--embedding_path", type=Path, default=EMBEDDING_PATH)
    p.add_argument("--vocab_path", type=Path, default=VOCAB_PATH)
    p.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--race_encoding", choices=("one_hot", "scalar"), default="one_hot")
    p.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--legacy_block", action="store_true")
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--no_layernorm", action="store_true")
    p.add_argument("--no_ffn", action="store_true")
    p.add_argument("--ffn_mult", type=int, default=DEFAULT_FFN_MULT)
    p.add_argument("--demo_hidden", type=int, default=DEFAULT_DEMO_HIDDEN)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr_backbone", type=float, default=DEFAULT_LR_BACKBONE)
    p.add_argument("--lr_age", type=float, default=DEFAULT_LR_AGE)
    p.add_argument("--lr_head", type=float, default=DEFAULT_LR_HEAD)
    p.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val_max_batches", type=int, default=50)
    p.add_argument("--age_test_batches", type=int, default=8)
    p.add_argument("--n_shuffle", type=int, default=N_SHUFFLE)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_examples", type=int, default=0,
                   help="If >0, subsample this many train/val/test examples (same seed).")
    p.add_argument("--max_shards", type=int, default=0,
                   help="If >0, symlink this many shards into a temp split dir.")
    p.add_argument("--skip_corpus_stats", action="store_true",
                   help="Use frozen MIMIC μ/σ instead of recomputing corpus_stats.")
    p.add_argument("--skip_split_ids", action="store_true")
    p.add_argument("--age_mean", type=float, default=None)
    p.add_argument("--age_sd", type=float, default=None)
    p.add_argument("--age_median", type=float, default=None)
    p.add_argument("--pool_temporal_bias", action="store_true",
                   help="Optional ablation: also apply λ0/β in pooling. Off for Stage-1.")
    return p


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_block_flags(args) -> tuple[bool, bool, bool]:
    if args.legacy_block:
        return False, False, False
    return not args.no_residual, not args.no_layernorm, not args.no_ffn


def _symlink_shard_subset(src: Path, dst: Path, n_shards: int, seed: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz in {src}")
    # First N shards (sorted). Random sampling can yield a val shard with no
    # two-visit patients, which makes the forecasting dataset empty.
    _ = seed
    pick = shards if (n_shards <= 0 or n_shards >= len(shards)) else shards[:n_shards]
    for p in pick:
        target = dst / p.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(p.resolve(), target)


def maybe_restrict_tensorized(tensorized_dir: Path, run_dir: Path, max_shards: int,
                              seed: int) -> Path:
    if max_shards <= 0:
        return tensorized_dir
    out = run_dir / "data_subset"
    for split in ("train", "val", "test"):
        src = tensorized_dir / split
        if src.exists():
            _symlink_shard_subset(src, out / split, max_shards, seed)
    return out


def make_loader(ds, batch_size, shuffle, num_workers, collate, drop_last=False) -> DataLoader:
    kw: dict = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                    collate_fn=collate, num_workers=num_workers,
                    worker_init_fn=dataloader_worker_init)
    if num_workers > 0:
        kw["pin_memory"] = True
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


def _grad_abs(param: torch.nn.Parameter) -> float:
    g = param.grad
    if g is None:
        return 0.0
    return float(g.detach().abs().mean().cpu())


def _write_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = [
        "epoch", "step", "train_bce", "val_bce", "lr_backbone", "lr_age", "lr_head",
        "lambda0", "beta", "lambda0_grad_abs", "beta_grad_abs",
        "temporal_bias_abs_mean", "content_abs_mean", "ratio_temporal_abs_to_content_abs",
        "micro_auroc", "macro_auroc", "micro_auprc", "macro_auprc",
    ]
    keys += [f"recall@{k}" for k in EVAL_KS] + [f"precision@{k}" for k in EVAL_KS]
    extra_age = [f"lambda_a_{int(a)}" for a in PROBE_AGES_YEARS]
    fieldnames = keys + extra_age
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rec in history:
            row = {k: rec.get(k) for k in keys}
            lam = rec.get("lambda_at_ages") or {}
            for a in PROBE_AGES_YEARS:
                row[f"lambda_a_{int(a)}"] = lam.get(str(a), lam.get(a))
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.arm = resolve_arm(args.arm)
    set_seed(args.seed)
    use_residual, use_layernorm, use_ffn = resolve_block_flags(args)

    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                              else "cpu")

    tensorized_dir = maybe_restrict_tensorized(
        Path(args.tensorized_dir), run_dir, args.max_shards, args.seed)

    train_ds = TensorizedPretrainDataset(tensorized_dir / "train", args.vocab_path,
                                         max_seq_len=args.max_seq_len)
    val_path = tensorized_dir / "val"
    val_ds = (TensorizedPretrainDataset(val_path, args.vocab_path, max_seq_len=args.max_seq_len)
              if val_path.exists() and any(val_path.glob("shard_*.npz")) else None)
    test_path = tensorized_dir / "test"
    test_ds = (TensorizedPretrainDataset(test_path, args.vocab_path, max_seq_len=args.max_seq_len)
               if test_path.exists() and any(test_path.glob("shard_*.npz")) else None)
    train_ds = maybe_subset(train_ds, args.max_examples, args.seed)
    if val_ds is not None:
        val_ds = maybe_subset(val_ds, args.max_examples, args.seed + 1)
    if test_ds is not None:
        test_ds = maybe_subset(test_ds, args.max_examples, args.seed + 2)

    demo_dim, demo_channels = demo_layout(args.race_encoding)
    collate = make_collate(args.race_encoding)
    if len(train_ds) == 0:
        raise RuntimeError(f"empty train split in {tensorized_dir / 'train'}")
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, collate,
                               drop_last=len(train_ds) >= args.batch_size)
    val_loader = (make_loader(val_ds, args.batch_size, False, args.num_workers, collate)
                  if val_ds is not None and len(val_ds) else None)
    test_loader = (make_loader(test_ds, args.batch_size, False, args.num_workers, collate)
                   if test_ds is not None and len(test_ds) else None)

    if args.skip_corpus_stats:
        age_mean = float(args.age_mean if args.age_mean is not None else MIMIC_AGE_MEAN_YEARS)
        age_sd = float(args.age_sd if args.age_sd is not None else MIMIC_AGE_STD_YEARS)
        age_median = float(args.age_median if args.age_median is not None else 58.702868326488705)
        stats_json = {"source": "frozen_mimic_defaults", "age_mean": age_mean,
                      "age_sd": age_sd, "age_median": age_median}
    else:
        stats = corpus_stats_cached(
            train_ds if not hasattr(train_ds, "dataset") else train_ds.dataset,
            tensorized_dir / "train", split="train", sample_windows=200, seed=args.seed,
            max_seq_len=args.max_seq_len)
        age_mean = float(args.age_mean if args.age_mean is not None else stats.event_age_mean)
        age_sd = float(args.age_sd if args.age_sd is not None else stats.event_age_sd)
        age_median = float(args.age_median if args.age_median is not None
                           else stats.event_age_median)
        stats_json = stats.to_json()

    model = MinimalDKMModel(
        num_codes=(train_ds.dataset.num_codes if hasattr(train_ds, "dataset")
                   else train_ds.num_codes),
        embedding_path=args.embedding_path, arm=args.arm, seed=args.seed,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        use_residual=use_residual, use_layernorm=use_layernorm, use_ffn=use_ffn,
        ffn_mult=args.ffn_mult, demo_dim=demo_dim, demo_channels=demo_channels,
        race_encoding=args.race_encoding, demo_hidden=args.demo_hidden,
        age_mean=age_mean, age_sd=age_sd,
        pool_temporal_bias=args.pool_temporal_bias,
    ).to(device)

    groups, group_report = build_param_groups(model, args.lr_backbone, args.lr_age, args.lr_head)
    opt = torch.optim.Adam(groups)

    split_summary = {}
    if not args.skip_split_ids:
        split_summary = save_split_ids(run_dir, tensorized_dir)

    # Imbalance on a small train sample (existing pipeline: no pos_weight).
    imb_targets = []
    for i, batch in enumerate(train_loader):
        imb_targets.append(batch["target_codes"])
        if i >= 7:
            break
    imbalance = class_imbalance_report(torch.cat(imb_targets)) if imb_targets else {}

    config = {
        "run_id": args.run_name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": args.seed,
        "arm": args.arm,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": model.config_dict(),
        "params": model.parameter_report(),
        "optim": {**group_report, "batch_size": args.batch_size, "epochs": args.epochs,
                  "optimizer": "Adam", "grad_clip": args.grad_clip},
        "data": {
            "paths": {"tensorized_dir": str(tensorized_dir),
                      "embedding_path": str(args.embedding_path),
                      "vocab_path": str(args.vocab_path)},
            "split_sizes": {"train": len(train_ds),
                            "val": len(val_ds) if val_ds is not None else 0,
                            "test": len(test_ds) if test_ds is not None else 0},
            "max_seq_len": args.max_seq_len,
            "age_transform": age_transform_spec(age_mean, age_sd),
            "tau_transform": tau_transform_spec(),
            "target": target_spec(),
            "corpus_stats": stats_json,
            "patient_splits": split_summary,
            "class_imbalance": imbalance,
        },
        "env": {"torch": torch.__version__, "python": sys.version.split()[0],
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "seed.json", {"seed": args.seed})

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    step = 0
    t_start = time.time()
    diag_batch = None
    model.train()

    def _ckpt(kind: str, epoch: int, val_bce: float) -> dict:
        return {
            "kind": kind, "epoch": epoch, "arm": args.arm, "seed": args.seed,
            "val_bce": val_bce,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "age_standardization": {"mean": age_mean, "sd": age_sd, "median": age_median},
            "config": config,
        }

    for epoch in range(1, args.epochs + 1):
        running, n_batches = 0.0, 0
        g_l0, g_b, n_g = 0.0, 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            if diag_batch is None:
                diag_batch = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                              for k, v in batch.items()}
            out = model(batch)
            loss = LOSS(out["code_logits"].float(), batch["target_codes"].float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite train loss at step {step}: {float(loss)}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g_l0 += _grad_abs(model.temporal.lambda0)
            g_b += _grad_abs(model.temporal.beta)
            n_g += 1
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], args.grad_clip)
            opt.step()
            running += loss.item()
            n_batches += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        val = (evaluate_loader(model, val_loader, device, max_batches=args.val_max_batches)
               if val_loader is not None else {"bce": float("nan")})
        diag = epoch_diagnostics(model, diag_batch, device) if diag_batch is not None else {}
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        record = {
            "epoch": epoch,
            "step": step,
            "wall_clock_s": time.time() - t_start,
            "train_bce": running / max(1, n_batches),
            "val_bce": val["bce"],
            "lr_backbone": lrs.get("backbone"),
            "lr_age": lrs.get("age"),
            "lr_head": lrs.get("head"),
            "lambda0": float(model.temporal.lambda0.detach().cpu()),
            "beta": float(model.temporal.beta.detach().cpu()),
            "lambda0_grad_abs": g_l0 / max(n_g, 1),
            "beta_grad_abs": g_b / max(n_g, 1),
            "lambda_at_ages": model.temporal.lambda_at_ages(PROBE_AGES_YEARS),
            **{k: val.get(k) for k in (
                "micro_auroc", "macro_auroc", "micro_auprc", "macro_auprc",
                "n_valid_classes_macro", "prevalence",
            )},
            **{f"recall@{k}": val.get(f"recall@{k}") for k in EVAL_KS},
            **{f"precision@{k}": val.get(f"precision@{k}") for k in EVAL_KS},
            **diag,
        }
        history.append(record)
        write_json(run_dir / "history.json", history)
        _write_history_csv(run_dir / "metrics.csv", history)
        write_json(run_dir / "lambda_trajectory.json", [
            {"epoch": r["epoch"], "lambda0": r["lambda0"], "beta": r["beta"],
             "lambda_at_ages": r["lambda_at_ages"]}
            for r in history
        ])
        print(
            f"[{args.arm} seed={args.seed}] epoch {epoch:02d} "
            f"train_bce={record['train_bce']:.5f} val_bce={record['val_bce']:.5f} "
            f"λ0={record['lambda0']:+.4f} β={record['beta']:+.4f} "
            f"|gλ0|={record['lambda0_grad_abs']:.3e} |gβ|={record['beta_grad_abs']:.3e}",
            flush=True,
        )

        torch.save(_ckpt("epoch", epoch, val["bce"]), run_dir / f"epoch_{epoch:03d}.pt")
        if val_loader is not None and val["bce"] == val["bce"] and val["bce"] < best_val - 1e-12:
            best_val = float(val["bce"])
            best_epoch = epoch
            torch.save(_ckpt("best", epoch, best_val), run_dir / "checkpoint_best.pt")
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(_ckpt("final", history[-1]["epoch"] if history else 0,
                     history[-1]["val_bce"] if history else float("nan")),
               run_dir / "checkpoint_final.pt")

    # Reload best for age tests + predictive metrics.
    best_path = run_dir / "checkpoint_best.pt"
    if best_path.exists():
        blob = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state_dict"])
    val_batches = (collect_batches(val_loader, device, args.age_test_batches)
                   if val_loader is not None else collect_batches(train_loader, device, 2))
    age_tests = run_age_tests(model, val_batches, device, age_mean, age_median,
                              n_shuffle=args.n_shuffle, seed=SHUFFLE_SEED + args.seed)
    write_json(run_dir / "age_tests.json", age_tests)

    val_metrics = (evaluate_loader(model, val_loader, device, max_batches=args.val_max_batches)
                   if val_loader is not None else {})
    test_metrics = (evaluate_loader(model, test_loader, device, max_batches=args.val_max_batches)
                    if test_loader is not None else {})
    predictive = {
        "best_epoch": best_epoch,
        "best_val_bce": best_val,
        "val": val_metrics,
        "test": test_metrics,
        "age_tests": {
            "delta_L_shuffle": age_tests.get("delta_L_shuffle"),
            "interpretation": age_tests.get("interpretation"),
        },
    }
    write_json(run_dir / "predictive_metrics.json", predictive)
    save_run_plots(run_dir, history, age_tests)

    print(f"done run_dir={run_dir} best_epoch={best_epoch} best_val_bce={best_val:.5f}",
          flush=True)
    print(f"age test: {age_tests.get('interpretation')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
