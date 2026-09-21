#!/usr/bin/env python3
"""Stage-2 NCH pediatric adaptation: no_interaction vs age_temporal.

    python -m stage2_nch.train --arm age_temporal --run_name adkm_nch_s0 --seed 0
    python -m stage2_nch.train --arm no_interaction --run_name nint_nch_s0 --seed 0

Both arms load the same Stage-1 ``adkm_s0`` checkpoint, reset β_P to 0, and
swap adult age μ/σ for z_P(a)=(a−9)/9. Do not launch full training unless asked.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_new.data import dataloader_worker_init, demo_layout
from model_new.diagnostics import write_json
from stage1_mimic_pretrain.evaluate import collect_batches, maybe_subset
from stage1_mimic_pretrain.model import build_param_groups
from stage2_nch.compatibility import write_compatibility_report
from stage2_nch.config import (
    ARM_CHOICES,
    CONSTANT_ATTENTION_AGE_YEARS,
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
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    EMBEDDING_PATH,
    EVAL_KS,
    N_SHUFFLE,
    NCH_SPLIT_DIR,
    NCH_TENSORIZED_DIR,
    PRIMARY_REPRESENTATION,
    PROBE_AGES_YEARS,
    REPO_ROOT,
    SHUFFLE_SEED,
    STAGE1_BEST_CKPT,
    STAGE2_RUN_ROOT,
    VOCAB_PATH,
    pediatric_age_transform_spec,
    resolve_arm,
    stage2_target_spec,
    tau_transform_spec,
)
from stage2_nch.dataset import NCHForecastDataset, make_nch_collate
from stage2_nch.evaluate import (
    epoch_diagnostics,
    evaluate_loader,
    run_age_tests,
    summarize_strata,
)
from stage2_nch.init_from_stage1 import build_stage2_model, logits_max_abs_diff
from stage2_nch.metrics import (
    evaluate_prevalence_baseline,
    history_bin_edges,
    param_l2_delta,
    pos_neg_bce,
)
from stage2_nch.plots import save_run_plots
from stage2_nch.sign_test import run_lambda0_sign_test
from stage2_nch.tensorize import ensure_forecast_shards

LOSS = torch.nn.BCEWithLogitsLoss()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARM_CHOICES, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_root", type=Path, default=STAGE2_RUN_ROOT)
    p.add_argument("--stage1_ckpt", type=Path, default=STAGE1_BEST_CKPT)
    p.add_argument("--tensorized_dir", type=Path,
                   default=NCH_TENSORIZED_DIR / PRIMARY_REPRESENTATION)
    p.add_argument("--embedding_path", type=Path, default=EMBEDDING_PATH)
    p.add_argument("--vocab_path", type=Path, default=VOCAB_PATH)
    p.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--race_encoding", choices=("one_hot", "scalar"), default="one_hot")
    p.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--n_heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--no_layernorm", action="store_true")
    p.add_argument("--no_ffn", action="store_true")
    p.add_argument("--ffn_mult", type=int, default=DEFAULT_FFN_MULT)
    p.add_argument("--demo_hidden", type=int, default=DEFAULT_DEMO_HIDDEN)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr_backbone", type=float, default=DEFAULT_LR_BACKBONE)
    p.add_argument("--lr_age", type=float, default=DEFAULT_LR_AGE)
    p.add_argument("--lr_head", type=float, default=DEFAULT_LR_HEAD)
    p.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    p.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--val_max_batches", type=int, default=0,
                   help="0 = full loader. NCH is small enough to score the whole split.")
    p.add_argument("--max_metric_examples", type=int, default=4096)
    p.add_argument("--age_test_batches", type=int, default=8)
    p.add_argument("--n_shuffle", type=int, default=N_SHUFFLE)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--pool_temporal_bias", action="store_true")
    p.add_argument("--skip_tensorize", action="store_true")
    p.add_argument("--force_tensorize", action="store_true")
    p.add_argument("--skip_init_identity", action="store_true")
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


def _peak_mem() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024 ** 2)


def _write_history_csv(path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = [
        "epoch", "step", "train_bce", "train_positive_bce", "train_negative_bce",
        "val_bce", "val_positive_bce", "val_negative_bce",
        "lr_backbone", "lr_age", "lr_head",
        "lambda0", "beta", "lambda0_grad_abs", "beta_grad_abs",
        "temporal_bias_abs_mean", "content_abs_mean", "ratio_temporal_abs_to_content_abs",
        "micro_auroc", "macro_auroc", "micro_auprc", "macro_auprc",
        "n_valid_classes_macro", "examples_per_sec", "epoch_s", "peak_gpu_mb",
        "delta_L_shuffle",
    ]
    keys += [f"recall@{k}" for k in EVAL_KS] + [f"precision@{k}" for k in EVAL_KS]
    extra = [f"lambda_a_{int(a)}" for a in PROBE_AGES_YEARS]
    fieldnames = keys + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rec in history:
            row = {k: rec.get(k) for k in keys}
            lam = rec.get("lambda_at_ages") or {}
            for a in PROBE_AGES_YEARS:
                row[f"lambda_a_{int(a)}"] = lam.get(str(a), lam.get(a))
            w.writerow(row)


def _assert_split_patients(train_ds, val_ds, test_ds, split_dir: Path) -> dict:
    raw = {}
    for name in ("train", "val", "test"):
        ids = json.loads((Path(split_dir) / f"{name}_patient_ids.json").read_text())
        raw[name] = set(int(x) for x in ids)
    overlap = {
        "train&val": len(raw["train"] & raw["val"]),
        "train&test": len(raw["train"] & raw["test"]),
        "val&test": len(raw["val"] & raw["test"]),
    }
    if any(overlap.values()):
        raise AssertionError(f"JSON split leakage: {overlap}")
    used = {}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if ds is None:
            used[name] = []
            continue
        pids = set(int(x) for x in ds.patient_ids().tolist())
        extra = pids - raw[name]
        if extra:
            raise AssertionError(f"{name} dataset has {len(extra)} patients not in split JSON")
        used[name] = sorted(pids)
    return {"json_n": {k: len(v) for k, v in raw.items()},
            "dataset_n": {k: len(v) for k, v in used.items()},
            "overlap": overlap}


def _prevalence_from_loader(loader: DataLoader, n_classes: int, max_batches: int = 0) -> np.ndarray:
    pos = np.zeros(n_classes, dtype=np.float64)
    n = 0
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        y = batch["target_codes"].numpy()
        pos += y.sum(axis=0)
        n += int(y.shape[0])
    return pos / max(n, 1)


def _cpu_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.arm = resolve_arm(args.arm)
    set_seed(args.seed)
    use_residual, use_layernorm, use_ffn = (
        not args.no_residual, not args.no_layernorm, not args.no_ffn)

    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                          else "cpu")

    tensorize_report = {}
    if not args.skip_tensorize:
        tensorize_report = ensure_forecast_shards(
            out_dir=args.tensorized_dir, force=args.force_tensorize)
        write_json(run_dir / "tensorize_report.json", tensorize_report)

    compat = write_compatibility_report(
        run_dir / "compatibility.json",
        vocab_path=args.vocab_path,
        embedding_path=args.embedding_path,
        stage1_ckpt=args.stage1_ckpt,
        tensorized_dir=args.tensorized_dir,
    )
    if not compat.get("passed"):
        raise RuntimeError(f"vocabulary/checkpoint compatibility failed: {compat}")

    sign = run_lambda0_sign_test(seed=args.seed)
    write_json(run_dir / "lambda0_sign_test.json", sign)

    train_ds = NCHForecastDataset(args.tensorized_dir / "train", args.vocab_path,
                                  max_seq_len=args.max_seq_len)
    val_path = args.tensorized_dir / "val"
    val_ds = (NCHForecastDataset(val_path, args.vocab_path, max_seq_len=args.max_seq_len)
              if val_path.exists() and any(val_path.glob("shard_*.npz")) else None)
    test_path = args.tensorized_dir / "test"
    test_ds = (NCHForecastDataset(test_path, args.vocab_path, max_seq_len=args.max_seq_len)
               if test_path.exists() and any(test_path.glob("shard_*.npz")) else None)
    train_ds = maybe_subset(train_ds, args.max_examples, args.seed)
    if val_ds is not None:
        val_ds = maybe_subset(val_ds, args.max_examples, args.seed + 1)
    if test_ds is not None:
        test_ds = maybe_subset(test_ds, args.max_examples, args.seed + 2)

    base_train = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    base_val = val_ds.dataset if val_ds is not None and hasattr(val_ds, "dataset") else val_ds
    base_test = test_ds.dataset if test_ds is not None and hasattr(test_ds, "dataset") else test_ds
    split_check = _assert_split_patients(base_train, base_val, base_test, NCH_SPLIT_DIR)
    write_json(run_dir / "split_ids.json", split_check)

    demo_dim, demo_channels = demo_layout(args.race_encoding)
    collate = make_nch_collate(args.race_encoding)
    if len(train_ds) == 0:
        raise RuntimeError(f"empty train split in {args.tensorized_dir / 'train'}")
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, collate,
                               drop_last=len(train_ds) >= args.batch_size)
    val_loader = (make_loader(val_ds, args.batch_size, False, args.num_workers, collate)
                  if val_ds is not None and len(val_ds) else None)
    test_loader = (make_loader(test_ds, args.batch_size, False, args.num_workers, collate)
                   if test_ds is not None and len(test_ds) else None)

    n_hist = []
    probe_n = min(len(train_ds), 2048)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(train_ds), size=probe_n, replace=False) if probe_n else []
    for i in pick:
        n_hist.append(int(train_ds[int(i)]["n_input_events"]))
    hist_edges = history_bin_edges(np.asarray(n_hist, dtype=np.float64))

    num_codes = base_train.num_codes
    model, transfer = build_stage2_model(
        num_codes=num_codes, arm=args.arm, embedding_path=args.embedding_path,
        stage1_ckpt=args.stage1_ckpt, seed=args.seed, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads, use_residual=use_residual,
        use_layernorm=use_layernorm, use_ffn=use_ffn, ffn_mult=args.ffn_mult,
        demo_dim=demo_dim, demo_channels=demo_channels, race_encoding=args.race_encoding,
        demo_hidden=args.demo_hidden, pool_temporal_bias=args.pool_temporal_bias,
    )
    model = model.to(device)
    clip_applied = bool((tensorize_report.get("age_out_of_range") or {}).get("clip_applied"))

    if not args.skip_init_identity:
        other_arm = "no_interaction" if args.arm == "age_temporal" else "age_temporal"
        other, _ = build_stage2_model(
            num_codes=num_codes, arm=other_arm, embedding_path=args.embedding_path,
            stage1_ckpt=args.stage1_ckpt, seed=args.seed, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads, use_residual=use_residual,
            use_layernorm=use_layernorm, use_ffn=use_ffn, ffn_mult=args.ffn_mult,
            demo_dim=demo_dim, demo_channels=demo_channels, race_encoding=args.race_encoding,
            demo_hidden=args.demo_hidden, pool_temporal_bias=args.pool_temporal_bias,
        )
        ident_batch = next(iter(make_loader(train_ds, min(4, len(train_ds)), False, 0, collate)))
        ident_batch = {k: (v if not isinstance(v, torch.Tensor) else v) for k, v in ident_batch.items()}
        diff = logits_max_abs_diff(model.cpu(), other.cpu(), ident_batch)
        model = model.to(device)
        write_json(run_dir / "init_identity.json", {
            "max_abs_logit_diff": diff,
            "passed": bool(diff < 1e-5),
            "beta_this": float(model.temporal.beta.detach().cpu()),
            "beta_other": float(other.temporal.beta.detach().cpu()),
        })
        if diff >= 1e-5:
            raise RuntimeError(f"arms differ at β=0 init: max|Δlogit|={diff}")
        del other

    groups, group_report = build_param_groups(model, args.lr_backbone, args.lr_age, args.lr_head)
    opt = torch.optim.Adam(groups)
    init_state = _cpu_state(model)

    prev_batches = min(32, max(1, len(train_loader)))
    p = _prevalence_from_loader(train_loader, num_codes, max_batches=prev_batches)
    write_json(run_dir / "prevalence_baseline.json", {
        "n_classes": int(p.size),
        "mean_prevalence": float(p.mean()),
        "n_always_zero": int((p == 0).sum()),
        "source": f"first {prev_batches} train batches",
    })

    config = {
        "run_id": args.run_name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": args.seed,
        "arm": args.arm,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "stage1": transfer,
        "lambda0_sign_test": {
            "positive_lambda0_is_recency": sign.get("positive_lambda0_is_recency"),
            "negative_lambda0_is_long_range": sign.get("negative_lambda0_is_long_range"),
            "interpretation": sign.get("interpretation"),
        },
        "model": model.config_dict(),
        "params": model.parameter_report(),
        "optim": {**group_report, "batch_size": args.batch_size, "epochs": args.epochs,
                  "patience": args.patience, "optimizer": "Adam", "grad_clip": args.grad_clip},
        "data": {
            "paths": {"tensorized_dir": str(args.tensorized_dir),
                      "embedding_path": str(args.embedding_path),
                      "vocab_path": str(args.vocab_path),
                      "stage1_ckpt": str(args.stage1_ckpt)},
            "split_sizes": {"train": len(train_ds),
                            "val": len(val_ds) if val_ds is not None else 0,
                            "test": len(test_ds) if test_ds is not None else 0},
            "max_seq_len": args.max_seq_len,
            "age_transform": pediatric_age_transform_spec(clip_applied=clip_applied),
            "tau_transform": tau_transform_spec(),
            "target": stage2_target_spec(),
            "history_bin_edges": hist_edges,
            "patient_splits": split_check,
            "tensorize": tensorize_report,
            "compatibility": {"passed": compat.get("passed")},
        },
        "env": {"torch": torch.__version__, "python": sys.version.split()[0],
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    # Overwrite adult age_transform written by Stage-1 config_dict.
    config["model"]["age_transform"] = pediatric_age_transform_spec(clip_applied=clip_applied)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "seed.json", {"seed": args.seed})

    history: list[dict] = []
    best_val = float("inf")
    best_auprc = -float("inf")
    best_epoch_bce = 0
    best_epoch_auprc = 0
    stale = 0
    step = 0
    t_start = time.time()
    diag_batch = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _ckpt(kind: str, epoch: int, val_bce: float, val_auprc: float | None = None) -> dict:
        return {
            "kind": kind, "epoch": epoch, "arm": args.arm, "seed": args.seed,
            "val_bce": val_bce, "val_micro_auprc": val_auprc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "age_transform": pediatric_age_transform_spec(clip_applied=clip_applied),
            "config": config,
        }

    def _eval_split(loader, collect=False):
        if loader is None:
            return {"bce": float("nan")}
        return evaluate_loader(
            model, loader, device, max_batches=args.val_max_batches,
            max_metric_examples=args.max_metric_examples, ks=EVAL_KS,
            history_edges=hist_edges, collect_strata=collect,
        )

    # Epoch 0: Stage-1 retention before any NCH update.
    val0 = _eval_split(val_loader)
    write_json(run_dir / "init_val_metrics.json", {
        "epoch": 0, "val_bce": val0.get("bce"), "micro_auprc": val0.get("micro_auprc"),
        "micro_auroc": val0.get("micro_auroc"),
        "note": "validation performance of the transferred Stage-1 weights, β_P=0, z_P ages",
    })

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = pos_run = neg_run = 0.0
        n_batches = n_pos = n_neg = 0
        g_l0 = g_b = 0.0
        n_g = 0
        n_ex = 0
        t_ep = time.time()
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
            pn = pos_neg_bce(out["code_logits"], batch["target_codes"])
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
            if pn["n_positive_labels"]:
                pos_run += pn["positive_bce"] * pn["n_positive_labels"]
                n_pos += pn["n_positive_labels"]
            if pn["n_negative_labels"]:
                neg_run += pn["negative_bce"] * pn["n_negative_labels"]
                n_neg += pn["n_negative_labels"]
            n_batches += 1
            n_ex += int(batch["target_codes"].shape[0])
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        epoch_s = time.time() - t_ep
        val = _eval_split(val_loader)
        diag = epoch_diagnostics(model, diag_batch, device) if diag_batch is not None else {}
        drift = param_l2_delta(model, init_state)
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        age_delta = float("nan")
        if val_loader is not None and (epoch == 1 or epoch % 2 == 0 or epoch == args.epochs):
            val_batches = collect_batches(val_loader, device, args.age_test_batches)
            age_now = run_age_tests(model, val_batches, device, n_shuffle=args.n_shuffle,
                                    seed=SHUFFLE_SEED + args.seed)
            age_delta = age_now.get("delta_L_shuffle")
            write_json(run_dir / "age_tests_latest.json", age_now)
        record = {
            "epoch": epoch,
            "step": step,
            "wall_clock_s": time.time() - t_start,
            "epoch_s": epoch_s,
            "examples_per_sec": n_ex / max(epoch_s, 1e-6),
            "peak_gpu_mb": _peak_mem(),
            "train_bce": running / max(1, n_batches),
            "train_positive_bce": pos_run / max(n_pos, 1) if n_pos else float("nan"),
            "train_negative_bce": neg_run / max(n_neg, 1) if n_neg else float("nan"),
            "val_bce": val["bce"],
            "val_positive_bce": val.get("positive_bce"),
            "val_negative_bce": val.get("negative_bce"),
            "val_micro_auprc": val.get("micro_auprc"),
            "lr_backbone": lrs.get("backbone"),
            "lr_age": lrs.get("age"),
            "lr_head": lrs.get("head"),
            "lambda0": float(model.temporal.lambda0.detach().cpu()),
            "beta": float(model.temporal.beta.detach().cpu()),
            "lambda0_grad_abs": g_l0 / max(n_g, 1),
            "beta_grad_abs": g_b / max(n_g, 1),
            "lambda_at_ages": model.temporal.lambda_at_ages(PROBE_AGES_YEARS),
            "delta_L_shuffle": age_delta,
            **{k: val.get(k) for k in (
                "micro_auroc", "macro_auroc", "micro_auprc", "macro_auprc",
                "n_valid_classes_macro", "prevalence",
            )},
            **{f"recall@{k}": val.get(f"recall@{k}") for k in EVAL_KS},
            **{f"precision@{k}": val.get(f"precision@{k}") for k in EVAL_KS},
            **diag,
            **drift,
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
            f"auprc={record.get('micro_auprc')} "
            f"λ0={record['lambda0']:+.4f} β={record['beta']:+.4f} "
            f"|gλ0|={record['lambda0_grad_abs']:.3e} |gβ|={record['beta_grad_abs']:.3e}",
            flush=True,
        )

        improved = val_loader is not None and val["bce"] == val["bce"] and val["bce"] < best_val - 1e-12
        if improved:
            best_val = float(val["bce"])
            best_epoch_bce = epoch
            stale = 0
            torch.save(_ckpt("best_bce", epoch, best_val, val.get("micro_auprc")),
                       run_dir / "checkpoint_best_bce.pt")
            torch.save(_ckpt("best_bce", epoch, best_val, val.get("micro_auprc")),
                       run_dir / "checkpoint_best.pt")
        else:
            stale += 1
        auprc = val.get("micro_auprc")
        if auprc is not None and auprc == auprc and auprc > best_auprc + 1e-12:
            best_auprc = float(auprc)
            best_epoch_auprc = epoch
            torch.save(_ckpt("best_auprc", epoch, val["bce"], best_auprc),
                       run_dir / "checkpoint_best_auprc.pt")
        if args.max_steps and step >= args.max_steps:
            break
        if args.patience and stale >= args.patience and epoch >= 2:
            print(f"early stop at epoch {epoch} (patience={args.patience})", flush=True)
            break

    torch.save(_ckpt("final", history[-1]["epoch"] if history else 0,
                     history[-1]["val_bce"] if history else float("nan"),
                     history[-1].get("micro_auprc") if history else None),
               run_dir / "checkpoint_final.pt")

    best_path = run_dir / "checkpoint_best_bce.pt"
    if not best_path.exists():
        best_path = run_dir / "checkpoint_final.pt"
    blob = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])

    val_batches = (collect_batches(val_loader, device, args.age_test_batches)
                   if val_loader is not None else collect_batches(train_loader, device, 2))
    age_tests = run_age_tests(model, val_batches, device, n_shuffle=args.n_shuffle,
                              seed=SHUFFLE_SEED + args.seed)
    write_json(run_dir / "age_tests.json", age_tests)

    val_metrics = _eval_split(val_loader, collect=True)
    test_metrics = _eval_split(test_loader, collect=True)

    def _drop_tensors(d: dict) -> dict:
        return {k: v for k, v in d.items()
                if k not in {"strata_rows", "_cap_logits", "_cap_targets"} and not torch.is_tensor(v)}

    age_val = summarize_strata(
        val_metrics.get("strata_rows") or [], val_metrics.get("_cap_logits"),
        val_metrics.get("_cap_targets"), key="age_band")
    hist_val = summarize_strata(
        val_metrics.get("strata_rows") or [], val_metrics.get("_cap_logits"),
        val_metrics.get("_cap_targets"), key="history_bin")
    age_test = summarize_strata(
        test_metrics.get("strata_rows") or [], test_metrics.get("_cap_logits"),
        test_metrics.get("_cap_targets"), key="age_band")
    hist_test = summarize_strata(
        test_metrics.get("strata_rows") or [], test_metrics.get("_cap_logits"),
        test_metrics.get("_cap_targets"), key="history_bin")
    write_json(run_dir / "age_stratified_metrics.json", {"val": age_val, "test": age_test})
    write_json(run_dir / "history_stratified_metrics.json", {"val": hist_val, "test": hist_test,
                                                             "edges": hist_edges})

    prev_targets = []
    if val_loader is not None:
        for i, batch in enumerate(val_loader):
            prev_targets.append(batch["target_codes"])
            if args.val_max_batches and i + 1 >= args.val_max_batches:
                break
            if len(prev_targets) >= 32:
                break
    if prev_targets:
        prev_metrics = evaluate_prevalence_baseline(p, torch.cat(prev_targets), ks=EVAL_KS)
        prev_path = run_dir / "prevalence_baseline.json"
        prev_json = json.loads(prev_path.read_text())
        prev_json["val"] = prev_metrics
        write_json(prev_path, prev_json)

    predictive = {
        "best_epoch_bce": best_epoch_bce,
        "best_val_bce": best_val,
        "best_epoch_auprc": best_epoch_auprc,
        "best_val_micro_auprc": best_auprc,
        "init_val_bce": val0.get("bce"),
        "constant_attention_age": CONSTANT_ATTENTION_AGE_YEARS,
        "val": _drop_tensors(val_metrics),
        "test": _drop_tensors(test_metrics),
        "age_tests": {
            "delta_L_shuffle": age_tests.get("delta_L_shuffle"),
            "delta_L_constant_mean": age_tests.get("delta_L_constant_mean"),
            "interpretation": age_tests.get("interpretation"),
        },
    }
    write_json(run_dir / "predictive_metrics.json", predictive)
    save_run_plots(run_dir, history, age_tests, age_val, hist_val)

    print(f"done run_dir={run_dir} best_epoch_bce={best_epoch_bce} "
          f"best_val_bce={best_val:.5f}", flush=True)
    print(f"age test: {age_tests.get('interpretation')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
