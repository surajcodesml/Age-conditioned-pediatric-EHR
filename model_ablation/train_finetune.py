#!/usr/bin/env python3
"""Arm-specific fine-tune on the CHD (heart-malformation) task.

Only ``--arm`` varies training behavior; data splits, seed, optimizer, hyper-
parameters, early-stop criterion, and eval are identical across the four arms
(enforced by shared defaults + a single code path). Locked: additive-logspace
kernel, no time/Weibull loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
    p.add_argument("--task_name", type=str, default="chd_heart_malformations",
                   help="label recorded in config.json (e.g. pneumonia, mortality, los_gt7)")
    p.add_argument("--pretrained_ckpt", type=Path, required=True)
    p.add_argument("--cohort_dir", type=Path, default=None)
    p.add_argument("--tensorized_dir", type=Path, default=None)
    p.add_argument("--events_parquet", type=Path, default=REPO_ROOT / "data/processed/patient_events_rolled_full.parquet")
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6,
                   help="early-stop after this many epochs with no val-AUPRC improvement")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_age", type=float, default=1e-3,
                   help="LR for the dedicated age-injection param group (kernel/additive/random_constant)")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--max_rows", type=int, default=None,
                   help="cap #samples per split (smoke tier 1); None = full data")
    p.add_argument("--seed", type=int, default=42)  # SHARED across arms
    p.add_argument("--num_workers", type=int, default=max(2, min(8, (os.cpu_count() or 8) - 2)))
    p.add_argument("--log_every", type=int, default=500, help="print a progress heartbeat every N steps")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--run_dir", type=Path, default=None)
    p.add_argument("--dry_run_one_epoch", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _grad_l2(params) -> float:
    """L2 norm of grads over a param subset (pre-clip). 0.0 if the subset has no grads
    (e.g. arms whose age pathway carries no parameters)."""
    sq, found = 0.0, False
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().norm().item()) ** 2
            found = True
    return sq ** 0.5 if found else 0.0


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
    if args.max_rows is not None:  # smoke tier 1: truncate each split's index in place
        for ds in (train_ds, val_ds, test_ds):
            if hasattr(ds, "_index"):
                del ds._index[args.max_rows:]
            elif hasattr(ds, "_rows"):
                del ds._rows[args.max_rows:]
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

    # INV-demo: age must never enter through demographics. demo_proj consumes sex/race only.
    assert model.backbone.demo_dim == 2, f"demo_dim must be 2 (sex, race); got {model.backbone.demo_dim}"

    # ---- dedicated age-injection optimizer group (the fix) --------------------------
    # Age-injection params = the kernel Delta-alpha generator (age_coeff_gen, both encoder-
    # attention and aggregation stages) and the additive-embedding MLP (additive_age_emb).
    # These previously sat inside the backbone group and trained at lr_backbone (1e-5),
    # which left them effectively frozen. We split them into their own group at lr_age so
    # `age params train at lr_age, everything else unchanged`. Partition by param id so no
    # param lands in two groups. Applied UNIFORMLY across arms:
    #   kernel / random_constant -> age_coeff_gen params (random_constant kept in the age
    #                               group so it stays capacity- AND lr-matched to kernel;
    #                               only the age INPUT differs between them);
    #   additive                 -> additive_age_emb params;
    #   vanilla                  -> no such params -> age group empty (omitted).
    # Base kernel coefficients (temporal_weight.coefficients) intentionally stay in the
    # backbone group for ALL arms ("everything else unchanged").
    def _is_age_name(n: str) -> bool:
        return ("age_coeff_gen" in n) or ("additive_age_emb" in n)

    age_named = [(n, p) for n, p in model.backbone.named_parameters() if _is_age_name(n)]
    age_params = [p for _, p in age_named]
    age_param_ids = {id(p) for p in age_params}
    backbone_params = [p for p in model.backbone.parameters() if id(p) not in age_param_ids]

    param_groups = [
        {"name": "backbone", "params": backbone_params, "lr": args.lr_backbone},
        {"name": "head", "params": list(model.classifier.parameters()), "lr": args.lr_head},
    ]
    if age_params:  # vanilla contributes no age params -> keep the group empty (omit)
        param_groups.append({"name": "age", "params": age_params, "lr": args.lr_age})
    optimizer = torch.optim.AdamW(param_groups)

    # Init snapshots so we can prove the age path actually MOVED. With Adam, the update
    # step is ~= lr even when the gradient is tiny, so grad-norm alone (small for the
    # kernel path by construction) does NOT show learning; parameter drift does.
    age_init = [p.detach().clone() for p in age_params]
    _tw = model.backbone.time_aware_attention.temporal_weight
    coeff_init = _tw.coefficients.detach().clone()  # age-free base kernel (stays in backbone group)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- config snapshot: proves all arms share the same backbone + settings -----
    config = {
        "task": args.task_name,
        "arm": args.arm,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "cohort_dir": str(args.cohort_dir) if args.cohort_dir else None,
        "tensorized_dir": str(args.tensorized_dir) if args.tensorized_dir else None,
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "pretrained_ckpt_sha256": _sha256(Path(args.pretrained_ckpt)),
        "pos_weight": pos_weight,
        "demo_dim": int(model.backbone.demo_dim),  # asserted == 2 above (no age leak via demographics)
        "age_pathway_param_count": int(model.backbone.age_pathway_param_count()),
        "age_pathway_param_names": [n for n, _ in age_named],
        "selection_metric": "val_auprc (max)",  # PRE-DECLARED before runs
        "early_stopping": {"metric": "val_auprc", "mode": "max", "patience": args.patience},
        "hyperparameters": {
            "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone, "lr_head": args.lr_head, "lr_age": args.lr_age,
            "max_seq_len": args.max_seq_len, "max_rows": args.max_rows, "num_workers": args.num_workers,
            "grad_clip_max_norm": 1.0, "optimizer": "AdamW", "amp": use_amp,
        },
        "lr_param_groups": [
            {"name": g.get("name", f"group{i}"), "lr": float(g["lr"]),
             "n_params": int(sum(p.numel() for p in g["params"])),
             "n_tensors": int(len(g["params"]))}
            for i, g in enumerate(optimizer.param_groups)
        ],
        "age_pathway_lr_group": "age",  # age-injection params now train at lr_age
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    for g in config["lr_param_groups"]:
        print(f"[optim] group={g['name']:<8} lr={g['lr']:.1e} n_params={g['n_params']:>10} "
              f"n_tensors={g['n_tensors']}", flush=True)

    best_auprc, best_epoch = -float("inf"), 0  # PRE-DECLARED selection metric: val AUPRC
    best_ckpt = run_dir / "best.pt"
    history: list[dict[str, Any]] = []
    epochs_no_improve = 0
    total_epochs = 1 if args.dry_run_one_epoch else args.epochs
    for epoch in range(1, total_epochs + 1):
        model.train()
        tl = []
        grad_norms: list[float] = []
        age_grad_norms: list[float] = []
        ep_t0 = time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                loss = criterion(model(batch), batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            age_grad_norms.append(_grad_l2(age_params))  # pre-clip age-pathway grad norm
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            tl.append(float(loss.item()))
            grad_norms.append(float(gn))  # pre-clip total grad norm
            if args.log_every and step % args.log_every == 0:
                elapsed = time.perf_counter() - ep_t0
                recent = tl[-args.log_every:]
                print(
                    f"  [{args.arm} epoch {epoch:03d} step {step:>6}] "
                    f"running_loss={np.mean(recent):.6f} gradnorm={grad_norms[-1]:.4f} "
                    f"age_gradnorm={age_grad_norms[-1]:.3e} {step / max(elapsed, 1e-9):.1f} it/s",
                    flush=True,
                )
        ep_secs = time.perf_counter() - ep_t0

        # Age-pathway parameter DRIFT from init (proves the path moved despite tiny grads).
        with torch.no_grad():
            age_drift = float(np.sqrt(sum(float((p - p0).norm().item()) ** 2
                                          for p, p0 in zip(age_params, age_init)))) if age_params else 0.0
            coeff_delta = (_tw.coefficients.detach() - coeff_init).cpu().numpy()
            coeff_drift = float(np.linalg.norm(coeff_delta))

        val_loss, vm, yt, yp, age = evaluate(model, val_loader, criterion, device)
        val_strat = age_stratified(yt, yp, np.clip(age, 0.0, None))
        print(f"Epoch {epoch:03d} | train_loss={np.mean(tl):.6f} | val_loss={val_loss:.6f} | "
              f"val_AUPRC={vm['auprc']:.6f} | val_AUROC={vm['auroc']:.6f} | "
              f"age_gradnorm(mean)={np.mean(age_grad_norms) if age_grad_norms else 0.0:.3e} | "
              f"age_drift={age_drift:.3e}", flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(tl)),
            "val_loss": float(val_loss),
            "val_auprc": vm["auprc"],   # PRIMARY
            "val_auroc": vm["auroc"],   # secondary
            "val_age_stratified": val_strat,
            "age_pathway_grad_norm_preclip": {
                "mean": float(np.mean(age_grad_norms)) if age_grad_norms else 0.0,
                "max": float(np.max(age_grad_norms)) if age_grad_norms else 0.0,
                "last": float(age_grad_norms[-1]) if age_grad_norms else 0.0,
            },
            # Cumulative drift from init: the real evidence the age path learned. Adam makes
            # the step ~= lr regardless of grad size, so this should be non-trivial once the
            # age params sit in their own lr_age group.
            "age_pathway_drift_l2": age_drift,
            "base_coeff_drift_l2": coeff_drift,
            "base_coeff_delta": [float(x) for x in coeff_delta],
            "total_grad_norm_preclip": {
                "mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                "max": float(np.max(grad_norms)) if grad_norms else float("nan"),
            },
            "lr_param_groups": [float(g["lr"]) for g in optimizer.param_groups],
            "epoch_seconds": ep_secs,
        })
        # Checkpoint selection on val AUPRC (pre-declared).
        rank = vm["auprc"] if np.isfinite(vm["auprc"]) else -float("inf")
        if rank > best_auprc:
            best_auprc, best_epoch = rank, epoch
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "arm": args.arm, "model_state_dict": model.state_dict()}, best_ckpt)
        else:
            epochs_no_improve += 1
        # Persist history after every epoch so a crash still leaves the convergence curve.
        with (run_dir / "history.json").open("w") as f:
            json.dump({"arm": args.arm, "config": config, "best_epoch": best_epoch,
                       "best_val_auprc": best_auprc if np.isfinite(best_auprc) else None,
                       "selection_metric": "val_auprc (max)", "history": history}, f, indent=2)
        # Early stopping on val AUPRC (disabled for dry-run).
        if not args.dry_run_one_epoch and epochs_no_improve >= args.patience:
            print(f"[early-stop] no val_AUPRC improvement for {epochs_no_improve} epochs "
                  f"(patience={args.patience}); best epoch {best_epoch} AUPRC={best_auprc:.6f}", flush=True)
            break

    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
    test_loss, tm, yt, yp, age = evaluate(model, test_loader, criterion, device)
    strat = age_stratified(yt, yp, np.clip(age, 0.0, None))
    print(f"[test @ best epoch {best_epoch}] AUPRC={tm['auprc']:.6f} AUROC={tm['auroc']:.6f}", flush=True)
    print("[age_stratified test]", flush=True)
    for band, rec in strat.items():
        print(f"  {band:>6}: n={rec['n']:5d} AUROC={rec['auroc']:.4f} AUPRC={rec['auprc']:.4f}", flush=True)
    with (run_dir / "history.json").open("w") as f:
        json.dump({"arm": args.arm, "config": config, "best_epoch": best_epoch,
                   "best_val_auprc": best_auprc if np.isfinite(best_auprc) else None,
                   "selection_metric": "val_auprc (max)", "history": history,
                   "test_metrics": tm, "test_age_stratified": strat}, f, indent=2)

    # Deterministic teardown (see train.py): spawned DataLoader workers can SIGABRT in a
    # C-extension destructor at interpreter shutdown, flipping the exit code even though
    # all checkpoints and history.json are already written. Hard-exit past that race.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
