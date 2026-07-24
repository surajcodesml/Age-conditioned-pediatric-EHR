#!/usr/bin/env python3
"""SHARED vanilla pretrain for the ablation (code loss only; no time/Weibull loss).

Produces the single backbone that all four fine-tune arms load. Age conditioning
is NOT present here (vanilla); it is introduced only at fine-tune, symmetrically
for the two age arms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_ablation.arms import ARMS
from model_ablation.dataset import TensorizedEHRDataset, _dataloader_worker_init, ehr_collate
from model_ablation.tale_ehr_age import TALEEHRAblation
from model_ablation.time_aware_attention_age import CHEB_TMAX, _chebyshev_powers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared vanilla pretrain (ablation).")
    p.add_argument("--tensorized_dir", type=Path, default=REPO_ROOT / "data/processed/tensorized_flat")
    p.add_argument("--embedding_path", type=Path, default=REPO_ROOT / "data/processed/bge_embeddings.pt")
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--arm", type=str, default="vanilla", choices=ARMS,
                   help="age-conditioning arm; the shared backbone pretrain is 'vanilla'")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--poly_degree", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=500, help="print a progress heartbeat every N steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_max_batches", type=int, default=50,
                   help="cap val batches per epoch eval (val set is huge); 0 = full val")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--save_dir", type=Path, default=REPO_ROOT / "checkpoints/ablation_pretrain")
    p.add_argument("--resume_from", type=Path, default=None,
                   help="checkpoint to resume from; continues at its epoch+1 up to --epochs")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Logging / diagnostics helpers (read-only; do not affect the training math).  #
# --------------------------------------------------------------------------- #
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


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def compute_metrics(code_logits: torch.Tensor, target_codes: torch.Tensor,
                    ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    """Next-code recall@k and (raveled) AUROC. Ported from model/train.py."""
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


@torch.no_grad()
def evaluate_pretrain(model, loader, device, use_amp, max_batches: int) -> dict[str, float]:
    """Val next-code BCE + recall@k + AUROC, averaged over (capped) val batches."""
    was_training = model.training
    model.eval()
    bces: list[float] = []
    agg: dict[str, list[float]] = {}
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            out = model(batch)
            loss = F.binary_cross_entropy_with_logits(out["code_logits"], batch["target_codes"])
        bces.append(float(loss.item()))
        for mk, mv in compute_metrics(out["code_logits"].float(), batch["target_codes"]).items():
            agg.setdefault(mk, []).append(mv)
    if was_training:
        model.train()
    res: dict[str, float] = {"val_bce": float(np.mean(bces)) if bces else float("nan")}
    for mk, vals in agg.items():
        res[f"val_{mk}"] = float(np.mean(vals)) if vals else float("nan")
    return res


@torch.no_grad()
def qk_norm_stats(model, batch) -> dict[str, float]:
    """Per-token ||q||, ||k|| over valid positions (evidence the kernel path is stable)."""
    ce = model.embedding_table[batch["code_indices"]]
    q = model.time_aware_attention.mlp_q(ce)
    k = model.time_aware_attention.mlp_k(ce)
    mask = batch["attention_mask"]
    qn = q.norm(dim=-1)[mask].float()
    kn = k.norm(dim=-1)[mask].float()
    return {
        "q_norm_mean": float(qn.mean()) if qn.numel() else float("nan"),
        "q_norm_std": float(qn.std()) if qn.numel() else float("nan"),
        "k_norm_mean": float(kn.mean()) if kn.numel() else float("nan"),
        "k_norm_std": float(kn.std()) if kn.numel() else float("nan"),
    }


@torch.no_grad()
def w_curve(model) -> dict[str, Any]:
    """The single global age-free w(Δt) kernel on a canonical Δt grid (vanilla: Δα≡0)."""
    days = [0.0, 1.0, 7.0, 30.0, 90.0, 365.0, 730.0, 1095.0]
    log_dt = torch.log1p(torch.tensor(days) / 7.0)
    x = 2.0 * log_dt / CHEB_TMAX - 1.0

    def curve(tw) -> list[float]:
        coeffs = tw.coefficients.detach().cpu()
        basis = _chebyshev_powers(x, tw.poly_degree)
        poly = torch.zeros_like(log_dt)
        for k in range(tw.poly_degree + 1):
            poly = poly + coeffs[k] * basis[k]
        return torch.sigmoid(poly).tolist()

    return {
        "days": days,
        "attn": curve(model.time_aware_attention.temporal_weight),
        "agg": curve(model.temporal_aggregation.temporal_weight),
    }


@torch.no_grad()
def alpha_norms(model) -> dict[str, Any]:
    """L2 norm of each site's BASE kernel coefficients (the vanilla kernel shape).

    Tracked per epoch because ||alpha|| is what sets the achievable row-centered
    logit change: the kernel can only move attention as far as its coefficients
    reach (see exp/e1_kernel_headroom.py).
    """
    out: dict[str, Any] = {}
    for site, tw in (("attn", model.time_aware_attention.temporal_weight),
                     ("agg", model.temporal_aggregation.temporal_weight)):
        c = tw.coefficients.detach().float().cpu()
        out[site] = {"l2": float(c.norm()), "coeffs": [float(v) for v in c]}
    return out


@torch.no_grad()
def attention_entropy(model, batch) -> dict[str, float]:
    """Mean within-row Shannon entropy (nats) of the attention distribution.

    Rebuilds the attention rows exactly as AgeConditionedTimeAwareAttention does
    -- additive log-space kernel, causal & padding mask -- so the number reflects
    the real forward pass. Reported alongside log(row_len), the uniform ceiling:
    entropy near that ceiling means the kernel is not concentrating anything.
    """
    att = model.time_aware_attention
    ce = model.embedding_table[batch["code_indices"]]
    mask = batch["attention_mask"]
    age_years = batch["age_years"]

    add_delta = model.additive_age_emb(model.additive_fourier(model._additive_age_years(age_years)))
    ce = ce + add_delta * mask.unsqueeze(-1).to(add_delta.dtype)

    q, k = att.mlp_q(ce), att.mlp_k(ce)
    scores = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(att.d_model)
    af = att.age_emb(model._kernel_age_years(age_years))
    scores = scores + F.logsigmoid(att.temporal_weight.poly_value(batch["delta_t"], af))

    l = scores.shape[1]
    causal = torch.tril(torch.ones((l, l), device=scores.device, dtype=torch.bool))
    full = (mask.unsqueeze(1) & mask.unsqueeze(2)) & causal.unsqueeze(0)
    scores = scores.masked_fill(~full, float("-inf"))
    attn = torch.softmax(scores, dim=-1).masked_fill(~full, 0.0)

    ent = -(attn.clamp_min(1e-12).log() * attn).sum(-1)      # [B, L] nats
    n_keys = full.sum(-1)                                     # keys visible per row
    valid = mask & (n_keys > 1)                               # 1-key rows have entropy 0 by construction
    if not bool(valid.any()):
        return {"attn_entropy_mean": float("nan"), "attn_entropy_uniform_ceiling": float("nan"),
                "attn_entropy_ratio": float("nan")}
    e = ent[valid].float()
    ceil = n_keys[valid].float().log()
    return {
        "attn_entropy_mean": float(e.mean()),
        "attn_entropy_uniform_ceiling": float(ceil.mean()),
        # 1.0 == indistinguishable from uniform attention over the visible row
        "attn_entropy_ratio": float((e / ceil.clamp_min(1e-12)).mean()),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    num_codes = len(json.load(open(args.vocab_path)))
    device = torch.device(args.device)

    train_ds = TensorizedEHRDataset(args.tensorized_dir / "train", args.vocab_path)
    val_ds = TensorizedEHRDataset(args.tensorized_dir / "val", args.vocab_path)
    kw = dict(batch_size=args.batch_size, collate_fn=ehr_collate, num_workers=args.num_workers,
              pin_memory=(args.device == "cuda"), persistent_workers=(args.num_workers > 0),
              prefetch_factor=(4 if args.num_workers > 0 else None), worker_init_fn=_dataloader_worker_init)
    if args.num_workers > 0:
        kw["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = TALEEHRAblation(embedding_path=args.embedding_path, num_codes=num_codes,
                            arm=args.arm, d_model=args.d_model,
                            poly_degree=args.poly_degree).to(device)
    model._variant_tag = args.arm
    # INV-demo: age must not be reachable through demo_proj. demo_dim == 2 (sex, race).
    assert model.demo_dim == 2, f"demo_dim must be 2 (sex, race), got {model.demo_dim}"
    assert model.demo_proj[0].in_features == 2, (
        f"demo_proj takes {model.demo_proj[0].in_features} inputs; age is leaking into demographics")
    opt = Adam(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # ---- optional resume: continue from a saved checkpoint's epoch+1 ---------
    start_epoch = 1
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("[resume] WARNING: checkpoint has no optimizer state; Adam moments reset "
                  "(fine for a short top-up under constant LR).", flush=True)
        if "scaler_state_dict" in ckpt and use_amp:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] loaded {args.resume_from} -> continuing at epoch {start_epoch} "
              f"through {args.epochs}", flush=True)
        if start_epoch > args.epochs:
            print(f"[resume] nothing to do: start_epoch {start_epoch} > --epochs {args.epochs}", flush=True)

    # ---- config snapshot (hyperparameters + provenance for the run) ---------
    config = {
        "task": "shared_vanilla_pretrain",
        "model_variant": args.arm,
        "arm": args.arm,
        "seed": args.seed,
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "num_codes": num_codes,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "n_params_total": int(sum(p.numel() for p in model.parameters())),
        "selection_metric": "val_bce (min)",
        "hyperparameters": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "d_model": args.d_model, "poly_degree": args.poly_degree,
            "num_workers": args.num_workers, "val_max_batches": args.val_max_batches,
            "grad_clip_max_norm": 1.0, "optimizer": "Adam", "amp": use_amp,
            "loss": "code_bce", "no_time_loss": True,
        },
        "paths": {
            "tensorized_dir": str(args.tensorized_dir),
            "embedding_path": str(args.embedding_path),
            "vocab_path": str(args.vocab_path),
            "save_dir": str(args.save_dir),
        },
        "diagnostics_notes": {
            "weibull": "N/A - the ablation model has no time/Weibull head (no_time_loss locked); "
                       "nothing to log.",
            "qk_norm": "not QK-normalized (locked); ||q||,||k|| logged as stability evidence only.",
            "w_curve": "single global age-free kernel (vanilla arm: Delta-alpha identically 0).",
        },
    }
    if args.resume_from is not None:
        config["resumed_from"] = str(args.resume_from)
        config["resume_start_epoch"] = start_epoch
    _write_json(args.save_dir / "config.json", config)

    metrics_path = args.save_dir / "pretrain_metrics.json"
    epochs_log: list[dict[str, Any]] = []
    best_epoch, best_val = 0, float("inf")
    # On resume, carry forward prior epochs + best tracking so the metrics file and
    # best-epoch selection span the whole run, not just the top-up epochs.
    if args.resume_from is not None and metrics_path.exists():
        try:
            prev = json.load(metrics_path.open())
            epochs_log = [e for e in prev.get("epochs", []) if int(e.get("epoch", 0)) < start_epoch]
            best_epoch = int(prev.get("best_epoch", 0) or 0)
            best_val = float(prev.get("best_val_bce", float("inf")))
        except Exception:
            pass
    run_t0 = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        grad_norms: list[float] = []
        ep_t0 = time.perf_counter()
        last_batch = None
        for step, batch in enumerate(train_loader, 1):
            if args.dry_run and step > 3:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(batch)
                loss = F.binary_cross_entropy_with_logits(out["code_logits"], batch["target_codes"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            grad_norms.append(float(gn))  # pre-clip total grad norm (clip_grad_norm_ return)
            last_batch = batch
            if args.log_every and step % args.log_every == 0:
                elapsed = time.perf_counter() - ep_t0
                recent = losses[-args.log_every:]
                print(
                    f"  [epoch {epoch:03d} step {step:>7}] "
                    f"running_bce={np.mean(recent):.6f} gradnorm={grad_norms[-1]:.4f} "
                    f"{step / max(elapsed, 1e-9):.1f} it/s elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )
        ep_secs = time.perf_counter() - ep_t0

        train_bce = float(np.mean(losses)) if losses else float("nan")
        val_metrics = evaluate_pretrain(model, val_loader, device, use_amp, args.val_max_batches)
        qk = qk_norm_stats(model, last_batch) if last_batch is not None else {}
        ent = attention_entropy(model, last_batch) if last_batch is not None else {}
        anorm = alpha_norms(model)
        gnorm = {
            "mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "max": float(np.max(grad_norms)) if grad_norms else float("nan"),
            "last": float(grad_norms[-1]) if grad_norms else float("nan"),
        }

        torch.save({"epoch": epoch, "model_variant": args.arm,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scaler_state_dict": scaler.state_dict()},
                   args.save_dir / f"epoch_{epoch:03d}.pt")

        val_bce = val_metrics.get("val_bce", float("nan"))
        if np.isfinite(val_bce) and val_bce < best_val:
            best_val, best_epoch = val_bce, epoch

        rec = {
            "epoch": epoch,
            "train_bce": train_bce,
            **val_metrics,
            "grad_norm_preclip": gnorm,
            "qk_norm": qk,
            "alpha_norm": anorm,
            "attn_entropy": ent,
            "w_curve": w_curve(model),
            "lr_param_groups": [float(g["lr"]) for g in opt.param_groups],
            "epoch_seconds": ep_secs,
        }
        epochs_log.append(rec)
        _write_json(metrics_path, {
            "config": config,
            "best_epoch": best_epoch,
            "best_val_bce": best_val if best_epoch else float("nan"),
            "selection_metric": "val_bce (min)",
            "total_seconds": time.perf_counter() - run_t0,
            "epochs": epochs_log,
        })

        print(
            f"Epoch {epoch:03d} | train_code_bce={train_bce:.6f} | "
            f"val_bce={val_bce:.6f} | val_auroc={val_metrics.get('val_auroc', float('nan')):.6f} | "
            f"val_r@10={val_metrics.get('val_recall@10', float('nan')):.4f} | "
            f"gradnorm(mean/max)={gnorm['mean']:.3f}/{gnorm['max']:.3f} | "
            f"||alpha||(attn/agg)={anorm['attn']['l2']:.4f}/{anorm['agg']['l2']:.4f} | "
            f"attn_H={ent.get('attn_entropy_mean', float('nan')):.4f}"
            f"/{ent.get('attn_entropy_uniform_ceiling', float('nan')):.4f} "
            f"(ratio {ent.get('attn_entropy_ratio', float('nan')):.4f}) | "
            f"best_epoch={best_epoch} | {ep_secs:.1f}s",
            flush=True,
        )
        if args.dry_run:
            break

    # Deterministic teardown. Spawned DataLoader workers can SIGABRT in a C-extension
    # destructor at interpreter shutdown (a benign post-training teardown race that
    # otherwise surfaces as "DataLoader worker killed by signal: Aborted", and can flip
    # the exit code). Training and the checkpoint are already complete and flushed to
    # disk here, so hard-exit and let the OS reap the daemonic workers, bypassing the
    # racy DataLoader __del__/_shutdown_workers path (which is what raises) entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())

