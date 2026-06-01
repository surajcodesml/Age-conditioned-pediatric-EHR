#!/usr/bin/env python3
"""Diagnostic run: pretrain TALE-EHR without time loss in optimization."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    from model.dataset import EHRDataset, TensorizedEHRDataset, ehr_collate
    from model.tale_ehr import TALEEHR
    from model.train import (
        _dataloader_worker_init,
        _move_batch_to_device,
        focal_code_loss,
        log_polynomial_diagnostics,
        bce_code_loss,
        temporal_point_process_loss,
    )
except ModuleNotFoundError:
    from dataset import EHRDataset, TensorizedEHRDataset, ehr_collate
    from tale_ehr import TALEEHR
    from train import (
        _dataloader_worker_init,
        _move_batch_to_device,
        focal_code_loss,
        log_polynomial_diagnostics,
        bce_code_loss,
        temporal_point_process_loss,
    )


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


def _resolve_split_path(data_dir: Path, split_name: str) -> Path:
    path = data_dir / f"{split_name}_events.parquet"
    if path.exists():
        return path
    if split_name == "val":
        fallback = data_dir / "test_events.parquet"
        if fallback.exists():
            print(f"Validation split not found; using fallback {fallback}", flush=True)
            return fallback
    raise FileNotFoundError(f"Missing split file: {path}")


def _tensor_percentiles(x: torch.Tensor, qs: list[float]) -> list[float]:
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    q = torch.tensor(qs, device=x.device, dtype=x.dtype)
    return [float(v) for v in torch.quantile(x, q)]


def _stats_line(prefix: str, x: torch.Tensor) -> str:
    x = x.float()
    std = float(x.std(unbiased=False)) if x.numel() > 0 else float("nan")
    return (
        f"{prefix} mean={float(x.mean()):.6f} std={std:.6f} "
        f"min={float(x.min()):.6f} max={float(x.max()):.6f}"
    )


def _compute_intensity_raw(model: nn.Module, batch: dict[str, torch.Tensor], h: torch.Tensor) -> torch.Tensor:
    b = batch["code_indices"].shape[0]
    device = batch["code_indices"].device
    lengths = batch["attention_mask"].sum(dim=1).long()
    last_idx = (lengths - 1).clamp(min=0)
    demo_last = batch["demographics"][torch.arange(b, device=device), last_idx]
    h_proj = model.history_proj(h)
    d_proj = model.demo_proj(demo_last)
    combined = torch.cat([h_proj, d_proj], dim=-1)
    return model.intensity_predictor(combined)


def run_no_time_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    gamma_loss: float,
    device: str,
    save_dir: Path,
    code_loss_name: str,
    bce_pos_weight: float,
    train_steps: int = 2500,
    val_steps: int = 200,
    w_curve_every: int = 200,
    intensity_diag_every: int = 500,
) -> None:
    device_t = torch.device(device)
    model.to(device_t)
    optimizer = Adam(model.parameters(), lr=lr)
    use_amp = device_t.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if code_loss_name == "focal":
        code_loss_fn = focal_code_loss
    else:
        if bce_pos_weight > 0:
            pos_weight_tensor = torch.full((1,), float(bce_pos_weight), device=device_t)
        else:
            pos_weight_tensor = None

        def code_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return bce_code_loss(logits, targets, pos_weight=pos_weight_tensor)

    log_file = save_dir / "train.log"
    log_fh = open(log_file, "w", buffering=1)
    with open(save_dir / "config.json", "w") as cfg_fh:
        json.dump(
            {
                "lr": lr,
                "gamma_loss": gamma_loss,
                "device": device,
                "code_loss": code_loss_name,
                "bce_pos_weight": bce_pos_weight,
                "train_steps": train_steps,
                "val_steps": val_steps,
                "w_curve_every": w_curve_every,
                "intensity_diag_every": intensity_diag_every,
                "mode": "no_time_loss_diagnostic",
            },
            cfg_fh,
            indent=2,
        )

    model.train()
    train_code_total = 0.0
    train_total = 0.0
    train_iter = iter(train_loader)
    step1_intensity_stats: tuple[float, float, float, float] | None = None
    step2500_intensity_stats: tuple[float, float, float, float] | None = None

    for step in range(1, train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = _move_batch_to_device(batch, device_t)
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()

        with ctx:
            out = model(batch)
            loss_code = code_loss_fn(out["code_logits"], batch["target_codes"])
            loss = gamma_loss * loss_code

        if step == 1:
            with torch.no_grad():
                print("=== STEP 1 DIAGNOSTIC START ===", flush=True)
                target_gap = batch["target_time_gap"].float()
                p05, p25, p50, p75, p95 = _tensor_percentiles(target_gap, [0.05, 0.25, 0.50, 0.75, 0.95])
                raw16 = target_gap[:16].detach().cpu().tolist()
                print(
                    "[diag-1] target_time_gap "
                    f"dtype={target_gap.dtype} shape={tuple(target_gap.shape)} "
                    f"min={float(target_gap.min()):.6f} max={float(target_gap.max()):.6f} "
                    f"mean={float(target_gap.mean()):.6f} std={float(target_gap.std(unbiased=False)):.6f} "
                    f"frac_le_0={float((target_gap <= 0).float().mean()):.6f} "
                    f"nan_count={int(torch.isnan(target_gap).sum().item())} "
                    f"p05={p05:.6f} p25={p25:.6f} p50={p50:.6f} p75={p75:.6f} p95={p95:.6f}",
                    flush=True,
                )
                print(
                    "[diag-1] target_time_gap first16="
                    + ", ".join(f"{float(v):.6f}" for v in raw16),
                    flush=True,
                )

                delta_t = batch["delta_t"].float()
                pair_mask = batch["attention_mask"].unsqueeze(2) & batch["attention_mask"].unsqueeze(1)
                delta_valid = delta_t[pair_mask]
                delta_p50, delta_p95 = _tensor_percentiles(delta_valid, [0.50, 0.95])
                frac_zero = float((delta_valid == 0).float().mean()) if delta_valid.numel() > 0 else float("nan")
                print(
                    "[diag-2] delta_t_valid "
                    f"min={float(delta_valid.min()):.6f} max={float(delta_valid.max()):.6f} "
                    f"p50={delta_p50:.6f} p95={delta_p95:.6f} frac_eq_0={frac_zero:.6f}",
                    flush=True,
                )

                intensity_raw = _compute_intensity_raw(model, batch, out["h"])
                print(_stats_line("[diag-3] intensity_raw", intensity_raw), flush=True)

                intensity_softplus = torch.nn.functional.softplus(intensity_raw)
                print(_stats_line("[diag-4] intensity_softplus", intensity_softplus), flush=True)

                temporal_would_be = temporal_point_process_loss(
                    out["intensity"], batch["target_time_gap"], T=float(torch.clamp(batch["target_time_gap"].max(), min=1.0).item())
                )
                focal_val = focal_code_loss(out["code_logits"], batch["target_codes"])
                print(
                    "[diag-5] losses_on_same_batch "
                    f"temporal_point_process_loss={float(temporal_would_be.item()):.6f} "
                    f"focal_code_loss={float(focal_val.item()):.6f}",
                    flush=True,
                )

                lengths = batch["attention_mask"].sum(dim=1).float()
                len_p50 = _tensor_percentiles(lengths, [0.50])[0]
                print(
                    "[diag-6] lengths "
                    f"min={float(lengths.min()):.1f} max={float(lengths.max()):.1f} "
                    f"mean={float(lengths.mean()):.3f} p50={len_p50:.1f}",
                    flush=True,
                )
                print("=== STEP 1 DIAGNOSTIC END ===", flush=True)

                s = intensity_softplus.squeeze(-1).float()
                step1_intensity_stats = (
                    float(s.mean()),
                    float(s.std(unbiased=False)),
                    float(s.min()),
                    float(s.max()),
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        required_report_steps = {1, 200, 500, 1000, 2000, train_steps}
        if w_curve_every > 0 and (
            step % w_curve_every == 0 or step in required_report_steps
        ):
            log_polynomial_diagnostics(model, step, log_fh)

        scaler.step(optimizer)
        scaler.update()

        train_total += float(loss.detach().item())
        train_code_total += float(loss_code.detach().item())

        if step % intensity_diag_every == 0:
            with torch.no_grad():
                raw_now = _compute_intensity_raw(model, batch, out["h"])
                sp_now = torch.nn.functional.softplus(raw_now).squeeze(-1).float()
                print(_stats_line(f"[intensity_softplus] step={step}", sp_now), flush=True)
                if step == train_steps:
                    step2500_intensity_stats = (
                        float(sp_now.mean()),
                        float(sp_now.std(unbiased=False)),
                        float(sp_now.min()),
                        float(sp_now.max()),
                    )

        if step % 50 == 0 or step == 1:
            print(
                f"step {step:6d} loss={float(loss.detach().item()):.6f} "
                f"(code={float(loss_code.detach().item()):.6f}, gamma*code={float(loss.detach().item()):.6f})",
                flush=True,
            )

    model.eval()
    val_code_total = 0.0
    val_total = 0.0
    val_iter = iter(val_loader)
    with torch.no_grad():
        for step in range(1, val_steps + 1):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)
            batch = _move_batch_to_device(batch, device_t)
            ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(batch)
                loss_code = code_loss_fn(out["code_logits"], batch["target_codes"])
                loss = gamma_loss * loss_code

            val_total += float(loss.detach().item())
            val_code_total += float(loss_code.detach().item())
            if step % 50 == 0 or step == 1:
                print(f"  val step {step:6d} loss={float(loss.detach().item()):.6f}", flush=True)

    final_train_code = train_code_total / max(train_steps, 1)
    final_train_total = train_total / max(train_steps, 1)
    final_val_code = val_code_total / max(val_steps, 1)
    final_val_total = val_total / max(val_steps, 1)

    print(
        f"Final | train={final_train_total:.6f} (code={final_train_code:.6f}) | "
        f"val={final_val_total:.6f} (code={final_val_code:.6f})",
        flush=True,
    )
    if step1_intensity_stats is not None and step2500_intensity_stats is not None:
        print(
            "Intensity drift summary | "
            f"step1(mean/std/min/max)=({step1_intensity_stats[0]:.6f}, {step1_intensity_stats[1]:.6f}, "
            f"{step1_intensity_stats[2]:.6f}, {step1_intensity_stats[3]:.6f}) | "
            f"step{train_steps}(mean/std/min/max)=({step2500_intensity_stats[0]:.6f}, {step2500_intensity_stats[1]:.6f}, "
            f"{step2500_intensity_stats[2]:.6f}, {step2500_intensity_stats[3]:.6f})",
            flush=True,
        )

    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-time-loss diagnostic run")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/"))
    parser.add_argument("--embedding_path", type=Path, default=Path("data/processed/bge_embeddings.pt"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--poly_degree", type=int, default=5)
    parser.add_argument("--gamma_loss", type=float, default=10.0)
    parser.add_argument("--code_loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--bce_pos_weight", type=float, default=0.0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints/"))
    parser.add_argument("--run_name", type=str, default="debug_no_time_loss")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_tensorized", action="store_true")
    parser.add_argument("--train_steps", type=int, default=2500)
    parser.add_argument("--val_steps", type=int, default=200)
    parser.add_argument("--w_curve_every", type=int, default=200)
    parser.add_argument("--intensity_diag_every", type=int, default=500)
    args = parser.parse_args()

    args.save_dir = args.save_dir / args.run_name
    args.save_dir.mkdir(parents=True, exist_ok=True)
    console_log_path = args.save_dir / "console.log"
    console_fh = open(console_log_path, "w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, console_fh)
    sys.stderr = _TeeStream(orig_stderr, console_fh)

    try:
        print(f"[run] checkpoints -> {args.save_dir}", flush=True)
        print(f"[run] console log -> {console_log_path}", flush=True)

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

        loader_kw = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=ehr_collate,
            pin_memory=(args.device == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(4 if args.num_workers > 0 else None),
            worker_init_fn=_dataloader_worker_init,
        )
        if args.num_workers > 0:
            loader_kw["multiprocessing_context"] = "spawn"
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

        model = TALEEHR(
            embedding_path=args.embedding_path,
            num_codes=num_codes,
            d_model=args.d_model,
            poly_degree=args.poly_degree,
        )

        run_no_time_loss(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            gamma_loss=args.gamma_loss,
            device=args.device,
            save_dir=args.save_dir,
            code_loss_name=args.code_loss,
            bce_pos_weight=args.bce_pos_weight,
            train_steps=args.train_steps,
            val_steps=args.val_steps,
            w_curve_every=args.w_curve_every,
            intensity_diag_every=args.intensity_diag_every,
        )
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        console_fh.close()
