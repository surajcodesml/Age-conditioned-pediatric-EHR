"""D4 — Loss mass (GPU, one forward; optional one backward)."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model_new import diagnostics as D
from model_new.audit.common import REPO_ROOT
from model_new.audit.signal.common import (
    add_common_args,
    assert_batch_hash,
    base_result_meta,
    ensure_batches,
    iter_packed_batches,
    iter_store_batches,
    load_arm_model,
    probe_precision,
    require_cuda,
    resolve_device,
    to_device,
    write_json_atomic,
)


def run_d4(ctx: dict, store: dict, device: torch.device, *,
           do_backward: bool = True) -> dict:
    assert_batch_hash(store, ctx["batch_meta"]["batch_list_hash"])
    require_cuda(device, batch_size=int(store["batch_size"]), label="D4")
    model, meta = load_arm_model(ctx, "vanilla", device)
    prec = probe_precision(model, store, device)
    dtype = torch.bfloat16 if prec["dtype"] == "bf16" else torch.float32

    pos_sum = 0.0
    neg_sum = 0.0
    n_pos = 0
    n_examples = 0
    from model_new.eval_pretrain import BatchOrderHash
    hasher = BatchOrderHash()

    model.eval()
    with torch.inference_mode():
        for raw in iter_store_batches(store):
            batch = {k: v for k, v in raw.items()
                     if k not in ("target_gap_days", "age_last") and not str(k).startswith("_")}
            hasher.update(batch)
            b = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=dtype,
                                enabled=(dtype != torch.float32 and device.type == "cuda")):
                logits = model(b)["code_logits"]
            logits_f = logits.float()
            targets = b["target_codes"].float()
            per = F.binary_cross_entropy_with_logits(
                logits_f, targets, reduction="none")
            pos_mask = targets > 0.5
            neg_mask = ~pos_mask
            pos_sum += float(per[pos_mask].sum().item()) if bool(pos_mask.any()) else 0.0
            neg_sum += float(per[neg_mask].sum().item()) if bool(neg_mask.any()) else 0.0
            n_pos += int(pos_mask.sum().item())
            n_examples += int(targets.shape[0])
    hasher_hash = hasher.hexdigest
    if hasher_hash != store["batch_list_hash"]:
        raise AssertionError(
            f"[HARD] D4 batch hash {hasher_hash} != {store['batch_list_hash']}")

    total = pos_sum + neg_sum
    pos_frac = float(pos_sum / total) if total > 0 else float("nan")
    mean_pos = float(n_pos / max(1, n_examples))

    grad_split = None
    if do_backward and device.type == "cuda":
        model.eval()
        raw = next(iter_packed_batches(store))
        batch = {k: v for k, v in raw.items()
                 if k not in ("target_gap_days", "age_last") and not str(k).startswith("_")}
        b = to_device(batch, device)
        model.zero_grad(set_to_none=True)
        logits = model(b)["code_logits"].float()
        targets = b["target_codes"].float()
        per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pos_mask = targets > 0.5
        loss_pos = per[pos_mask].sum() if bool(pos_mask.any()) else logits.sum() * 0
        loss_pos.backward(retain_graph=True)
        pos_gn = 0.0
        for p in model.head_parameters():
            if p.grad is not None:
                pos_gn += float(p.grad.detach().float().pow(2).sum().sqrt().item())
        model.zero_grad(set_to_none=True)
        loss_neg = per[~pos_mask].sum() if bool((~pos_mask).any()) else logits.sum() * 0
        loss_neg.backward()
        neg_gn = 0.0
        for p in model.head_parameters():
            if p.grad is not None:
                neg_gn += float(p.grad.detach().float().pow(2).sum().sqrt().item())
        grad_split = {
            "head_grad_norm_from_pos": pos_gn,
            "head_grad_norm_from_neg": neg_gn,
            "pos_over_neg": float(pos_gn / neg_gn) if neg_gn > 0 else float("nan"),
        }
        model.zero_grad(set_to_none=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = {
        **base_result_meta(ctx, store),
        "arm": "vanilla",
        "checkpoint": meta,
        "precision": prec,
        "positive_loss_mass_fraction": pos_frac,
        "negative_loss_mass_fraction": float(1.0 - pos_frac) if np.isfinite(pos_frac) else float("nan"),
        "mean_positives_per_example": mean_pos,
        "pos_loss_sum": pos_sum,
        "neg_loss_sum": neg_sum,
        "grad_split": grad_split,
        "verdict": {
            "pos_mass_lt_5pct": bool(pos_frac < 0.05) if np.isfinite(pos_frac) else None,
            "route": (
                "negatives dominate → sampled softmax"
                if (np.isfinite(pos_frac) and pos_frac < 0.05)
                else "positive mass adequate for full softmax"
            ),
        },
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--no_backward", action="store_true")
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    D.print_block("D4 loss mass", [
        f"out={out_dir}  smoke={args.smoke}  device={device}",
    ])
    ctx, store = ensure_batches(
        out_dir, smoke=args.smoke, batch_size=args.batch_size,
        force=args.force, run_root=args.run_root,
    )
    ctx["seed"] = args.seed
    result = run_d4(ctx, store, device, do_backward=not args.no_backward)
    write_json_atomic(out_dir / "d4_lossmass.json", result)
    D.print_block("D4 results", [
        f"batch_list_hash={result['batch_list_hash']}",
        f"pos_mass={result['positive_loss_mass_fraction']:.4%}  "
        f"mean_pos/ex={result['mean_positives_per_example']:.2f}",
        f"precision={result['precision']['dtype']} ({result['precision']['reason']})",
        f"verdict: {result['verdict']['route']}",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
