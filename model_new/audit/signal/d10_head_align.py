"""D10 — Head sensitivity alignment (GPU).

Is Δh from a 25-vs-75 age intervention aligned with ∇_h L?
Near-random alignment ⇒ kernel writes orthogonally to the objective.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model_new import diagnostics as D
from model_new.audit.common import REPO_ROOT
from model_new.audit.signal import T4_SIGMA_CONTENT
from model_new.audit.signal.common import (
    SIGNAL_SEED,
    add_common_args,
    assert_batch_hash,
    base_result_meta,
    ensure_batches,
    iter_store_batches,
    load_arm_model,
    require_cuda,
    resolve_device,
    to_device,
    write_json_atomic,
)
from model_new.train import set_seed


def _broadcast_age(batch: dict, age: float) -> dict:
    """Replace age_years only; demographics (incl. standardized age channel) stay true."""
    b = dict(batch)
    mask = batch["attention_mask"].float()
    b["age_years"] = torch.full_like(batch["age_years"], float(age)) * mask
    return b


def _grad_h_loss(model, batch: dict) -> torch.Tensor:
    """Per-example ∇_h L for mean-over-codes BCE. Returns [B, d_model]."""
    model.eval()
    # Forward to pooled h with grad enabled through pooling/encoder.
    for p in model.parameters():
        p.requires_grad_(False)
    out = model(batch, need_diagnostics=False)
    h = out["h"].detach().requires_grad_(True)

    lengths = batch["lengths"]
    rows = torch.arange(lengths.shape[0], device=lengths.device)
    last = lengths - 1
    demo_last = model.standardize_demo_age(batch["demographics"][rows, last])
    parts = [h, model.demo_proj(demo_last)]
    if model.additive_age is not None:
        age_last = batch["age_years"][rows, last]
        parts.append(model.additive_age(age_last))
    logits = model.head(torch.cat(parts, dim=-1))
    targets = batch["target_codes"].float()
    loss_per = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none").mean(dim=-1)  # [B]

    grads = []
    for i in range(loss_per.shape[0]):
        g, = torch.autograd.grad(loss_per[i], h, retain_graph=True, allow_unused=False)
        grads.append(g[i].detach())
    return torch.stack(grads, dim=0)


@torch.inference_mode()
def _h_at_age(model, batch: dict, age: float) -> torch.Tensor:
    out = model(_broadcast_age(batch, age))
    return out["h"].detach()


def run_d10(ctx: dict, store: dict, device: torch.device) -> dict:
    flags = ctx["flags"]
    seed = int(ctx.get("seed", SIGNAL_SEED))
    set_seed(seed)
    require_cuda(device, batch_size=int(store["batch_size"]), label="D10")
    assert_batch_hash(store, ctx["batch_meta"]["batch_list_hash"])

    model, meta = load_arm_model(ctx, "kernel", device)
    age_lo, age_hi = 25.0, 75.0

    cos_vals: list[float] = []
    cos_rand: list[float] = []
    frac_top10: list[float] = []
    rel_dh: list[float] = []
    all_grads: list[torch.Tensor] = []
    all_dh: list[torch.Tensor] = []
    rng = np.random.default_rng(seed + 91)
    n_examples = 0

    for raw in iter_store_batches(store):
        batch = {k: v for k, v in raw.items()
                 if k not in ("target_gap_days", "age_last") and not str(k).startswith("_")}
        b = to_device(batch, device)

        # ∇_h L (needs grad)
        with torch.enable_grad():
            g = _grad_h_loss(model, b)  # [B, D]

        h25 = _h_at_age(model, b, age_lo)
        h75 = _h_at_age(model, b, age_hi)
        dh = h75 - h25  # [B, D]  (75-vs-25; sign arbitrary for cos abs later)

        # Random Δh of matched norm.
        rand = torch.from_numpy(
            rng.standard_normal(size=tuple(dh.shape)).astype(np.float32)
        ).to(device)
        rand = rand / rand.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rand = rand * dh.norm(dim=-1, keepdim=True)

        cos = F.cosine_similarity(dh, g, dim=-1)
        cos_r = F.cosine_similarity(rand, g, dim=-1)
        cos_vals.extend(cos.detach().cpu().numpy().tolist())
        cos_rand.extend(cos_r.detach().cpu().numpy().tolist())
        rel_dh.extend(
            (dh.norm(dim=-1) / h25.norm(dim=-1).clamp_min(1e-12))
            .detach().cpu().numpy().tolist()
        )
        all_grads.append(g.detach().cpu())
        all_dh.append(dh.detach().cpu())
        n_examples += int(dh.shape[0])

    G = torch.cat(all_grads, dim=0).numpy().astype(np.float64)  # [N, D]
    DH = torch.cat(all_dh, dim=0).numpy().astype(np.float64)

    # Top-10 principal directions of ∇_h L across the batch.
    G_c = G - G.mean(axis=0, keepdims=True)
    # Economy SVD; D is small (~64–256).
    _, _, Vt = np.linalg.svd(G_c, full_matrices=False)
    top = Vt[:10]  # [10, D]
    # Fraction of ‖Δh‖ lying in that subspace.
    for i in range(DH.shape[0]):
        dh_i = DH[i]
        proj = top @ dh_i
        frac = float(np.linalg.norm(proj) / (np.linalg.norm(dh_i) + 1e-12))
        frac_top10.append(frac)

    cos_a = np.asarray(cos_vals, dtype=np.float64)
    cos_r_a = np.asarray(cos_rand, dtype=np.float64)
    frac_a = np.asarray(frac_top10, dtype=np.float64)

    mean_cos = float(np.nanmean(cos_a))
    mean_rand = float(np.nanmean(cos_r_a))
    # Near random if |mean_cos| within ~2× |mean_rand| or CI overlap-ish.
    near_random = bool(abs(mean_cos) <= max(0.05, 2.0 * abs(mean_rand) + 0.02))

    out = {
        **base_result_meta(ctx, store),
        "arm": "kernel",
        "checkpoint": meta,
        "device": str(device),
        "t4_sigma_content": float(T4_SIGMA_CONTENT),
        "intervention_ages": {"lo": age_lo, "hi": age_hi},
        "n_examples": n_examples,
        "cos_dh_grad": {
            "mean": mean_cos,
            "std": float(np.nanstd(cos_a)),
            "p10": float(np.nanpercentile(cos_a, 10)),
            "p50": float(np.nanpercentile(cos_a, 50)),
            "p90": float(np.nanpercentile(cos_a, 90)),
        },
        "cos_random_control": {
            "mean": mean_rand,
            "std": float(np.nanstd(cos_r_a)),
            "p50": float(np.nanpercentile(cos_r_a, 50)),
        },
        "frac_dh_in_top10_grad_pcs": {
            "mean": float(np.nanmean(frac_a)),
            "p50": float(np.nanpercentile(frac_a, 50)),
            "p90": float(np.nanpercentile(frac_a, 90)),
        },
        "rel_dh_mean": float(np.nanmean(rel_dh)),
        "verdict": {
            "near_random_control": near_random,
            "route": (
                "kernel writes orthogonally to the objective → head/objective failure, not attention"
                if near_random
                else "Δh aligns with ∇_h L → head can see the age-route write"
            ),
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    D.print_block("D10 head sensitivity alignment", [
        f"out={out_dir}  smoke={args.smoke}  device={device}",
        f"T4 σ_content={T4_SIGMA_CONTENT} (reused, not recomputed)",
    ])
    ctx, store = ensure_batches(
        out_dir, smoke=args.smoke, batch_size=args.batch_size,
        force=args.force, run_root=args.run_root,
    )
    ctx["seed"] = args.seed
    result = run_d10(ctx, store, device)
    write_json_atomic(out_dir / "d10_head_align.json", result)
    D.print_block("D10 results", [
        f"batch_list_hash={result['batch_list_hash']}",
        f"cos(Δh, ∇hL) mean={result['cos_dh_grad']['mean']:.4f}  "
        f"random={result['cos_random_control']['mean']:.4f}",
        f"frac ‖Δh‖ in top-10 ∇ PCs mean={result['frac_dh_in_top10_grad_pcs']['mean']:.4f}",
        f"verdict: {result['verdict']['route']}",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
