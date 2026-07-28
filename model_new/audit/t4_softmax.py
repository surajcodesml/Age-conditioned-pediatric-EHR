"""T4 — Does Δα variation survive the softmax? Centered kernel vs content scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model_new import diagnostics as D
from model_new.audit.common import (
    ARMS,
    build_model,
    iter_batches,
    load_checkpoint,
    to_device,
)
from model_new.data import pairwise_tau, sample_empirical_taus
from model_new.eval_pretrain import make_val_loader
from model_new.encoder import build_pair_mask


AGE_BAND_PAIRS = (
    (25.0, 50.0),
    (25.0, 75.0),
    (50.0, 75.0),
    (18.0, 40.0),
    (40.0, 65.0),
    (18.0, 65.0),
)


@torch.no_grad()
def _content_logit_sd(model, loader, device, max_batches, n_batches_cap: int = 20) -> float:
    """σ_content = sd of QK/√d logits on real batches (encoder layer 0)."""
    vals: list[torch.Tensor] = []
    n = 0
    for batch in iter_batches(loader, max_batches):
        b = to_device(batch, device)
        x = model.embedding_table[b["code_indices"]]
        blk = model.encoder.blocks[0]
        h = blk.ln_attn(x) if blk.ln_attn is not None else x
        attn = blk.attn
        q = attn.mlp_q(h).view(h.shape[0], h.shape[1], attn.n_heads, attn.d_head).transpose(1, 2)
        k = attn.mlp_k(h).view(h.shape[0], h.shape[1], attn.n_heads, attn.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        pair = build_pair_mask(b["attention_mask"]).unsqueeze(1)
        vals.append(scores[pair].detach().float().cpu())
        n += 1
        if n >= n_batches_cap:
            break
    if not vals:
        return float("nan")
    cat = torch.cat([v.reshape(-1) for v in vals])
    return float(cat.std())


@torch.no_grad()
def _D_age_pair(site, tau: torch.Tensor, tau_density: torch.Tensor,
                a1: float, a2: float) -> float:
    """D(a1,a2) = std_τ[ w̃(τ|a1) − w̃(τ|a2) ], τ-weighted by empirical density."""
    ages = torch.tensor([a1, a2], device=tau.device, dtype=torch.float32)
    alpha = site.alpha_base + site.age(ages)  # [2, s]
    log_w = site.kernel(tau.unsqueeze(0), alpha, count=False)  # [2, T]
    # Center over τ with empirical density weights.
    w = tau_density / tau_density.sum().clamp_min(1e-12)
    mean = (log_w * w.unsqueeze(0)).sum(dim=-1, keepdim=True)
    centered = log_w - mean
    diff = centered[0] - centered[1]
    # Weighted std.
    mu = (diff * w).sum()
    var = ((diff - mu).pow(2) * w).sum()
    return float(var.sqrt())


@torch.no_grad()
def run_t4(ctx: dict) -> dict:
    shared = ctx["shared"]
    device = ctx["device"]
    selected = ctx["selected"]
    ds = ctx["dataset"]
    seed = ctx["seed"]

    # Empirical within-row pairwise lags on the pretrain (train) split via existing helper.
    # sample_empirical_taus expects a TensorizedPretrainDataset — use train for density,
    # matching the corpus the kernel was fit against; also sample from val batches.
    from model_new.audit.common import REPO_ROOT
    from model_new.data import TensorizedPretrainDataset

    train_ds = TensorizedPretrainDataset(
        REPO_ROOT / shared["tensorized_dir"] / "train",
        REPO_ROOT / shared["vocab_path"],
        max_seq_len=shared["max_seq_len"],
    )
    tau_np = sample_empirical_taus(train_ds, n_examples=400, seed=seed)
    if tau_np.size < 100:
        raise AssertionError(f"[HARD] too few empirical τ samples: {tau_np.size}")

    # Histogram density on a fixed τ grid covering [0, tau_max].
    tau_max = float(shared["tau_max"])
    n_grid = 257
    edges = np.linspace(0.0, tau_max, n_grid + 1)
    hist, _ = np.histogram(np.clip(tau_np, 0, tau_max), bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # density at centers; floor to avoid zeros.
    dens = hist + 1e-12

    tau_t = torch.as_tensor(centers, dtype=torch.float32, device=device)
    dens_t = torch.as_tensor(dens, dtype=torch.float32, device=device)

    per_arm: dict[str, dict] = {}
    for arm in ("kernel", "random_constant", "vanilla"):
        m = build_model(shared, arm)
        load_checkpoint(m, Path(selected[arm]["checkpoint"]), arm=arm,
                        epoch=selected[arm]["epoch"], device=device)
        loader = make_val_loader(ds, ctx["batch_size"], 0, shared["race_encoding"])
        sigma = _content_logit_sd(m, loader, device, ctx["max_val_batches"])
        sites = {}
        for name, site in m.kernel_sites():
            Ds = {}
            for a1, a2 in AGE_BAND_PAIRS:
                Ds[f"{a1:g}_vs_{a2:g}"] = _D_age_pair(site, tau_t, dens_t, a1, a2)
            max_D = max(Ds.values()) if Ds else 0.0
            R = (max_D / sigma) if sigma > 0 else float("nan")
            sites[name] = {
                "D_by_pair": Ds,
                "max_D": max_D,
                "sigma_content": sigma,
                "R": R,
                "inert_softmax": bool(R < 0.01) if np.isfinite(R) else True,
            }
        per_arm[arm] = sites
        del m

    # Headline: max R over kernel sites.
    kernel_Rs = {s: per_arm["kernel"][s]["R"] for s in per_arm["kernel"]}
    max_R = max(kernel_Rs.values()) if kernel_Rs else float("nan")
    out = {
        "n_tau_samples": int(tau_np.size),
        "tau_grid": centers.tolist(),
        "tau_density": dens.tolist(),
        "age_pairs": [list(p) for p in AGE_BAND_PAIRS],
        "per_arm": per_arm,
        "kernel_max_R": max_R,
        "headline": (
            f"R={max_R:.4e}: learned kernel cannot move attention (R < 0.01)."
            if (np.isfinite(max_R) and max_R < 0.01) else
            f"R={max_R:.4e}: kernel shape variation is large enough relative to content "
            f"logits to matter in principle."
        ),
    }
    lines = [out["headline"]]
    for arm, sites in per_arm.items():
        for name, v in sites.items():
            lines.append(
                f"{arm}/{name}: max_D={v['max_D']:.4e} σ_content={v['sigma_content']:.4e} "
                f"R={v['R']:.4e} inert={v['inert_softmax']}")
    D.print_block("T4 softmax survival", lines)
    return out
