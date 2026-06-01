#!/usr/bin/env python3
"""Diagnostics for age-conditioned polynomial coefficients."""

from __future__ import annotations

import math

import torch


def compute_alpha_delta_stats(model, batch: dict[str, torch.Tensor]) -> dict:
    """Compute Delta-alpha norm summary statistics for a batch."""
    age_years = batch["demographics"][..., 0]
    mask = batch["attention_mask"]
    age_emb = model.time_aware_attention.age_emb
    coeff_gen = model.time_aware_attention.temporal_weight.age_coeff_gen
    agg_age_emb = model.temporal_aggregation.age_emb
    agg_coeff_gen = model.temporal_aggregation.temporal_weight.age_coeff_gen

    with torch.no_grad():
        gamma = age_emb(age_years.clamp(min=0.0))
        delta_alpha = coeff_gen(gamma)
        norms = delta_alpha.norm(dim=-1)
        lengths = mask.sum(dim=1).to(dtype=torch.long)
        batch_idx = torch.arange(age_years.shape[0], device=age_years.device)
        age_current = age_years[batch_idx, lengths - 1].clamp(min=0.0)
        gamma_agg = agg_age_emb(age_current)
        delta_alpha_agg = agg_coeff_gen(gamma_agg)
        norms_agg = delta_alpha_agg.norm(dim=-1)

    valid = mask.bool()
    norms_valid = norms[valid]
    ages_valid = age_years[valid]

    if norms_valid.numel() == 0:
        mean_val = float("nan")
        std_val = float("nan")
    else:
        mean_val = norms_valid.mean().detach().item()
        std_val = norms_valid.std(unbiased=False).detach().item()

    bucket_defs = [
        ("neonate", lambda a: a < (1.0 / 12.0)),
        ("infant", lambda a: (a >= (1.0 / 12.0)) & (a < 2.0)),
        ("child", lambda a: (a >= 2.0) & (a < 12.0)),
        ("adolescent", lambda a: (a >= 12.0) & (a < 18.0)),
        ("young_adult", lambda a: (a >= 18.0) & (a < 40.0)),
        ("middle_age", lambda a: (a >= 40.0) & (a < 65.0)),
        ("older_adult", lambda a: a >= 65.0),
    ]

    by_bucket: dict[str, float] = {}
    for name, cond_fn in bucket_defs:
        if norms_valid.numel() == 0:
            by_bucket[name] = float("nan")
            continue
        bmask = cond_fn(ages_valid)
        if bool(bmask.any()):
            by_bucket[name] = norms_valid[bmask].mean().detach().item()
        else:
            by_bucket[name] = float("nan")

    return {
        "delta_alpha_norm_mean": float(mean_val),
        "delta_alpha_norm_std": float(std_val),
        "delta_alpha_norm_by_age_bucket": by_bucket,
        "delta_alpha_agg_norm_mean": float(norms_agg.mean().detach().item()),
        "delta_alpha_agg_norm_std": float(norms_agg.std(unbiased=False).detach().item()),
        "delta_alpha_agg_norm_min": float(norms_agg.min().detach().item()),
        "delta_alpha_agg_norm_max": float(norms_agg.max().detach().item()),
    }


def log_alpha_delta_stats(stats: dict, step: int, prefix: str = "") -> str:
    """Format a one-line diagnostics message."""
    mean_val = float(stats.get("delta_alpha_norm_mean", float("nan")))
    std_val = float(stats.get("delta_alpha_norm_std", float("nan")))
    by_bucket = stats.get("delta_alpha_norm_by_age_bucket", {})
    agg_mean = float(stats.get("delta_alpha_agg_norm_mean", float("nan")))
    agg_std = float(stats.get("delta_alpha_agg_norm_std", float("nan")))
    agg_min = float(stats.get("delta_alpha_agg_norm_min", float("nan")))
    agg_max = float(stats.get("delta_alpha_agg_norm_max", float("nan")))

    header = (
        f"{prefix}[age_diag] step {step} ||Δα||={mean_val:.3f}±{std_val:.3f} "
        f"||Δα_agg(a_current)||={agg_mean:.3f}±{agg_std:.3f} "
        f"[{agg_min:.3f},{agg_max:.3f}]"
    )
    parts: list[str] = [header]
    for name in [
        "neonate",
        "infant",
        "child",
        "adolescent",
        "young_adult",
        "middle_age",
        "older_adult",
    ]:
        if name not in by_bucket:
            continue
        value = float(by_bucket[name])
        if math.isnan(value):
            continue
        parts.append(f"{name}={value:.3f}")
    return " | ".join(parts)
