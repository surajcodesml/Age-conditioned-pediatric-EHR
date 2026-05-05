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

    with torch.no_grad():
        gamma = age_emb(age_years.clamp(min=0.0))
        delta_alpha = coeff_gen(gamma)
        norms = delta_alpha.norm(dim=-1)

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
    }


def log_alpha_delta_stats(stats: dict, step: int, prefix: str = "") -> str:
    """Format a one-line diagnostics message."""
    mean_val = float(stats.get("delta_alpha_norm_mean", float("nan")))
    std_val = float(stats.get("delta_alpha_norm_std", float("nan")))
    by_bucket = stats.get("delta_alpha_norm_by_age_bucket", {})

    header = f"{prefix}[age_diag] step {step} ||Δα||={mean_val:.3f}±{std_val:.3f}"
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
