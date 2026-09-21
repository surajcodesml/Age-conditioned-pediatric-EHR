"""Controlled λ0 sign test: near vs distant attention, β=0.

s_ij = q⊤k/√d − λ0 τ_ij.  Positive λ0 subtracts more from long lags (recency).
Negative λ0 adds to long lags (anti-recency / long-range preference).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from model_new.data import pretrain_collate
from stage1_mimic_pretrain.model import MinimalDKMModel
from stage2_nch.config import PEDIATRIC_AGE_CENTER_YEARS, PEDIATRIC_AGE_SCALE_YEARS


def _toy_batch() -> dict:
    """Three events: t=0, t=1d, t=1000d. Query is the last event."""
    items = [{
        "code_indices": np.array([2, 3, 4], dtype=np.int64),
        "timestamps_days": np.array([0.0, 1.0, 1000.0], dtype=np.float32),
        "age_days": np.full(3, 9.0 * 365.25, dtype=np.float32),
        "sex": 1,
        "race": 0,
        "unk_vocab_index": 8,
        "target_codes": np.zeros(8, dtype=np.float32),
        "target_time": 1001.0,
    }]
    return pretrain_collate(items)


@torch.no_grad()
def _distant_mass(model: MinimalDKMModel, batch: dict, lambda0: float) -> dict[str, float]:
    model.eval()
    model.temporal.lambda0.fill_(lambda0)
    model.temporal.beta.zero_()
    out = model(batch, need_diagnostics=True)
    attn = out["attn"]  # [B, H, L, L]
    # Last query (index 2) attending to event 0 (distant) vs event 2 (self / near).
    mass_distant = float(attn[0, :, 2, 0].mean().cpu())
    mass_mid = float(attn[0, :, 2, 1].mean().cpu())
    mass_self = float(attn[0, :, 2, 2].mean().cpu())
    return {
        "lambda0": float(lambda0),
        "attn_last_to_t0_distant": mass_distant,
        "attn_last_to_t1_near": mass_mid,
        "attn_last_to_self": mass_self,
    }


def run_lambda0_sign_test(seed: int = 0) -> dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(10, 16, generator=g)
    model = MinimalDKMModel(
        num_codes=8, embedding_table=table, arm="no_interaction", seed=seed,
        d_model=32, n_layers=1, n_heads=4, demo_dim=9, demo_hidden=8,
        age_mean=PEDIATRIC_AGE_CENTER_YEARS, age_sd=PEDIATRIC_AGE_SCALE_YEARS,
    )
    batch = _toy_batch()
    zero = _distant_mass(model, batch, 0.0)
    pos = _distant_mass(model, batch, 2.0)
    neg = _distant_mass(model, batch, -2.0)
    pos_reduces_distant = pos["attn_last_to_t0_distant"] < zero["attn_last_to_t0_distant"]
    neg_increases_distant = neg["attn_last_to_t0_distant"] > zero["attn_last_to_t0_distant"]
    interpretation = (
        "Positive λ0 decreases attention from the last event to the distant (t=0) "
        "event relative to λ0=0 (recency). Negative λ0 increases that mass "
        "(long-range preference). Stage-1 learned λ0<0, so the transferred prior "
        "is long-range, not recency."
    )
    return {
        "equation": "s_ij = q^T k / sqrt(d) - lambda0 * tau_ij   (beta=0)",
        "tau": "log1p(|dt|/7)",
        "sequence_timestamps_days": [0.0, 1.0, 1000.0],
        "lambda0_0": zero,
        "lambda0_plus_2": pos,
        "lambda0_minus_2": neg,
        "positive_lambda0_is_recency": bool(pos_reduces_distant),
        "negative_lambda0_is_long_range": bool(neg_increases_distant),
        "interpretation": interpretation,
        "passed": bool(pos_reduces_distant and neg_increases_distant),
    }
