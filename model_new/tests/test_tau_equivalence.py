#!/usr/bin/env python3
"""The regression that matters for the tau-to-GPU move: the tau the model computes from
``timestamps_days`` matches the values the old collate produced, to 1e-6 at every valid
position, and padded positions are still zero.

The reference here is an INDEPENDENT inline reimplementation of the old collate arithmetic,
not a call into `data`, so this test would catch a change in either path.
"""

from __future__ import annotations

import numpy as np
import torch

from model_new.data import pretrain_collate, tau_from_timestamps
from model_new.tests.conftest import make_items


def _reference_tau(ts_f64: torch.Tensor, mask: torch.Tensor):
    """The old collate's arithmetic, written out independently, in float64."""
    t = ts_f64.double()
    dt = (t[:, :, None] - t[:, None, :]).abs()
    tau = torch.log1p(dt / 7.0)
    pair = mask[:, :, None] & mask[:, None, :]
    tau = tau * pair.double()

    lengths = mask.sum(dim=1).long()
    rows = torch.arange(t.shape[0])
    t_last = t[rows, lengths - 1][:, None]
    tau_to_now = torch.log1p((t_last - t).abs() / 7.0) * mask.double()
    return tau.float(), tau_to_now.float(), pair


def test_gpu_tau_matches_old_collate_values():
    for seed in range(4):
        b = pretrain_collate(make_items(np.random.default_rng(seed)))
        tau, tau_to_now = tau_from_timestamps(
            b["timestamps_days"], b["attention_mask"], b["lengths"])
        ref_tau, ref_ttn, pair = _reference_tau(b["timestamps_days"], b["attention_mask"])

        assert torch.allclose(tau[pair], ref_tau[pair], atol=1e-6), f"seed {seed}: tau mismatch"
        valid = b["attention_mask"]
        assert torch.allclose(tau_to_now[valid], ref_ttn[valid], atol=1e-6), \
            f"seed {seed}: tau_to_now mismatch"


def test_padded_positions_are_zero():
    b = pretrain_collate(make_items(np.random.default_rng(0), lengths=(3, 9, 1, 5)))
    tau, tau_to_now = tau_from_timestamps(
        b["timestamps_days"], b["attention_mask"], b["lengths"])
    mask = b["attention_mask"]
    pair = mask[:, :, None] & mask[:, None, :]
    assert float(tau[~pair].abs().max()) == 0.0
    assert float(tau_to_now[~mask].abs().max()) == 0.0


def test_lengths_agree_with_mask():
    b = pretrain_collate(make_items(np.random.default_rng(1)))
    assert torch.equal(b["lengths"], b["attention_mask"].sum(dim=1).long())


def test_timestamps_are_float64_in_batch():
    """f64 storage is what lets the in-model differencing keep sub-minute resolution."""
    b = pretrain_collate(make_items(np.random.default_rng(0)))
    assert b["timestamps_days"].dtype == torch.float64
    assert "tau" not in b and "tau_to_now" not in b, "tau must not be shipped in the batch"


def test_model_forward_uses_computed_tau(model_factory):
    """End to end: the model runs on a batch that contains no tau, only timestamps."""
    b = pretrain_collate(make_items(np.random.default_rng(2)))
    m = model_factory("kernel").eval()
    with torch.no_grad():
        out = m(b)
    assert torch.isfinite(out["code_logits"]).all()
