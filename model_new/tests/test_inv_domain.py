#!/usr/bin/env python3
"""INV-DOMAIN -- tau_tilde in [-1, 1] for every valid pair, at pretrain and fine-tune."""

from __future__ import annotations

import numpy as np
import torch

from model_new.basis import ChebyshevKernel
from model_new.data import pretrain_collate, tau_from_timestamps
from model_new.data_finetune import finetune_collate
from model_new.tests.conftest import VOCAB, make_items


def _tau_tilde_in_range(tau: torch.Tensor, tau_max: float) -> None:
    k = ChebyshevKernel(s=5, tau_max=tau_max)
    tt = k.rescale(tau)
    assert float(tt.min()) >= -1.0
    assert float(tt.max()) <= 1.0


def _batch_taus(b):
    # tau is no longer shipped in the batch; recompute it the way the model does.
    return tau_from_timestamps(b["timestamps_days"], b["attention_mask"], b.get("lengths"))


def test_pretrain_batch_stays_in_domain(batch_factory):
    for seed in range(4):
        b = batch_factory(seed)
        tau, tau_to_now = _batch_taus(b)
        _tau_tilde_in_range(tau, 6.5)
        _tau_tilde_in_range(tau_to_now, 6.5)


def test_finetune_batch_stays_in_domain():
    rng = np.random.default_rng(3)
    items = make_items(rng)
    for it in items:
        it["label"] = 1.0
        it["subject_id"] = 1
        it["hadm_id"] = 2
    b = finetune_collate(items)
    tau, tau_to_now = _batch_taus(b)
    _tau_tilde_in_range(tau, 6.5)
    _tau_tilde_in_range(tau_to_now, 6.5)


def test_out_of_domain_is_clamped_and_counted():
    k = ChebyshevKernel(s=5, tau_max=1.0)
    tau = torch.tensor([0.0, 0.5, 1.0, 5.0, 50.0])   # 3 of 5 land above tau_max
    tt = k.rescale(tau)
    assert float(tt.max()) == 1.0
    assert k.clamp_fraction > 0.0
    assert k.clamp_fraction == 2 / 5   # 5.0 and 50.0; tau == tau_max maps to exactly 1.0
    k.reset_clamp_stats()
    assert k.clamp_fraction == 0.0


def test_model_forward_keeps_domain(model_factory, batch):
    m = model_factory("kernel")
    m.reset_clamp_stats()
    m(batch)
    for _, site in m.kernel_sites():
        assert site.kernel.clamp_fraction == 0.0, "synthetic corpus should not clamp at tau_max=6.5"
