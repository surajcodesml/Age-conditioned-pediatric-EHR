#!/usr/bin/env python3
"""INV-QUERY -- perturbing age_years[:, j] changes encoder output row j and no other row.

Conditioning is on the QUERY age: alpha[..., k:k+1] broadcasts along the key axis, so one
softmax row carries one kernel shape. Conditioning on the key age would put a different
shape in every entry of a row and make the weights incomparable.

At n_layers > 1 this cannot hold for the stack as a whole -- layer 2's row i reads layer 1's
row j as a *value* -- so the invariant is checked on the full encoder at n_layers=1 and on
block 0's output at n_layers=2.
"""

from __future__ import annotations

import torch

from model_new.tests.conftest import wake_generators_


def _changed_rows(fn, batch, j: int) -> torch.Tensor:
    with torch.no_grad():
        base = fn(batch["age_years"])
        bumped = batch["age_years"].clone()
        bumped[:, j] += 7.0
        after = fn(bumped)
    return (after - base).abs().amax(dim=-1)  # [B, L]


def _tau(batch):
    from model_new.data import pairwise_tau
    return pairwise_tau(batch["timestamps_days"], batch["attention_mask"])


def test_single_layer_encoder_row_locality(model_factory, batch):
    m = model_factory("kernel", n_layers=1)
    wake_generators_(m)
    mask, j = batch["attention_mask"], 2
    tau = _tau(batch)

    def fwd(ages):
        return m.encoder(m.embedding_table[batch["code_indices"]], tau, mask, ages)

    diff = _changed_rows(fwd, batch, j)
    valid = mask.clone()
    for b in range(mask.shape[0]):
        if not bool(mask[b, j]):
            continue
        for i in range(mask.shape[1]):
            if not bool(valid[b, i]):
                continue
            if i == j:
                assert float(diff[b, i]) > 0, f"row {i} of batch {b} must change"
            else:
                assert float(diff[b, i]) == 0, f"row {i} of batch {b} must NOT change"


def test_two_layer_first_block_row_locality(model_factory, batch):
    m = model_factory("kernel", n_layers=2)
    wake_generators_(m)
    mask, j = batch["attention_mask"], 2
    blk = m.encoder.blocks[0]
    tau = _tau(batch)

    def fwd(ages):
        return blk(m.embedding_table[batch["code_indices"]], tau, mask, ages)

    diff = _changed_rows(fwd, batch, j)
    for b in range(mask.shape[0]):
        if not bool(mask[b, j]):
            continue
        for i in range(mask.shape[1]):
            if bool(mask[b, i]) and i != j:
                assert float(diff[b, i]) == 0


def test_pooling_uses_last_event_age_only(model_factory, batch):
    """The pooling site is conditioned on a_n, so only the last valid age can move it."""
    m = model_factory("kernel")
    wake_generators_(m)
    mask = batch["attention_mask"]
    last = m.pooling.last_valid_index(mask)
    with torch.no_grad():
        rows = torch.arange(mask.shape[0])
        a = batch["age_years"][rows, last]
        base = m.pooling.alpha(a)
        assert not torch.allclose(base, m.pooling.alpha(a + 5.0)), "pathway is asleep"
