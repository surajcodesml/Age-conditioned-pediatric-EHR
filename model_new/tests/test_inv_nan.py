#!/usr/bin/env python3
"""INV-NAN -- a ragged batch yields finite gradients for every parameter, in every arm."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model_new.arms import ARMS
from model_new.encoder import build_pair_mask
from model_new.tests.conftest import wake_generators_


def test_ragged_batch_gives_finite_gradients(model_factory, batch_factory):
    b = batch_factory(0, lengths=(1, 12, 3, 9))   # length-1 row + heavily padded rows
    for arm in ARMS:
        m = model_factory(arm)
        wake_generators_(m)                        # zero-init would hide a NaN in the pathway
        out = m(b)
        loss = F.binary_cross_entropy_with_logits(out["code_logits"], b["target_codes"])
        m.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss), arm
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, f"{arm}: {name} received no gradient"
            assert torch.isfinite(p.grad).all(), f"{arm}: {name} has non-finite gradient"


def test_forward_is_finite_with_extreme_padding(model_factory, batch_factory):
    b = batch_factory(1, lengths=(1, 1, 40))
    for arm in ARMS:
        m = model_factory(arm)
        wake_generators_(m)
        with torch.no_grad():
            out = m(b)
        assert torch.isfinite(out["code_logits"]).all(), arm
        assert torch.isfinite(out["h"]).all(), arm


def test_padded_rows_never_produce_an_all_minus_inf_softmax(batch_factory):
    """The diagonal of the pair mask is forced True precisely so this cannot happen."""
    b = batch_factory(2, lengths=(1, 15))
    pair = build_pair_mask(b["attention_mask"])
    assert bool(pair.any(dim=-1).all()), "some row has no permitted key -> softmax would be NaN"
    assert bool(pair.diagonal(dim1=-2, dim2=-1).all())


def test_zero_length_row_raises_rather_than_wrapping(model_factory, batch):
    """`lengths - 1` on an empty row would wrap to the end of the sequence.

    Drop the collate's `lengths` so the model derives length from the tampered mask and
    reaches the pooling/tau_to_now zero-length guard (rather than the earlier
    lengths-vs-mask consistency assertion, which is covered by the next test)."""
    m = model_factory("kernel")
    bad = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    bad.pop("lengths", None)
    bad["attention_mask"][1] = False
    try:
        m(bad)
    except ValueError as exc:
        assert "zero-length" in str(exc)
        return
    raise AssertionError("a zero-length sequence was silently accepted")


def test_stale_lengths_are_rejected(model_factory, batch):
    """The new HARD guard: lengths that disagree with the mask must raise, not silently
    disagree with the tau_to_now / pooling last-index computations."""
    m = model_factory("kernel")
    bad = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    bad["attention_mask"][1] = False   # lengths now stale
    try:
        m(bad)
    except AssertionError as exc:
        assert "lengths" in str(exc)
        return
    raise AssertionError("stale lengths were silently accepted")
