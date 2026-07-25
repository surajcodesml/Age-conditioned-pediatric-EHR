#!/usr/bin/env python3
"""INV-ZERO-B -- for additive, zeroing the generator's concat columns leaves logits unchanged
at init.

additive has a wider head than the other arms, so a single cross-arm bit-identity test is
not well posed; this is the arm-appropriate form.
"""

from __future__ import annotations

import torch


def test_zeroing_concat_columns_is_a_no_op_at_init(model_factory, batch):
    m = model_factory("additive").eval()
    with torch.no_grad():
        before = m(batch)["code_logits"].clone()
        assert m.head.net[0].weight.shape[1] == m.d_model + m.demo_hidden + m.s
        m.head.net[0].weight[:, -m.s:].zero_()
        after = m(batch)["code_logits"]
    assert torch.equal(before, after)


def test_generator_output_is_zero_at_init(model_factory, batch):
    m = model_factory("additive")
    with torch.no_grad():
        last = m.pooling.last_valid_index(batch["attention_mask"])
        rows = torch.arange(batch["code_indices"].shape[0])
        delta = m.additive_age(batch["age_years"][rows, last])
    assert delta.shape == (batch["code_indices"].shape[0], m.s)
    assert int(torch.count_nonzero(delta)) == 0


def test_pathway_is_live_once_woken(model_factory, batch):
    """Guards against the test above passing because the pathway is disconnected."""
    from model_new.tests.conftest import wake_generators_
    m = model_factory("additive").eval()
    with torch.no_grad():
        before = m(batch)["code_logits"].clone()
    wake_generators_(m)
    with torch.no_grad():
        after = m(batch)["code_logits"]
    assert not torch.equal(before, after), "additive concat columns are not connected"
