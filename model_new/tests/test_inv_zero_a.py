#!/usr/bin/env python3
"""INV-ZERO-A -- all four arms produce the same logits at init.

The three arms whose head input has the same width (vanilla, kernel, random_constant) are
**bit-identical**. ``additive``'s head input is ``s`` columns wider, and although those
columns are multiplied by an exactly-zero generator output, the matmul reduces over a longer
axis and so accumulates in a different order. That leaves float32 rounding noise of order
1e-7 -- a property of floating-point summation, not an initialization difference.

Before the head-init fix this gap was ~3.2 in logit units, because xavier's ``fan_in``
depended on the arm and rescaled *every* weight in the layer. ``INV-ZERO-B`` cannot detect
that: it only checks that the concat columns contribute zero given zero input, which held
either way.
"""

from __future__ import annotations

import torch

SAME_WIDTH_ARMS = ("vanilla", "kernel", "random_constant")
FLOAT_TOL = 1e-5   # far below the ~3.2 the unfixed init produced, far above matmul noise


def _logits(model_factory, arm, batch):
    m = model_factory(arm).eval()
    with torch.no_grad():
        return m(batch)["code_logits"]


def test_same_width_arms_are_bit_identical_at_init(model_factory, batch):
    ref = None
    for arm in SAME_WIDTH_ARMS:
        logits = _logits(model_factory, arm, batch)
        if ref is None:
            ref = logits
        else:
            assert torch.equal(logits, ref), f"{arm} differs from vanilla at init"


def test_additive_matches_within_float_tolerance(model_factory, batch):
    ref = _logits(model_factory, "vanilla", batch)
    got = _logits(model_factory, "additive", batch)
    delta = float((got - ref).abs().max())
    assert delta < FLOAT_TOL, (
        f"additive differs from vanilla by {delta:.3e} at init. Anything above float noise "
        f"means the head is no longer drawn at the widest arm's width and sliced.")


def test_head_first_layer_shared_columns_are_bit_identical(model_factory):
    """The mechanism behind the test above, checked directly on the weights."""
    ref = None
    for arm in ("vanilla", "kernel", "random_constant", "additive"):
        m = model_factory(arm)
        w = m.head.net[0].weight.detach()
        shared = m.d_model + m.demo_hidden
        assert w.shape[0] == m.head_in_max, "head hidden width must be arm-independent"
        if ref is None:
            ref = w[:, :shared].clone()
        else:
            assert torch.equal(w[:, :shared], ref), f"{arm}: shared head columns differ"


def test_head_second_layer_is_bit_identical(model_factory):
    ref = None
    for arm in ("vanilla", "kernel", "random_constant", "additive"):
        w = model_factory(arm).head.net[2].weight.detach()
        if ref is None:
            ref = w.clone()
        else:
            assert torch.equal(w, ref), f"{arm}: second head layer differs"


def test_additive_extra_columns_are_not_zero_initialised(model_factory):
    """Zeroing them as well as the generator's final layer would make both gradients vanish
    permanently (dL/dW_c = 0 because g = 0, dL/dg = 0 because W_c = 0), so the additive
    pathway could never start and the arm would be vanilla with dead parameters."""
    m = model_factory("additive")
    extra = m.head.net[0].weight.detach()[:, -m.s:]
    assert int(torch.count_nonzero(extra)) == extra.numel(), (
        "additive's concat columns must get a normal draw, not zeros")


def test_shared_backbone_is_bit_identical_in_all_four_arms(model_factory):
    """Per-parameter seeding means a shape change anywhere cannot shift another draw."""
    ref = None
    for arm in ("vanilla", "kernel", "random_constant", "additive"):
        m = model_factory(arm)
        age_ids = {id(p) for p in m.age_parameters()}
        shared = {name: p.detach().clone()
                  for name, p in m.named_parameters()
                  if id(p) not in age_ids and not name.startswith("head.")}
        if ref is None:
            ref = shared
        else:
            assert set(shared) == set(ref)
            for name in shared:
                assert torch.equal(shared[name], ref[name]), f"{arm}: {name} differs"


def test_delta_alpha_is_exactly_zero_at_init(model_factory, batch):
    ages = batch["age_years"][batch["attention_mask"]]
    for arm in ("vanilla", "kernel", "random_constant", "additive"):
        m = model_factory(arm)
        for name, site in m.kernel_sites():
            with torch.no_grad():
                d = site.age(ages)
            assert int(torch.count_nonzero(d)) == 0, f"{arm}/{name}"
