#!/usr/bin/env python3
"""INV-ARM -- per-arm parameter structure.

kernel and random_constant must match exactly (that is the identifying comparison).
vanilla must have zero age parameters. additive must have a generator and no kernel-side age
parameters. NO constraint is placed on additive's total: its architecture differs and the
count is reported honestly rather than padded.
"""

from __future__ import annotations

import pytest

from model_new.arms import ARMS, resolve_arm, assert_arm_invariants


def _counts(m):
    return {k: v for k, v in m.parameter_report().items()}


def test_kernel_and_random_constant_are_exactly_matched(model_factory):
    a, b = model_factory("kernel"), model_factory("random_constant")
    ca, cb = _counts(a), _counts(b)
    for key in ("backbone", "age", "head", "total_trainable"):
        assert ca[key] == cb[key], f"{key}: kernel={ca[key]} random_constant={cb[key]}"


def test_vanilla_has_no_age_parameters(model_factory):
    m = model_factory("vanilla")
    assert m.age_parameters() == []
    assert _counts(m)["age"] == 0
    for _, site in m.kernel_sites():
        assert site.age.generator.mlp is None, "the module must not be constructed at all"


def test_additive_has_a_generator_and_no_kernel_side_age_params(model_factory):
    m = model_factory("additive")
    assert m.additive_age is not None
    assert len(m.age_parameters()) > 0
    for _, site in m.kernel_sites():
        assert site.age.generator.mlp is None
        assert site.age.age_parameters() == []


def test_additive_total_is_unconstrained(model_factory):
    """Documented, not asserted: additive's head is s wider by construction."""
    m, k = model_factory("additive"), model_factory("kernel")
    assert m.head_in == k.head_in + k.s
    assert _counts(m)["head"] > _counts(k)["head"]


def test_arm_config_resolution():
    for arm in ARMS:
        cfg = resolve_arm(arm)
        assert cfg.arm == arm
        assert_arm_invariants(cfg, center_delta_alpha=False)
    with pytest.raises(ValueError):
        resolve_arm("nope")


def test_center_delta_alpha_refuses_random_constant():
    """Centering makes a constant Delta-alpha exactly zero, collapsing the control onto
    vanilla. See INVARIANTS.md."""
    with pytest.raises(AssertionError, match=r"collapses random_constant"):
        assert_arm_invariants(resolve_arm("random_constant"), center_delta_alpha=True)
