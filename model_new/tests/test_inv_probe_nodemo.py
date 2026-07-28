#!/usr/bin/env python3
"""INV-PROBE-NODEMO -- demographics do not enter ``h_pool``.

Perturbing ``demographics`` leaves ``h_pool`` bit-identical for every arm. That is the
extraction-point check: the probe reads the pooled sequence vector *before* demographic
combination.

Paired sanity (same test, not a separate ID): perturbing ``age_years`` **must** change
``h_pool`` for ``kernel`` and must **not** change it for ``vanilla``. If the kernel arm
fails, age conditioning is not reaching the representation and the probe is meaningless.
"""

from __future__ import annotations

import torch

from model_new.arms import ARMS
from model_new.tests.conftest import wake_generators_


def _extract(model, batch):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        return model.extract_representations(batch)


def test_demographics_do_not_affect_h_pool(model_factory, batch):
    for arm in ARMS:
        m = model_factory(arm)
        wake_generators_(m)
        base = _extract(m, batch)["h_pool"]
        bumped = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items()}
        bumped["demographics"] = bumped["demographics"] + 11.0
        after = _extract(m, bumped)["h_pool"]
        assert torch.equal(base, after), (
            f"[INV-PROBE-NODEMO] arm={arm}: h_pool changed when demographics were "
            f"perturbed (max abs diff={float((base - after).abs().max())})")


def test_age_years_reaches_h_pool_for_kernel_not_vanilla(model_factory, batch):
    """Paired sanity inside INV-PROBE-NODEMO."""
    # kernel: age must move h_pool
    mk = model_factory("kernel")
    wake_generators_(mk)
    base_k = _extract(mk, batch)["h_pool"]
    bumped = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    bumped["age_years"] = bumped["age_years"] + 9.0
    # keep demographic age channel in sync so the only intentional change is age_years
    # for the kernel path; demographics still must not be what moves h_pool.
    after_k = _extract(mk, bumped)["h_pool"]
    assert not torch.equal(base_k, after_k), (
        "[INV-PROBE-NODEMO] kernel h_pool unchanged after age_years perturbation — "
        "age conditioning is not reaching the representation")

    # vanilla: age_years must NOT move h_pool
    mv = model_factory("vanilla")
    wake_generators_(mv)
    base_v = _extract(mv, batch)["h_pool"]
    after_v = _extract(mv, bumped)["h_pool"]
    assert torch.equal(base_v, after_v), (
        "[INV-PROBE-NODEMO] vanilla h_pool changed after age_years perturbation — "
        f"max abs diff={float((base_v - after_v).abs().max())}")


def test_h_head_includes_additive_concat(model_factory, batch):
    m = model_factory("additive")
    out = _extract(m, batch)
    assert out["h_head"].shape[-1] == out["h_pool"].shape[-1] + m.s
    m2 = model_factory("vanilla")
    out2 = _extract(m2, batch)
    assert torch.equal(out2["h_pool"], out2["h_head"])
