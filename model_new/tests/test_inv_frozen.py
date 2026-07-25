#!/usr/bin/env python3
"""INV-FROZEN -- embedding_table and Fourier frequency buffers are frozen, persistent, and
restored from the checkpoint rather than rebuilt from defaults."""

from __future__ import annotations

import torch

from model_new.arms import ARMS


def _frequency_buffers(m):
    return {name: buf for name, buf in m.named_buffers() if name.endswith("frequencies")}


def test_embedding_table_is_frozen(model_factory):
    for arm in ARMS:
        m = model_factory(arm)
        assert m.embedding_table.requires_grad is False
        assert not any(p is m.embedding_table for p in m.parameters())


def test_frequency_buffers_are_frozen_and_present(model_factory):
    m = model_factory("kernel")
    freqs = _frequency_buffers(m)
    assert freqs, "no Fourier frequency buffers found"
    for name, buf in freqs.items():
        assert buf.requires_grad is False, name


def test_buffers_are_persistent(model_factory):
    m = model_factory("kernel")
    sd = m.state_dict()
    assert "embedding_table" in sd
    for name in _frequency_buffers(m):
        assert name in sd, f"{name} must serialise, or it is silently rebuilt at load"


def test_buffers_are_restored_not_rebuilt(model_factory, tmp_path):
    """Perturb the buffers, round-trip, and confirm the perturbation survives. If load
    rebuilt them from defaults, this would come back to the default band."""
    m = model_factory("kernel")
    with torch.no_grad():
        m.embedding_table.mul_(3.0)
        for _, buf in _frequency_buffers(m).items():
            buf.mul_(1.5)
    p = tmp_path / "ck.pt"
    torch.save(m.state_dict(), p)

    m2 = model_factory("kernel")
    m2.load_state_dict(torch.load(p, weights_only=False))
    assert torch.equal(m2.embedding_table, m.embedding_table)
    a, b = _frequency_buffers(m), _frequency_buffers(m2)
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_frozen_tensors_receive_no_gradient(model_factory, batch):
    m = model_factory("kernel")
    m(batch)["code_logits"].sum().backward()
    assert m.embedding_table.grad is None
    for name, buf in _frequency_buffers(m).items():
        assert buf.grad is None, name
