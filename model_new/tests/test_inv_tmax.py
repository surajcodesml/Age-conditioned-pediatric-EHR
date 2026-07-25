#!/usr/bin/env python3
"""INV-TMAX -- tau_max at fine-tune equals the checkpoint value bit-for-bit."""

from __future__ import annotations

import pytest
import torch

from model_new.train_finetune import resolve_tau_max


def test_checkpoint_value_is_used_verbatim():
    ckpt = {"tau_max": 6.437218189239502}
    assert resolve_tau_max(ckpt, None) == ckpt["tau_max"]
    assert resolve_tau_max(ckpt, 6.437218189239502) == ckpt["tau_max"]


def test_disagreeing_override_raises():
    ckpt = {"tau_max": 6.5}
    with pytest.raises(AssertionError, match=r"INV-TMAX"):
        resolve_tau_max(ckpt, 6.4999999)


def test_missing_tau_max_raises():
    with pytest.raises(AssertionError, match=r"INV-TMAX"):
        resolve_tau_max({}, None)


def test_tau_max_is_one_source_of_truth_across_sites(model_factory):
    m = model_factory("kernel", tau_max=6.25)
    assert m.tau_max == 6.25
    # Desynchronise one site by hand: the property must refuse to answer.
    with torch.no_grad():
        m.kernel_sites()[0][1].kernel.tau_max.fill_(6.0)
    with pytest.raises(AssertionError, match=r"INV-TMAX"):
        _ = m.tau_max
    m.set_tau_max(6.25)
    assert m.tau_max == 6.25


def test_tau_max_serialises(model_factory, tmp_path):
    m = model_factory("kernel", tau_max=5.75)
    p = tmp_path / "ck.pt"
    torch.save({"model_state_dict": m.state_dict(), "tau_max": m.tau_max}, p)
    ck = torch.load(p, weights_only=False)
    m2 = model_factory("kernel", tau_max=1.0)      # deliberately wrong
    m2.load_state_dict(ck["model_state_dict"])
    assert m2.tau_max == 5.75, "tau_max must ride in the state_dict, not be rebuilt"
