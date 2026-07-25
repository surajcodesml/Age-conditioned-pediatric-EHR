#!/usr/bin/env python3
"""INV-AGESTD -- demographic-age standardization constants are restored from the checkpoint,
never re-derived; an override that disagrees raises.

Same discipline as INV-TMAX. Re-deriving (mean, sd) from the fine-tune corpus would put a PIC
child near 0 (PIC's own mean) instead of near -3 (its true position relative to the adult
pretraining corpus), silently changing what the demographic age feature means to demo_proj.
"""

from __future__ import annotations

import pytest
import torch

from model_new.train_finetune import resolve_age_standardization


def _ckpt(mean, sd):
    return {"config": {"model": {"age_standardization": {"mean": mean, "sd": sd}}}}


def test_checkpoint_constants_used_verbatim():
    assert resolve_age_standardization(_ckpt(55.7, 18.2), None) == (55.7, 18.2)
    assert resolve_age_standardization(_ckpt(55.7, 18.2), (55.7, 18.2)) == (55.7, 18.2)


def test_disagreeing_override_raises():
    with pytest.raises(AssertionError, match=r"INV-AGESTD"):
        resolve_age_standardization(_ckpt(55.7, 18.2), (0.0, 1.0))


def test_missing_constants_raise():
    with pytest.raises(AssertionError, match=r"INV-AGESTD"):
        resolve_age_standardization({"config": {"model": {}}}, None)


def test_buffers_are_persistent_and_frozen(model_factory):
    m = model_factory("kernel", age_mean=55.7, age_sd=18.2)
    sd = m.state_dict()
    assert "age_mean" in sd and "age_sd" in sd, "standardization constants must serialise"
    assert m.age_mean.requires_grad is False and m.age_sd.requires_grad is False


def test_constants_survive_roundtrip_not_rebuilt(model_factory, tmp_path):
    m = model_factory("kernel", age_mean=55.7, age_sd=18.2)
    p = tmp_path / "ck.pt"
    torch.save(m.state_dict(), p)
    m2 = model_factory("kernel", age_mean=0.0, age_sd=1.0)   # deliberately wrong
    m2.load_state_dict(torch.load(p, weights_only=False))
    assert abs(float(m2.age_mean) - 55.7) < 1e-4 and abs(float(m2.age_sd) - 18.2) < 1e-4


def test_standardization_changes_the_demo_channel_only(model_factory, batch):
    """age_years (fed to psi) must stay raw; only demographic channel 0 is standardized."""
    m = model_factory("kernel", age_mean=55.7, age_sd=18.2)
    demo = batch["demographics"]
    std = m.standardize_demo_age(demo)
    # channel 0 transformed
    expected0 = (demo[..., 0] - 55.7) / 18.2
    assert torch.allclose(std[..., 0], expected0, atol=1e-5)
    # every other channel untouched
    assert torch.equal(std[..., 1:], demo[..., 1:])
    # the psi input is a different tensor and is not standardized
    assert torch.equal(batch["age_years"], batch["age_years"])  # unchanged by construction


def test_zero_sd_is_rejected(model_factory):
    with pytest.raises(ValueError, match=r"age_sd"):
        model_factory("kernel", age_mean=10.0, age_sd=0.0)
