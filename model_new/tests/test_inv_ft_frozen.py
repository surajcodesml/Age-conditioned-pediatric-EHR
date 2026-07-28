#!/usr/bin/env python3
"""INV-FT-FROZEN -- every constant frozen at pretraining is bit-identical at fine-tune.

Five quantities, one test each for the failure. The shared point: each of them silently
*redefines* a learned weight rather than breaking anything, so the run would finish and
report numbers. That is why they are hard errors.
"""

from __future__ import annotations

import copy

import pytest
import torch

from model_new import diagnostics as D
from model_new.arms import ARMS
from model_new.data import demo_layout
from model_new.train_finetune import assert_frozen_constants

from .conftest import TAU_MAX

DEMO_DIM, DEMO_CHANNELS = demo_layout("one_hot")


@pytest.fixture
def ft_model(model_factory):
    """A model built the way ``train_finetune`` builds one: the demographic channel names
    come from ``data.demo_layout``, so the race ordering is a recorded fact."""
    def _make(arm: str = "kernel", **kw):
        return model_factory(arm, demo_dim=DEMO_DIM, demo_channels=DEMO_CHANNELS, **kw)
    return _make


def _checkpoint(model) -> dict:
    """The subset of a pretrain checkpoint that assert_frozen_constants reads."""
    cfg = model.config_dict()
    return {
        "arm": model.cfg.arm,
        "tau_max": model.tau_max,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "config": {"arm": model.cfg.arm, "model": cfg},
    }


@pytest.mark.parametrize("arm", ARMS)
def test_a_freshly_loaded_model_passes(ft_model, arm):
    m = ft_model(arm)
    report = assert_frozen_constants(m, _checkpoint(m))
    assert report["tau_max"]["value"] == TAU_MAX
    assert report["s"]["value"] == m.s
    assert report["race_encoding"]["channels"] == list(m.demo_channels)
    # every coefficient generator's Fourier band was checked, at every site
    assert set(report["fourier_buffers"]) == {n for n, _ in D.age_conditioner_sites(m)}


def test_tau_max_drift_raises(ft_model):
    m = ft_model("kernel")
    ck = _checkpoint(m)
    m.set_tau_max(TAU_MAX + 1e-3)
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN.*tau_max"):
        assert_frozen_constants(m, ck)


def test_age_standardization_drift_raises(ft_model):
    m = ft_model("kernel", age_mean=63.336, age_sd=16.5748)
    ck = _checkpoint(m)
    # Exactly what re-deriving on PIC would do: a child moves from ~-3.5 to ~0.
    m.set_age_standardization(1.5, 3.0)
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN.*age standardization"):
        assert_frozen_constants(m, ck)


def test_rebuilt_fourier_band_raises(ft_model):
    m = ft_model("kernel")
    ck = _checkpoint(m)
    site = m.kernel_sites()[0][1]
    with torch.no_grad():
        site.age.fourier.frequencies[0] *= 1.0000001
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN.*frequencies"):
        assert_frozen_constants(m, ck)


def test_missing_fourier_buffer_in_checkpoint_raises(ft_model):
    m = ft_model("kernel")
    ck = _checkpoint(m)
    key = next(k for k in ck["model_state_dict"] if k.endswith(".fourier.frequencies"))
    del ck["model_state_dict"][key]
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN.*no buffer"):
        assert_frozen_constants(m, ck)


def test_permuted_race_ordering_raises(ft_model):
    m = ft_model("kernel")
    ck = _checkpoint(m)
    ch = list(ck["config"]["model"]["demo_channels"])
    ch[2], ch[3] = ch[3], ch[2]          # swap two race columns
    ck["config"]["model"]["demo_channels"] = ch
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN.*ordering"):
        assert_frozen_constants(m, ck)


def test_missing_demo_channels_raises(ft_model):
    m = ft_model("kernel")
    ck = _checkpoint(m)
    ck["config"]["model"]["demo_channels"] = []
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN.*demo_channels"):
        assert_frozen_constants(m, ck)


def test_degree_s_mismatch_raises(ft_model):
    m = ft_model("kernel")
    ck = _checkpoint(m)
    ck["config"]["model"]["s"] = m.s + 1
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN"):
        assert_frozen_constants(m, ck)


def test_config_tau_max_disagreeing_with_the_field_raises(ft_model):
    m = ft_model("kernel")
    ck = _checkpoint(m)
    ck["config"]["model"]["tau_max"] = TAU_MAX + 0.5
    with pytest.raises(AssertionError, match=r"INV-FT-FROZEN"):
        assert_frozen_constants(m, ck)
