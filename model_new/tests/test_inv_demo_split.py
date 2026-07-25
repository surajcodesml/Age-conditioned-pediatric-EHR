#!/usr/bin/env python3
"""INV-DEMO-SPLIT -- age_years is its own batch key; no module reads age from demographics.

The demographic vector still CONTAINS age and every arm receives it identically. The point
of the split is that the kernel input and the demographic feature can be varied
independently, not that age is withheld from the baseline.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from model_new import encoder, model as model_mod, pooling
from model_new.arms import ARMS
from model_new.data import demo_layout

PKG = Path(__file__).resolve().parents[1]


def test_age_years_is_a_separate_key(batch):
    assert "age_years" in batch
    assert batch["age_years"].shape == batch["attention_mask"].shape


def test_demographics_still_contains_age(batch):
    demo_dim, names = demo_layout("one_hot")
    assert batch["demographics"].shape[-1] == demo_dim == 9
    assert names[0] == "age_years"
    expected = batch["age_years"] * batch["attention_mask"]
    assert torch.allclose(batch["demographics"][..., 0], expected, atol=1e-6)


def test_every_arm_receives_the_identical_demographic_tensor(model_factory, batch):
    ref = None
    for arm in ARMS:
        m = model_factory(arm)
        with torch.no_grad():
            rows = torch.arange(batch["code_indices"].shape[0])
            last = m.pooling.last_valid_index(batch["attention_mask"])
            demo_last = batch["demographics"][rows, last]
        if ref is None:
            ref = demo_last
        assert torch.equal(demo_last, ref)


def test_no_kernel_module_indexes_demographics():
    """The legacy bug was `age_years = demographics[..., 0]` inside the model."""
    for mod in (encoder, pooling):
        src = inspect.getsource(mod)
        assert "demographics" not in src, f"{mod.__name__} must not mention demographics"
    src = inspect.getsource(model_mod.DKMModel.forward)
    assert "demographics[..., 0]" not in src
    assert 'batch["age_years"]' in src


def test_wrong_demo_width_raises(model_factory, batch):
    m = model_factory("kernel")
    bad = dict(batch)
    bad["demographics"] = batch["demographics"][..., :3]
    with pytest.raises(AssertionError, match=r"INV-DEMO-SPLIT"):
        m(bad)


def test_missing_age_years_raises(model_factory, batch):
    m = model_factory("kernel")
    bad = {k: v for k, v in batch.items() if k != "age_years"}
    with pytest.raises(AssertionError, match=r"INV-DEMO-SPLIT"):
        m(bad)
