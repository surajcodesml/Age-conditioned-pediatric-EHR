#!/usr/bin/env python3
"""INV-GROUPS -- the three optimizer groups are pairwise disjoint and their union is exactly
the trainable set. Membership comes from module-declared sets, never from name matching."""

from __future__ import annotations

import inspect

import pytest
import torch

from model_new import optim
from model_new.arms import ARMS
from model_new.optim import build_param_groups


def test_groups_partition_the_trainable_set(model_factory):
    for arm in ARMS:
        m = model_factory(arm)
        groups, report = build_param_groups(m, 1e-4, 1e-3, 1e-3)
        ids = [{id(p) for p in g["params"]} for g in groups]
        assert ids[0] & ids[1] == set()
        assert ids[0] & ids[2] == set()
        assert ids[1] & ids[2] == set()
        union = ids[0] | ids[1] | ids[2]
        trainable = {id(p) for p in m.parameters() if p.requires_grad}
        assert union == trainable, arm
        assert sum(report["n_params"].values()) == sum(
            p.numel() for p in m.parameters() if p.requires_grad)


def test_age_group_is_empty_only_for_vanilla(model_factory):
    for arm in ARMS:
        m = model_factory(arm)
        groups, _ = build_param_groups(m, 1e-4, 1e-3, 1e-3)
        age = next(g for g in groups if g["name"] == "age")
        if arm == "vanilla":
            assert len(age["params"]) == 0
        else:
            assert len(age["params"]) > 0, arm


def test_learning_rates_are_attached_per_group(model_factory):
    m = model_factory("kernel")
    groups, _ = build_param_groups(m, 1e-4, 1e-3, 5e-3)
    opt = torch.optim.Adam(groups)
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs == {"backbone": 1e-4, "age": 1e-3, "head": 5e-3}


def test_no_name_string_matching_in_optim():
    """The original failure was a renamed parameter falling into the backbone group at a
    3,400x smaller learning rate. A declaration cannot be broken by a rename."""
    src = inspect.getsource(optim)
    for needle in ("startswith(", "endswith(", "in name", "named_parameters("):
        assert needle not in src, f"optim.py must not inspect parameter names: found {needle!r}"


def test_declared_membership_survives_a_rename(model_factory):
    """Rename the attribute holding a generator; group membership must not change."""
    m = model_factory("kernel")
    before = {id(p) for p in m.age_parameters()}
    site = m.kernel_sites()[0][1]
    site.add_module("renamed_generator_xyz", site.age.generator)
    after = {id(p) for p in m.age_parameters()}
    assert before == after
