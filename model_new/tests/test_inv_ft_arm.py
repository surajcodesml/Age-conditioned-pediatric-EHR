#!/usr/bin/env python3
"""INV-FT-ARM -- the arm is read from the checkpoint, never asserted by the caller."""

from __future__ import annotations

import pytest

from model_new.arms import ARMS
from model_new.train_finetune import checkpoint_arm, resolve_arm_from_checkpoint


def _ckpt(arm: str | None, inner: str | None = "__same__") -> dict:
    cfg_arm = arm if inner == "__same__" else inner
    ck: dict = {"config": {}}
    if arm is not None:
        ck["arm"] = arm
    if cfg_arm is not None:
        ck["config"]["arm"] = cfg_arm
    return ck


@pytest.mark.parametrize("arm", ARMS)
def test_arm_comes_from_the_checkpoint(arm):
    assert resolve_arm_from_checkpoint(_ckpt(arm), None) == arm
    assert resolve_arm_from_checkpoint(_ckpt(arm), arm) == arm


def test_only_one_of_the_two_fields_is_enough():
    assert resolve_arm_from_checkpoint({"arm": "kernel", "config": {}}, None) == "kernel"
    assert resolve_arm_from_checkpoint({"config": {"arm": "kernel"}}, None) == "kernel"


def test_disagreeing_flag_raises():
    with pytest.raises(AssertionError, match=r"INV-FT-ARM"):
        resolve_arm_from_checkpoint(_ckpt("kernel"), "vanilla")


def test_shared_backbone_mismatch_must_be_declared():
    """DECISION D2's shared-vanilla design is a deliberate mismatch. It is allowed only
    when named, and then the effective arm is the flag's, not the checkpoint's."""
    ck = _ckpt("vanilla")
    assert resolve_arm_from_checkpoint(ck, "kernel", allow_mismatch=True) == "kernel"
    assert checkpoint_arm(ck) == "vanilla", "the pretrain arm is still recoverable"
    with pytest.raises(AssertionError, match=r"INV-FT-ARM"):
        resolve_arm_from_checkpoint(ck, "kernel")


def test_checkpoint_disagreeing_with_itself_raises():
    with pytest.raises(AssertionError, match=r"INV-FT-ARM"):
        resolve_arm_from_checkpoint(_ckpt("kernel", inner="additive"), None)


def test_missing_arm_raises():
    with pytest.raises(AssertionError, match=r"INV-FT-ARM"):
        resolve_arm_from_checkpoint({"config": {}}, None)
    # ... and supplying it on the command line does not rescue it: the arm is a property
    # of the weights, so a checkpoint that does not record it cannot be fine-tuned.
    with pytest.raises(AssertionError, match=r"INV-FT-ARM"):
        resolve_arm_from_checkpoint({"config": {}}, "kernel")


def test_unknown_arm_raises():
    with pytest.raises(AssertionError, match=r"INV-FT-ARM"):
        resolve_arm_from_checkpoint(_ckpt("kernel_v2"), None)
