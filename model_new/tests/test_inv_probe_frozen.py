#!/usr/bin/env python3
"""INV-PROBE-FROZEN -- the probe never trains or mutates the encoder.

After a probe extraction pass every parameter is bit-identical to its value at load, and
no parameter had ``requires_grad=True`` at any point during extraction.

Also asserts the extraction path is unreachable from ``train.py`` / ``train_finetune.py``
(D9: a fine-tune path that skipped the head left pooling-site age parameters gradient-dead).
"""

from __future__ import annotations

from pathlib import Path

import torch

from model_new.probe import (
    assert_no_grad_params, assert_state_unchanged, extract_split, freeze_model_,
    snapshot_state,
)
from model_new.tests.conftest import wake_generators_

PKG = Path(__file__).resolve().parents[1]


def test_extract_path_unreachable_from_training_entrypoints():
    """Call sites / kwargs only — documentation mentions of D9 are allowed."""
    import ast

    needle_call = "extract_representations"
    offenders = []
    for name in ("train.py", "train_finetune.py"):
        tree = ast.parse((PKG / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == needle_call:
                    offenders.append(f"{name}:call")
                if isinstance(func, ast.Name) and func.id == needle_call:
                    offenders.append(f"{name}:call")
            if isinstance(node, ast.keyword) and node.arg == "return_repr_only":
                offenders.append(f"{name}:return_repr_only kwarg")
    assert not offenders, (
        f"[INV-PROBE-FROZEN] training entry points must not call extract_representations / "
        f"return_repr_only: {offenders}")


def test_no_return_repr_only_flag_on_forward():
    """D9 regression: forward must not grow a training-reachable repr-only path."""
    import ast
    src = (PKG / "model.py").read_text()
    assert "def extract_representations" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
            assert "return_repr_only" not in args, (
                "[INV-PROBE-FROZEN] forward must not accept return_repr_only")
        if isinstance(node, ast.keyword) and node.arg == "return_repr_only":
            raise AssertionError(
                "[INV-PROBE-FROZEN] return_repr_only kwarg must not appear in model.py")


def test_probe_leaves_parameters_bit_identical(model_factory, batch):
    m = model_factory("kernel")
    wake_generators_(m)
    freeze_model_(m)
    assert_no_grad_params(m)
    before = snapshot_state(m)

    # One extraction pass, the same path probe.py uses.
    from torch.utils.data import DataLoader

    class _One:
        def __iter__(self):
            yield batch

    store = extract_split(m, _One(), torch.device("cpu"), collect_pretrain_targets=True)
    assert store["h_pool"].ndim == 2
    assert_no_grad_params(m)
    assert_state_unchanged(before, m)

    # A second pass still must not mutate.
    _ = extract_split(m, _One(), torch.device("cpu"))
    assert_state_unchanged(before, m)


def test_requires_grad_false_throughout_extraction(model_factory, batch):
    m = model_factory("additive")
    freeze_model_(m)
    # Deliberately try to flip one flag; extract path must still assert.
    for p in m.parameters():
        p.requires_grad_(True)
        break
    try:
        assert_no_grad_params(m)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "[INV-PROBE-FROZEN] expected assert_no_grad_params to fire"
