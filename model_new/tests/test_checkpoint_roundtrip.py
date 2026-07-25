#!/usr/bin/env python3
"""Save -> load reproduces bit-identical outputs for every arm, and a fine-tune whose corpus
tau_max differs from the checkpoint's raises rather than silently recomputing."""

from __future__ import annotations

import pytest
import torch

from model_new.arms import ARMS
from model_new.tests.conftest import wake_generators_
from model_new.train_finetune import resolve_tau_max


def test_roundtrip_is_bit_identical_for_every_arm(model_factory, batch, tmp_path):
    for arm in ARMS:
        m = model_factory(arm).eval()
        wake_generators_(m)                     # trained-like state, not the zero init
        with torch.no_grad():
            before = m(batch)["code_logits"].clone()
        p = tmp_path / f"{arm}.pt"
        torch.save({"model_state_dict": m.state_dict(), "tau_max": m.tau_max}, p)

        ck = torch.load(p, weights_only=False)
        m2 = model_factory(arm, tau_max=1.0).eval()   # deliberately wrong tau_max
        m2.load_state_dict(ck["model_state_dict"])
        assert m2.tau_max == m.tau_max, f"{arm}: tau_max was rebuilt, not restored"
        with torch.no_grad():
            after = m2(batch)["code_logits"]
        assert torch.equal(before, after), arm


def test_finetune_refuses_a_different_tau_max():
    """The failure mode this guards: a fine-tune corpus with a different lag distribution
    silently re-deriving tau_max, which changes the meaning of every learned coefficient."""
    ckpt = {"tau_max": 6.437218189239502}
    corpus_derived = 5.9012345          # what PIC would give if we recomputed
    with pytest.raises(AssertionError, match=r"INV-TMAX"):
        resolve_tau_max(ckpt, corpus_derived)
    assert resolve_tau_max(ckpt, None) == ckpt["tau_max"]


def test_classification_head_loads_from_a_pretrain_checkpoint(model_factory, tmp_path,
                                                              embedding_table):
    from model_new.model import DKMModel
    from model_new.train_finetune import load_backbone

    pre = model_factory("kernel")
    wake_generators_(pre)
    p = tmp_path / "pre.pt"
    torch.save({"model_state_dict": pre.state_dict(), "tau_max": pre.tau_max}, p)

    ft = DKMModel(num_codes=pre.num_codes, embedding_table=embedding_table, arm="kernel",
                  seed=0, d_model=pre.d_model, s=pre.s, tau_max=pre.tau_max, age_M=4,
                  age_hidden=8, demo_hidden=8, demo_dim=9, task="classification")
    info = load_backbone(ft, torch.load(p, weights_only=False)["model_state_dict"], "kernel")
    # Only head keys may differ: the backbone and the age modules must have transferred.
    assert all(k.startswith("head.") for k in info["unexpected_keys"]), info
    assert all(k.startswith("head.") for k in info["missing_keys"]), info
    for (na, pa), (nb, pb) in zip(sorted(pre.encoder.named_parameters()),
                                  sorted(ft.encoder.named_parameters())):
        assert na == nb and torch.equal(pa, pb), na
