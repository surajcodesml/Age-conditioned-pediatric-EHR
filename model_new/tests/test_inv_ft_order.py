#!/usr/bin/env python3
"""INV-FT-ORDER -- the four arms consume the same data in the same order.

The failure this exists to catch is silent and specific: with the default sampler the
shuffle is driven by the global RNG, and constructing the age modules consumes a different
number of global draws in each arm, so ``kernel`` and ``vanilla`` would see different
batches and every downstream difference would be confounded with batch order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from model_new.arms import ARMS
from model_new.data_finetune import make_finetune_collate
from model_new.model import DKMModel
from model_new.train_finetune import (
    assert_order_matches_siblings, eval_order_hash, train_order_hash,
)

from .conftest import VOCAB


class _Tiny(torch.utils.data.Dataset):
    """Six labelled sequences; the smallest thing with a nontrivial batch order."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, i: int) -> dict:
        n = 2 + (i % 3)
        return {
            "code_indices": np.arange(i, i + n) % VOCAB,
            "timestamps_days": np.linspace(0.0, 0.9, n).astype(np.float32),
            "age_days": np.linspace(30.0, 30.0 + n, n).astype(np.float32),
            "sex": i % 2, "race": 6, "unk_vocab_index": VOCAB,
            "label": float(i % 2), "subject_id": i, "hadm_id": 100 + i,
        }


def _loader(shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(_Tiny(), batch_size=2, shuffle=shuffle, generator=generator,
                      collate_fn=make_finetune_collate("one_hot"))


def test_train_order_hash_is_seed_determined_and_not_arm_determined(embedding_table):
    """The whole point: same seed -> same order, regardless of how many global RNG draws
    building the arm's age modules consumed."""
    hashes = []
    for arm in ARMS:
        torch.manual_seed(0)
        DKMModel(num_codes=VOCAB, embedding_table=embedding_table, arm=arm, seed=0,
                 d_model=16, n_layers=1, age_M=4, age_hidden=8, demo_hidden=8, demo_dim=9)
        hashes.append(train_order_hash(6, seed=0, epochs=3)["hash"])
    assert len(set(hashes)) == 1, f"arms disagree on the training order: {hashes}"


def test_train_order_hash_separates_seeds_and_epochs():
    a = train_order_hash(6, seed=0, epochs=3)["hash"]
    assert a == train_order_hash(6, seed=0, epochs=3)["hash"]
    assert a != train_order_hash(6, seed=1, epochs=3)["hash"]
    assert a != train_order_hash(6, seed=0, epochs=2)["hash"]


def test_an_owned_generator_makes_the_shuffle_arm_independent():
    """A DataLoader driven by an owned generator reproduces its order; one driven by the
    global RNG does not, once anything else has drawn from it."""
    def owned() -> list[int]:
        return [int(v) for b in _loader(True, torch.Generator().manual_seed(0))
                for v in b["subject_id"]]

    assert owned() == owned()

    torch.manual_seed(0)
    a = [int(v) for b in _loader(True) for v in b["subject_id"]]
    torch.manual_seed(0)
    torch.randn(17)                       # stands in for constructing the age modules
    b = [int(v) for b_ in _loader(True) for v in b_["subject_id"]]
    assert a != b, "the global-RNG shuffle must be the fragile one this invariant avoids"


def test_eval_order_hash_is_deterministic_and_content_sensitive():
    h1 = eval_order_hash(_loader(False))
    h2 = eval_order_hash(_loader(False))
    assert h1["hash"] == h2["hash"]
    assert h1["n_batches"] == 3 and h1["n_rows"] == 6
    assert eval_order_hash(_loader(True, torch.Generator().manual_seed(3)))["hash"] \
        != h1["hash"]


def _write_sibling(root, run, arm, task, hashes) -> None:
    d = root / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "pic_config.json").write_text(json.dumps(
        {"arm": arm, "task": task, "data_order": {"task": task, "hashes": hashes}}))


def test_matching_siblings_pass_and_mismatching_ones_raise(tmp_path):
    mine = {"task": "heart_malformations",
            "hashes": {"train": {"hash": "aaaa"}, "val": {"hash": "bbbb"}}}
    _write_sibling(tmp_path, "vanilla_s0", "vanilla", "heart_malformations",
                   {"train": {"hash": "aaaa"}, "val": {"hash": "bbbb"}})
    rep = assert_order_matches_siblings(tmp_path, "kernel_s0", "kernel", mine)
    assert rep["n_siblings_checked"] == 1

    _write_sibling(tmp_path, "additive_s0", "additive", "heart_malformations",
                   {"train": {"hash": "ZZZZ"}, "val": {"hash": "bbbb"}})
    with pytest.raises(AssertionError, match=r"INV-FT-ORDER"):
        assert_order_matches_siblings(tmp_path, "kernel_s0", "kernel", mine)


def test_a_different_task_is_not_a_sibling(tmp_path):
    """Two tasks legitimately have different data orders; only same-task runs are compared."""
    mine = {"task": "pneumonia", "hashes": {"train": {"hash": "aaaa"}}}
    _write_sibling(tmp_path, "vanilla_s0", "vanilla", "mortality",
                   {"train": {"hash": "ZZZZ"}})
    assert assert_order_matches_siblings(tmp_path, "kernel_s0", "kernel",
                                         mine)["n_siblings_checked"] == 0
