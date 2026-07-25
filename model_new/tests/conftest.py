#!/usr/bin/env python3
"""Shared fixtures: a tiny synthetic corpus, CPU only, well under a minute."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model_new.data import pretrain_collate
from model_new.model import DKMModel

VOCAB = 32
D_IN = 24
D_MODEL = 16
S = 5
TAU_MAX = 6.5


def make_items(rng: np.random.Generator, lengths=(6, 1, 9, 4)) -> list[dict]:
    """Ragged: includes a length-1 row and rows that will be heavily padded (INV-NAN)."""
    items = []
    for n in lengths:
        t = np.sort(rng.random(n) * 900.0).astype(np.float32)
        items.append({
            "code_indices": rng.integers(0, VOCAB, size=n),
            "timestamps_days": t,
            "age_days": (np.sort(rng.random(n) * 900.0) + rng.uniform(200, 28000)
                         ).astype(np.float32),
            "sex": int(rng.integers(0, 2)),
            "race": int(rng.integers(0, 7)),
            "unk_vocab_index": VOCAB,
            "target_codes": (rng.random(VOCAB) < 0.1).astype(np.float32),
        })
    return items


@pytest.fixture(scope="session")
def embedding_table() -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randn(VOCAB + 2, D_IN, generator=g)


@pytest.fixture
def batch() -> dict:
    return pretrain_collate(make_items(np.random.default_rng(0)))


@pytest.fixture
def batch_factory():
    def _make(seed: int = 0, lengths=(6, 1, 9, 4)) -> dict:
        return pretrain_collate(make_items(np.random.default_rng(seed), lengths))
    return _make


@pytest.fixture
def model_factory(embedding_table):
    def _make(arm: str = "vanilla", **kw) -> DKMModel:
        params = dict(num_codes=VOCAB, embedding_table=embedding_table, arm=arm, seed=0,
                      d_model=D_MODEL, n_layers=1, s=S, tau_max=TAU_MAX, age_M=4,
                      age_hidden=8, demo_hidden=8, demo_dim=9)
        params.update(kw)
        return DKMModel(**params)
    return _make


def wake_generators_(model: DKMModel, scale: float = 0.5, seed: int = 7) -> None:
    """The generator's final layer is zero-initialised, so Delta-alpha is identically zero at
    init and age cannot move anything. Tests that probe age *sensitivity* must wake it up
    first, or they pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.age_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
