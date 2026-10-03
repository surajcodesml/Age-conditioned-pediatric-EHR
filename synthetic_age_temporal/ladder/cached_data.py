"""Precomputed encounter tensors for the ladder.

The training math is unchanged. Each example is padded to the split maximum
and the model already ignores pad positions, so a batch matches dynamic
padding on every value the network reads. The DataLoader still shuffles with
the default sampler, which draws its seed from the torch RNG after
``set_seed`` inside the existing trainer.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from baselines.synthetic.data_adapter import make_dtr_baseline_loaders

_CACHE: dict[tuple[str, int, int], tuple[Any, ...]] = {}

_TENSOR_KEYS = (
    "enc_code_ids",
    "enc_code_mask",
    "enc_tau",
    "enc_lag_days",
    "enc_padding_mask",
    "enc_n_signal",
    "age",
    "z_age",
    "labels",
)


class _CachedEncounters(Dataset):
    def __init__(self, store: dict[str, Any]) -> None:
        self.store = store

    def __len__(self) -> int:
        return int(self.store["age"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        store = self.store
        item = {key: store[key][idx] for key in _TENSOR_KEYS}
        item["example_id"] = int(store["example_ids"][idx])
        item["patient_id"] = store["patient_ids"][idx]
        return item


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {key: torch.stack([row[key] for row in batch]) for key in _TENSOR_KEYS}
    out["example_ids"] = torch.tensor([row["example_id"] for row in batch], dtype=torch.long)
    out["patient_ids"] = [row["patient_id"] for row in batch]
    return out


def _materialize(dataset) -> _CachedEncounters:
    n = len(dataset)
    items = [dataset[i] for i in range(n)]
    width = max((item["n_encounters"] for item in items), default=1)
    width = max(width, 1)
    codes = max(
        (max((len(row) for row in item["enc_codes"]), default=1) for item in items),
        default=1,
    )
    codes = max(codes, 1)
    n_targets = int(items[0]["labels"].shape[0]) if items else 1
    store: dict[str, Any] = {
        "enc_code_ids": torch.zeros(n, width, codes, dtype=torch.long),
        "enc_code_mask": torch.zeros(n, width, codes, dtype=torch.bool),
        "enc_tau": torch.zeros(n, width, dtype=torch.float32),
        "enc_lag_days": torch.zeros(n, width, dtype=torch.float32),
        "enc_padding_mask": torch.ones(n, width, dtype=torch.bool),
        "enc_n_signal": torch.zeros(n, width, dtype=torch.long),
        "age": torch.zeros(n, dtype=torch.float32),
        "z_age": torch.zeros(n, dtype=torch.float32),
        "labels": torch.zeros(n, n_targets, dtype=torch.float32),
        "example_ids": torch.zeros(n, dtype=torch.long),
        "patient_ids": [""] * n,
    }
    for i, item in enumerate(items):
        store["age"][i] = item["age"]
        store["z_age"][i] = item["z_age"]
        store["labels"][i] = torch.tensor(item["labels"], dtype=torch.float32)
        store["example_ids"][i] = int(item["example_id"])
        store["patient_ids"][i] = str(item["patient_id"])
        for j, ids in enumerate(item["enc_codes"]):
            n_ids = len(ids)
            if n_ids == 0:
                continue
            store["enc_code_ids"][i, j, :n_ids] = torch.tensor(ids, dtype=torch.long)
            store["enc_code_mask"][i, j, :n_ids] = True
            store["enc_tau"][i, j] = item["enc_tau"][j]
            store["enc_lag_days"][i, j] = item["enc_lag_days"][j]
            store["enc_n_signal"][i, j] = item["enc_n_signal"][j]
            store["enc_padding_mask"][i, j] = False
    return _CachedEncounters(store)


def cached_dtr_loaders(
    scenario: str,
    *,
    data_seed: int,
    batch_size: int,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Train/val/test loaders whose batches match ``make_dtr_baseline_loaders``."""
    key = (str(scenario), int(data_seed), int(batch_size))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    train_loader, val_loader, test_loader, vocab, info = make_dtr_baseline_loaders(
        scenario,
        data_seed=int(data_seed),
        batch_size=int(batch_size),
    )
    info = dict(info)
    info.pop("examples", None)
    info.pop("labels", None)
    pin = torch.cuda.is_available()
    loaders = []
    for loader in (train_loader, val_loader, test_loader):
        loaders.append(
            DataLoader(
                _materialize(loader.dataset),
                batch_size=int(batch_size),
                shuffle=(loader is train_loader),
                num_workers=0,
                pin_memory=pin,
                collate_fn=_collate,
            )
        )
    built = (loaders[0], loaders[1], loaders[2], vocab, info)
    _CACHE[key] = built
    return built
