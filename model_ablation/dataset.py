#!/usr/bin/env python3
"""Pretrain dataset + collate for the ablation's SHARED vanilla pretrain.

Self-contained port of the frozen tensorized pretrain reader. Same age-split
convention as ``dataset_finetune.py``: ``age_years`` is a separate [B, L] field and
``demographics`` is [B, L, 2] = (sex, race). No time/Weibull target is used
(``no_time_loss`` is locked on), so only ``target_codes`` is produced for the loss.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from model_ablation.dataset_finetune import _dataloader_worker_init, encode_race

__all__ = ["TensorizedEHRDataset", "ehr_collate", "_dataloader_worker_init"]


class TensorizedEHRDataset(Dataset):
    """Visit-level next-code prediction samples from tensorized pretrain shards."""

    def __init__(self, tensorized_dir: str | Path, code_vocab_path: str | Path,
                 max_seq_len: int = 1024, shard_cache_size: int = 4) -> None:
        self.tensorized_dir = Path(tensorized_dir)
        self.code_vocab_path = Path(code_vocab_path)
        self.max_seq_len = int(max_seq_len)
        self.shard_cache_size = int(shard_cache_size)
        with self.code_vocab_path.open("r", encoding="utf-8") as f:
            self.code_vocab: dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}
        self.num_codes = len(self.code_vocab)
        self.unk_vocab_index = self.num_codes

        self._shard_paths = sorted(self.tensorized_dir.glob("shard_*.npz"))
        if not self._shard_paths:
            raise FileNotFoundError(f"No shards in {self.tensorized_dir}")
        self._index: list[tuple[int, int, int]] = []
        for shard_id, shard_path in enumerate(self._shard_paths):
            meta = np.load(shard_path, allow_pickle=True)
            visit_spans_arr = meta["visit_spans"]
            for pos in range(len(visit_spans_arr)):
                spans = np.asarray(visit_spans_arr[pos], dtype=np.int32)
                for v in range(max(0, int(spans.shape[0]) - 1)):
                    self._index.append((shard_id, pos, v))
            del meta
        self._shard_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._shard_cache:
            shard = self._shard_cache.pop(shard_id)
            self._shard_cache[shard_id] = shard
            return shard
        if len(self._shard_cache) >= self.shard_cache_size:
            del self._shard_cache[next(iter(self._shard_cache))]
        npz = np.load(self._shard_paths[shard_id], allow_pickle=True)
        shard = {k: npz[k] for k in npz.files}
        npz.close()
        self._shard_cache[shard_id] = shard
        return shard

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id, pos, visit_k = self._index[idx]
        shard = self._load_shard(shard_id)
        spans = np.asarray(shard["visit_spans"][pos], dtype=np.int32)
        if visit_k >= len(spans) - 1:
            raise IndexError(f"visit_k={visit_k} out of range ({len(spans)} visits)")
        end_curr = int(spans[visit_k][1])
        start_next = int(spans[visit_k + 1][0])
        end_next = int(spans[visit_k + 1][1])

        code_all = np.asarray(shard["code_id"][pos], dtype=object)
        ts_all = np.asarray(shard["timestamps_days"][pos], dtype=np.float32)
        age_all = np.asarray(shard["age_days"][pos], dtype=np.float32)

        code_slice = code_all[:end_curr]
        timestamps_days = ts_all[:end_curr].copy()
        age_days = age_all[:end_curr].copy()
        if code_slice.shape[0] > self.max_seq_len:
            sl = slice(-self.max_seq_len, None)
            code_slice, timestamps_days, age_days = code_slice[sl], timestamps_days[sl], age_days[sl]

        code_indices = np.fromiter(
            (self.code_vocab.get(str(c), self.unk_vocab_index) for c in code_slice),
            dtype=np.int64, count=len(code_slice),
        )
        target = np.zeros(self.num_codes, dtype=np.float32)
        next_codes = np.fromiter(
            (self.code_vocab.get(str(c), self.unk_vocab_index)
             for c in np.asarray(code_all[start_next:end_next], dtype=object)),
            dtype=np.int64, count=int(end_next - start_next),
        )
        valid = next_codes[next_codes != self.unk_vocab_index]
        if valid.size:
            target[np.unique(valid)] = 1.0

        return {
            "code_indices": code_indices,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": int(shard["sex"][pos]),
            "race": int(encode_race(shard["race"][pos])),
            "unk_vocab_index": self.unk_vocab_index,
            "target_codes": target,
        }


def ehr_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad; emit age_years separately and demographics=[sex, race]. Code loss only."""
    if not batch:
        raise ValueError("empty batch")
    bsz = len(batch)
    unk_id = int(batch[0]["unk_vocab_index"])
    max_len = max(int(item["code_indices"].shape[0]) for item in batch)

    code_np = np.full((bsz, max_len), -1, dtype=np.int64)
    ts_np = np.zeros((bsz, max_len), dtype=np.float32)
    age_np = np.zeros((bsz, max_len), dtype=np.float32)
    mask_np = np.zeros((bsz, max_len), dtype=bool)
    sex_arr = np.array([item["sex"] for item in batch], dtype=np.float32)
    race_arr = np.array([item["race"] for item in batch], dtype=np.float32)
    target_codes_np = np.stack([item["target_codes"] for item in batch], axis=0)

    for b, item in enumerate(batch):
        L = int(item["code_indices"].shape[0])
        if L == 0:
            continue
        code_np[b, :L] = item["code_indices"].astype(np.int64)
        ts_np[b, :L] = item["timestamps_days"].astype(np.float32)
        age_np[b, :L] = item["age_days"].astype(np.float32)
        mask_np[b, :L] = True

    code_indices = torch.from_numpy(code_np)
    attention_mask = torch.from_numpy(mask_np)
    timestamps_days = torch.from_numpy(ts_np)
    bge_codes = torch.where(
        attention_mask,
        torch.where(code_indices == unk_id, torch.ones((), dtype=torch.long), code_indices + 2),
        torch.zeros((), dtype=torch.long),
    )
    t = timestamps_days
    delta_t = torch.log1p(torch.abs(t.unsqueeze(2) - t.unsqueeze(1)) / 7.0)
    pair_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    delta_t = delta_t * pair_mask.float()

    age_years = (age_np / 365.25) * mask_np.astype(np.float32)
    demo = np.zeros((bsz, max_len, 2), dtype=np.float32)
    demo[:, :, 0] = sex_arr[:, np.newaxis]
    demo[:, :, 1] = race_arr[:, np.newaxis]
    demo = demo * mask_np[:, :, np.newaxis].astype(np.float32)

    return {
        "code_indices": bge_codes,
        "timestamps_days": timestamps_days,
        "delta_t": delta_t,
        "attention_mask": attention_mask,
        "age_years": torch.from_numpy(age_years),
        "demographics": torch.from_numpy(demo),
        "target_codes": torch.from_numpy(target_codes_np),
    }
