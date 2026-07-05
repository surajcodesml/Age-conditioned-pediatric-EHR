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

__all__ = ["TensorizedEHRDataset", "ehr_collate", "_dataloader_worker_init", "encode_race"]


# Defined locally (not imported from dataset_finetune) so the tensorized pretrain
# path never pulls duckdb into the spawn-worker bootstrap. The pretrain tensorized
# reader is pure numpy and has no duckdb dependency.
def _dataloader_worker_init(_worker_id: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def encode_race(race_val) -> int:
    if race_val is None:
        return 6
    if isinstance(race_val, float) and np.isnan(race_val):
        return 6
    s = str(race_val).strip().upper()
    if not s or s == "NAN":
        return 6
    if s in {"UNKNOWN", "UNABLE TO OBTAIN", "PREFER NOT TO SAY", "N/A", "DECLINED"}:
        return 6
    if s.startswith("WHITE"):
        return 0
    if s.startswith("BLACK"):
        return 1
    if s.startswith("ASIAN"):
        return 2
    if s.startswith("HISPANIC"):
        return 3
    if s.startswith("AMERICAN INDIAN") or s.startswith("ALASKA NATIVE"):
        return 4
    if s == "OTHER" or s.startswith("OTHER "):
        return 5
    return 5


class TensorizedEHRDataset(Dataset):
    """Visit-level next-code prediction samples from flat tensorized pretrain shards.

    Reads the spawn-safe flat schema produced by ``tensorize_pretrain.py`` (flat
    numeric arrays + offsets, ``allow_pickle=False``, mmap). This mirrors the
    fine-tune reader (``TensorizedDiseaseClassificationDataset``) which is proven
    clean under a spawned DataLoader, replacing the legacy object-array shards whose
    ``allow_pickle=True`` unpickling races at worker teardown.
    """

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
        # Sample index: (shard_id, patient_pos, visit_k) for each of the (n_visits-1)
        # next-visit prediction targets. Built from visit_offsets only (small, no pickle).
        self._index: list[tuple[int, int, int]] = []
        for shard_id, shard_path in enumerate(self._shard_paths):
            npz = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            if "visit_offsets" not in npz.files:
                raise RuntimeError(f"{shard_path} is old object-array format; re-run tensorize_pretrain.py.")
            visit_offsets = np.asarray(npz["visit_offsets"])
            npz.close()
            n_patients = int(visit_offsets.shape[0]) - 1
            for pos in range(n_patients):
                n_visits = int(visit_offsets[pos + 1] - visit_offsets[pos])
                for v in range(max(0, n_visits - 1)):
                    self._index.append((shard_id, pos, v))
        self._shard_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._shard_cache:
            d = self._shard_cache.pop(shard_id)
            self._shard_cache[shard_id] = d
            return d
        if len(self._shard_cache) >= self.shard_cache_size:
            _, old = self._shard_cache.popitem(last=False)
            npz_old = old.get("_npz")
            if npz_old is not None:
                try:
                    npz_old.close()
                except Exception:
                    pass
        npz = np.load(self._shard_paths[shard_id], mmap_mode="r", allow_pickle=False)
        d: dict[str, Any] = {
            "_npz": npz,
            "event_offsets": npz["event_offsets"],
            "code_indices": npz["code_indices"],
            "timestamps_days": npz["timestamps_days"],
            "age_days": npz["age_days"],
            "visit_offsets": npz["visit_offsets"],
            "visit_starts": npz["visit_starts"],
            "visit_ends": npz["visit_ends"],
            "sex": npz["sex"],
            "race": npz["race"],
            "unk_vocab_index": int(np.asarray(npz["unk_vocab_index"]).reshape(-1)[0]),
        }
        self._shard_cache[shard_id] = d
        return d

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id, pos, visit_k = self._index[idx]
        s = self._load_shard(shard_id)

        ev_start = int(s["event_offsets"][pos])
        vis_start = int(s["visit_offsets"][pos])
        vis_end = int(s["visit_offsets"][pos + 1])
        n_visits = vis_end - vis_start
        if visit_k >= n_visits - 1:
            raise IndexError(f"visit_k={visit_k} out of range ({n_visits} visits)")

        # Visit spans are stored RELATIVE to the patient's event block.
        end_curr = int(s["visit_ends"][vis_start + visit_k])
        start_next = int(s["visit_starts"][vis_start + visit_k + 1])
        end_next = int(s["visit_ends"][vis_start + visit_k + 1])

        code_block = s["code_indices"]
        ts_block = s["timestamps_days"]
        age_block = s["age_days"]

        code_slice = np.asarray(code_block[ev_start:ev_start + end_curr], dtype=np.int64)
        timestamps_days = np.asarray(ts_block[ev_start:ev_start + end_curr], dtype=np.float32)
        age_days = np.asarray(age_block[ev_start:ev_start + end_curr], dtype=np.float32)
        if code_slice.shape[0] > self.max_seq_len:
            sl = slice(-self.max_seq_len, None)
            code_slice, timestamps_days, age_days = code_slice[sl], timestamps_days[sl], age_days[sl]

        unk = int(s["unk_vocab_index"])
        next_codes = np.asarray(
            code_block[ev_start + start_next:ev_start + end_next], dtype=np.int64,
        )
        target = np.zeros(self.num_codes, dtype=np.float32)
        valid = next_codes[next_codes != unk]
        if valid.size:
            target[np.unique(valid)] = 1.0

        return {
            "code_indices": code_slice,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": int(s["sex"][pos]),
            "race": int(s["race"][pos]),
            "unk_vocab_index": unk,
            "target_codes": target,
        }

    def __del__(self) -> None:
        for shard in getattr(self, "_shard_cache", {}).values():
            try:
                npz = shard.get("_npz")
                if npz is not None:
                    npz.close()
            except Exception:
                pass
        if hasattr(self, "_shard_cache"):
            self._shard_cache.clear()


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
