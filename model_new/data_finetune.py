#!/usr/bin/env python3
"""Fine-tune (binary disease classification) dataset + collate.

Identical batch contract to :mod:`model_new.data` plus ``labels [B]``, and it reuses that
module's padding and demographic layout. Like the pretrain collate it ships
``timestamps_days`` + ``lengths`` and lets the model compute ``tau`` on the GPU -- so the
``/7`` and ``log1p`` convention still lives in exactly one place (``data.lag_to_tau``).

``tau_max`` is **not** derived here. It comes from the pretraining checkpoint and is reused
verbatim (D8, INV-TMAX). :func:`check_tau_max` exists to report how much of the fine-tune
corpus falls outside that domain; the clamp rate is a MEASURE quantity, not a reason to
re-derive the constant.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from model_new.data import (
    DAYS_PER_YEAR, _build_demographics, _pad_common, _sample_indices, spans_to_tau,
)

__all__ = ["TensorizedFinetuneDataset", "finetune_collate", "make_finetune_collate",
           "check_tau_max"]


class TensorizedFinetuneDataset(Dataset):
    """Flat mmap-able classification shards: one row per (subject, admission) example."""

    def __init__(self, tensorized_split_dir: str | Path, max_seq_len: int = 1024,
                 shard_cache_size: int = 4) -> None:
        self.dir = Path(tensorized_split_dir)
        self.max_seq_len = int(max_seq_len)
        self.shard_cache_size = int(shard_cache_size)
        self._shard_paths = sorted(self.dir.glob("shard_*.npz"))
        if not self._shard_paths:
            raise FileNotFoundError(f"no shard_*.npz in {self.dir}")
        self._index: list[tuple[int, int]] = []
        for shard_id, path in enumerate(self._shard_paths):
            npz = np.load(path, mmap_mode="r", allow_pickle=False)
            if "offsets" not in npz.files:
                npz.close()
                raise RuntimeError(f"{path} is not the flat schema; re-tensorize.")
            n = int(len(npz["subject_id"]))
            npz.close()
            self._index.extend((shard_id, pos) for pos in range(n))
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._cache:
            d = self._cache.pop(shard_id)
            self._cache[shard_id] = d
            return d
        if len(self._cache) >= self.shard_cache_size:
            _, old = self._cache.popitem(last=False)
            try:
                old["_npz"].close()
            except Exception:
                pass
        npz = np.load(self._shard_paths[shard_id], mmap_mode="r", allow_pickle=False)
        d: dict[str, Any] = {
            "_npz": npz,
            "unk_vocab_index": int(np.asarray(npz["unk_vocab_index"]).reshape(-1)[0]),
            "hadm_id": npz["hadm_id"] if "hadm_id" in npz.files else None,
        }
        for key in ("offsets", "code_indices", "timestamps_days", "age_days", "subject_id",
                    "sex", "race", "label"):
            d[key] = npz[key]
        self._cache[shard_id] = d
        return d

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id, pos = self._index[idx]
        s = self._load_shard(shard_id)
        start, end = int(s["offsets"][pos]), int(s["offsets"][pos + 1])
        codes = np.asarray(s["code_indices"][start:end], dtype=np.int64)
        ts = np.asarray(s["timestamps_days"][start:end], dtype=np.float32)
        ages = np.asarray(s["age_days"][start:end], dtype=np.float32)
        if codes.shape[0] > self.max_seq_len:
            sl = slice(-self.max_seq_len, None)
            codes, ts, ages = codes[sl], ts[sl], ages[sl]
        return {
            "code_indices": codes,
            "timestamps_days": ts,
            "age_days": ages,
            "sex": int(s["sex"][pos]),
            "race": int(s["race"][pos]),
            "unk_vocab_index": int(s["unk_vocab_index"]),
            "label": float(s["label"][pos]),
            "subject_id": int(s["subject_id"][pos]),
            "hadm_id": int(s["hadm_id"][pos]) if s["hadm_id"] is not None else -1,
        }

    def __del__(self) -> None:
        for shard in getattr(self, "_cache", {}).values():
            try:
                shard["_npz"].close()
            except Exception:
                pass
        if hasattr(self, "_cache"):
            self._cache.clear()


def finetune_collate(batch: list[dict[str, Any]], *, race_encoding: str = "one_hot") -> dict:
    common = _pad_common(batch)
    out = {k: v for k, v in common.items() if not k.startswith("_")}
    out["demographics"] = _build_demographics(common, race_encoding)
    out["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    out["subject_id"] = [int(item["subject_id"]) for item in batch]
    out["hadm_id"] = [int(item["hadm_id"]) for item in batch]
    return out


def make_finetune_collate(race_encoding: str = "one_hot"):
    return partial(finetune_collate, race_encoding=race_encoding)


def check_tau_max(dataset: TensorizedFinetuneDataset, tau_max_from_checkpoint: float,
                  n_samples: int = 4000, seed: int = 0) -> dict:
    """MEASURE: what fraction of this corpus lies outside the checkpoint's ``tau`` domain.

    Reported, never acted on. Re-deriving ``tau_max`` from the fine-tune corpus would change
    the meaning of every learned coefficient, which is exactly what INV-TMAX forbids.
    """
    idxs = _sample_indices(len(dataset), n_samples, seed)
    spans = np.empty(len(idxs), dtype=np.float64)
    ages: list[np.ndarray] = []
    for i, j in enumerate(idxs):
        item = dataset[int(j)]
        ts = item["timestamps_days"]
        spans[i] = float(ts.max() - ts.min()) if ts.size else 0.0
        ages.append(item["age_days"].astype(np.float64) / DAYS_PER_YEAR)
    taus = spans_to_tau(spans)
    a = np.concatenate(ages) if ages else np.zeros(0)
    return {
        "tau_max_checkpoint": float(tau_max_from_checkpoint),
        "corpus_tau_max_observed": float(taus.max()),
        "fraction_windows_exceeding": float((taus > tau_max_from_checkpoint).mean()),
        "corpus_tau_p50": float(np.percentile(taus, 50)),
        "corpus_tau_p99": float(np.percentile(taus, 99)),
        "n_sampled": int(len(idxs)),
        "min_age": float(a.min()) if a.size else float("nan"),
        "max_age": float(a.max()) if a.size else float("nan"),
        "median_age": float(np.median(a)) if a.size else float("nan"),
    }


def _smoke() -> None:
    from model_new import diagnostics

    rng = np.random.default_rng(0)
    items = []
    for n, lab in ((4, 1.0), (7, 0.0)):
        items.append({
            "code_indices": rng.integers(0, 50, size=n),
            "timestamps_days": np.sort(rng.random(n) * 20).astype(np.float32),
            "age_days": (np.sort(rng.random(n) * 20) + 300).astype(np.float32),
            "sex": 0, "race": 6, "unk_vocab_index": 50, "label": lab,
            "subject_id": 1, "hadm_id": 2,
        })
    b = finetune_collate(items)
    diagnostics.print_block("data_finetune.py smoke", [
        f"keys           : {sorted(k for k in b if isinstance(b[k], torch.Tensor))}",
        f"labels         : {b['labels'].tolist()}",
        f"timestamps_days: {tuple(b['timestamps_days'].shape)} {b['timestamps_days'].dtype}  "
        f"(tau computed in the model, not shipped)",
        f"lengths        : {b['lengths'].tolist()}",
        f"demographics   : {tuple(b['demographics'].shape)}",
        f"age_years      : {[round(x, 3) for x in b['age_years'][0].tolist()]}",
        "tau_max is NOT computed here: it is restored from the checkpoint (INV-TMAX).",
    ])


if __name__ == "__main__":
    _smoke()
