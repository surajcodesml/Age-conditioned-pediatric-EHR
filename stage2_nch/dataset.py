"""NCH next-encounter dataset on top of the Stage-1 shard schema."""
from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch

from model_new.data import DAYS_PER_YEAR, TensorizedPretrainDataset, pretrain_collate
from stage2_nch.config import age_band_name


class NCHForecastDataset(TensorizedPretrainDataset):
    """Same INV-HORIZON windows as Stage-1, plus stratification metadata."""

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        d = super()._load_shard(shard_id)
        npz = d["_npz"]
        if "subject_id" in npz.files and "subject_id" not in d:
            d["subject_id"] = npz["subject_id"]
        return d

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id, pos, visit_k = self._index[idx]
        item = super().__getitem__(idx)
        s = self._load_shard(shard_id)
        last_age_days = float(item["age_days"][-1]) if item["age_days"].size else float("nan")
        item["patient_id"] = int(s["subject_id"][pos]) if "subject_id" in s else -1
        item["n_prior_visits"] = int(visit_k + 1)
        item["n_input_events"] = int(item["code_indices"].shape[0])
        item["last_age_days"] = last_age_days
        item["last_age_years"] = last_age_days / DAYS_PER_YEAR
        item["age_band"] = age_band_name(item["last_age_years"])
        return item

    def patient_ids(self) -> np.ndarray:
        ids = []
        for shard_id in range(len(self._shard_paths)):
            s = self._load_shard(shard_id)
            if "subject_id" in s:
                ids.append(np.asarray(s["subject_id"], dtype=np.int64))
        if not ids:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(ids))


def nch_collate(batch: list[dict[str, Any]], *, race_encoding: str = "one_hot",
                assert_horizon: bool | None = None) -> dict:
    out = pretrain_collate(batch, race_encoding=race_encoding, assert_horizon=assert_horizon)
    out["patient_id"] = torch.tensor([int(item["patient_id"]) for item in batch], dtype=torch.long)
    out["n_input_events"] = torch.tensor([int(item["n_input_events"]) for item in batch],
                                         dtype=torch.long)
    out["n_prior_visits"] = torch.tensor([int(item["n_prior_visits"]) for item in batch],
                                         dtype=torch.long)
    out["last_age_years"] = torch.tensor([float(item["last_age_years"]) for item in batch],
                                         dtype=torch.float32)
    return out


def make_nch_collate(race_encoding: str = "one_hot", assert_horizon: bool | None = None):
    return partial(nch_collate, race_encoding=race_encoding, assert_horizon=assert_horizon)
