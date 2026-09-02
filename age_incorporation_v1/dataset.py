"""Synthea age-signal benchmark loader.

Dataset-specific. Future NCH / ECHO / PIC loaders should emit the same batch
keys as ``collate_batch`` so ``model.py`` stays unchanged.

Never exposes DOB, calendar dates, label-generation probabilities, or
has_SIGNAL_* flags as model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import AGE_GROUPS, LABEL_COL, PAD, UNK, Config


def time_norm_days(days_before_index: np.ndarray, age_scale_years: float) -> np.ndarray:
    denom = np.log1p(age_scale_years * 365.25)
    x = np.log1p(np.asarray(days_before_index, dtype=np.float64)) / denom
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def age_norm_years(age_years: np.ndarray, age_scale_years: float) -> np.ndarray:
    x = np.asarray(age_years, dtype=np.float64) / age_scale_years
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def build_vocab(values: pd.Series) -> dict[str, int]:
    uniq = sorted(str(v) for v in values.dropna().unique())
    vocab = {PAD: 0, UNK: 1}
    for i, v in enumerate(uniq, start=2):
        vocab[v] = i
    return vocab


def encode(values: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    unk = vocab[UNK]
    return np.asarray([vocab.get(str(v), unk) for v in values.tolist()], dtype=np.int64)


@dataclass
class TruncationStats:
    n_patients: int
    n_over_max: int
    frac_over_max: float
    max_seq_len_raw: int
    n_signal_events: int
    n_signal_events_kept: int
    frac_signal_events_lost: float
    n_signal_a: int
    n_signal_a_lost: int
    n_signal_b: int
    n_signal_b_lost: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_patients": self.n_patients,
            "n_over_max": self.n_over_max,
            "frac_over_max": self.frac_over_max,
            "max_seq_len_raw": self.max_seq_len_raw,
            "n_signal_events": self.n_signal_events,
            "n_signal_events_kept": self.n_signal_events_kept,
            "frac_signal_events_lost": self.frac_signal_events_lost,
            "n_signal_a": self.n_signal_a,
            "n_signal_a_lost": self.n_signal_a_lost,
            "n_signal_b": self.n_signal_b,
            "n_signal_b_lost": self.n_signal_b_lost,
        }


class PatientSequenceDataset(Dataset):
    """Generic padded sequence dataset. Not Synthea-specific beyond construction."""

    def __init__(
        self,
        code_ids: list[np.ndarray],
        type_ids: list[np.ndarray],
        time_norm: list[np.ndarray],
        age_event_norm: list[np.ndarray],
        index_age_norm: np.ndarray,
        labels: np.ndarray,
        age_groups: list[str],
        patient_ids: list[str],
    ) -> None:
        self.code_ids = code_ids
        self.type_ids = type_ids
        self.time_norm = time_norm
        self.age_event_norm = age_event_norm
        self.index_age_norm = index_age_norm
        self.labels = labels
        self.age_groups = age_groups
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "code_ids": torch.from_numpy(self.code_ids[idx]),
            "type_ids": torch.from_numpy(self.type_ids[idx]),
            "time_norm": torch.from_numpy(self.time_norm[idx]),
            "age_event_norm": torch.from_numpy(self.age_event_norm[idx]),
            "index_age_norm": torch.tensor(self.index_age_norm[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "age_group": self.age_groups[idx],
            "patient_id": self.patient_ids[idx],
        }


def collate_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _pad(key: str, pad_value: float | int) -> torch.Tensor:
        seqs = [r[key] for r in rows]
        return pad_sequence(seqs, batch_first=True, padding_value=pad_value)

    code_ids = _pad("code_ids", 0)
    padding_mask = code_ids == 0
    return {
        "code_ids": code_ids,
        "type_ids": _pad("type_ids", 0),
        "time_norm": _pad("time_norm", 0.0),
        "age_event_norm": _pad("age_event_norm", 0.0),
        "padding_mask": padding_mask,
        "index_age_norm": torch.stack([r["index_age_norm"] for r in rows]),
        "labels": torch.stack([r["label"] for r in rows]),
        "age_group": [r["age_group"] for r in rows],
        "patient_id": [r["patient_id"] for r in rows],
    }


class SyntheaBenchmark:
    """Loads the sep1-exp processed tables and builds per-split sequence datasets."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        data_dir = cfg.data_path
        patients_path = data_dir / "patients.parquet"
        events_path = data_dir / "events.parquet"
        if not patients_path.exists() or not events_path.exists():
            raise FileNotFoundError(
                f"Missing benchmark tables in {data_dir}. "
                "Expected patients.parquet and events.parquet."
            )
        self.patients = pd.read_parquet(patients_path)
        events = pd.read_parquet(events_path)

        pre = events["time_before_index_days"] > 0
        self.n_dropped_non_preindex = int((~pre).sum())
        events = events.loc[pre].copy()
        events = events.sort_values(["patient_id", "event_timestamp"], kind="mergesort")

        train_ids = set(self.patients.loc[self.patients["split"] == "train", "patient_id"])
        train_events = events.loc[events["patient_id"].isin(train_ids)]
        self.code_vocab = build_vocab(train_events["event_code"])
        self.type_vocab = build_vocab(train_events["event_type"])

        events["code_id"] = encode(events["event_code"], self.code_vocab)
        events["type_id"] = encode(events["event_type"], self.type_vocab)
        events["time_n"] = time_norm_days(
            events["time_before_index_days"].to_numpy(), cfg.age_scale_years
        )
        events["age_n"] = age_norm_years(
            events["age_at_event"].to_numpy(), cfg.age_scale_years
        )

        self.truncation = self._truncate_and_pack(events)
        self.split_counts = {
            s: int((self.patients["split"] == s).sum()) for s in ("train", "val", "test")
        }

    def _truncate_and_pack(self, events: pd.DataFrame) -> TruncationStats:
        max_len = self.cfg.max_seq_len
        self._rows: dict[str, dict[str, np.ndarray]] = {}
        n_over = 0
        max_raw = 0
        n_sig = n_sig_kept = 0
        n_a = n_a_lost = 0
        n_b = n_b_lost = 0

        for pid, g in events.groupby("patient_id", sort=False):
            raw_len = len(g)
            max_raw = max(max_raw, raw_len)
            if raw_len > max_len:
                n_over += 1
            codes_raw = g["event_code"].to_numpy()
            is_a = codes_raw == "SIGNAL_A"
            is_b = codes_raw == "SIGNAL_B"
            n_a_here = int(is_a.sum())
            n_b_here = int(is_b.sum())
            n_a += n_a_here
            n_b += n_b_here
            n_sig += n_a_here + n_b_here

            kept = g.tail(max_len)
            kept_codes = kept["event_code"].to_numpy()
            kept_a = int((kept_codes == "SIGNAL_A").sum())
            kept_b = int((kept_codes == "SIGNAL_B").sum())
            n_sig_kept += kept_a + kept_b
            n_a_lost += n_a_here - kept_a
            n_b_lost += n_b_here - kept_b

            self._rows[str(pid)] = {
                "code_ids": kept["code_id"].to_numpy(dtype=np.int64),
                "type_ids": kept["type_id"].to_numpy(dtype=np.int64),
                "time_norm": kept["time_n"].to_numpy(dtype=np.float32),
                "age_event_norm": kept["age_n"].to_numpy(dtype=np.float32),
            }

        n_patients = len(self._rows)
        n_sig_lost = n_sig - n_sig_kept
        return TruncationStats(
            n_patients=n_patients,
            n_over_max=n_over,
            frac_over_max=n_over / n_patients if n_patients else 0.0,
            max_seq_len_raw=max_raw,
            n_signal_events=n_sig,
            n_signal_events_kept=n_sig_kept,
            frac_signal_events_lost=n_sig_lost / n_sig if n_sig else 0.0,
            n_signal_a=n_a,
            n_signal_a_lost=n_a_lost,
            n_signal_b=n_b,
            n_signal_b_lost=n_b_lost,
        )

    def make_dataset(self, split: str, task: str) -> PatientSequenceDataset:
        label_col = LABEL_COL[task]
        sub = self.patients.loc[self.patients["split"] == split].copy()
        code_ids: list[np.ndarray] = []
        type_ids: list[np.ndarray] = []
        time_norm: list[np.ndarray] = []
        age_event: list[np.ndarray] = []
        index_age: list[float] = []
        labels: list[float] = []
        groups: list[str] = []
        pids: list[str] = []
        age_scale = self.cfg.age_scale_years
        for row in sub.itertuples(index=False):
            pid = str(row.patient_id)
            packed = self._rows[pid]
            code_ids.append(packed["code_ids"])
            type_ids.append(packed["type_ids"])
            time_norm.append(packed["time_norm"])
            age_event.append(packed["age_event_norm"])
            index_age.append(float(np.clip(row.age_at_index / age_scale, 0.0, 1.0)))
            labels.append(float(getattr(row, label_col)))
            group = str(row.developmental_age_group)
            groups.append(group if group in AGE_GROUPS else group)
            pids.append(pid)
        return PatientSequenceDataset(
            code_ids=code_ids,
            type_ids=type_ids,
            time_norm=time_norm,
            age_event_norm=age_event,
            index_age_norm=np.asarray(index_age, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.float32),
            age_groups=groups,
            patient_ids=pids,
        )

    def make_loader(self, split: str, task: str, shuffle: bool) -> DataLoader:
        ds = self.make_dataset(split, task)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_batch,
            drop_last=False,
        )
