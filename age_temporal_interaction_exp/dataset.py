"""Sequence dataset for the age × temporal interaction experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import (
    AGE_GROUPS,
    LABEL_COL,
    NEG_CODE,
    PAD,
    POS_CODE,
    QUERY_CODE,
    UNK,
    Config,
    tau_from_days,
)


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
class PackedPatient:
    code_ids: np.ndarray
    type_ids: np.ndarray
    days_before: np.ndarray
    time_norm: np.ndarray
    is_query: np.ndarray
    is_signal: np.ndarray
    polarity: np.ndarray
    z_age: float
    age_years: float
    age_group: str
    patient_id: str
    labels: dict[str, float]
    true_w: dict[str, np.ndarray]


def _signal_only(row: PackedPatient) -> PackedPatient:
    keep = row.is_query | row.is_signal
    return PackedPatient(
        code_ids=row.code_ids[keep],
        type_ids=row.type_ids[keep],
        days_before=row.days_before[keep],
        time_norm=row.time_norm[keep],
        is_query=row.is_query[keep],
        is_signal=row.is_signal[keep],
        polarity=row.polarity[keep],
        z_age=row.z_age,
        age_years=row.age_years,
        age_group=row.age_group,
        patient_id=row.patient_id,
        labels=row.labels,
        true_w={k: v[keep] for k, v in row.true_w.items()},
    )


class InteractionDataset(Dataset):
    def __init__(self, rows: list[PackedPatient], task: str) -> None:
        self.rows = rows
        self.task = task

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        return {
            "code_ids": torch.from_numpy(r.code_ids),
            "type_ids": torch.from_numpy(r.type_ids),
            "days_before": torch.from_numpy(r.days_before),
            "time_norm": torch.from_numpy(r.time_norm),
            "is_query": torch.from_numpy(r.is_query),
            "is_signal": torch.from_numpy(r.is_signal),
            "polarity": torch.from_numpy(r.polarity),
            "z_age": torch.tensor(r.z_age, dtype=torch.float32),
            "age_years": torch.tensor(r.age_years, dtype=torch.float32),
            "label": torch.tensor(r.labels[self.task], dtype=torch.float32),
            "true_w": torch.from_numpy(r.true_w[self.task]),
            "age_group": r.age_group,
            "patient_id": r.patient_id,
        }


def collate_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _pad(key: str, value) -> torch.Tensor:
        return pad_sequence([r[key] for r in rows], batch_first=True, padding_value=value)

    code_ids = _pad("code_ids", 0)
    padding_mask = code_ids == 0
    return {
        "code_ids": code_ids,
        "type_ids": _pad("type_ids", 0),
        "days_before": _pad("days_before", 0.0),
        "time_norm": _pad("time_norm", 0.0),
        "is_query": _pad("is_query", False),
        "is_signal": _pad("is_signal", False),
        "polarity": _pad("polarity", 0),
        "true_w": _pad("true_w", 0.0),
        "padding_mask": padding_mask,
        "z_age": torch.stack([r["z_age"] for r in rows]),
        "age_years": torch.stack([r["age_years"] for r in rows]),
        "labels": torch.stack([r["label"] for r in rows]),
        "age_group": [r["age_group"] for r in rows],
        "patient_id": [r["patient_id"] for r in rows],
    }


class InteractionBenchmark:
    def __init__(self, cfg: Config, data_dir: Path | None = None) -> None:
        self.cfg = cfg
        data_dir = Path(data_dir) if data_dir is not None else cfg.out_path / "data"
        patients = pd.read_parquet(data_dir / "patients.parquet")
        events = pd.read_parquet(data_dir / "events.parquet")
        signals = pd.read_parquet(data_dir / "signals.parquet")
        self.patients = patients
        self.signals = signals
        self.age_mean = float(patients["age_mean_train"].iloc[0])
        self.age_std = float(patients["age_std_train"].iloc[0])

        events = events.sort_values(["patient_id", "event_timestamp"], kind="mergesort")
        train_ids = set(patients.loc[patients["split"] == "train", "patient_id"])
        train_events = events.loc[events["patient_id"].isin(train_ids)]
        self.code_vocab = build_vocab(train_events["event_code"])
        self.type_vocab = build_vocab(train_events["event_type"])
        assert QUERY_CODE in self.code_vocab
        assert POS_CODE in self.code_vocab
        assert NEG_CODE in self.code_vocab

        events = events.copy()
        events["code_id"] = encode(events["event_code"], self.code_vocab)
        events["type_id"] = encode(events["event_type"], self.type_vocab)
        events["time_n"] = tau_from_days(events["time_before_index_days"].to_numpy()).astype(
            np.float32
        )

        w_lookup = {
            pid: g.sort_values("signal_idx")
            for pid, g in signals.groupby("patient_id", sort=False)
        }
        self._rows: dict[str, PackedPatient] = {}
        self.truncation = self._pack(events, patients, w_lookup)

    def _pack(
        self,
        events: pd.DataFrame,
        patients: pd.DataFrame,
        w_lookup: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        max_len = self.cfg.max_seq_len
        n_over = 0
        n_sig_lost = 0
        n_sig = 0
        max_raw = 0
        meta = patients.set_index("patient_id")
        for pid, g in events.groupby("patient_id", sort=False):
            pid = str(pid)
            raw_len = len(g)
            max_raw = max(max_raw, raw_len)
            inj = g["source"] == "age_temporal_injected"
            bg = g.loc[~inj]
            kept_inj = g.loc[inj]
            budget = max(max_len - len(kept_inj), 0)
            if len(bg) > budget:
                n_over += 1
                kept_bg = bg.tail(budget)
            else:
                kept_bg = bg
            kept = pd.concat([kept_bg, kept_inj], axis=0)
            kept = kept.sort_values("event_timestamp", kind="mergesort")
            if len(kept) > max_len:
                extra = len(kept) - max_len
                drop_idx = kept.index[~kept["source"].eq("age_temporal_injected")][:extra]
                kept = kept.drop(index=drop_idx)

            is_signal = kept["is_signal"].to_numpy(dtype=bool)
            n_sig += int(is_signal.sum())
            n_here = int((g["is_signal"] == True).sum())  # noqa: E712
            n_sig_lost += n_here - int(is_signal.sum())

            true_w = {"T0": np.zeros(len(kept), dtype=np.float32),
                      "T1": np.zeros(len(kept), dtype=np.float32),
                      "T2": np.zeros(len(kept), dtype=np.float32)}
            if pid in w_lookup:
                sg = w_lookup[pid]
                sig_pos = np.flatnonzero(is_signal)
                if len(sig_pos) == len(sg):
                    for task in ("T0", "T1", "T2"):
                        true_w[task][sig_pos] = sg[f"w_{task}"].to_numpy(dtype=np.float32)

            row = meta.loc[pid]
            group = str(row["developmental_age_group"])
            self._rows[pid] = PackedPatient(
                code_ids=kept["code_id"].to_numpy(dtype=np.int64),
                type_ids=kept["type_id"].to_numpy(dtype=np.int64),
                days_before=kept["time_before_index_days"].to_numpy(dtype=np.float32),
                time_norm=kept["time_n"].to_numpy(dtype=np.float32),
                is_query=kept["is_query"].to_numpy(dtype=bool),
                is_signal=is_signal,
                polarity=kept["polarity"].to_numpy(dtype=np.int64),
                z_age=float(row["z_age"]),
                age_years=float(row["age_at_index"]),
                age_group=group if group in AGE_GROUPS else group,
                patient_id=pid,
                labels={t: float(row[LABEL_COL[t]]) for t in LABEL_COL},
                true_w=true_w,
            )
        n_patients = len(self._rows)
        return {
            "n_patients": n_patients,
            "n_over_max": n_over,
            "max_seq_len_raw": max_raw,
            "n_signal_events": n_sig + n_sig_lost,
            "n_signal_events_kept": n_sig,
            "frac_signal_lost": n_sig_lost / max(n_sig + n_sig_lost, 1),
        }

    def make_dataset(
        self,
        split: str,
        task: str,
        max_examples: int | None = None,
        seed: int = 0,
        signal_only: bool = False,
    ) -> InteractionDataset:
        sub = self.patients.loc[self.patients["split"] == split]
        pids = [str(p) for p in sub["patient_id"].tolist() if str(p) in self._rows]
        if max_examples is not None and max_examples < len(pids):
            rng = np.random.default_rng(seed)
            rng.shuffle(pids)
            pids = pids[:max_examples]
        rows = [self._rows[p] for p in pids]
        if signal_only:
            rows = [_signal_only(r) for r in rows]
        return InteractionDataset(rows, task)

    def make_loader(
        self,
        split: str,
        task: str,
        shuffle: bool,
        max_examples: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        signal_only: bool = False,
    ) -> DataLoader:
        ds = self.make_dataset(
            split, task, max_examples=max_examples, seed=seed, signal_only=signal_only
        )
        return DataLoader(
            ds,
            batch_size=batch_size or self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_batch,
            drop_last=False,
        )
