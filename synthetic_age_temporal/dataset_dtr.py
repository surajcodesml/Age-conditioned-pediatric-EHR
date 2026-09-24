"""Encounter-level dataset / loaders for Developmental Temporal Retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import FORBIDDEN_MODEL_KEYS, MAX_BACKGROUND_EVENTS, MAX_SEQ_LEN
from dataset import Vocab, build_vocab, load_scenario_dir
from encounters import truncate_and_group

MAX_ENCOUNTERS = 64
MAX_CODES_PER_ENCOUNTER = 32


class DTRDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        labels: np.ndarray,
        split: str,
        vocab: Vocab,
        max_seq_len: int = MAX_SEQ_LEN,
        max_background: int = MAX_BACKGROUND_EVENTS,
        max_encounters: int = MAX_ENCOUNTERS,
        max_codes_per_encounter: int = MAX_CODES_PER_ENCOUNTER,
        target_idx: list[int] | None = None,
    ) -> None:
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_background = max_background
        self.max_encounters = max_encounters
        self.max_codes = max_codes_per_encounter
        self.target_idx = target_idx
        mask = examples["split"].to_numpy() == split
        self.examples = examples.loc[mask].reset_index(drop=True)
        self.labels = labels[mask]
        if target_idx is not None:
            self.labels = self.labels[:, target_idx]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples.iloc[idx]
        encs = truncate_and_group(
            list(row["history_codes"]),
            list(row["history_types"]),
            list(row["history_lag_days"]),
            list(row["history_tau"]) if isinstance(row["history_tau"], list) else None,
            max_seq_len=self.max_seq_len,
            max_background=self.max_background,
            max_encounters=self.max_encounters,
        )
        enc_codes: list[list[int]] = []
        enc_lags: list[float] = []
        enc_taus: list[float] = []
        enc_n_sig: list[int] = []
        for e in encs:
            ids = [self.vocab.encode(c) for c in e.codes[: self.max_codes]]
            enc_codes.append(ids)
            enc_lags.append(e.lag_days)
            enc_taus.append(e.tau)
            enc_n_sig.append(e.n_signal)
        item = {
            "enc_codes": enc_codes,
            "enc_lag_days": enc_lags,
            "enc_tau": enc_taus,
            "enc_n_signal": enc_n_sig,
            "n_encounters": len(encs),
            "age": float(row["age_at_cutoff"]),
            "z_age": float(row["z_age"]),
            "labels": self.labels[idx].astype(np.float32),
            "patient_id": str(row["patient_id"]),
            "example_id": int(row["example_id"]),
        }
        for k in FORBIDDEN_MODEL_KEYS:
            assert k not in item
        return item


def collate_dtr(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    B = len(batch)
    M = max((b["n_encounters"] for b in batch), default=1)
    M = max(M, 1)
    K = max(
        (max((len(c) for c in b["enc_codes"]), default=1) for b in batch),
        default=1,
    )
    K = max(K, 1)
    code = torch.zeros(B, M, K, dtype=torch.long)
    code_mask = torch.zeros(B, M, K, dtype=torch.bool)
    tau = torch.zeros(B, M, dtype=torch.float32)
    lag = torch.zeros(B, M, dtype=torch.float32)
    pad = torch.ones(B, M, dtype=torch.bool)
    n_sig = torch.zeros(B, M, dtype=torch.long)
    age = torch.zeros(B, dtype=torch.float32)
    z = torch.zeros(B, dtype=torch.float32)
    y = torch.stack([torch.tensor(b["labels"], dtype=torch.float32) for b in batch])
    patient_ids = [b["patient_id"] for b in batch]
    example_ids = torch.tensor([b["example_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        for j, ids in enumerate(b["enc_codes"]):
            n = len(ids)
            if n == 0:
                continue
            code[i, j, :n] = torch.tensor(ids, dtype=torch.long)
            code_mask[i, j, :n] = True
            tau[i, j] = b["enc_tau"][j]
            lag[i, j] = b["enc_lag_days"][j]
            n_sig[i, j] = b["enc_n_signal"][j]
            pad[i, j] = False
        age[i] = b["age"]
        z[i] = b["z_age"]

    out = {
        "enc_code_ids": code,
        "enc_code_mask": code_mask,
        "enc_tau": tau,
        "enc_lag_days": lag,
        "enc_padding_mask": pad,
        "enc_n_signal": n_sig,
        "age": age,
        "z_age": z,
        "labels": y,
        "example_ids": example_ids,
        "patient_ids": patient_ids,
    }
    for k in FORBIDDEN_MODEL_KEYS:
        assert k not in out
    return out


def make_dtr_loaders(
    scenario_dir: Path,
    batch_size: int,
    max_seq_len: int = MAX_SEQ_LEN,
    max_background: int = MAX_BACKGROUND_EVENTS,
    max_encounters: int = MAX_ENCOUNTERS,
    target_idx: list[int] | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab, dict]:
    examples, labels, meta, specs = load_scenario_dir(scenario_dir)
    vocab = build_vocab(examples)
    n_targets = labels.shape[1] if target_idx is None else len(target_idx)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = DTRDataset(
            examples,
            labels,
            split,
            vocab,
            max_seq_len=max_seq_len,
            max_background=max_background,
            max_encounters=max_encounters,
            target_idx=target_idx,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_dtr,
        )
    return loaders["train"], loaders["val"], loaders["test"], vocab, {
        "meta": meta,
        "specs": specs,
        "n_codes": len(vocab),
        "n_targets": n_targets,
        "target_idx": target_idx,
        "examples": examples,
        "labels": labels,
    }
