"""PyTorch dataset and collate for the synthetic age × temporal benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    FORBIDDEN_MODEL_KEYS,
    MAX_BACKGROUND_EVENTS,
    MAX_SEQ_LEN,
    PAD,
    QUERY_CODE,
    QUERY_TYPE,
    UNK,
    all_signal_codes,
)


class Vocab:
    def __init__(self) -> None:
        self.stoi = {PAD: 0, UNK: 1, QUERY_CODE: 2}
        for c in all_signal_codes():
            self.stoi[c] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

    def add(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.stoi)
            self.itos[self.stoi[token]] = token
        return self.stoi[token]

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[UNK])

    def __len__(self) -> int:
        return len(self.stoi)


TYPE_STOI = {
    PAD: 0,
    UNK: 1,
    QUERY_TYPE: 2,
    "signal": 3,
    "condition": 4,
    "encounter": 5,
    "medication": 6,
    "procedure": 7,
    "observation": 8,
    "immunization": 9,
    "background": 10,
}


def encode_type(t: str) -> int:
    return TYPE_STOI.get(t, TYPE_STOI[UNK])


def apply_model_truncation(
    codes: list,
    types: list,
    lags: list,
    taus: list | None,
    *,
    max_seq_len: int = MAX_SEQ_LEN,
    max_background: int = MAX_BACKGROUND_EVENTS,
) -> dict[str, Any]:
    """Exact truncation used by BenchmarkDataset (shared with visibility audit).

    1. Keep all signal events; subsample background to ``max_background`` newest.
    2. If still over ``max_seq_len - 1``, drop oldest events (long-lag signals can die here).
    3. Append prediction-time query token.
    """
    from config import tau_from_days

    codes = list(codes)
    types = list(types)
    lags = [float(x) for x in lags]
    if taus is None or (isinstance(taus, float) and np.isnan(taus)):
        taus = tau_from_days(lags).tolist()
    else:
        taus = [float(x) for x in list(taus)]

    n_before = len(codes)
    n_sig_before = sum(1 for t in types if t == "signal")
    sig_idx = [i for i, t in enumerate(types) if t == "signal"]
    bg_idx = [i for i, t in enumerate(types) if t != "signal"]
    if len(bg_idx) > max_background:
        bg_idx = bg_idx[-max_background:]
    keep = sorted(bg_idx + sig_idx)
    codes = [codes[i] for i in keep]
    types = [types[i] for i in keep]
    lags = [lags[i] for i in keep]
    taus = [taus[i] for i in keep]

    budget = max_seq_len - 1
    truncated = False
    if len(codes) > budget:
        truncated = True
        codes, types, lags, taus = (
            codes[-budget:],
            types[-budget:],
            lags[-budget:],
            taus[-budget:],
        )
    n_sig_after = sum(1 for t in types if t == "signal")
    retained_signal_lags = [lags[i] for i, t in enumerate(types) if t == "signal"]
    retained_signal_codes = [codes[i] for i, t in enumerate(types) if t == "signal"]

    codes = codes + [QUERY_CODE]
    types = types + [QUERY_TYPE]
    lags = lags + [0.0]
    taus = taus + [0.0]
    is_query = [0] * (len(codes) - 1) + [1]
    is_signal = [1 if t == "signal" else 0 for t in types]
    return {
        "codes": codes,
        "types": types,
        "lags": lags,
        "taus": taus,
        "is_query": is_query,
        "is_signal": is_signal,
        "n_events_before": n_before,
        "n_signal_before": n_sig_before,
        "n_signal_after": n_sig_after,
        "truncated": truncated,
        "retained_signal_lags": retained_signal_lags,
        "retained_signal_codes": retained_signal_codes,
        "seq_len_history": len(codes) - 1,
    }


class BenchmarkDataset(Dataset):
    """Model-visible tensors only — ground-truth mechanism fields are excluded."""

    def __init__(
        self,
        examples: pd.DataFrame,
        labels: np.ndarray,
        split: str,
        vocab: Vocab,
        max_seq_len: int = MAX_SEQ_LEN,
        max_background: int = MAX_BACKGROUND_EVENTS,
        target_idx: list[int] | None = None,
    ) -> None:
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_background = max_background
        self.target_idx = target_idx
        mask = examples["split"].to_numpy() == split
        self.examples = examples.loc[mask].reset_index(drop=True)
        self.labels = labels[mask]
        if target_idx is not None:
            self.labels = self.labels[:, target_idx]
        self.split = split

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.examples.iloc[idx]
        trunc = apply_model_truncation(
            list(row["history_codes"]),
            list(row["history_types"]),
            list(row["history_lag_days"]),
            list(row["history_tau"]) if isinstance(row["history_tau"], list) else None,
            max_seq_len=self.max_seq_len,
            max_background=self.max_background,
        )
        codes = trunc["codes"]
        types = trunc["types"]
        lags = trunc["lags"]
        taus = trunc["taus"]
        is_query = trunc["is_query"]
        is_signal = trunc["is_signal"]

        item = {
            "code_ids": [self.vocab.encode(c) for c in codes],
            "type_ids": [encode_type(t) for t in types],
            "lag_days": lags,
            "tau": taus,
            "is_query": is_query,
            "is_signal": is_signal,
            "age": float(row["age_at_cutoff"]),
            "z_age": float(row["z_age"]),
            "labels": self.labels[idx].astype(np.float32),
            "patient_id": str(row["patient_id"]),
            "example_id": int(row["example_id"]),
        }
        # Hard guarantee: no forbidden keys.
        for k in FORBIDDEN_MODEL_KEYS:
            assert k not in item
        return item


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    B = len(batch)
    L = max(len(b["code_ids"]) for b in batch)
    code = torch.zeros(B, L, dtype=torch.long)
    typ = torch.zeros(B, L, dtype=torch.long)
    lag = torch.zeros(B, L, dtype=torch.float32)
    tau = torch.zeros(B, L, dtype=torch.float32)
    is_q = torch.zeros(B, L, dtype=torch.bool)
    is_sig = torch.zeros(B, L, dtype=torch.bool)
    pad = torch.ones(B, L, dtype=torch.bool)
    age = torch.zeros(B, dtype=torch.float32)
    z = torch.zeros(B, dtype=torch.float32)
    y = torch.stack([torch.tensor(b["labels"], dtype=torch.float32) for b in batch])

    for i, b in enumerate(batch):
        n = len(b["code_ids"])
        code[i, :n] = torch.tensor(b["code_ids"], dtype=torch.long)
        typ[i, :n] = torch.tensor(b["type_ids"], dtype=torch.long)
        lag[i, :n] = torch.tensor(b["lag_days"], dtype=torch.float32)
        tau[i, :n] = torch.tensor(b["tau"], dtype=torch.float32)
        is_q[i, :n] = torch.tensor(b["is_query"], dtype=torch.bool)
        is_sig[i, :n] = torch.tensor(b["is_signal"], dtype=torch.bool)
        pad[i, :n] = False
        age[i] = b["age"]
        z[i] = b["z_age"]

    out = {
        "code_ids": code,
        "type_ids": typ,
        "lag_days": lag,
        "tau": tau,
        "is_query": is_q,
        "is_signal": is_sig,
        "padding_mask": pad,
        "age": age,
        "z_age": z,
        "labels": y,
    }
    for k in FORBIDDEN_MODEL_KEYS:
        assert k not in out
    return out


def build_vocab(examples: pd.DataFrame) -> Vocab:
    vocab = Vocab()
    for codes in examples["history_codes"]:
        for c in codes:
            vocab.add(str(c))
    return vocab


def load_scenario_dir(scenario_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict, list]:
    examples = pd.read_parquet(scenario_dir / "examples.parquet")
    labels = np.load(scenario_dir / "labels.npz")["Y"]
    with (scenario_dir / "meta.json").open() as f:
        meta = json.load(f)
    with (scenario_dir / "target_specs.json").open() as f:
        specs = json.load(f)
    return examples, labels, meta, specs


def make_loaders(
    scenario_dir: Path,
    batch_size: int,
    max_seq_len: int = MAX_SEQ_LEN,
    num_workers: int = 0,
    target_idx: list[int] | None = None,
    max_background: int = MAX_BACKGROUND_EVENTS,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocab, dict]:
    examples, labels, meta, specs = load_scenario_dir(scenario_dir)
    vocab = build_vocab(examples)
    n_targets = labels.shape[1] if target_idx is None else len(target_idx)
    loaders = {}
    for split in ("train", "val", "test"):
        ds = BenchmarkDataset(
            examples,
            labels,
            split,
            vocab,
            max_seq_len=max_seq_len,
            max_background=max_background,
            target_idx=target_idx,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_batch,
        )
    return loaders["train"], loaders["val"], loaders["test"], vocab, {
        "meta": meta,
        "specs": specs,
        "n_codes": len(vocab),
        "n_types": max(TYPE_STOI.values()) + 1,
        "n_targets": n_targets,
        "target_idx": target_idx,
    }
