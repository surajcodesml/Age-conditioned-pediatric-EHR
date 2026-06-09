#!/usr/bin/env python3
"""Dataset and collate for TALE-EHR binary disease classification fine-tuning."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def encode_race(race_val: Any) -> int:
    """Map MIMIC-IV race string to 7 buckets (prefix/literal rules)."""
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


class DiseaseClassificationDataset(Dataset):
    """Patient-level disease classification samples ending at cohort-specified index."""

    def __init__(
        self,
        cohort_parquet: str | Path,
        events_parquet: str | Path,
        code_vocab_path: str | Path,
        max_seq_len: int = 1024,
    ) -> None:
        self.cohort_parquet = Path(cohort_parquet)
        self.events_parquet = Path(events_parquet)
        self.code_vocab_path = Path(code_vocab_path)
        self.max_seq_len = int(max_seq_len)

        if not self.cohort_parquet.exists():
            raise FileNotFoundError(f"Missing cohort parquet: {self.cohort_parquet}")
        if not self.events_parquet.exists():
            raise FileNotFoundError(f"Missing events parquet: {self.events_parquet}")
        if not self.code_vocab_path.exists():
            raise FileNotFoundError(f"Missing code vocab: {self.code_vocab_path}")

        with self.code_vocab_path.open("r", encoding="utf-8") as f:
            self.code_vocab: dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}
        self.num_codes = len(self.code_vocab)
        self.unk_vocab_index = self.num_codes

        self._events_sql = _esc(self.events_parquet)
        self._rows = self._load_cohort_rows()
        self._con: duckdb.DuckDBPyConnection | None = None

    def _load_cohort_rows(self) -> list[dict[str, int]]:
        con = duckdb.connect()
        try:
            schema_cur = con.execute(
                "SELECT * FROM read_parquet(?) LIMIT 0",
                [_esc(self.cohort_parquet)],
            )
            available_cols = {str(col_meta[0]).lower() for col_meta in (schema_cur.description or [])}
            hadm_expr = "CAST(hadm_id AS BIGINT) AS hadm_id" if "hadm_id" in available_cols else "CAST(-1 AS BIGINT) AS hadm_id"
            n_events_expr = (
                "CAST(n_events_in_window AS BIGINT) AS n_events_in_window"
                if "n_events_in_window" in available_cols
                else "CAST(0 AS BIGINT) AS n_events_in_window"
            )
            cols = con.execute(
                f"""
                SELECT
                    CAST(subject_id AS BIGINT) AS subject_id,
                    {hadm_expr},
                    CAST(label AS INTEGER) AS label,
                    CAST(last_event_idx AS BIGINT) AS last_event_idx,
                    {n_events_expr}
                FROM read_parquet(?)
                ORDER BY subject_id, hadm_id
                """,
                [_esc(self.cohort_parquet)],
            ).fetchnumpy()
        finally:
            con.close()

        out: list[dict[str, int]] = []
        for i in range(len(cols["subject_id"])):
            out.append(
                {
                    "subject_id": int(cols["subject_id"][i]),
                    "hadm_id": int(cols["hadm_id"][i]),
                    "label": int(cols["label"][i]),
                    "last_event_idx": int(cols["last_event_idx"][i]),
                    "n_events_in_window": int(cols["n_events_in_window"][i]),
                }
            )
        return out

    def __len__(self) -> int:
        return len(self._rows)

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect()
            # Keep per-worker DuckDB memory bounded so larger worker counts fit in 32GB RAM.
            self._con.execute("PRAGMA memory_limit='768MB'")
            self._con.execute("PRAGMA threads=1")
        return self._con

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]
        subject_id = int(row["subject_id"])
        label = int(row["label"])
        last_event_idx = int(row["last_event_idx"])

        con = self._get_connection()
        cols = con.execute(
            f"""
            SELECT
                code_id,
                timestamp_days,
                age_at_event_days,
                sex,
                race
            FROM read_parquet('{self._events_sql}')
            WHERE subject_id = ?
            ORDER BY timestamp_days, event_time, code_id
            """,
            [subject_id],
        ).fetchnumpy()

        n = len(cols["code_id"])
        if n == 0:
            raise RuntimeError(
                f"Empty context for subject_id={subject_id}, idx={idx}, last_event_idx={last_event_idx}"
            )

        trunc_n = min(last_event_idx + 1, n)
        code_ids = cols["code_id"][:trunc_n].astype(str)
        timestamps_days = cols["timestamp_days"][:trunc_n].astype(np.float32)
        age_days = cols["age_at_event_days"][:trunc_n].astype(np.float32)
        sex = int(cols["sex"][0]) if n > 0 else 0
        race = encode_race(cols["race"][0]) if n > 0 else 6

        if trunc_n > self.max_seq_len:
            sl = slice(-self.max_seq_len, None)
            code_ids = code_ids[sl]
            timestamps_days = timestamps_days[sl]
            age_days = age_days[sl]

        code_indices = np.fromiter(
            (self.code_vocab.get(str(c), self.unk_vocab_index) for c in code_ids),
            dtype=np.int64,
            count=len(code_ids),
        )

        return {
            "subject_id": subject_id,
            "hadm_id": int(row["hadm_id"]),
            "code_indices": code_indices,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": sex,
            "race": race,
            "unk_vocab_index": self.unk_vocab_index,
            "n_events_in_window": int(row["n_events_in_window"]),
            "label": float(label),
        }

    def __del__(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None


class TensorizedDiseaseClassificationDataset(Dataset):
    """Fast shard-backed disease classification dataset for multi-worker loading."""

    def __init__(
        self,
        tensorized_split_dir: str | Path,
        max_seq_len: int = 1024,
        shard_cache_size: int = 4,
    ) -> None:
        self.tensorized_split_dir = Path(tensorized_split_dir)
        self.max_seq_len = int(max_seq_len)
        self.shard_cache_size = int(shard_cache_size)

        if not self.tensorized_split_dir.exists():
            raise FileNotFoundError(f"Missing tensorized split dir: {self.tensorized_split_dir}")
        self._shard_paths = sorted(self.tensorized_split_dir.glob("shard_*.npz"))
        if not self._shard_paths:
            raise FileNotFoundError(f"No shard_*.npz files in {self.tensorized_split_dir}")

        self._index: list[tuple[int, int]] = []
        for shard_id, shard_path in enumerate(self._shard_paths):
            npz = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            n = int(len(npz["subject_id"]))
            npz.close()
            for pos in range(n):
                self._index.append((shard_id, pos))

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
        # Old object-array shards lack the flat `offsets` index; touching their
        # ragged arrays would raise a raw "Object arrays cannot be loaded" error
        # (allow_pickle=False), so detect the legacy layout up front.
        if "offsets" not in npz.files:
            raise RuntimeError(
                f"{self._shard_paths[shard_id]} is in the old object-array format; re-run tensorization."
            )
        d: dict[str, Any] = {
            "_npz": npz,
            "offsets": npz["offsets"],
            "code_indices": npz["code_indices"],
            "timestamps_days": npz["timestamps_days"],
            "age_days": npz["age_days"],
            "subject_id": npz["subject_id"],
            "hadm_id": npz["hadm_id"] if "hadm_id" in npz.files else None,
            "sex": npz["sex"],
            "race": npz["race"],
            "label": npz["label"],
            "unk_vocab_index": int(np.asarray(npz["unk_vocab_index"]).reshape(-1)[0]),
            "n_events_in_window": npz["n_events_in_window"] if "n_events_in_window" in npz.files else None,
        }
        self._shard_cache[shard_id] = d
        return d

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id, pos = self._index[idx]
        s = self._load_shard(shard_id)
        off = s["offsets"]
        start, end = int(off[pos]), int(off[pos + 1])
        code_indices = np.asarray(s["code_indices"][start:end], dtype=np.int64)
        timestamps_days = np.asarray(s["timestamps_days"][start:end], dtype=np.float32)
        age_days = np.asarray(s["age_days"][start:end], dtype=np.float32)
        if code_indices.shape[0] > self.max_seq_len:
            sl = slice(-self.max_seq_len, None)
            code_indices = code_indices[sl]
            timestamps_days = timestamps_days[sl]
            age_days = age_days[sl]

        return {
            "subject_id": int(s["subject_id"][pos]),
            "hadm_id": int(s["hadm_id"][pos]) if s["hadm_id"] is not None else -1,
            "code_indices": code_indices,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": int(s["sex"][pos]),
            "race": int(s["race"][pos]),
            "unk_vocab_index": int(s["unk_vocab_index"]),
            "n_events_in_window": int(s["n_events_in_window"][pos]) if s["n_events_in_window"] is not None else 0,
            "label": float(s["label"][pos]),
        }

    def __del__(self) -> None:
        for shard in self._shard_cache.values():
            try:
                npz = shard.get("_npz")
                if npz is not None:
                    npz.close()
            except Exception:
                pass
        self._shard_cache.clear()


def _dataloader_worker_init(_worker_id: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def disease_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad sequences and build pairwise temporal matrix for disease classification."""
    if not batch:
        raise ValueError("empty batch")

    bsz = len(batch)
    unk_id = int(batch[0]["unk_vocab_index"])
    max_len = max(int(item["code_indices"].shape[0]) for item in batch)

    code_np = np.full((bsz, max_len), -1, dtype=np.int64)
    ts_np = np.zeros((bsz, max_len), dtype=np.float32)
    age_np = np.zeros((bsz, max_len), dtype=np.float32)
    mask_np = np.zeros((bsz, max_len), dtype=bool)
    labels_np = np.array([item["label"] for item in batch], dtype=np.float32)
    sex_arr = np.array([item["sex"] for item in batch], dtype=np.float32)
    race_arr = np.array([item["race"] for item in batch], dtype=np.float32)

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

    # BGE-aligned row ids: PAD=0, UNK=1, known vocab ids shifted by +2
    bge_codes = torch.where(
        attention_mask,
        torch.where(
            code_indices == int(unk_id),
            torch.ones((), dtype=torch.long),
            code_indices + 2,
        ),
        torch.zeros((), dtype=torch.long),
    )

    t = timestamps_days
    delta_t = torch.log1p(torch.abs(t.unsqueeze(2) - t.unsqueeze(1)) / 7.0)
    pair_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    delta_t = delta_t * pair_mask.float()

    age_years = age_np / 365.25
    demo = np.zeros((bsz, max_len, 3), dtype=np.float32)
    demo[:, :, 0] = age_years
    demo[:, :, 1] = sex_arr[:, np.newaxis]
    demo[:, :, 2] = race_arr[:, np.newaxis]
    demo = demo * mask_np[:, :, np.newaxis].astype(np.float32)

    return {
        "code_indices": bge_codes,
        "timestamps_days": timestamps_days,
        "delta_t": delta_t,
        "attention_mask": attention_mask,
        "demographics": torch.from_numpy(demo),
        "labels": torch.from_numpy(labels_np),
        "subject_id": [int(item["subject_id"]) for item in batch],
        "hadm_id": [int(item["hadm_id"]) for item in batch],
        "n_events_in_window": [int(item["n_events_in_window"]) for item in batch],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test disease dataset + collate.")
    parser.add_argument("--cohort_parquet", type=Path, required=True)
    parser.add_argument(
        "--events_parquet",
        type=Path,
        default=Path("data/processed/patient_events_rolled_full.parquet"),
    )
    parser.add_argument("--code_vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    ds = DiseaseClassificationDataset(
        cohort_parquet=args.cohort_parquet,
        events_parquet=args.events_parquet,
        code_vocab_path=args.code_vocab_path,
        max_seq_len=1024,
    )
    sample = ds[0]
    print(
        {
            "subject_id": sample["subject_id"],
            "n_events": int(sample["code_indices"].shape[0]),
            "first_ts": float(sample["timestamps_days"][0]),
            "last_ts": float(sample["timestamps_days"][-1]),
            "label": int(sample["label"]),
        }
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=disease_collate, num_workers=0)
    batch = next(iter(loader))
    print({k: tuple(v.shape) for k, v in batch.items()})
