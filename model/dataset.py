#!/usr/bin/env python3
"""PyTorch Dataset and collate for TALE-EHR pretraining on MIMIC-IV event sequences."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import duckdb

LOGGER = logging.getLogger("ehr_dataset")


def _dataloader_worker_init(_worker_id: int) -> None:
    """Avoid oversubscription / teardown issues when DuckDB and PyTorch share worker processes."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _duckdb_escape_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def encode_race(race_val: Any) -> int:
    """Map MIMIC-IV race string to 7 buckets (prefix / literal rules)."""
    if race_val is None:
        return 6
    if isinstance(race_val, float) and np.isnan(race_val):
        return 6
    s = str(race_val).strip().upper()
    if not s or s == "NAN":
        return 6
    unknown_literals = {"UNKNOWN", "UNABLE TO OBTAIN", "PREFER NOT TO SAY", "N/A", "DECLINED"}
    if s in unknown_literals:
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


def encode_sex(val: Any) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        v = int(val)
    except (TypeError, ValueError):
        return 0
    return 1 if v == 1 else 0


class EHRDataset(Dataset):
    """Event-sequence samples for next-visit code and timing prediction.

    Phase 1 (``__init__``): builds a lightweight sample index with DuckDB over the parquet on disk
    (no full-table pandas load). Phase 2 (``__getitem__``): reads one patient's rows per access via
    DuckDB predicate pushdown.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        code_vocab_path: str | Path,
        max_seq_len: int = 1024,
        max_rows: int | None = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.code_vocab_path = Path(code_vocab_path)
        self.max_seq_len = int(max_seq_len)
        self.max_rows = max_rows

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet: {self.parquet_path}")
        if not self.code_vocab_path.exists():
            raise FileNotFoundError(f"Missing code vocab: {self.code_vocab_path}")

        with self.code_vocab_path.open("r", encoding="utf-8") as f:
            self.code_vocab: dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}
        self.num_codes = len(self.code_vocab)
        self.unk_vocab_index = int(self.num_codes)

        self._parquet_sql = _duckdb_escape_path(self.parquet_path)

        t0 = time.perf_counter()
        self._samples: list[dict[str, Any]] = self._build_sample_index_duckdb()
        elapsed = time.perf_counter() - t0

        n_patients = len({s["subject_id"] for s in self._samples})
        LOGGER.info(
            "EHRDataset: %d patients, %d (patient, visit) samples from %s (index build %.1fs)",
            n_patients,
            len(self._samples),
            self.parquet_path,
            elapsed,
        )

    def _build_sample_index_duckdb(self) -> list[dict[str, Any]]:
        """Build sample index using robust hadm-slot visits."""
        limit_clause = ""
        if self.max_rows is not None and self.max_rows > 0:
            limit_clause = f"LIMIT {int(self.max_rows)}"

        sql = f"""
        WITH events AS (
            SELECT subject_id, hadm_id, timestamp_days, code_id
            FROM read_parquet('{self._parquet_sql}')
            {limit_clause}
        ),
        visit_slots AS (
            SELECT
                subject_id,
                COALESCE(TRY_CAST(hadm_id AS BIGINT), CAST(-1 AS BIGINT)) AS hadm_fill,
                MIN(timestamp_days) AS first_ts
            FROM events
            GROUP BY subject_id, hadm_fill
        ),
        ranked AS (
            SELECT
                subject_id,
                hadm_fill,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id
                    ORDER BY first_ts, hadm_fill
                ) AS visit_idx
            FROM visit_slots
        )
        SELECT
            subject_id,
            COUNT(*)::BIGINT AS n_visits
        FROM ranked
        GROUP BY subject_id
        ORDER BY subject_id
        """

        con = duckdb.connect()
        try:
            cols = con.execute(sql).fetchnumpy()
        finally:
            con.close()

        subject_ids = cols["subject_id"]
        n_visits = cols["n_visits"]
        n = len(subject_ids)
        samples: list[dict[str, Any]] = []

        for i in range(n):
            sid = int(subject_ids[i])
            visit_count = int(n_visits[i])
            if visit_count >= 2:
                for visit_k in range(visit_count - 1):
                    samples.append({"subject_id": sid, "visit_k": visit_k})

        return samples

    def _materialize_patient(self, subject_id: int) -> dict[str, Any]:
        sql = f"""
        SELECT code_id, hadm_id, timestamp_days, age_at_event_days, sex, race
        FROM read_parquet('{self._parquet_sql}')
        WHERE subject_id = ?
        ORDER BY timestamp_days, code_id
        """
        con = duckdb.connect()
        try:
            cols = con.execute(sql, [subject_id]).fetchnumpy()
        finally:
            con.close()

        g = pd.DataFrame(
            {
                "code_id": cols["code_id"],
                "hadm_id": cols["hadm_id"],
                "timestamp_days": cols["timestamp_days"],
                "age_at_event_days": cols["age_at_event_days"],
                "sex": cols["sex"],
                "race": cols["race"],
            }
        )
        n_events = len(g)
        if n_events == 0:
            raise RuntimeError(f"No events for subject_id={subject_id}")

        # Assign each event to a visit slot, ordered by the first timestamp seen for that hadm_id.
        g["_hadm_fill"] = g["hadm_id"].fillna(-1).astype("int64")
        first_ts_per_hadm = g.groupby("_hadm_fill")["timestamp_days"].min().sort_values(kind="mergesort")
        hadm_to_visit_idx = {h: k for k, h in enumerate(first_ts_per_hadm.index)}
        g["_visit_idx"] = g["_hadm_fill"].map(hadm_to_visit_idx).astype("int64")

        # Re-sort so events within each visit are in (timestamp_days, code_id) order and visits are contiguous.
        g = g.sort_values(["_visit_idx", "timestamp_days", "code_id"], kind="mergesort").reset_index(drop=True)

        n_events = len(g)
        visit_idx_arr = g["_visit_idx"].to_numpy()
        visit_spans: list[tuple[int, int]] = []
        start_i = 0
        for i in range(1, n_events):
            if visit_idx_arr[i] != visit_idx_arr[i - 1]:
                visit_spans.append((start_i, i))
                start_i = i
        visit_spans.append((start_i, n_events))

        assert visit_spans[0][0] == 0, f"bad first span for subject {subject_id}"
        assert visit_spans[-1][1] == n_events, f"bad last span for subject {subject_id}"
        for _k in range(len(visit_spans) - 1):
            assert visit_spans[_k][1] == visit_spans[_k + 1][0], (
                f"non-contiguous spans for subject {subject_id} at boundary {_k}"
            )

        code_id_arr = g["code_id"].astype(str).to_numpy()
        code_indices = np.fromiter(
            (self.code_vocab.get(c, self.unk_vocab_index) for c in code_id_arr),
            dtype=np.int32,
            count=n_events,
        )
        timestamps_days = g["timestamp_days"].fillna(0.0).astype("float32").to_numpy()
        age_days = g["age_at_event_days"].fillna(0.0).astype("float32").to_numpy()

        sex = encode_sex(g["sex"].iloc[0])
        race = encode_race(g["race"].iloc[0])

        return {
            "code_indices": code_indices,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": sex,
            "race": race,
            "visit_spans": visit_spans,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def _target_codes_for_visit(self, pat: dict[str, Any], visit_idx: int) -> np.ndarray:
        start, end = pat["visit_spans"][visit_idx]
        target = np.zeros(self.num_codes, dtype=np.float32)
        codes = pat["code_indices"][start:end]
        valid = codes[codes != self.unk_vocab_index]
        if valid.size:
            target[np.unique(valid)] = 1.0
        return target

    def __getitem__(self, idx: int) -> dict[str, Any]:
        spec = self._samples[idx]
        subject_id = int(spec["subject_id"])
        visit_k = int(spec["visit_k"])

        pat = self._materialize_patient(subject_id)
        visit_spans: list[tuple[int, int]] = pat["visit_spans"]

        if visit_k >= len(visit_spans) - 1:
            raise IndexError(
                f"visit_k={visit_k} out of range for subject {subject_id} "
                f"with {len(visit_spans)} visits (index/desync?)"
            )

        end_curr = visit_spans[visit_k][1]
        start_next = visit_spans[visit_k + 1][0]

        code_indices = pat["code_indices"][:end_curr].copy()
        timestamps_days = pat["timestamps_days"][:end_curr].copy()
        age_days = pat["age_days"][:end_curr].copy()

        if code_indices.shape[0] > self.max_seq_len:
            sl = slice(-self.max_seq_len, None)
            code_indices = code_indices[sl]
            timestamps_days = timestamps_days[sl]
            age_days = age_days[sl]

        target_codes = self._target_codes_for_visit(pat, visit_k + 1)

        last_curr = visit_spans[visit_k][1] - 1
        ts_curr_last = float(pat["timestamps_days"][last_curr])
        ts_next_first = float(pat["timestamps_days"][start_next])
        target_time_gap = abs(float(ts_next_first - ts_curr_last))

        return {
            "code_indices": code_indices,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": int(pat["sex"]),
            "race": int(pat["race"]),
            "target_codes": target_codes,
            "target_time_gap": target_time_gap,
        }


class TensorizedEHRDataset(Dataset):
    def __init__(
        self,
        tensorized_dir: str | Path,
        code_vocab_path: str | Path,
        max_seq_len: int = 1024,
        shard_cache_size: int = 4,
    ):
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
            raise FileNotFoundError(f"No shards in {self.tensorized_dir}. Run: python preprocessing/tensorize.py")

        self._index: list[tuple[int, int, int]] = []
        for shard_id, shard_path in enumerate(self._shard_paths):
            meta = np.load(shard_path, allow_pickle=True)
            visit_spans_arr = meta["visit_spans"]
            for pos in range(len(visit_spans_arr)):
                spans = np.asarray(visit_spans_arr[pos], dtype=np.int32)
                n_visits = int(spans.shape[0])
                for v in range(max(0, n_visits - 1)):
                    self._index.append((shard_id, pos, v))
            del meta

        self._shard_cache: dict[int, dict[str, Any]] = {}
        LOGGER.info(
            "TensorizedEHRDataset: %d samples across %d shards (%s)",
            len(self._index),
            len(self._shard_paths),
            self.tensorized_dir,
        )

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._shard_cache:
            shard = self._shard_cache.pop(shard_id)
            self._shard_cache[shard_id] = shard
            return shard

        if len(self._shard_cache) >= self.shard_cache_size:
            oldest_key = next(iter(self._shard_cache))
            del self._shard_cache[oldest_key]

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
            raise IndexError(
                f"visit_k={visit_k} out of range for shard {shard_id} pos {pos} with {len(spans)} visits"
            )

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
            code_slice = code_slice[sl]
            timestamps_days = timestamps_days[sl]
            age_days = age_days[sl]

        code_indices = np.fromiter(
            (self.code_vocab.get(str(c), self.unk_vocab_index) for c in code_slice),
            dtype=np.int32,
            count=len(code_slice),
        )

        target = np.zeros(self.num_codes, dtype=np.float32)
        next_codes_str = np.asarray(code_all[start_next:end_next], dtype=object)
        next_codes = np.fromiter(
            (self.code_vocab.get(str(c), self.unk_vocab_index) for c in next_codes_str),
            dtype=np.int32,
            count=len(next_codes_str),
        )
        valid = next_codes[next_codes != self.unk_vocab_index]
        if valid.size:
            target[np.unique(valid)] = 1.0

        target_time_gap = abs(float(ts_all[start_next] - ts_all[end_curr - 1]))

        return {
            "code_indices": code_indices,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": int(shard["sex"][pos]),
            "race": int(encode_race(shard["race"][pos])),
            "target_codes": target,
            "target_time_gap": target_time_gap,
        }


def ehr_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences and build pairwise Δt and BGE-aligned code indices."""
    if not batch:
        raise ValueError("empty batch")

    bsz = len(batch)
    num_codes = batch[0]["target_codes"].shape[0]
    unk_id = num_codes

    max_len = max(int(item["code_indices"].shape[0]) for item in batch)

    code_np = np.full((bsz, max_len), -1, dtype=np.int64)
    ts_np = np.zeros((bsz, max_len), dtype=np.float32)
    age_np = np.zeros((bsz, max_len), dtype=np.float32)
    mask_np = np.zeros((bsz, max_len), dtype=bool)

    target_codes_np = np.stack([item["target_codes"] for item in batch], axis=0)
    target_gap_np = np.array([item["target_time_gap"] for item in batch], dtype=np.float32)

    sex_arr = np.array([item["sex"] for item in batch], dtype=np.float32)
    race_arr = np.array([item["race"] for item in batch], dtype=np.float32)

    for b, item in enumerate(batch):
        seq = item["code_indices"]
        L = int(seq.shape[0])
        if L == 0:
            continue
        code_np[b, :L] = seq.astype(np.int64)
        ts_np[b, :L] = item["timestamps_days"].astype(np.float32)
        age_np[b, :L] = item["age_days"].astype(np.float32)
        mask_np[b, :L] = True

    code_indices = torch.from_numpy(code_np)
    timestamps_days = torch.from_numpy(ts_np)
    attention_mask = torch.from_numpy(mask_np)

    # BGE row indices: PAD=0, UNK=1, real vocab v -> v+2
    bge_codes = torch.where(
        attention_mask,
        torch.where(
            code_indices == unk_id,
            torch.ones((), dtype=torch.long),
            code_indices + 2,
        ),
        torch.zeros((), dtype=torch.long),
    )

    t = timestamps_days
    # Convert pairwise day-deltas to weeks before log1p (paper Section 4.1)
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
        "target_codes": torch.from_numpy(target_codes_np),
        "target_time_gap": torch.from_numpy(target_gap_np),
    }


def _invert_vocab(code_vocab: dict[str, int]) -> list[str]:
    n = len(code_vocab)
    out = [""] * n
    for cid, i in code_vocab.items():
        if 0 <= int(i) < n:
            out[int(i)] = str(cid)
    return out


if __name__ == "__main__":
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    setup_logging()
    parser = argparse.ArgumentParser(description="Smoke test EHRDataset + ehr_collate")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Use test_events.parquet instead of train_events.parquet",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=200_000,
        help="Limit raw parquet rows when building the visit index (smoke tests); 0 = full scan.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    processed = repo_root / "data" / "processed"
    split_name = "test_events.parquet" if args.test_mode else "train_events.parquet"
    parquet_path = processed / split_name
    vocab_path = processed / "code_vocab.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        code_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    index_to_code = _invert_vocab(code_vocab)

    row_limit = None if args.max_rows == 0 else args.max_rows
    dataset = EHRDataset(parquet_path=parquet_path, code_vocab_path=vocab_path, max_rows=row_limit)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=ehr_collate, num_workers=0)
    batch = next(iter(loader))

    print("Tensor shapes and dtypes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    bsz, L = batch["code_indices"].shape
    assert batch["delta_t"].shape == (bsz, L, L), "delta_t must be [B, L, L]"
    for k, v in batch.items():
        assert torch.isfinite(v).all(), f"non-finite values in {k}"

    print("\nSample code_indices (BGE rows) first two positions:", batch["code_indices"][:2, : min(8, L)])

    print("\nExample vocab strings for first row (map BGE index j -> vocab j-2 for j>=2):")
    row0 = batch["code_indices"][0, : min(10, L)]
    mask0 = batch["attention_mask"][0, : min(10, L)]
    for i in range(min(10, L)):
        if not mask0[i]:
            break
        j = int(row0[i].item())
        if j <= 1:
            print(f"  pos {i}: BGE={j} (PAD/UNK)")
        else:
            vi = j - 2
            cid = index_to_code[vi] if 0 <= vi < len(index_to_code) else "?"
            print(f"  pos {i}: BGE={j} vocab={vi} code_id={cid!r}")

    pos = batch["target_codes"].sum(dim=1)
    print("\nPositive labels per sample in batch:", pos.tolist())

    # Workers: DuckDB is not fork-safe; use spawn (matches model/train.py).
    ctx = mp.get_context("spawn")
    loader_w = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=ehr_collate,
        num_workers=2,
        multiprocessing_context=ctx,
        worker_init_fn=_dataloader_worker_init,
    )
    it = iter(loader_w)
    _ = next(it)
    del it
    del loader_w

    print("\nSmoke test passed.")
