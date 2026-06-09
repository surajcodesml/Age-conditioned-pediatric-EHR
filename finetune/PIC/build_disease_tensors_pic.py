#!/usr/bin/env python3
"""Tensorize PIC disease cohorts into sharded NPZ files for fast loading.

Same shard layout as ``finetune/build_disease_tensors.py`` but additionally
writes the ``hadm_id`` and ``n_events_in_window`` arrays that
``TensorizedDiseaseClassificationDataset`` reads (the parent tensorizer in this
checkout omits ``hadm_id``, which the dataset requires). Event ordering matches
``(timestamp_days, event_time, code_id)`` exactly so ``last_event_idx`` lines up
with the cohort builder.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from finetune.dataset import encode_race  # reused, read-only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PIC tensorized disease shards.")
    parser.add_argument("--cohort_dir", type=Path, required=True)
    parser.add_argument("--events_parquet", type=Path, required=True)
    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--shard_size", type=int, default=50_000)
    return parser.parse_args()


def _build_subject_event_map(
    con: duckdb.DuckDBPyConnection,
    events_parquet: Path,
    cohort_chunk: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    con.register("chunk_cohort", cohort_chunk)
    events_sql = str(events_parquet.resolve()).replace("'", "''")
    try:
        df = con.execute(
            f"""
            WITH ordered AS (
                SELECT
                    e.subject_id,
                    e.hadm_id,
                    e.code_id,
                    e.timestamp_days,
                    e.age_at_event_days,
                    e.sex,
                    e.race,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.subject_id
                        ORDER BY e.timestamp_days, e.event_time, e.code_id
                    ) - 1 AS event_idx
                FROM read_parquet('{events_sql}') e
                JOIN chunk_cohort c USING (subject_id)
            )
            SELECT o.subject_id, o.hadm_id, o.code_id, o.timestamp_days,
                   o.age_at_event_days, o.sex, o.race, o.event_idx
            FROM ordered o
            JOIN chunk_cohort c USING (subject_id)
            WHERE o.event_idx <= c.last_event_idx
            ORDER BY o.subject_id, o.event_idx
            """
        ).df()
    finally:
        con.unregister("chunk_cohort")

    out: dict[int, pd.DataFrame] = {}
    if len(df) == 0:
        return out
    for sid, g in df.groupby("subject_id", sort=False):
        out[int(sid)] = g
    return out


def _process_split(
    con: duckdb.DuckDBPyConnection,
    split: str,
    cohort_parquet: Path,
    events_parquet: Path,
    code_vocab: dict[str, int],
    out_split_dir: Path,
    max_seq_len: int,
    shard_size: int,
) -> tuple[int, int]:
    cohort = con.execute(
        "SELECT subject_id, label, last_event_idx FROM read_parquet(?) ORDER BY subject_id",
        [str(cohort_parquet.resolve())],
    ).df()
    if cohort.empty:
        raise RuntimeError(f"Empty cohort for split={split}: {cohort_parquet}")

    unk_vocab_index = int(len(code_vocab))
    out_split_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(cohort)
    n_shards = int(math.ceil(n_rows / shard_size))
    print(f"[{split}] rows={n_rows:,} shard_size={shard_size:,} -> shards={n_shards}", flush=True)

    total_written = 0
    t0 = time.perf_counter()
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, n_rows)
        cohort_chunk = cohort.iloc[start:end].copy()
        event_map = _build_subject_event_map(con, events_parquet, cohort_chunk)

        subject_ids: list[int] = []
        hadm_ids: list[int] = []
        labels: list[float] = []
        sexs: list[int] = []
        races: list[int] = []
        n_events_window: list[int] = []
        code_indices_list: list[np.ndarray] = []
        timestamps_list: list[np.ndarray] = []
        age_days_list: list[np.ndarray] = []

        for row in cohort_chunk.itertuples(index=False):
            sid = int(row.subject_id)
            label = float(row.label)
            g = event_map.get(sid)
            if g is None or g.empty:
                raise RuntimeError(f"No tensorizable events for subject_id={sid} in split={split}")

            code_ids = g["code_id"].astype(str).to_numpy()
            timestamps_days = g["timestamp_days"].astype(np.float32).to_numpy()
            age_days = g["age_at_event_days"].astype(np.float32).to_numpy()
            sex = int(g["sex"].iloc[0])
            race = encode_race(g["race"].iloc[0])
            # hadm of the last in-window event (index-admission proxy / metadata only)
            hadm_val = g["hadm_id"].iloc[-1]
            hadm_id = int(hadm_val) if pd.notna(hadm_val) else -1
            n_evt = int(code_ids.shape[0])

            if code_ids.shape[0] > max_seq_len:
                sl = slice(-max_seq_len, None)
                code_ids = code_ids[sl]
                timestamps_days = timestamps_days[sl]
                age_days = age_days[sl]

            code_indices = np.fromiter(
                (code_vocab.get(str(c), unk_vocab_index) for c in code_ids),
                dtype=np.int64,
                count=len(code_ids),
            )

            subject_ids.append(sid)
            hadm_ids.append(hadm_id)
            labels.append(label)
            sexs.append(sex)
            races.append(race)
            n_events_window.append(n_evt)
            code_indices_list.append(code_indices)
            timestamps_list.append(timestamps_days.astype(np.float32, copy=False))
            age_days_list.append(age_days.astype(np.float32, copy=False))

        shard_path = out_split_dir / f"shard_{shard_idx:05d}.npz"
        seq_len_arr = np.asarray([len(c) for c in code_indices_list], dtype=np.int64)
        offsets = np.zeros(len(code_indices_list) + 1, dtype=np.int64)
        np.cumsum(seq_len_arr, out=offsets[1:])
        code_concat = np.concatenate(code_indices_list).astype(np.int64) if code_indices_list else np.zeros(0, np.int64)
        ts_concat = np.concatenate(timestamps_list).astype(np.float32) if timestamps_list else np.zeros(0, np.float32)
        age_concat = np.concatenate(age_days_list).astype(np.float32) if age_days_list else np.zeros(0, np.float32)
        np.savez(
            shard_path,
            subject_id=np.asarray(subject_ids, dtype=np.int64),
            hadm_id=np.asarray(hadm_ids, dtype=np.int64),
            label=np.asarray(labels, dtype=np.float32),
            sex=np.asarray(sexs, dtype=np.int8),
            race=np.asarray(races, dtype=np.int16),
            n_events_in_window=np.asarray(n_events_window, dtype=np.int64),
            unk_vocab_index=np.asarray([unk_vocab_index], dtype=np.int64),
            offsets=offsets,
            code_indices=code_concat,
            timestamps_days=ts_concat,
            age_days=age_concat,
        )
        total_written += len(subject_ids)
        print(f"[{split}] shard {shard_idx + 1}/{n_shards} rows={len(subject_ids):,} total={total_written:,}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[{split}] done in {elapsed:.1f}s | rows={total_written:,} shards={n_shards}", flush=True)
    return total_written, n_shards


def main() -> int:
    args = parse_args()
    for p in (args.cohort_dir, args.events_parquet, args.vocab_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    with args.vocab_path.open("r", encoding="utf-8") as f:
        code_vocab: dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit='8GB'")
        con.execute("PRAGMA threads=8")
        split_totals: dict[str, tuple[int, int]] = {}
        for split in ("train", "val", "test"):
            cohort_parquet = args.cohort_dir / f"{split}_cohort.parquet"
            if not cohort_parquet.exists():
                raise FileNotFoundError(f"Missing split cohort: {cohort_parquet}")
            split_totals[split] = _process_split(
                con=con,
                split=split,
                cohort_parquet=cohort_parquet,
                events_parquet=args.events_parquet,
                code_vocab=code_vocab,
                out_split_dir=args.out_dir / split,
                max_seq_len=args.max_seq_len,
                shard_size=args.shard_size,
            )
        print("[done] tensorized PIC disease dataset", flush=True)
        for split in ("train", "val", "test"):
            rows, shards = split_totals[split]
            print(f"  - {split}: rows={rows:,}, shards={shards}, dir={args.out_dir / split}", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
