#!/usr/bin/env python3
"""Tensorize disease cohorts into sharded NPZ files for fast multi-worker loading."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tensorized shards for disease fine-tuning cohorts.")
    parser.add_argument("--cohort_dir", type=Path, required=True, help="Directory with train/val/test cohort parquet files.")
    parser.add_argument("--events_parquet", type=Path, default=Path("data/processed/patient_events_rolled_full.parquet"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--out_dir", type=Path, required=True, help="Output tensorized root directory.")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--shard_size", type=int, default=50_000)
    return parser.parse_args()


def encode_race(race_val: Any) -> int:
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
            SELECT
                o.subject_id,
                o.code_id,
                o.timestamp_days,
                o.age_at_event_days,
                o.sex,
                o.race,
                o.event_idx
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
        labels: list[float] = []
        sexs: list[int] = []
        races: list[int] = []
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
            labels.append(label)
            sexs.append(sex)
            races.append(race)
            code_indices_list.append(code_indices)
            timestamps_list.append(timestamps_days.astype(np.float32, copy=False))
            age_days_list.append(age_days.astype(np.float32, copy=False))

        shard_path = out_split_dir / f"shard_{shard_idx:05d}.npz"
        np.savez(
            shard_path,
            subject_id=np.asarray(subject_ids, dtype=np.int64),
            label=np.asarray(labels, dtype=np.float32),
            sex=np.asarray(sexs, dtype=np.int8),
            race=np.asarray(races, dtype=np.int16),
            unk_vocab_index=np.asarray([unk_vocab_index], dtype=np.int64),
            code_indices=np.asarray(code_indices_list, dtype=object),
            timestamps_days=np.asarray(timestamps_list, dtype=object),
            age_days=np.asarray(age_days_list, dtype=object),
        )
        total_written += len(subject_ids)
        if shard_idx % 10 == 0 or shard_idx == (n_shards - 1):
            print(
                f"[{split}] shard {shard_idx + 1}/{n_shards} "
                f"rows={len(subject_ids):,} total_written={total_written:,}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"[{split}] done in {elapsed:.1f}s | rows={total_written:,} shards={n_shards}", flush=True)
    return total_written, n_shards


def main() -> int:
    args = parse_args()
    if not args.cohort_dir.exists():
        raise FileNotFoundError(f"Missing cohort dir: {args.cohort_dir}")
    if not args.events_parquet.exists():
        raise FileNotFoundError(f"Missing events parquet: {args.events_parquet}")
    if not args.vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab path: {args.vocab_path}")

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
            out_split_dir = args.out_dir / split
            split_totals[split] = _process_split(
                con=con,
                split=split,
                cohort_parquet=cohort_parquet,
                events_parquet=args.events_parquet,
                code_vocab=code_vocab,
                out_split_dir=out_split_dir,
                max_seq_len=args.max_seq_len,
                shard_size=args.shard_size,
            )

        print("[done] tensorized disease dataset", flush=True)
        for split in ("train", "val", "test"):
            rows, shards = split_totals[split]
            print(f"  - {split}: rows={rows:,}, shards={shards}, dir={args.out_dir / split}", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
