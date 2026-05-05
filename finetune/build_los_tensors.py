#!/usr/bin/env python3
"""Tensorize LOS cohorts with admission-aware event cutoffs."""

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
    parser = argparse.ArgumentParser(description="Build tensorized shards for LOS admission cohorts.")
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
    per_subject_max = (
        cohort_chunk.groupby("subject_id", as_index=False, sort=False)["last_event_idx"]
        .max()
        .astype({"subject_id": "int64", "last_event_idx": "int64"})
        .rename(columns={"last_event_idx": "max_last_event_idx"})
    )
    con.register("chunk_subject_bounds", per_subject_max)
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
                JOIN chunk_subject_bounds b USING (subject_id)
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
            JOIN chunk_subject_bounds b USING (subject_id)
            WHERE o.event_idx <= b.max_last_event_idx
            ORDER BY o.subject_id, o.event_idx
            """
        ).df()
    finally:
        con.unregister("chunk_subject_bounds")

    out: dict[int, pd.DataFrame] = {}
    if df.empty:
        return out
    for sid, g in df.groupby("subject_id", sort=False):
        out[int(sid)] = g
    return out


def _cohort_with_row_order(con: duckdb.DuckDBPyConnection, cohort_parquet: Path) -> pd.DataFrame:
    cohort = con.execute(
        """
        SELECT
            ROW_NUMBER() OVER () - 1 AS cohort_row_idx,
            subject_id,
            hadm_id,
            label,
            last_event_idx,
            los_days
        FROM read_parquet(?)
        """,
        [str(cohort_parquet.resolve())],
    ).df()
    if cohort.empty:
        raise RuntimeError(f"Empty cohort: {cohort_parquet}")
    return cohort


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
    cohort = _cohort_with_row_order(con, cohort_parquet)
    unk_vocab_index = int(len(code_vocab))
    out_split_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(cohort)
    n_unique_subjects = int(cohort["subject_id"].nunique())
    admissions_per_subject_mean = float(n_rows / max(n_unique_subjects, 1))
    n_shards = int(math.ceil(n_rows / shard_size))
    print(
        f"[{split}] n_admissions={n_rows:,} n_unique_subjects={n_unique_subjects:,} "
        f"admissions_per_subject_mean={admissions_per_subject_mean:.3f}",
        flush=True,
    )

    total_written = 0
    seq_lens: list[int] = []
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
        los_days_out: list[float] = []
        code_indices_list: list[np.ndarray] = []
        timestamps_list: list[np.ndarray] = []
        age_days_list: list[np.ndarray] = []

        for row in cohort_chunk.itertuples(index=False):
            sid = int(row.subject_id)
            hadm_id = int(row.hadm_id)
            label = float(row.label)
            row_last_event_idx = int(row.last_event_idx)
            row_los_days = float(row.los_days)

            g_all = event_map.get(sid)
            if g_all is None or g_all.empty:
                raise RuntimeError(f"No tensorizable events for subject_id={sid} (split={split}, hadm_id={hadm_id})")

            g = g_all.loc[g_all["event_idx"].astype(np.int64) <= row_last_event_idx]
            if g.empty:
                raise RuntimeError(
                    f"No events <= last_event_idx for subject_id={sid} hadm_id={hadm_id} last_event_idx={row_last_event_idx}"
                )

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
            hadm_ids.append(hadm_id)
            labels.append(label)
            sexs.append(sex)
            races.append(race)
            los_days_out.append(row_los_days)
            code_indices_list.append(code_indices)
            timestamps_list.append(timestamps_days.astype(np.float32, copy=False))
            age_days_list.append(age_days.astype(np.float32, copy=False))
            seq_lens.append(int(len(code_indices)))

        shard_path = out_split_dir / f"shard_{shard_idx:05d}.npz"
        np.savez(
            shard_path,
            subject_id=np.asarray(subject_ids, dtype=np.int64),
            hadm_id=np.asarray(hadm_ids, dtype=np.int64),
            label=np.asarray(labels, dtype=np.float32),
            sex=np.asarray(sexs, dtype=np.int8),
            race=np.asarray(races, dtype=np.int16),
            los_days=np.asarray(los_days_out, dtype=np.float32),
            unk_vocab_index=np.asarray([unk_vocab_index], dtype=np.int64),
            code_indices=np.asarray(code_indices_list, dtype=object),
            timestamps_days=np.asarray(timestamps_list, dtype=object),
            age_days=np.asarray(age_days_list, dtype=object),
        )
        total_written += len(subject_ids)
        if shard_idx % 10 == 0 or shard_idx == (n_shards - 1):
            print(
                f"[{split}] shard {shard_idx + 1}/{n_shards} rows={len(subject_ids):,} total_written={total_written:,}",
                flush=True,
            )

    seq_arr = np.asarray(seq_lens, dtype=np.int64)
    seq_mean = float(np.mean(seq_arr)) if seq_arr.size > 0 else float("nan")
    seq_median = float(np.median(seq_arr)) if seq_arr.size > 0 else float("nan")
    seq_p95 = float(np.percentile(seq_arr, 95)) if seq_arr.size > 0 else float("nan")
    elapsed = time.perf_counter() - t0
    print(
        f"[{split}] shards_written={n_shards} rows_written={total_written:,} "
        f"seq_len mean={seq_mean:.2f} median={seq_median:.2f} p95={seq_p95:.2f} time={elapsed:.1f}s",
        flush=True,
    )
    return total_written, n_shards


def _verify_train_admission_behavior(
    con: duckdb.DuckDBPyConnection,
    train_cohort: pd.DataFrame,
    train_shard_dir: Path,
    events_parquet: Path,
    code_vocab: dict[str, int],
    max_seq_len: int,
) -> tuple[bool, str]:
    candidate_subjects: list[int] = []
    multi_counts = train_cohort.groupby("subject_id")["hadm_id"].size()
    for sid in sorted([int(s) for s, n in multi_counts.items() if int(n) >= 2]):
        cohort_sub = train_cohort.loc[train_cohort["subject_id"] == sid].copy()
        cohort_sub = cohort_sub.sort_values("hadm_id", kind="mergesort")
        if cohort_sub["last_event_idx"].is_monotonic_increasing:
            candidate_subjects.append(sid)
        if len(candidate_subjects) == 5:
            break
    if len(candidate_subjects) < 5:
        return False, f"need >=5 multi-admission subjects with monotonic hadm_id ordering, found {len(candidate_subjects)}"

    selected = set(candidate_subjects)
    captured: dict[int, list[dict[str, Any]]] = {sid: [] for sid in candidate_subjects}
    offset = 0
    shard_paths = sorted(train_shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        return False, "no train shards found for verification"

    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=True) as z:
            sids = z["subject_id"].astype(np.int64)
            hadms = z["hadm_id"].astype(np.int64)
            labels = z["label"].astype(np.float32)
            los_days = z["los_days"].astype(np.float32)
            codes = z["code_indices"]

            for i in range(len(sids)):
                sid = int(sids[i])
                if sid in selected:
                    captured[sid].append(
                        {
                            "cohort_row_idx": offset + i,
                            "hadm_id": int(hadms[i]),
                            "label": float(labels[i]),
                            "los_days": float(los_days[i]),
                            "code_indices": np.asarray(codes[i], dtype=np.int64),
                        }
                    )
        offset += len(sids)

    events_sql = str(events_parquet.resolve()).replace("'", "''")
    unk_vocab_index = int(len(code_vocab))

    for sid in candidate_subjects:
        cohort_sub = train_cohort.loc[train_cohort["subject_id"] == sid].copy()
        cohort_sub = cohort_sub.sort_values("hadm_id", kind="mergesort").reset_index(drop=True)
        if not cohort_sub["last_event_idx"].is_monotonic_increasing:
            return False, f"subject_id={sid} has non-monotonic last_event_idx after hadm_id sort"

        entry_map = {entry["hadm_id"]: entry for entry in captured[sid]}
        if len(entry_map) != len(cohort_sub):
            return False, f"subject_id={sid} missing shard entries (expected {len(cohort_sub)} got {len(entry_map)})"

        sid_max_idx = int(cohort_sub["last_event_idx"].max())
        ev = con.execute(
            f"""
            WITH ordered AS (
                SELECT
                    e.code_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.subject_id
                        ORDER BY e.timestamp_days, e.event_time, e.code_id
                    ) - 1 AS event_idx
                FROM read_parquet('{events_sql}') e
                WHERE e.subject_id = ?
            )
            SELECT code_id, event_idx
            FROM ordered
            WHERE event_idx <= ?
            ORDER BY event_idx
            """,
            [sid, sid_max_idx],
        ).df()
        if ev.empty:
            return False, f"subject_id={sid} has no ordered events for verification"

        event_codes = ev["code_id"].astype(str).to_numpy()

        for _, row in cohort_sub.iterrows():
            hadm_id = int(row["hadm_id"])
            last_idx = int(row["last_event_idx"])
            expected_codes = event_codes[: last_idx + 1]
            expected_idx = np.fromiter(
                (code_vocab.get(str(c), unk_vocab_index) for c in expected_codes),
                dtype=np.int64,
                count=len(expected_codes),
            )
            if expected_idx.shape[0] > max_seq_len:
                expected_idx = expected_idx[-max_seq_len:]

            got = entry_map[hadm_id]["code_indices"]
            if not np.array_equal(got, expected_idx):
                return False, f"subject_id={sid} hadm_id={hadm_id} shard sequence mismatch vs expected cutoff"

        full_sequences = [event_codes[: int(le) + 1] for le in cohort_sub["last_event_idx"].astype(np.int64).tolist()]
        for i in range(1, len(full_sequences)):
            prev_seq = full_sequences[i - 1]
            cur_seq = full_sequences[i]
            if len(prev_seq) > len(cur_seq) or not np.array_equal(prev_seq, cur_seq[: len(prev_seq)]):
                return False, f"subject_id={sid} earlier admission sequence is not prefix of later admission sequence"

        labels_series = cohort_sub["label"]
        los_series = cohort_sub["los_days"]
        captured_labels = np.array([entry_map[int(h)]["label"] for h in cohort_sub["hadm_id"]], dtype=np.float32)
        captured_los = np.array([entry_map[int(h)]["los_days"] for h in cohort_sub["hadm_id"]], dtype=np.float32)
        if not np.array_equal(captured_labels.astype(np.int8), labels_series.astype(np.int8).to_numpy()):
            return False, f"subject_id={sid} shard labels do not match cohort labels"
        if not np.allclose(captured_los, los_series.astype(np.float32).to_numpy(), atol=1e-4):
            return False, f"subject_id={sid} shard los_days do not match cohort los_days"
        if labels_series.nunique() > 1 and len(np.unique(captured_labels)) <= 1:
            return False, f"subject_id={sid} expected varying labels across admissions, found constant labels in shard"
        if los_series.nunique() > 1 and len(np.unique(captured_los)) <= 1:
            return False, f"subject_id={sid} expected varying los_days across admissions, found constant los_days in shard"

    return True, "checked 5 multi-admission train subjects"


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
        con.execute("PRAGMA memory_limit='24GB'")
        con.execute("PRAGMA threads=14")
        split_totals: dict[str, tuple[int, int]] = {}
        train_cohort: pd.DataFrame | None = None
        for split in ("train", "val", "test"):
            cohort_parquet = args.cohort_dir / f"{split}_cohort.parquet"
            if not cohort_parquet.exists():
                raise FileNotFoundError(f"Missing split cohort: {cohort_parquet}")
            out_split_dir = args.out_dir / split
            if split == "train":
                train_cohort = _cohort_with_row_order(con, cohort_parquet)
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

        if train_cohort is None:
            raise RuntimeError("Train cohort did not load; cannot run verification.")
        ok, detail = _verify_train_admission_behavior(
            con=con,
            train_cohort=train_cohort,
            train_shard_dir=args.out_dir / "train",
            events_parquet=args.events_parquet,
            code_vocab=code_vocab,
            max_seq_len=args.max_seq_len,
        )
        if ok:
            print(f"[verify] OK: {detail}", flush=True)
        else:
            print(f"[verify] FAIL: {detail}", flush=True)
            raise RuntimeError(detail)

        print("[done] tensorized LOS dataset", flush=True)
        for split in ("train", "val", "test"):
            rows, shards = split_totals[split]
            print(f"  - {split}: rows={rows:,}, shards={shards}, dir={args.out_dir / split}", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
