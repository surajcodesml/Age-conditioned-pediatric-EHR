#!/usr/bin/env python3
"""Offline tensorization into subject shards for TALE-EHR training."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def encode_sex(val) -> int:
    if val is None:
        return 0
    try:
        if isinstance(val, float) and np.isnan(val):
            return 0
    except Exception:
        pass
    try:
        v = int(val)
    except (TypeError, ValueError):
        return 0
    return 1 if v == 1 else 0


def _build_subject_payload(g, code_vocab: dict[str, int]) -> dict | None:
    g = g.copy()
    # Robust visit-slot construction.
    g["_hadm_fill"] = g["hadm_id"].fillna(-1).astype("int64")
    first_ts_per_hadm = g.groupby("_hadm_fill")["event_time"].min().sort_values(kind="mergesort")
    hadm_to_visit_idx = {h: k for k, h in enumerate(first_ts_per_hadm.index)}
    g["_visit_idx"] = g["_hadm_fill"].map(hadm_to_visit_idx).astype("int64")
    g = g.sort_values(["_visit_idx", "event_time", "code_id"], kind="mergesort").reset_index(drop=True)

    n_events = len(g)
    visit_idx_arr = g["_visit_idx"].to_numpy()
    visit_spans: list[tuple[int, int]] = []
    start_i = 0
    for i in range(1, n_events):
        if visit_idx_arr[i] != visit_idx_arr[i - 1]:
            visit_spans.append((start_i, i))
            start_i = i
    visit_spans.append((start_i, n_events))

    if len(visit_spans) < 2:
        return None

    return {
        "subject_id": int(g["subject_id"].iloc[0]),
        "code_id": g["code_id"].astype(str).to_numpy(),
        "timestamps_days": g["timestamp_days"].fillna(0.0).astype("float32").to_numpy(),
        "age_days": g["age_at_event_days"].fillna(0.0).astype("float32").to_numpy(),
        "sex": np.int8(encode_sex(g["sex"].iloc[0])),
        "race": "" if g["race"].iloc[0] is None else str(g["race"].iloc[0]),
        "visit_spans": np.asarray(visit_spans, dtype=np.int32),
    }


def _tensorize_worker(args: tuple[str, list[int], str, str]) -> int:
    parquet_path, subject_ids, out_npz_path, vocab_path = args
    with Path(vocab_path).open("r", encoding="utf-8") as f:
        code_vocab = {str(k): int(v) for k, v in json.load(f).items()}

    table = pq.read_table(parquet_path, filters=[("subject_id", "in", subject_ids)])
    df = table.to_pandas()
    if df.empty:
        np.savez_compressed(
            out_npz_path,
            subject_id=np.array([], dtype=np.int64),
            code_id=np.array([], dtype=object),
            timestamps_days=np.array([], dtype=object),
            age_days=np.array([], dtype=object),
            sex=np.array([], dtype=np.int8),
            race=np.array([], dtype=object),
            visit_spans=np.array([], dtype=object),
        )
        return 0

    payloads: list[dict] = []
    for _, g in df.groupby("subject_id", sort=False):
        p = _build_subject_payload(g, code_vocab)
        if p is not None:
            payloads.append(p)

    subject_id = np.array([p["subject_id"] for p in payloads], dtype=np.int64)
    code_id = np.array([p["code_id"] for p in payloads], dtype=object)
    timestamps_days = np.array([p["timestamps_days"] for p in payloads], dtype=object)
    age_days = np.array([p["age_days"] for p in payloads], dtype=object)
    sex = np.array([p["sex"] for p in payloads], dtype=np.int8)
    race = np.array([p["race"] for p in payloads], dtype=object)
    visit_spans = np.array([p["visit_spans"] for p in payloads], dtype=object)

    np.savez_compressed(
        out_npz_path,
        subject_id=subject_id,
        code_id=code_id,
        timestamps_days=timestamps_days,
        age_days=age_days,
        sex=sex,
        race=race,
        visit_spans=visit_spans,
    )
    return int(len(payloads))


def _chunk(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tensorize EHR parquet into subject shards.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/tensorized"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--num_workers", type=int, default=max(mp.cpu_count() // 2, 1))
    parser.add_argument("--shard_size", type=int, default=512)
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        parquet_path = args.data_dir / f"{split}_events.parquet"
        if not parquet_path.exists():
            continue

        ids_table = pq.read_table(parquet_path, columns=["subject_id"])
        subject_ids = np.unique(ids_table["subject_id"].to_numpy()).astype(np.int64)
        subject_ids.sort()
        shards = _chunk(subject_ids.tolist(), args.shard_size)
        split_out = args.out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        print(f"[{split}] {len(subject_ids)} subjects -> {len(shards)} shards")
        jobs: list[tuple[str, list[int], str, str]] = []
        for i, shard_subjects in enumerate(shards):
            out_npz = split_out / f"shard_{i:04d}.npz"
            jobs.append((str(parquet_path), shard_subjects, str(out_npz), str(args.vocab_path)))

        with mp.Pool(processes=args.num_workers) as pool:
            for done_i, n_written in enumerate(pool.imap_unordered(_tensorize_worker, jobs), start=1):
                print(f"[{split}] shard {done_i}/{len(jobs)} done ({n_written} subjects)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

