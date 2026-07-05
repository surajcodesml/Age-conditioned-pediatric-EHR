#!/usr/bin/env python3
"""Re-tensorize the SHARED pretrain shards into a flat, spawn-safe numeric format.

Motivation
----------
The legacy pretrain shards (``preprocessing/tensorize.py``) store ragged per-patient
sequences as ``dtype=object`` arrays and are read with ``allow_pickle=True``. Under a
spawned DataLoader worker, the numpy object-array/pickle machinery races at process
teardown and turns a benign worker shutdown into ``std::terminate`` (nonzero exit),
even though training and the checkpoint are fine.

This script emits the SAME flat schema the fine-tune tensorizer already uses
(``finetune/build_disease_tensors.py``) -- flat numeric arrays + offsets, no object
arrays, written **uncompressed** so the reader can ``mmap`` with ``allow_pickle=False``.
That configuration is proven clean under spawn in this repo.

Semantics are byte-for-byte equivalent to the legacy pretrain reader:
  * visit ordering, visit spans, and the >=2-visit filter come straight from the
    reused ``_build_subject_payload`` (same code path that built the old shards);
  * code strings are vocab-mapped at build time (unk -> ``unk_vocab_index``);
  * race is encoded at build time (same ``encode_race`` as the reader);
  * full per-patient event history is kept (truncation to ``max_seq_len`` stays in
    ``__getitem__`` because different visit targets need different context lengths).

Flat shard schema (all numeric, ``allow_pickle=False``-loadable, mmap-able):
  subject_id      int64  [n]
  sex             int8   [n]
  race            int16  [n]
  event_offsets   int64  [n+1]   slice into the flat event arrays, per patient
  code_indices    int64  [E]     vocab-mapped codes, concatenated over patients
  timestamps_days float32[E]
  age_days        float32[E]
  visit_offsets   int64  [n+1]   slice into the flat visit arrays, per patient
  visit_starts    int32  [V]     visit start, RELATIVE to the patient's event block
  visit_ends      int32  [V]     visit end,   RELATIVE to the patient's event block
  unk_vocab_index int64  [1]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact visit-span construction that built the legacy shards so the
# re-tensorized data is semantically identical (same ordering / >=2-visit filter).
from preprocessing.tensorize import _build_subject_payload
from model_ablation.dataset import encode_race

_EVENT_COLUMNS = [
    "subject_id", "hadm_id", "event_time", "code_id",
    "timestamp_days", "age_at_event_days", "sex", "race",
]


def _tensorize_worker(args: tuple[str, list[int], str, str]) -> tuple[int, int]:
    parquet_path, subject_ids, out_npz_path, vocab_path = args
    with Path(vocab_path).open("r", encoding="utf-8") as f:
        code_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    unk = len(code_vocab)

    table = pq.read_table(
        parquet_path,
        columns=_EVENT_COLUMNS,
        filters=[("subject_id", "in", subject_ids)],
    )
    df = table.to_pandas()

    subj: list[int] = []
    sexs: list[int] = []
    races: list[int] = []
    code_blocks: list[np.ndarray] = []
    ts_blocks: list[np.ndarray] = []
    age_blocks: list[np.ndarray] = []
    vs_blocks: list[np.ndarray] = []
    ve_blocks: list[np.ndarray] = []
    ev_lens: list[int] = []
    vis_counts: list[int] = []

    if not df.empty:
        for _, g in df.groupby("subject_id", sort=False):
            payload = _build_subject_payload(g, code_vocab)
            if payload is None:  # fewer than 2 visits -> not a training sample
                continue
            codes = payload["code_id"]
            code_ids = np.fromiter(
                (code_vocab.get(str(c), unk) for c in codes),
                dtype=np.int64, count=len(codes),
            )
            spans = np.asarray(payload["visit_spans"], dtype=np.int32)  # [n_visits, 2]

            subj.append(int(payload["subject_id"]))
            sexs.append(int(payload["sex"]))
            races.append(int(encode_race(payload["race"])))
            code_blocks.append(code_ids)
            ts_blocks.append(np.asarray(payload["timestamps_days"], dtype=np.float32))
            age_blocks.append(np.asarray(payload["age_days"], dtype=np.float32))
            vs_blocks.append(spans[:, 0].astype(np.int32, copy=False))
            ve_blocks.append(spans[:, 1].astype(np.int32, copy=False))
            ev_lens.append(int(code_ids.shape[0]))
            vis_counts.append(int(spans.shape[0]))

    n = len(subj)
    event_offsets = np.zeros(n + 1, dtype=np.int64)
    visit_offsets = np.zeros(n + 1, dtype=np.int64)
    if n:
        np.cumsum(np.asarray(ev_lens, dtype=np.int64), out=event_offsets[1:])
        np.cumsum(np.asarray(vis_counts, dtype=np.int64), out=visit_offsets[1:])

    code_concat = np.concatenate(code_blocks) if code_blocks else np.zeros(0, np.int64)
    ts_concat = np.concatenate(ts_blocks) if ts_blocks else np.zeros(0, np.float32)
    age_concat = np.concatenate(age_blocks) if age_blocks else np.zeros(0, np.float32)
    vs_concat = np.concatenate(vs_blocks) if vs_blocks else np.zeros(0, np.int32)
    ve_concat = np.concatenate(ve_blocks) if ve_blocks else np.zeros(0, np.int32)

    np.savez(  # uncompressed on purpose: required for mmap_mode="r"
        out_npz_path,
        subject_id=np.asarray(subj, dtype=np.int64),
        sex=np.asarray(sexs, dtype=np.int8),
        race=np.asarray(races, dtype=np.int16),
        event_offsets=event_offsets,
        code_indices=code_concat.astype(np.int64, copy=False),
        timestamps_days=ts_concat.astype(np.float32, copy=False),
        age_days=age_concat.astype(np.float32, copy=False),
        visit_offsets=visit_offsets,
        visit_starts=vs_concat.astype(np.int32, copy=False),
        visit_ends=ve_concat.astype(np.int32, copy=False),
        unk_vocab_index=np.asarray([unk], dtype=np.int64),
    )
    return n, int(code_concat.shape[0])


def _chunk(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def main() -> int:
    p = argparse.ArgumentParser(description="Re-tensorize pretrain shards into flat spawn-safe format.")
    p.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data/processed")
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / "data/processed/tensorized_flat")
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--num_workers", type=int, default=max(mp.cpu_count() // 2, 1))
    p.add_argument("--shard_size", type=int, default=512, help="subjects per shard")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    if not args.vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab: {args.vocab_path}")

    for split in args.splits:
        parquet_path = args.data_dir / f"{split}_events.parquet"
        if not parquet_path.exists():
            print(f"[{split}] SKIP (missing {parquet_path})", flush=True)
            continue

        ids_table = pq.read_table(parquet_path, columns=["subject_id"])
        subject_ids = np.unique(ids_table["subject_id"].to_numpy()).astype(np.int64)
        subject_ids.sort()
        shards = _chunk(subject_ids.tolist(), args.shard_size)
        split_out = args.out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        jobs = [
            (str(parquet_path), shard_subjects, str(split_out / f"shard_{i:04d}.npz"), str(args.vocab_path))
            for i, shard_subjects in enumerate(shards)
        ]
        print(f"[{split}] {len(subject_ids):,} subjects -> {len(jobs)} shards "
              f"(shard_size={args.shard_size}, workers={args.num_workers})", flush=True)

        t0 = time.perf_counter()
        total_patients = 0
        total_events = 0
        with mp.Pool(processes=args.num_workers) as pool:
            for done_i, (n_pat, n_ev) in enumerate(pool.imap_unordered(_tensorize_worker, jobs), start=1):
                total_patients += n_pat
                total_events += n_ev
                if done_i % 25 == 0 or done_i == len(jobs):
                    rate = done_i / max(time.perf_counter() - t0, 1e-9)
                    eta = (len(jobs) - done_i) / max(rate, 1e-9)
                    print(f"[{split}] shard {done_i}/{len(jobs)} | "
                          f"patients={total_patients:,} events={total_events:,} | "
                          f"{rate:.2f} shard/s | eta {eta/60:.1f} min", flush=True)
        elapsed = time.perf_counter() - t0
        print(f"[{split}] DONE in {elapsed/60:.1f} min | patients={total_patients:,} "
              f"events={total_events:,} shards={len(jobs)} dir={split_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
