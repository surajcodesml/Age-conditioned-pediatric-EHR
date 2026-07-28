#!/usr/bin/env python3
"""Out-of-core deterministic rebuild of PIC fine-tune classification shards (Phase 4).

Mirrors :mod:`model_new.tensorize_pretrain`: DuckDB partitions events by a stable
``hash(subject_id) % N_BUCKETS``, then ``ProcessPoolExecutor`` builds one
``shard_{bucket:05d}.npz`` per nonempty bucket. Self-contained (no imports from
``finetune/``). Output schema matches :class:`model_new.data_finetune.TensorizedFinetuneDataset`.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from model_new import diagnostics as D
from model_new.data import encode_race
from model_new.tensorize_pretrain import (
    _mem_available_gb,
    _peak_rss_mb,
    encode_sex,
    recommend_workers,
)

__all__ = ["rebuild_pic_task"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _partition_pic_events(events_parquet: Path, cohort_parquet: Path, part_dir: Path,
                          *, n_buckets: int, duckdb_mem: str = "4GB") -> dict[str, Any]:
    import duckdb
    import shutil

    part_dir = Path(part_dir)
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    tmp = part_dir / "_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA memory_limit='{duckdb_mem}'")
        con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
        con.execute("PRAGMA threads=4")
        # Restrict to cohort subjects; order events as the legacy PIC builder did
        # (timestamp_days, event_time, code_id) so last_event_idx truncation matches.
        ev = events_parquet.resolve().as_posix()
        co = cohort_parquet.resolve().as_posix()
        out = part_dir.resolve().as_posix()
        t0 = time.perf_counter()
        con.execute(f"""
            COPY (
                SELECT
                    e.subject_id, e.hadm_id, e.event_time, e.code_id,
                    e.timestamp_days, e.age_at_event_days, e.sex, e.race,
                    c.label, c.last_event_idx,
                    CAST(abs(hash(e.subject_id)) % {int(n_buckets)} AS INTEGER) AS bucket,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.subject_id
                        ORDER BY e.timestamp_days, e.event_time, e.code_id
                    ) - 1 AS event_idx
                FROM read_parquet('{ev}') e
                JOIN read_parquet('{co}') c USING (subject_id)
                QUALIFY event_idx <= c.last_event_idx
                ORDER BY bucket, e.subject_id, event_idx
            ) TO '{out}'
              (FORMAT PARQUET, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE, COMPRESSION ZSTD)
        """)
        wall = time.perf_counter() - t0
    finally:
        con.close()

    buckets = sorted(
        int(p.name.split("=", 1)[1])
        for p in part_dir.iterdir()
        if p.is_dir() and p.name.startswith("bucket=")
    )
    return {
        "n_buckets_requested": int(n_buckets),
        "n_buckets_nonempty": len(buckets),
        "buckets": buckets,
        "wall_s": float(wall),
        "part_dir": str(part_dir),
    }


def _tensorize_pic_bucket(bucket_dir: Path, out_npz: Path, vocab_path: Path,
                          max_seq_len: int) -> dict[str, Any]:
    import pyarrow.dataset as ds

    t0 = time.perf_counter()
    with Path(vocab_path).open("r", encoding="utf-8") as f:
        code_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    unk = len(code_vocab)

    dataset = ds.dataset(str(bucket_dir), format="parquet")
    cols = ["subject_id", "hadm_id", "code_id", "timestamp_days", "age_at_event_days",
            "sex", "race", "label", "event_idx"]
    table = dataset.to_table(columns=cols)
    if table.num_rows == 0:
        return {"out": str(out_npz), "n_rows": 0, "peak_rss_mb": _peak_rss_mb(),
                "wall_s": 0.0, "skipped": True}

    df = table.to_pandas()
    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    labels: list[float] = []
    sexs: list[int] = []
    races: list[int] = []
    n_events_window: list[int] = []
    code_list: list[np.ndarray] = []
    ts_list: list[np.ndarray] = []
    age_list: list[np.ndarray] = []

    for sid, g in df.groupby("subject_id", sort=True):
        g = g.sort_values("event_idx", kind="mergesort")
        code_ids = g["code_id"].astype(str).to_numpy()
        timestamps = g["timestamp_days"].astype(np.float32).to_numpy()
        ages = g["age_at_event_days"].astype(np.float32).to_numpy()
        n_evt = int(code_ids.shape[0])
        if n_evt > max_seq_len:
            sl = slice(-max_seq_len, None)
            code_ids, timestamps, ages = code_ids[sl], timestamps[sl], ages[sl]
        code_indices = np.fromiter(
            (code_vocab.get(str(c), unk) for c in code_ids),
            dtype=np.int64, count=len(code_ids),
        )
        hadm_val = g["hadm_id"].iloc[-1]
        hadm_id = int(hadm_val) if hadm_val == hadm_val and hadm_val is not None else -1
        subject_ids.append(int(sid))
        hadm_ids.append(hadm_id)
        labels.append(float(g["label"].iloc[0]))
        sexs.append(encode_sex(g["sex"].iloc[0]))
        races.append(encode_race(g["race"].iloc[0]))
        n_events_window.append(n_evt)
        code_list.append(code_indices)
        ts_list.append(timestamps)
        age_list.append(ages)

    if not subject_ids:
        return {"out": str(out_npz), "n_rows": 0, "peak_rss_mb": _peak_rss_mb(),
                "wall_s": float(time.perf_counter() - t0), "skipped": True}

    seq_len = np.asarray([len(c) for c in code_list], dtype=np.int64)
    offsets = np.zeros(len(code_list) + 1, dtype=np.int64)
    np.cumsum(seq_len, out=offsets[1:])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        subject_id=np.asarray(subject_ids, dtype=np.int64),
        hadm_id=np.asarray(hadm_ids, dtype=np.int64),
        label=np.asarray(labels, dtype=np.float32),
        sex=np.asarray(sexs, dtype=np.int8),
        race=np.asarray(races, dtype=np.int16),
        n_events_in_window=np.asarray(n_events_window, dtype=np.int64),
        unk_vocab_index=np.asarray([unk], dtype=np.int64),
        offsets=offsets,
        code_indices=np.concatenate(code_list).astype(np.int64),
        timestamps_days=np.concatenate(ts_list).astype(np.float32),
        age_days=np.concatenate(age_list).astype(np.float32),
    )
    return {
        "out": str(out_npz),
        "n_rows": len(subject_ids),
        "peak_rss_mb": _peak_rss_mb(),
        "wall_s": float(time.perf_counter() - t0),
        "skipped": False,
    }


def _pic_job(args: tuple[str, str, str, int, int]) -> dict[str, Any]:
    bucket_dir, out_npz, vocab_path, bucket_id, max_seq_len = args
    r = _tensorize_pic_bucket(Path(bucket_dir), Path(out_npz), Path(vocab_path), max_seq_len)
    r["bucket_id"] = int(bucket_id)
    return r


def rebuild_pic_task(
    cohort_dir: Path,
    events_parquet: Path,
    out_dir: Path,
    vocab_path: Path,
    *,
    n_buckets: int = 64,
    max_workers: int = 14,
    max_seq_len: int = 1024,
    duckdb_mem: str = "4GB",
) -> dict[str, Any]:
    cohort_dir = Path(cohort_dir)
    out_dir = Path(out_dir)
    workers = recommend_workers(max_workers)
    split_summaries = {}
    t_all = time.perf_counter()

    for split in ("train", "val", "test"):
        cohort = cohort_dir / f"{split}_cohort.parquet"
        if not cohort.exists():
            raise FileNotFoundError(cohort)
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        for old in split_out.glob("shard_*.npz"):
            old.unlink()
        part_dir = out_dir / "_parts" / split
        part_meta = _partition_pic_events(
            events_parquet, cohort, part_dir, n_buckets=n_buckets, duckdb_mem=duckdb_mem)

        jobs = [
            (str(part_dir / f"bucket={b}"),
             str(split_out / f"shard_{b:05d}.npz"),
             str(vocab_path), int(b), int(max_seq_len))
            for b in part_meta["buckets"]
        ]
        results: list[dict[str, Any]] = []
        if workers == 1:
            results = [_pic_job(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_pic_job, j) for j in jobs]
                for fut in as_completed(futs):
                    results.append(fut.result())

        kept = []
        for r in results:
            if r.get("skipped") or r.get("n_rows", 0) == 0:
                p = Path(r["out"])
                if p.exists():
                    p.unlink()
            else:
                kept.append(r)
        kept.sort(key=lambda r: int(r["bucket_id"]))
        split_summaries[split] = {
            "n_rows": int(sum(r["n_rows"] for r in kept)),
            "n_shards": len(kept),
            "partition_wall_s": part_meta["wall_s"],
            "peak_rss_mb_worker_max": float(np.nanmax([r["peak_rss_mb"] for r in kept]))
            if kept else float("nan"),
        }
        D.write_json(split_out / "rebuild_manifest.json",
                     {"split": split, **split_summaries[split], "partition": part_meta})

    summary = {
        "out_dir": str(out_dir),
        "events_parquet": str(events_parquet),
        "cohort_dir": str(cohort_dir),
        "n_buckets": int(n_buckets),
        "max_workers_used": int(workers),
        "mem_available_gb_at_start": _mem_available_gb(),
        "wall_s": float(time.perf_counter() - t_all),
        "splits": split_summaries,
    }
    D.write_json(out_dir / "rebuild_all_manifest.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rebuild PIC fine-tune shards (Phase 4).")
    p.add_argument("--pic_ft_root", type=Path,
                   default=REPO_ROOT / "data/processed/pic/finetune")
    p.add_argument("--out_root", type=Path,
                   default=REPO_ROOT / "data/tensorized/pic_horizon")
    p.add_argument("--vocab_path", type=Path,
                   default=REPO_ROOT / "data/processed/pic/code_vocab_pic.json")
    p.add_argument("--tasks", nargs="+",
                   default=["mortality", "los_gt7", "pneumonia", "heart_malformations"])
    p.add_argument("--n_buckets", type=int, default=64)
    p.add_argument("--max_workers", type=int, default=14)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--duckdb_mem", type=str, default="4GB")
    p.add_argument("--swap_into", type=Path, default=None,
                   help="Replace this root (default data/tensorized/pic) with --out_root.")
    args = p.parse_args(argv)

    workers = recommend_workers(args.max_workers)
    D.print_kv("tensorize_pic", {
        "out_root": str(args.out_root),
        "tasks": args.tasks,
        "max_workers_requested": args.max_workers,
        "max_workers_used": workers,
        "mem_available_gb": round(_mem_available_gb(), 2),
    })

    all_sum = []
    for task in args.tasks:
        cohort_dir = args.pic_ft_root / task / "cohort"
        events = args.pic_ft_root / task / "events.parquet"
        out_dir = args.out_root / task
        summary = rebuild_pic_task(
            cohort_dir, events, out_dir, args.vocab_path,
            n_buckets=args.n_buckets, max_workers=args.max_workers,
            max_seq_len=args.max_seq_len, duckdb_mem=args.duckdb_mem,
        )
        all_sum.append({"task": task, **summary})
        D.print_kv(f"rebuild pic/{task}", {
            "wall_s": round(summary["wall_s"], 1),
            "workers": summary["max_workers_used"],
            **{f"{s}_rows": summary["splits"][s]["n_rows"] for s in summary["splits"]},
        })

    if args.swap_into is not None:
        import shutil
        target = Path(args.swap_into)
        backup = target.with_name(target.name + "_pre_horizon")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        Path(args.out_root).rename(target)
        D.print_kv("swap_pic", {"live": str(target), "backup": str(backup)})

    D.write_json((args.swap_into or args.out_root) / "rebuild_pic_manifest.json",
                 {"tasks": all_sum})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
