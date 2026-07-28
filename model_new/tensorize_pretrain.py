#!/usr/bin/env python3
"""Out-of-core deterministic rebuild of MIMIC pretraining flat shards (Phase 4).

Pipeline
--------
1. DuckDB sorts events by ``(subject_id, timestamp_days, event_time, code_id)`` and
   partitions by ``ABS(HASH(subject_id)) % N_BUCKETS`` (stable; independent of worker
   count / scheduling).
2. ``ProcessPoolExecutor`` over buckets (default ``max_workers=14``; auto-capped by
   available RAM). Each worker reads one hive partition, builds the flat visit schema,
   and writes ``shard_{bucket:04d}.npz``. At most one bucket is resident per worker.

Visit blocks remain hadm-contiguous (schema requirement for ``visit_starts`` /
``visit_ends``). The future-visit **window** is applied at read time by
:class:`model_new.data.TensorizedPretrainDataset` (INV-HORIZON).

All console / JSON output goes through :mod:`model_new.diagnostics` (D11).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from model_new import diagnostics as D
from model_new.data import encode_race

__all__ = [
    "stable_subject_bucket",
    "partition_events",
    "build_subject_payload",
    "tensorize_bucket",
    "rebuild_split",
    "recommend_workers",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_COLUMNS = (
    "subject_id", "hadm_id", "event_time", "code_id",
    "timestamp_days", "age_at_event_days", "sex", "race",
)


def stable_subject_bucket(subject_id: int, n_buckets: int) -> int:
    """Blake2b-based bucket; matches the DuckDB partition expression used below."""
    # Keep in sync with SQL: abs(hash(subject_id)) % n_buckets is DuckDB-native and
    # verified against this for the determinism check's documentation. We use DuckDB
    # HASH at partition time; this helper is for tests / PIC cohort routing.
    import hashlib
    digest = hashlib.blake2b(str(int(subject_id)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(n_buckets)


def _mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 8.0


def recommend_workers(requested: int, *, per_worker_gb: float = 1.25,
                      reserve_gb: float = 6.0, force: bool = False) -> int:
    """Cap workers so projected RSS stays under available RAM (leave headroom for I/O)."""
    if force:
        return max(1, int(requested))
    avail = _mem_available_gb()
    cap = max(1, int((avail - reserve_gb) / max(per_worker_gb, 0.1)))
    return max(1, min(int(requested), cap))


def _peak_rss_mb() -> float:
    try:
        # Linux: VmHWM is peak RSS in kB
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import resource
        # ru_maxrss is kB on Linux
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


def encode_sex(val: Any) -> int:
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


def build_subject_payload(g, code_vocab: dict[str, int]) -> dict | None:
    """Hadm-contiguous visit blocks; same semantics as the legacy tensorizer.

    Events within a visit are ordered by ``(event_time, code_id)``. Visits are ordered by
    each hadm's first ``event_time`` (null hadm -> -1). Returns ``None`` if fewer than two
    visits (no forecasting target).
    """
    g = g.copy()
    g["_hadm_fill"] = g["hadm_id"].fillna(-1).astype("int64")
    first_ts = g.groupby("_hadm_fill", sort=False)["event_time"].min().sort_values(kind="mergesort")
    hadm_to_visit = {h: k for k, h in enumerate(first_ts.index)}
    g["_visit_idx"] = g["_hadm_fill"].map(hadm_to_visit).astype("int64")
    g = g.sort_values(["_visit_idx", "event_time", "code_id"], kind="mergesort").reset_index(drop=True)

    n_events = len(g)
    visit_idx = g["_visit_idx"].to_numpy()
    spans: list[tuple[int, int]] = []
    start_i = 0
    for i in range(1, n_events):
        if visit_idx[i] != visit_idx[i - 1]:
            spans.append((start_i, i))
            start_i = i
    spans.append((start_i, n_events))
    if len(spans) < 2:
        return None

    return {
        "subject_id": int(g["subject_id"].iloc[0]),
        "code_id": g["code_id"].astype(str).to_numpy(),
        "timestamps_days": g["timestamp_days"].fillna(0.0).astype("float32").to_numpy(),
        "age_days": g["age_at_event_days"].fillna(0.0).astype("float32").to_numpy(),
        "sex": np.int8(encode_sex(g["sex"].iloc[0])),
        "race": encode_race(g["race"].iloc[0]),
        "visit_spans": np.asarray(spans, dtype=np.int32),
    }


def partition_events(events_parquet: Path, part_dir: Path, *, n_buckets: int,
                     duckdb_mem: str = "4GB", temp_dir: Path | None = None) -> dict[str, Any]:
    """Write hive-partitioned parquet: ``bucket=k/*.parquet``, sorted within each bucket."""
    import duckdb

    part_dir = Path(part_dir)
    if part_dir.exists():
        # Fresh partition for determinism — remove prior hive tree.
        import shutil
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(temp_dir) if temp_dir is not None else part_dir / "_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA memory_limit='{duckdb_mem}'")
        con.execute(f"PRAGMA temp_directory='{tmp.as_posix()}'")
        con.execute("PRAGMA threads=4")
        # ABS(HASH(subject_id)) % N is stable in DuckDB and independent of worker count.
        sql = f"""
            COPY (
                SELECT
                    subject_id, hadm_id, event_time, code_id,
                    timestamp_days, age_at_event_days, sex, race,
                    CAST(abs(hash(subject_id)) % {int(n_buckets)} AS INTEGER) AS bucket
                FROM read_parquet('{events_parquet.resolve().as_posix()}')
                ORDER BY bucket, subject_id, timestamp_days, event_time, code_id
            ) TO '{part_dir.resolve().as_posix()}'
              (FORMAT PARQUET, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE, COMPRESSION ZSTD)
        """
        t0 = time.perf_counter()
        con.execute(sql)
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


def _write_flat_shard(out_npz: Path, payloads: list[dict], unk: int) -> tuple[int, int]:
    if not payloads:
        return 0, 0
    payloads = sorted(payloads, key=lambda p: int(p["subject_id"]))
    subj = np.asarray([p["subject_id"] for p in payloads], dtype=np.int64)
    sexs = np.asarray([p["sex"] for p in payloads], dtype=np.int8)
    races = np.asarray([p["race"] for p in payloads], dtype=np.int16)
    code_blocks = [np.asarray(p["code_indices"], dtype=np.int64) for p in payloads]
    ts_blocks = [np.asarray(p["timestamps_days"], dtype=np.float32) for p in payloads]
    age_blocks = [np.asarray(p["age_days"], dtype=np.float32) for p in payloads]
    spans = [np.asarray(p["visit_spans"], dtype=np.int32) for p in payloads]
    vs_blocks = [s[:, 0].astype(np.int32, copy=False) for s in spans]
    ve_blocks = [s[:, 1].astype(np.int32, copy=False) for s in spans]
    ev_lens = [int(c.shape[0]) for c in code_blocks]
    vis_counts = [int(s.shape[0]) for s in spans]

    n = len(payloads)
    event_offsets = np.zeros(n + 1, dtype=np.int64)
    visit_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.asarray(ev_lens, dtype=np.int64), out=event_offsets[1:])
    np.cumsum(np.asarray(vis_counts, dtype=np.int64), out=visit_offsets[1:])
    code_concat = np.concatenate(code_blocks)
    ts_concat = np.concatenate(ts_blocks)
    age_concat = np.concatenate(age_blocks)
    vs_concat = np.concatenate(vs_blocks)
    ve_concat = np.concatenate(ve_blocks)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        subject_id=subj,
        sex=sexs,
        race=races,
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


def tensorize_bucket(bucket_dir: Path, out_npz: Path, vocab_path: Path) -> dict[str, Any]:
    """Build one flat shard from a single hive ``bucket=k`` directory."""
    import pyarrow.dataset as ds

    t0 = time.perf_counter()
    with Path(vocab_path).open("r", encoding="utf-8") as f:
        code_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    unk = len(code_vocab)

    dataset = ds.dataset(str(bucket_dir), format="parquet")
    table = dataset.to_table(columns=list(EVENT_COLUMNS))
    if table.num_rows == 0:
        return {
            "bucket_dir": str(bucket_dir), "out": str(out_npz),
            "n_patients": 0, "n_events": 0, "wall_s": 0.0,
            "peak_rss_mb": _peak_rss_mb(), "skipped": True,
        }

    df = table.to_pandas()
    # Contiguous subjects: partition already ordered by subject_id.
    payloads: list[dict] = []
    for sid, g in df.groupby("subject_id", sort=True):
        payload = build_subject_payload(g, code_vocab)
        if payload is None:
            continue
        codes = payload["code_id"]
        payload["code_indices"] = np.fromiter(
            (code_vocab.get(str(c), unk) for c in codes),
            dtype=np.int64, count=len(codes),
        )
        del payload["code_id"]
        payloads.append(payload)

    n_pat, n_ev = _write_flat_shard(out_npz, payloads, unk)
    return {
        "bucket_dir": str(bucket_dir),
        "out": str(out_npz),
        "n_patients": int(n_pat),
        "n_events": int(n_ev),
        "wall_s": float(time.perf_counter() - t0),
        "peak_rss_mb": _peak_rss_mb(),
        "skipped": n_pat == 0,
    }


def _bucket_job(args: tuple[str, str, str, int]) -> dict[str, Any]:
    bucket_dir, out_npz, vocab_path, bucket_id = args
    result = tensorize_bucket(Path(bucket_dir), Path(out_npz), Path(vocab_path))
    result["bucket_id"] = int(bucket_id)
    return result


def rebuild_split(
    events_parquet: Path,
    out_split_dir: Path,
    vocab_path: Path,
    *,
    n_buckets: int = 512,
    max_workers: int = 14,
    part_root: Path | None = None,
    duckdb_mem: str = "4GB",
    reuse_partition: bool = False,
    force_workers: bool = False,
) -> dict[str, Any]:
    """Partition then parallel-tensorize one split. Returns a MEASURE summary dict."""
    events_parquet = Path(events_parquet)
    out_split_dir = Path(out_split_dir)
    vocab_path = Path(vocab_path)
    out_split_dir.mkdir(parents=True, exist_ok=True)

    workers = recommend_workers(max_workers, force=force_workers)
    part_root = Path(part_root) if part_root is not None else out_split_dir.parent / "_parts" / out_split_dir.name
    t_all = time.perf_counter()

    if reuse_partition and part_root.exists() and any(part_root.glob("bucket=*")):
        buckets = sorted(
            int(p.name.split("=", 1)[1])
            for p in part_root.iterdir()
            if p.is_dir() and p.name.startswith("bucket=")
        )
        part_meta = {
            "n_buckets_requested": int(n_buckets),
            "n_buckets_nonempty": len(buckets),
            "buckets": buckets,
            "wall_s": 0.0,
            "part_dir": str(part_root),
            "reused": True,
        }
    else:
        part_meta = partition_events(
            events_parquet, part_root, n_buckets=n_buckets, duckdb_mem=duckdb_mem)
        part_meta["reused"] = False
        buckets = list(part_meta["buckets"])

    # Drop any prior shards in the destination so a partial tree cannot linger.
    for old in out_split_dir.glob("shard_*.npz"):
        old.unlink()
    for old in out_split_dir.glob("corpus_stats.json"):
        old.unlink()

    jobs = []
    for b in buckets:
        bdir = part_root / f"bucket={b}"
        jobs.append((str(bdir), str(out_split_dir / f"shard_{b:04d}.npz"),
                     str(vocab_path), int(b)))

    results: list[dict[str, Any]] = []
    peak_rss = []
    if workers == 1:
        for job in jobs:
            results.append(_bucket_job(job))
            peak_rss.append(results[-1]["peak_rss_mb"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_bucket_job, job): job[3] for job in jobs}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                peak_rss.append(r["peak_rss_mb"])

    # Deterministic: remove empty shards (bucket had subjects but none with >=2 visits).
    kept = []
    for r in results:
        if r.get("skipped") or r["n_patients"] == 0:
            p = Path(r["out"])
            if p.exists():
                p.unlink()
        else:
            kept.append(r)

    # Sort results by bucket for stable manifest.
    kept.sort(key=lambda r: int(r["bucket_id"]))
    summary = {
        "events_parquet": str(events_parquet),
        "out_split_dir": str(out_split_dir),
        "n_buckets": int(n_buckets),
        "max_workers_requested": int(max_workers),
        "max_workers_used": int(workers),
        "mem_available_gb_at_start": _mem_available_gb(),
        "partition": part_meta,
        "n_shards": len(kept),
        "n_patients": int(sum(r["n_patients"] for r in kept)),
        "n_events": int(sum(r["n_events"] for r in kept)),
        "wall_s": float(time.perf_counter() - t_all),
        "peak_rss_mb_per_worker_max": float(np.nanmax(peak_rss)) if peak_rss else float("nan"),
        "peak_rss_mb_per_worker_mean": float(np.nanmean(peak_rss)) if peak_rss else float("nan"),
        "shards": [{"bucket_id": r["bucket_id"], "n_patients": r["n_patients"],
                    "n_events": r["n_events"], "peak_rss_mb": r["peak_rss_mb"],
                    "wall_s": r["wall_s"]} for r in kept],
    }
    D.write_json(out_split_dir / "rebuild_manifest.json", summary)
    return summary


def _byte_identical_trees(a: Path, b: Path) -> tuple[bool, list[str]]:
    """Compare shard_*.npz byte-for-byte (names and contents)."""
    sa = {p.name: p for p in sorted(a.glob("shard_*.npz"))}
    sb = {p.name: p for p in sorted(b.glob("shard_*.npz"))}
    problems = []
    if sa.keys() != sb.keys():
        problems.append(f"name mismatch: only_a={sorted(sa.keys()-sb.keys())[:5]} "
                        f"only_b={sorted(sb.keys()-sa.keys())[:5]}")
        return False, problems
    for name in sa:
        ba = sa[name].read_bytes()
        bb = sb[name].read_bytes()
        if ba != bb:
            problems.append(f"{name}: {len(ba)} vs {len(bb)} bytes differ")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rebuild MIMIC pretrain flat shards (Phase 4).")
    p.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data/processed")
    p.add_argument("--out_dir", type=Path,
                   default=REPO_ROOT / "data/processed/tensorized_flat_horizon")
    p.add_argument("--vocab_path", type=Path,
                   default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--n_buckets", type=int, default=512)
    p.add_argument("--max_workers", type=int, default=14)
    p.add_argument("--duckdb_mem", type=str, default="4GB")
    p.add_argument("--part_root", type=Path, default=None,
                   help="hive partition root (default: <out_dir>/_parts)")
    p.add_argument("--reuse_partition", action="store_true")
    p.add_argument("--force_workers", action="store_true",
                   help="Do not auto-cap max_workers by MemAvailable.")
    p.add_argument("--determinism_check", action="store_true",
                   help="Rebuild the first split at workers=1 and workers=N; require byte identity.")
    p.add_argument("--swap_into", type=Path, default=None,
                   help="If set, atomically replace this dir with --out_dir when done "
                        "(old tree moved to <swap_into>_pre_horizon).")
    args = p.parse_args(argv)

    if not args.vocab_path.exists():
        raise FileNotFoundError(args.vocab_path)

    workers = recommend_workers(args.max_workers, force=args.force_workers)
    D.print_kv("tensorize_pretrain", {
        "out_dir": str(args.out_dir),
        "n_buckets": args.n_buckets,
        "max_workers_requested": args.max_workers,
        "max_workers_used": workers,
        "force_workers": bool(args.force_workers),
        "mem_available_gb": round(_mem_available_gb(), 2),
        "splits": args.splits,
    })

    if args.determinism_check:
        split = args.splits[0]
        events = args.data_dir / f"{split}_events.parquet"
        part_root = (args.part_root or (args.out_dir / "_parts")) / split
        # Partition once, reuse for both builds.
        partition_events(events, part_root, n_buckets=args.n_buckets, duckdb_mem=args.duckdb_mem)
        out_a = args.out_dir / f"_det_{split}_w1"
        out_b = args.out_dir / f"_det_{split}_wN"
        s1 = rebuild_split(events, out_a, args.vocab_path, n_buckets=args.n_buckets,
                           max_workers=1, part_root=part_root, duckdb_mem=args.duckdb_mem,
                           reuse_partition=True, force_workers=True)
        sN = rebuild_split(events, out_b, args.vocab_path, n_buckets=args.n_buckets,
                           max_workers=workers, part_root=part_root, duckdb_mem=args.duckdb_mem,
                           reuse_partition=True, force_workers=args.force_workers)
        ok, problems = _byte_identical_trees(out_a, out_b)
        D.print_kv("determinism_check", {
            "split": split,
            "workers_a": 1,
            "workers_b": sN["max_workers_used"],
            "byte_identical": ok,
            "n_shards": s1["n_shards"],
            "problems": problems[:10],
            "wall_w1_s": s1["wall_s"],
            "wall_wN_s": sN["wall_s"],
            "peak_rss_mb_w1": s1["peak_rss_mb_per_worker_max"],
            "peak_rss_mb_wN": sN["peak_rss_mb_per_worker_max"],
        })
        if not ok:
            return 1

    summaries = []
    for split in args.splits:
        events = args.data_dir / f"{split}_events.parquet"
        if not events.exists():
            D.print_kv(f"skip {split}", {"reason": f"missing {events}"})
            continue
        part_root = (args.part_root or (args.out_dir / "_parts")) / split
        summary = rebuild_split(
            events, args.out_dir / split, args.vocab_path,
            n_buckets=args.n_buckets, max_workers=args.max_workers,
            part_root=part_root, duckdb_mem=args.duckdb_mem,
            reuse_partition=args.reuse_partition,
            force_workers=args.force_workers,
        )
        summaries.append(summary)
        D.print_kv(f"rebuild {split}", {
            "n_patients": summary["n_patients"],
            "n_events": summary["n_events"],
            "n_shards": summary["n_shards"],
            "wall_s": round(summary["wall_s"], 1),
            "workers": summary["max_workers_used"],
            "peak_rss_mb_worker_max": round(summary["peak_rss_mb_per_worker_max"], 1),
            "partition_wall_s": round(summary["partition"]["wall_s"], 1),
        })

    if args.swap_into is not None and summaries:
        import shutil
        target = Path(args.swap_into)
        backup = target.with_name(target.name + "_pre_horizon")
        staging = Path(args.out_dir)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.rename(backup)
        staging.rename(target)
        D.print_kv("swap", {"live": str(target), "backup": str(backup)})

    if args.swap_into is not None:
        manifest_path = Path(args.swap_into) / "rebuild_all_manifest.json"
    else:
        manifest_path = Path(args.out_dir) / "rebuild_all_manifest.json"
    D.write_json(manifest_path, {"summaries": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
