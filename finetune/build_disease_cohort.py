#!/usr/bin/env python3
"""Build binary disease prediction cohorts from rolled patient timelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disease cohorts for fine-tuning.")
    parser.add_argument("--disease", type=str, required=True, help="Disease short name (e.g., t2d, aki, hf).")
    parser.add_argument("--code_prefix", type=str, required=True, help="Disease code prefix to match on code_id.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def build_split_cohort(
    con: duckdb.DuckDBPyConnection,
    rolled_path: Path,
    split_events_path: Path,
    code_prefix: str,
    out_path: Path,
    rng: np.random.Generator,
) -> dict[str, int | float]:
    rolled_sql = _esc(rolled_path)
    split_sql = _esc(split_events_path)
    out_sql = _esc(out_path)

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE split_subjects AS
        SELECT DISTINCT subject_id
        FROM read_parquet(?)
        """,
        [split_sql],
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE patient_first_last AS
        WITH events AS (
            SELECT
                e.subject_id,
                e.code_id,
                ROW_NUMBER() OVER (
                    PARTITION BY e.subject_id
                    ORDER BY e.timestamp_days, e.event_time, e.code_id
                ) - 1 AS event_idx
            FROM read_parquet(?) e
            JOIN split_subjects s USING (subject_id)
        )
        SELECT
            subject_id,
            MIN(CASE WHEN starts_with(code_id, ?) THEN event_idx END) AS first_occurrence_idx,
            MAX(event_idx) AS last_idx
        FROM events
        GROUP BY subject_id
        """,
        [rolled_sql, code_prefix],
    )

    positives = con.execute(
        """
        SELECT subject_id, (first_occurrence_idx - 1) AS last_event_idx
        FROM patient_first_last
        WHERE first_occurrence_idx > 0
        ORDER BY subject_id
        """
    ).df()
    negatives = con.execute(
        """
        SELECT subject_id, last_idx
        FROM patient_first_last
        WHERE first_occurrence_idx IS NULL
        ORDER BY subject_id
        """
    ).df()
    n_drop_first_event = int(
        con.execute("SELECT COUNT(*) FROM patient_first_last WHERE first_occurrence_idx = 0").fetchone()[0]
    )

    n_pos = int(len(positives))
    if n_pos == 0:
        raise RuntimeError("No positives in split; cannot construct matched negative truncation distribution.")

    # Drop negatives with too-short timelines (<5 events).
    neg_eligible = negatives.loc[negatives["last_idx"].astype(int) >= 4].copy()
    n_drop_short_neg = int(len(negatives) - len(neg_eligible))

    pos_ctx = positives["last_event_idx"].to_numpy(dtype=np.int64)
    pos_ctx_sorted = np.sort(pos_ctx)
    neg_last_idx = neg_eligible["last_idx"].to_numpy(dtype=np.int64)
    neg_lengths = neg_last_idx + 1

    # Quantile-matched sampling from the empirical positive CDF: t = F_pos^{-1}(q), q~U[0,1).
    q = rng.random(size=len(neg_eligible))
    q_idx = np.floor(q * len(pos_ctx_sorted)).astype(np.int64)
    q_idx = np.clip(q_idx, 0, len(pos_ctx_sorted) - 1)
    sampled_pos_ctx = pos_ctx_sorted[q_idx]

    # Keep only negatives that can realize sampled context without clamping.
    keep_mask = sampled_pos_ctx < neg_lengths
    n_drop_short_neg_unmatchable = int(np.sum(~keep_mask))
    neg_kept = neg_eligible.loc[keep_mask].copy()
    neg_assigned_last = sampled_pos_ctx[keep_mask]

    positives_out = pd.DataFrame(
        {
            "subject_id": positives["subject_id"].astype(np.int64),
            "label": np.ones(len(positives), dtype=np.int8),
            "last_event_idx": pos_ctx.astype(np.int64),
        }
    )
    negatives_out = pd.DataFrame(
        {
            "subject_id": neg_kept["subject_id"].astype(np.int64),
            "label": np.zeros(len(neg_kept), dtype=np.int8),
            "last_event_idx": neg_assigned_last.astype(np.int64),
        }
    )
    cohort_out = pd.concat([positives_out, negatives_out], axis=0, ignore_index=True).sort_values(
        "subject_id", kind="mergesort"
    )

    con.register("cohort_out_df", cohort_out)
    try:
        con.execute(
            f"""
            COPY (
                SELECT subject_id, label, last_event_idx
                FROM cohort_out_df
                ORDER BY subject_id
            ) TO '{out_sql}' (FORMAT PARQUET)
            """
        )
    finally:
        con.unregister("cohort_out_df")

    n_neg = int(len(negatives_out))

    con.execute("DROP TABLE IF EXISTS patient_first_last")
    con.execute("DROP TABLE IF EXISTS split_subjects")
    neg_lost_unmatchable_rate = float(n_drop_short_neg_unmatchable / max(len(negatives), 1))
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_neg_original": int(len(negatives)),
        "n_drop_first_event": n_drop_first_event,
        "n_drop_short_neg": n_drop_short_neg,
        "n_drop_short_neg_unmatchable": n_drop_short_neg_unmatchable,
        "neg_lost_unmatchable_rate": neg_lost_unmatchable_rate,
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rolled_path = args.data_dir / "patient_events_rolled_full.parquet"
    split_paths = {
        "train": args.data_dir / "train_events.parquet",
        "val": args.data_dir / "val_events.parquet",
        "test": args.data_dir / "test_events.parquet",
    }
    out_paths = {
        "train": args.out_dir / "train_cohort.parquet",
        "val": args.out_dir / "val_cohort.parquet",
        "test": args.out_dir / "test_cohort.parquet",
    }

    if not rolled_path.exists():
        raise FileNotFoundError(f"Missing rolled events parquet: {rolled_path}")
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split parquet for {name}: {path}")

    tmp_dir = args.out_dir / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit='4GB'")
        con.execute(f"PRAGMA temp_directory='{_esc(tmp_dir)}'")
        con.execute("PRAGMA threads=2")
        rng = np.random.default_rng(42)

        print(f"[disease] {args.disease}")
        print(f"[code_prefix] {args.code_prefix}")
        print(f"[rolled] {rolled_path}")
        print(f"[out_dir] {args.out_dir}")

        for split in ("train", "val", "test"):
            stats = build_split_cohort(
                con=con,
                rolled_path=rolled_path,
                split_events_path=split_paths[split],
                code_prefix=args.code_prefix,
                out_path=out_paths[split],
                rng=rng,
            )
            n_pos = int(stats["n_pos"])
            n_neg = int(stats["n_neg"])
            n_neg_original = int(stats["n_neg_original"])
            n_drop_first_event = int(stats["n_drop_first_event"])
            n_drop_short_neg = int(stats["n_drop_short_neg"])
            n_drop_short_neg_unmatchable = int(stats["n_drop_short_neg_unmatchable"])
            neg_lost_unmatchable_rate = float(stats["neg_lost_unmatchable_rate"])
            total = n_pos + n_neg
            prevalence = (n_pos / total) if total > 0 else 0.0
            print(
                f"[{split}] positives={n_pos:,} negatives={n_neg:,} "
                f"dropped_first_event={n_drop_first_event:,} dropped_short_neg={n_drop_short_neg:,} "
                f"dropped_short_neg_unmatchable={n_drop_short_neg_unmatchable:,} "
                f"neg_lost_unmatchable={neg_lost_unmatchable_rate:.2%} "
                f"neg_original={n_neg_original:,} prevalence={prevalence:.4%}"
            )
            ctx = con.execute(
                """
                SELECT
                    AVG(CASE WHEN label = 1 THEN last_event_idx + 1 END) AS pos_mean,
                    MEDIAN(CASE WHEN label = 1 THEN last_event_idx + 1 END) AS pos_median,
                    AVG(CASE WHEN label = 0 THEN last_event_idx + 1 END) AS neg_mean,
                    MEDIAN(CASE WHEN label = 0 THEN last_event_idx + 1 END) AS neg_median
                FROM read_parquet(?)
                """,
                [str(out_paths[split].resolve())],
            ).fetchone()
            pos_mean = float(ctx[0]) if ctx[0] is not None else float("nan")
            pos_median = float(ctx[1]) if ctx[1] is not None else float("nan")
            neg_mean = float(ctx[2]) if ctx[2] is not None else float("nan")
            neg_median = float(ctx[3]) if ctx[3] is not None else float("nan")
            ratio = (neg_mean / pos_mean) if pos_mean > 0 else float("nan")
            print(
                f"[{split}:context_len] pos mean={pos_mean:.2f} median={pos_median:.2f} | "
                f"neg mean={neg_mean:.2f} median={neg_median:.2f} | neg/pos ratio={ratio:.4f}"
            )

        print("Wrote:")
        for split in ("train", "val", "test"):
            print(f"  - {out_paths[split]}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
