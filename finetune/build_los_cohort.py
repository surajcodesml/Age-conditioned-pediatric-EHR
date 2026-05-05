#!/usr/bin/env python3
"""Build admission-anchored LOS>threshold cohorts from MIMIC-IV admissions."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LOS>threshold fine-tuning cohorts.")
    parser.add_argument("--observation_hours", type=int, default=24)
    parser.add_argument("--los_threshold_days", type=float, default=7.0)
    parser.add_argument("--min_events_in_window", type=int, default=5)
    parser.add_argument("--mimic_root", type=Path, required=True, help="Path to MIMIC-IV hosp dir containing admissions.csv.gz")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        float(np.percentile(values, 95)),
    )


def build_split_cohort(
    con: duckdb.DuckDBPyConnection,
    admissions_path: Path,
    rolled_path: Path,
    split_events_path: Path,
    out_path: Path,
    observation_hours: int,
    los_threshold_days: float,
    min_events_in_window: int,
) -> dict[str, float | int]:
    admissions_sql = _esc(admissions_path)
    rolled_sql = _esc(rolled_path)
    split_sql = _esc(split_events_path)
    out_sql = _esc(out_path)

    observation_interval = f"{int(observation_hours)} hours"
    obs_days = float(observation_hours) / 24.0

    per_admission = con.execute(
        f"""
        WITH split_subjects AS (
            SELECT DISTINCT CAST(subject_id AS BIGINT) AS subject_id
            FROM read_parquet('{split_sql}')
        ),
        split_admissions AS (
            SELECT
                CAST(a.subject_id AS BIGINT) AS subject_id,
                CAST(a.hadm_id AS BIGINT) AS hadm_id,
                CAST(a.admittime AS TIMESTAMP) AS admittime,
                CAST(a.dischtime AS TIMESTAMP) AS dischtime,
                (epoch(CAST(a.dischtime AS TIMESTAMP)) - epoch(CAST(a.admittime AS TIMESTAMP))) / 86400.0 AS los_days,
                CAST(a.admittime AS TIMESTAMP) + INTERVAL '{observation_interval}' AS landmark_time
            FROM read_csv_auto('{admissions_sql}', header=true, sample_size=-1) a
            JOIN split_subjects s ON CAST(a.subject_id AS BIGINT) = s.subject_id
            WHERE a.admittime IS NOT NULL
              AND a.dischtime IS NOT NULL
              AND CAST(a.dischtime AS TIMESTAMP) > CAST(a.admittime AS TIMESTAMP)
        ),
        ordered_events AS (
            SELECT
                CAST(e.subject_id AS BIGINT) AS subject_id,
                CAST(e.event_time AS TIMESTAMP) AS event_time,
                CAST(e.timestamp_days AS DOUBLE) AS timestamp_days,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(e.subject_id AS BIGINT)
                    ORDER BY e.timestamp_days, e.event_time, e.code_id
                ) - 1 AS event_idx
            FROM read_parquet('{rolled_sql}') e
            JOIN split_subjects s ON CAST(e.subject_id AS BIGINT) = s.subject_id
        )
        SELECT
            a.subject_id,
            a.hadm_id,
            a.admittime,
            a.dischtime,
            a.landmark_time,
            a.los_days,
            SUM(
                CASE
                    WHEN o.event_time >= a.admittime
                     AND o.event_time <= a.landmark_time
                    THEN 1 ELSE 0
                END
            )::INTEGER AS n_events_in_window,
            MAX(CASE WHEN o.event_time <= a.landmark_time THEN o.event_idx END)::BIGINT AS last_event_idx,
            MAX(CASE WHEN o.event_time <= a.landmark_time THEN o.timestamp_days END)::DOUBLE AS t_landmark_days
        FROM split_admissions a
        LEFT JOIN ordered_events o USING (subject_id)
        GROUP BY
            a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.landmark_time, a.los_days
        ORDER BY a.subject_id, a.hadm_id
        """
    ).df()

    if per_admission.empty:
        raise RuntimeError(f"No admissions found for split parquet: {split_events_path}")

    los_days = pd.to_numeric(per_admission["los_days"], errors="coerce").to_numpy(dtype=np.float64)
    n_events_in_window = pd.to_numeric(per_admission["n_events_in_window"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    last_event_idx = pd.to_numeric(per_admission["last_event_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    t_landmark_days = pd.to_numeric(per_admission["t_landmark_days"], errors="coerce").to_numpy(dtype=np.float64)

    remaining = np.ones(len(per_admission), dtype=bool)

    short_admission = los_days <= obs_days
    dropped_short_admission = int(np.sum(remaining & short_admission))
    remaining &= ~short_admission

    too_few_events = n_events_in_window < int(min_events_in_window)
    dropped_too_few_events = int(np.sum(remaining & too_few_events))
    remaining &= ~too_few_events

    valid_landmark = np.isfinite(t_landmark_days) & (last_event_idx >= 0)
    # Keep this folded into the same drop bucket to preserve the requested drop schema.
    dropped_too_few_events += int(np.sum(remaining & ~valid_landmark))
    remaining &= valid_landmark

    labels = (los_days > float(los_threshold_days)).astype(np.int8)
    out = pd.DataFrame(
        {
            "subject_id": per_admission["subject_id"].astype(np.int64),
            "hadm_id": per_admission["hadm_id"].astype(np.int64),
            "label": labels,
            "last_event_idx": last_event_idx.astype(np.int64),
            "t_landmark_days": t_landmark_days.astype(np.float32),
            "los_days": los_days.astype(np.float32),
            "n_events_in_window": n_events_in_window.astype(np.int32),
            "observation_hours": np.full(len(per_admission), int(observation_hours), dtype=np.int32),
            "los_threshold_days": np.full(len(per_admission), int(los_threshold_days), dtype=np.int32),
        }
    )
    out = out.loc[remaining].sort_values(["subject_id", "hadm_id"], kind="mergesort").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("LOS cohort is empty after filtering.")

    con.register("cohort_out_df", out)
    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    subject_id,
                    hadm_id,
                    label,
                    last_event_idx,
                    t_landmark_days,
                    los_days,
                    n_events_in_window,
                    observation_hours,
                    los_threshold_days
                FROM cohort_out_df
                ORDER BY subject_id, hadm_id
            ) TO '{out_sql}' (FORMAT PARQUET)
            """
        )
    finally:
        con.unregister("cohort_out_df")

    pos = out.loc[out["label"] == 1]
    neg = out.loc[out["label"] == 0]
    n_pos = int(len(pos))
    n_neg = int(len(neg))
    n_admissions = int(len(out))
    prevalence = float(n_pos / max(n_pos + n_neg, 1))
    n_unique_subjects = int(out["subject_id"].nunique())

    los_mean, los_median, los_p25, los_p75, los_p95 = _stats(out["los_days"].to_numpy(dtype=np.float64))
    pos_len = pos["n_events_in_window"].to_numpy(dtype=np.float64)
    neg_len = neg["n_events_in_window"].to_numpy(dtype=np.float64)
    pos_len_mean = float(np.mean(pos_len)) if pos_len.size > 0 else float("nan")
    pos_len_median = float(np.median(pos_len)) if pos_len.size > 0 else float("nan")
    neg_len_mean = float(np.mean(neg_len)) if neg_len.size > 0 else float("nan")
    neg_len_median = float(np.median(neg_len)) if neg_len.size > 0 else float("nan")
    len_ratio = float(neg_len_mean / pos_len_mean) if pos_len_mean > 0 else float("nan")

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_admissions": n_admissions,
        "prevalence": prevalence,
        "dropped_short_admission": dropped_short_admission,
        "dropped_too_few_events": dropped_too_few_events,
        "los_mean": los_mean,
        "los_median": los_median,
        "los_p25": los_p25,
        "los_p75": los_p75,
        "los_p95": los_p95,
        "pos_len_mean": pos_len_mean,
        "pos_len_median": pos_len_median,
        "neg_len_mean": neg_len_mean,
        "neg_len_median": neg_len_median,
        "len_ratio": len_ratio,
        "n_unique_subjects": n_unique_subjects,
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    admissions_path = args.mimic_root / "admissions.csv.gz"
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

    if not admissions_path.exists():
        raise FileNotFoundError(f"Missing admissions CSV: {admissions_path}")
    if not rolled_path.exists():
        raise FileNotFoundError(f"Missing rolled events parquet: {rolled_path}")
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing split parquet for {name}: {path}")

    tmp_dir = args.out_dir / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # No sampling is used, but keep deterministic RNG initialization for parity with other cohort scripts.
    _rng = np.random.default_rng(42)
    del _rng

    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit='24GB'")
        con.execute(f"PRAGMA temp_directory='{_esc(tmp_dir)}'")
        con.execute("PRAGMA threads=14")
        con.execute("SET preserve_insertion_order=false")

        print(f"[task] los_gt{int(args.los_threshold_days)}")
        print(f"[admissions] {admissions_path}")
        print(f"[rolled] {rolled_path}")
        print(f"[out_dir] {args.out_dir}")
        print(
            f"[config] observation_hours={args.observation_hours} "
            f"los_threshold_days={args.los_threshold_days} min_events_in_window={args.min_events_in_window}"
        )

        split_subject_sets: dict[str, set[int]] = {}
        for split in ("train", "val", "test"):
            split_subject_sets[split] = set(
                con.execute(
                    "SELECT DISTINCT CAST(subject_id AS BIGINT) FROM read_parquet(?)",
                    [str(split_paths[split].resolve())],
                )
                .df()
                .iloc[:, 0]
                .astype(np.int64)
                .tolist()
            )

            stats = build_split_cohort(
                con=con,
                admissions_path=admissions_path,
                rolled_path=rolled_path,
                split_events_path=split_paths[split],
                out_path=out_paths[split],
                observation_hours=args.observation_hours,
                los_threshold_days=args.los_threshold_days,
                min_events_in_window=args.min_events_in_window,
            )

            print(
                f"[{split}] n_pos={int(stats['n_pos']):,} n_neg={int(stats['n_neg']):,} "
                f"n_admissions={int(stats['n_admissions']):,} prevalence={float(stats['prevalence']):.4%}"
            )
            print(
                f"[{split}:drops] dropped_short_admission={int(stats['dropped_short_admission']):,} "
                f"dropped_too_few_events={int(stats['dropped_too_few_events']):,}"
            )
            print(
                f"[{split}:los_days] mean={float(stats['los_mean']):.2f} median={float(stats['los_median']):.2f} "
                f"p25={float(stats['los_p25']):.2f} p75={float(stats['los_p75']):.2f} p95={float(stats['los_p95']):.2f}"
            )
            print(
                f"[{split}:len_stats] pos mean={float(stats['pos_len_mean']):.2f} "
                f"median={float(stats['pos_len_median']):.2f} | neg mean={float(stats['neg_len_mean']):.2f} "
                f"median={float(stats['neg_len_median']):.2f} | neg/pos ratio={float(stats['len_ratio']):.4f}"
            )
            print(
                f"[{split}:subjects_vs_admissions] unique_subjects={int(stats['n_unique_subjects']):,} "
                f"admissions={int(stats['n_admissions']):,}"
            )

        for a_name, b_name in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = split_subject_sets[a_name].intersection(split_subject_sets[b_name])
            if overlap:
                raise RuntimeError(f"Split leakage detected between {a_name} and {b_name}: {len(overlap)} overlapping subject_ids")
        print("[split_check] patient-level split assignments are mutually exclusive")

        print("Wrote:")
        for split in ("train", "val", "test"):
            print(f"  - {out_paths[split]}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
