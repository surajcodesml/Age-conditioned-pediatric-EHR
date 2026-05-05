#!/usr/bin/env python3
"""Build anchored landmark disease cohorts from rolled patient timelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anchored landmark cohorts for disease prediction.")
    parser.add_argument("--disease", type=str, required=True, help="Disease short name (e.g., t2d, aki, hf).")
    parser.add_argument("--code_prefix", type=str, required=True, help="Disease code prefix to match on code_id.")
    parser.add_argument("--obs_window_days", type=int, default=365)
    parser.add_argument("--gap_days", type=int, default=90)
    parser.add_argument("--pred_window_days", type=int, default=365)
    parser.add_argument("--min_obs_window_days", type=int, default=30)
    parser.add_argument("--min_events_in_window", type=int, default=5)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args()


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    )


def build_split_cohort(
    con: duckdb.DuckDBPyConnection,
    rolled_path: Path,
    split_events_path: Path,
    code_prefix: str,
    out_path: Path,
    obs_window_days: int,
    gap_days: int,
    pred_window_days: int,
    min_obs_window_days: int,
    min_events_in_window: int,
) -> dict[str, float | int]:
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

    per_subject = con.execute(
        """
        SELECT
            e.subject_id,
            MIN(CASE WHEN starts_with(e.code_id, ?) THEN CAST(e.timestamp_days AS DOUBLE) END) AS t_first_disease,
            MAX(CAST(e.timestamp_days AS DOUBLE)) AS t_max
        FROM read_parquet(?) e
        JOIN split_subjects s USING (subject_id)
        GROUP BY e.subject_id
        ORDER BY subject_id
        """,
        [code_prefix, rolled_sql],
    ).df()

    con.execute("DROP TABLE IF EXISTS split_subjects")

    if per_subject.empty:
        raise RuntimeError(f"No subjects found for split parquet: {split_events_path}")

    t_first_disease = pd.to_numeric(per_subject["t_first_disease"], errors="coerce").to_numpy(dtype=np.float64)
    t_max = pd.to_numeric(per_subject["t_max"], errors="coerce").to_numpy(dtype=np.float64)
    has_disease = np.isfinite(t_first_disease)
    n_subjects = len(per_subject)

    t_landmark = np.full(n_subjects, np.nan, dtype=np.float64)
    t_landmark[has_disease] = t_first_disease[has_disease] - float(gap_days) - float(pred_window_days)

    pos_landmarks_for_sampling = t_landmark[has_disease]
    if pos_landmarks_for_sampling.size == 0:
        raise RuntimeError("No positive subjects in split; cannot sample negative landmarks.")

    rng = np.random.default_rng(42)
    neg_mask = ~has_disease
    dropped_no_eligible_mask = np.zeros(n_subjects, dtype=bool)
    if np.any(neg_mask):
        horizon_days = float(gap_days + pred_window_days)
        neg_indices = np.flatnonzero(neg_mask)
        for idx in neg_indices:
            eligible_pool = pos_landmarks_for_sampling[(pos_landmarks_for_sampling + horizon_days) <= t_max[idx]]
            if eligible_pool.size == 0:
                dropped_no_eligible_mask[idx] = True
                continue
            t_landmark[idx] = float(rng.choice(eligible_pool))
    dropped_no_eligible_landmark = int(np.sum(dropped_no_eligible_mask))

    remaining = np.ones(n_subjects, dtype=bool)
    remaining &= ~dropped_no_eligible_mask

    landmark_ok = np.isfinite(t_landmark) & (t_landmark >= float(min_obs_window_days))
    dropped_no_landmark_room = int(np.sum(remaining & ~landmark_ok))
    remaining &= landmark_ok

    record_ok = t_max >= (t_landmark + float(gap_days + pred_window_days))
    dropped_short_record = int(np.sum(remaining & ~record_ok))
    remaining &= record_ok

    metric_subjects = pd.DataFrame(
        {
            "subject_id": per_subject.loc[remaining, "subject_id"].astype(np.int64),
            "t_landmark": t_landmark[remaining].astype(np.float64),
        }
    )
    n_events_in_window = np.zeros(n_subjects, dtype=np.int32)
    last_event_idx = np.full(n_subjects, -1, dtype=np.int64)
    if not metric_subjects.empty:
        con.register("metric_subjects_df", metric_subjects)
        try:
            metric_df = con.execute(
                f"""
                WITH ordered AS (
                    SELECT
                        e.subject_id,
                        CAST(e.timestamp_days AS DOUBLE) AS timestamp_days,
                        ROW_NUMBER() OVER (
                            PARTITION BY e.subject_id
                            ORDER BY e.timestamp_days, e.event_time, e.code_id
                        ) - 1 AS event_idx
                    FROM read_parquet('{rolled_sql}') e
                    JOIN metric_subjects_df m USING (subject_id)
                )
                SELECT
                    o.subject_id,
                    SUM(
                        CASE
                            WHEN o.timestamp_days >= (m.t_landmark - {float(obs_window_days)})
                             AND o.timestamp_days <= m.t_landmark
                            THEN 1 ELSE 0
                        END
                    )::INTEGER AS n_events_in_window,
                    MAX(CASE WHEN o.timestamp_days <= m.t_landmark THEN o.event_idx END)::BIGINT AS last_event_idx
                FROM ordered o
                JOIN metric_subjects_df m USING (subject_id)
                GROUP BY o.subject_id
                ORDER BY o.subject_id
                """
            ).df()
        finally:
            con.unregister("metric_subjects_df")

        metric_idx = pd.Series(np.arange(n_subjects, dtype=np.int64), index=per_subject["subject_id"].astype(np.int64))
        idx = metric_idx.reindex(metric_df["subject_id"].astype(np.int64)).to_numpy(dtype=np.int64)
        n_events_in_window[idx] = metric_df["n_events_in_window"].fillna(0).astype(np.int32).to_numpy()
        last_event_idx[idx] = metric_df["last_event_idx"].fillna(-1).astype(np.int64).to_numpy()

    enough_events = n_events_in_window >= int(min_events_in_window)
    dropped_too_few_events = int(np.sum(remaining & ~enough_events))
    remaining &= enough_events

    ambiguous = has_disease & (t_first_disease >= (t_landmark - float(obs_window_days))) & (
        t_first_disease < (t_landmark + float(gap_days))
    )
    dropped_ambiguous = int(np.sum(remaining & ambiguous))
    remaining &= ~ambiguous

    positive_label = has_disease & (t_first_disease >= (t_landmark + float(gap_days))) & (
        t_first_disease <= (t_landmark + float(gap_days + pred_window_days))
    )
    negative_label = ~has_disease
    labelable = positive_label | negative_label
    remaining &= labelable

    labels = np.where(positive_label, 1, 0).astype(np.int8)
    out = pd.DataFrame(
        {
            "subject_id": per_subject["subject_id"].astype(np.int64),
            "label": labels,
            "last_event_idx": last_event_idx.astype(np.int64),
            "t_landmark_days": t_landmark.astype(np.float32),
            "n_events_in_window": n_events_in_window.astype(np.int32),
            "obs_window_days": np.full(n_subjects, int(obs_window_days), dtype=np.int32),
            "gap_days": np.full(n_subjects, int(gap_days), dtype=np.int32),
            "pred_window_days": np.full(n_subjects, int(pred_window_days), dtype=np.int32),
        }
    )

    out = out.loc[remaining].sort_values("subject_id", kind="mergesort").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Cohort is empty after filtering.")
    if (out["last_event_idx"] < 0).any():
        raise RuntimeError("Found rows without events at or before landmark. Check eligibility logic.")

    con.register("cohort_out_df", out)
    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    subject_id,
                    label,
                    last_event_idx,
                    t_landmark_days,
                    n_events_in_window,
                    obs_window_days,
                    gap_days,
                    pred_window_days
                FROM cohort_out_df
                ORDER BY subject_id
            ) TO '{out_sql}' (FORMAT PARQUET)
            """
        )
    finally:
        con.unregister("cohort_out_df")

    pos = out.loc[out["label"] == 1]
    neg = out.loc[out["label"] == 0]
    n_pos = int(len(pos))
    n_neg = int(len(neg))
    prevalence = float(n_pos / max(n_pos + n_neg, 1))

    pos_len = pos["n_events_in_window"].to_numpy(dtype=np.float64)
    neg_len = neg["n_events_in_window"].to_numpy(dtype=np.float64)
    pos_len_mean, pos_len_median, pos_len_p25, pos_len_p75 = _stats(pos_len)
    neg_len_mean, neg_len_median, neg_len_p25, neg_len_p75 = _stats(neg_len)
    len_ratio = float(neg_len_mean / pos_len_mean) if pos_len_mean > 0 else float("nan")

    pos_lm = pos["t_landmark_days"].to_numpy(dtype=np.float64)
    neg_lm = neg["t_landmark_days"].to_numpy(dtype=np.float64)
    pos_lm_mean, pos_lm_median, _, _ = _stats(pos_lm)
    neg_lm_mean, neg_lm_median, _, _ = _stats(neg_lm)

    pos_mask_kept = remaining & positive_label
    disease_to_landmark = t_first_disease[pos_mask_kept] - t_landmark[pos_mask_kept]
    d2l_mean, d2l_median, d2l_p25, d2l_p75 = _stats(disease_to_landmark)
    d2l_min = float(np.min(disease_to_landmark)) if disease_to_landmark.size > 0 else float("nan")
    d2l_max = float(np.max(disease_to_landmark)) if disease_to_landmark.size > 0 else float("nan")

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prevalence,
        "dropped_no_eligible_landmark": dropped_no_eligible_landmark,
        "dropped_no_landmark_room": dropped_no_landmark_room,
        "dropped_short_record": dropped_short_record,
        "dropped_too_few_events": dropped_too_few_events,
        "dropped_ambiguous": dropped_ambiguous,
        "pos_len_mean": pos_len_mean,
        "pos_len_median": pos_len_median,
        "pos_len_p25": pos_len_p25,
        "pos_len_p75": pos_len_p75,
        "neg_len_mean": neg_len_mean,
        "neg_len_median": neg_len_median,
        "neg_len_p25": neg_len_p25,
        "neg_len_p75": neg_len_p75,
        "len_ratio": len_ratio,
        "pos_lm_mean": pos_lm_mean,
        "pos_lm_median": pos_lm_median,
        "neg_lm_mean": neg_lm_mean,
        "neg_lm_median": neg_lm_median,
        "d2l_mean": d2l_mean,
        "d2l_median": d2l_median,
        "d2l_p25": d2l_p25,
        "d2l_p75": d2l_p75,
        "d2l_min": d2l_min,
        "d2l_max": d2l_max,
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
        con.execute("PRAGMA threads=1")
        con.execute("SET preserve_insertion_order=false")

        print(f"[disease] {args.disease}")
        print(f"[code_prefix] {args.code_prefix}")
        print(f"[rolled] {rolled_path}")
        print(f"[out_dir] {args.out_dir}")
        print(
            f"[config] obs_window_days={args.obs_window_days} gap_days={args.gap_days} "
            f"pred_window_days={args.pred_window_days} min_obs_window_days={args.min_obs_window_days} "
            f"min_events_in_window={args.min_events_in_window}"
        )

        for split in ("train", "val", "test"):
            stats = build_split_cohort(
                con=con,
                rolled_path=rolled_path,
                split_events_path=split_paths[split],
                code_prefix=args.code_prefix,
                out_path=out_paths[split],
                obs_window_days=args.obs_window_days,
                gap_days=args.gap_days,
                pred_window_days=args.pred_window_days,
                min_obs_window_days=args.min_obs_window_days,
                min_events_in_window=args.min_events_in_window,
            )
            print(
                f"[{split}] n_pos={int(stats['n_pos']):,} n_neg={int(stats['n_neg']):,} "
                f"prevalence={float(stats['prevalence']):.4%}"
            )
            print(
                f"[{split}:drops] dropped_no_eligible_landmark={int(stats['dropped_no_eligible_landmark']):,} "
                f"dropped_no_landmark_room={int(stats['dropped_no_landmark_room']):,} "
                f"dropped_short_record={int(stats['dropped_short_record']):,} "
                f"dropped_too_few_events={int(stats['dropped_too_few_events']):,} "
                f"dropped_ambiguous={int(stats['dropped_ambiguous']):,}"
            )
            print(
                f"[{split}:len_stats] pos mean={float(stats['pos_len_mean']):.2f} "
                f"median={float(stats['pos_len_median']):.2f} p25={float(stats['pos_len_p25']):.2f} "
                f"p75={float(stats['pos_len_p75']):.2f} | neg mean={float(stats['neg_len_mean']):.2f} "
                f"median={float(stats['neg_len_median']):.2f} p25={float(stats['neg_len_p25']):.2f} "
                f"p75={float(stats['neg_len_p75']):.2f} | neg/pos ratio={float(stats['len_ratio']):.4f}"
            )
            print(
                f"[{split}:landmark_stats] pos mean={float(stats['pos_lm_mean']):.2f} "
                f"median={float(stats['pos_lm_median']):.2f} | neg mean={float(stats['neg_lm_mean']):.2f} "
                f"median={float(stats['neg_lm_median']):.2f}"
            )
            print(
                f"[{split}:disease_minus_landmark] min={float(stats['d2l_min']):.2f} "
                f"p25={float(stats['d2l_p25']):.2f} mean={float(stats['d2l_mean']):.2f} "
                f"median={float(stats['d2l_median']):.2f} p75={float(stats['d2l_p75']):.2f} "
                f"max={float(stats['d2l_max']):.2f} expected_range=[{args.gap_days}, {args.gap_days + args.pred_window_days}]"
            )

        print("Wrote:")
        for split in ("train", "val", "test"):
            print(f"  - {out_paths[split]}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
