#!/usr/bin/env python3
"""STEP 4 gate: per-candidate-task prevalence in the rolled Synthea parquet.

For each candidate target code_id, reports across all retained subjects:
  - n_with_code        : subjects who ever have the code
  - n_onset_after_prior: subjects whose FIRST occurrence index > 0 (i.e. there is
    >=1 prior event) -- these are the usable positives for build_disease_cohort,
    which requires first_occurrence_idx > 0.
  - n_onset_at_first   : subjects whose first event IS the code (dropped by cohort)
  - prevalence         : n_onset_after_prior / n_subjects
This confirms each task has enough positives that onset after >=1 prior event
before committing to it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

CANDIDATES = {
    "obesity": "COND_414916001",
    "t2d": "COND_44054006",
    "osa": "COND_78275009",
    "asthma": "COND_195967001",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report candidate disease-task prevalence.")
    p.add_argument(
        "--rolled",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthea" / "processed"
        / "patient_events_rolled_full.parquet",
    )
    return p.parse_args()


def _esc(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


def main() -> int:
    args = parse_args()
    con = duckdb.connect()
    rolled = _esc(args.rolled)
    n_subjects = int(con.execute(
        f"SELECT COUNT(DISTINCT subject_id) FROM read_parquet('{rolled}')").fetchone()[0])
    print(f"retained subjects: {n_subjects:,}\n")
    print(f"{'task':<10}{'code_id':<22}{'with_code':>11}{'onset>0':>10}{'at_first':>10}{'prev':>9}")
    for task, code in CANDIDATES.items():
        df = con.execute(
            f"""
            WITH ev AS (
                SELECT subject_id, code_id,
                       ROW_NUMBER() OVER (PARTITION BY subject_id
                           ORDER BY timestamp_days, event_time, code_id) - 1 AS idx
                FROM read_parquet('{rolled}')
            ),
            firsts AS (
                SELECT subject_id,
                       MIN(CASE WHEN starts_with(code_id, '{code}') THEN idx END) AS first_idx
                FROM ev GROUP BY subject_id
            )
            SELECT
                COUNT(*) FILTER (WHERE first_idx IS NOT NULL) AS with_code,
                COUNT(*) FILTER (WHERE first_idx > 0) AS onset_gt0,
                COUNT(*) FILTER (WHERE first_idx = 0) AS onset_at_first
            FROM firsts
            """
        ).fetchone()
        with_code, onset_gt0, at_first = int(df[0]), int(df[1]), int(df[2])
        prev = onset_gt0 / n_subjects if n_subjects else 0.0
        print(f"{task:<10}{code:<22}{with_code:>11,}{onset_gt0:>10,}{at_first:>10,}{prev:>8.2%}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
