#!/usr/bin/env python3
"""STEP 2: roll Synthea CSV exports into the event-table parquet consumed by the
existing TALE-EHR pipeline.

Emits, into ``--out_dir`` (default ``data/synthea/processed``):
  - ``patient_events_rolled_full.parquet`` : one row per (patient, encounter,
    code, time) event, with the EXACT column schema the MIMIC pipeline produces
    and that ``build_splits.py`` / ``preprocessing/tensorize.py`` /
    ``finetune/build_disease_cohort.py`` / ``build_disease_tensors.py`` /
    ``finetune/dataset.py`` read:
        subject_id (BIGINT), hadm_id (BIGINT), event_time (TIMESTAMP),
        code_id (VARCHAR), code_type (VARCHAR), timestamp_days (DOUBLE),
        log_delta_t (DOUBLE), age_at_event_days (DOUBLE), sex (TINYINT),
        race (VARCHAR)
  - ``code_descriptions.json`` : code_id -> Synthea's own human-readable
    DESCRIPTION (feeds compute_bge_embeddings.py; no SNOMED->PheCode mapping).

Design choices anchored to the consumers (not line numbers):
  * subject_id / hadm_id are STABLE INTEGERS: dense-ranked from the sorted
    patient UUID and encounter UUID respectively. tensorize builds visits from
    hadm_id and DROPS patients with < 2 distinct encounters.
  * code_id = source prefix (COND_/MED_/OBS_/PROC_) + native code, pulled from
    conditions, medications, observations (BMI/vitals + labs) and procedures.
  * timestamp_days = days since the patient's first event;
    age_at_event_days = days since birth.
  * sex: M -> 1 else 0 (matches build_event_table.py's encoding and
    tensorize.encode_sex). race: emitted as MIMIC-style strings so the existing
    dataset/tensorizer encode_race() numericalizes them with no new categories
    and the UNKNOWN (bucket 6) fallback.
  * MIMIC retention filters applied: >= 5 events, > 1 distinct timestamp_days,
    >= 2 distinct encounters.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb

LOGGER = logging.getLogger("build_synthea_events")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roll Synthea CSVs into the TALE-EHR event parquet.")
    p.add_argument(
        "--csv_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "csv",
        help="Directory with Synthea patients/encounters/conditions/medications/observations/procedures CSVs.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthea" / "processed",
        help="Output directory for the rolled parquet + code_descriptions.json.",
    )
    return p.parse_args()


def _esc(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


# Map a Synthea race/ethnicity to a MIMIC-style race string that the existing
# encode_race() recognizes (no new categories; UNKNOWN -> bucket 6 fallback).
_RACE_CASE_SQL = """
    CASE
        WHEN lower(coalesce(p.ETHNICITY, '')) = 'hispanic' THEN 'HISPANIC'
        WHEN lower(coalesce(p.RACE, '')) = 'white' THEN 'WHITE'
        WHEN lower(coalesce(p.RACE, '')) = 'black' THEN 'BLACK'
        WHEN lower(coalesce(p.RACE, '')) = 'asian' THEN 'ASIAN'
        WHEN lower(coalesce(p.RACE, '')) = 'native' THEN 'AMERICAN INDIAN'
        WHEN lower(coalesce(p.RACE, '')) = 'hawaiian' THEN 'OTHER'
        WHEN lower(coalesce(p.RACE, '')) = 'other' THEN 'OTHER'
        ELSE 'UNKNOWN'
    END
"""


def main() -> int:
    setup_logging()
    args = parse_args()
    csv_dir: Path = args.csv_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {name: csv_dir / f"{name}.csv" for name in
             ("patients", "encounters", "conditions", "medications", "observations", "procedures")}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Synthea CSV: {path}")

    rolled_out = out_dir / "patient_events_rolled_full.parquet"
    desc_out = out_dir / "code_descriptions.json"

    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit='12GB'")
        tmp = (out_dir / "duckdb_tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{_esc(tmp)}'")

        # --- patients: stable int subject_id, sex, race string, birthdate ---
        con.execute(
            f"""
            CREATE OR REPLACE TABLE patients AS
            SELECT
                p.Id AS patient_uuid,
                CAST(DENSE_RANK() OVER (ORDER BY p.Id) AS BIGINT) AS subject_id,
                CASE WHEN upper(coalesce(p.GENDER, '')) = 'M' THEN CAST(1 AS TINYINT) ELSE CAST(0 AS TINYINT) END AS sex,
                {_RACE_CASE_SQL} AS race,
                CAST(p.BIRTHDATE AS TIMESTAMP) AS birthdate
            FROM read_csv_auto('{_esc(files['patients'])}') p
            """
        )

        # --- stable int hadm_id from encounter UUIDs ---
        con.execute(
            f"""
            CREATE OR REPLACE TABLE enc_map AS
            SELECT
                e.Id AS encounter_uuid,
                CAST(DENSE_RANK() OVER (ORDER BY e.Id) AS BIGINT) AS hadm_id
            FROM read_csv_auto('{_esc(files['encounters'])}') e
            """
        )

        # --- unified raw events from the four sources (code_id prefixed + description) ---
        con.execute(
            f"""
            CREATE OR REPLACE TABLE events_src AS
            SELECT PATIENT AS patient_uuid, ENCOUNTER AS encounter_uuid,
                   CAST(START AS TIMESTAMP) AS event_time,
                   'COND_' || CAST(CODE AS VARCHAR) AS code_id,
                   'condition' AS code_type,
                   CAST(DESCRIPTION AS VARCHAR) AS description
            FROM read_csv_auto('{_esc(files['conditions'])}')
            WHERE CODE IS NOT NULL AND START IS NOT NULL
            UNION ALL
            SELECT PATIENT, ENCOUNTER, CAST(START AS TIMESTAMP),
                   'MED_' || CAST(CODE AS VARCHAR), 'medication', CAST(DESCRIPTION AS VARCHAR)
            FROM read_csv_auto('{_esc(files['medications'])}')
            WHERE CODE IS NOT NULL AND START IS NOT NULL
            UNION ALL
            SELECT PATIENT, ENCOUNTER, CAST(DATE AS TIMESTAMP),
                   'OBS_' || CAST(CODE AS VARCHAR), 'observation', CAST(DESCRIPTION AS VARCHAR)
            FROM read_csv_auto('{_esc(files['observations'])}')
            WHERE CODE IS NOT NULL AND DATE IS NOT NULL
            UNION ALL
            SELECT PATIENT, ENCOUNTER, CAST(START AS TIMESTAMP),
                   'PROC_' || CAST(CODE AS VARCHAR), 'procedure', CAST(DESCRIPTION AS VARCHAR)
            FROM read_csv_auto('{_esc(files['procedures'])}')
            WHERE CODE IS NOT NULL AND START IS NOT NULL
            """
        )

        n_raw_events = int(con.execute("SELECT COUNT(*) FROM events_src").fetchone()[0])
        n_raw_subjects = int(con.execute("SELECT COUNT(DISTINCT patient_uuid) FROM events_src").fetchone()[0])

        # --- join ids/demographics, compute time features ---
        con.execute(
            """
            CREATE OR REPLACE TABLE events_joined AS
            WITH base AS (
                SELECT
                    pt.subject_id,
                    COALESCE(em.hadm_id, CAST(-1 AS BIGINT)) AS hadm_id,
                    es.event_time,
                    es.code_id,
                    es.code_type,
                    es.description,
                    pt.sex,
                    pt.race,
                    (epoch(es.event_time) - epoch(pt.birthdate)) / 86400.0 AS age_at_event_days
                FROM events_src es
                JOIN patients pt ON es.patient_uuid = pt.patient_uuid
                LEFT JOIN enc_map em ON es.encounter_uuid = em.encounter_uuid
            ),
            ts AS (
                SELECT
                    *,
                    (epoch(event_time) - epoch(MIN(event_time) OVER (PARTITION BY subject_id))) / 86400.0
                        AS timestamp_days
                FROM base
            ),
            step AS (
                SELECT
                    *,
                    LAG(timestamp_days) OVER (
                        PARTITION BY subject_id ORDER BY event_time, code_id, code_type, hadm_id
                    ) AS prev_ts,
                    ROW_NUMBER() OVER (
                        PARTITION BY subject_id ORDER BY event_time, code_id, code_type, hadm_id
                    ) AS rn
                FROM ts
            )
            SELECT
                subject_id, hadm_id, event_time, code_id, code_type,
                CAST(timestamp_days AS DOUBLE) AS timestamp_days,
                CASE WHEN rn = 1 THEN 0.0
                     ELSE LN(1 + GREATEST(timestamp_days - COALESCE(prev_ts, 0.0), 0.0)) END AS log_delta_t,
                CAST(age_at_event_days AS DOUBLE) AS age_at_event_days,
                sex, race, description
            FROM step
            """
        )

        # --- MIMIC retention filters: >=5 events, >1 distinct timestamp, >=2 distinct encounters ---
        con.execute(
            """
            CREATE OR REPLACE TABLE subj_stats AS
            SELECT subject_id,
                   COUNT(*) AS n_events,
                   COUNT(DISTINCT timestamp_days) AS n_ts,
                   COUNT(DISTINCT hadm_id) AS n_enc
            FROM events_joined
            GROUP BY subject_id
            """
        )
        n_ge5 = int(con.execute("SELECT COUNT(*) FROM subj_stats WHERE n_events >= 5").fetchone()[0])
        n_ge5_ts = int(con.execute("SELECT COUNT(*) FROM subj_stats WHERE n_events >= 5 AND n_ts > 1").fetchone()[0])
        n_keep = int(con.execute(
            "SELECT COUNT(*) FROM subj_stats WHERE n_events >= 5 AND n_ts > 1 AND n_enc >= 2").fetchone()[0])

        con.execute(
            """
            CREATE OR REPLACE TABLE subj_keep AS
            SELECT subject_id FROM subj_stats
            WHERE n_events >= 5 AND n_ts > 1 AND n_enc >= 2
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE final_events AS
            SELECT e.subject_id, e.hadm_id, e.event_time, e.code_id, e.code_type,
                   e.timestamp_days, e.log_delta_t, e.age_at_event_days, e.sex, e.race
            FROM events_joined e
            JOIN subj_keep k USING (subject_id)
            """
        )

        # --- write rolled parquet (same column order as the MIMIC rolled output) ---
        con.execute(
            f"""
            COPY (
                SELECT subject_id, hadm_id, event_time, code_id, code_type,
                       timestamp_days, log_delta_t, age_at_event_days, sex, race
                FROM final_events
                ORDER BY subject_id, event_time, code_id
            ) TO '{_esc(rolled_out)}' (FORMAT PARQUET)
            """
        )

        # --- code_descriptions.json : code_id -> Synthea DESCRIPTION (+ type hint) ---
        desc_rows = con.execute(
            """
            WITH ranked AS (
                SELECT code_id, code_type, description, COUNT(*) AS cnt,
                       ROW_NUMBER() OVER (PARTITION BY code_id ORDER BY COUNT(*) DESC) AS rn
                FROM events_joined ej JOIN subj_keep k USING (subject_id)
                WHERE description IS NOT NULL AND TRIM(description) <> ''
                GROUP BY code_id, code_type, description
            )
            SELECT code_id, code_type, description FROM ranked WHERE rn = 1
            """
        ).fetchall()
        descriptions = {
            str(code_id): f"{str(desc)} ({str(code_type)})"
            for (code_id, code_type, desc) in desc_rows
        }
        # Guarantee every code in the parquet has a description (fallback to the code).
        all_codes = [str(r[0]) for r in con.execute(
            "SELECT DISTINCT code_id FROM final_events").fetchall()]
        for c in all_codes:
            descriptions.setdefault(c, c)
        with desc_out.open("w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=True, indent=2)

        # --- sanity report (drop reasons + counts) ---
        total_rows = int(con.execute("SELECT COUNT(*) FROM final_events").fetchone()[0])
        n_codes = int(con.execute("SELECT COUNT(DISTINCT code_id) FROM final_events").fetchone()[0])
        per_type = con.execute(
            "SELECT code_type, COUNT(*) c FROM final_events GROUP BY code_type ORDER BY c DESC").fetchall()
        ts_rng = con.execute(
            "SELECT MIN(timestamp_days), MAX(timestamp_days) FROM final_events").fetchone()
        age_rng = con.execute(
            "SELECT MIN(age_at_event_days), MAX(age_at_event_days) FROM final_events").fetchone()

        print("=== build_synthea_events sanity ===")
        print(f"raw events: {n_raw_events:,} | raw subjects: {n_raw_subjects:,}")
        print("patients dropped and why:")
        print(f"  - excluded_fewer_than_5_events : {n_raw_subjects - n_ge5:,}")
        print(f"  - excluded_single_unique_timestamp : {n_ge5 - n_ge5_ts:,}")
        print(f"  - excluded_fewer_than_2_encounters : {n_ge5_ts - n_keep:,}")
        print(f"RETAINED subjects: {n_keep:,}")
        print(f"final rows: {total_rows:,} | distinct code_ids: {n_codes:,}")
        print(f"timestamp_days range: [{ts_rng[0]:.3f}, {ts_rng[1]:.3f}]")
        print(f"age_at_event_days range: [{age_rng[0]:.3f}, {age_rng[1]:.3f}]  (min must be >= 0)")
        print("rows per code_type:")
        for ct, c in per_type:
            print(f"  - {ct}: {int(c):,}")
        print(f"descriptions written: {len(descriptions):,}")
        print(f"wrote: {rolled_out}")
        print(f"wrote: {desc_out}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
