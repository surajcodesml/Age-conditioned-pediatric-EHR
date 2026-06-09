#!/usr/bin/env python3
"""Roll up raw PIC event codes to PheCodes and build English code descriptions.

Mirrors ``preprocessing/rollup_and_describe.py`` (MIMIC). Reuses the PheWAS PheCode
map loader/builder (``create_phecode_maps``) from the parent package unchanged.

Diagnoses (``ICD10_*``) are rolled up using the WHO ICD-10 code carried in the
``aux_code`` column (PIC's ``D_ICD_DIAGNOSES.ICD10_CODE``). The PheWAS ICD-10 map is
US ICD-10-CM, so a two-pass lookup is used:
  1. full WHO code (dot-stripped), then
  2. fall back to the 3-character ICD-10 category prefix.
Mapped -> ``PHE_<phecode>``; unmapped -> kept as ``ICD10_<code>`` with English title.
The mapping rate at full-code vs 3-char-fallback is logged separately.

Labs / meds / chart / exam codes are already final from script 1; their English
descriptions are carried through.

Outputs (to ``data/processed/pic/``):
- ``patient_events_rolled_pic.parquet`` (MIMIC 10-column schema)
- ``code_descriptions_pic.json``        ({final code_id: English description})
- ``pic_lab_loinc_map.json``            (verified present; authored by script 1)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import duckdb

from _shared import create_phecode_maps, save_json

LOGGER = logging.getLogger("rollup_and_describe_pic")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll up PIC codes and build descriptions.")
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed",
        help="Repo data/processed dir (holds the shared mappings/ cache and the pic/ subdir).",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    processed_dir: Path = args.processed_dir
    pic_dir = processed_dir / "pic"
    # Reuse the SAME PheWAS download/cache directory as the MIMIC pipeline.
    mappings_dir = processed_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    input_parquet = pic_dir / "patient_events_pic.parquet"
    rolled_parquet = pic_dir / "patient_events_rolled_pic.parquet"
    desc_json = pic_dir / "code_descriptions_pic.json"
    loinc_json = pic_dir / "pic_lab_loinc_map.json"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_parquet}")

    db_path = pic_dir / "rollup_and_describe_pic.duckdb"
    wal_path = pic_dir / "rollup_and_describe_pic.duckdb.wal"
    for p in (db_path, wal_path):
        if p.exists():
            p.unlink()

    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA memory_limit='24GB'")
        duckdb_tmp = (pic_dir / "duckdb_tmp").resolve()
        duckdb_tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{str(duckdb_tmp).replace(chr(39), chr(39)*2)}'")
        con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 8) - 2)}")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE patient_events AS
            SELECT * FROM read_parquet('{str(input_parquet).replace("'", "''")}')
            """
        )

        # Reused, unchanged: builds phe_icd9 / phe_icd10 / phe_defs in this connection.
        create_phecode_maps(con, mappings_dir)

        # ---------------- two-pass PheCode rollup on the WHO ICD-10 code ----------------
        LOGGER.info("Rolling up diagnoses via WHO ICD-10 (full code, then 3-char prefix).")
        con.execute(
            """
            CREATE OR REPLACE TABLE diag_mapped AS
            WITH diag AS (
                SELECT
                    *,
                    REPLACE(UPPER(TRIM(CAST(aux_code AS VARCHAR))), '.', '') AS who_norm
                FROM patient_events
                WHERE code_type = 'diagnosis'
            )
            SELECT
                d.* EXCLUDE (who_norm),
                pf.phecode AS phe_full,
                pc.phecode AS phe_cat3,
                CASE
                    WHEN pf.phecode IS NOT NULL THEN 'full'
                    WHEN pc.phecode IS NOT NULL THEN 'cat3'
                    ELSE 'none'
                END AS map_source,
                CASE
                    WHEN pf.phecode IS NOT NULL THEN 'PHE_' || pf.phecode
                    WHEN pc.phecode IS NOT NULL THEN 'PHE_' || pc.phecode
                    ELSE d.code_id
                END AS rolled_code_id
            FROM diag d
            LEFT JOIN phe_icd10 pf ON pf.icd_code_norm = d.who_norm
            LEFT JOIN phe_icd10 pc ON pc.icd_code_norm = LEFT(d.who_norm, 3)
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE patient_events_rolled AS
            SELECT
                subject_id, hadm_id, event_time, rolled_code_id AS code_id,
                CASE WHEN map_source = 'none' THEN 'diagnosis_unmapped' ELSE 'diagnosis' END AS code_type,
                timestamp_days, log_delta_t, age_at_event_days, sex, race
            FROM diag_mapped
            UNION ALL
            SELECT
                subject_id, hadm_id, event_time, code_id, code_type,
                timestamp_days, log_delta_t, age_at_event_days, sex, race
            FROM patient_events
            WHERE code_type <> 'diagnosis'
            """
        )

        escaped_out = str(rolled_parquet).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT subject_id, hadm_id, event_time, code_id, code_type,
                       timestamp_days, log_delta_t, age_at_event_days, sex, race
                FROM patient_events_rolled
                ORDER BY subject_id, event_time, code_id
            ) TO '{escaped_out}' (FORMAT PARQUET)
            """
        )

        # ---------------- code descriptions ----------------
        # Carry English description from the raw events (consistent per raw code_id).
        con.execute(
            """
            CREATE OR REPLACE TABLE desc_carry AS
            SELECT code_id, ANY_VALUE(description) AS description
            FROM patient_events
            WHERE description IS NOT NULL AND TRIM(description) <> ''
            GROUP BY code_id
            """
        )
        rows = con.execute(
            """
            WITH codes AS (SELECT DISTINCT code_id FROM patient_events_rolled)
            SELECT
                c.code_id,
                COALESCE(
                    CASE WHEN c.code_id LIKE 'PHE_%'
                         THEN COALESCE(pd.phenotype, 'PheCode ' || SUBSTR(c.code_id, 5))
                              || COALESCE(' (' || pd.category || ')', '')
                    END,
                    dc.description,
                    c.code_id
                ) AS description
            FROM codes c
            LEFT JOIN phe_defs pd ON c.code_id LIKE 'PHE_%' AND SUBSTR(c.code_id, 5) = pd.phecode
            LEFT JOIN desc_carry dc ON c.code_id = dc.code_id
            """
        ).fetchall()
        descriptions = {str(c): str(d) for c, d in rows}
        save_json(desc_json, descriptions)

        # ---------------- LOINC sidecar (authored by script 1) ----------------
        if not loinc_json.exists():
            raise FileNotFoundError(
                f"Expected {loinc_json} from script 1; run build_event_table_pic.py first."
            )

        # ---------------- sanity / mapping-rate logging ----------------
        diag_total = int(con.execute("SELECT COUNT(*) FROM diag_mapped").fetchone()[0])
        n_full = int(con.execute("SELECT COUNT(*) FROM diag_mapped WHERE map_source='full'").fetchone()[0])
        n_cat3 = int(con.execute("SELECT COUNT(*) FROM diag_mapped WHERE map_source='cat3'").fetchone()[0])
        n_none = int(con.execute("SELECT COUNT(*) FROM diag_mapped WHERE map_source='none'").fetchone()[0])
        before = int(con.execute("SELECT COUNT(DISTINCT code_id) FROM patient_events").fetchone()[0])
        after = int(con.execute("SELECT COUNT(DISTINCT code_id) FROM patient_events_rolled").fetchone()[0])

        def pct(x: int) -> float:
            return (100.0 * x / diag_total) if diag_total else 0.0

        print("=== PIC rollup ===")
        print(f"Unique codes before rollup: {before:,}")
        print(f"Unique codes after rollup:  {after:,}")
        print(f"Diagnosis events: {diag_total:,}")
        print(f"  PheCode mapped (full code):       {n_full:,} ({pct(n_full):.2f}%)")
        print(f"  PheCode mapped (3-char fallback): {n_cat3:,} ({pct(n_cat3):.2f}%)")
        print(f"  Unmapped (kept as ICD10_):        {n_none:,} ({pct(n_none):.2f}%)")
        print(f"  Overall PheCode mapping rate:     {pct(n_full + n_cat3):.2f}%")

        missing = [c for c in (str(r[0]) for r in con.execute(
            "SELECT DISTINCT code_id FROM patient_events_rolled").fetchall())
            if c not in descriptions or not descriptions[c]]
        if missing:
            raise AssertionError(f"Missing descriptions for {len(missing)} codes, e.g. {missing[:10]}")
        print(f"Total codes with descriptions: {len(descriptions):,}")

        LOGGER.info("Wrote rolled parquet: %s", rolled_parquet)
        LOGGER.info("Wrote code descriptions: %s", desc_json)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
