#!/usr/bin/env python3
"""Build a unified patient event table from the PIC v1.1.0 database via DuckDB.

Mirrors ``preprocessing/build_event_table.py`` (MIMIC-IV) so downstream code runs
unchanged. PIC differences handled here:
- PIC tables are bilingual (Chinese + English); we always use the English column.
- PIC ``ITEMID`` is a CHARACTER field -> cast to VARCHAR everywhere.
- PIC diagnoses use ICD-10 (7-char Chinese clinical extension ``ICD10_CODE_CN``);
  the dictionary also exposes the WHO ICD-10 ``ICD10_CODE`` which we carry in the
  ``aux_code`` column for PheCode rollup in script 2.
- PIC has no usable race field -> race is stored as the literal ``'UNKNOWN'`` so the
  shared ``encode_race`` maps every PIC subject to the UNK bucket (index 6).

Column mappings are NOT hardcoded blindly: each table's CSV header is printed and the
chosen role->column mapping is logged before loading.

Output (to ``data/processed/pic/``):
- ``patient_events_pic.parquet``  (raw codes, pre-rollup; carries aux_code + description)
- ``code_vocab_raw_pic.json``     ({code_id: frequency})
- ``pic_lab_loinc_map.json``      ({LAB_<itemid>: loinc_code})
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import threading
import time
from pathlib import Path

import duckdb

LOGGER = logging.getLogger("build_event_table_pic")

DEFAULT_PIC_ROOT = Path(
    "/home/suraj/Data/PIC/physionet.org/files/picdb/1.1.0/V1.1.0"
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified PIC event table.")
    parser.add_argument(
        "--pic_root",
        type=Path,
        default=DEFAULT_PIC_ROOT,
        help="Path to PIC v1.1.0 folder containing the uncompressed CSV tables.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "pic",
        help="Directory for output parquet/json files.",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="If set, sample 1,000 patients and write patient_events_test_pic.parquet.",
    )
    return parser.parse_args()


def _csv_header(filepath: Path) -> list[str]:
    with filepath.open("r", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def _log_mapping(table: str, filepath: Path, mapping: dict[str, str]) -> None:
    header = _csv_header(filepath)
    LOGGER.info("[%s] CSV header: %s", table, header)
    LOGGER.info("[%s] chosen column mapping (role -> column):", table)
    for role, col in mapping.items():
        LOGGER.info("    %-22s -> %s", role, col)


def _csv_source(filepath: Path) -> str:
    """read_csv_auto source string with ALL_VARCHAR for type-sniff robustness.

    Quote/escape are pinned to '"' because PIC's free-text fields (e.g. microbiology
    results) embed commas inside quotes, which the auto-detector otherwise misreads.
    """
    escaped = str(filepath).replace("'", "''")
    return (
        f"read_csv_auto('{escaped}', ALL_VARCHAR=TRUE, header=TRUE, "
        f"quote='\"', escape='\"')"
    )


def _execute(con: duckdb.DuckDBPyConnection, sql: str, action: str) -> None:
    start = time.monotonic()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(60):
            LOGGER.info("%s still running... elapsed %.1f min", action, (time.monotonic() - start) / 60.0)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    LOGGER.info("Started %s", action)
    try:
        con.execute(sql)
    finally:
        stop_event.set()
        hb.join(timeout=0.1)
        LOGGER.info("Finished %s in %.1f min", action, (time.monotonic() - start) / 60.0)


def _scalar_int(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(con.execute(query).fetchone()[0])


def _scalar_float(con: duckdb.DuckDBPyConnection, query: str) -> float:
    value = con.execute(query).fetchone()[0]
    return float(value) if value is not None else float("nan")


# Normalize an English string into a slug: lowercase, trimmed, whitespace collapsed.
_SLUG_SQL = "LOWER(TRIM(REGEXP_REPLACE({col}, '\\s+', ' ', 'g')))"


def main() -> int:
    setup_logging()
    args = parse_args()

    pic_root: Path = args.pic_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "patients": pic_root / "PATIENTS.csv",
        "admissions": pic_root / "ADMISSIONS.csv",
        "diagnoses": pic_root / "DIAGNOSES_ICD.csv",
        "d_icd": pic_root / "D_ICD_DIAGNOSES.csv",
        "labevents": pic_root / "LABEVENTS.csv",
        "d_labitems": pic_root / "D_LABITEMS.csv",
        "prescriptions": pic_root / "PRESCRIPTIONS.csv",
        "chartevents": pic_root / "CHARTEVENTS.csv",
        "d_items": pic_root / "D_ITEMS.csv",
        "exam": pic_root / "OR_EXAM_REPORTS.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required PIC files:\n" + "\n".join(missing))

    # DuckDB hygiene: persistent on-disk DB; remove stale .duckdb/.wal before connect.
    db_path = output_dir / "build_event_table_pic.duckdb"
    wal_path = output_dir / "build_event_table_pic.duckdb.wal"
    for p in (db_path, wal_path):
        if p.exists():
            p.unlink()

    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA memory_limit='24GB'")
        escaped_temp = str(temp_dir).replace("'", "''")
        con.execute(f"PRAGMA temp_directory='{escaped_temp}'")
        con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 8) - 2)}")

        LOGGER.info("Starting PIC build_event_table (test_mode=%s)", args.test_mode)

        # ---------------- PATIENTS ----------------
        _log_mapping(
            "PATIENTS", paths["patients"],
            {"subject_id": "SUBJECT_ID", "gender(sex)": "GENDER", "dob": "DOB"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE patients_raw AS
            SELECT
                TRY_CAST(SUBJECT_ID AS BIGINT) AS subject_id,
                CAST(GENDER AS VARCHAR) AS gender,
                TRY_CAST(DOB AS TIMESTAMP) AS dob
            FROM {_csv_source(paths['patients'])}
            WHERE TRY_CAST(SUBJECT_ID AS BIGINT) IS NOT NULL
            """,
            "loading PATIENTS",
        )

        if args.test_mode:
            ids = con.execute("SELECT subject_id FROM patients_raw").df()["subject_id"]
            if len(ids) < 1000:
                raise ValueError(f"Need 1000 patients, only {len(ids)} present.")
            sampled = ids.sample(n=1000, random_state=42).astype("int64").tolist()
            con.execute("CREATE OR REPLACE TEMP TABLE sample_subjects(subject_id BIGINT)")
            con.executemany("INSERT INTO sample_subjects VALUES (?)", [(s,) for s in sampled])
            LOGGER.info("Sampled %d subjects for test mode.", len(sampled))
            sj = lambda a: f"JOIN sample_subjects ss ON TRY_CAST({a}.SUBJECT_ID AS BIGINT) = ss.subject_id"  # noqa: E731
        else:
            sj = lambda a: ""  # noqa: E731

        # ---------------- ADMISSIONS (for diagnosis event_time + cohort) ----------------
        _log_mapping(
            "ADMISSIONS", paths["admissions"],
            {"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "admittime(event_time)": "ADMITTIME"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE admissions AS
            SELECT
                TRY_CAST(t.SUBJECT_ID AS BIGINT) AS subject_id,
                TRY_CAST(t.HADM_ID AS BIGINT) AS hadm_id,
                TRY_CAST(t.ADMITTIME AS TIMESTAMP) AS admittime
            FROM {_csv_source(paths['admissions'])} t
            {sj('t')}
            WHERE TRY_CAST(t.SUBJECT_ID AS BIGINT) IS NOT NULL
            """,
            "loading ADMISSIONS",
        )
        con.execute(
            "CREATE OR REPLACE TABLE admission_subjects AS SELECT DISTINCT subject_id FROM admissions"
        )

        # ---------------- D_ICD_DIAGNOSES (English title + WHO ICD-10) ----------------
        _log_mapping(
            "D_ICD_DIAGNOSES", paths["d_icd"],
            {"join_key(icd_cn)": "ICD10_CODE_CN", "who_icd10": "ICD10_CODE", "english_title": "TITLE"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE d_icd AS
            SELECT
                REPLACE(UPPER(TRIM(CAST(ICD10_CODE_CN AS VARCHAR))), '.', '') AS icd_cn_norm,
                CAST(ICD10_CODE AS VARCHAR) AS icd_who,
                CAST(TITLE AS VARCHAR) AS title
            FROM {_csv_source(paths['d_icd'])}
            WHERE ICD10_CODE_CN IS NOT NULL
            """,
            "loading D_ICD_DIAGNOSES",
        )

        # ---------------- D_LABITEMS (English label + LOINC) ----------------
        _log_mapping(
            "D_LABITEMS", paths["d_labitems"],
            {"itemid": "ITEMID", "english_label": "LABEL", "loinc": "LOINC_CODE"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE d_labitems AS
            SELECT
                CAST(ITEMID AS VARCHAR) AS itemid,
                CAST(LABEL AS VARCHAR) AS label,
                CAST(LOINC_CODE AS VARCHAR) AS loinc_code
            FROM {_csv_source(paths['d_labitems'])}
            WHERE ITEMID IS NOT NULL
            """,
            "loading D_LABITEMS",
        )

        # ---------------- D_ITEMS (English label for chartevents) ----------------
        _log_mapping(
            "D_ITEMS", paths["d_items"],
            {"itemid": "ITEMID", "english_label": "LABEL"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE d_items AS
            SELECT
                CAST(ITEMID AS VARCHAR) AS itemid,
                CAST(LABEL AS VARCHAR) AS label
            FROM {_csv_source(paths['d_items'])}
            WHERE ITEMID IS NOT NULL
            """,
            "loading D_ITEMS",
        )

        # ---------------- events: DIAGNOSES_ICD ----------------
        _log_mapping(
            "DIAGNOSES_ICD", paths["diagnoses"],
            {"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "icd10_cn(code)": "ICD10_CODE_CN",
             "event_time": "ADMISSIONS.ADMITTIME", "english_desc": "D_ICD_DIAGNOSES.TITLE",
             "aux_code(who)": "D_ICD_DIAGNOSES.ICD10_CODE"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE events_diagnosis AS
            SELECT
                TRY_CAST(d.SUBJECT_ID AS BIGINT) AS subject_id,
                TRY_CAST(d.HADM_ID AS BIGINT) AS hadm_id,
                a.admittime AS event_time,
                'ICD10_' || REPLACE(UPPER(TRIM(CAST(d.ICD10_CODE_CN AS VARCHAR))), '.', '') AS code_id,
                'diagnosis' AS code_type,
                di.icd_who AS aux_code,
                COALESCE(di.title, CAST(d.ICD10_CODE_CN AS VARCHAR)) AS description
            FROM {_csv_source(paths['diagnoses'])} d
            {sj('d')}
            JOIN admissions a ON TRY_CAST(d.HADM_ID AS BIGINT) = a.hadm_id
            LEFT JOIN d_icd di
                ON REPLACE(UPPER(TRIM(CAST(d.ICD10_CODE_CN AS VARCHAR))), '.', '') = di.icd_cn_norm
            WHERE d.ICD10_CODE_CN IS NOT NULL
            """,
            "building events_diagnosis",
        )
        LOGGER.info("Diagnosis events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_diagnosis"))

        # ---------------- events: LABEVENTS (~10M; stream) ----------------
        _log_mapping(
            "LABEVENTS", paths["labevents"],
            {"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "itemid": "ITEMID",
             "event_time": "CHARTTIME", "english_desc": "D_LABITEMS.LABEL"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE events_lab AS
            SELECT
                TRY_CAST(l.SUBJECT_ID AS BIGINT) AS subject_id,
                TRY_CAST(l.HADM_ID AS BIGINT) AS hadm_id,
                TRY_CAST(l.CHARTTIME AS TIMESTAMP) AS event_time,
                'LAB_' || CAST(l.ITEMID AS VARCHAR) AS code_id,
                'lab' AS code_type,
                CAST(NULL AS VARCHAR) AS aux_code,
                dl.label AS description
            FROM {_csv_source(paths['labevents'])} l
            {sj('l')}
            JOIN d_labitems dl ON CAST(l.ITEMID AS VARCHAR) = dl.itemid
            WHERE l.CHARTTIME IS NOT NULL
            """,
            "building events_lab",
        )
        LOGGER.info("Lab events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_lab"))

        # ---------------- events: PRESCRIPTIONS (~1.26M; stream) ----------------
        _log_mapping(
            "PRESCRIPTIONS", paths["prescriptions"],
            {"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "event_time": "STARTDATE",
             "english_drug(code+desc)": "DRUG_NAME_EN"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE events_medication AS
            SELECT
                TRY_CAST(m.SUBJECT_ID AS BIGINT) AS subject_id,
                TRY_CAST(m.HADM_ID AS BIGINT) AS hadm_id,
                TRY_CAST(m.STARTDATE AS TIMESTAMP) AS event_time,
                'DRUG_' || {_SLUG_SQL.format(col='CAST(m.DRUG_NAME_EN AS VARCHAR)')} AS code_id,
                'medication' AS code_type,
                CAST(NULL AS VARCHAR) AS aux_code,
                TRIM(CAST(m.DRUG_NAME_EN AS VARCHAR)) AS description
            FROM {_csv_source(paths['prescriptions'])} m
            {sj('m')}
            WHERE m.STARTDATE IS NOT NULL
              AND m.DRUG_NAME_EN IS NOT NULL
              AND TRIM(CAST(m.DRUG_NAME_EN AS VARCHAR)) <> ''
            """,
            "building events_medication",
        )
        LOGGER.info("Medication events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_medication"))

        # ---------------- events: CHARTEVENTS (stream) ----------------
        _log_mapping(
            "CHARTEVENTS", paths["chartevents"],
            {"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "itemid": "ITEMID",
             "event_time": "CHARTTIME", "english_desc": "D_ITEMS.LABEL"},
        )
        _execute(
            con,
            f"""
            CREATE OR REPLACE TABLE events_chart AS
            SELECT
                TRY_CAST(c.SUBJECT_ID AS BIGINT) AS subject_id,
                TRY_CAST(c.HADM_ID AS BIGINT) AS hadm_id,
                TRY_CAST(c.CHARTTIME AS TIMESTAMP) AS event_time,
                'CHART_' || CAST(c.ITEMID AS VARCHAR) AS code_id,
                'chart' AS code_type,
                CAST(NULL AS VARCHAR) AS aux_code,
                dit.label AS description
            FROM {_csv_source(paths['chartevents'])} c
            {sj('c')}
            JOIN d_items dit ON CAST(c.ITEMID AS VARCHAR) = dit.itemid
            WHERE c.CHARTTIME IS NOT NULL
              AND c.VALUE IS NOT NULL
              AND TRIM(CAST(c.VALUE AS VARCHAR)) <> ''
            """,
            "building events_chart",
        )
        LOGGER.info("Chart events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_chart"))

        # ---------------- events: OR_EXAM_REPORTS (English exam-name column present) ----------------
        exam_header = {c.strip().upper() for c in _csv_header(paths["exam"])}
        if "EXAM_ITEM_NAME" in exam_header:
            _log_mapping(
                "OR_EXAM_REPORTS", paths["exam"],
                {"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID", "event_time": "EXAMTIME",
                 "english_exam(code+desc)": "EXAM_ITEM_NAME"},
            )
            _execute(
                con,
                f"""
                CREATE OR REPLACE TABLE events_exam AS
                SELECT
                    TRY_CAST(e.SUBJECT_ID AS BIGINT) AS subject_id,
                    TRY_CAST(e.HADM_ID AS BIGINT) AS hadm_id,
                    TRY_CAST(e.EXAMTIME AS TIMESTAMP) AS event_time,
                    'EXAM_' || {_SLUG_SQL.format(col='CAST(e.EXAM_ITEM_NAME AS VARCHAR)')} AS code_id,
                    'exam' AS code_type,
                    CAST(NULL AS VARCHAR) AS aux_code,
                    TRIM(CAST(e.EXAM_ITEM_NAME AS VARCHAR)) AS description
                FROM {_csv_source(paths['exam'])} e
                {sj('e')}
                WHERE e.EXAMTIME IS NOT NULL
                  AND e.EXAM_ITEM_NAME IS NOT NULL
                  AND TRIM(CAST(e.EXAM_ITEM_NAME AS VARCHAR)) <> ''
                """,
                "building events_exam",
            )
            LOGGER.info("Exam events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_exam"))
            include_exam = True
        else:
            LOGGER.warning(
                "OR_EXAM_REPORTS has no EXAM_ITEM_NAME column (header=%s); skipping exam source.",
                sorted(exam_header),
            )
            include_exam = False

        # ---------------- union all event sources ----------------
        LOGGER.info("Concatenating event sources.")
        union_parts = [
            "SELECT subject_id, hadm_id, event_time, code_id, code_type, aux_code, description FROM events_diagnosis",
            "SELECT subject_id, hadm_id, event_time, code_id, code_type, aux_code, description FROM events_lab",
            "SELECT subject_id, hadm_id, event_time, code_id, code_type, aux_code, description FROM events_medication",
            "SELECT subject_id, hadm_id, event_time, code_id, code_type, aux_code, description FROM events_chart",
        ]
        if include_exam:
            union_parts.append(
                "SELECT subject_id, hadm_id, event_time, code_id, code_type, aux_code, description FROM events_exam"
            )
        con.execute("CREATE OR REPLACE TABLE events_raw AS " + "\nUNION ALL\n".join(union_parts))

        # ---------------- cohort filters (mirror MIMIC pipeline) ----------------
        con.execute(
            """
            CREATE OR REPLACE TABLE events_adm AS
            SELECT e.* FROM events_raw e
            JOIN admission_subjects a ON e.subject_id = a.subject_id
            """
        )
        con.execute(
            "CREATE OR REPLACE TABLE events_nonnull AS SELECT * FROM events_adm WHERE event_time IS NOT NULL"
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE subjects_with_hadm AS
            SELECT DISTINCT subject_id FROM events_nonnull WHERE hadm_id IS NOT NULL
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE events_hadm AS
            SELECT e.* FROM events_nonnull e
            JOIN subjects_with_hadm s ON e.subject_id = s.subject_id
            """
        )

        LOGGER.info("Computing timestamp features.")
        con.execute(
            """
            CREATE OR REPLACE TABLE events_time AS
            WITH base AS (
                SELECT
                    subject_id, hadm_id, event_time, code_id, code_type, aux_code, description,
                    (EPOCH(event_time) - EPOCH(MIN(event_time) OVER (PARTITION BY subject_id))) / 86400.0 AS timestamp_days
                FROM events_hadm
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
                FROM base
            )
            SELECT
                subject_id, hadm_id, event_time, code_id, code_type, aux_code, description,
                CAST(timestamp_days AS DOUBLE) AS timestamp_days,
                CASE WHEN rn = 1 THEN 0.0
                     ELSE LN(1 + GREATEST(timestamp_days - COALESCE(prev_ts, 0.0), 0.0))
                END AS log_delta_t
            FROM step
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE subjects_ge5 AS
            SELECT subject_id FROM events_time GROUP BY subject_id HAVING COUNT(*) >= 5
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE events_ge5 AS
            SELECT e.* FROM events_time e JOIN subjects_ge5 s ON e.subject_id = s.subject_id
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE subjects_multi_ts AS
            SELECT subject_id FROM events_ge5 GROUP BY subject_id HAVING COUNT(DISTINCT timestamp_days) > 1
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE events_filtered AS
            SELECT e.* FROM events_ge5 e JOIN subjects_multi_ts s ON e.subject_id = s.subject_id
            """
        )

        # ---------------- demographics ----------------
        con.execute(
            """
            CREATE OR REPLACE TABLE patients_demo AS
            SELECT
                subject_id,
                CASE WHEN gender = 'M' THEN CAST(1 AS TINYINT) ELSE CAST(0 AS TINYINT) END AS sex,
                dob
            FROM patients_raw
            """
        )

        LOGGER.info("Joining demographics and finalizing schema.")
        con.execute(
            """
            CREATE OR REPLACE TABLE final_events AS
            SELECT
                CAST(e.subject_id AS BIGINT) AS subject_id,
                CAST(e.hadm_id AS BIGINT) AS hadm_id,
                CAST(e.event_time AS TIMESTAMP) AS event_time,
                CAST(e.code_id AS VARCHAR) AS code_id,
                CAST(e.code_type AS VARCHAR) AS code_type,
                CAST(e.timestamp_days AS DOUBLE) AS timestamp_days,
                CAST(e.log_delta_t AS DOUBLE) AS log_delta_t,
                CAST((EPOCH(e.event_time) - EPOCH(p.dob)) / 86400.0 AS DOUBLE) AS age_at_event_days,
                CAST(COALESCE(p.sex, 0) AS TINYINT) AS sex,
                CAST('UNKNOWN' AS VARCHAR) AS race,
                CAST(e.aux_code AS VARCHAR) AS aux_code,
                CAST(e.description AS VARCHAR) AS description
            FROM events_filtered e
            LEFT JOIN patients_demo p ON e.subject_id = p.subject_id
            WHERE e.event_time IS NOT NULL
            """
        )

        # ---------------- write outputs ----------------
        suffix = "test" if args.test_mode else "full"
        output_file = output_dir / f"patient_events_{suffix}_pic.parquet"
        # Canonical name used downstream by script 2 (full run only keeps the stable name too).
        canonical = output_dir / "patient_events_pic.parquet"
        vocab_file = output_dir / "code_vocab_raw_pic.json"
        loinc_file = output_dir / "pic_lab_loinc_map.json"

        escaped_output = str(output_file).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT subject_id, hadm_id, event_time, code_id, code_type,
                       timestamp_days, log_delta_t, age_at_event_days, sex, race,
                       aux_code, description
                FROM final_events
                ORDER BY subject_id, event_time, code_id
            ) TO '{escaped_output}' (FORMAT PARQUET)
            """
        )
        if output_file != canonical:
            import shutil
            shutil.copyfile(output_file, canonical)

        vocab_rows = con.execute(
            "SELECT code_id, COUNT(*) AS freq FROM final_events GROUP BY code_id ORDER BY freq DESC"
        ).fetchall()
        with vocab_file.open("w", encoding="utf-8") as f:
            json.dump({str(c): int(v) for c, v in vocab_rows}, f, ensure_ascii=True, indent=2)

        # LAB -> LOINC sidecar (only labs that actually appear, with a LOINC code).
        loinc_rows = con.execute(
            """
            SELECT DISTINCT 'LAB_' || dl.itemid AS code_id, dl.loinc_code
            FROM d_labitems dl
            WHERE dl.loinc_code IS NOT NULL AND TRIM(dl.loinc_code) <> ''
              AND 'LAB_' || dl.itemid IN (SELECT DISTINCT code_id FROM final_events WHERE code_id LIKE 'LAB_%')
            """
        ).fetchall()
        with loinc_file.open("w", encoding="utf-8") as f:
            json.dump({str(c): str(l) for c, l in loinc_rows}, f, ensure_ascii=True, indent=2)

        # ---------------- sanity prints ----------------
        n_raw = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_raw")
        n_adm = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_adm")
        n_hadm = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_hadm")
        n_ge5 = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_ge5")
        n_final = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM final_events")
        total_rows = _scalar_int(con, "SELECT COUNT(*) FROM final_events")
        unique_codes = _scalar_int(con, "SELECT COUNT(DISTINCT code_id) FROM final_events")
        ts_min = _scalar_float(con, "SELECT MIN(timestamp_days) FROM final_events")
        ts_max = _scalar_float(con, "SELECT MAX(timestamp_days) FROM final_events")
        age_min = _scalar_float(con, "SELECT MIN(age_at_event_days) FROM final_events")
        age_max = _scalar_float(con, "SELECT MAX(age_at_event_days) FROM final_events")
        neg_age = _scalar_int(con, "SELECT COUNT(*) FROM final_events WHERE age_at_event_days < 0")
        code_type_counts = con.execute(
            "SELECT code_type, COUNT(*) c FROM final_events GROUP BY code_type ORDER BY c DESC"
        ).fetchall()

        print(f"Total rows: {total_rows:,}")
        print(f"Unique subject_ids: {n_final:,}")
        print(f"Unique raw code_ids: {unique_codes:,}")
        print(f"Timestamp range (days): min={ts_min:.4f}, max={ts_max:.4f}")
        print(f"age_at_event_days range: min={age_min:.2f}, max={age_max:.2f}, negatives={neg_age:,}")
        print("Row counts per code_type:")
        for ct, c in code_type_counts:
            print(f"  - {ct}: {int(c):,}")
        print("Subjects retained through filters:")
        print(f"  - with any event: {n_raw:,}")
        print(f"  - with admission record: {n_adm:,}")
        print(f"  - with hadm_id event: {n_hadm:,}")
        print(f"  - >=5 events: {n_ge5:,}")
        print(f"  - >1 distinct timestamp (final): {n_final:,}")
        print(f"LAB->LOINC sidecar entries: {len(loinc_rows):,}")

        LOGGER.info("Wrote %s (and canonical %s)", output_file, canonical)
        LOGGER.info("Wrote %s, %s", vocab_file, loinc_file)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
