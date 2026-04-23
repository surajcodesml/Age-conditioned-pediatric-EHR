#!/usr/bin/env python3
"""Build a unified patient event table from MIMIC-IV v3.1 hosp + icu modules."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

LOGGER = logging.getLogger("build_event_table")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _default_ed_root(mimic_root: Path) -> Path:
    """Sibling MIMIC-IV-ED folder under the same PhysioNet files directory as hosp/icu."""
    # mimic_root: .../files/mimiciv/3.1  ->  .../files/mimiciv-ed/2.2/ed
    files_dir = mimic_root.parent.parent
    return files_dir / "mimiciv-ed" / "2.2" / "ed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified event table for TALE-EHR preprocessing.")
    parser.add_argument(
        "--mimic_root",
        type=Path,
        default=Path("/home/suraj/Data/MIMIC-IV/physionet.org/files/mimiciv/3.1"),
        help="Path to MIMIC-IV v3.1 root containing hosp/ and icu/.",
    )
    parser.add_argument(
        "--ed_root",
        type=Path,
        default=None,
        help="Path to MIMIC-IV-ED ed/ folder (CSV.gz tables). Default: sibling of mimiciv under files/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed",
        help="Directory for output parquet/json files.",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="If set, sample 1,000 patients and write patient_events_test.parquet.",
    )
    args = parser.parse_args()
    if args.ed_root is None:
        args.ed_root = _default_ed_root(args.mimic_root)
    return args


def _gzip_csv_header_columns(filepath: Path) -> list[str]:
    with gzip.open(filepath, "rt", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def _execute_csv_sql(con: duckdb.DuckDBPyConnection, sql_template: str, filepath: Path) -> None:
    action = f"loading {filepath.name}"
    start = time.monotonic()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(60):
            elapsed_min = (time.monotonic() - start) / 60.0
            LOGGER.info("%s still running... elapsed %.1f min", action, elapsed_min)

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    LOGGER.info("Started %s", action)
    try:
        escaped = str(filepath).replace("'", "''")
        direct_source = f"read_csv_auto('{escaped}')"
        try:
            con.execute(sql_template.format(source=direct_source))
            return
        except duckdb.IOException as exc:
            if "GZIP" not in str(exc).upper():
                raise

        with tempfile.TemporaryDirectory() as tmp_dir:
            fifo_path = Path(tmp_dir) / "stream.csv"
            os.mkfifo(fifo_path)
            proc = subprocess.Popen(
                ["bash", "-lc", f"gzip -dc \"{filepath}\" > \"{fifo_path}\""],
                stderr=subprocess.PIPE,
            )
            try:
                escaped_fifo = str(fifo_path).replace("'", "''")
                fifo_source = f"read_csv_auto('{escaped_fifo}')"
                con.execute(sql_template.format(source=fifo_source))
            finally:
                _, stderr = proc.communicate()
                if proc.returncode not in (0, 2):
                    stderr_text = stderr.decode("utf-8", errors="ignore") if stderr else ""
                    raise RuntimeError(f"gzip stream failed for {filepath}: {stderr_text}")
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=0.1)
        elapsed_min = (time.monotonic() - start) / 60.0
        LOGGER.info("Finished %s in %.1f min", action, elapsed_min)


def _scalar_int(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(con.execute(query).fetchone()[0])


def _scalar_float(con: duckdb.DuckDBPyConnection, query: str) -> float:
    value = con.execute(query).fetchone()[0]
    return float(value) if value is not None else float("nan")


def main() -> int:
    setup_logging()
    args = parse_args()

    mimic_root = args.mimic_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    hosp_dir = mimic_root / "hosp"
    icu_dir = mimic_root / "icu"
    required_paths = [
        hosp_dir / "patients.csv.gz",
        hosp_dir / "admissions.csv.gz",
        hosp_dir / "diagnoses_icd.csv.gz",
        hosp_dir / "procedures_icd.csv.gz",
        hosp_dir / "prescriptions.csv.gz",
        hosp_dir / "labevents.csv.gz",
        hosp_dir / "d_labitems.csv.gz",
        hosp_dir / "drgcodes.csv.gz",
        hosp_dir / "hcpcsevents.csv.gz",
        icu_dir / "chartevents.csv.gz",
        icu_dir / "inputevents.csv.gz",
        icu_dir / "outputevents.csv.gz",
        icu_dir / "procedureevents.csv.gz",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    db_path = output_dir / "build_event_table.duckdb"
    wal_path = output_dir / "build_event_table.duckdb.wal"
    if db_path.exists():
        db_path.unlink()
    if wal_path.exists():
        wal_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        # 32GB machine: cap duckdb memory below system max and spill intermediates to disk.
        con.execute("PRAGMA memory_limit='24GB'")
        escaped_temp_dir = str(temp_dir).replace("'", "''")
        con.execute(f"PRAGMA temp_directory='{escaped_temp_dir}'")
        con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 8) - 2)}")

        LOGGER.info("Starting build_event_table with test_mode=%s", args.test_mode)

        LOGGER.info("Loading patients table.")
        _execute_csv_sql(
            con,
            """
            CREATE OR REPLACE TABLE patients_raw AS
            SELECT
                TRY_CAST(subject_id AS BIGINT) AS subject_id,
                CAST(gender AS VARCHAR) AS gender,
                TRY_CAST(anchor_age AS DOUBLE) AS anchor_age,
                TRY_CAST(anchor_year AS DOUBLE) AS anchor_year
            FROM {source}
            WHERE TRY_CAST(subject_id AS BIGINT) IS NOT NULL
            """,
            hosp_dir / "patients.csv.gz",
        )

        if args.test_mode:
            patient_ids = con.execute("SELECT subject_id FROM patients_raw").df()["subject_id"]
            if len(patient_ids) < 1000:
                raise ValueError(f"Requested 1000 sampled patients but only found {len(patient_ids)} in patients table.")
            sampled_ids = patient_ids.sample(n=1000, random_state=42).astype("int64").tolist()
            con.execute("CREATE OR REPLACE TEMP TABLE sample_subjects(subject_id BIGINT)")
            con.executemany("INSERT INTO sample_subjects VALUES (?)", [(sid,) for sid in sampled_ids])
            LOGGER.info("Sampled %d subject_ids for test mode.", len(sampled_ids))
            subject_join = "JOIN sample_subjects ss ON TRY_CAST(t.subject_id AS BIGINT) = ss.subject_id"
        else:
            subject_join = ""

        LOGGER.info("Loading admissions table.")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE admissions AS
            SELECT
                TRY_CAST(t.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(t.hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(t.admittime AS TIMESTAMP) AS admittime,
                CAST(t.race AS VARCHAR) AS race
            FROM {{source}} t
            {subject_join}
            WHERE TRY_CAST(t.subject_id AS BIGINT) IS NOT NULL
            """,
            hosp_dir / "admissions.csv.gz",
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE admission_subjects AS
            SELECT DISTINCT subject_id
            FROM admissions
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE race_by_subject AS
            SELECT subject_id, race
            FROM (
                SELECT
                    subject_id,
                    race,
                    ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime DESC) AS rn
                FROM admissions
                WHERE race IS NOT NULL AND TRIM(race) != ''
            ) x
            WHERE rn = 1
            """
        )

        LOGGER.info("Loading d_labitems table (dynamic lab item mapping).")
        _execute_csv_sql(
            con,
            """
            CREATE OR REPLACE TABLE d_labitems AS
            SELECT DISTINCT TRY_CAST(itemid AS BIGINT) AS itemid
            FROM {source}
            WHERE TRY_CAST(itemid AS BIGINT) IS NOT NULL
            """,
            hosp_dir / "d_labitems.csv.gz",
        )

        diag_subject_join = "JOIN sample_subjects ss ON TRY_CAST(d.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        proc_subject_join = "JOIN sample_subjects ss ON TRY_CAST(p.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        med_subject_join = "JOIN sample_subjects ss ON TRY_CAST(m.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        lab_subject_join = "JOIN sample_subjects ss ON TRY_CAST(l.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        chart_subject_join = "JOIN sample_subjects ss ON TRY_CAST(c.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        drg_subject_join = "JOIN sample_subjects ss ON TRY_CAST(g.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        input_subject_join = "JOIN sample_subjects ss ON TRY_CAST(i.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        output_subject_join = "JOIN sample_subjects ss ON TRY_CAST(o.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        icuproc_subject_join = "JOIN sample_subjects ss ON TRY_CAST(pe.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
        hcpcs_subject_join = "JOIN sample_subjects ss ON TRY_CAST(h.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""

        LOGGER.info("Building event source: diagnoses_icd")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_diagnosis AS
            SELECT
                TRY_CAST(d.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(d.hadm_id AS BIGINT) AS hadm_id,
                a.admittime AS event_time,
                CASE
                    WHEN TRY_CAST(d.icd_version AS BIGINT) = 9 THEN 'ICD9_'
                    ELSE 'ICD10_'
                END || CAST(d.icd_code AS VARCHAR) AS code_id,
                'diagnosis' AS code_type
            FROM {{source}} d
            {diag_subject_join}
            JOIN admissions a ON TRY_CAST(d.hadm_id AS BIGINT) = a.hadm_id
            WHERE d.icd_code IS NOT NULL
            """,
            hosp_dir / "diagnoses_icd.csv.gz",
        )
        LOGGER.info("Diagnoses events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_diagnosis"))

        LOGGER.info("Building event source: procedures_icd")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_procedure AS
            SELECT
                TRY_CAST(p.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(p.hadm_id AS BIGINT) AS hadm_id,
                a.admittime AS event_time,
                CASE
                    WHEN TRY_CAST(p.icd_version AS BIGINT) = 9 THEN 'PROC9_'
                    ELSE 'PROC10_'
                END || CAST(p.icd_code AS VARCHAR) AS code_id,
                'procedure' AS code_type
            FROM {{source}} p
            {proc_subject_join}
            JOIN admissions a ON TRY_CAST(p.hadm_id AS BIGINT) = a.hadm_id
            WHERE p.icd_code IS NOT NULL
            """,
            hosp_dir / "procedures_icd.csv.gz",
        )
        LOGGER.info("Procedure events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_procedure"))

        LOGGER.info("Building event source: prescriptions")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_medication AS
            SELECT
                TRY_CAST(m.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(m.hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(m.starttime AS TIMESTAMP) AS event_time,
                'NDC_' || TRIM(CAST(m.ndc AS VARCHAR)) AS code_id,
                'medication' AS code_type
            FROM {{source}} m
            {med_subject_join}
            WHERE m.starttime IS NOT NULL
              AND m.ndc IS NOT NULL
              AND TRIM(CAST(m.ndc AS VARCHAR)) != ''
              AND TRIM(CAST(m.ndc AS VARCHAR)) != '0'
            """,
            hosp_dir / "prescriptions.csv.gz",
        )
        LOGGER.info("Medication events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_medication"))

        LOGGER.info("Building event source: labevents")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_lab AS
            SELECT
                TRY_CAST(l.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(l.hadm_id AS BIGINT) AS hadm_id,
                COALESCE(TRY_CAST(l.charttime AS TIMESTAMP), TRY_CAST(l.storetime AS TIMESTAMP)) AS event_time,
                'LAB_' || CAST(TRY_CAST(l.itemid AS BIGINT) AS VARCHAR) AS code_id,
                'lab' AS code_type
            FROM {{source}} l
            {lab_subject_join}
            JOIN d_labitems dl ON TRY_CAST(l.itemid AS BIGINT) = dl.itemid
            WHERE l.charttime IS NOT NULL OR l.storetime IS NOT NULL
            """,
            hosp_dir / "labevents.csv.gz",
        )
        LOGGER.info("Lab events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_lab"))

        LOGGER.info("Building event source: chartevents")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_chart AS
            SELECT
                TRY_CAST(c.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(c.hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(c.charttime AS TIMESTAMP) AS event_time,
                'CHART_' || CAST(TRY_CAST(c.itemid AS BIGINT) AS VARCHAR) AS code_id,
                'chart' AS code_type
            FROM {{source}} c
            {chart_subject_join}
            WHERE c.charttime IS NOT NULL
              AND c.value IS NOT NULL
              AND TRIM(CAST(c.value AS VARCHAR)) != ''
            """,
            icu_dir / "chartevents.csv.gz",
        )
        LOGGER.info("Chart events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_chart"))

        LOGGER.info("Building event source: drgcodes")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_drg AS
            SELECT
                TRY_CAST(g.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(g.hadm_id AS BIGINT) AS hadm_id,
                a.admittime AS event_time,
                'DRG_' || CAST(g.drg_code AS VARCHAR) AS code_id,
                'drg' AS code_type
            FROM {{source}} g
            {drg_subject_join}
            JOIN admissions a ON TRY_CAST(g.hadm_id AS BIGINT) = a.hadm_id
            WHERE g.drg_code IS NOT NULL
            """,
            hosp_dir / "drgcodes.csv.gz",
        )
        LOGGER.info("DRG events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_drg"))

        LOGGER.info("Building event source: inputevents")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_input AS
            SELECT
                TRY_CAST(i.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(i.hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(i.starttime AS TIMESTAMP) AS event_time,
                'INPUT_' || CAST(TRY_CAST(i.itemid AS BIGINT) AS VARCHAR) AS code_id,
                'input' AS code_type
            FROM {{source}} i
            {input_subject_join}
            WHERE i.starttime IS NOT NULL AND i.itemid IS NOT NULL
            """,
            icu_dir / "inputevents.csv.gz",
        )
        LOGGER.info("Input events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_input"))

        LOGGER.info("Building event source: outputevents")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_output AS
            SELECT
                TRY_CAST(o.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(o.hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(o.charttime AS TIMESTAMP) AS event_time,
                'OUTPUT_' || CAST(TRY_CAST(o.itemid AS BIGINT) AS VARCHAR) AS code_id,
                'output' AS code_type
            FROM {{source}} o
            {output_subject_join}
            WHERE o.charttime IS NOT NULL AND o.itemid IS NOT NULL
            """,
            icu_dir / "outputevents.csv.gz",
        )
        LOGGER.info("Output events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_output"))

        LOGGER.info("Building event source: procedureevents")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_icu_procedure AS
            SELECT
                TRY_CAST(pe.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(pe.hadm_id AS BIGINT) AS hadm_id,
                TRY_CAST(pe.starttime AS TIMESTAMP) AS event_time,
                'ICUPROC_' || CAST(TRY_CAST(pe.itemid AS BIGINT) AS VARCHAR) AS code_id,
                'icu_procedure' AS code_type
            FROM {{source}} pe
            {icuproc_subject_join}
            WHERE pe.starttime IS NOT NULL AND pe.itemid IS NOT NULL
            """,
            icu_dir / "procedureevents.csv.gz",
        )
        LOGGER.info("ICU procedure events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_icu_procedure"))

        LOGGER.info("Building event source: hcpcsevents")
        _execute_csv_sql(
            con,
            f"""
            CREATE OR REPLACE TABLE events_hcpcs AS
            SELECT
                TRY_CAST(h.subject_id AS BIGINT) AS subject_id,
                TRY_CAST(h.hadm_id AS BIGINT) AS hadm_id,
                a.admittime AS event_time,
                'HCPCS_' || CAST(h.hcpcs_cd AS VARCHAR) AS code_id,
                'hcpcs' AS code_type
            FROM {{source}} h
            {hcpcs_subject_join}
            JOIN admissions a ON TRY_CAST(h.hadm_id AS BIGINT) = a.hadm_id
            WHERE h.hcpcs_cd IS NOT NULL
            """,
            hosp_dir / "hcpcsevents.csv.gz",
        )
        LOGGER.info("HCPCS events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_hcpcs"))

        ed_union_fragments: list[str] = []
        ed_dir = args.ed_root
        if not ed_dir.is_dir():
            LOGGER.warning(
                "MIMIC-IV-ED directory not found at %s; skipping ED event sources.",
                ed_dir,
            )
        else:
            ed_stays_path = ed_dir / "edstays.csv.gz"
            if not ed_stays_path.exists():
                LOGGER.warning(
                    "MIMIC-IV-ED edstays.csv.gz missing under %s; skipping ED event sources.",
                    ed_dir,
                )
            else:
                ed_stay_subject_join = (
                    "JOIN sample_subjects ss ON TRY_CAST(t.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
                )
                LOGGER.info("Loading ED edstays for joins.")
                _execute_csv_sql(
                    con,
                    f"""
                    CREATE OR REPLACE TABLE ed_edstays AS
                    SELECT
                        TRY_CAST(t.subject_id AS BIGINT) AS subject_id,
                        TRY_CAST(t.stay_id AS BIGINT) AS stay_id,
                        TRY_CAST(t.hadm_id AS BIGINT) AS hadm_id,
                        TRY_CAST(t.intime AS TIMESTAMP) AS intime
                    FROM {{source}} t
                    {ed_stay_subject_join}
                    WHERE TRY_CAST(t.subject_id AS BIGINT) IS NOT NULL
                      AND TRY_CAST(t.stay_id AS BIGINT) IS NOT NULL
                    """,
                    ed_stays_path,
                )
                ed_diag_subject_join = (
                    "JOIN sample_subjects ss ON TRY_CAST(d.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
                )
                ed_med_subject_join = (
                    "JOIN sample_subjects ss ON TRY_CAST(m.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
                )
                ed_pyx_subject_join = (
                    "JOIN sample_subjects ss ON TRY_CAST(px.subject_id AS BIGINT) = ss.subject_id" if args.test_mode else ""
                )

                ed_diag_path = ed_dir / "diagnosis.csv.gz"
                if ed_diag_path.exists():
                    LOGGER.info("Building event source: ED diagnosis")
                    _execute_csv_sql(
                        con,
                        f"""
                        CREATE OR REPLACE TABLE events_ed_diagnosis AS
                        SELECT
                            TRY_CAST(d.subject_id AS BIGINT) AS subject_id,
                            TRY_CAST(es.hadm_id AS BIGINT) AS hadm_id,
                            TRY_CAST(es.intime AS TIMESTAMP) AS event_time,
                            CASE
                                WHEN TRY_CAST(d.icd_version AS BIGINT) = 9 THEN 'ICD9_'
                                ELSE 'ICD10_'
                            END || CAST(d.icd_code AS VARCHAR) AS code_id,
                            'diagnosis' AS code_type
                        FROM {{source}} d
                        {ed_diag_subject_join}
                        JOIN ed_edstays es ON TRY_CAST(d.stay_id AS BIGINT) = es.stay_id
                        WHERE d.icd_code IS NOT NULL
                        """,
                        ed_diag_path,
                    )
                    LOGGER.info("ED diagnosis events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_ed_diagnosis"))
                    ed_union_fragments.append(
                        "SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_ed_diagnosis"
                    )
                else:
                    LOGGER.warning("ED diagnosis.csv.gz not found under %s; skipping.", ed_dir)

                ed_med_path = ed_dir / "medrecon.csv.gz"
                if ed_med_path.exists():
                    LOGGER.info("Building event source: ED medrecon")
                    _execute_csv_sql(
                        con,
                        f"""
                        CREATE OR REPLACE TABLE events_ed_medrecon AS
                        SELECT
                            TRY_CAST(m.subject_id AS BIGINT) AS subject_id,
                            TRY_CAST(es.hadm_id AS BIGINT) AS hadm_id,
                            TRY_CAST(m.charttime AS TIMESTAMP) AS event_time,
                            'NDC_' || TRIM(CAST(m.ndc AS VARCHAR)) AS code_id,
                            'medication' AS code_type
                        FROM {{source}} m
                        {ed_med_subject_join}
                        JOIN ed_edstays es ON TRY_CAST(m.stay_id AS BIGINT) = es.stay_id
                        WHERE m.charttime IS NOT NULL
                          AND m.ndc IS NOT NULL
                          AND TRIM(CAST(m.ndc AS VARCHAR)) != ''
                          AND TRIM(CAST(m.ndc AS VARCHAR)) != '0'
                        """,
                        ed_med_path,
                    )
                    LOGGER.info("ED medrecon events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_ed_medrecon"))
                    ed_union_fragments.append(
                        "SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_ed_medrecon"
                    )
                else:
                    LOGGER.warning("ED medrecon.csv.gz not found under %s; skipping.", ed_dir)

                ed_pyx_path = ed_dir / "pyxis.csv.gz"
                if ed_pyx_path.exists():
                    pyx_cols = {c.strip().lower() for c in _gzip_csv_header_columns(ed_pyx_path)}
                    if "ndc" in pyx_cols:
                        LOGGER.info("Building event source: ED pyxis (NDC column present)")
                        _execute_csv_sql(
                            con,
                            f"""
                            CREATE OR REPLACE TABLE events_ed_pyxis AS
                            SELECT
                                TRY_CAST(px.subject_id AS BIGINT) AS subject_id,
                                TRY_CAST(es.hadm_id AS BIGINT) AS hadm_id,
                                TRY_CAST(px.charttime AS TIMESTAMP) AS event_time,
                                'NDC_' || TRIM(CAST(px.ndc AS VARCHAR)) AS code_id,
                                'medication' AS code_type
                            FROM {{source}} px
                            {ed_pyx_subject_join}
                            JOIN ed_edstays es ON TRY_CAST(px.stay_id AS BIGINT) = es.stay_id
                            WHERE px.charttime IS NOT NULL
                              AND px.ndc IS NOT NULL
                              AND TRIM(CAST(px.ndc AS VARCHAR)) != ''
                              AND TRIM(CAST(px.ndc AS VARCHAR)) != '0'
                            """,
                            ed_pyx_path,
                        )
                    else:
                        LOGGER.info("Building event source: ED pyxis (GSN)")
                        _execute_csv_sql(
                            con,
                            f"""
                            CREATE OR REPLACE TABLE events_ed_pyxis AS
                            SELECT
                                TRY_CAST(px.subject_id AS BIGINT) AS subject_id,
                                TRY_CAST(es.hadm_id AS BIGINT) AS hadm_id,
                                TRY_CAST(px.charttime AS TIMESTAMP) AS event_time,
                                'PYXIS_' || TRIM(CAST(px.gsn AS VARCHAR)) AS code_id,
                                'pyxis_medication' AS code_type
                            FROM {{source}} px
                            {ed_pyx_subject_join}
                            JOIN ed_edstays es ON TRY_CAST(px.stay_id AS BIGINT) = es.stay_id
                            WHERE px.charttime IS NOT NULL
                              AND px.gsn IS NOT NULL
                              AND TRIM(CAST(px.gsn AS VARCHAR)) != ''
                            """,
                            ed_pyx_path,
                        )
                    LOGGER.info("ED pyxis events: %d", _scalar_int(con, "SELECT COUNT(*) FROM events_ed_pyxis"))
                    ed_union_fragments.append(
                        "SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_ed_pyxis"
                    )
                else:
                    LOGGER.warning("ED pyxis.csv.gz not found under %s; skipping.", ed_dir)

        LOGGER.info("Concatenating all event sources.")
        union_sql = """
            CREATE OR REPLACE TABLE events_raw AS
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_diagnosis
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_procedure
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_medication
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_lab
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_chart
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_drg
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_input
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_output
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_icu_procedure
            UNION ALL
            SELECT subject_id, hadm_id, event_time, code_id, code_type FROM events_hcpcs
        """
        if ed_union_fragments:
            union_sql += "\n            UNION ALL\n            " + "\n            UNION ALL\n            ".join(
                ed_union_fragments
            )
        con.execute(union_sql)

        con.execute(
            """
            CREATE OR REPLACE TABLE events_adm AS
            SELECT e.*
            FROM events_raw e
            JOIN admission_subjects a ON e.subject_id = a.subject_id
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE events_nonnull AS
            SELECT *
            FROM events_adm
            WHERE event_time IS NOT NULL
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE subjects_with_hadm AS
            SELECT DISTINCT subject_id
            FROM events_nonnull
            WHERE hadm_id IS NOT NULL
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE events_hadm AS
            SELECT e.*
            FROM events_nonnull e
            JOIN subjects_with_hadm s ON e.subject_id = s.subject_id
            """
        )

        LOGGER.info("Computing timestamp features.")
        con.execute(
            """
            CREATE OR REPLACE TABLE events_time AS
            WITH base AS (
                SELECT
                    subject_id,
                    hadm_id,
                    event_time,
                    code_id,
                    code_type,
                    (EPOCH(event_time) - EPOCH(MIN(event_time) OVER (PARTITION BY subject_id))) / 86400.0 AS timestamp_days
                FROM events_hadm
            ),
            step AS (
                SELECT
                    *,
                    LAG(timestamp_days) OVER (
                        PARTITION BY subject_id
                        ORDER BY event_time, code_id, code_type, hadm_id
                    ) AS prev_ts,
                    ROW_NUMBER() OVER (
                        PARTITION BY subject_id
                        ORDER BY event_time, code_id, code_type, hadm_id
                    ) AS rn
                FROM base
            )
            SELECT
                subject_id,
                hadm_id,
                event_time,
                code_id,
                code_type,
                CAST(timestamp_days AS DOUBLE) AS timestamp_days,
                CASE
                    WHEN rn = 1 THEN 0.0
                    ELSE LN(1 + GREATEST(timestamp_days - COALESCE(prev_ts, 0.0), 0.0))
                END AS log_delta_t
            FROM step
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE subjects_ge5 AS
            SELECT subject_id
            FROM events_time
            GROUP BY subject_id
            HAVING COUNT(*) >= 5
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE events_ge5 AS
            SELECT e.*
            FROM events_time e
            JOIN subjects_ge5 s ON e.subject_id = s.subject_id
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE subjects_multi_ts AS
            SELECT subject_id
            FROM events_ge5
            GROUP BY subject_id
            HAVING COUNT(DISTINCT timestamp_days) > 1
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE events_filtered AS
            SELECT e.*
            FROM events_ge5 e
            JOIN subjects_multi_ts s ON e.subject_id = s.subject_id
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE patients_demo AS
            SELECT
                p.subject_id,
                CASE WHEN p.gender = 'M' THEN CAST(1 AS TINYINT) ELSE CAST(0 AS TINYINT) END AS sex,
                TRY_STRPTIME(
                    CAST(CAST(ROUND(p.anchor_year - p.anchor_age) AS BIGINT) AS VARCHAR) || '-07-01',
                    '%Y-%m-%d'
                ) AS dob
            FROM patients_raw p
            """
        )

        LOGGER.info("Joining demographics and finalizing output schema.")
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
                CAST(r.race AS VARCHAR) AS race
            FROM events_filtered e
            LEFT JOIN patients_demo p ON e.subject_id = p.subject_id
            LEFT JOIN race_by_subject r ON e.subject_id = r.subject_id
            WHERE e.event_time IS NOT NULL
            """
        )

        output_file = output_dir / ("patient_events_test.parquet" if args.test_mode else "patient_events_full.parquet")
        vocab_file = output_dir / "code_vocab_raw.json"
        escaped_output = str(output_file).replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT
                    subject_id,
                    hadm_id,
                    event_time,
                    code_id,
                    code_type,
                    timestamp_days,
                    log_delta_t,
                    age_at_event_days,
                    sex,
                    race
                FROM final_events
                ORDER BY subject_id, event_time, code_id
            ) TO '{escaped_output}' (FORMAT PARQUET)
            """
        )

        vocab_rows = con.execute(
            """
            SELECT code_id, COUNT(*) AS freq
            FROM final_events
            GROUP BY code_id
            ORDER BY freq DESC
            """
        ).fetchall()
        vocab = {str(code): int(freq) for code, freq in vocab_rows}
        with vocab_file.open("w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=True, indent=2)

        n_subjects_raw = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_raw")
        n_subjects_adm = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_adm")
        n_subjects_nonnull = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_nonnull")
        n_subjects_hadm = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_hadm")
        n_subjects_ge5 = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM events_ge5")
        n_subjects_final = _scalar_int(con, "SELECT COUNT(DISTINCT subject_id) FROM final_events")
        null_event_subjects = _scalar_int(
            con,
            "SELECT COUNT(DISTINCT subject_id) FROM events_adm WHERE event_time IS NULL",
        )

        dropped = {
            "excluded_no_admission_record": n_subjects_raw - n_subjects_adm,
            "excluded_null_event_time": null_event_subjects,
            "excluded_no_event_with_hadm_id": n_subjects_nonnull - n_subjects_hadm,
            "excluded_fewer_than_5_events": n_subjects_hadm - n_subjects_ge5,
            "excluded_single_unique_timestamp": n_subjects_ge5 - n_subjects_final,
        }

        total_rows = _scalar_int(con, "SELECT COUNT(*) FROM final_events")
        unique_subjects = n_subjects_final
        unique_codes = _scalar_int(con, "SELECT COUNT(DISTINCT code_id) FROM final_events")
        ts_min = _scalar_float(con, "SELECT MIN(timestamp_days) FROM final_events")
        ts_max = _scalar_float(con, "SELECT MAX(timestamp_days) FROM final_events")
        code_type_counts = con.execute(
            """
            SELECT code_type, COUNT(*) AS cnt
            FROM final_events
            GROUP BY code_type
            ORDER BY cnt DESC
            """
        ).fetchall()

        print(f"Total rows: {total_rows:,}")
        print(f"Unique subject_ids: {unique_subjects:,}")
        print(f"Unique code_ids: {unique_codes:,}")
        print(f"Timestamp range (days): min={ts_min:.6f}, max={ts_max:.6f}")
        print("Row counts per code_type:")
        for code_type, count in code_type_counts:
            print(f"  - {code_type}: {int(count):,}")
        print("Patients dropped and why:")
        for reason, count in dropped.items():
            print(f"  - {reason}: {int(count):,}")

        LOGGER.info("Wrote output parquet: %s", output_file)
        LOGGER.info("Wrote code vocab: %s", vocab_file)
        LOGGER.info("Completed event table build.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
 