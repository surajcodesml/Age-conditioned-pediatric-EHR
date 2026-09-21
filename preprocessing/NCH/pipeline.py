"""NCH inventory, canonical events, SleepBank-indexed sequences, and audits.

CPU-only. Reuses Stage-1 vocabulary, PheCode maps, lag_to_tau, encode_race, and
frozen age/tau constants. Never writes under the MIMIC processed tree or the
active ``adkm_s0`` run directory.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import mapping, paths
from .mapping import MappedCode

DAYS_PER_YEAR = 365.25
WEEK_DAYS = 7.0
MAX_SEQ_LEN = 1024

PARSE_TS = """
COALESCE(
  try_strptime(TRIM(CAST({col} AS VARCHAR)), '%Y-%m-%d %H:%M:%S'),
  try_strptime(TRIM(CAST({col} AS VARCHAR)), '%Y-%m-%d %H:%M:%S.%f'),
  try_strptime(TRIM(CAST({col} AS VARCHAR)), '%Y-%m-%d'),
  try_strptime(TRIM(CAST({col} AS VARCHAR)), '%m/%d/%Y %H:%M:%S'),
  try_strptime(TRIM(CAST({col} AS VARCHAR)), '%m/%d/%Y %H:%M'),
  try_strptime(TRIM(CAST({col} AS VARCHAR)), '%m/%d/%Y')
)
"""

CSV_OPTS = "ALL_VARCHAR=TRUE, HEADER=TRUE, quote='\"', escape='\"'"

HISTORY_CUTS_DAYS = {
    "ge_30d": 30,
    "ge_90d": 90,
    "ge_180d": 180,
    "ge_1y": 365.25,
    "ge_2y": 730.5,
    "ge_5y": 1826.25,
}

STAGE2_BANDS = (
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.0),
    (">=18", 18.0, float("inf")),
)


def connect(mem: str = "6GB", threads: int = 4):
    import duckdb

    paths.ensure_output_dirs()
    db = paths.WORK_DIR / "nch_stage2.duckdb"
    con = duckdb.connect(str(db))
    con.execute(f"PRAGMA memory_limit='{mem}'")
    con.execute(f"PRAGMA temp_directory='{(paths.WORK_DIR / 'duckdb_tmp').as_posix()}'")
    (paths.WORK_DIR / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA threads={int(threads)}")
    return con


def _csv(path: Path) -> str:
    return f"read_csv_auto('{path.as_posix()}', {CSV_OPTS})"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def load_vocab() -> dict[str, int]:
    with paths.VOCAB_PATH.open("r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def load_frozen_constants() -> dict:
    from stage1_mimic_pretrain.config import MIMIC_AGE_MEAN_YEARS, MIMIC_AGE_STD_YEARS

    tau_max = None
    age_mean, age_sd = float(MIMIC_AGE_MEAN_YEARS), float(MIMIC_AGE_STD_YEARS)
    if paths.STAGE1_RUN_CONFIG.exists():
        cfg = json.loads(paths.STAGE1_RUN_CONFIG.read_text())
        stats = (cfg.get("data") or {}).get("corpus_stats") or {}
        if stats.get("tau_max") is not None:
            tau_max = float(stats["tau_max"])
        at = (cfg.get("data") or {}).get("age_transform") or {}
        if at.get("mean") is not None:
            age_mean = float(at["mean"])
        if at.get("sd") is not None:
            age_sd = float(at["sd"])
    if tau_max is None and paths.CORPUS_STATS_PATH.exists():
        stats = json.loads(paths.CORPUS_STATS_PATH.read_text())
        tau_max = float(stats["tau_max"])
    return {
        "age_mean": age_mean,
        "age_sd": age_sd,
        "tau_max": tau_max,
        "week_days": WEEK_DAYS,
        "days_per_year": DAYS_PER_YEAR,
        "max_seq_len": MAX_SEQ_LEN,
        "pad_id": 0,
        "unk_model_id": 1,
    }


# --------------------------------------------------------------------------- #
# Inventory                                                                   #
# --------------------------------------------------------------------------- #
def inventory_nch(con) -> dict:
    from model_new.data import encode_race

    out: dict = {"root": str(paths.NCH_ROOT), "tables": {}, "psg_waveforms_local": {}}
    for name, path in paths.NCH_CSVS.items():
        src = _csv(path)
        n = int(con.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0])
        cols = con.execute(f"SELECT * FROM {src} LIMIT 0").df().columns.tolist()
        out["tables"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "n_rows": n,
            "columns": cols,
        }

    # Demographics
    demo = _csv(paths.NCH_CSVS["demographic"])
    out["patients"] = con.execute(f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT STUDY_PAT_ID) AS n_patients,
          SUM(CASE WHEN BIRTH_DATE IS NULL OR TRIM(BIRTH_DATE)='' THEN 1 ELSE 0 END) AS missing_dob,
          SUM(CASE WHEN PCORI_GENDER_CD IS NULL OR TRIM(PCORI_GENDER_CD)='' THEN 1 ELSE 0 END) AS missing_sex,
          SUM(CASE WHEN RACE_DESCR IS NULL OR TRIM(RACE_DESCR)='' THEN 1 ELSE 0 END) AS missing_race
        FROM {demo}
    """).df().to_dict("records")[0]
    out["sex_values"] = con.execute(
        f"SELECT PCORI_GENDER_CD, GENDER_DESCR, COUNT(*) n FROM {demo} GROUP BY 1,2 ORDER BY n DESC"
    ).df().to_dict("records")
    out["race_values"] = con.execute(
        f"SELECT PCORI_RACE_CD, RACE_DESCR, COUNT(*) n FROM {demo} GROUP BY 1,2 ORDER BY n DESC"
    ).df().to_dict("records")
    out["hispanic_values"] = con.execute(
        f"SELECT PCORI_HISPANIC_CD, ETHNICITY_DESCR, COUNT(*) n FROM {demo} GROUP BY 1,2 ORDER BY n DESC"
    ).df().to_dict("records")

    enc = _csv(paths.NCH_CSVS["encounter"])
    out["encounters"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT STUDY_ENC_ID) n_encounters,
               COUNT(DISTINCT STUDY_PAT_ID) n_patients,
               SUM(CASE WHEN ENCOUNTER_DATE IS NULL OR TRIM(ENCOUNTER_DATE)='' THEN 1 ELSE 0 END) missing_date
        FROM {enc}
    """).df().to_dict("records")[0]
    out["encounter_types"] = con.execute(
        f"SELECT ENCOUNTER_TYPE, COUNT(*) n FROM {enc} GROUP BY 1 ORDER BY n DESC LIMIT 30"
    ).df().to_dict("records")

    dx = _csv(paths.NCH_CSVS["diagnosis"])
    out["diagnoses"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT STUDY_PAT_ID) n_patients,
               COUNT(DISTINCT STUDY_ENC_ID) n_encounters,
               COUNT(DISTINCT DX_CODE) n_raw_codes,
               SUM(CASE WHEN DX_CODE IS NULL OR TRIM(DX_CODE)='' THEN 1 ELSE 0 END) missing_code,
               SUM(CASE WHEN DX_START_DATETIME IS NULL OR TRIM(DX_START_DATETIME)='' THEN 1 ELSE 0 END) missing_time
        FROM {dx}
    """).df().to_dict("records")[0]
    out["dx_code_types"] = con.execute(
        f"SELECT DX_CODE_TYPE, COUNT(*) n, COUNT(DISTINCT DX_CODE) n_codes FROM {dx} GROUP BY 1 ORDER BY n DESC"
    ).df().to_dict("records")
    out["dx_source_types"] = con.execute(
        f"SELECT DX_SOURCE_TYPE, COUNT(*) n FROM {dx} GROUP BY 1 ORDER BY n DESC LIMIT 20"
    ).df().to_dict("records")

    pr = _csv(paths.NCH_CSVS["procedure"])
    out["procedures"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT STUDY_PAT_ID) n_patients,
               COUNT(DISTINCT PROC_CODE) n_raw_codes,
               COUNT(DISTINCT PROC_ID_NCH) n_nch_local_ids,
               SUM(CASE WHEN PROC_CODE IS NULL OR TRIM(PROC_CODE)='' THEN 1 ELSE 0 END) missing_code,
               SUM(CASE WHEN PROCEDURE_DATETIME IS NULL OR TRIM(PROCEDURE_DATETIME)='' THEN 1 ELSE 0 END) missing_time
        FROM {pr}
    """).df().to_dict("records")[0]
    out["proc_code_types"] = con.execute(
        f"SELECT PROC_CODE_TYPE, COUNT(*) n, COUNT(DISTINCT PROC_CODE) n_codes FROM {pr} GROUP BY 1 ORDER BY n DESC"
    ).df().to_dict("records")

    hx = _csv(paths.NCH_CSVS["procedure_surg_hx"])
    out["procedure_surg_hx"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT STUDY_PAT_ID) n_patients,
               COUNT(DISTINCT PROC_CODE) n_proc_code, COUNT(DISTINCT CPT_CODE) n_cpt
        FROM {hx}
    """).df().to_dict("records")[0]

    med = _csv(paths.NCH_CSVS["medication"])
    out["medications"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT STUDY_PAT_ID) n_patients,
               COUNT(DISTINCT RXNORM_CODE) n_rxcui,
               SUM(CASE WHEN RXNORM_CODE IS NULL OR TRIM(RXNORM_CODE)='' OR TRIM(RXNORM_CODE)='0' THEN 1 ELSE 0 END) missing_rxcui,
               SUM(CASE WHEN MED_TAKEN_DATETIME IS NULL OR TRIM(MED_TAKEN_DATETIME)='' THEN 1 ELSE 0 END) missing_taken,
               SUM(CASE WHEN MED_START_DATETIME IS NULL OR TRIM(MED_START_DATETIME)='' THEN 1 ELSE 0 END) missing_start
        FROM {med}
    """).df().to_dict("records")[0]
    out["med_source_types"] = con.execute(
        f"SELECT MED_SOURCE_TYPE, COUNT(*) n FROM {med} GROUP BY 1 ORDER BY n DESC LIMIT 20"
    ).df().to_dict("records")

    meas = _csv(paths.NCH_CSVS["measurement"])
    out["measurements"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT STUDY_PAT_ID) n_patients,
               COUNT(DISTINCT MEAS_TYPE) n_types
        FROM {meas}
    """).df().to_dict("records")[0]
    out["measurement_types"] = con.execute(
        f"SELECT MEAS_TYPE, COUNT(*) n FROM {meas} GROUP BY 1 ORDER BY n DESC"
    ).df().to_dict("records")

    ss = _csv(paths.NCH_CSVS["sleep_study"])
    out["sleep_studies"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT STUDY_PAT_ID) n_patients,
               COUNT(DISTINCT SLEEP_STUDY_ID) n_studies,
               SUM(CASE WHEN SLEEP_STUDY_START_DATETIME IS NULL OR TRIM(SLEEP_STUDY_START_DATETIME)='' THEN 1 ELSE 0 END) missing_start,
               MIN(TRY_CAST(AGE_AT_SLEEP_STUDY_DAYS AS DOUBLE)) min_age_days,
               MAX(TRY_CAST(AGE_AT_SLEEP_STUDY_DAYS AS DOUBLE)) max_age_days,
               MEDIAN(TRY_CAST(AGE_AT_SLEEP_STUDY_DAYS AS DOUBLE)) median_age_days
        FROM {ss}
    """).df().to_dict("records")[0]
    out["patients_with_multiple_studies"] = int(con.execute(f"""
        SELECT COUNT(*) FROM (
          SELECT STUDY_PAT_ID FROM {ss} GROUP BY 1 HAVING COUNT(*) > 1
        )
    """).fetchone()[0])
    out["studies_per_patient"] = con.execute(f"""
        SELECT n_studies, COUNT(*) n_patients FROM (
          SELECT STUDY_PAT_ID, COUNT(*) n_studies FROM {ss} GROUP BY 1
        ) GROUP BY 1 ORDER BY 1
    """).df().to_dict("records")

    se = _csv(paths.NCH_CSVS["sleep_enc_id"])
    out["sleep_enc_id"] = con.execute(f"""
        SELECT COUNT(*) n_rows, COUNT(DISTINCT SLEEP_STUDY_ID) n_studies,
               COUNT(DISTINCT STUDY_ENC_ID) n_encounters
        FROM {se}
    """).df().to_dict("records")[0]

    # Duplicate keys
    out["duplicates"] = {
        "diagnosis_id": int(con.execute(
            f"SELECT COUNT(*) FROM (SELECT STUDY_DX_ID FROM {dx} GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0]),
        "procedure_id": int(con.execute(
            f"SELECT COUNT(*) FROM (SELECT STUDY_PROC_ID FROM {pr} GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0]),
        "medication_id": int(con.execute(
            f"SELECT COUNT(*) FROM (SELECT STUDY_MED_ID FROM {med} GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0]),
        "sleep_study_id": int(con.execute(
            f"SELECT COUNT(*) FROM (SELECT SLEEP_STUDY_ID FROM {ss} GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0]),
    }

    # Date ranges (shifted/deidentified)
    out["date_ranges"] = {}
    out["date_ranges"]["diagnosis"] = con.execute(f"""
        SELECT MIN({PARSE_TS.format(col='DX_START_DATETIME')}) min_t,
               MAX({PARSE_TS.format(col='DX_START_DATETIME')}) max_t
        FROM {dx}
    """).df().to_dict("records")[0]
    out["date_ranges"]["encounter"] = con.execute(f"""
        SELECT MIN({PARSE_TS.format(col='ENCOUNTER_DATE')}) min_t,
               MAX({PARSE_TS.format(col='ENCOUNTER_DATE')}) max_t
        FROM {enc}
    """).df().to_dict("records")[0]
    out["date_ranges"]["sleep_study"] = con.execute(f"""
        SELECT MIN({PARSE_TS.format(col='SLEEP_STUDY_START_DATETIME')}) min_t,
               MAX({PARSE_TS.format(col='SLEEP_STUDY_START_DATETIME')}) max_t
        FROM {ss}
    """).df().to_dict("records")[0]
    out["date_ranges"]["birth"] = con.execute(f"""
        SELECT MIN({PARSE_TS.format(col='BIRTH_DATE')}) min_t,
               MAX({PARSE_TS.format(col='BIRTH_DATE')}) max_t
        FROM {demo}
    """).df().to_dict("records")[0]

    # Local PSG waveforms: sample only; full RECORDS lists 3984 studies.
    n_edf = len(list(paths.NCH_SLEEP_DATA.glob("*.edf"))) if paths.NCH_SLEEP_DATA.exists() else 0
    n_records = sum(1 for _ in paths.NCH_RECORDS.open()) if paths.NCH_RECORDS.exists() else 0
    out["psg_waveforms_local"] = {
        "edf_files_downloaded": n_edf,
        "records_file_lines": n_records,
        "used_for_ehr_representation": False,
        "role": "index time / future label source only",
    }

    # Redacted-like diagnosis codes
    out["dx_redacted_like"] = con.execute(f"""
        SELECT DX_CODE, COUNT(*) n FROM {dx}
        WHERE DX_CODE IS NULL OR TRIM(DX_CODE)=''
           OR UPPER(TRIM(DX_CODE)) IN ('REDACTED','NI','UN','UNKNOWN','NULL','NA','N/A','*','-','.')
           OR UPPER(DX_CODE) LIKE '%REDACT%'
        GROUP BY 1 ORDER BY n DESC LIMIT 20
    """).df().to_dict("records")

    _write_json(paths.COMPAT_DIR / "nch_inventory.json", out)
    return out


# --------------------------------------------------------------------------- #
# Patients + sleep studies                                                    #
# --------------------------------------------------------------------------- #
def build_patients_and_studies(con) -> None:
    from model_new.data import encode_race

    demo = _csv(paths.NCH_CSVS["demographic"])
    ss = _csv(paths.NCH_CSVS["sleep_study"])
    se = _csv(paths.NCH_CSVS["sleep_enc_id"])
    con.execute(f"""
        CREATE OR REPLACE TABLE nch_patients AS
        SELECT
          TRY_CAST(STUDY_PAT_ID AS BIGINT) AS patient_id,
          {PARSE_TS.format(col='BIRTH_DATE')} AS dob,
          TRIM(CAST(PCORI_GENDER_CD AS VARCHAR)) AS gender_cd,
          TRIM(CAST(GENDER_DESCR AS VARCHAR)) AS gender_descr,
          TRIM(CAST(PCORI_RACE_CD AS VARCHAR)) AS pcori_race_cd,
          TRIM(CAST(RACE_DESCR AS VARCHAR)) AS race_descr,
          TRIM(CAST(PCORI_HISPANIC_CD AS VARCHAR)) AS hispanic_cd,
          TRIM(CAST(ETHNICITY_DESCR AS VARCHAR)) AS ethnicity_descr,
          TRY_CAST(PEDS_GEST_AGE_NUM_WEEKS AS DOUBLE) AS gest_weeks,
          TRY_CAST(PEDS_GEST_AGE_NUM_DAYS AS DOUBLE) AS gest_days
        FROM {demo}
        WHERE TRY_CAST(STUDY_PAT_ID AS BIGINT) IS NOT NULL
    """)
    # Python-side sex/race encoding (must match MIMIC helpers).
    df = con.execute("SELECT * FROM nch_patients").df()
    sex = [mapping.nch_sex_to_mimic(g if g not in (None, "") else d)
           for g, d in zip(df["gender_cd"], df["gender_descr"])]
    race_str = [
        mapping.nch_race_to_mimic_string(rd, h, eth, pr)
        for rd, h, eth, pr in zip(
            df["race_descr"], df["hispanic_cd"], df["ethnicity_descr"], df["pcori_race_cd"]
        )
    ]
    race = [encode_race(s) for s in race_str]
    df["sex"] = np.asarray(sex, dtype=np.int8)
    df["race"] = np.asarray(race, dtype=np.int16)
    df["race_string"] = race_str
    con.register("patients_encoded", df)
    con.execute("CREATE OR REPLACE TABLE nch_patients AS SELECT * FROM patients_encoded")
    con.unregister("patients_encoded")

    con.execute(f"""
        CREATE OR REPLACE TABLE nch_sleep_studies AS
        SELECT
          TRY_CAST(s.STUDY_PAT_ID AS BIGINT) AS patient_id,
          TRY_CAST(s.SLEEP_STUDY_ID AS BIGINT) AS sleep_study_id,
          {PARSE_TS.format(col='s.SLEEP_STUDY_START_DATETIME')} AS index_time,
          TRIM(CAST(s.SLEEP_STUDY_DURATION_DATETIME AS VARCHAR)) AS duration_raw,
          TRY_CAST(s.AGE_AT_SLEEP_STUDY_DAYS AS DOUBLE) AS age_at_sleep_study_days,
          TRY_CAST(e.STUDY_ENC_ID AS BIGINT) AS study_enc_id
        FROM {ss} s
        LEFT JOIN {se} e
          ON TRY_CAST(s.STUDY_PAT_ID AS BIGINT) = TRY_CAST(e.STUDY_PAT_ID AS BIGINT)
         AND TRY_CAST(s.SLEEP_STUDY_ID AS BIGINT) = TRY_CAST(e.SLEEP_STUDY_ID AS BIGINT)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE nch_sleep_studies AS
        SELECT
          ss.*,
          p.dob,
          p.sex,
          p.race,
          p.race_string,
          CASE WHEN ss.index_time IS NOT NULL AND p.dob IS NOT NULL
               THEN (EPOCH(ss.index_time) - EPOCH(p.dob)) / 86400.0
               ELSE ss.age_at_sleep_study_days END AS index_age_days,
          ROW_NUMBER() OVER (
            PARTITION BY ss.patient_id
            ORDER BY ss.index_time NULLS LAST, ss.sleep_study_id
          ) AS study_ord
        FROM nch_sleep_studies ss
        LEFT JOIN nch_patients p USING (patient_id)
    """)
    con.execute(f"""
        COPY nch_patients TO '{(paths.PROCESSED_DIR / 'patients.parquet').as_posix()}'
          (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
        COPY nch_sleep_studies TO '{(paths.PROCESSED_DIR / 'sleep_studies.parquet').as_posix()}'
          (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


# --------------------------------------------------------------------------- #
# Raw events                                                                  #
# --------------------------------------------------------------------------- #
def build_raw_events(con) -> None:
    dx = _csv(paths.NCH_CSVS["diagnosis"])
    pr = _csv(paths.NCH_CSVS["procedure"])
    hx = _csv(paths.NCH_CSVS["procedure_surg_hx"])
    med = _csv(paths.NCH_CSVS["medication"])
    enc = _csv(paths.NCH_CSVS["encounter"])
    meas = _csv(paths.NCH_CSVS["measurement"])

    con.execute(f"""
        CREATE OR REPLACE TABLE raw_events AS
        SELECT
          TRY_CAST(STUDY_PAT_ID AS BIGINT) AS patient_id,
          TRY_CAST(STUDY_ENC_ID AS BIGINT) AS encounter_id,
          {PARSE_TS.format(col='DX_START_DATETIME')} AS event_time,
          'diagnosis' AS event_type,
          TRIM(CAST(DX_CODE_TYPE AS VARCHAR)) AS code_system,
          TRIM(CAST(DX_CODE AS VARCHAR)) AS raw_code,
          TRIM(CAST(DX_NAME AS VARCHAR)) AS description,
          CAST(NULL AS DOUBLE) AS value,
          'DIAGNOSIS' AS source_table,
          TRIM(CAST(DX_ALT_CODE AS VARCHAR)) AS alt_code
        FROM {dx}
        UNION ALL
        SELECT
          TRY_CAST(STUDY_PAT_ID AS BIGINT),
          TRY_CAST(STUDY_ENC_ID AS BIGINT),
          {PARSE_TS.format(col='PROCEDURE_DATETIME')},
          'procedure',
          TRIM(CAST(PROC_CODE_TYPE AS VARCHAR)),
          TRIM(CAST(PROC_CODE AS VARCHAR)),
          TRIM(CAST(PROC_DESCR AS VARCHAR)),
          CAST(NULL AS DOUBLE),
          'PROCEDURE',
          TRIM(CAST(PROC_ID_NCH AS VARCHAR))
        FROM {pr}
        UNION ALL
        SELECT
          TRY_CAST(STUDY_PAT_ID AS BIGINT),
          NULL,
          COALESCE({PARSE_TS.format(col='PROC_START_TIME')}, {PARSE_TS.format(col='PROC_NOTED_DATE')}),
          'procedure',
          CASE
            WHEN CPT_CODE IS NOT NULL AND UPPER(TRIM(CAST(CPT_CODE AS VARCHAR))) LIKE 'SHX%' THEN 'NCH_LOCAL'
            ELSE 'CPT'
          END,
          COALESCE(NULLIF(TRIM(CAST(PROC_CODE AS VARCHAR)), ''), TRIM(CAST(CPT_CODE AS VARCHAR))),
          TRIM(CAST(PROC_DESCR AS VARCHAR)),
          CAST(NULL AS DOUBLE),
          'PROCEDURE_SURG_HX',
          TRIM(CAST(CPT_CODE AS VARCHAR))
        FROM {hx}
        UNION ALL
        SELECT
          TRY_CAST(STUDY_PAT_ID AS BIGINT),
          TRY_CAST(STUDY_ENC_ID AS BIGINT),
          COALESCE(
            {PARSE_TS.format(col='MED_TAKEN_DATETIME')},
            {PARSE_TS.format(col='MED_START_DATETIME')},
            {PARSE_TS.format(col='MED_ORDER_DATETIME')}
          ),
          'medication',
          'RXNORM',
          TRIM(CAST(RXNORM_CODE AS VARCHAR)),
          COALESCE(TRIM(CAST(GENERIC_DRUG_DESCR AS VARCHAR)), TRIM(CAST(MEDICATION_DESCR AS VARCHAR))),
          TRY_CAST(EFFECTIVE_DRUG_DOSE AS DOUBLE),
          'MEDICATION',
          TRIM(CAST(GENERIC_DRUG_DESCR AS VARCHAR))
        FROM {med}
        UNION ALL
        SELECT
          TRY_CAST(STUDY_PAT_ID AS BIGINT),
          TRY_CAST(STUDY_ENC_ID AS BIGINT),
          COALESCE({PARSE_TS.format(col='VISIT_START_DATETIME')}, {PARSE_TS.format(col='ENCOUNTER_DATE')}),
          'drg',
          'DRG',
          TRIM(CAST(DRG_CODE AS VARCHAR)),
          TRIM(CAST(DRG_NAME AS VARCHAR)),
          CAST(NULL AS DOUBLE),
          'ENCOUNTER',
          NULL
        FROM {enc}
        WHERE DRG_CODE IS NOT NULL AND TRIM(CAST(DRG_CODE AS VARCHAR)) != ''
        UNION ALL
        SELECT
          TRY_CAST(STUDY_PAT_ID AS BIGINT),
          TRY_CAST(STUDY_ENC_ID AS BIGINT),
          {PARSE_TS.format(col='MEAS_RECORDED_DATETIME')},
          'measurement',
          'NCH_MEAS',
          TRIM(CAST(MEAS_TYPE AS VARCHAR)),
          TRIM(CAST(MEAS_VALUE_TEXT AS VARCHAR)),
          TRY_CAST(MEAS_VALUE_NUMBER AS DOUBLE),
          'MEASUREMENT',
          TRIM(CAST(MEAS_SOURCE AS VARCHAR))
        FROM {meas}
    """)
    con.execute("""
        CREATE OR REPLACE TABLE raw_events AS
        SELECT r.*,
               p.dob, p.sex, p.race, p.race_string,
               CASE WHEN r.event_time IS NOT NULL AND p.dob IS NOT NULL
                    THEN (EPOCH(r.event_time) - EPOCH(p.dob)) / 86400.0
                    ELSE NULL END AS age_at_event_days
        FROM raw_events r
        LEFT JOIN nch_patients p USING (patient_id)
    """)


def build_code_mapping(con, res: dict) -> None:
    uniq = con.execute("""
        SELECT event_type, code_system, raw_code,
               COUNT(*) AS n_events,
               COUNT(DISTINCT patient_id) AS n_patients,
               ANY_VALUE(description) AS sample_description,
               ANY_VALUE(alt_code) AS sample_alt
        FROM raw_events
        WHERE event_type IN ('diagnosis','procedure','medication','drg')
        GROUP BY 1,2,3
    """).df()
    rows = []
    for rec in uniq.itertuples(index=False):
        extra = {
            "generic": rec.sample_alt if rec.event_type == "medication" else None,
            "description": rec.sample_description,
            "nch_local_id": rec.sample_alt if rec.event_type == "procedure" else None,
        }
        m = mapping.map_row(rec.event_type, rec.raw_code, rec.code_system, res, **extra)
        d = mapping.mapped_to_dict(m)
        d["source_code_system"] = rec.code_system
        d["n_events"] = int(rec.n_events)
        d["n_patients"] = int(rec.n_patients)
        d["sample_description"] = rec.sample_description
        rows.append(d)
    import pandas as pd
    map_df = pd.DataFrame(rows)
    con.register("code_mapping_df", map_df)
    con.execute("CREATE OR REPLACE TABLE code_mapping AS SELECT * FROM code_mapping_df")
    con.unregister("code_mapping_df")
    out_csv = paths.COMPAT_DIR / "code_mapping.csv"
    con.execute(f"COPY code_mapping TO '{out_csv.as_posix()}' (HEADER, DELIMITER ',')")


def build_canonical_events(con) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE canonical_events AS
        SELECT
          r.patient_id,
          r.encounter_id,
          r.event_time,
          r.age_at_event_days,
          r.event_type,
          COALESCE(m.code_system, r.code_system) AS code_system,
          r.raw_code,
          m.normalized_code,
          m.mimic_token,
          m.mimic_token_id,
          COALESCE(m.description, r.description) AS description,
          r.value,
          r.source_table,
          m.mapping_status,
          m.oov_cause,
          m.needed_normalization,
          r.sex,
          r.race,
          r.dob
        FROM raw_events r
        LEFT JOIN code_mapping m
          ON r.event_type = m.event_type
         AND (r.code_system IS NOT DISTINCT FROM m.source_code_system)
         AND (r.raw_code IS NOT DISTINCT FROM m.raw_code)
    """)
    out = paths.PROCESSED_DIR / "canonical_events.parquet"
    con.execute(f"""
        COPY (
          SELECT * FROM canonical_events
          ORDER BY patient_id, event_time, event_type, COALESCE(mimic_token, raw_code)
        ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


# --------------------------------------------------------------------------- #
# Sequences                                                                   #
# --------------------------------------------------------------------------- #
def _write_sequence_npz(path: Path, samples: list[dict], unk: int) -> None:
    if not samples:
        return
    seq_len = np.asarray([int(s["code_indices"].shape[0]) for s in samples], dtype=np.int64)
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    np.cumsum(seq_len, out=offsets[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        subject_id=np.asarray([s["patient_id"] for s in samples], dtype=np.int64),
        hadm_id=np.asarray([s["encounter_id"] for s in samples], dtype=np.int64),
        sleep_study_id=np.asarray([s["sleep_study_id"] for s in samples], dtype=np.int64),
        label=np.full(len(samples), np.nan, dtype=np.float32),
        sex=np.asarray([s["sex"] for s in samples], dtype=np.int8),
        race=np.asarray([s["race"] for s in samples], dtype=np.int16),
        n_events_in_window=np.asarray([s["n_history_events"] for s in samples], dtype=np.int64),
        unk_vocab_index=np.asarray([unk], dtype=np.int64),
        offsets=offsets,
        code_indices=np.concatenate([s["code_indices"] for s in samples]).astype(np.int64),
        timestamps_days=np.concatenate([s["timestamps_days"] for s in samples]).astype(np.float32),
        age_days=np.concatenate([s["age_days"] for s in samples]).astype(np.float32),
        index_age_days=np.asarray([s["index_age_days"] for s in samples], dtype=np.float32),
        history_duration_days=np.asarray([s["history_duration"] for s in samples], dtype=np.float32),
        seq_len_before=np.asarray([s["sequence_length_before_truncation"] for s in samples], dtype=np.int64),
        seq_len_after=np.asarray([s["sequence_length_after_truncation"] for s in samples], dtype=np.int64),
    )


def build_sequences(con, vocab: dict[str, int], constants: dict,
                    modalities: tuple[str, ...], cohort: str) -> dict:
    """Build SleepBank-indexed sequences. cohort in {first, all}."""
    import pandas as pd

    unk = len(vocab)
    types = list(modalities)
    type_sql = ",".join("'" + t + "'" for t in types)
    cohort_pred = "AND s.study_ord = 1" if cohort == "first" else ""

    # Retain clinical tokens present in the frozen vocab. OOV rows stay in the
    # canonical table for audit but do not enter the encoder input.
    df = con.execute(f"""
        WITH hist AS (
          SELECT
            s.patient_id,
            s.sleep_study_id,
            s.index_time,
            s.index_age_days,
            s.study_enc_id,
            s.sex,
            s.race,
            s.study_ord,
            e.event_time,
            e.age_at_event_days,
            e.mimic_token,
            e.mimic_token_id,
            e.event_type
          FROM nch_sleep_studies s
          JOIN canonical_events e
            ON e.patient_id = s.patient_id
          WHERE s.index_time IS NOT NULL
            {cohort_pred}
            AND e.event_type IN ({type_sql})
            AND e.mimic_token_id IS NOT NULL
            AND e.event_time IS NOT NULL
            AND e.event_time < s.index_time
        )
        SELECT * FROM hist
        ORDER BY patient_id, sleep_study_id, event_time, mimic_token
    """).df()

    samples = []
    meta_rows = []
    if df.empty:
        return {"n_samples": 0, "modalities": types, "cohort": cohort}

    for (pid, sid), g in df.groupby(["patient_id", "sleep_study_id"], sort=False):
        g = g.sort_values(["event_time", "mimic_token"], kind="mergesort")
        times = pd.to_datetime(g["event_time"], utc=False)
        t0 = times.iloc[0]
        ts_days = ((times - t0).dt.total_seconds().to_numpy(dtype=np.float64)) / 86400.0
        ages = g["age_at_event_days"].to_numpy(dtype=np.float64)
        codes = g["mimic_token_id"].to_numpy(dtype=np.int64)
        n_before = int(codes.shape[0])
        if n_before > MAX_SEQ_LEN:
            codes = codes[-MAX_SEQ_LEN:]
            ts_days = ts_days[-MAX_SEQ_LEN:]
            ages = ages[-MAX_SEQ_LEN:]
        n_after = int(codes.shape[0])
        index_time = pd.Timestamp(g["index_time"].iloc[0])
        index_age = float(g["index_age_days"].iloc[0]) if g["index_age_days"].iloc[0] == g["index_age_days"].iloc[0] else float("nan")
        hist_start = pd.Timestamp(times.iloc[0])
        duration = (index_time - hist_start).total_seconds() / 86400.0
        enc = g["study_enc_id"].iloc[0]
        try:
            enc_id = int(enc) if enc is not None and not pd.isna(enc) else -1
        except (TypeError, ValueError):
            enc_id = -1
        max_event_time = pd.Timestamp(times.iloc[-1])
        sample = {
            "patient_id": int(pid),
            "sleep_study_id": int(sid),
            "encounter_id": enc_id,
            "index_time": index_time,
            "index_age_days": index_age,
            "n_history_events": n_before,
            "history_start": hist_start,
            "history_duration": float(duration),
            "sequence_length_before_truncation": n_before,
            "sequence_length_after_truncation": n_after,
            "sex": int(g["sex"].iloc[0]) if g["sex"].iloc[0] == g["sex"].iloc[0] else 0,
            "race": int(g["race"].iloc[0]) if g["race"].iloc[0] == g["race"].iloc[0] else 6,
            "code_indices": codes,
            "timestamps_days": ts_days.astype(np.float32),
            "age_days": np.nan_to_num(ages, nan=0.0).astype(np.float32),
            "max_event_time": max_event_time,
        }
        samples.append(sample)
        meta_rows.append({
            "patient_id": int(pid),
            "sleep_study_id": int(sid),
            "index_time": str(index_time),
            "index_age_days": index_age,
            "index_age_years": index_age / DAYS_PER_YEAR if index_age == index_age else float("nan"),
            "n_history_events": n_before,
            "history_start": str(hist_start),
            "history_duration": float(duration),
            "sequence_length_before_truncation": n_before,
            "sequence_length_after_truncation": n_after,
            "sex": sample["sex"],
            "race": sample["race"],
            "study_ord": int(g["study_ord"].iloc[0]),
        })

    tag = f"{cohort}_study_" + "_".join(types)
    npz_path = paths.PROCESSED_DIR / f"{tag}_sequences.npz"
    meta_path = paths.PROCESSED_DIR / f"{tag}_sequences.parquet"
    _write_sequence_npz(npz_path, samples, unk)
    pd.DataFrame(meta_rows).to_parquet(meta_path, index=False)

    # Also write default names requested in the brief for the dx+procedure first/all sets.
    if types == ["diagnosis", "procedure"] and cohort == "first":
        _write_sequence_npz(paths.PROCESSED_DIR / "first_study_sequences.npz", samples, unk)
        pd.DataFrame(meta_rows).to_parquet(paths.PROCESSED_DIR / "first_study_sequences.parquet", index=False)
    if types == ["diagnosis", "procedure"] and cohort == "all":
        _write_sequence_npz(paths.PROCESSED_DIR / "all_study_sequences.npz", samples, unk)
        pd.DataFrame(meta_rows).to_parquet(paths.PROCESSED_DIR / "all_study_sequences.parquet", index=False)

    n_pat = len({s["patient_id"] for s in samples})
    return {
        "cohort": cohort,
        "modalities": types,
        "n_samples": len(samples),
        "n_patients": n_pat,
        "npz": str(npz_path),
        "meta": str(meta_path),
        "unk_vocab_index": unk,
        "samples": samples,
        "meta_rows": meta_rows,
    }


# --------------------------------------------------------------------------- #
# Compatibility                                                               #
# --------------------------------------------------------------------------- #
def compatibility_tables(con) -> dict:
    fam = con.execute("""
        SELECT
          event_type,
          COUNT(*) AS nch_rows,
          COUNT(DISTINCT raw_code) AS n_raw_codes,
          COUNT(DISTINCT normalized_code) AS n_normalized_codes,
          COUNT(DISTINCT mimic_token) FILTER (WHERE mimic_token_id IS NOT NULL) AS n_matched_tokens,
          COUNT(*) FILTER (WHERE mimic_token_id IS NOT NULL) AS n_matched_events,
          COUNT(*) FILTER (WHERE mimic_token_id IS NULL) AS n_oov_events,
          COUNT(DISTINCT patient_id) AS n_patients
        FROM canonical_events
        WHERE event_type IN ('diagnosis','procedure','medication','drg','measurement')
        GROUP BY 1
        ORDER BY 1
    """).df()
    # Simpler patient-with-OOV query
    oov_pats = con.execute("""
        SELECT event_type,
               COUNT(DISTINCT patient_id) AS n_patients_with_oov
        FROM canonical_events
        WHERE mimic_token_id IS NULL AND event_type IN ('diagnosis','procedure','medication','drg')
        GROUP BY 1
    """).df()
    oov_map = dict(zip(oov_pats.event_type, oov_pats.n_patients_with_oov))

    rows = []
    for rec in fam.itertuples(index=False):
        n_rows = int(rec.nch_rows)
        n_raw = int(rec.n_raw_codes)
        n_norm = int(rec.n_normalized_codes or 0)
        n_match_tok = int(rec.n_matched_tokens or 0)
        n_match_ev = int(rec.n_matched_events or 0)
        n_oov_ev = int(rec.n_oov_events or 0)
        rows.append({
            "event_type": rec.event_type,
            "nch_rows": n_rows,
            "nch_unique_raw_codes": n_raw,
            "nch_unique_normalized_codes": n_norm,
            "exact_mimic_vocab_matches": n_match_tok,
            "unique_code_coverage_pct": 100.0 * n_match_tok / n_norm if n_norm else 0.0,
            "event_weighted_coverage_pct": 100.0 * n_match_ev / n_rows if n_rows else 0.0,
            "oov_event_count": n_oov_ev,
            "oov_event_pct": 100.0 * n_oov_ev / n_rows if n_rows else 0.0,
            "n_patients": int(rec.n_patients),
            "n_patients_with_oov": int(oov_map.get(rec.event_type, 0)),
            "pct_patients_with_oov": 100.0 * oov_map.get(rec.event_type, 0) / rec.n_patients if rec.n_patients else 0.0,
        })
    import pandas as pd
    cov = pd.DataFrame(rows)
    cov.to_csv(paths.COMPAT_DIR / "vocab_coverage.csv", index=False)

    oov = con.execute("""
        SELECT event_type, code_system, raw_code, normalized_code, mimic_token,
               mapping_status, oov_cause, n_events, n_patients, sample_description
        FROM code_mapping
        WHERE mimic_token_id IS NULL
        ORDER BY n_events DESC
    """).df()
    oov.to_csv(paths.COMPAT_DIR / "oov_codes.csv", index=False)

    causes = con.execute("""
        SELECT event_type, oov_cause,
               SUM(n_events) AS n_events,
               COUNT(*) AS n_codes,
               SUM(n_patients) AS n_patient_code_pairs
        FROM code_mapping
        WHERE mimic_token_id IS NULL
        GROUP BY 1,2
        ORDER BY 1, n_events DESC
    """).df()

    top_matched = con.execute("""
        SELECT event_type, mimic_token, ANY_VALUE(description) AS description,
               SUM(n_events) AS n_events, COUNT(*) AS n_raw_variants
        FROM code_mapping
        WHERE mimic_token_id IS NOT NULL
        GROUP BY 1,2
        ORDER BY n_events DESC
        LIMIT 80
    """).df()

    top50_oov = oov.head(50).to_dict("records")
    summary = {
        "by_family": cov.to_dict("records"),
        "oov_causes": causes.to_dict("records"),
        "top_matched": top_matched.to_dict("records"),
        "top50_oov": top50_oov,
        "normalization_rescues": int(con.execute("""
            SELECT COUNT(*) FROM code_mapping
            WHERE needed_normalization AND mimic_token_id IS NOT NULL
        """).fetchone()[0]),
        "measurements_not_tokenized": True,
        "measurement_note": (
            "NCH measurements (BMI, percentiles, vitals) have no counterpart in the "
            "frozen MIMIC LAB_/CHART_ itemid vocabulary and are retained only as covariates."
        ),
    }
    _write_json(paths.COMPAT_DIR / "compatibility_summary.json", summary)
    return summary


def write_compatibility_markdown(summary: dict, inventory: dict) -> None:
    lines = [
        "# MIMIC ↔ NCH vocabulary compatibility",
        "",
        "NCH codes are normalized with the Stage-1 rule `strip/upper/remove-dots` and",
        "rolled through the **same** PheWAS ICD→PheCode maps used for MIMIC.",
        "The frozen `data/processed/code_vocab.json` (`|V|=30635`) is never modified.",
        "",
        "PSG waveforms are **not** part of the EHR vocabulary. Sleep studies are index times.",
        "",
        "## Coverage by event family",
        "",
        "| family | NCH rows | unique raw | unique norm | vocab matches | unique % | event % | OOV events | OOV % | patients with OOV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["by_family"]:
        lines.append(
            f"| {r['event_type']} | {r['nch_rows']:,} | {r['nch_unique_raw_codes']:,} | "
            f"{r['nch_unique_normalized_codes']:,} | {r['exact_mimic_vocab_matches']:,} | "
            f"{r['unique_code_coverage_pct']:.1f} | {r['event_weighted_coverage_pct']:.1f} | "
            f"{r['oov_event_count']:,} | {r['oov_event_pct']:.1f} | "
            f"{r['n_patients_with_oov']:,} ({r['pct_patients_with_oov']:.1f}%) |"
        )
    lines += [
        "",
        f"Codes that matched only after formatting normalization: **{summary['normalization_rescues']}** unique rows in `code_mapping.csv`.",
        "",
        "## OOV cause breakdown (unique-code table, event-weighted)",
        "",
        "| family | cause | events | unique codes |",
        "|---|---|---:|---:|",
    ]
    for r in summary["oov_causes"]:
        lines.append(
            f"| {r['event_type']} | {r['oov_cause'] or ''} | {int(r['n_events']):,} | {int(r['n_codes']):,} |"
        )
    lines += [
        "",
        "Cause legend:",
        "",
        "1. `formatting_mismatch` — would match after case/dot/zero normalization (these are mapped, not left OOV).",
        "2. `coding_system_mismatch` — NCH system is not one Stage-1 rolled (e.g. local procedure IDs).",
        "3. `code_version_mismatch` — ICD-9 vs ICD-10 leftover not in the frozen vocab.",
        "4. `nch_local` — NCH internal identifiers (`PROC_ID_NCH`, `SHX*` surgical history).",
        "5. `pediatric_absent` — maps to a well-formed Stage-1 *namespace* token (PHE_/RXN_) that adult MIMIC never kept.",
        "6. `redacted` — placeholder / missing clinical code.",
        "7. `unknown_unmappable` — no justified map onto the frozen vocabulary.",
        "",
        "## Most frequent matched concepts",
        "",
    ]
    for r in summary["top_matched"][:25]:
        lines.append(f"- `{r['mimic_token']}` ({r['event_type']}): {int(r['n_events']):,} events — {r.get('description') or ''}")
    lines += ["", "## Top 50 OOV concepts by event frequency", ""]
    for r in summary["top50_oov"]:
        lines.append(
            f"- `{r.get('raw_code')}` [{r.get('event_type')}/{r.get('oov_cause')}] "
            f"{int(r.get('n_events') or 0):,} events — {r.get('sample_description') or ''}"
        )
    lines += [
        "",
        "## Measurements",
        "",
        summary["measurement_note"],
        "",
        f"NCH measurement rows: {inventory.get('measurements', {}).get('n_rows')}.",
        "",
        "## Inventory snapshot",
        "",
        f"- Patients: {inventory.get('patients', {}).get('n_patients')}",
        f"- Sleep studies: {inventory.get('sleep_studies', {}).get('n_studies')}",
        f"- Patients with >1 study: {inventory.get('patients_with_multiple_studies')}",
        f"- Local EDF files downloaded: {inventory.get('psg_waveforms_local', {}).get('edf_files_downloaded')} "
        f"(RECORDS lists {inventory.get('psg_waveforms_local', {}).get('records_file_lines')}; waveforms unused)",
        "",
    ]
    (paths.COMPAT_DIR / "compatibility_report.md").write_text("\n".join(lines), encoding="utf-8")


def _span_tau(span_days: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(np.asarray(span_days, dtype=np.float64)) / WEEK_DAYS)


def _summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    return {
        "n": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "quantiles": {str(q): float(np.quantile(x, q)) for q in qs},
    }


def temporal_audit(con, seq_pack: dict, constants: dict) -> dict:
    """Compare MIMIC corpus_stats / val_events vs NCH indexed sequences."""
    tau_max = float(constants["tau_max"])
    age_mean = float(constants["age_mean"])
    age_sd = float(constants["age_sd"])

    mimic_corpus = {}
    if paths.CORPUS_STATS_PATH.exists():
        raw = json.loads(paths.CORPUS_STATS_PATH.read_text())
        mimic_corpus = raw.get("stats", raw)

    samples = seq_pack.get("samples") or []
    n_ev = np.asarray([s["n_history_events"] for s in samples], dtype=np.float64)
    seq_after = np.asarray([s["sequence_length_after_truncation"] for s in samples], dtype=np.float64)
    dur = np.asarray([s["history_duration"] for s in samples], dtype=np.float64)
    idx_age = np.asarray([s["index_age_days"] / DAYS_PER_YEAR for s in samples], dtype=np.float64)
    span_tau = _span_tau(dur)

    # Consecutive lags inside truncated windows (raw days and tau).
    raw_dt = []
    tau_consec = []
    tau_pairs_sample = []
    z_ages = []
    event_ages = []
    for s in samples:
        ts = np.asarray(s["timestamps_days"], dtype=np.float64)
        ag = np.asarray(s["age_days"], dtype=np.float64) / DAYS_PER_YEAR
        event_ages.append(ag)
        z_ages.append((ag - age_mean) / max(age_sd, 1e-6))
        if ts.size >= 2:
            d = np.diff(ts)
            raw_dt.append(d)
            tau_consec.append(np.log1p(np.abs(d) / WEEK_DAYS))
            # subsample pairwise without forming L^2 for long sequences
            L = ts.size
            if L <= 64:
                tau = np.log1p(np.abs(ts[:, None] - ts[None, :]) / WEEK_DAYS)
                iu = np.triu_indices(L, k=1)
                tau_pairs_sample.append(tau[iu])
            else:
                rng = np.random.default_rng(0)
                i = rng.integers(0, L, size=512)
                j = rng.integers(0, L, size=512)
                tau_pairs_sample.append(np.log1p(np.abs(ts[i] - ts[j]) / WEEK_DAYS))

    raw_dt = np.concatenate(raw_dt) if raw_dt else np.zeros(0)
    tau_consec = np.concatenate(tau_consec) if tau_consec else np.zeros(0)
    tau_pairs = np.concatenate(tau_pairs_sample) if tau_pairs_sample else np.zeros(0)
    event_ages = np.concatenate(event_ages) if event_ages else np.zeros(0)
    z_ages = np.concatenate(z_ages) if z_ages else np.zeros(0)

    hist_frac = {name: float((dur >= d).mean()) if dur.size else 0.0
                 for name, d in HISTORY_CUTS_DAYS.items()}
    nch_tau_over = float((span_tau > tau_max).mean()) if span_tau.size else 0.0
    nch_tau_at_boundary = float((span_tau >= tau_max * 0.99).mean()) if span_tau.size else 0.0
    # Fraction of pairwise tau occupying the lower 10% of MIMIC support
    low = float((tau_pairs < 0.1 * tau_max).mean()) if tau_pairs.size else float("nan")

    # MIMIC consecutive dt from val_events (avoid the train shards the live job is reading).
    mimic_dt = {}
    if paths.VAL_EVENTS.exists():
        mimic_dt = con.execute(f"""
            SELECT
              MIN(log_delta_t) min_log_dt,
              MAX(log_delta_t) max_log_dt,
              AVG(log_delta_t) mean_log_dt,
              QUANTILE_CONT(log_delta_t, 0.5) median_log_dt,
              QUANTILE_CONT(age_at_event_days/365.25, 0.5) median_age,
              COUNT(*) n
            FROM read_parquet('{paths.VAL_EVENTS.as_posix()}')
        """).df().to_dict("records")[0]
        # Approximate consecutive days from stored log_delta_t: dt = exp(x)-1
        # This is the build_event_table column, NOT model tau.

    out = {
        "mimic_corpus_stats_path": str(paths.CORPUS_STATS_PATH),
        "mimic": {
            "n_events": mimic_corpus.get("n_events"),
            "event_age": {
                "min": mimic_corpus.get("event_age_min"),
                "max": mimic_corpus.get("event_age_max"),
                "median": mimic_corpus.get("event_age_median"),
                "mean": mimic_corpus.get("event_age_mean"),
                "sd": mimic_corpus.get("event_age_sd"),
            },
            "seq_len_quantiles": mimic_corpus.get("seq_len_quantiles"),
            "span_days_max": mimic_corpus.get("span_days_max"),
            "span_tau_quantiles": mimic_corpus.get("span_tau_quantiles"),
            "tau_quantiles_pairwise_sample": mimic_corpus.get("tau_quantiles"),
            "tau_max": mimic_corpus.get("tau_max"),
            "val_events_log_delta_t": mimic_dt,
        },
        "nch_first_study_dx_proc": {
            "n_samples": int(len(samples)),
            "events_per_sample_before_truncation": _summarize(n_ev),
            "sequence_length_after_truncation": _summarize(seq_after),
            "history_duration_days": _summarize(dur),
            "index_age_years": _summarize(idx_age),
            "event_age_years": _summarize(event_ages),
            "z_age_using_mimic_moments": _summarize(z_ages),
            "consecutive_dt_days": _summarize(raw_dt),
            "consecutive_tau": _summarize(tau_consec),
            "pairwise_tau_sample": _summarize(tau_pairs),
            "span_tau": _summarize(span_tau),
            "history_fractions": hist_frac,
            "tau_max_mimic": tau_max,
            "fraction_span_tau_exceeding_mimic_tau_max": nch_tau_over,
            "fraction_span_tau_at_99pct_boundary": nch_tau_at_boundary,
            "fraction_pairwise_tau_in_lowest_10pct_of_tau_max": low,
            "mean_span_tau_over_tau_max": float(span_tau.mean() / tau_max) if span_tau.size else None,
            "note_stage1_does_not_clip_tau": True,
        },
    }
    _write_json(paths.COMPAT_DIR / "temporal_summary.json", out)
    return out


def make_figures(temporal: dict, seq_pack: dict, constants: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    samples = seq_pack.get("samples") or []
    written = []

    def save(fig, name):
        p = paths.FIGURE_DIR / name
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p))

    idx_age = np.asarray([s["index_age_days"] / DAYS_PER_YEAR for s in samples])
    dur = np.asarray([s["history_duration"] for s in samples])
    slen = np.asarray([s["sequence_length_after_truncation"] for s in samples])
    n_before = np.asarray([s["n_history_events"] for s in samples])
    span_tau = _span_tau(dur)
    tau_max = float(constants["tau_max"])
    age_mean = float(constants["age_mean"])
    age_sd = float(constants["age_sd"])

    raw_dt, tau_c = [], []
    event_ages = []
    for s in samples:
        ts = np.asarray(s["timestamps_days"], dtype=np.float64)
        event_ages.append(np.asarray(s["age_days"], dtype=np.float64) / DAYS_PER_YEAR)
        if ts.size >= 2:
            d = np.diff(ts)
            raw_dt.append(d)
            tau_c.append(np.log1p(np.abs(d) / WEEK_DAYS))
    raw_dt = np.concatenate(raw_dt) if raw_dt else np.zeros(0)
    tau_c = np.concatenate(tau_c) if tau_c else np.zeros(0)
    event_ages = np.concatenate(event_ages) if event_ages else np.zeros(0)

    mimic = temporal.get("mimic") or {}
    m_age = mimic.get("event_age") or {}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if event_ages.size:
        ax.hist(event_ages, bins=40, density=True, alpha=0.7, label="NCH event age")
    if idx_age.size:
        ax.hist(idx_age, bins=40, density=True, alpha=0.5, label="NCH index age")
    if m_age.get("mean") is not None:
        ax.axvline(m_age["mean"], color="k", ls="--", label=f"MIMIC event mean {m_age['mean']:.1f}y")
        ax.axvline(m_age.get("median", 0), color="k", ls=":", label=f"MIMIC event median {m_age.get('median'):.1f}y")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("density")
    ax.set_title("Age: NCH vs MIMIC train moments")
    ax.legend(fontsize=8)
    save(fig, "age_distribution.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if slen.size:
        ax.hist(slen, bins=40, alpha=0.7, label="NCH after truncation")
        ax.hist(np.clip(n_before, 0, 4000), bins=40, alpha=0.4, label="NCH before truncation (clip 4000)")
    q = mimic.get("seq_len_quantiles") or {}
    if q:
        ax.axvline(float(q.get("0.5", 0)), color="k", ls="--", label="MIMIC seq_len median")
        ax.axvline(float(q.get("0.95", 1024)), color="k", ls=":", label="MIMIC seq_len p95")
    ax.set_xlabel("sequence length")
    ax.set_title("Sequence length")
    ax.legend(fontsize=8)
    save(fig, "sequence_lengths.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if dur.size:
        ax.hist(np.clip(dur, 0, 4000), bins=40, alpha=0.8)
    ax.set_xlabel("history duration (days, clipped at 4000)")
    ax.set_title("NCH history length before PSG")
    save(fig, "history_duration.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    if raw_dt.size:
        axes[0].hist(np.clip(raw_dt, 0, 365), bins=50, alpha=0.85)
    axes[0].set_xlabel("consecutive Δt (days, clip 365)")
    axes[0].set_title("NCH raw lag")
    if tau_c.size:
        axes[1].hist(tau_c, bins=50, alpha=0.85, label="NCH consecutive τ")
    mq = mimic.get("tau_quantiles_pairwise_sample") or {}
    if mq.get("0.5") is not None:
        axes[1].axvline(float(mq["0.5"]), color="k", ls="--", label="MIMIC pairwise τ median")
        axes[1].axvline(float(mq.get("0.95", 0)), color="k", ls=":", label="MIMIC pairwise τ p95")
    axes[1].set_xlabel("τ = log1p(|Δt|/7)")
    axes[1].set_title("log lag")
    axes[1].legend(fontsize=7)
    save(fig, "temporal_lag.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if span_tau.size:
        ax.hist(span_tau / tau_max, bins=40, alpha=0.85, label="NCH span τ / MIMIC τ_max")
    ax.axvline(1.0, color="r", ls="--", label="MIMIC τ_max")
    ax.set_xlabel("τ / τ_max (diagnostic; Stage-1 does not rescale)")
    ax.set_title("Normalized temporal support")
    ax.legend(fontsize=8)
    save(fig, "normalized_temporal_lag.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    z = (event_ages - age_mean) / max(age_sd, 1e-6)
    if z.size:
        ax.hist(z, bins=40, alpha=0.85)
    ax.axvline(0, color="k", ls="--", label="MIMIC z=0 (adult mean)")
    ax.set_xlabel("z(a) using frozen MIMIC μ/σ")
    ax.set_title("NCH ages in the Stage-1 standardized age domain")
    ax.legend(fontsize=8)
    save(fig, "age_z_mimic_domain.png")

    return written
