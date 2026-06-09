#!/usr/bin/env python3
"""Step 1: build PIC fine-tune cohorts for 4 tasks x 3 splits.

Tasks: Pneumonia, Heart Malformations (diagnosis tasks), ICU LOS>7, In-hospital Mortality.

Cohort parquet contract (read by finetune/dataset.py DiseaseClassificationDataset):
    columns = subject_id (BIGINT), label (INT 0/1), last_event_idx (INT)

CRITICAL ordering rule: DiseaseClassificationDataset orders a subject's events by
(timestamp_days, event_time, code_id) and truncates to the first last_event_idx+1 events.
This builder replicates that EXACT ordering over the EXACT events parquet the dataset will
read (FILTERED parquet for diagnosis tasks, FULL split parquet for mortality/LOS), so
last_event_idx is always consistent.

Target code sets for the diagnosis tasks are DERIVED from the PheWAS ICD-10->PheCode map
(reused via preprocessing's create_phecode_maps) plus any unmapped ICD10_ codes in the PIC
vocab whose raw category falls in the requested ranges. Nothing is hardcoded; the resolved
sets are printed for human verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Reuse the PheWAS PheCode map loader from the preprocessing package (read-only).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "preprocessing"))
from rollup_and_describe import create_phecode_maps  # noqa: E402

LOGGER = logging.getLogger("build_cohorts_pic")

DEFAULT_PIC_ROOT = Path("/home/suraj/Data/PIC/physionet.org/files/picdb/1.1.0/V1.1.0")

PNEUMONIA_CATS = [f"J{n}" for n in range(12, 19)]          # J12..J18
HEART_CATS = [f"Q{n}" for n in range(20, 29)]              # Q20..Q28
SPLITS = ("train", "val", "test")

# Each diagnosis task is restricted to a single PheCode phenotype family: the root
# PheCode and its children (root.*). Off-family PheCodes that some ICD-10 codes in the
# category range happen to map to are dropped (and logged with their description).
TASK_FAMILY = {
    "pneumonia": {"cats": PNEUMONIA_CATS, "phe_root": "480"},            # 480 = Pneumonia
    "heart_malformations": {"cats": HEART_CATS, "phe_root": "747"},      # 747 = cardiac congenital anomalies
}


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _csv(path: Path) -> str:
    return f"read_csv_auto('{_esc(path)}', ALL_VARCHAR=TRUE, header=TRUE, quote='\"', escape='\"')"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PIC fine-tune cohorts.")
    parser.add_argument("--obs_window_days", type=float, default=1.0)
    parser.add_argument("--los_threshold_days", type=float, default=7.0)
    parser.add_argument("--pic_root", type=Path, default=DEFAULT_PIC_ROOT)
    parser.add_argument(
        "--pic_dir", type=Path,
        default=_REPO_ROOT / "data" / "processed" / "pic",
    )
    return parser.parse_args()


def derive_target_sets(
    con: duckdb.DuckDBPyConnection,
    mappings_dir: Path,
    vocab_codes: set[str],
    descriptions: dict[str, str] | None = None,
) -> dict[str, set[str]]:
    """Derive {task: set(code_id)} from the PheWAS map + unmapped ICD10_ vocab codes.

    Each diagnosis task is restricted to a single PheCode phenotype family (root + root.*);
    off-family PheCodes that some category-range ICD-10 codes map to are dropped and logged.
    The unmapped-ICD10 fallback (raw codes in the category range) is kept as-is.
    """
    descriptions = descriptions or {}
    create_phecode_maps(con, mappings_dir)  # builds phe_icd9 / phe_icd10 / phe_defs

    def desc(code: str) -> str:
        if code in descriptions:
            return descriptions[code]
        return "<not in PIC vocab>" if code not in vocab_codes else "<no description>"

    def in_family(code: str, root: str) -> bool:
        sub = code[4:]  # strip 'PHE_'
        return sub == root or sub.startswith(root + ".")

    targets: dict[str, set[str]] = {}
    for task, cfg in TASK_FAMILY.items():
        cats = cfg["cats"]
        root = cfg["phe_root"]
        cat_list = ",".join(f"'{c}'" for c in cats)
        phecodes = [
            str(r[0])
            for r in con.execute(
                f"SELECT DISTINCT phecode FROM phe_icd10 WHERE LEFT(icd_code_norm, 3) IN ({cat_list})"
            ).fetchall()
        ]
        phe_all = {f"PHE_{p}" for p in phecodes}
        phe_keep = {c for c in phe_all if in_family(c, root)}
        phe_drop = phe_all - phe_keep
        # Unmapped ICD10_ codes present in the PIC vocab whose 3-char category is in range.
        icd10_set = {c for c in vocab_codes if c.startswith("ICD10_") and c[6:9] in set(cats)}

        target = (phe_keep | icd10_set) & vocab_codes  # only codes that actually occur in PIC
        targets[task] = target

        print(f"\n=== target codes: {task} (ICD-10 {cats[0]}..{cats[-1]}, family PHE_{root}.*) ===")
        print(f"  PheCodes derived from map: {len(phe_all)} | kept in-family: {len(phe_keep)} | dropped off-family: {len(phe_drop)}")
        print(f"  RESOLVED target set (present in PIC, n={len(target)}):")
        for c in sorted(target):
            print(f"      {c}  ::  {desc(c)}")
        if icd10_set:
            print(f"  (includes unmapped ICD10_ fallback: {sorted(icd10_set)})")
        else:
            print("  (no unmapped ICD10_ fallback codes present)")
        if phe_drop:
            print(f"  DROPPED off-family PheCodes ({len(phe_drop)}):")
            for c in sorted(phe_drop):
                print(f"      {c}  ::  {desc(c)}")
    return targets


def _register_targets(con: duckdb.DuckDBPyConnection, codes: set[str]) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE target_codes(code_id VARCHAR)")
    if codes:
        con.executemany("INSERT INTO target_codes VALUES (?)", [(c,) for c in sorted(codes)])


def _build_split_subjects(con: duckdb.DuckDBPyConnection, split_events: Path) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE split_subjects AS
        SELECT DISTINCT CAST(subject_id AS BIGINT) AS subject_id
        FROM read_parquet('{_esc(split_events)}')
        """
    )


def _build_index_adm(con: duckdb.DuckDBPyConnection, admissions: Path) -> None:
    """First admission (earliest ADMITTIME) per subject = index admission."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE index_adm AS
        WITH adm AS (
            SELECT
                TRY_CAST(a.SUBJECT_ID AS BIGINT) AS subject_id,
                TRY_CAST(a.HADM_ID AS BIGINT) AS hadm_id,
                TRY_CAST(a.ADMITTIME AS TIMESTAMP) AS admittime,
                TRY_CAST(a.DISCHTIME AS TIMESTAMP) AS dischtime,
                TRY_CAST(a.DEATHTIME AS TIMESTAMP) AS deathtime,
                TRY_CAST(a.HOSPITAL_EXPIRE_FLAG AS INTEGER) AS hosp_expire
            FROM {_csv(admissions)} a
            JOIN split_subjects s ON TRY_CAST(a.SUBJECT_ID AS BIGINT) = s.subject_id
            WHERE TRY_CAST(a.ADMITTIME AS TIMESTAMP) IS NOT NULL
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS rn
            FROM adm
        )
        SELECT subject_id, hadm_id, admittime, dischtime, deathtime, hosp_expire
        FROM ranked WHERE rn = 1
        """
    )


def _build_window(con: duckdb.DuckDBPyConnection, events: Path, obs: float, name: str) -> None:
    """Per-subject window stats over `events`, ordered exactly like the dataset."""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {name} AS
        WITH ev AS (
            SELECT
                CAST(e.subject_id AS BIGINT) AS subject_id,
                CAST(e.timestamp_days AS DOUBLE) AS timestamp_days,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(e.subject_id AS BIGINT)
                    ORDER BY e.timestamp_days, e.event_time, e.code_id
                ) - 1 AS event_idx
            FROM read_parquet('{_esc(events)}') e
            JOIN split_subjects s ON CAST(e.subject_id AS BIGINT) = s.subject_id
        )
        SELECT
            subject_id,
            SUM(CASE WHEN timestamp_days <= {obs} THEN 1 ELSE 0 END) AS n_in_window,
            MAX(CASE WHEN timestamp_days <= {obs} THEN event_idx END) AS last_event_idx
        FROM ev
        GROUP BY subject_id
        """
    )


def _write_cohort(df: pd.DataFrame, out_path: Path, con: duckdb.DuckDBPyConnection) -> None:
    out = pd.DataFrame(
        {
            "subject_id": df["subject_id"].astype(np.int64),
            "label": df["label"].astype(np.int32),
            "last_event_idx": df["last_event_idx"].astype(np.int64),
        }
    ).sort_values("subject_id", kind="mergesort").reset_index(drop=True)
    con.register("cohort_df", out)
    try:
        con.execute(
            f"""
            COPY (SELECT subject_id, label, last_event_idx FROM cohort_df ORDER BY subject_id)
            TO '{_esc(out_path)}' (FORMAT PARQUET)
            """
        )
    finally:
        con.unregister("cohort_df")


def _report(task: str, split: str, df: pd.DataFrame, excl: dict[str, int]) -> None:
    n = len(df)
    prev = float(df["label"].mean()) if n else float("nan")
    excl_str = " ".join(f"{k}={v:,}" for k, v in excl.items())
    print(f"[{task}:{split}] n_subjects={n:,} prevalence={prev:.4%} | excluded: {excl_str}")


def build_mortality(con, split, split_events, out_path, obs):
    _build_window(con, split_events, obs, "win_full")
    df = con.execute(
        """
        SELECT a.subject_id, a.admittime, a.deathtime,
               CASE WHEN a.hosp_expire = 1 THEN 1 ELSE 0 END AS label,
               COALESCE(w.n_in_window, 0) AS n_in_window, w.last_event_idx
        FROM index_adm a LEFT JOIN win_full w USING (subject_id)
        """
    ).df()
    died_in_window = df["deathtime"].notna() & (
        (df["deathtime"] - df["admittime"]).dt.total_seconds() / 86400.0 <= obs
    )
    no_window = (df["n_in_window"] == 0) | df["last_event_idx"].isna()
    keep = ~died_in_window & ~no_window
    excl = {
        "died_within_window": int((died_in_window).sum()),
        "no_events_in_window": int((no_window & ~died_in_window).sum()),
    }
    kept = df.loc[keep].copy()
    _write_cohort(kept, out_path, con)
    _report("mortality", split, kept, excl)


def build_los(con, split, split_events, out_path, obs, icustays, los_thr):
    _build_window(con, split_events, obs, "win_full")
    df = con.execute(
        f"""
        WITH icu AS (
            SELECT TRY_CAST(HADM_ID AS BIGINT) AS hadm_id,
                   TRY_CAST(LOS AS DOUBLE) AS los_days,
                   ROW_NUMBER() OVER (PARTITION BY TRY_CAST(HADM_ID AS BIGINT)
                                      ORDER BY TRY_CAST(INTIME AS TIMESTAMP)) AS rn
            FROM {_csv(icustays)}
        )
        SELECT a.subject_id, i.los_days,
               COALESCE(w.n_in_window, 0) AS n_in_window, w.last_event_idx
        FROM index_adm a
        LEFT JOIN icu i ON a.hadm_id = i.hadm_id AND i.rn = 1
        LEFT JOIN win_full w USING (subject_id)
        """
    ).df()
    df["label"] = (df["los_days"] > float(los_thr)).astype(np.int32)
    no_icu = df["los_days"].isna()
    short_icu = (~no_icu) & (df["los_days"] <= obs)
    no_window = (df["n_in_window"] == 0) | df["last_event_idx"].isna()
    keep = ~no_icu & ~short_icu & ~no_window
    excl = {
        "no_icu_stay": int(no_icu.sum()),
        "icu_shorter_than_window": int((short_icu & ~no_icu).sum()),
        "no_events_in_window": int((no_window & ~no_icu & ~short_icu).sum()),
    }
    kept = df.loc[keep].copy()
    _write_cohort(kept, out_path, con)
    _report("los_gt7", split, kept, excl)


def build_diagnosis(con, task, split, split_events, filtered_path, out_path, obs, target_codes):
    _register_targets(con, target_codes)
    # Filtered events parquet = split events with all target-family codes removed.
    con.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet('{_esc(split_events)}')
            WHERE code_id NOT IN (SELECT code_id FROM target_codes)
        ) TO '{_esc(filtered_path)}' (FORMAT PARQUET)
        """
    )
    # Label: any target code in the subject's INDEX-ADMISSION events (unfiltered).
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pos_subjects AS
        SELECT DISTINCT e.subject_id
        FROM read_parquet('{_esc(split_events)}') e
        JOIN index_adm a ON CAST(e.subject_id AS BIGINT) = a.subject_id
                        AND CAST(e.hadm_id AS BIGINT) = a.hadm_id
        WHERE e.code_id IN (SELECT code_id FROM target_codes)
        """
    )
    # Context window computed on the FILTERED stream.
    _build_window(con, filtered_path, obs, "win_filt")
    df = con.execute(
        """
        SELECT a.subject_id,
               CASE WHEN p.subject_id IS NOT NULL THEN 1 ELSE 0 END AS label,
               COALESCE(w.n_in_window, 0) AS n_in_window, w.last_event_idx
        FROM index_adm a
        LEFT JOIN win_filt w USING (subject_id)
        LEFT JOIN pos_subjects p USING (subject_id)
        """
    ).df()
    no_window = (df["n_in_window"] == 0) | df["last_event_idx"].isna()
    keep = ~no_window
    excl = {"no_events_in_window_after_filter": int(no_window.sum())}
    kept = df.loc[keep].copy()
    _write_cohort(kept, out_path, con)
    _report(task, split, kept, excl)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    pic_dir: Path = args.pic_dir
    out_dir = pic_dir / "cohorts"
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = _REPO_ROOT / "data" / "processed" / "mappings"

    admissions = args.pic_root / "ADMISSIONS.csv"
    icustays = args.pic_root / "ICUSTAYS.csv"
    for p in (admissions, icustays):
        if not p.exists():
            raise FileNotFoundError(f"Missing PIC label table: {p}")

    split_events = {s: pic_dir / f"{s}_events.parquet" for s in SPLITS}
    for s, p in split_events.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing split events parquet: {p}")

    vocab_codes = set(json.load((pic_dir / "code_vocab_pic.json").open()).keys())
    descriptions = json.load((pic_dir / "code_descriptions_pic.json").open())

    obs = float(args.obs_window_days)
    print(f"[config] OBS_WINDOW_DAYS={obs} LOS_THRESHOLD_DAYS={args.los_threshold_days}")
    print(f"[admissions] {admissions}")
    print(f"[icustays]   {icustays}")
    print(f"[out_dir]    {out_dir}")

    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit='24GB'")
        tmp = out_dir / "duckdb_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"PRAGMA temp_directory='{_esc(tmp)}'")
        con.execute("PRAGMA threads=8")

        targets = derive_target_sets(con, mappings_dir, vocab_codes, descriptions)

        for split in SPLITS:
            _build_split_subjects(con, split_events[split])
            _build_index_adm(con, admissions)

            build_mortality(con, split, split_events[split],
                             out_dir / f"mortality_{split}_cohort.parquet", obs)
            build_los(con, split, split_events[split],
                      out_dir / f"los_gt7_{split}_cohort.parquet", obs, icustays, args.los_threshold_days)
            for task in ("pneumonia", "heart_malformations"):
                build_diagnosis(
                    con, task, split, split_events[split],
                    out_dir / f"{task}_{split}_events.parquet",
                    out_dir / f"{task}_{split}_cohort.parquet",
                    obs, targets[task],
                )

        print("\nWrote cohorts + filtered events to", out_dir)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
