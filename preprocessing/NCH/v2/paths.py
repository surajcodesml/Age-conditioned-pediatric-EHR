"""NCH Stage-2 v2 artifact paths (does not overwrite v1 audit)."""
from __future__ import annotations

from pathlib import Path

from preprocessing.NCH import paths as v1

REPO_ROOT = v1.REPO_ROOT
ARTIFACT_V2 = v1.ARTIFACT_ROOT / "v2"

DIRS = {
    "cleanup": ARTIFACT_V2 / "cleanup",
    "vocab_recovery": ARTIFACT_V2 / "vocab_recovery",
    "procedure_mapping": ARTIFACT_V2 / "procedure_mapping",
    "medication_mapping": ARTIFACT_V2 / "medication_mapping",
    "extended_vocab": ARTIFACT_V2 / "extended_vocab",
    "age_extrapolation": ARTIFACT_V2 / "age_extrapolation",
    "labels": ARTIFACT_V2 / "labels",
    "splits": ARTIFACT_V2 / "splits",
    "validation": ARTIFACT_V2 / "validation_v2",
    "processed": ARTIFACT_V2 / "processed",
    "figures": ARTIFACT_V2 / "figures",
    "reports": ARTIFACT_V2 / "reports",
    "work": ARTIFACT_V2 / "processed" / "_work",
}

VOCAB_PATH = v1.VOCAB_PATH
DESCRIPTIONS_PATH = v1.DESCRIPTIONS_PATH
EMBEDDING_PATH = v1.EMBEDDING_PATH
MAPPINGS_DIR = v1.MAPPINGS_DIR
TRAIN_EVENTS = v1.TRAIN_EVENTS
VAL_EVENTS = v1.VAL_EVENTS
TEST_EVENTS = v1.TEST_EVENTS
NCH_CSVS = v1.NCH_CSVS
NCH_SLEEP_DATA = v1.NCH_SLEEP_DATA
NCH_RECORDS = v1.NCH_RECORDS
NCH_ROOT = v1.NCH_ROOT
STAGE1_BEST = REPO_ROOT / "stage1_mimic_pretrain" / "run" / "adkm_s0" / "checkpoint_best.pt"
STAGE1_FINAL = REPO_ROOT / "stage1_mimic_pretrain" / "run" / "adkm_s0" / "checkpoint_final.pt"
STAGE1_CONFIG = REPO_ROOT / "stage1_mimic_pretrain" / "run" / "adkm_s0" / "config.json"
V1_WORK_DB = v1.WORK_DIR / "nch_stage2.duckdb"
V1_CANONICAL = v1.PROCESSED_DIR / "canonical_events.parquet"
V1_PATIENTS = v1.PROCESSED_DIR / "patients.parquet"
V1_SLEEP = v1.PROCESSED_DIR / "sleep_studies.parquet"

DAYS_PER_YEAR = 365.25
WEEK_DAYS = 7.0
MAX_SEQ_LEN = 1024
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)


def ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
