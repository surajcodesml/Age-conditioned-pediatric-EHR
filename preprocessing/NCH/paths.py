"""Filesystem locations for NCH Stage-2 preprocessing.

Does not point at the active Stage-1 run directory except for read-only contract
inspection of already-written config files.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Stage-1 MIMIC artifacts (read-only; never overwrite) ------------------- #
MIMIC_PROCESSED = REPO_ROOT / "data" / "processed"
VOCAB_PATH = MIMIC_PROCESSED / "code_vocab.json"
DESCRIPTIONS_PATH = MIMIC_PROCESSED / "code_descriptions.json"
EMBEDDING_PATH = MIMIC_PROCESSED / "bge_embeddings.pt"
MAPPINGS_DIR = MIMIC_PROCESSED / "mappings"
TENSORIZED_DIR = MIMIC_PROCESSED / "tensorized_flat"
TRAIN_EVENTS = MIMIC_PROCESSED / "train_events.parquet"
VAL_EVENTS = MIMIC_PROCESSED / "val_events.parquet"
TEST_EVENTS = MIMIC_PROCESSED / "test_events.parquet"
CORPUS_STATS_PATH = TENSORIZED_DIR / "train" / "corpus_stats.json"
STAGE1_RUN_CONFIG = REPO_ROOT / "stage1_mimic_pretrain" / "run" / "adkm_s0" / "config.json"

# --- NCH Sleep DataBank (PhysioNet v3.1.0) --------------------------------- #
NCH_ROOT = Path("/home/suraj/Data/NCH-Sleep-DataBank/physionet.org/files/nch-sleep/3.1.0")
NCH_HEALTH = NCH_ROOT / "Health_Data"
NCH_SLEEP_DATA = NCH_ROOT / "Sleep_Data"
NCH_RECORDS = NCH_ROOT / "RECORDS"

NCH_CSVS = {
    "demographic": NCH_HEALTH / "DEMOGRAPHIC.csv",
    "encounter": NCH_HEALTH / "ENCOUNTER.csv",
    "diagnosis": NCH_HEALTH / "DIAGNOSIS.csv",
    "procedure": NCH_HEALTH / "PROCEDURE.csv",
    "procedure_surg_hx": NCH_HEALTH / "PROCEDURE_SURG_HX.csv",
    "medication": NCH_HEALTH / "MEDICATION.csv",
    "measurement": NCH_HEALTH / "MEASUREMENT.csv",
    "sleep_study": NCH_HEALTH / "SLEEP_STUDY.csv",
    "sleep_enc_id": NCH_HEALTH / "SLEEP_ENC_ID.csv",
}

# --- Outputs (never write under data/processed MIMIC or stage1 run dirs) ---- #
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "nch_stage2"
CONTRACT_DIR = ARTIFACT_ROOT / "pretraining_contract"
COMPAT_DIR = ARTIFACT_ROOT / "compatibility"
FIGURE_DIR = COMPAT_DIR / "figures"
PROCESSED_DIR = ARTIFACT_ROOT / "processed"
VALIDATION_DIR = ARTIFACT_ROOT / "validation"
WORK_DIR = PROCESSED_DIR / "_work"


def ensure_output_dirs() -> None:
    for p in (CONTRACT_DIR, COMPAT_DIR, FIGURE_DIR, PROCESSED_DIR, VALIDATION_DIR, WORK_DIR):
        p.mkdir(parents=True, exist_ok=True)
