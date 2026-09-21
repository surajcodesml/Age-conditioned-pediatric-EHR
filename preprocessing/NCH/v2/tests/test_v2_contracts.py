"""Automated tests for NCH Stage-2 v2 preprocessing contracts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.NCH.v2 import paths as P
from preprocessing.NCH.v2.cleanup import SENTINEL_YEARS, clean_canonical_events
from preprocessing.NCH.v2.sequences import assert_no_index_encounter_leakage, build_indexed_sequences
from preprocessing.NCH.v2.splits import make_splits
from preprocessing.NCH.v2.procedures import cpt_chapter_token


def test_sentinel_date_filtering():
    sleep = pd.DataFrame({
        "patient_id": [1],
        "sleep_study_id": [10],
        "study_enc_id": [100],
        "index_time": [pd.Timestamp("2018-01-01")],
        "index_age_days": [2000.0],
        "sex": [1],
        "race": [0],
    })
    events = pd.DataFrame({
        "patient_id": [1, 1, 1, 1],
        "encounter_id": [99, 99, 99, 100],
        "event_time": [
            pd.Timestamp("1899-01-01"),
            pd.Timestamp("2015-01-01"),
            pd.Timestamp("2010-01-01"),
            pd.Timestamp("2018-01-01"),
        ],
        "age_at_event_days": [1000.0, -50.0, 500.0, 2000.0],
        "event_type": ["diagnosis"] * 4,
        "code_system": ["ICD10"] * 4,
        "raw_code": ["J459", "J459", "J459", "G4733"],
        "normalized_code": ["J459"] * 4,
        "mimic_token": ["PHE_495"] * 4,
        "mimic_token_id": [10, 10, 10, 10],
        "description": [""] * 4,
        "value": [np.nan] * 4,
        "source_table": ["DIAGNOSIS"] * 4,
        "mapping_status": ["mapped"] * 4,
        "oov_cause": [None] * 4,
        "needed_normalization": [False] * 4,
        "sex": [1] * 4,
        "race": [0] * 4,
        "dob": [pd.Timestamp("2012-01-01")] * 4,
    })
    # Don't write into real artifact dirs during unit test — monkey via tmp
    # Use functions' logic indirectly: call clean and check reasons
    import preprocessing.NCH.v2.paths as paths_mod
    old = paths_mod.DIRS
    tmp = Path("/tmp/nch_v2_test_cleanup")
    paths_mod.DIRS = {k: tmp / k for k in old}
    for p in paths_mod.DIRS.values():
        p.mkdir(parents=True, exist_ok=True)
    try:
        clean, report = clean_canonical_events(events, sleep)
        assert "sentinel_year" in report["removed_by_reason"]
        assert report["removed_by_reason"]["age_before_dob"] >= 1
        # PSG encounter removed
        assert report["psg_encounter_removed"]["n_events"] >= 1
        assert (clean["encounter_id"] != 100).all() or clean.empty or 100 not in set(clean["encounter_id"].dropna())
        assert 1899 in SENTINEL_YEARS
    finally:
        paths_mod.DIRS = old


def test_no_index_encounter_leakage_assertion():
    indexes = pd.DataFrame({
        "patient_id": [1], "sleep_study_id": [1], "study_enc_id": [55],
        "index_time": [pd.Timestamp("2020-01-01")], "index_age_days": [1000.0],
        "sex": [1], "race": [0],
    })
    clean = pd.DataFrame({
        "patient_id": [1, 1],
        "encounter_id": [54, 53],
        "event_time": [pd.Timestamp("2019-01-01")] * 2,
    })
    r = assert_no_index_encounter_leakage(indexes, clean)
    assert r["passed"]
    dirty = pd.DataFrame({
        "patient_id": [1], "encounter_id": [55],
        "event_time": [pd.Timestamp("2019-01-01")],
    })
    r2 = assert_no_index_encounter_leakage(indexes, dirty)
    assert not r2["passed"]


def test_no_adults_in_primary_cohort_artifact():
    path = P.DIRS["processed"] / "sleep_studies_pediatric_first.parquet"
    if not path.exists():
        pytest.skip("v2 artifacts not built yet")
    df = pd.read_parquet(path)
    assert (df["index_age_days"] / P.DAYS_PER_YEAR < 18).all()


def test_vocab_ids_unchanged_for_existing_mimic_tokens():
    vocab = json.loads(P.VOCAB_PATH.read_text())
    ext = P.DIRS["extended_vocab"] / "code_vocab_extended_sample.json"
    if not ext.exists():
        pytest.skip("extended vocab sample not built")
    newv = json.loads(ext.read_text())
    for tok, i in vocab.items():
        assert newv[tok] == i
    assert all(v >= len(vocab) for t, v in newv.items() if t not in vocab)


def test_unk_behavior_retains_oov():
    vocab = {"PHE_495": 0}
    events = pd.DataFrame({
        "patient_id": [1, 1],
        "encounter_id": [1, 1],
        "event_time": [pd.Timestamp("2017-01-01"), pd.Timestamp("2017-06-01")],
        "age_at_event_days": [1000.0, 1200.0],
        "event_type": ["diagnosis", "diagnosis"],
        "raw_code": ["J459", "ZZZZZ"],
        "mimic_token": ["PHE_495", None],
        "mimic_token_id": [0, np.nan],
    })
    indexes = pd.DataFrame({
        "patient_id": [1], "sleep_study_id": [1], "study_enc_id": [9],
        "index_time": [pd.Timestamp("2018-01-01")], "index_age_days": [1500.0],
        "sex": [1], "race": [0],
    })
    samples, meta, summary = build_indexed_sequences(
        events, indexes, vocab, modalities=("diagnosis",), representation="t",
        retain_oov_as_unk=True,
    )
    assert len(samples) == 1
    assert 1 in samples[0]["code_indices"] or samples[0]["n_unk"] >= 1  # unk vocab index = 1
    assert summary["unk_event_total"] >= 1


def test_mapping_determinism_cpt_chapter():
    assert cpt_chapter_token("71045") == cpt_chapter_token("71045")
    assert cpt_chapter_token("71045") == "CPT_CHAPTER_RAD"
    assert cpt_chapter_token("99213") == "CPT_CHAPTER_EAM"


def test_patient_split_isolation():
    path = P.DIRS["splits"] / "pediatric_first_splits.parquet"
    if not path.exists():
        pytest.skip("splits not built")
    df = pd.read_parquet(path)
    sets = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_label_leakage_prevention_psg_not_in_clean():
    leak = P.DIRS["validation"] / "leakage_test.json"
    if not leak.exists():
        pytest.skip("leakage test not run")
    data = json.loads(leak.read_text())
    assert data["passed"] is True
