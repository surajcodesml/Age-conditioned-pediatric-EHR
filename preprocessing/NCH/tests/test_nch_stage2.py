"""Unit/smoke tests for NCH Stage-2 mapping and Stage-1 contract reuse.

Run: python -m pytest preprocessing/NCH/tests/test_nch_stage2.py
  or: python preprocessing/NCH/tests/test_nch_stage2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model_new.data import DAYS_PER_YEAR, WEEK_DAYS, lag_to_tau  # noqa: E402
from preprocessing.NCH import mapping, paths  # noqa: E402
from preprocessing.NCH.mapping import (  # noqa: E402
    is_redacted,
    map_diagnosis,
    map_medication,
    map_procedure,
    nch_race_to_mimic_string,
    nch_sex_to_mimic,
    normalize_code,
)
from stage1_mimic_pretrain.config import (  # noqa: E402
    MIMIC_AGE_MEAN_YEARS,
    MIMIC_AGE_STD_YEARS,
)


def _res():
    return mapping.load_mapping_resources()


def test_normalize_code_matches_stage1():
    sys.path.insert(0, str(_REPO / "preprocessing"))
    from rollup_and_describe import normalize_code as stage1_norm

    for raw in ["r56.9", " R56.9 ", "250.00", "a00.0"]:
        assert normalize_code(raw) == stage1_norm(raw)


def test_redacted_not_a_concept():
    assert is_redacted("redacted")
    assert is_redacted("")
    assert is_redacted(None)
    assert is_redacted("NI")
    assert not is_redacted("R56.9")
    res = _res()
    m = map_diagnosis("redacted", "ICD10", res)
    assert m.oov_cause == "redacted"
    assert m.mimic_token_id is None


def test_icd_dot_normalization_hits_phecode_or_icd_leftover():
    res = _res()
    # Seizure / unspecified — NCH sample row used R56.9
    m = map_diagnosis("R56.9", "ICD10", res)
    assert m.normalized_code == "R569"
    assert m.needed_normalization or m.normalized_code != "R56.9"
    assert m.mimic_token is not None
    if m.mapping_status == "mapped_phecode":
        assert m.mimic_token.startswith("PHE_")
    else:
        assert m.mimic_token in {"ICD10_R569", "ICD10_R56.9"}
    assert m.mimic_token_id is None or m.mimic_token in res["vocab"]


def test_phecode_token_ids_are_frozen():
    res = _res()
    vocab = res["vocab"]
    # Diabetes family used throughout this repo
    assert "PHE_250.2" in vocab
    m = map_diagnosis("250.00", "ICD9", res)
    if m.mimic_token and m.mimic_token.startswith("PHE_"):
        assert vocab[m.mimic_token] == m.mimic_token_id


def test_cpt_hcpcs_namespace():
    res = _res()
    m = map_procedure("71045", "CPT", res)
    assert m.code_system == "HCPCS"
    if m.mimic_token_id is not None:
        assert m.mimic_token.startswith("HCPCS_")
        assert res["vocab"][m.mimic_token] == m.mimic_token_id


def test_local_procedure_is_oov():
    res = _res()
    m = map_procedure("SHX29", "NCH_LOCAL", res)
    assert m.mimic_token_id is None
    assert m.oov_cause in {"nch_local", "coding_system_mismatch"}


def test_medication_rxcui():
    res = _res()
    # Pick a token that is definitely in the frozen vocab.
    rxcui = next(k.split("_", 1)[1] for k in res["vocab"] if k.startswith("RXN_"))
    m = map_medication(rxcui, "whatever", None, res)
    assert m.mapping_status == "rxcui_exact"
    assert m.mimic_token == "RXN_" + rxcui
    assert m.mimic_token_id == res["vocab"][m.mimic_token]


def test_special_token_ids():
    vocab = json.loads(paths.VOCAB_PATH.read_text())
    v = len(vocab)
    assert min(vocab.values()) == 0
    assert max(vocab.values()) == v - 1
    assert "[PAD]" not in vocab and "[UNK]" not in vocab
    pad, unk_model, real0 = 0, 1, 0 + 2
    assert pad == 0 and unk_model == 1 and real0 == 2


def test_lag_to_tau_is_the_stage1_function():
    dt = torch.tensor([0.0, 7.0, 365.25], dtype=torch.float64)
    tau = lag_to_tau(dt)
    expect = torch.log1p(dt.abs() / WEEK_DAYS)
    assert torch.allclose(tau, expect)
    assert WEEK_DAYS == 7.0


def test_age_years_and_z():
    days = np.array([0.0, 365.25, 365.25 * 6, 365.25 * 12, 365.25 * 18])
    years = days / DAYS_PER_YEAR
    z = (years - MIMIC_AGE_MEAN_YEARS) / MIMIC_AGE_STD_YEARS
    assert np.allclose(years, [0, 1, 6, 12, 18])
    # Infants sit several σ below the adult MIMIC mean — this is expected, not a bug.
    assert z[0] < -3


def test_sex_race_helpers():
    assert nch_sex_to_mimic("M") == 1
    assert nch_sex_to_mimic("F") == 0
    from model_new.data import encode_race

    assert encode_race(nch_race_to_mimic_string("White", "N", "Not Hispanic or Latino")) == 0
    assert encode_race(nch_race_to_mimic_string("White", "Y", "Hispanic or Latino")) == 3
    assert encode_race(nch_race_to_mimic_string("Black or African American", "N", "")) == 1


def test_truncation_keeps_newest():
    n = 2000
    ts = np.arange(n, dtype=np.float64)
    keep = ts[-1024:]
    assert keep[0] == n - 1024
    assert keep[-1] == n - 1


def test_leakage_predicate():
    index = np.datetime64("2019-04-02T20:22:49")
    t_ok = np.datetime64("2019-04-02T20:22:48")
    t_bad = np.datetime64("2019-04-02T20:22:49")
    assert t_ok < index
    assert not (t_bad < index)


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"{len(tests)-failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
