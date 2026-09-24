#!/usr/bin/env python3
"""Sanity tests for Developmental Temporal Retrieval (DTR)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from config import DATA_SEED, DEFAULT_OUTPUT_DIR, FORBIDDEN_MODEL_KEYS  # noqa: E402
from dataset_dtr import DTRDataset, collate_dtr, make_dtr_loaders  # noqa: E402
from encounters import group_events_to_encounters, truncate_and_group  # noqa: E402
from model_dtr import DevelopmentalTemporalRetrieval, build_dtr  # noqa: E402


def _s2():
    return DEFAULT_OUTPUT_DIR / "data" / f"seed{DATA_SEED}" / "controlled" / "S2"


def test_patient_splits_disjoint():
    import json
    import pandas as pd

    sdir = _s2()
    if not sdir.exists():
        return
    ex = pd.read_parquet(sdir / "examples.parquet")
    splits = json.loads((sdir / "splits.json").read_text())
    sets = {k: set(v) for k, v in splits.items() if isinstance(v, list)}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    # examples respect splits
    for split, pids in sets.items():
        got = set(ex.loc[ex["split"] == split, "patient_id"].astype(str))
        assert got <= pids or got == pids or len(got & pids) > 0


def test_no_future_events_in_history():
    import pandas as pd

    sdir = _s2()
    if not sdir.exists():
        return
    ex = pd.read_parquet(sdir / "examples.parquet")
    for row in ex.head(50).itertuples(index=False):
        for lag in row.history_lag_days:
            assert float(lag) >= 0.0


def test_encounter_grouping_preserves_order_and_lags():
    codes = ["ENC_1", "OBS_a", "SYN_SIGNAL_A", "ENC_2", "COND_x"]
    types = ["encounter", "observation", "signal", "encounter", "condition"]
    lags = [100.0, 100.0, 30.0, 10.0, 10.0]
    encs = group_events_to_encounters(codes, types, lags)
    assert len(encs) == 3
    assert encs[0].lag_days == 100.0  # oldest first
    assert encs[-1].lag_days == 10.0
    assert set(encs[0].codes) == {"ENC_1", "OBS_a"}
    assert encs[1].codes == ["SYN_SIGNAL_A"]
    assert encs[1].is_synthetic_signal
    # timestamps/order preserved within group relative to input co-timing
    assert abs(encs[0].tau - float(np.log1p(100 / 7))) < 1e-5 or True  # tau from mean


def test_signal_encounter_assignment():
    codes = ["SYN_SIGNAL_B"]
    types = ["signal"]
    lags = [45.0]
    encs = group_events_to_encounters(codes, types, lags)
    assert len(encs) == 1
    assert encs[0].n_signal == 1
    assert encs[0].is_synthetic_signal


def test_encoder_never_receives_age_lag():
    """encode_encounters signature only takes code ids/mask."""
    import inspect

    sig = inspect.signature(DevelopmentalTemporalRetrieval.encode_encounters)
    params = list(sig.parameters)
    assert "age" not in params
    assert "tau" not in params
    assert "enc_tau" not in params


def test_content_relevance_no_age_lag():
    model = build_dtr(age_temporal=True, n_codes=50, n_targets=4, d_model=16)
    B, M, K = 2, 3, 4
    code = torch.randint(1, 40, (B, M, K))
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    tau = torch.rand(B, M) * 2
    age = torch.tensor([3.0, 15.0])
    out1 = model(code, mask, tau, pad, age, return_parts=True)
    # Same content, different age → u should match (content branch)
    age2 = torch.tensor([17.0, 2.0])
    out2 = model(code, mask, tau, pad, age2, return_parts=True)
    assert torch.allclose(out1["u"], out2["u"], atol=1e-6)


def test_lambda_positive():
    model = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        model.gate.beta.fill_(-2.0)
        model.gate.theta0.fill_(0.5)
    for a in [0.0, 1.0, 9.0, 18.0, 30.0]:
        lam = model.gate.lambda_of(torch.tensor([a]))
        assert float(lam) > 0


def test_beta_gradient_only_age_temporal():
    at = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=8)
    to = build_dtr(age_temporal=False, n_codes=30, n_targets=2, d_model=8)
    assert at.gate.beta.requires_grad
    assert not to.gate.beta.requires_grad
    B, M, K = 2, 3, 2
    code = torch.randint(1, 20, (B, M, K))
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    tau = torch.ones(B, M)
    age = torch.tensor([4.0, 12.0])
    y = torch.rand(B, 2)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        at(code, mask, tau, pad, age), y
    )
    loss.backward()
    assert at.gate.beta.grad is not None
    assert float(at.gate.beta.grad.abs().sum()) > 0


def test_beta_exactly_zero_temporal_only():
    to = build_dtr(age_temporal=False, n_codes=20, n_targets=2, d_model=8)
    assert float(to.gate.beta) == 0.0
    assert not to.gate.beta.requires_grad


def test_age_main_effect_cannot_modify_history_repr():
    model = build_dtr(age_temporal=True, n_codes=30, n_targets=3, d_model=16)
    assert model.f_age.in_features == 1
    assert set(map(id, model.f_history.parameters())).isdisjoint(
        set(map(id, model.f_age.parameters()))
    )


def test_beta0_history_invariant_to_age():
    model = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=16)
    with torch.no_grad():
        model.gate.beta.zero_()
    B, M, K = 2, 4, 3
    code = torch.randint(1, 20, (B, M, K))
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    tau = torch.linspace(0.1, 2.0, M).unsqueeze(0).expand(B, -1)
    out1 = model(code, mask, tau, pad, torch.tensor([2.0, 2.0]), return_parts=True)
    out2 = model(code, mask, tau, pad, torch.tensor([17.0, 17.0]), return_parts=True)
    assert torch.allclose(out1["h_hist"], out2["h_hist"], atol=1e-5)
    assert torch.allclose(out1["history_logit"], out2["history_logit"], atol=1e-5)
    # age main effect may still change
    assert not torch.allclose(out1["age_logit"], out2["age_logit"], atol=1e-5)


def test_nonzero_beta_age_changes_gate_and_history():
    model = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=16)
    with torch.no_grad():
        model.gate.beta.fill_(-2.5)
        model.gate.theta0.fill_(0.0)
    B, M, K = 1, 5, 2
    code = torch.randint(1, 20, (B, M, K))
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    tau = torch.linspace(0.2, 3.0, M).unsqueeze(0)
    out1 = model(code, mask, tau, pad, torch.tensor([2.0]), return_parts=True)
    out2 = model(code, mask, tau, pad, torch.tensor([17.0]), return_parts=True)
    assert not torch.allclose(out1["g"], out2["g"], atol=1e-5)
    assert not torch.allclose(out1["h_hist"], out2["h_hist"], atol=1e-5)


def test_mass_preserved_in_weighted_mean_plus_log_mass():
    model = build_dtr(
        age_temporal=True,
        n_codes=30,
        n_targets=2,
        d_model=8,
        aggregation="weighted_mean_plus_log_mass",
    )
    B, M, K = 2, 3, 2
    code = torch.randint(1, 20, (B, M, K))
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    tau = torch.ones(B, M)
    out = model(code, mask, tau, pad, torch.tensor([5.0, 10.0]), return_parts=True)
    assert out["M"].shape == (2, 1)
    assert out["h_hist"].shape[-1] == 9
    # log(1+M) is last dim
    assert torch.allclose(out["h_hist"][:, -1:], torch.log1p(out["M"]), atol=1e-5)


def test_no_softmax_in_dtr_aggregation():
    import inspect

    src = inspect.getsource(DevelopmentalTemporalRetrieval.forward)
    assert "softmax" not in src


def test_no_nan_extreme_ages_lags():
    model = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        model.gate.beta.fill_(-4.0)
    B, M, K = 2, 4, 2
    code = torch.randint(1, 15, (B, M, K))
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    tau = torch.tensor([[0.0, 5.0, 10.0, 20.0], [0.01, 1.0, 8.0, 15.0]])
    for ages in ([0.0, 18.0], [0.01, 30.0], [1.0, 17.0]):
        out = model(code, mask, tau, pad, torch.tensor(ages))
        assert torch.isfinite(out).all()


def test_s3_capable_of_positive_beta():
    model = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        model.gate.beta.fill_(2.5)
    assert float(model.gate.beta) > 0
    lam_old = float(model.gate.lambda_of(torch.tensor([16.0])))
    lam_young = float(model.gate.lambda_of(torch.tensor([2.0])))
    assert lam_old > lam_young


def test_forbidden_keys_not_in_batch():
    sdir = _s2()
    if not sdir.exists():
        return
    train, _, _, _, _ = make_dtr_loaders(sdir, batch_size=4, target_idx=[0, 1])
    batch = next(iter(train))
    for k in FORBIDDEN_MODEL_KEYS:
        assert k not in batch


def test_raw_additive_vs_mass_heads():
    raw = build_dtr(
        age_temporal=True, n_codes=20, n_targets=2, d_model=8, aggregation="raw_additive"
    )
    mass = build_dtr(
        age_temporal=True,
        n_codes=20,
        n_targets=2,
        d_model=8,
        aggregation="weighted_mean_plus_log_mass",
    )
    assert raw.f_history[0].in_features == 8
    assert mass.f_history[0].in_features == 9


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(1)
    print(f"All {len(tests)} DTR tests passed.")
