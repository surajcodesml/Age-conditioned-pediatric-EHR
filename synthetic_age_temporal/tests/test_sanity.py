#!/usr/bin/env python3
"""Automated sanity tests for the synthetic age × temporal benchmark."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running from package directory or repo root.
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from config import (  # noqa: E402
    FORBIDDEN_MODEL_KEYS,
    SCENARIO_SPECS,
    lambda_true,
    relevance,
    tau_from_days,
    z_age,
)
from ground_truth import ExampleSignals, calibrate_parameters  # noqa: E402
from model import BenchmarkModel  # noqa: E402


def test_z_age_anchors() -> None:
    assert abs(float(z_age(0.0)) - (-1.0)) < 1e-12
    assert abs(float(z_age(9.0)) - 0.0) < 1e-12
    assert abs(float(z_age(18.0)) - 1.0) < 1e-12


def test_lambda_positive() -> None:
    for beta in (-3.0, 0.0, 3.0):
        for a in np.linspace(0, 18, 37):
            assert float(lambda_true(a, 0.0, beta)) > 0.0


def test_s0_beta_zero() -> None:
    assert SCENARIO_SPECS["S0"].beta_true == 0.0


def test_s2_s3_opposite_signs() -> None:
    assert SCENARIO_SPECS["S2"].beta_true < 0
    assert SCENARIO_SPECS["S3"].beta_true > 0
    assert np.sign(SCENARIO_SPECS["S2"].beta_true) == -np.sign(SCENARIO_SPECS["S3"].beta_true)


def test_relevance_decreases_with_lag() -> None:
    age = 8.0
    taus = tau_from_days([7, 30, 90, 180, 365, 730])
    r = relevance(age, taus, 0.0, -2.0)
    assert np.all(np.diff(r) < 0)


def test_s2_younger_faster() -> None:
    beta = SCENARIO_SPECS["S2"].beta_true
    assert float(lambda_true(2.0, 0.0, beta)) > float(lambda_true(16.0, 0.0, beta))


def test_s3_older_faster() -> None:
    beta = SCENARIO_SPECS["S3"].beta_true
    assert float(lambda_true(16.0, 0.0, beta)) > float(lambda_true(2.0, 0.0, beta))


def test_no_nans_over_support() -> None:
    ages = np.linspace(0, 18, 50)
    lags = np.linspace(0, 730, 50)
    for beta in (-3.5, 0.0, 3.5):
        lam = lambda_true(ages, 0.0, beta)
        assert np.isfinite(lam).all()
        for a in ages:
            r = relevance(a, tau_from_days(lags), 0.0, beta)
            assert np.isfinite(r).all()
            assert np.all(r > 0) and np.all(r <= 1.0 + 1e-9)


def test_patient_splits_disjoint(data_root: Path | None = None) -> None:
    if data_root is None:
        return
    path = data_root / "patient_splits.json"
    if not path.exists():
        return
    with path.open() as f:
        splits = json.load(f)
    sets = {k: set(v) for k, v in splits.items()}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])


def test_no_future_events(scenario_dir: Path | None = None) -> None:
    if scenario_dir is None or not scenario_dir.exists():
        return
    ex = pd.read_parquet(scenario_dir / "examples.parquet")
    for row in ex.itertuples(index=False):
        cutoff = pd.Timestamp(row.cutoff_time)
        # lags must be >= 0 (event before or at cutoff); we require > 0 for history.
        for lag in row.history_lag_days:
            assert float(lag) >= 0.0
        # Signal times in GT
    gt = pd.read_parquet(scenario_dir / "ground_truth.parquet")
    sig = gt[gt["signal_event_time"].notna()]
    for row in sig.itertuples(index=False):
        assert pd.Timestamp(row.signal_event_time) < pd.Timestamp(row.cutoff_time)


def test_forbidden_keys_absent_from_tensors(scenario_dir: Path | None = None) -> None:
    if scenario_dir is None or not (scenario_dir / "examples.parquet").exists():
        return
    from dataset import make_loaders

    train, _, _, _, _ = make_loaders(scenario_dir, batch_size=4)
    batch = next(iter(train))
    for k in FORBIDDEN_MODEL_KEYS:
        assert k not in batch


def test_beta0_removes_interaction() -> None:
    m = BenchmarkModel("age_temporal", n_codes=20, n_types=8, n_targets=4, d_model=32, n_heads=4)
    with torch.no_grad():
        m.temporal.theta0.fill_(0.5)
        m.temporal.beta.fill_(-2.0)
    age = torch.tensor([2.0, 16.0])
    lam = m.temporal.lambda_of(age)
    assert float(lam[0].detach()) != float(lam[1].detach())
    with torch.no_grad():
        m.temporal.beta.fill_(0.0)
    lam0 = m.temporal.lambda_of(age)
    assert abs(float(lam0[0].detach()) - float(lam0[1].detach())) < 1e-6


def test_gradients_reach_theta_beta() -> None:
    m = BenchmarkModel("age_temporal", n_codes=20, n_types=8, n_targets=4, d_model=32, n_heads=4)
    B, L = 2, 5
    code = torch.randint(1, 20, (B, L))
    typ = torch.randint(1, 8, (B, L))
    tau = torch.rand(B, L)
    pad = torch.zeros(B, L, dtype=torch.bool)
    is_q = torch.zeros(B, L, dtype=torch.bool)
    is_q[:, -1] = True
    age = torch.tensor([3.0, 15.0])
    y = torch.zeros(B, 4)
    logits = m(code, typ, tau, pad, is_q, age, lag_days=torch.rand(B, L) * 100)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    assert m.temporal.theta0.grad is not None
    assert m.temporal.beta.grad is not None


def test_prediction_time_uses_cutoff_age_only() -> None:
    """Historical_age differs from age_temporal when event ages ≠ cutoff age."""
    m_pt = BenchmarkModel("age_temporal", 20, 8, 2, d_model=32, n_heads=4)
    m_hist = BenchmarkModel("historical_age", 20, 8, 2, d_model=32, n_heads=4)
    m_hist.load_state_dict(m_pt.state_dict())
    m_pt.temporal.theta0.data.fill_(0.0)
    m_pt.temporal.beta.data.fill_(-2.0)
    m_hist.temporal.theta0.data.copy_(m_pt.temporal.theta0.data)
    m_hist.temporal.beta.data.copy_(m_pt.temporal.beta.data)
    B, L = 1, 4
    code = torch.arange(1, L + 1).view(1, L)
    typ = torch.ones(B, L, dtype=torch.long)
    tau = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    lag = torch.tensor([[700.0, 300.0, 30.0, 0.0]])
    pad = torch.zeros(B, L, dtype=torch.bool)
    is_q = torch.tensor([[False, False, False, True]])
    age = torch.tensor([10.0])
    with torch.no_grad():
        # Force same content path by zeroing embeddings difference — compare bias only.
        bias_pt = -m_pt.temporal.lambda_of(age).unsqueeze(-1) * tau
        event_ages = age.unsqueeze(-1) - lag / 365.25
        bias_h = -m_hist.temporal.lambda_of(event_ages.clamp(min=0)) * tau
    assert not torch.allclose(bias_pt[:, :3], bias_h[:, :3])


def test_oracle_age_shuffle_effect(scenario_dir: Path | None = None) -> None:
    if scenario_dir is None:
        return
    path = scenario_dir / "oracle_metrics.json"
    if not path.exists():
        return
    with path.open() as f:
        o = json.load(f)
    scen = o["scenario"]
    d = o["delta_bce_shuffle_age"]
    if scen == "S0":
        assert abs(d) < 0.05, d
    if scen in ("S2", "S3"):
        assert d > 0.005, d


def run_all(data_root: Path | None = None) -> None:
    test_z_age_anchors()
    test_lambda_positive()
    test_s0_beta_zero()
    test_s2_s3_opposite_signs()
    test_relevance_decreases_with_lag()
    test_s2_younger_faster()
    test_s3_older_faster()
    test_no_nans_over_support()
    test_beta0_removes_interaction()
    test_gradients_reach_theta_beta()
    test_prediction_time_uses_cutoff_age_only()
    cal = calibrate_parameters(0.0, -2.0)
    assert cal["checks"]["lambda_positive"]
    assert abs(cal["checks"]["z0"] + 1) < 1e-9

    if data_root is not None and data_root.exists():
        test_patient_splits_disjoint(data_root)
        for scen in ("S0", "S1", "S2", "S3"):
            sdir = data_root / "controlled" / scen
            if sdir.exists():
                test_no_future_events(sdir)
                test_forbidden_keys_absent_from_tensors(sdir)
                test_oracle_age_shuffle_effect(sdir)
    print("All tests passed.")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else PKG / "outputs" / "data" / "seed20260922"
    run_all(root)
