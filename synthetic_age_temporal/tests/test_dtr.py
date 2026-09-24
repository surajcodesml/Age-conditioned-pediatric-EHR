#!/usr/bin/env python3
"""Unit tests for canonical Content-Persistence Developmental Temporal Retrieval."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from config import DATA_SEED, DEFAULT_OUTPUT_DIR, FORBIDDEN_MODEL_KEYS  # noqa: E402
from dataset_dtr import make_dtr_loaders  # noqa: E402
from encounters import group_events_to_encounters  # noqa: E402
from model_dtr import (  # noqa: E402
    CONTENT_SCORE_EXP_CLAMP,
    DevelopmentalTemporalRetrieval,
    build_dtr,
    load_legacy_dtr_checkpoint,
    matched_arm_init_check,
)


def _batch(B=2, M=4, K=3, n_codes=30, n_targets=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    code = torch.randint(1, n_codes - 1, (B, M, K), generator=g)
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    # last encounter padded in each example
    pad[:, -1] = True
    tau = torch.linspace(0.2, 2.5, M).unsqueeze(0).expand(B, -1).contiguous()
    age = torch.tensor([3.0, 15.0][:B], dtype=torch.float32)
    return {
        "enc_code_ids": code,
        "enc_code_mask": mask,
        "enc_tau": tau,
        "enc_padding_mask": pad,
        "age": age,
        "labels": torch.rand(B, n_targets, generator=g),
    }


def _s2():
    return DEFAULT_OUTPUT_DIR / "data" / f"seed{DATA_SEED}" / "controlled" / "S2"


# ----- existing data sanity (unchanged) -----


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
    assert encs[0].lag_days == 100.0
    assert encs[-1].lag_days == 10.0


def test_signal_encounter_assignment():
    encs = group_events_to_encounters(["SYN_SIGNAL_B"], ["signal"], [45.0])
    assert encs[0].n_signal == 1


# ----- required canonical tests (1–20) -----


def test_01_beta0_arms_identical_logits():
    """1. beta=0 makes temporal-only and age-temporal logits identical."""
    torch.manual_seed(0)
    at = build_dtr(age_temporal=True, n_codes=40, n_targets=2, d_model=16)
    to = build_dtr(age_temporal=False, n_codes=40, n_targets=2, d_model=16)
    to.load_state_dict(at.state_dict(), strict=False)
    with torch.no_grad():
        at.beta.zero_()
        to.beta.zero_()
    b = _batch(n_codes=40)
    diff = matched_arm_init_check(at, to, b)
    assert diff < 1e-6, diff


def test_02_grad_beta_only_age_temporal():
    """2. Gradients reach beta only in age-temporal."""
    at = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=8)
    to = build_dtr(age_temporal=False, n_codes=30, n_targets=2, d_model=8)
    assert at.beta.requires_grad and not to.beta.requires_grad
    b = _batch(n_codes=30)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        at(b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"], b["age"]),
        b["labels"],
    )
    loss.backward()
    assert at.beta.grad is not None and float(at.beta.grad.abs().sum()) > 0


def test_03_grad_theta0():
    """3. Gradients reach theta0."""
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=8)
    b = _batch(n_codes=30)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        m(b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"], b["age"]),
        b["labels"],
    )
    loss.backward()
    assert m.theta0.grad is not None and float(m.theta0.grad.abs().sum()) > 0


def test_04_grad_persistence_projection():
    """4. Gradients reach persistence_projection."""
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=8)
    b = _batch(n_codes=30)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        m(b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"], b["age"]),
        b["labels"],
    )
    loss.backward()
    assert m.persistence_projection.weight.grad is not None
    assert float(m.persistence_projection.weight.grad.abs().sum()) > 0


def test_05_zero_init_persistence_still_gets_grad():
    """5. Zero-initialized persistence projection still receives gradient."""
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=8)
    assert float(m.persistence_projection.weight.abs().sum()) == 0.0
    assert float(m.persistence_projection.bias.abs().sum()) == 0.0
    b = _batch(n_codes=30)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        m(b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"], b["age"]),
        b["labels"],
    )
    loss.backward()
    assert float(m.persistence_projection.weight.grad.abs().sum()) > 0


def test_06_content_changes_persistence():
    """6. Changing encounter content can change persistence."""
    m = build_dtr(age_temporal=True, n_codes=40, n_targets=2, d_model=16)
    with torch.no_grad():
        m.persistence_projection.weight.normal_(0, 0.1)
    b1 = _batch(n_codes=40, seed=1)
    b2 = _batch(n_codes=40, seed=2)
    o1 = m(
        b1["enc_code_ids"], b1["enc_code_mask"], b1["enc_tau"], b1["enc_padding_mask"], b1["age"],
        return_parts=True,
    )
    o2 = m(
        b2["enc_code_ids"], b2["enc_code_mask"], b2["enc_tau"], b2["enc_padding_mask"], b2["age"],
        return_parts=True,
    )
    assert not torch.allclose(o1["theta_content"], o2["theta_content"], atol=1e-5)


def test_07_beta0_history_invariant_to_age():
    """7. Changing age with beta=0 does not change the history representation."""
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=16)
    with torch.no_grad():
        m.beta.zero_()
    b = _batch(n_codes=30)
    o1 = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        torch.tensor([2.0, 2.0]), return_parts=True,
    )
    o2 = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        torch.tensor([17.0, 17.0]), return_parts=True,
    )
    assert torch.allclose(o1["h_hist"], o2["h_hist"], atol=1e-5)
    assert torch.allclose(o1["history_logit"], o2["history_logit"], atol=1e-5)
    assert not torch.allclose(o1["age_logit"], o2["age_logit"], atol=1e-5)


def test_08_nonzero_beta_age_changes_gate():
    """8. Changing age with beta!=0 does change the temporal gate."""
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=16)
    with torch.no_grad():
        m.beta.fill_(-2.5)
    b = _batch(B=1, n_codes=30)
    o1 = m(
        b["enc_code_ids"][:1], b["enc_code_mask"][:1], b["enc_tau"][:1],
        b["enc_padding_mask"][:1], torch.tensor([2.0]), return_parts=True,
    )
    o2 = m(
        b["enc_code_ids"][:1], b["enc_code_mask"][:1], b["enc_tau"][:1],
        b["enc_padding_mask"][:1], torch.tensor([17.0]), return_parts=True,
    )
    assert not torch.allclose(o1["g"], o2["g"], atol=1e-5)
    assert not torch.allclose(o1["h_hist"], o2["h_hist"], atol=1e-5)


def test_09_content_relevance_no_age_lag_input():
    """9. Content relevance receives no explicit age or lag input."""
    # Signature / wiring: content_key only takes v; u independent of age.
    m = build_dtr(age_temporal=True, n_codes=40, n_targets=2, d_model=16)
    b = _batch(n_codes=40)
    o1 = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        torch.tensor([3.0, 15.0]), return_parts=True,
    )
    o2 = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        torch.tensor([17.0, 2.0]), return_parts=True,
    )
    assert torch.allclose(o1["u"], o2["u"], atol=1e-6)
    sig = inspect.signature(m.content_key.forward)
    assert "age" not in sig.parameters and "tau" not in sig.parameters


def test_10_persistence_projection_no_age_lag_input():
    """10. Persistence projection receives no explicit age or lag input."""
    m = build_dtr(age_temporal=True, n_codes=40, n_targets=2, d_model=16)
    with torch.no_grad():
        m.persistence_projection.weight.normal_(0, 0.05)
    b = _batch(n_codes=40)
    o1 = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        torch.tensor([3.0, 15.0]), return_parts=True,
    )
    o2 = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        torch.tensor([17.0, 2.0]), return_parts=True,
    )
    assert torch.allclose(o1["theta_content"], o2["theta_content"], atol=1e-6)
    src = inspect.getsource(DevelopmentalTemporalRetrieval.forward)
    # persistence_projection called on v only
    assert "persistence_projection(v)" in src or "persistence_offset(v)" in src


def test_11_lambda_always_positive():
    """11. Lambda is always positive."""
    m = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        m.beta.fill_(-4.0)
        m.theta0.fill_(-2.0)
        m.persistence_projection.weight.normal_(0, 1.0)
    b = _batch(n_codes=20)
    out = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        b["age"], return_parts=True,
    )
    hist = ~b["enc_padding_mask"]
    assert (out["lambda"][hist] > 0).all()


def test_12_larger_lambda_stronger_decay():
    """12. Larger lambda produces stronger decay with lag."""
    m = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    tau = torch.tensor([[1.0, 2.0]])
    age = torch.tensor([9.0])
    g_lo, _ = m.temporal_gate(age, tau, persistence_offset=torch.tensor([[-1.0, -1.0]]))
    g_hi, _ = m.temporal_gate(age, tau, persistence_offset=torch.tensor([[2.0, 2.0]]))
    # higher persistence offset → higher λ → smaller gate
    assert (g_hi < g_lo).all()
    # within a row, larger tau → smaller gate
    assert g_lo[0, 1] < g_lo[0, 0]


def test_13_padded_encounters_zero_weight():
    """13. Padded encounters contribute zero history weight."""
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=8)
    b = _batch(n_codes=30)
    out = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        b["age"], return_parts=True,
    )
    assert (out["w"][b["enc_padding_mask"]] == 0).all()
    assert (out["theta_content"][b["enc_padding_mask"]] == 0).all()


def test_14_raw_additive_aggregation():
    """14. Raw additive aggregation is used."""
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=8)
    assert m.aggregation == "raw_additive"
    b = _batch(n_codes=30)
    out = m(
        b["enc_code_ids"], b["enc_code_mask"], b["enc_tau"], b["enc_padding_mask"],
        b["age"], return_parts=True,
    )
    # h = Σ w v
    v = out["v"]
    expected = (out["w"].unsqueeze(-1) * v).sum(dim=1)
    assert torch.allclose(out["h_hist"], expected, atol=1e-5)


def test_15_no_softmax_over_history_weights():
    """15. No softmax is applied over temporal history weights."""
    src = inspect.getsource(DevelopmentalTemporalRetrieval.forward)
    # Disallow actual softmax calls (comments mentioning "NOT softmax" are fine).
    assert "torch.softmax" not in src
    assert "F.softmax" not in src
    assert ".softmax(" not in src


def test_16_s2_can_learn_negative_beta():
    """16. S2 can learn negative beta (capacity / sign check)."""
    m = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        m.beta.fill_(-2.5)
    assert float(m.beta) < 0
    lam_young = float(m.lambda_of(torch.tensor([2.0])))
    lam_old = float(m.lambda_of(torch.tensor([16.0])))
    # β<0 → younger has larger λ
    assert lam_young > lam_old


def test_17_s3_can_learn_positive_beta():
    """17. S3 can learn positive beta."""
    m = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        m.beta.fill_(2.5)
    assert float(m.beta) > 0
    lam_old = float(m.lambda_of(torch.tensor([16.0])))
    lam_young = float(m.lambda_of(torch.tensor([2.0])))
    assert lam_old > lam_young


def test_18_no_nan_extreme_ages_lags():
    """18. No NaN/Inf for extreme valid ages/lags."""
    m = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        m.beta.fill_(-4.0)
        m.persistence_projection.weight.normal_(0, 0.5)
    B, M, K = 2, 4, 2
    code = torch.randint(1, 15, (B, M, K))
    mask = torch.ones(B, M, K, dtype=torch.bool)
    pad = torch.zeros(B, M, dtype=torch.bool)
    tau = torch.tensor([[0.0, 5.0, 10.0, 20.0], [0.01, 1.0, 8.0, 15.0]])
    for ages in ([0.0, 18.0], [0.01, 30.0], [1.0, 17.0]):
        out = m(code, mask, tau, pad, torch.tensor(ages))
        assert torch.isfinite(out).all()


def test_19_gt_persistence_labels_never_inputs():
    """19. Ground-truth synthetic persistence labels are never inputs."""
    forbidden = set(FORBIDDEN_MODEL_KEYS) | {
        "persistence_class",
        "persistence_group",
        "true_theta_content",
        "oracle_relevance",
        "beta_true",
    }
    sdir = _s2()
    if not sdir.exists():
        return
    train, _, _, _, _ = make_dtr_loaders(sdir, batch_size=4, target_idx=[0, 1])
    batch = next(iter(train))
    for k in forbidden:
        assert k not in batch


def test_20_matched_arms_identical_init():
    """20. Matched arms share identical initialization."""
    torch.manual_seed(42)
    at = build_dtr(age_temporal=True, n_codes=30, n_targets=2, d_model=16)
    to = build_dtr(age_temporal=False, n_codes=30, n_targets=2, d_model=16)
    to.load_state_dict(at.state_dict(), strict=False)
    for (n1, p1), (n2, p2) in zip(at.named_parameters(), to.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2), n1
    assert float(at.beta) == 0.0 and float(to.beta) == 0.0


def test_encoder_never_receives_age_lag():
    sig = inspect.signature(DevelopmentalTemporalRetrieval.encode_encounters)
    params = list(sig.parameters)
    assert "age" not in params and "tau" not in params


def test_age_main_effect_separated():
    m = build_dtr(age_temporal=True, n_codes=30, n_targets=3, d_model=16)
    assert m.age_head.in_features == 1
    assert set(map(id, m.history_head.parameters())).isdisjoint(
        set(map(id, m.age_head.parameters()))
    )


def test_legacy_checkpoint_migration_warns_on_missing_persistence():
    m = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    # Minimal legacy-like dict without persistence_projection
    legacy = {
        "code_emb.weight": m.code_emb.weight.detach().clone(),
        "enc_mlp.0.weight": m.enc_mlp[0].weight.detach().clone(),
        "enc_mlp.0.bias": m.enc_mlp[0].bias.detach().clone(),
        "enc_mlp.3.weight": m.enc_mlp[3].weight.detach().clone(),
        "enc_mlp.3.bias": m.enc_mlp[3].bias.detach().clone(),
        "W_k.weight": m.content_key.weight.detach().clone(),
        "q": m.content_query.detach().clone(),
        "gate.theta0": m.theta0.detach().clone(),
        "gate.beta": m.beta.detach().clone(),
        "f_history.0.weight": m.history_head[0].weight.detach().clone(),
        "f_history.0.bias": m.history_head[0].bias.detach().clone(),
        "f_history.2.weight": m.history_head[2].weight.detach().clone(),
        "f_history.2.bias": m.history_head[2].bias.detach().clone(),
        "f_age.weight": m.age_head.weight.detach().clone(),
        "f_age.bias": m.age_head.bias.detach().clone(),
        "bias": m.bias.detach().clone(),
    }
    m2 = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    with torch.no_grad():
        m2.persistence_projection.weight.fill_(9.0)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_legacy_dtr_checkpoint(m2, legacy, strict=False)
        assert any("persistence_projection" in str(x.message) for x in w)
    # Missing persist → left at prior init of m2 unless remapped; we warn and do not
    # overwrite with garbage. Zero-init preferred: re-zero after warn path is caller's job.
    # Migration should have transferred theta0/beta/etc.
    assert torch.allclose(m2.theta0, m.theta0)


def test_content_score_clamp_documented():
    assert CONTENT_SCORE_EXP_CLAMP == 20.0
    src = inspect.getsource(DevelopmentalTemporalRetrieval.forward)
    assert "CONTENT_SCORE_EXP_CLAMP" in src or "clamp(max=" in src


def test_raw_additive_default_not_mass():
    m = build_dtr(age_temporal=True, n_codes=20, n_targets=2, d_model=8)
    assert m.history_head[0].in_features == 8  # d_model, no log-mass


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
