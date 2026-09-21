#!/usr/bin/env python3
"""Sanity tests for Stage-2 NCH pediatric adaptation."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from model_new.data import pretrain_collate, select_forecast_input_indices
from stage1_mimic_pretrain.metrics import shuffle_age_years
from stage1_mimic_pretrain.model import MinimalDKMModel
from stage2_nch.config import (
    EMBEDDING_PATH,
    NCH_SPLIT_DIR,
    PEDIATRIC_AGE_CENTER_YEARS,
    PEDIATRIC_AGE_SCALE_YEARS,
    STAGE1_BEST_CKPT,
    VOCAB_PATH,
    z_pediatric_numpy,
)
from stage2_nch.init_from_stage1 import build_stage2_model, logits_max_abs_diff
from stage2_nch.sign_test import run_lambda0_sign_test

VOCAB = 24
D_IN = 16
D_MODEL = 32
N_HEADS = 4


def _table(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB + 2, D_IN, generator=g)


def _items(rng: np.random.Generator, lengths=(8, 5, 6, 3), ages=None) -> list[dict]:
    items = []
    for i, n in enumerate(lengths):
        t = np.sort(rng.random(n) * 400.0).astype(np.float32)
        age_years = 9.0 if ages is None else float(ages[i])
        items.append({
            "code_indices": rng.integers(0, VOCAB, size=n),
            "timestamps_days": t,
            "age_days": np.full(n, age_years * 365.25, dtype=np.float32),
            "sex": 1,
            "race": 0,
            "unk_vocab_index": VOCAB,
            "target_codes": (rng.random(VOCAB) < 0.15).astype(np.float32),
            "target_time": float(t.max() + 1.0),
        })
    return items


def _batch(seed: int = 0, **kw) -> dict:
    return pretrain_collate(_items(np.random.default_rng(seed), **kw))


def _model(arm: str = "age_temporal", seed: int = 0, table=None) -> MinimalDKMModel:
    return MinimalDKMModel(
        num_codes=VOCAB, embedding_table=table if table is not None else _table(seed),
        arm=arm, seed=seed, d_model=D_MODEL, n_layers=1, n_heads=N_HEADS,
        demo_dim=9, demo_hidden=8,
        age_mean=PEDIATRIC_AGE_CENTER_YEARS, age_sd=PEDIATRIC_AGE_SCALE_YEARS,
    )


def test_pediatric_z_endpoints():
    z = z_pediatric_numpy([0.0, 9.0, 18.0], clip=False)
    assert abs(z[0] + 1.0) < 1e-12
    assert abs(z[1] - 0.0) < 1e-12
    assert abs(z[2] - 1.0) < 1e-12
    m = _model("age_temporal")
    age = torch.tensor([0.0, 9.0, 18.0])
    zz = m.temporal.z_of(age)
    assert torch.allclose(zz, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-6)


def test_arms_identical_at_beta_zero():
    table = _table(1)
    batch = _batch(2, ages=(1.0, 5.0, 10.0, 17.0))
    a = _model("age_temporal", seed=3, table=table)
    b = _model("no_interaction", seed=3, table=table)
    with torch.no_grad():
        a.temporal.lambda0.fill_(-1.6)
        b.temporal.lambda0.fill_(-1.6)
        a.temporal.beta.zero_()
        assert float(b.temporal.beta.detach()) == 0.0
        assert not b.temporal.beta.requires_grad
        la = a(batch)["code_logits"]
        lb = b(batch)["code_logits"]
    assert torch.allclose(la, lb, atol=1e-6, rtol=1e-5)


def test_age_changes_logits_only_when_beta_nonzero():
    batch = _batch(4, ages=(1.0, 5.0, 10.0, 17.0))
    m = _model("age_temporal", seed=4)
    with torch.no_grad():
        m.temporal.lambda0.fill_(0.4)
        m.temporal.beta.fill_(1.5)
        base = m(batch)["code_logits"]
        bumped = dict(batch)
        bumped["age_years"] = batch["age_years"] + 4.0 * batch["attention_mask"].float()
        alt = m(bumped)["code_logits"]
    assert not torch.allclose(base, alt, atol=1e-6)
    m0 = _model("no_interaction", seed=4)
    with torch.no_grad():
        m0.temporal.lambda0.fill_(0.4)
        b0 = m0(batch)["code_logits"]
        bumped = dict(batch)
        bumped["age_years"] = batch["age_years"] + 4.0 * batch["attention_mask"].float()
        a0 = m0(bumped)["code_logits"]
    assert torch.allclose(b0, a0, atol=1e-6, rtol=1e-5)


def test_beta_and_lambda0_gradients():
    batch = _batch(6, ages=(0.5, 4.0, 11.0, 16.0))
    m = _model("age_temporal", seed=6)
    loss = F.binary_cross_entropy_with_logits(m(batch)["code_logits"], batch["target_codes"])
    loss.backward()
    assert m.temporal.lambda0.grad is not None and float(m.temporal.lambda0.grad.abs()) > 0
    assert m.temporal.beta.grad is not None and float(m.temporal.beta.grad.abs()) > 0
    m2 = _model("no_interaction", seed=6)
    loss2 = F.binary_cross_entropy_with_logits(m2(batch)["code_logits"], batch["target_codes"])
    loss2.backward()
    assert m2.temporal.lambda0.grad is not None and float(m2.temporal.lambda0.grad.abs()) > 0
    assert float(m2.temporal.beta.detach()) == 0.0
    assert (m2.temporal.beta.grad is None) or float(m2.temporal.beta.grad.abs()) == 0.0


def test_padding_mask():
    m = _model("age_temporal", seed=9)
    batch = _batch(9, lengths=(8, 3, 5, 1))
    with torch.no_grad():
        out = m(batch, need_diagnostics=True)
        attn = out["attn"]
        mask = batch["attention_mask"]
        for b in range(mask.shape[0]):
            pad = ~mask[b]
            valid = mask[b]
            if pad.any() and valid.any():
                mass = attn[b][:, valid][:, :, pad]
                assert float(mass.abs().max()) < 1e-6


def test_no_future_leakage_in_window():
    ts = np.array([0.0, 1.0, 5.0, 10.0, 10.0, 12.0], dtype=np.float64)
    target_time = 10.0
    sel = select_forecast_input_indices(ts, target_time, 1024)
    assert float(ts[sel].max()) < target_time
    assert 3 not in sel.tolist() and 4 not in sel.tolist()


def test_nan_free_pediatric_extremes():
    m = _model("age_temporal", seed=10)
    with torch.no_grad():
        m.temporal.lambda0.fill_(-2.0)
        m.temporal.beta.fill_(1.5)
    batch = _batch(10, ages=(0.0, 1.0, 9.0, 18.0))
    ts = batch["timestamps_days"].clone()
    ts[:, -1] = 1.0e6
    batch = dict(batch)
    batch["timestamps_days"] = ts
    out = m(batch, need_diagnostics=True)
    for key in ("code_logits", "h", "attn", "temporal_bias"):
        assert torch.isfinite(out[key]).all(), key


def test_pooling_independent_of_beta():
    m = _model("age_temporal", seed=2)
    assert m.pool_temporal_bias is False
    with torch.no_grad():
        m.temporal.beta.fill_(3.0)
        m.temporal.lambda0.fill_(1.2)
    e = torch.randn(2, 6, D_MODEL)
    tau = torch.rand(2, 6)
    mask = torch.ones(2, 6, dtype=torch.bool)
    a1 = torch.tensor([0.0, 18.0])
    a2 = torch.tensor([18.0, 0.0])
    with torch.no_grad():
        h1, attn1, bias1 = m.pooling(e, tau, mask, a1, need_weights=True)
        h2, attn2, bias2 = m.pooling(e, tau, mask, a2, need_weights=True)
    assert bias1 is None and bias2 is None
    assert torch.allclose(h1, h2)
    assert torch.allclose(attn1, attn2)


def test_shuffle_changes_outputs_when_beta_set():
    m = _model("age_temporal", seed=8)
    with torch.no_grad():
        m.temporal.beta.fill_(2.0)
        m.temporal.lambda0.fill_(0.3)
    batch = _batch(8, ages=(0.5, 4.0, 12.0, 17.0))
    g = torch.Generator().manual_seed(0)
    shuffled = shuffle_age_years(batch["age_years"], batch["attention_mask"], g)
    with torch.no_grad():
        a = m(batch)["code_logits"]
        b = m(batch, age_years_for_bias=shuffled)["code_logits"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_lambda0_sign():
    rep = run_lambda0_sign_test(seed=0)
    assert rep["passed"], rep
    assert rep["positive_lambda0_is_recency"]
    assert rep["negative_lambda0_is_long_range"]


def test_stage1_checkpoint_loads_and_resets_beta():
    if not STAGE1_BEST_CKPT.exists() or not EMBEDDING_PATH.exists():
        return
    a, ra = build_stage2_model(
        num_codes=30635, arm="age_temporal", embedding_path=EMBEDDING_PATH,
        stage1_ckpt=STAGE1_BEST_CKPT, seed=0, demo_hidden=64,
    )
    b, _rb = build_stage2_model(
        num_codes=30635, arm="no_interaction", embedding_path=EMBEDDING_PATH,
        stage1_ckpt=STAGE1_BEST_CKPT, seed=0, demo_hidden=64,
    )
    assert abs(float(a.temporal.beta.detach())) < 1e-12
    assert abs(float(b.temporal.beta.detach())) < 1e-12
    assert a.temporal.beta.requires_grad
    assert not b.temporal.beta.requires_grad
    assert abs(float(a.age_mean) - 9.0) < 1e-6
    assert abs(float(a.age_sd) - 9.0) < 1e-6
    assert not a.embedding_table.requires_grad
    dummy = {
        "code_indices": torch.tensor([[2, 3, 4, 0]]),
        "timestamps_days": torch.tensor([[0.0, 2.0, 9.0, 0.0]], dtype=torch.float64),
        "attention_mask": torch.tensor([[True, True, True, False]]),
        "lengths": torch.tensor([3]),
        "age_years": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "demographics": torch.zeros(1, 4, 9),
        "target_codes": torch.zeros(1, 30635),
    }
    dummy["demographics"][0, :, 0] = dummy["age_years"]
    dummy["demographics"][0, :, 2] = 1.0
    dummy["code_indices"] = dummy["code_indices"] * dummy["attention_mask"].long()
    diff = logits_max_abs_diff(a, b, dummy)
    assert diff < 1e-5, diff
    assert ra["transferred"]["lambda0"] != 0.0


def test_vocab_indices_align():
    if not VOCAB_PATH.exists() or not EMBEDDING_PATH.exists() or not STAGE1_BEST_CKPT.exists():
        return
    vocab = json.loads(VOCAB_PATH.read_text())
    v = len(vocab)
    blob = torch.load(STAGE1_BEST_CKPT, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    assert sd["embedding_table"].shape[0] == v + 2
    assert sd["head.net.2.weight"].shape[0] == v
    emb = torch.load(EMBEDDING_PATH, map_location="cpu", weights_only=False)
    table = emb["embeddings"] if isinstance(emb, dict) else emb
    assert table.shape[0] == v + 2
    assert int(sd["embedding_table"].shape[1]) == int(table.shape[1])


def test_patient_split_json_disjoint():
    ids = {}
    for name in ("train", "val", "test"):
        path = NCH_SPLIT_DIR / f"{name}_patient_ids.json"
        if not path.exists():
            return
        ids[name] = set(json.loads(path.read_text()))
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])


def test_tiny_overfit():
    m = _model("age_temporal", seed=11)
    batch = _batch(11, lengths=(6, 6, 6, 6), ages=(1.0, 5.0, 10.0, 16.0))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(m(batch)["code_logits"], batch["target_codes"])
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.02, (losses[0], losses[-1])


TESTS = [
    test_pediatric_z_endpoints,
    test_arms_identical_at_beta_zero,
    test_age_changes_logits_only_when_beta_nonzero,
    test_beta_and_lambda0_gradients,
    test_padding_mask,
    test_no_future_leakage_in_window,
    test_nan_free_pediatric_extremes,
    test_pooling_independent_of_beta,
    test_shuffle_changes_outputs_when_beta_set,
    test_lambda0_sign,
    test_stage1_checkpoint_loads_and_resets_beta,
    test_vocab_indices_align,
    test_patient_split_json_disjoint,
    test_tiny_overfit,
]


def run_all() -> dict:
    results = []
    for fn in TESTS:
        try:
            fn()
            results.append({"name": fn.__name__, "ok": True, "error": None})
            print(f"PASS  {fn.__name__}", flush=True)
        except Exception as exc:
            results.append({"name": fn.__name__, "ok": False, "error": repr(exc)})
            print(f"FAIL  {fn.__name__}: {exc!r}", flush=True)
    n_ok = sum(1 for r in results if r["ok"])
    print(f"{n_ok}/{len(results)} passed", flush=True)
    return {"n": len(results), "n_ok": n_ok, "results": results}


if __name__ == "__main__":
    summary = run_all()
    raise SystemExit(0 if summary["n_ok"] == summary["n"] else 1)
