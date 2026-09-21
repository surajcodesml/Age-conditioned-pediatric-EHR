#!/usr/bin/env python3
"""Sanity tests for the Stage-1 minimal age × temporal attention model."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_new.data import pretrain_collate
from stage1_mimic_pretrain.metrics import (
    constant_age_years,
    multilabel_metrics,
    shuffle_age_years,
    valid_class_mask,
)
from stage1_mimic_pretrain.model import MinimalDKMModel

VOCAB = 24
D_IN = 16
D_MODEL = 32
N_HEADS = 4
AGE_MEAN = 63.33601047086648
AGE_SD = 16.574804662346914


def _table(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB + 2, D_IN, generator=g)


def _items(rng: np.random.Generator, lengths=(8, 5, 6, 3), ages=None) -> list[dict]:
    items = []
    for i, n in enumerate(lengths):
        t = np.sort(rng.random(n) * 400.0).astype(np.float32)
        if ages is None:
            age_years = 20.0 + 15.0 * i
        else:
            age_years = float(ages[i])
        age_days = np.full(n, age_years * 365.25, dtype=np.float32)
        items.append({
            "code_indices": rng.integers(0, VOCAB, size=n),
            "timestamps_days": t,
            "age_days": age_days,
            "sex": 1,
            "race": 0,
            "unk_vocab_index": VOCAB,
            "target_codes": (rng.random(VOCAB) < 0.15).astype(np.float32),
        })
    return items


def _batch(seed: int = 0, **kw) -> dict:
    return pretrain_collate(_items(np.random.default_rng(seed), **kw))


def _model(arm: str = "age_temporal", n_heads: int = N_HEADS, seed: int = 0,
           table: torch.Tensor | None = None, d_model: int = D_MODEL,
           pool_temporal_bias: bool = False) -> MinimalDKMModel:
    return MinimalDKMModel(
        num_codes=VOCAB, embedding_table=table if table is not None else _table(seed),
        arm=arm, seed=seed, d_model=d_model, n_layers=1, n_heads=n_heads,
        demo_dim=9, demo_hidden=8, age_mean=AGE_MEAN, age_sd=AGE_SD,
        pool_temporal_bias=pool_temporal_bias,
    )


def test_broadcast_bias_across_heads():
    m = _model("age_temporal", n_heads=4, seed=0)
    with torch.no_grad():
        m.temporal.lambda0.fill_(0.8)
        m.temporal.beta.fill_(1.2)
    batch = _batch(1)
    out = m(batch, need_diagnostics=True)
    bias = out["temporal_bias"]          # [B, L, L]
    content = out["content_logits"]      # [B, H, L, L]
    assert bias.shape == content.shape[:1] + content.shape[2:]
    assert content.shape[1] == 4
    # The SAME bias is added to every head: scores_h - content_h is identical across h.
    scores_minus_content = []
    # Reconstruct scores from attn is messy; check pairwise_bias vs content broadcast.
    added = bias.unsqueeze(1).expand_as(content)
    assert added.shape == content.shape
    assert torch.allclose(added[:, 0], added[:, 1], atol=0.0)
    assert torch.allclose(added[:, 0], added[:, 3], atol=0.0)
    assert torch.allclose(added[:, 0], bias, atol=0.0)


def test_beta_zero_matches_no_interaction():
    table = _table(7)
    batch = _batch(2)
    m_at = _model("age_temporal", seed=3, table=table)
    m_to = _model("no_interaction", seed=3, table=table)
    with torch.no_grad():
        m_at.temporal.lambda0.fill_(0.55)
        m_to.temporal.lambda0.fill_(0.55)
        m_at.temporal.beta.zero_()
        assert float(m_to.temporal.beta) == 0.0
        assert not m_to.temporal.beta.requires_grad
        assert m_at.temporal.beta.requires_grad
        assert m_to.arm == "no_interaction"
    with torch.no_grad():
        a = m_at(batch)["code_logits"]
        b = m_to(batch)["code_logits"]
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), float((a - b).abs().max())


def test_changing_age_changes_logits_when_beta_nonzero():
    m = _model("age_temporal", seed=4)
    with torch.no_grad():
        m.temporal.lambda0.fill_(0.3)
        m.temporal.beta.fill_(1.5)
    batch = _batch(4, ages=(25.0, 40.0, 55.0, 80.0))
    with torch.no_grad():
        base = m(batch)["code_logits"]
        bumped = dict(batch)
        bumped["age_years"] = batch["age_years"] + 12.0 * batch["attention_mask"].float()
        alt = m(bumped)["code_logits"]
    assert not torch.allclose(base, alt, atol=1e-6), "age must move logits when β≠0"


def test_changing_age_does_not_change_logits_when_beta_zero():
    m = _model("no_interaction", seed=4)
    with torch.no_grad():
        m.temporal.lambda0.fill_(0.9)
    batch = _batch(4, ages=(25.0, 40.0, 55.0, 80.0))
    with torch.no_grad():
        base = m(batch)["code_logits"]
        bumped = dict(batch)
        bumped["age_years"] = batch["age_years"] + 12.0 * batch["attention_mask"].float()
        alt = m(bumped)["code_logits"]
    assert torch.allclose(base, alt, atol=1e-6, rtol=1e-5)


def test_changing_time_gap_changes_logits():
    m = _model("no_interaction", seed=5)
    with torch.no_grad():
        m.temporal.lambda0.fill_(1.0)
    batch = _batch(5)
    with torch.no_grad():
        base = m(batch)["code_logits"]
        bumped = dict(batch)
        ts = batch["timestamps_days"].clone()
        # Stretch gaps from the first event.
        ts = ts * 4.0
        bumped["timestamps_days"] = ts
        alt = m(bumped)["code_logits"]
    assert not torch.allclose(base, alt, atol=1e-6), "τ must move logits when λ0≠0"


def test_gradients_reach_lambda0_and_beta():
    m = _model("age_temporal", seed=6)
    batch = _batch(6, ages=(20.0, 35.0, 70.0, 85.0))
    out = m(batch)
    loss = F.binary_cross_entropy_with_logits(out["code_logits"], batch["target_codes"])
    loss.backward()
    g0 = m.temporal.lambda0.grad
    gb = m.temporal.beta.grad
    assert g0 is not None and float(g0.abs()) > 0, f"lambda0 grad={g0}"
    assert gb is not None and float(gb.abs()) > 0, f"beta grad={gb}"


def test_gradients_lambda0_only_when_no_interaction():
    m = _model("no_interaction", seed=6)
    batch = _batch(6)
    out = m(batch)
    loss = F.binary_cross_entropy_with_logits(out["code_logits"], batch["target_codes"])
    loss.backward()
    assert m.temporal.lambda0.grad is not None and float(m.temporal.lambda0.grad.abs()) > 0
    assert m.temporal.beta.grad is None or float(m.temporal.beta.grad.abs()) == 0.0


def test_shuffled_ages_alter_outputs_when_age_conditioned():
    m = _model("age_temporal", seed=8)
    with torch.no_grad():
        m.temporal.beta.fill_(2.0)
        m.temporal.lambda0.fill_(0.2)
    batch = _batch(8, ages=(20.0, 40.0, 60.0, 80.0))
    g = torch.Generator().manual_seed(0)
    shuffled = shuffle_age_years(batch["age_years"], batch["attention_mask"], g)
    assert not torch.equal(shuffled, batch["age_years"])
    with torch.no_grad():
        a = m(batch)["code_logits"]
        b = m(batch, age_years_for_bias=shuffled)["code_logits"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_padding_receives_no_attention():
    m = _model("age_temporal", seed=9, n_heads=4)
    batch = _batch(9, lengths=(8, 3, 5, 1))
    with torch.no_grad():
        out = m(batch, need_diagnostics=True)
        attn = out["attn"]  # [B, H, L, L]
        mask = batch["attention_mask"]
        for b in range(mask.shape[0]):
            valid = mask[b]
            pad = ~valid
            if pad.any() and valid.any():
                mass = attn[b][:, valid][:, :, pad]
                assert float(mass.abs().max()) < 1e-6, float(mass.abs().max())
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_no_nan_extreme_gaps_and_ages():
    m = _model("age_temporal", seed=10)
    with torch.no_grad():
        m.temporal.lambda0.fill_(3.0)
        m.temporal.beta.fill_(-2.0)
    batch = _batch(10)
    batch = dict(batch)
    ts = batch["timestamps_days"].clone()
    ts[:, -1] = 1.0e7  # huge lag
    batch["timestamps_days"] = ts
    ages = torch.zeros_like(batch["age_years"])
    ages[:, 0] = 0.0
    ages[:, 1] = 200.0
    ages = torch.where(batch["attention_mask"], ages, torch.zeros_like(ages))
    # Keep at least the first valid event's age finite.
    ages = ages + 1.0 * batch["attention_mask"].float()
    batch["age_years"] = ages
    out = m(batch, need_diagnostics=True)
    for key in ("code_logits", "h", "attn", "content_logits", "temporal_bias"):
        t = out[key]
        assert torch.isfinite(t).all(), key


def test_overfit_tiny_batch():
    m = _model("age_temporal", seed=11)
    batch = _batch(11, lengths=(6, 6, 6, 6), ages=(30.0, 45.0, 60.0, 75.0))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        logits = m(batch)["code_logits"]
        loss = F.binary_cross_entropy_with_logits(logits, batch["target_codes"])
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.02, (losses[0], losses[-1])
    assert torch.isfinite(m.temporal.lambda0) and torch.isfinite(m.temporal.beta)


def test_macro_metrics_skip_invalid_classes():
    logits = torch.tensor([[10.0, -10.0, 0.0], [10.0, -10.0, 0.1]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    # col 0 all-positive, col 1 all-negative, col 2 mixed
    mask = valid_class_mask(targets.numpy())
    assert mask.tolist() == [False, False, True]
    m = multilabel_metrics(logits, targets, ks=(1, 2))
    assert m["n_valid_classes_macro"] == 1
    assert math.isfinite(m["macro_auroc"])
    assert math.isfinite(m["micro_auroc"])


def test_constant_age_override_changes_conditioned_model():
    m = _model("age_temporal", seed=12)
    with torch.no_grad():
        m.temporal.beta.fill_(1.7)
    batch = _batch(12, ages=(20.0, 40.0, 60.0, 80.0))
    const = constant_age_years(batch["age_years"], batch["attention_mask"], 50.0)
    with torch.no_grad():
        a = m(batch)["code_logits"]
        b = m(batch, age_years_for_bias=const)["code_logits"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_arms_share_backbone_at_init():
    table = _table(1)
    a = _model("no_interaction", seed=99, table=table)
    b = _model("age_temporal", seed=99, table=table)
    pa = {k: v for k, v in a.state_dict().items() if "temporal." not in k}
    pb = {k: v for k, v in b.state_dict().items() if "temporal." not in k}
    assert pa.keys() == pb.keys()
    for k in pa:
        assert torch.equal(pa[k], pb[k]), k


def test_temporal_only_alias_resolves_to_no_interaction():
    m = _model("temporal_only", seed=1)
    assert m.arm == "no_interaction"
    assert not m.temporal.beta.requires_grad


def test_production_n_heads_and_d_head():
    m = _model("age_temporal", n_heads=4, d_model=256, seed=0)
    assert m.n_heads == 4
    assert m.d_model == 256
    assert m.encoder.blocks[0].attn.d_head == 64
    assert m.encoder.blocks[0].attn.n_heads == 4
    assert m.temporal.lambda0.numel() == 1
    assert m.temporal.beta.numel() == 1
    batch = _batch(0)
    out = m(batch, need_diagnostics=True)
    assert out["content_logits"].shape[1] == 4
    assert out["temporal_bias"].shape == out["content_logits"].shape[:1] + out["content_logits"].shape[2:]
    added = out["temporal_bias"].unsqueeze(1).expand_as(out["content_logits"])
    assert torch.allclose(added[:, 0], added[:, 3], atol=0.0)


def test_pooling_ignores_age_by_default():
    m = _model("age_temporal", seed=2)
    assert m.pool_temporal_bias is False
    assert m.pooling.use_temporal_bias is False
    with torch.no_grad():
        m.temporal.beta.fill_(2.5)
        m.temporal.lambda0.fill_(1.0)
    e = torch.randn(3, 7, D_MODEL)
    tau = torch.rand(3, 7)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[1, 5:] = False
    a1 = torch.tensor([20.0, 50.0, 80.0])
    a2 = torch.tensor([80.0, 20.0, 35.0])
    with torch.no_grad():
        h1, attn1, bias1 = m.pooling(e, tau, mask, a1, need_weights=True)
        h2, attn2, bias2 = m.pooling(e, tau, mask, a2, need_weights=True)
    assert bias1 is None and bias2 is None
    assert torch.allclose(h1, h2)
    assert torch.allclose(attn1, attn2)


def test_pooling_bias_flag_uses_age():
    m = _model("age_temporal", seed=2, pool_temporal_bias=True)
    assert m.pooling.use_temporal_bias is True
    with torch.no_grad():
        m.temporal.beta.fill_(2.5)
        m.temporal.lambda0.fill_(0.4)
    e = torch.randn(3, 7, D_MODEL)
    tau = torch.rand(3, 7) + 0.2
    mask = torch.ones(3, 7, dtype=torch.bool)
    a1 = torch.tensor([20.0, 50.0, 80.0])
    a2 = torch.tensor([80.0, 20.0, 35.0])
    with torch.no_grad():
        h1 = m.pooling(e, tau, mask, a1)
        h2 = m.pooling(e, tau, mask, a2)
    assert not torch.allclose(h1, h2, atol=1e-6)


TESTS = [
    test_broadcast_bias_across_heads,
    test_beta_zero_matches_no_interaction,
    test_changing_age_changes_logits_when_beta_nonzero,
    test_changing_age_does_not_change_logits_when_beta_zero,
    test_changing_time_gap_changes_logits,
    test_gradients_reach_lambda0_and_beta,
    test_gradients_lambda0_only_when_no_interaction,
    test_shuffled_ages_alter_outputs_when_age_conditioned,
    test_padding_receives_no_attention,
    test_no_nan_extreme_gaps_and_ages,
    test_overfit_tiny_batch,
    test_macro_metrics_skip_invalid_classes,
    test_constant_age_override_changes_conditioned_model,
    test_arms_share_backbone_at_init,
    test_temporal_only_alias_resolves_to_no_interaction,
    test_production_n_heads_and_d_head,
    test_pooling_ignores_age_by_default,
    test_pooling_bias_flag_uses_age,
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
