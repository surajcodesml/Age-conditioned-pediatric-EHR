"""Contract tests for D00/D01/D02. No benchmark fitting."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAT))
sys.path.insert(0, str(ROOT))

from baselines.common.training import set_seed  # noqa: E402
from high_impact.models import build_high_impact  # noqa: E402


def _cfg(variant: str):
    return {
        "variant": variant,
        "d_model": 64,
        "dropout": 0.0,
        "n_heads": 4,
        "d_head": 16,
    }


def _batch():
    torch.manual_seed(0)
    return {
        "enc_code_ids": torch.randint(1, 40, (3, 4, 5)),
        "enc_code_mask": torch.ones(3, 4, 5, dtype=torch.bool),
        "enc_tau": torch.rand(3, 4) + 0.2,
        "enc_padding_mask": torch.zeros(3, 4, dtype=torch.bool),
        "age": torch.tensor([3.0, 9.0, 15.0]),
        "labels": torch.rand(3, 8),
    }


def _forward(model, batch, **kwargs):
    return model(
        batch["enc_code_ids"], batch["enc_code_mask"], batch["enc_tau"],
        batch["enc_padding_mask"], batch["age"], return_parts=True, **kwargs,
    )


def test_d00_uses_oracle_and_freezes_temporal():
    set_seed(0)
    model = build_high_impact(_cfg("oracle_gate"), 40, 8, age_temporal=True, oracle_theta0=0.0, oracle_beta=-2.5)
    model.set_oracle_(0.0, -2.5)
    assert model.temporal_parameters() == []
    assert not model.base.theta0.requires_grad
    assert not model.base.beta.requires_grad
    batch = _batch()
    parts = _forward(model, batch)
    expected = F.softplus(torch.tensor(0.0) + (-2.5) * model.base.z_of(batch["age"]))
    assert torch.allclose(parts["lambda"], expected, atol=1e-5)
    # Age head unchanged under gate shuffle.
    shuffled = _forward(model, batch, gate_age=torch.tensor([1.0, 18.0, 5.0]))
    assert torch.allclose(parts["age_logit"], shuffled["age_logit"])
    assert not torch.allclose(parts["g"], shuffled["g"])


def test_d01_width_and_shared_gate():
    set_seed(1)
    model = build_high_impact(_cfg("multihead_shared"), 40, 8, age_temporal=True)
    batch = _batch()
    parts = _forward(model, batch)
    assert parts["h_heads"].shape == (3, 4, 16)
    assert parts["u"].shape == (3, 4, 4)
    assert parts["g_heads"].shape[-1] == 4
    # Shared gate: all heads equal
    assert torch.allclose(parts["g_heads"][..., 0], parts["g_heads"][..., 1], atol=1e-6)
    assert model.content_queries.shape == (4, 16)


def test_d02_identical_to_d01_at_init():
    batch = _batch()
    set_seed(2)
    d01 = build_high_impact(_cfg("multihead_shared"), 40, 8, age_temporal=True)
    set_seed(2)
    d02 = build_high_impact(_cfg("multihead_dev"), 40, 8, age_temporal=True)
    # Align multihead weights after separate beta params created.
    d02.key_proj.load_state_dict(d01.key_proj.state_dict())
    d02.value_proj.load_state_dict(d01.value_proj.state_dict())
    with torch.no_grad():
        d02.content_queries.copy_(d01.content_queries)
        d02.base.load_state_dict(d01.base.state_dict())
        d02.beta_global.zero_()
        d02.delta.zero_()
        d01.base.beta.zero_()
    left = _forward(d01, batch)["logits"]
    right = _forward(d02, batch)["logits"]
    assert torch.allclose(left, right, atol=1e-5)
    betas = d02.head_betas()
    assert torch.allclose(betas, torch.zeros_like(betas))


def test_d02_centered_delta_and_independent_head_gates():
    set_seed(3)
    model = build_high_impact(_cfg("multihead_dev"), 40, 8, age_temporal=True)
    with torch.no_grad():
        model.beta_global.fill_(0.5)
        model.delta.copy_(torch.tensor([1.0, -0.5, 0.0, -0.5]))
    betas = model.head_betas()
    assert abs(float(betas.sum() - 4 * model.beta_global)) < 1e-5
    assert abs(float((betas - model.beta_global).sum())) < 1e-5
    batch = _batch()
    parts = _forward(model, batch)
    assert not torch.allclose(parts["g_heads"][..., 0], parts["g_heads"][..., 1])


def test_head_ablation_changes_logits():
    set_seed(4)
    model = build_high_impact(_cfg("multihead_shared"), 40, 8, age_temporal=True)
    batch = _batch()
    base = _forward(model, batch)["logits"]
    ablated = _forward(model, batch, ablate_head=0)["logits"]
    assert not torch.allclose(base, ablated)


def test_pi_free_age_in_content_projections():
    set_seed(5)
    model = build_high_impact(_cfg("multihead_shared"), 40, 8, age_temporal=True)
    batch = _batch()
    a = _forward(model, batch)["u"]
    b = _forward(model, batch, gate_age=torch.tensor([0.0, 18.0, 9.0]))["u"]
    assert torch.allclose(a, b)


if __name__ == "__main__":
    tests = [v for n, v in list(globals().items()) if n.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    if failed:
        raise SystemExit(1)
    print(f"All {len(tests)} high-impact tests passed.")
