"""Contract tests for the atomic DTR follow-up. No benchmark fitting."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAT))
sys.path.insert(0, str(ROOT))

import atomic  # noqa: E402
from baselines.common.training import set_seed  # noqa: E402
from atomic.models import build_atomic  # noqa: E402
from ladder.models.common import inverse_softplus_value  # noqa: E402

CFG = {
    "variant": "current",
    "d_model": 8,
    "dropout": 0.0,
    "lambda_init": 0.1,
    "n_components": 3,
}


def _batch(n_codes: int = 12, n_targets: int = 4, encounters: int = 3, extra_pad: bool = False):
    m = encounters + (1 if extra_pad else 0)
    codes = torch.randint(1, n_codes, (2, m, 4))
    mask = torch.ones(2, m, 4, dtype=torch.bool)
    tau = torch.rand(2, m)
    pad = torch.zeros(2, m, dtype=torch.bool)
    if extra_pad:
        pad[:, -1] = True
        tau[:, -1] = 40
        codes[:, -1] = 0
        mask[:, -1] = False
    age = torch.tensor([4.0, 15.0])
    labels = torch.zeros(2, n_targets)
    return {
        "enc_code_ids": codes,
        "enc_code_mask": mask,
        "enc_tau": tau,
        "enc_padding_mask": pad,
        "age": age,
        "labels": labels,
    }


def _cfg(variant: str) -> dict:
    cfg = dict(CFG)
    cfg["variant"] = variant
    return cfg


def _model(variant: str, *, age_temporal: bool = True, seed: int = 0):
    set_seed(seed)
    return build_atomic(_cfg(variant), 12, 4, age_temporal=age_temporal)


def _forward(model, batch, **kwargs):
    return model(
        batch["enc_code_ids"], batch["enc_code_mask"], batch["enc_tau"],
        batch["enc_padding_mask"], batch["age"], return_parts=True, **kwargs,
    )


def test_gate_age_matches_true_age_when_equal():
    model = _model("current")
    batch = _batch()
    same = _forward(model, batch, gate_age=batch["age"])
    default = _forward(model, batch)
    assert torch.allclose(same["logits"], default["logits"])


def test_gate_shuffle_spares_age_head():
    model = _model("current")
    with torch.no_grad():
        model.base.beta.fill_(1.5)
    batch = _batch()
    gate_age = torch.tensor([1.0, 18.0])
    original = _forward(model, batch)
    shuffled = _forward(model, batch, gate_age=gate_age)
    assert torch.allclose(original["age_logit"], shuffled["age_logit"])
    assert not torch.allclose(original["g"], shuffled["g"])


def test_temporal_only_ignores_gate_age():
    model = _model("current", age_temporal=False)
    batch = _batch()
    original = _forward(model, batch)
    shuffled = _forward(model, batch, gate_age=torch.tensor([0.0, 18.0]))
    assert torch.allclose(original["logits"], shuffled["logits"])


def test_c04_and_c05_match_c00_at_init():
    batch = _batch()
    current = _model("current", seed=3)
    removed = _model("no_persistence", seed=3)
    mixture = _model("shared_beta_mixture", seed=3)
    assert torch.allclose(_forward(current, batch)["logits"], _forward(removed, batch)["logits"], atol=1e-6)
    assert torch.allclose(_forward(current, batch)["logits"], _forward(mixture, batch)["logits"], atol=1e-6)


def test_c05_and_c06_match_when_beta_is_zero():
    batch = _batch()
    shared = _model("shared_beta_mixture", seed=4)
    component = _model("component_beta_mixture", seed=4)
    with torch.no_grad():
        shared.beta.zero_()
        component.beta_k.zero_()
    assert torch.allclose(_forward(shared, batch)["logits"], _forward(component, batch)["logits"], atol=1e-6)


def test_weak_init_effective_lambda_is_0_1():
    model = _model("weak_init")
    batch = _batch()
    assert torch.allclose(F.softplus(model.base.theta0), torch.tensor([0.1]), atol=1e-6)
    assert float(model.base.persistence_projection.bias.detach().abs().sum()) == 0.0
    parts = _forward(model, batch)
    valid = ~batch["enc_padding_mask"]
    assert torch.allclose(parts["lambda"][valid], torch.full((int(valid.sum()),), 0.1), atol=1e-5)


def test_mass_copies_original_readout_columns():
    set_seed(1)
    current = build_atomic(_cfg("current"), 12, 4, age_temporal=True)
    set_seed(1)
    mass = build_atomic(_cfg("mass"), 12, 4, age_temporal=True)
    old = current.base.history_head[0].weight
    new = mass.base.history_head[0].weight
    assert new.shape[1] == old.shape[1] + 1
    assert torch.allclose(new[:, : old.shape[1]], old)
    assert torch.count_nonzero(new[:, -1]) == 0
    assert torch.allclose(mass.base.history_head[2].weight, current.base.history_head[2].weight)


def test_mixture_pi_sums_to_one_and_ignores_age():
    model = _model("shared_beta_mixture")
    batch = _batch()
    first = _forward(model, batch)
    other = _forward(model, batch, gate_age=torch.tensor([0.0, 18.0]))
    hist = ~batch["enc_padding_mask"]
    totals = first["pi"].sum(dim=-1)[hist]
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    assert torch.allclose(first["pi"], other["pi"])


def test_matched_arms_share_init_and_beta_grad():
    set_seed(2)
    age = build_atomic(_cfg("current"), 12, 4, age_temporal=True)
    set_seed(2)
    temporal = build_atomic(_cfg("current"), 12, 4, age_temporal=False)
    for left, right in zip(age.state_dict().values(), temporal.state_dict().values()):
        assert torch.equal(left, right)
    assert age.base.beta.requires_grad
    assert not temporal.base.beta.requires_grad
    batch = _batch()
    age.base.history_head[2].weight.data.fill_(0.2)
    loss = _forward(age, batch)["logits"].sum()
    loss.backward()
    assert age.base.beta.grad is not None and float(age.base.beta.grad.abs().sum()) > 0
    loss_t = _forward(temporal, batch)["logits"].sum()
    loss_t.backward()
    assert temporal.base.beta.grad is None


def test_padding_does_not_change_logits():
    model = _model("component_beta_mixture", seed=5)
    with torch.no_grad():
        model.beta_k.copy_(torch.tensor([0.2, -0.4, 0.7]))
    short = _batch(extra_pad=False)
    long = _batch(extra_pad=False)
    # Rebuild the longer batch from the same short values plus a pad column.
    m = short["enc_tau"].shape[1]
    pad_batch = {
        "enc_code_ids": torch.nn.functional.pad(short["enc_code_ids"], (0, 0, 0, 1)),
        "enc_code_mask": torch.nn.functional.pad(short["enc_code_mask"], (0, 0, 0, 1)),
        "enc_tau": torch.nn.functional.pad(short["enc_tau"], (0, 1), value=40),
        "enc_padding_mask": torch.nn.functional.pad(short["enc_padding_mask"], (0, 1), value=True),
        "age": short["age"],
    }
    left = _forward(model, short)["logits"]
    right = _forward(model, pad_batch)["logits"]
    assert torch.allclose(left, right, atol=1e-5)
    del m, long


def test_saved_predictions_reproduce_metrics(tmp_path=Path("/tmp/atomic_metric_contract")):
    from atomic.io import read_predictions, write_predictions
    from atomic.metrics import compute_run_metrics
    from evaluate import classification_metrics

    tmp_path.mkdir(parents=True, exist_ok=True)
    pred = tmp_path / "predictions.parquet"
    if pred.exists():
        pred.unlink()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=(6, 3)).astype(np.float64)
    logits = rng.normal(size=(6, 3))
    write_predictions(
        pred,
        example_id=np.arange(6),
        patient_id=[f"p{i//2}" for i in range(6)],
        age=np.linspace(1, 16, 6),
        labels=labels,
        logits=logits,
        logits_beta0=logits + 0.1,
        logits_full_age_shuffle=logits + 0.2,
        logits_gate_age_shuffle=logits + 0.05,
    )
    ages = np.array([0.0, 9.0])
    lags = np.array([0.0, 7.0])
    surface = rng.random((2, 2, 3))
    np.savez_compressed(
        tmp_path / "mechanism_outputs.npz",
        architecture=np.asarray("current"),
        surface_ages=ages,
        surface_lags=lags,
        cf_ages=ages,
        cf_lags=lags,
        surface_model=surface,
        surface_oracle=surface + 0.1,
        cf_age_model=rng.random((2, 3)),
        cf_age_oracle=rng.random((2, 3)),
        cf_lag_model=rng.random((2, 3)),
        cf_lag_oracle=rng.random((2, 3)),
        beta=np.array([-0.4]),
        theta=np.array([0.0]),
        beta_true=np.array(-2.5),
        theta0_true=np.array(0.0),
        has_global_gate=np.array(0),
        lambda_true_age=np.array([0.2, 0.4]),
        gate_signal_hat=np.array([0.5, 0.25, 0.8]),
        gate_signal_true=np.array([0.4, 0.2, 0.7]),
        gate_signal_code=np.array(["SYN_SIGNAL_A", "SYN_SIGNAL_A", "SYN_SIGNAL_B"]),
    )
    result = compute_run_metrics(tmp_path, n_boot=4)
    loaded = read_predictions(pred)
    direct = classification_metrics(loaded["labels"], loaded["logits"])
    assert abs(result["metrics"]["bce"] - direct["bce"]) < 1e-12
    assert result["metrics"]["delta_bce_gate_age_shuffle"] != result["metrics"]["delta_bce_full_age_shuffle"]
    assert result["mechanism"]["gate_signal_rmse"] > 0
    assert result["mechanism"]["gate_surface_rmse"] is None


def test_inverse_softplus_roundtrip():
    value = inverse_softplus_value(0.1)
    assert abs(float(F.softplus(torch.tensor(value))) - 0.1) < 1e-6


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
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
    print(f"All {len(tests)} atomic follow-up tests passed.")
