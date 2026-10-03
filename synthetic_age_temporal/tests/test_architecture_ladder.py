"""Contract tests for the architecture ladder. No benchmark fitting."""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAT))
sys.path.insert(0, str(ROOT))

import ladder  # noqa: E402
from baselines.common.counterfactual import (  # noqa: E402
    CF_AGES,
    CF_LAGS_DAYS,
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
)
from evaluate import classification_metrics  # noqa: E402
from ladder.artifacts import read_predictions, write_predictions  # noqa: E402
from ladder.evaluation.metrics import compute_run_metrics  # noqa: E402
from ladder.models.channels import ContentChannelDTR  # noqa: E402
from ladder.models.common import inverse_softplus_value  # noqa: E402
from ladder.models.direct import DirectEvidenceDTR  # noqa: E402
from ladder.models.factory import build_model  # noqa: E402
from ladder.models.integrated import integrate_piecewise_linear  # noqa: E402
from ladder.training.loop import fork_matched_arms, max_state_diff  # noqa: E402


def _cfg(architecture: str, **extra):
    cfg = {
        "architecture": architecture,
        "d_model": 8,
        "dropout": 0.0,
        "lambda_init": 0.1,
        "aggregation": "raw_additive",
        "mass_eps": 1e-6,
        "n_channels": 4,
        "n_components": 3,
        "knots": [0.0, 6.0, 12.0, 18.0],
    }
    cfg.update(extra)
    return cfg


def _batch(B=2, M=3, K=2, n_codes=20, n_targets=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(1, n_codes, (B, M, K), generator=g)
    code_mask = torch.ones(B, M, K, dtype=torch.bool)
    tau = torch.rand(B, M, generator=g) + 0.2
    pad = torch.zeros(B, M, dtype=torch.bool)
    age = torch.tensor([1.0, 17.0][:B] if B <= 2 else [float(i % 18) for i in range(B)])
    lag = torch.expm1(tau) * 7.0
    labels = torch.zeros(B, n_targets)
    labels[:, 0] = 1.0
    return {
        "enc_code_ids": codes,
        "enc_code_mask": code_mask,
        "enc_tau": tau,
        "enc_padding_mask": pad,
        "enc_lag_days": lag,
        "age": age,
        "labels": labels,
    }


def _logits(model, batch):
    return model(
        enc_code_ids=batch["enc_code_ids"],
        enc_code_mask=batch["enc_code_mask"],
        enc_tau=batch["enc_tau"],
        enc_padding_mask=batch["enc_padding_mask"],
        age=batch["age"],
        enc_lag_days=batch["enc_lag_days"],
    )


def _pad_one(batch):
    out = {}
    B = batch["age"].shape[0]
    K = batch["enc_code_ids"].shape[-1]
    out["enc_code_ids"] = torch.cat([batch["enc_code_ids"], torch.zeros(B, 1, K, dtype=torch.long)], dim=1)
    out["enc_code_mask"] = torch.cat([batch["enc_code_mask"], torch.zeros(B, 1, K, dtype=torch.bool)], dim=1)
    out["enc_tau"] = torch.cat([batch["enc_tau"], torch.full((B, 1), 40.0)], dim=1)
    out["enc_padding_mask"] = torch.cat([batch["enc_padding_mask"], torch.ones(B, 1, dtype=torch.bool)], dim=1)
    out["enc_lag_days"] = torch.cat([batch["enc_lag_days"], torch.full((B, 1), 9999.0)], dim=1)
    out["age"] = batch["age"].clone()
    out["labels"] = batch["labels"].clone()
    return out


def test_inverse_softplus_and_weak_init():
    theta = inverse_softplus_value(0.1)
    recovered = float(F.softplus(torch.tensor(theta)))
    assert abs(recovered - 0.1) < 1e-6
    model = DirectEvidenceDTR(n_codes=20, n_targets=2, d_model=8, age_temporal=True, lambda_init=0.1)
    age = torch.tensor([9.0])
    lam = float(model.developmental_lambda(age))
    assert abs(lam - 0.1) < 1e-5
    tau = math.log1p(730.0 / 7.0)
    gate = math.exp(-0.1 * tau)
    assert gate > 0.5


def test_matched_arms_identical_at_beta0():
    batch = _batch()
    for architecture in ("direct", "channels", "mixture", "integrated_hazard"):
        torch.manual_seed(0)
        left = build_model(_cfg(architecture), 20, 4, age_temporal=True)
        torch.manual_seed(0)
        right = build_model(_cfg(architecture), 20, 4, age_temporal=False)
        with torch.no_grad():
            left.beta_param().zero_()
            right.beta_param().zero_()
        a = _logits(left, batch)
        b = _logits(right, batch)
        assert torch.allclose(a, b, atol=1e-5), architecture


def test_temporal_only_beta_has_no_gradient_and_age_temporal_does():
    batch = _batch()
    for architecture in ("direct", "channels", "mixture", "integrated_hazard"):
        frozen = build_model(_cfg(architecture), 20, 4, age_temporal=False)
        assert frozen.beta_param().requires_grad is False
        loss = _logits(frozen, batch).sum()
        loss.backward()
        assert frozen.beta_param().grad is None

        live = build_model(_cfg(architecture), 20, 4, age_temporal=True)
        with torch.no_grad():
            live.W_history.weight.fill_(0.2)
        loss = F.binary_cross_entropy_with_logits(_logits(live, batch), batch["labels"])
        loss.backward()
        grad = live.beta_param().grad
        assert grad is not None, architecture
        assert float(grad.abs().sum()) > 0, architecture


def test_padding_does_not_change_predictions():
    batch = _batch()
    padded = _pad_one(batch)
    for architecture in ("direct", "channels", "mixture", "integrated_hazard"):
        torch.manual_seed(1)
        model = build_model(_cfg(architecture), 20, 4, age_temporal=True)
        model.eval()
        with torch.no_grad():
            a = _logits(model, batch)
            b = _logits(model, padded)
        assert torch.allclose(a, b, atol=1e-5), architecture


def test_e01_has_no_content_query_or_persistence_or_mlp():
    model = DirectEvidenceDTR(n_codes=15, n_targets=3, d_model=8, aggregation="raw_additive")
    keys = " ".join(model.state_dict())
    for banned in ("content_query", "content_key", "persistence"):
        assert banned not in keys
    assert isinstance(model.W_history, torch.nn.Linear)
    assert model.W_history.in_features == 8
    mass = DirectEvidenceDTR(n_codes=15, n_targets=3, d_model=8, aggregation="mass")
    assert mass.W_history.in_features == 9


def test_e02_fork_is_identical_before_beta_training():
    cfg = _cfg("direct")
    torch.manual_seed(3)
    base = build_model(cfg, 20, 4, age_temporal=False)
    with torch.no_grad():
        base.W_history.weight.add_(0.3)
        base.theta_param().add_(0.2)
    at, to = fork_matched_arms(base.state_dict(), cfg, 20, 4)
    assert max_state_diff(at, to) == 0.0
    assert at.beta_param().requires_grad is True
    assert to.beta_param().requires_grad is False
    batch = _batch()
    with torch.no_grad():
        gap = (_logits(at, batch) - _logits(to, batch)).abs().max().item()
    assert gap < 1e-6


def test_e04_channels_cannot_see_age_or_lag():
    params = inspect.signature(ContentChannelDTR.channel_scores).parameters
    assert "age" not in params
    assert "tau" not in params
    assert "enc_lag_days" not in params
    model = ContentChannelDTR(n_codes=20, n_targets=3, d_model=8, n_channels=4, age_temporal=True)
    model.eval()
    batch = _batch()
    other = _batch()
    other["enc_code_ids"] = batch["enc_code_ids"].clone()
    other["enc_code_mask"] = batch["enc_code_mask"].clone()
    other["enc_padding_mask"] = batch["enc_padding_mask"].clone()
    other["age"] = torch.tensor([0.0, 18.0])
    other["enc_tau"] = batch["enc_tau"] + 3
    other["enc_lag_days"] = batch["enc_lag_days"] + 100
    with torch.no_grad():
        first = model(
            batch["enc_code_ids"], batch["enc_code_mask"], batch["enc_tau"],
            batch["enc_padding_mask"], batch["age"], batch["enc_lag_days"], return_parts=True,
        )
        second = model(
            other["enc_code_ids"], other["enc_code_mask"], other["enc_tau"],
            other["enc_padding_mask"], other["age"], other["enc_lag_days"], return_parts=True,
        )
    assert torch.allclose(first["channel_scores"], second["channel_scores"], atol=1e-6)


def test_e05_mixture_weights_sum_to_one():
    model = build_model(_cfg("mixture"), 20, 4, age_temporal=True)
    batch = _batch()
    v = model.encounter_encoder(batch["enc_code_ids"], batch["enc_code_mask"])
    pi = model.mixture_weights(v)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(pi.shape[:-1]), atol=1e-5)
    parts = model(
        batch["enc_code_ids"], batch["enc_code_mask"], batch["enc_tau"],
        batch["enc_padding_mask"], batch["age"], batch["enc_lag_days"], return_parts=True,
    )
    valid = ~batch["enc_padding_mask"]
    assert torch.allclose(parts["pi"][valid].sum(dim=-1), torch.ones(int(valid.sum())), atol=1e-5)


def test_e06_integration_matches_closed_form_and_quadrature():
    knots = torch.tensor([0.0, 10.0])
    rho = torch.tensor([0.2, 0.2])
    exact = integrate_piecewise_linear(rho, knots, torch.tensor(-2.0), torch.tensor(12.0))
    assert abs(float(exact) - 0.2 * 14.0) < 1e-5
    rho = torch.tensor([0.0, 10.0], requires_grad=True)
    # Linear from 0 at age 0 to 10 at age 10. Integral over [2, 4] is 6.
    mid = integrate_piecewise_linear(rho, knots, torch.tensor(2.0), torch.tensor(4.0))
    assert abs(float(mid) - 6.0) < 1e-4
    mid.sum().backward()
    assert rho.grad is not None and float(rho.grad.abs().sum()) > 0

    xs = np.linspace(-1.0, 20.0, 20001)
    knot_np = np.asarray([0.0, 6.0, 12.0, 18.0])
    rho_np = np.asarray([0.1, 0.4, 0.2, 0.8])
    ys = np.interp(xs, knot_np, rho_np)
    reference = float(np.trapz(ys, xs))
    got = float(integrate_piecewise_linear(
        torch.tensor(rho_np), torch.tensor(knot_np), torch.tensor(-1.0), torch.tensor(20.0),
    ))
    assert abs(got - reference) < 1e-3

    model = build_model(_cfg("integrated_hazard"), 20, 4, age_temporal=False)
    age_event = torch.tensor([[2.0]])
    age_current = torch.tensor([[11.0]])
    integral = model.integrate(age_event, age_current)
    rho0 = float(model.rho_knots()[0])
    assert abs(float(integral) - rho0 * 9.0) < 1e-4


def test_saved_predictions_reproduce_metrics(tmp_path=None):
    directory = Path(tmp_path) if tmp_path is not None else Path("/tmp/ladder_metric_contract")
    if directory.exists():
        for child in directory.iterdir():
            child.unlink()
    else:
        directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n, t = 12, 3
    labels = rng.integers(0, 2, size=(n, t)).astype(np.float64)
    labels[0, :] = 0
    labels[1, :] = 1
    logits = rng.normal(size=(n, t))
    write_predictions(
        directory / "predictions.parquet",
        example_id=np.arange(n),
        patient_id=[f"p{i // 2}" for i in range(n)],
        age=rng.uniform(0, 18, size=n),
        labels=labels,
        logits=logits,
        logits_beta0=logits + 0.4,
        logits_age_shuffle=logits - 0.2,
    )
    loaded = read_predictions(directory / "predictions.parquet")
    direct = classification_metrics(loaded["labels"], loaded["logits"])
    ages = np.asarray(SURFACE_AGES, dtype=np.float64)
    lags = np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64)
    cf_ages = np.asarray(CF_AGES, dtype=np.float64)
    cf_lags = np.asarray(CF_LAGS_DAYS, dtype=np.float64)
    surface_model = rng.random((len(ages), len(lags), t))
    surface_oracle = rng.random((len(ages), len(lags), t))
    np.savez_compressed(
        directory / "mechanism_outputs.npz",
        beta=np.asarray([-1.2]),
        theta=np.asarray([0.3]),
        surface_ages=ages,
        surface_lags=lags,
        cf_ages=cf_ages,
        cf_lags=cf_lags,
        lambda_age=np.linspace(0.2, 1.0, len(ages)),
        lambda_true_age=np.linspace(0.1, 1.1, len(ages)),
        surface_model=surface_model,
        surface_oracle=surface_oracle,
        cf_age_model=rng.random((len(cf_ages), t)),
        cf_age_oracle=rng.random((len(cf_ages), t)),
        cf_lag_model=rng.random((len(cf_lags), t)),
        cf_lag_oracle=rng.random((len(cf_lags), t)),
        beta_true=np.asarray(-2.5),
        theta0_true=np.asarray(0.0),
        architecture=np.asarray("direct"),
    )
    result = compute_run_metrics(directory, n_boot=8)
    assert abs(result["metrics"]["bce"] - direct["bce"]) < 1e-12
    again = classification_metrics(
        read_predictions(directory / "predictions.parquet")["labels"],
        read_predictions(directory / "predictions.parquet")["logits"],
    )
    assert abs(again["bce"] - direct["bce"]) < 1e-12
    assert result["mechanism"]["surface_rmse"] > 0


def test_cached_loader_matches_dynamic_batches():
    """Split-max padding and the default shuffle reproduce the slow loader."""
    from baselines.common.training import set_seed
    from baselines.synthetic.data_adapter import make_dtr_baseline_loaders
    from ladder.cached_data import cached_dtr_loaders

    slow_train, _, _, _, info = make_dtr_baseline_loaders(
        "S2", data_seed=20260922, batch_size=32
    )
    fast_train, _, _, _, _ = cached_dtr_loaders(
        "S2", data_seed=20260922, batch_size=32
    )
    set_seed(0)
    slow_ids = torch.cat([batch["example_ids"] for batch in slow_train])
    set_seed(0)
    fast_ids = torch.cat([batch["example_ids"] for batch in fast_train])
    assert torch.equal(slow_ids, fast_ids)

    set_seed(1)
    slow_batch = next(iter(slow_train))
    set_seed(1)
    fast_batch = next(iter(fast_train))
    assert torch.equal(slow_batch["labels"], fast_batch["labels"])
    assert torch.equal(slow_batch["example_ids"], fast_batch["example_ids"])
    width = slow_batch["enc_code_ids"].shape[1]
    codes = slow_batch["enc_code_ids"].shape[2]
    assert torch.equal(slow_batch["enc_code_ids"], fast_batch["enc_code_ids"][:, :width, :codes])
    assert torch.equal(slow_batch["enc_tau"], fast_batch["enc_tau"][:, :width])
    assert torch.equal(slow_batch["enc_lag_days"], fast_batch["enc_lag_days"][:, :width])
    assert torch.equal(slow_batch["enc_padding_mask"], fast_batch["enc_padding_mask"][:, :width])
    if fast_batch["enc_padding_mask"].shape[1] > width:
        assert bool(fast_batch["enc_padding_mask"][:, width:].all())
    model = DirectEvidenceDTR(
        info["n_codes"], info["n_targets"], d_model=64, dropout=0.0, age_temporal=True
    )
    model.eval()
    with torch.no_grad():
        slow_logits = model(
            slow_batch["enc_code_ids"],
            slow_batch["enc_code_mask"],
            slow_batch["enc_tau"],
            slow_batch["enc_padding_mask"],
            slow_batch["age"],
            slow_batch["enc_lag_days"],
        )
        fast_logits = model(
            fast_batch["enc_code_ids"],
            fast_batch["enc_code_mask"],
            fast_batch["enc_tau"],
            fast_batch["enc_padding_mask"],
            fast_batch["age"],
            fast_batch["enc_lag_days"],
        )
    assert torch.allclose(slow_logits, fast_logits, atol=1e-6)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        raise SystemExit(1)
    print(f"All {len(tests)} architecture-ladder tests passed.")
