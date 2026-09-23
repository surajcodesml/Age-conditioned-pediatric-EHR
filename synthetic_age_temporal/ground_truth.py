"""Ground-truth age × temporal mechanism: softplus λ(a) and target logits."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    AGE_CENTER,
    AGE_SCALE,
    PROBE_AGES,
    RELEVANCE_LOGIT_SCALE,
    SCENARIO_SPECS,
    SURFACE_LAGS_DAYS,
    TARGET_MECHANISM_COUNTS,
    TARGET_PREVALENCE,
    TEMPORAL_ONLY_LAMBDA0,
    THETA0_DEFAULT,
    all_signal_codes,
    lambda_true,
    relevance,
    softplus_np,
    tau_from_days,
    z_age,
)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_parameters(
    theta0: float = THETA0_DEFAULT,
    beta: float = -2.0,
    ages: tuple[float, ...] = PROBE_AGES,
    lags_days: tuple[float, ...] = SURFACE_LAGS_DAYS,
) -> dict[str, Any]:
    """Report λ(a) and R(a,τ) tables used to validate non-trivial decay."""
    report: dict[str, Any] = {
        "theta0": float(theta0),
        "beta": float(beta),
        "z_formula": f"(a - {AGE_CENTER}) / {AGE_SCALE}",
        "lambda_formula": "softplus(theta0 + beta * z(a))",
        "relevance_formula": "exp(-lambda(a) * tau)",
        "lambda_by_age": {},
        "relevance_by_age_lag": {},
    }
    for a in ages:
        lam = float(lambda_true(a, theta0, beta))
        report["lambda_by_age"][str(a)] = {
            "age": a,
            "z": float(z_age(a)),
            "lambda": lam,
        }
        row = {}
        for d in lags_days:
            if d <= 0:
                continue
            t = float(tau_from_days(d))
            row[f"{int(d)}d"] = {
                "lag_days": d,
                "tau": t,
                "relevance": float(relevance(a, t, theta0, beta)),
            }
        report["relevance_by_age_lag"][str(a)] = row

    # Sanity checks used by the build pipeline.
    lam_young = float(lambda_true(1.0, theta0, beta))
    lam_old = float(lambda_true(18.0, theta0, beta))
    report["checks"] = {
        "lambda_positive": all(
            float(lambda_true(a, theta0, beta)) > 0 for a in ages
        ),
        "younger_faster_if_beta_neg": (lam_young > lam_old) if beta < 0 else None,
        "older_faster_if_beta_pos": (lam_old > lam_young) if beta > 0 else None,
        "z0": float(z_age(0.0)),
        "z9": float(z_age(9.0)),
        "z18": float(z_age(18.0)),
    }
    return report


def build_target_specs(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Create multi-label target definitions with known mechanism classes."""
    codes = all_signal_codes()
    n_codes = len(codes)
    specs: list[dict[str, Any]] = []
    idx = 0
    for mech, count in TARGET_MECHANISM_COUNTS.items():
        for _ in range(count):
            # Sparse weights over signal types; not age-dependent.
            w = rng.normal(0.0, 1.2, size=n_codes)
            # Zero out half so content is not dense noise.
            mask = rng.random(n_codes) < 0.55
            w = np.where(mask, w, 0.0)
            if not np.any(np.abs(w) > 0):
                w[rng.integers(0, n_codes)] = rng.choice([-1.0, 1.0])
            if mech in ("interaction", "temporal_only"):
                w = w * 1.5  # stronger history contribution
            gamma = float(rng.uniform(0.6, 1.4) * rng.choice([-1.0, 1.0]))
            specs.append(
                {
                    "target_id": idx,
                    "name": f"Y{idx:02d}",
                    "mechanism": mech,
                    "weights": {codes[i]: float(w[i]) for i in range(n_codes)},
                    "weight_vector": w.astype(np.float64).tolist(),
                    "gamma": gamma,  # age main-effect coefficient when used
                    "bias": 0.0,  # calibrated later
                    "null_p": float(TARGET_PREVALENCE),
                    "relevance_scale": float(RELEVANCE_LOGIT_SCALE),
                }
            )
            idx += 1
    assert idx == sum(TARGET_MECHANISM_COUNTS.values())
    return specs


def _find_intercept(lp: np.ndarray, noise: np.ndarray, target_p: float) -> float:
    lo, hi = -10.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mean_p = float(np.mean(sigmoid(mid + lp + noise)))
        if mean_p < target_p:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


@dataclass
class ExampleSignals:
    """Per-example injected signal events (already before cutoff)."""

    codes: np.ndarray  # (K,) str or object
    lag_days: np.ndarray  # (K,)
    tau: np.ndarray  # (K,)
    times: np.ndarray  # (K,) datetime64 or float days


def compute_target_logits(
    *,
    age: float,
    signals: ExampleSignals,
    specs: list[dict[str, Any]],
    scenario: str,
    theta0: float,
    beta: float,
    noise: np.ndarray,
    temporal_lambda0: float = TEMPORAL_ONLY_LAMBDA0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return logits, probs, mechanism-weighted relevance sum, and true λ(a*)."""
    spec_sc = SCENARIO_SPECS[scenario]
    z = float(z_age(age))
    lam_interact = float(lambda_true(age, theta0, beta))
    codes = list(all_signal_codes())
    code_to_i = {c: i for i, c in enumerate(codes)}

    # Event relevances under interaction λ and under fixed temporal λ.
    if signals.tau.size == 0:
        R_int = np.zeros(0, dtype=np.float64)
        R_temp = np.zeros(0, dtype=np.float64)
        code_idx = np.zeros(0, dtype=np.int64)
    else:
        R_int = np.exp(-lam_interact * signals.tau)
        R_temp = np.exp(-temporal_lambda0 * signals.tau)
        code_idx = np.array([code_to_i[c] for c in signals.codes], dtype=np.int64)

    n_t = len(specs)
    logits = np.zeros(n_t, dtype=np.float64)
    for k, sp in enumerate(specs):
        w = np.asarray(sp["weight_vector"], dtype=np.float64)
        mech = sp["mechanism"]
        b = float(sp["bias"])
        gamma = float(sp["gamma"])
        if mech == "null":
            # Fixed prevalence via calibrated bias; ignore history.
            logits[k] = b
            continue
        if mech == "age_only":
            age_coef = spec_sc.age_main_effect if spec_sc.age_main_effect != 0 else 1.0
            logits[k] = b + age_coef * gamma * z
            continue
        if signals.tau.size == 0:
            content = 0.0
            weighted_R = 0.0
        else:
            w_ev = w[code_idx]
            if mech == "content_only":
                # Occurrence only (ignore lag/age).
                present = np.zeros(len(codes), dtype=np.float64)
                for ci in code_idx:
                    present[ci] = 1.0
                content = float(np.dot(w, present))
                logits[k] = b + content
                continue
            if mech == "temporal_only" or (
                mech == "interaction" and not spec_sc.has_interaction
            ):
                scale = float(sp.get("relevance_scale", RELEVANCE_LOGIT_SCALE))
                weighted_R = float(np.dot(w_ev, R_temp)) * scale
                age_term = 0.0
                if scenario == "S1":
                    age_term = float(spec_sc.age_main_effect) * gamma * z
                logits[k] = b + age_term + weighted_R
                continue
            # interaction with age-dependent R
            scale = float(sp.get("relevance_scale", RELEVANCE_LOGIT_SCALE))
            weighted_R = float(np.dot(w_ev, R_int)) * scale
            age_term = float(spec_sc.age_main_effect) * gamma * z
            logits[k] = b + age_term + weighted_R

    logits = logits + noise
    probs = np.asarray(sigmoid(logits), dtype=np.float64)
    return logits, probs, R_int if signals.tau.size else np.zeros(0), lam_interact


def calibrate_target_biases(
    *,
    ages: np.ndarray,
    signal_list: list[ExampleSignals],
    specs: list[dict[str, Any]],
    scenario: str,
    theta0: float,
    beta: float,
    rng: np.random.Generator,
    target_p: float = TARGET_PREVALENCE,
    noise_std: float = 0.35,
) -> list[dict[str, Any]]:
    """Set per-target bias so mean prevalence ≈ target_p on the train cohort."""
    from config import NOISE_STD

    noise_std = noise_std if noise_std is not None else NOISE_STD
    n = len(ages)
    n_t = len(specs)
    # Zero biases, compute linear predictors without bias/noise, then calibrate.
    for sp in specs:
        sp["bias"] = 0.0

    lp = np.zeros((n, n_t), dtype=np.float64)
    for i in range(n):
        noise0 = np.zeros(n_t, dtype=np.float64)
        logits, _, _, _ = compute_target_logits(
            age=float(ages[i]),
            signals=signal_list[i],
            specs=specs,
            scenario=scenario,
            theta0=theta0,
            beta=beta,
            noise=noise0,
        )
        lp[i] = logits

    noise = rng.normal(0.0, noise_std, size=(n, n_t))
    out = []
    for k, sp in enumerate(specs):
        sp = dict(sp)
        if sp["mechanism"] == "null":
            # logit(p) with no noise mean
            p = float(sp["null_p"])
            sp["bias"] = float(np.log(p / (1.0 - p)))
        else:
            sp["bias"] = _find_intercept(lp[:, k], noise[:, k], target_p)
        out.append(sp)
    return out


def sample_targets(
    probs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return (rng.random(probs.shape) < probs).astype(np.float32)


def oracle_predict(
    *,
    ages: np.ndarray,
    signal_list: list[ExampleSignals],
    specs: list[dict[str, Any]],
    scenario: str,
    theta0: float,
    beta: float,
    mode: str = "correct",
    shuffle_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Oracle probabilities under correct / shuffled-age / shuffled-lag / no-interaction."""
    n = len(ages)
    n_t = len(specs)
    out = np.zeros((n, n_t), dtype=np.float64)
    ages_use = ages.copy()
    if mode == "shuffle_age":
        assert shuffle_rng is not None
        ages_use = shuffle_rng.permutation(ages_use)
    # β=0 keeps softplus(θ0) age-independent decay (true interaction ablation).
    beta_use = 0.0 if mode == "no_interaction" else beta

    for i in range(n):
        sig = signal_list[i]
        if mode == "shuffle_lag" and sig.tau.size > 0:
            assert shuffle_rng is not None
            order = shuffle_rng.permutation(sig.tau.size)
            sig = ExampleSignals(
                codes=sig.codes,
                lag_days=sig.lag_days[order],
                tau=sig.tau[order],
                times=sig.times,
            )
        # Remove age from age_only / age main terms by setting z=0 via age=9.
        age_i = 9.0 if mode == "remove_age" else float(ages_use[i])
        _, probs, _, _ = compute_target_logits(
            age=age_i,
            signals=sig,
            specs=specs,
            scenario=scenario,
            theta0=theta0,
            beta=beta_use,
            noise=np.zeros(n_t),
        )
        out[i] = probs
    return out


def save_calibration(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    for name, beta in [("S0", 0.0), ("S2", -2.0), ("S3", 2.0)]:
        r = calibrate_parameters(beta=beta)
        print(name, json.dumps(r["lambda_by_age"], indent=2))
        print("checks", r["checks"])
