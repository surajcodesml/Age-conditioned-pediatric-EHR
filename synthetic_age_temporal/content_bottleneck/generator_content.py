"""Canonical generator w_kj and oracle content evidence (read from target_specs)."""
from __future__ import annotations

from typing import Any

import numpy as np

from config import (
    RELEVANCE_LOGIT_SCALE,
    SCENARIO_SPECS,
    TEMPORAL_ONLY_LAMBDA0,
    all_signal_codes,
    lambda_true,
    z_age,
)


def load_weight_matrix(specs: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return W[target, signal], signal names, mechanism per target."""
    signals = list(all_signal_codes())
    W = np.asarray([sp["weight_vector"] for sp in specs], dtype=np.float64)
    mechs = [str(sp["mechanism"]) for sp in specs]
    assert W.shape == (len(specs), len(signals))
    return W, signals, mechs


def signal_id_map(vocab) -> dict[int, int]:
    """Map vocab token id -> signal index 0..11. Non-signals omitted."""
    codes = list(all_signal_codes())
    code_to_j = {c: i for i, c in enumerate(codes)}
    out = {}
    for idx, token in vocab.itos.items():
        j = code_to_j.get(str(token))
        if j is not None:
            out[int(idx)] = j
    return out


def encounter_signal_membership(
    code_ids: np.ndarray,
    code_mask: np.ndarray,
    signal_map: dict[int, int],
    n_signals: int = 12,
) -> np.ndarray:
    """Multi-hot SYN_SIGNAL membership for one encounter. code_ids [K]."""
    mem = np.zeros(n_signals, dtype=np.float64)
    for k, cid in enumerate(code_ids):
        if not bool(code_mask[k]):
            continue
        j = signal_map.get(int(cid))
        if j is not None:
            mem[j] = 1.0
    return mem


def n_background_codes(
    code_ids: np.ndarray,
    code_mask: np.ndarray,
    signal_map: dict[int, int],
) -> int:
    n = 0
    for k, cid in enumerate(code_ids):
        if not bool(code_mask[k]):
            continue
        if int(cid) == 0:
            continue
        if int(cid) not in signal_map:
            n += 1
    return n


def oracle_content_pre_decay(
    membership: np.ndarray,
    specs: list[dict[str, Any]],
    W: np.ndarray,
) -> np.ndarray:
    """Target vector of content evidence BEFORE temporal decay for one encounter.

    interaction / temporal_only: scale * W @ membership
    content_only: W @ membership  (occurrence; scale not applied in generator)
    null / age_only: zeros
    """
    n_t = len(specs)
    out = np.zeros(n_t, dtype=np.float64)
    if not np.any(membership > 0):
        return out
    for k, sp in enumerate(specs):
        mech = sp["mechanism"]
        if mech in ("null", "age_only"):
            continue
        w_dot = float(np.dot(W[k], membership))
        if mech == "content_only":
            out[k] = w_dot
        else:
            scale = float(sp.get("relevance_scale", RELEVANCE_LOGIT_SCALE))
            out[k] = scale * w_dot
    return out


def oracle_age_bias_terms(
    age: float,
    specs: list[dict[str, Any]],
    scenario: str,
) -> np.ndarray:
    """Exact generator age-main-effect + bias (no history, no noise)."""
    spec_sc = SCENARIO_SPECS[scenario]
    z = float(z_age(age))
    out = np.zeros(len(specs), dtype=np.float64)
    for k, sp in enumerate(specs):
        b = float(sp["bias"])
        gamma = float(sp["gamma"])
        mech = sp["mechanism"]
        if mech == "null":
            out[k] = b
        elif mech == "age_only":
            age_coef = spec_sc.age_main_effect if spec_sc.age_main_effect != 0 else 1.0
            out[k] = b + age_coef * gamma * z
        elif mech == "temporal_only" and scenario == "S1":
            out[k] = b + float(spec_sc.age_main_effect) * gamma * z
        elif mech == "interaction":
            out[k] = b + float(spec_sc.age_main_effect) * gamma * z
        else:
            out[k] = b
    return out


def oracle_gate_for_target(
    age: float,
    tau: float,
    *,
    scenario: str,
    mechanism: str,
    theta0: float,
    beta_true: float,
) -> float:
    """Generator temporal gate for one (target mechanism, age, tau)."""
    spec_sc = SCENARIO_SPECS[scenario]
    if mechanism == "content_only":
        return 1.0
    if mechanism in ("null", "age_only"):
        return 0.0
    if mechanism == "temporal_only" or (mechanism == "interaction" and not spec_sc.has_interaction):
        return float(np.exp(-TEMPORAL_ONLY_LAMBDA0 * tau))
    return float(np.exp(-lambda_true(age, theta0, beta_true) * tau))


def build_oracle_logits_with_gates(
    *,
    ages: np.ndarray,
    enc_tau: np.ndarray,
    enc_pad: np.ndarray,
    enc_membership: np.ndarray,
    gates: np.ndarray,
    specs: list[dict[str, Any]],
    W: np.ndarray,
    scenario: str,
    use_model_gates: bool,
    theta0: float,
    beta_true: float,
) -> np.ndarray:
    """logits [B, T] from oracle content + chosen gates + generator age/bias.

    enc_membership: [B, M, S]
    gates: [B, M] model gates when use_model_gates else ignored
    """
    bsz, n_enc, _ = enc_membership.shape
    n_t = len(specs)
    logits = np.zeros((bsz, n_t), dtype=np.float64)
    for b in range(bsz):
        logits[b] = oracle_age_bias_terms(float(ages[b]), specs, scenario)
        for m in range(n_enc):
            if bool(enc_pad[b, m]):
                continue
            mem = enc_membership[b, m]
            if not np.any(mem > 0):
                continue
            pre = oracle_content_pre_decay(mem, specs, W)
            tau = float(enc_tau[b, m])
            for k, sp in enumerate(specs):
                mech = sp["mechanism"]
                if abs(pre[k]) < 1e-15:
                    continue
                if use_model_gates:
                    if mech == "content_only":
                        g = 1.0
                    elif mech in ("null", "age_only"):
                        continue
                    else:
                        g = float(gates[b, m])
                else:
                    g = oracle_gate_for_target(
                        float(ages[b]), tau, scenario=scenario, mechanism=mech,
                        theta0=theta0, beta_true=beta_true,
                    )
                logits[b, k] += g * pre[k]
    return logits
