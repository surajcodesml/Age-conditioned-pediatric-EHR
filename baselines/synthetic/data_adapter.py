"""Synthetic dataset adapter for baseline training/evaluation.

Provides a uniform interface over controlled scenarios S0–S3 (core
age × temporal) and S5 (heterogeneous persistence).

Evaluation-only fields (persistence groups, is_signal, oracle metadata)
are never passed into model training/predict paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure synthetic_age_temporal is importable (uses bare ``config`` imports).
_SAT = Path(__file__).resolve().parents[2] / "synthetic_age_temporal"
if str(_SAT) not in sys.path:
    sys.path.insert(0, str(_SAT))
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from synthetic_age_temporal.config import Config, MAX_SEQ_LEN
from synthetic_age_temporal.dataset import make_loaders

# Core age × temporal mechanism benchmark.
CORE_SCENARIOS: tuple[str, ...] = ("S0", "S1", "S2", "S3")
# Heterogeneous temporal persistence (alongside core).
S5_SCENARIO = "S5"
# Full synthetic baseline suite (no S6 / multi-horizon).
BENCHMARK_SCENARIOS: tuple[str, ...] = CORE_SCENARIOS + (S5_SCENARIO,)

# Keys allowed into model forward / training_step.
MODEL_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "code_ids",
        "type_ids",
        "lag_days",
        "tau",
        "is_query",
        "padding_mask",
        "age",
        "z_age",
        "labels",
    }
)

# Present in loaders for evaluation / template selection only.
EVAL_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "is_signal",
        "patient_ids",
        "example_ids",
        "persistence_group",
        "persistence_class",
        "true_lambda",
        "true_relevance",
        "true_event_relevance",
        "beta_true",
        "theta0_true",
        "oracle_surface",
        "decay_parameters",
    }
)

# Evaluation-only S5 signal groups (never model inputs).
S5_PERSISTENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "acute": ("SYN_SIGNAL_A", "SYN_SIGNAL_B", "SYN_SIGNAL_C", "SYN_SIGNAL_D"),
    "intermediate": ("SYN_SIGNAL_E", "SYN_SIGNAL_F", "SYN_SIGNAL_G", "SYN_SIGNAL_H"),
    "chronic": ("SYN_SIGNAL_I", "SYN_SIGNAL_J", "SYN_SIGNAL_K", "SYN_SIGNAL_L"),
}

# Ground-truth baseline θ per group (S5 generator; evaluation only).
S5_GROUP_THETA: dict[str, float] = {
    "acute": 1.0,
    "intermediate": 0.0,
    "chronic": -1.0,
}


def resolve_scenarios(scenario_arg: str) -> list[str]:
    """Map CLI scenario argument to concrete scenario list."""
    if scenario_arg == "all":
        return list(BENCHMARK_SCENARIOS)
    if scenario_arg == "core":
        return list(CORE_SCENARIOS)
    if scenario_arg not in BENCHMARK_SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario_arg!r}; "
            f"expected one of {BENCHMARK_SCENARIOS + ('all', 'core')}"
        )
    return [scenario_arg]


def scenario_dir(scenario: str, data_seed: int = 20260922) -> Path:
    cfg = Config(data_seed=data_seed)
    return cfg.data_dir() / "controlled" / scenario


def load_splits(scenario: str, data_seed: int = 20260922) -> dict[str, list[str]]:
    path = scenario_dir(scenario, data_seed) / "splits.json"
    with path.open() as f:
        return json.load(f)


def strip_eval_only(batch: dict[str, Any]) -> dict[str, Any]:
    """Return a copy containing only model-visible tensors."""
    return {k: v for k, v in batch.items() if k in MODEL_INPUT_KEYS}


def assert_no_eval_leakage(batch: dict[str, Any]) -> None:
    """Raise if evaluation-only / GT mechanism fields are present."""
    leaked = set(batch.keys()) & EVAL_ONLY_KEYS
    # Also catch any key that looks like persistence metadata.
    for k in batch.keys():
        kl = k.lower()
        if "persistence" in kl or kl.startswith("true_") or "oracle" in kl:
            leaked.add(k)
    if leaked:
        raise AssertionError(f"Evaluation-only keys leaked into model batch: {sorted(leaked)}")


def make_baseline_loaders(
    scenario: str,
    *,
    data_seed: int = 20260922,
    batch_size: int = 32,
    max_seq_len: int = MAX_SEQ_LEN,
    num_workers: int = 0,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Load train/val/test for any baseline-compatible scenario (S0–S3, S5).

    Returns the same structure as ``make_loaders``, plus adapter metadata
    under ``info['adapter']``.
    """
    if scenario not in BENCHMARK_SCENARIOS:
        raise ValueError(
            f"Scenario {scenario} is not part of the synthetic baseline benchmark. "
            f"Supported: {BENCHMARK_SCENARIOS}"
        )
    sdir = scenario_dir(scenario, data_seed)
    if not sdir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {sdir}")

    train_loader, val_loader, test_loader, vocab, info = make_loaders(
        sdir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
    )
    info = dict(info)
    info["adapter"] = {
        "scenario": scenario,
        "scenario_dir": str(sdir),
        "is_core": scenario in CORE_SCENARIOS,
        "is_s5": scenario == S5_SCENARIO,
        "model_input_keys": sorted(MODEL_INPUT_KEYS),
        "eval_only_keys": sorted(EVAL_ONLY_KEYS),
        "s5_persistence_groups": (
            {g: list(codes) for g, codes in S5_PERSISTENCE_GROUPS.items()}
            if scenario == S5_SCENARIO
            else None
        ),
    }
    info["vocab"] = vocab
    info["scenario_dir"] = sdir
    return train_loader, val_loader, test_loader, vocab, info


def model_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Strip evaluation-only fields and assert no leakage."""
    out = strip_eval_only(batch)
    assert_no_eval_leakage(out)
    return out


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone tensor fields (for counterfactual mutation)."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        else:
            out[k] = v
    return out
