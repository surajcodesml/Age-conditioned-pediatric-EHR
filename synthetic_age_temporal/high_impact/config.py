"""Resolved configs for D00–D02. Order fixed before fitting."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from high_impact import CONFIG_PATH

EXPERIMENT_ORDER = (
    "D00_oracle_gate",
    "D01_multihead_shared",
    "D02_multihead_dev",
)

_CARDS: dict[str, dict[str, Any]] = {
    "D00_oracle_gate": {
        "variant": "oracle_gate",
        "staged": True,
        "hypothesis": "With a perfect temporal gate, the remaining C01 content encoder/retrieval/readout sets the prediction ceiling.",
        "change": "Replace the learned developmental gate with the generator lambda_true(a). No trainable theta/beta. Everything else matches C01.",
    },
    "D01_multihead_shared": {
        "variant": "multihead_shared",
        "staged": True,
        "hypothesis": "Four content retrieval heads with fixed total width improve recovery while sharing one developmental gate.",
        "change": "Replace the single content query/key with H=4 heads of width 16. Shared lambda/gate across heads. Concatenate head histories to width 64.",
    },
    "D02_multihead_dev": {
        "variant": "multihead_dev",
        "staged": True,
        "hypothesis": "After multi-head content specialization, head-specific centered developmental slopes help further.",
        "change": "Same multi-head content as D01. beta_h = beta_global + centered delta_h. Identical to D01 at initialization.",
    },
}


def load_base() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def experiment_configs() -> list[dict[str, Any]]:
    base = load_base()
    configs = []
    for experiment_id in EXPERIMENT_ORDER:
        cfg = dict(base)
        cfg.update(_CARDS[experiment_id])
        cfg["experiment_id"] = experiment_id
        payload = {k: v for k, v in cfg.items() if k != "config_hash"}
        cfg["config_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        configs.append(cfg)
    return configs
