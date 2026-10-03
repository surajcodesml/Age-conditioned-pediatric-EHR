"""Resolved configs for C00–C06. Later experiments are not edited from earlier results."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

from atomic import CONFIG_PATH

EXPERIMENT_ORDER = (
    "C00_current_dtr",
    "C01_staged_current",
    "C02_mass_current",
    "C03_weak_lambda_init",
    "C04_no_content_persistence",
    "C05_shared_beta_mixture",
    "C06_component_beta_mixture",
)

_CARDS: dict[str, dict[str, Any]] = {
    "C00_current_dtr": {
        "variant": "current",
        "staged": False,
        "hypothesis": "Multi-seed reference for the current Content-Persistence DTR, with matched arm initialization.",
        "change": "None. Current architecture and the single AdamW group used for dtr_*_new.",
    },
    "C01_staged_current": {
        "variant": "current",
        "staged": True,
        "hypothesis": "Staged optimization improves the current architecture.",
        "change": "No architectural change. Stage A trains beta=0, then both arms clone that checkpoint. Stage B trains theta0/beta with the encoder and readout frozen. Stage C jointly fine-tunes.",
    },
    "C02_mass_current": {
        "variant": "mass",
        "staged": False,
        "hypothesis": "Keeping evidence mass beside the normalized history improves the current DTR.",
        "change": "Replace h=sum w v by concat(sum(w v)/(M+eps), log1p(M)). The history MLP stays; only its first input dimension grows by one.",
    },
    "C03_weak_lambda_init": {
        "variant": "weak_init",
        "staged": False,
        "hypothesis": "Initializing effective lambda at 0.1 instead of softplus(0) lets long-range evidence survive early training.",
        "change": "Set theta0 so softplus(theta0) = 0.1. Persistence intercept stays 0, so the initial effective lambda is 0.1.",
    },
    "C04_no_content_persistence": {
        "variant": "no_persistence",
        "staged": False,
        "hypothesis": "Removing r·v + b_r lets the developmental slope be the only persistence mechanism.",
        "change": "Drop the content persistence offset. Keep content query, exp(u), raw aggregation, history MLP, and the current optimizer.",
    },
    "C05_shared_beta_mixture": {
        "variant": "shared_beta_mixture",
        "staged": False,
        "hypothesis": "A few content-specific baseline timescales help when they share one developmental slope.",
        "change": "Replace r·v + b_r with K=3 mixture weights from content only and lambda_k=softplus(theta_k + beta z). beta is one shared scalar.",
    },
    "C06_component_beta_mixture": {
        "variant": "component_beta_mixture",
        "staged": False,
        "hypothesis": "Separate developmental slopes across mixture components are required for the E05-style gain.",
        "change": "Identical to C05 except each component has its own beta_k.",
    },
}


def load_base() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def experiment_configs() -> list[dict[str, Any]]:
    base = load_base()
    configs = []
    for experiment_id in EXPERIMENT_ORDER:
        card = _CARDS[experiment_id]
        cfg = dict(base)
        cfg.update(card)
        cfg["experiment_id"] = experiment_id
        payload = {k: v for k, v in cfg.items() if k != "config_hash"}
        cfg["config_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        configs.append(cfg)
    return configs
