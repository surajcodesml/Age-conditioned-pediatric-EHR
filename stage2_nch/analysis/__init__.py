"""Verified Stage-2 age×temporal contract (from implementation, not names)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "artifacts" / "nch_stage2" / "analysis_age_temporal"
ADKM_DIR = REPO / "stage2_nch" / "run" / "adkm_nch_s0"
NINT_DIR = REPO / "stage2_nch" / "run" / "nint_nch_s0"
STAGE1_BEST = REPO / "stage1_mimic_pretrain" / "run" / "adkm_s0" / "checkpoint_best.pt"

# Primary predictive selection for paper figures (documented in training).
PRIMARY_CKPT_NAME = "checkpoint_best_auprc.pt"

CONTRACT = {
    "attention_equation": (
        "s_ij^(h) = q_i^(h)^T k_j^(h) / sqrt(d_h) "
        "- [lambda0 + beta * z(a_i)] * tau_ij"
    ),
    "kernel_bias_K": "K(a, Delta_t) = -lambda(a) * tau(Delta_t)",
    "lambda_of_age": "lambda(a) = lambda0 + beta * z(a)",
    "age_standardization_stage2": {
        "name": "z_P",
        "formula": "z_P(a) = (a - 9) / 9",
        "center_years": 9.0,
        "scale_years": 9.0,
        "note": "Fixed pediatric transform; MIMIC adult mu/sigma NOT reused",
    },
    "conditioning_age": {
        "symbol": "a_i",
        "meaning": "per-query event age in years (age_at_event_days / 365.25)",
        "tensor": "batch['age_years'] shape [B, L]",
        "not": "not solely index age; each attention query uses its own event age",
        "strata_proxy": (
            "age-band performance strata use last_age_years (age of the last "
            "input event / query at the sequence end), which approximates "
            "developmental age at the forecast window"
        ),
    },
    "tau_transform": {
        "formula": "tau_ij = log(1 + |t_i - t_j| / 7)",
        "c_days": 7.0,
        "t_unit": "days from first event in the window (timestamps_days)",
        "implementation": "model_new.data.lag_to_tau / tau_from_timestamps",
    },
    "parameterization": {
        "lambda0": "scalar nn.Parameter, shared across all attention heads",
        "beta": "scalar nn.Parameter, shared across all attention heads",
        "head_specific": False,
        "vector_valued": False,
        "applied_in": "Transformer self-attention only (pool_temporal_bias=False)",
    },
    "sign_convention": {
        "source": "stage2_nch/sign_test.py + run lambda0_sign_test.json",
        "positive_lambda_is_recency": True,
        "negative_lambda_is_long_range": True,
        "mechanism": (
            "Larger positive lambda makes -lambda*tau more negative at large tau, "
            "reducing attention mass on distant keys (recency). "
            "Negative lambda increases distant mass (long-range preference)."
        ),
        "multiplicative_relevance": (
            "exp(K) = exp(-lambda(a)*tau); for lambda>0 this decays with lag; "
            "for lambda<0 it grows with lag (before softmax competition)."
        ),
    },
    "stage2_initialization_from_stage1": {
        "kept": ["temporal.lambda0", "encoder", "pooling", "demo_proj", "head", "embedding_table"],
        "reset": {
            "temporal.beta": 0.0,
            "age_mean": 9.0,
            "age_sd": 9.0,
        },
        "not_transferred": ["adult beta", "MIMIC age mu/sigma"],
        "stage1_ckpt": str(STAGE1_BEST),
    },
    "arms": {
        "age_temporal": "beta trainable (initialized at 0)",
        "no_interaction": "beta frozen at 0",
    },
    "task": {
        "name": "future_encounter_codes",
        "representation": "diagnoses_only",
        "loss": "BCEWithLogitsLoss unweighted",
        "primary_selection_for_this_analysis": PRIMARY_CKPT_NAME,
        "held_out_split": "artifacts/nch_stage2/v2/tensorized_forecast/diagnoses_only/test",
    },
}


def write_contract(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (ANALYSIS / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "age_temporal_contract.json"
    path.write_text(json.dumps(CONTRACT, indent=2) + "\n", encoding="utf-8")
    return path
