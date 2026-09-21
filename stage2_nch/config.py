"""Stage-2 NCH pediatric adaptation: shared constants and documented transforms.

Attention (identical for every head h; architecture unchanged from Stage-1):

    s_ij^(h) = q_i^(h)T k_j^(h) / sqrt(d_h) - [lambda0 + beta_P * z_P(a_i)] * tau_ij

Pediatric age (do NOT reuse MIMIC adult mu/sigma)
-------------------------------------------------
    z_P(a) = (a - 9) / 9
    a      = age_at_event_days / 365.25     (years)

Processed pediatric first-study, pre-index diagnosis ages are in [0, 17.98].
Clipping is therefore NOT applied in the primary formula. If any age outside
[0, 18] is observed at tensorize time it is clipped to that interval (equivalent
to clip(z_P, -1, 1)) and counted in the tensorize report.

Arms
----
no_interaction  : beta_P frozen at 0 after Stage-1 load
age_temporal    : beta_P initialized to 0 and trained

Both arms load the same Stage-1 checkpoint (lambda0 kept, adult beta discarded).
"""
from __future__ import annotations

from pathlib import Path

from stage1_mimic_pretrain.config import (  # noqa: F401  (re-export)
    ARM_ALIASES,
    ARM_CHOICES,
    ARMS,
    DAYS_PER_YEAR,
    DEFAULT_D_MODEL,
    DEFAULT_DEMO_HIDDEN,
    DEFAULT_FFN_MULT,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR_AGE,
    DEFAULT_LR_BACKBONE,
    DEFAULT_LR_HEAD,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    EMBEDDING_PATH,
    HEAD_FINAL_BIAS_PRETRAIN,
    N_SHUFFLE,
    REPO_ROOT,
    SHUFFLE_SEED,
    VOCAB_PATH,
    WEEK_DAYS,
    resolve_arm,
    tau_transform_spec,
)

PEDIATRIC_AGE_CENTER_YEARS = 9.0
PEDIATRIC_AGE_SCALE_YEARS = 9.0
PEDIATRIC_AGE_MIN_YEARS = 0.0
PEDIATRIC_AGE_MAX_YEARS = 18.0
PROBE_AGES_YEARS: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 15.0, 18.0)
CONSTANT_ATTENTION_AGE_YEARS = 9.0  # z_P(9) = 0

EVAL_KS: tuple[int, ...] = (5, 10, 20, 50)

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 12
DEFAULT_PATIENCE = 4
DEFAULT_NUM_WORKERS = 6

AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.0),
)

STAGE1_BEST_CKPT = REPO_ROOT / "stage1_mimic_pretrain/run/adkm_s0/checkpoint_best.pt"
STAGE1_BEST_AUPRC_NOTE = (
    "adkm_s0 selected on validation BCE (epoch 7). micro-AUPRC at that "
    "checkpoint is used as the companion ranking metric; no separate Stage-1 "
    "AUPRC checkpoint was saved."
)

NCH_V2 = REPO_ROOT / "artifacts/nch_stage2/v2"
NCH_CLEAN_EVENTS = NCH_V2 / "processed/canonical_events_clean.parquet"
NCH_INDEX_FIRST = NCH_V2 / "processed/sleep_studies_pediatric_first.parquet"
NCH_SPLIT_DIR = NCH_V2 / "splits"
NCH_TENSORIZED_DIR = NCH_V2 / "tensorized_forecast"
STAGE2_RUN_ROOT = REPO_ROOT / "stage2_nch/run"

PRIMARY_REPRESENTATION = "diagnoses_only"
PRIMARY_MODALITIES: tuple[str, ...] = ("diagnosis",)


def z_pediatric_numpy(age_years, clip: bool = False):
    """Fixed pediatric map z_P(a) = (a − 9) / 9."""
    import numpy as np
    z = (np.asarray(age_years, dtype=np.float64) - PEDIATRIC_AGE_CENTER_YEARS) / PEDIATRIC_AGE_SCALE_YEARS
    if clip:
        z = np.clip(z, -1.0, 1.0)
    return z


def pediatric_age_transform_spec(*, clip_applied: bool) -> dict:
    formula = "z_P(a) = (a - 9) / 9"
    if clip_applied:
        formula = "z_P(a) = clip((a - 9) / 9, -1, 1)"
    return {
        "name": "pediatric_fixed",
        "formula": formula,
        "a_unit": "years (age_at_event_days / 365.25)",
        "center_years": PEDIATRIC_AGE_CENTER_YEARS,
        "scale_years": PEDIATRIC_AGE_SCALE_YEARS,
        "intended_domain_years": [PEDIATRIC_AGE_MIN_YEARS, PEDIATRIC_AGE_MAX_YEARS],
        "clip_applied": bool(clip_applied),
        "source": "Stage-2 spec; not estimated from NCH or MIMIC moments",
        "trainable": False,
        "fourier": False,
        "chebyshev": False,
        "age_mlp": False,
        "mimic_adult_mu_sigma_reused": False,
    }


def stage2_target_spec() -> dict:
    return {
        "name": "future_encounter_codes",
        "input": "events with timestamp < start_time(V_{m+1}) (strict); ties go to the target",
        "target": "multi-hot code set of the next NCH encounter, UNK dropped, duplicates collapsed",
        "horizon": "next clinical encounter (encounter_id), not a fixed calendar window",
        "index_cut": "only events strictly before the first pediatric sleep-study index_time",
        "masking": "padding_only (not causal); target encounter is outside the input window",
        "representation": PRIMARY_REPRESENTATION,
        "modalities": list(PRIMARY_MODALITIES),
        "head": "full Stage-1 vocabulary |V|=30635 retained",
        "oov": "input UNK (vocab index |V| → model id 1); UNK dropped from the target",
        "loss": "BCEWithLogitsLoss, unweighted, no pos_weight",
        "implementation": "stage2_nch.tensorize + stage2_nch.dataset.NCHForecastDataset",
    }


def age_band_name(age_years: float) -> str:
    if age_years != age_years:
        return "missing"
    for name, lo, hi in AGE_BANDS:
        if lo <= float(age_years) < hi:
            return name
    if float(age_years) >= 18.0:
        return ">=18"
    return "<0"
