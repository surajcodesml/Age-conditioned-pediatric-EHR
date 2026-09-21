"""Stage-1 MIMIC-IV pretraining: shared constants and documented transforms.

Attention (identical for every head h):

    s_ij^(h) = q_i^(h)T k_j^(h) / sqrt(d_h) - [lambda0 + beta * z(a_i)] * tau_ij

Arms
----
no_interaction  : demographic age remains; beta frozen at 0  (lambda(a) = lambda0)
age_temporal    : lambda0 and beta are both trained in self-attention only
temporal_only   : alias of no_interaction (kept so old flags still parse)

Age normalization (adult MIMIC, frozen from the train corpus)
-------------------------------------------------------------
    z(a) = (a - mu) / sigma
    a    = age_at_event_days / 365.25     (years)
    mu   = 63.33601047086648              (event-level mean, years)
    sigma= 16.574804662346914             (event-level std, years)

These mu, sigma are the exact event_age_mean / event_age_sd computed by
model_new.data.corpus_stats on the train split (N_events = 405,519,425).
They are not re-estimated at fine-tune or per-run. A live
corpus_stats_cached pass overrides them only if the corpus key matches.

Temporal distance (existing convention; not redefined)
------------------------------------------------------
    tau_ij = log(1 + |t_i - t_j| / c)
    c      = 7 days
    t      in days from the first event (timestamps_days)

Differencing and log1p run in float64 inside model_new.data.lag_to_tau.
There is no Chebyshev rescaling tau_tilde = 2 tau / tau_max - 1.

Future-event target (existing; not redefined)
---------------------------------------------
One sample per (patient, target visit V_{m+1}) with at least one prior event:

    input  = every event with timestamp < start_time(V_{m+1})  (strict)
    target = multi-hot code set of V_{m+1}, length |V|
             (UNK dropped; duplicates collapsed)

Visits are hadm-derived blocks. Truncation keeps the newest max_seq_len
pre-boundary events. Padding-only attention is not leakage because no input
event is at or after the target boundary (INV-HORIZON).

Loss
----
    L = BCEWithLogitsLoss(logits, y)     # unweighted; no pos_weight
    logits = f(h)                        # sigmoid is not applied first

The existing pipeline never used pos_weight. Class imbalance is reported
but not re-weighted.
"""
from __future__ import annotations

from pathlib import Path

ARMS = ("no_interaction", "age_temporal")
ARM_ALIASES = {"temporal_only": "no_interaction"}
ARM_CHOICES = ARMS + tuple(ARM_ALIASES)


def resolve_arm(arm: str) -> str:
    """Map CLI/legacy names onto the canonical arm."""
    name = ARM_ALIASES.get(str(arm), str(arm))
    if name not in ARMS:
        raise ValueError(f"arm must be one of {ARM_CHOICES}, got {arm!r}")
    return name


REPO_ROOT = Path(__file__).resolve().parents[1]

# Frozen adult-MIMIC event-level age moments (train split, corpus_stats).
MIMIC_AGE_MEAN_YEARS = 63.33601047086648
MIMIC_AGE_STD_YEARS = 16.574804662346914
DAYS_PER_YEAR = 365.25
WEEK_DAYS = 7.0  # c in τ = log1p(Δt / c)

# Probe ages for λ(a) = λ0 + β z(a). Adult range, as specified.
PROBE_AGES_YEARS: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)

# Existing ranking cutoffs used in model_new.train (recall@5/10/20).
EVAL_KS: tuple[int, ...] = (5, 10, 20)

# Production backbone: 4 heads, d_model=256 → d_h=64. λ0/β stay shared across heads.
DEFAULT_D_MODEL = 256
DEFAULT_N_LAYERS = 1
DEFAULT_N_HEADS = 4
DEFAULT_FFN_MULT = 4
DEFAULT_DEMO_HIDDEN = 64
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 8
DEFAULT_LR_BACKBONE = 1e-4
DEFAULT_LR_AGE = 1e-3
DEFAULT_LR_HEAD = 1e-3
DEFAULT_GRAD_CLIP = 1.0
HEAD_FINAL_BIAS_PRETRAIN = -7.0

N_SHUFFLE = 5
SHUFFLE_SEED = 0

TENSORIZED_DIR = REPO_ROOT / "data/processed/tensorized_flat"
EMBEDDING_PATH = REPO_ROOT / "data/processed/bge_embeddings.pt"
VOCAB_PATH = REPO_ROOT / "data/processed/code_vocab.json"


def z_age_numpy(age_years, mean: float = MIMIC_AGE_MEAN_YEARS,
                sd: float = MIMIC_AGE_STD_YEARS):
    """Fixed affine map used everywhere ``z(a)`` appears."""
    import numpy as np
    return (np.asarray(age_years, dtype=np.float64) - float(mean)) / max(float(sd), 1e-6)


def age_transform_spec(mean: float = MIMIC_AGE_MEAN_YEARS,
                       sd: float = MIMIC_AGE_STD_YEARS) -> dict:
    return {
        "name": "standardize_years",
        "formula": "z(a) = (a - mean) / sd",
        "a_unit": "years (age_at_event_days / 365.25)",
        "mean": float(mean),
        "sd": float(sd),
        "source": "model_new.data.corpus_stats event-level moments on MIMIC train",
        "trainable": False,
        "fourier": False,
        "chebyshev": False,
        "age_mlp": False,
    }


def tau_transform_spec() -> dict:
    return {
        "name": "log1p_weeks",
        "formula": "tau_ij = log(1 + |t_i - t_j| / c)",
        "c_days": WEEK_DAYS,
        "t_unit": "days from first event (timestamps_days)",
        "implementation": "model_new.data.lag_to_tau (float64 abs, then log1p)",
        "chebyshev_rescale": False,
        "tau_max_used": False,
    }


def target_spec() -> dict:
    return {
        "name": "future_visit_codes",
        "input": "events with timestamp < start_time(V_{m+1}) (strict); ties go to target",
        "target": "multi-hot code set of hadm-visit V_{m+1}, UNK dropped",
        "horizon": "next admission/visit; not a fixed calendar window",
        "masking": "padding_only (not causal); target visit is outside the input window",
        "implementation": "model_new.data.TensorizedPretrainDataset",
        "loss": "BCEWithLogitsLoss, unweighted, no pos_weight",
    }
