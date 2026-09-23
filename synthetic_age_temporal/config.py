"""Benchmark configuration: transforms, scenarios, and training defaults."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = Path(__file__).resolve().parent

# Existing ~10k pediatric Synthea cohort (sep1-exp full strata). Do not regenerate.
DEFAULT_SYNTHEA_PROCESSED = (
    REPO_ROOT / "synthea" / "sep1-exp" / "output" / "full" / "processed"
)
DEFAULT_SYNTHEA_ENGINE = Path("/home/suraj/Git/synthea")
DEFAULT_OUTPUT_DIR = PKG_DIR / "outputs"
DEFAULT_RESULTS_DIR = PKG_DIR / "results"

# Recorded Synthea provenance (from sep1-exp generation metadata).
SYNTHEA_VERSION = "master-branch-latest"
SYNTHEA_COMMIT = "aa0772fb5e92e48a776c51508c00eddc0d9d27ff"
SYNTHEA_GEOGRAPHY = "Massachusetts"
SYNTHEA_REFERENCE_DATE = "20260101"
SYNTHEA_STRATUM_SEEDS = {
    "infant": 202601011,
    "early_childhood": 202601012,
    "school_age": 202601013,
    "adolescent": 202601014,
}

# Pediatric age normalization: z(0)=-1, z(9)=0, z(18)=1.
AGE_CENTER = 9.0
AGE_SCALE = 9.0
AGE_MIN = 0.0
AGE_MAX = 18.0

# τ = log(1 + Δt_days / 7). Same convention as Stage-1 / Stage-2.
TAU_WEEK_SCALE = 7.0
DAYS_PER_YEAR = 365.25

# Controlled support region (avoids max-lag ↔ age shortcuts).
CONTROLLED_MIN_AGE = 2.0
CONTROLLED_MAX_SIGNAL_LAG_DAYS = 730.0  # 2 years

SIGNAL_LAGS_DAYS = (7.0, 30.0, 90.0, 180.0, 365.0, 730.0)
N_SIGNAL_TYPES = 12  # SYN_SIGNAL_A .. SYN_SIGNAL_L
SIGNAL_PREFIX = "SYN_SIGNAL_"

# Multi-label target inventory.
N_TARGETS = 32
TARGET_MECHANISM_COUNTS = {
    "interaction": 8,
    "temporal_only": 6,
    "age_only": 6,
    "content_only": 6,
    "null": 6,
}

# Ground-truth softplus parameters (calibrated; see ground_truth.calibrate).
# λ(a) = softplus(θ0 + β z(a)); R = exp(-λ τ).
THETA0_DEFAULT = 0.0
BETA_S2 = -2.5  # younger → faster decay
BETA_S3 = 2.5   # reversed
BETA_S0 = 0.0
BETA_S1 = 0.0   # age main effect only; temporal λ fixed

# Multiplicative scale on Σ w_kj R(a,τ) for interaction / temporal targets.
RELEVANCE_LOGIT_SCALE = 2.5

# Interaction-strength sweep for S2 (absolute |β|).
INTERACTION_STRENGTHS = {
    "weak": 1.2,
    "medium": 2.5,
    "strong": 4.0,
}

# Outcome generation.
TARGET_PREVALENCE = 0.25
NOISE_STD = 0.35
TEMPORAL_ONLY_LAMBDA0 = 0.8  # fixed λ for temporal-only targets

# Splits / seeds.
SPLIT_SEED = 20260922
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
DATA_SEED = 20260922
MODEL_SEED = 0

# Model arms (matched architecture).
ARMS = (
    "no_age",
    "age_only",
    "temporal_only",
    "age_temporal",
    "historical_age",
    "temporal_only_per_head",
    "age_temporal_per_head",
)
SCENARIOS = ("S0", "S1", "S2", "S3")

# Training (matched across arms).
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 1
DIM_FF = 512
DROPOUT = 0.10
MAX_SEQ_LEN = 96
MAX_BACKGROUND_EVENTS = 48  # subsample; all signal events always kept
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 25
PATIENCE = 5
GRAD_CLIP = 1.0

# Evaluation.
PRECISION_K = 5
RECALL_K = 5
PROBE_AGES = (1.0, 5.0, 10.0, 15.0, 18.0)
SURFACE_AGES = tuple(float(x) for x in range(0, 19))
SURFACE_LAGS_DAYS = (0.0, 7.0, 30.0, 90.0, 180.0, 365.0, 730.0)

PAD = "<PAD>"
UNK = "<UNK>"
QUERY_CODE = "PRED_QUERY"
SIGNAL_TYPE = "signal"
BACKGROUND_TYPE = "background"
QUERY_TYPE = "query"

# Forbidden model-input keys (ground-truth leakage).
FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "true_lambda",
        "true_event_relevance",
        "true_relevance",
        "scenario",
        "target_mechanism_type",
        "mechanism_type",
        "beta_true",
        "theta0_true",
        "true_target_logit",
        "true_target_probability",
    }
)


def z_age(age_years) -> Any:
    """Pediatric age normalization z(a)=(a-9)/9."""
    import numpy as np

    a = np.asarray(age_years, dtype=np.float64)
    return (a - AGE_CENTER) / AGE_SCALE


def tau_from_days(delta_days) -> Any:
    """τ = log(1 + Δt_days / 7). Unnormalized (matches Stage-1 convention)."""
    import numpy as np

    return np.log1p(np.asarray(delta_days, dtype=np.float64) / TAU_WEEK_SCALE)


def softplus_np(x) -> Any:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -40.0, 40.0))))


def lambda_true(age_years, theta0: float, beta: float) -> Any:
    """λ_true(a) = softplus(θ0 + β z(a)) > 0."""
    return softplus_np(theta0 + beta * z_age(age_years))


def relevance(age_years, tau, theta0: float, beta: float) -> Any:
    """R(a,τ) = exp(-λ(a) τ) ∈ (0, 1]."""
    import numpy as np

    lam = lambda_true(age_years, theta0, beta)
    return np.exp(-lam * np.asarray(tau, dtype=np.float64))


def signal_code(i: int) -> str:
    return f"{SIGNAL_PREFIX}{chr(ord('A') + i)}"


def all_signal_codes() -> tuple[str, ...]:
    return tuple(signal_code(i) for i in range(N_SIGNAL_TYPES))


@dataclass
class ScenarioSpec:
    name: str
    beta_true: float
    theta0: float = THETA0_DEFAULT
    # If True, interaction targets use age×temporal; else temporal-only λ.
    has_interaction: bool = True
    # Age main-effect strength on interaction / age_only targets.
    age_main_effect: float = 0.0
    description: str = ""


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "S0": ScenarioSpec(
        name="S0",
        beta_true=BETA_S0,
        has_interaction=False,
        age_main_effect=0.0,
        description="Temporal relevance exists; age does not modify decay.",
    ),
    "S1": ScenarioSpec(
        name="S1",
        beta_true=BETA_S1,
        has_interaction=False,
        age_main_effect=1.2,
        description="Age main effect only; temporal λ age-independent.",
    ),
    "S2": ScenarioSpec(
        name="S2",
        beta_true=BETA_S2,
        has_interaction=True,
        age_main_effect=0.0,
        description="Developmental age × temporal: β<0 (younger decays faster).",
    ),
    "S3": ScenarioSpec(
        name="S3",
        beta_true=BETA_S3,
        has_interaction=True,
        age_main_effect=0.0,
        description="Reversed interaction: β>0 (older decays faster).",
    ),
}


@dataclass
class Config:
    scenario: str = "S2"
    arm: str = "age_temporal"
    strength: str = "medium"  # S2 sweep only
    cohort: str = "controlled"  # controlled | full
    data_seed: int = DATA_SEED
    model_seed: int = MODEL_SEED
    synthea_dir: str = str(DEFAULT_SYNTHEA_PROCESSED)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    results_dir: str = str(DEFAULT_RESULTS_DIR)

    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    dim_feedforward: int = DIM_FF
    dropout: float = DROPOUT
    max_seq_len: int = MAX_SEQ_LEN
    batch_size: int = BATCH_SIZE
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    grad_clip: float = GRAD_CLIP
    num_workers: int = 0
    device: str = "cuda"
    run_tag: str = ""
    interaction_only: bool = False
    track_mechanism_each_epoch: bool = False

    # Overridable ground-truth params (filled from scenario/strength).
    theta0: float = THETA0_DEFAULT
    beta_true: float = BETA_S2

    def resolve_gt(self) -> None:
        spec = SCENARIO_SPECS[self.scenario]
        self.theta0 = spec.theta0
        if self.scenario == "S2" and self.strength in INTERACTION_STRENGTHS:
            self.beta_true = -abs(INTERACTION_STRENGTHS[self.strength])
        elif self.scenario == "S3" and self.strength in INTERACTION_STRENGTHS:
            # S3 keeps positive sign; strength scales magnitude.
            self.beta_true = abs(INTERACTION_STRENGTHS.get(self.strength, BETA_S3))
            if self.strength == "medium":
                self.beta_true = BETA_S3
        else:
            self.beta_true = spec.beta_true

    def to_dict(self) -> dict[str, Any]:
        self.resolve_gt()
        d = asdict(self)
        d["z_formula"] = "(a - 9) / 9"
        d["tau_formula"] = "log(1 + dt_days / 7)"
        d["lambda_formula"] = "softplus(theta0 + beta * z(a))"
        d["relevance_formula"] = "exp(-lambda(a) * tau)"
        d["synthea_version"] = SYNTHEA_VERSION
        d["synthea_commit"] = SYNTHEA_COMMIT
        return d

    @property
    def out_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def res_path(self) -> Path:
        return Path(self.results_dir)

    @property
    def synthea_path(self) -> Path:
        return Path(self.synthea_dir)

    def data_dir(self) -> Path:
        return self.out_path / "data" / f"seed{self.data_seed}"

    def scenario_dir(self) -> Path:
        self.resolve_gt()
        name = self.scenario
        if self.scenario in ("S2", "S3") and self.strength != "medium":
            name = f"{self.scenario}_{self.strength}"
        return self.data_dir() / self.cohort / name

    def run_dir(self) -> Path:
        self.resolve_gt()
        name = f"{self.scenario}_{self.arm}_d{self.data_seed}_m{self.model_seed}"
        if self.scenario == "S2" and self.strength != "medium":
            name = f"{self.scenario}_{self.strength}_{self.arm}_d{self.data_seed}_m{self.model_seed}"
        if self.interaction_only:
            name = f"{name}_interonly"
        if self.max_seq_len != MAX_SEQ_LEN:
            name = f"{name}_L{self.max_seq_len}"
        if self.run_tag:
            name = f"{self.run_tag}_{name}"
        return self.out_path / "runs" / self.cohort / name
