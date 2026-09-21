"""Hyperparameters for the minimal age × temporal interaction experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_SYNTHEA_DIR = (
    REPO_ROOT / "synthea" / "sep1-exp" / "output" / "full" / "processed"
)
DEFAULT_OUTPUT_DIR = EXP_DIR / "outputs"
DEFAULT_RESULTS_DIR = EXP_DIR / "results"

ARMS = ("no_age", "temporal_only", "late_age", "age_temporal")
TASKS = ("T0", "T1", "T2")
AGE_GROUPS = ("<1", "1-5", "6-11", "12-17")
FULL_SEEDS = (0, 1, 2, 3, 4)
PROBE_AGES = (1.0, 3.0, 5.0, 8.0, 11.0, 14.0, 17.0)
KERNEL_AGES = (2.0, 6.0, 10.0, 14.0, 17.0)

PAD = "<PAD>"
UNK = "<UNK>"
QUERY_CODE = "QUERY"
POS_CODE = "POS_SIGNAL"
NEG_CODE = "NEG_SIGNAL"
QUERY_TYPE = "query"
SIGNAL_TYPE = "injected_signal"

LABEL_COL = {"T0": "y_T0", "T1": "y_T1", "T2": "y_T2"}
BETA_TRUE = {"T0": 0.0, "T1": 1.0, "T2": -1.0}
LAMBDA0_TRUE = 0.5

# τ = log(1 + Δt_days / TAU_WEEK_SCALE) / log(1 + GAP_MAX_DAYS / TAU_WEEK_SCALE)
TAU_WEEK_SCALE = 7.0
GAP_MIN_DAYS = 7.0
GAP_MAX_DAYS = 90.0
GAP_NEAR = (7.0, 14.0)
GAP_FAR = (60.0, 90.0)
N_SIG_MIN = 4
N_SIG_MAX = 8
MIN_HISTORY_DAYS = 90.0

DATA_SEED = 20260918
MATCHED_SEED = 20260919
MATCHED_AGE_YOUNG = 2.0
MATCHED_AGE_OLD = 16.0
MATCHED_MARGIN = 0.02
MATCHED_TASKS = ("T1", "T2")
DEFAULT_MATCHED_OUTPUT_DIR = EXP_DIR / "outputs"
DEFAULT_MATCHED_RESULTS_DIR = EXP_DIR / "results" / "matched"
DEFAULT_MATCHED_DATA_DIR = EXP_DIR / "outputs" / "matched_data"


def tau_max() -> float:
    import math

    return math.log1p(GAP_MAX_DAYS / TAU_WEEK_SCALE)


def tau_from_days(delta_days) -> Any:
    import math

    import numpy as np

    x = np.log1p(np.asarray(delta_days, dtype=np.float64) / TAU_WEEK_SCALE)
    return np.clip(x / tau_max(), 0.0, 1.0)


@dataclass
class Config:
    experiment: str = "minimal_age_temporal_interaction"
    synthea_dir: str = str(DEFAULT_SYNTHEA_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    results_dir: str = str(DEFAULT_RESULTS_DIR)

    task: str = "T1"
    arm: str = "age_temporal"
    seed: int = 0

    d_model: int = 64
    n_layers: int = 1
    n_heads: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.10
    max_seq_len: int = 128
    head_hidden: int = 32

    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 30
    grad_clip: float = 1.0
    patience: int = 6

    num_workers: int = 0
    keep_background: bool = True
    run_tag: str = ""

    lambda0_true: float = LAMBDA0_TRUE
    beta_true: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tau_week_scale"] = TAU_WEEK_SCALE
        d["gap_near_days"] = list(GAP_NEAR)
        d["gap_far_days"] = list(GAP_FAR)
        d["tau_max"] = tau_max()
        d["tau_formula"] = "log(1 + dt_days / 7) / log(1 + 90 / 7), clipped to [0, 1]"
        return d

    @property
    def synthea_path(self) -> Path:
        return Path(self.synthea_dir)

    @property
    def out_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def res_path(self) -> Path:
        return Path(self.results_dir)

    def run_dir(self) -> Path:
        name = f"{self.task}_{self.arm}_seed{self.seed}"
        if self.run_tag:
            name = f"{self.run_tag}_{name}"
        return self.out_path / "runs" / name
