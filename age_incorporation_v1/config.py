"""Hyperparameters for Developmental Age Incorporation Benchmark v1.

Shared across all arms and tasks. Do not search these.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    REPO_ROOT / "synthea" / "sep1-exp" / "output" / "full" / "processed"
)
DEFAULT_OUTPUT_DIR = EXP_DIR / "outputs"

ARMS = ("no_age", "late_age", "additive_age", "conditioned_age")
TASKS = ("S0", "S1", "S2")
AGE_GROUPS = ("<1", "1-5", "6-11", "12-17")
FULL_SEEDS = (0, 1, 2, 3, 4)

PAD = "<PAD>"
UNK = "<UNK>"
LABEL_COL = {"S0": "y_S0", "S1": "y_S1", "S2": "y_S2"}


@dataclass
class Config:
    experiment: str = "Developmental Age Incorporation Benchmark v1"
    data_dir: str = str(DEFAULT_DATA_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    task: str = "S2"
    arm: str = "no_age"
    seed: int = 0

    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.10
    max_seq_len: int = 1024
    age_hidden: int = 32
    head_hidden: int = 64
    age_scale_years: float = 18.0

    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 30
    grad_clip: float = 1.0
    patience: int = 5

    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def out_path(self) -> Path:
        return Path(self.output_dir)

    def run_dir(self) -> Path:
        name = f"{self.task}_{self.arm}_seed{self.seed}"
        return self.out_path / name
