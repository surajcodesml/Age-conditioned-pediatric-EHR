"""BaselineModel ABC and standard ModelOutput.

Every baseline exposes the same interface so that training loops,
evaluation harnesses, and counterfactual probes work uniformly.
Models must NOT be forced into identical internal representations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Standard output from any baseline model."""
    logits: torch.Tensor                                  # [B, n_targets]
    patient_repr: torch.Tensor                            # [B, d]
    token_reprs: Optional[torch.Tensor] = None            # [B, L, d]
    attention: Optional[torch.Tensor] = None              # model-specific
    extras: dict[str, Any] = field(default_factory=dict)  # model-specific


class BaselineModel(ABC):
    """Abstract interface that every baseline must implement.

    The common training loop, evaluation code, and counterfactual probes
    call only these methods, so any model that satisfies this contract
    plugs in without modification.

    Neural baselines inherit from both BaselineModel and nn.Module.
    LightGBM inherits only BaselineModel.
    """

    # ----- required properties -----
    @property
    @abstractmethod
    def name(self) -> str:
        """Short model identifier, e.g. 'behrt', 'medbert'."""

    @property
    @abstractmethod
    def has_age_input(self) -> bool:
        """True if model receives an explicit age feature."""

    @property
    @abstractmethod
    def has_time_input(self) -> bool:
        """True if model receives explicit timestamp / lag / τ features."""

    @property
    @abstractmethod
    def model_card(self) -> dict[str, Any]:
        """Architecture metadata for capacity reporting.

        Must include at minimum:
            trainable_params, embedding_params, layers, heads,
            hidden_size, ffn_size, max_seq_len
        """

    # ----- required methods -----
    @abstractmethod
    def predict(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        """Forward pass → standard ModelOutput. No gradient."""

    @abstractmethod
    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """One gradient step. Returns at least {'loss': ...}."""

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Serialize model state."""

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Restore model state."""

    # ----- optional overrides -----
    def encode_history(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Patient representation without the prediction head.

        Default implementation calls predict() and returns patient_repr.
        """
        return self.predict(batch).patient_repr

    def get_optimizer_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Return parameter groups for AdamW.

        Default: single group with all trainable parameters.
        """
        if isinstance(self, nn.Module):
            return [{"params": [p for p in self.parameters() if p.requires_grad],
                     "lr": lr, "weight_decay": weight_decay}]
        raise NotImplementedError("Non-neural models must override get_optimizer_groups")
