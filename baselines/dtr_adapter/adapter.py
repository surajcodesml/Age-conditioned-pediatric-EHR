"""DTR adapter — Content-Persistence Developmental Temporal Retrieval.

Wraps ``synthetic_age_temporal.model_dtr.DevelopmentalTemporalRetrieval`` for the
baseline evaluation framework.

Ablation contract (locked):
  temporal_only:  λ = softplus(θ₀)            (β frozen at 0)
  age_temporal:   λ(a) = softplus(θ₀ + β z(a))

Both arms share the same age main-effect head f_age(z). Age×history interaction
exists only through β inside λ. History aggregation is raw-additive (mass-
preserving), not softmax.

Legacy arms ``no_age`` / ``age_only`` still use the Transformer BenchmarkModel
(token-level) for backward-compatible ablation scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import register_baseline

# Arms served by Content-Persistence DTR (encounter-level).
_CP_ARMS = frozenset({"temporal_only", "age_temporal"})
# Legacy Transformer arms (token-level).
_LEGACY_ARMS = frozenset({"no_age", "age_only", "age_temporal_per_head",
                          "temporal_only_per_head", "historical_age"})


def _sat_path() -> Path:
    return Path(__file__).resolve().parents[2] / "synthetic_age_temporal"


@register_baseline("dtr")
class DTRAdapter(BaselineModel, nn.Module):
    """Content-Persistence DTR for temporal_only / age_temporal; legacy otherwise."""

    def __init__(
        self,
        arm: str,
        n_codes: int,
        n_types: int,
        n_targets: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        aggregation: str = "raw_additive",
    ) -> None:
        nn.Module.__init__(self)
        import sys

        sat = str(_sat_path())
        if sat not in sys.path:
            sys.path.insert(0, sat)

        self.arm = arm
        self._n_codes = n_codes
        self._n_types = n_types
        self._n_targets = n_targets
        self._d_model = d_model
        self._uses_content_persistence = arm in _CP_ARMS

        if self._uses_content_persistence:
            from model_dtr import DevelopmentalTemporalRetrieval

            self._model = DevelopmentalTemporalRetrieval(
                n_codes=n_codes,
                n_targets=n_targets,
                d_model=d_model,
                age_temporal=(arm == "age_temporal"),
                aggregation=aggregation,
                dropout=dropout,
            )
        else:
            from model import BenchmarkModel

            self._model = BenchmarkModel(
                arm=arm,
                n_codes=n_codes,
                n_types=n_types,
                n_targets=n_targets,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

    @property
    def name(self) -> str:
        return f"dtr_{self.arm}"

    @property
    def uses_encounter_batch(self) -> bool:
        """True when forward expects enc_* tensors (Content-Persistence)."""
        return self._uses_content_persistence

    @property
    def has_age_input(self) -> bool:
        if self._uses_content_persistence:
            # Both CP arms receive age (main-effect head); interaction via β only
            # for age_temporal.
            return True
        return self.arm in (
            "age_only",
            "age_temporal",
            "age_temporal_per_head",
            "historical_age",
        )

    @property
    def has_time_input(self) -> bool:
        if self._uses_content_persistence:
            return True
        return self.arm in (
            "temporal_only",
            "age_temporal",
            "temporal_only_per_head",
            "age_temporal_per_head",
            "historical_age",
        )

    @property
    def model_card(self) -> dict[str, Any]:
        tp = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        card: dict[str, Any] = {
            "trainable_params": tp,
            "hidden_size": self._d_model,
            "arm": self.arm,
            "architecture": (
                "Content-Persistence DTR"
                if self._uses_content_persistence
                else "BenchmarkModel Transformer"
            ),
        }
        if self._uses_content_persistence:
            card.update(self._model.architecture_config())
            card["embedding_params"] = sum(
                p.numel() for p in self._model.encounter_encoder.parameters()
            )
        else:
            card["embedding_params"] = sum(
                p.numel() for p in self._model.code_emb.parameters()
            )
            card["layers"] = self._model.n_layers
            card["heads"] = self._model.n_heads
        return card

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._model.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def to(self, *args, **kwargs):
        self._model.to(*args, **kwargs)
        return self

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        if self._uses_content_persistence:
            logits = self._model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=batch["age"],
            )
        else:
            logits = self._model(
                code_ids=batch["code_ids"],
                type_ids=batch["type_ids"],
                tau=batch["tau"],
                padding_mask=batch["padding_mask"],
                is_query=batch["is_query"],
                age=batch["age"],
                lag_days=batch.get("lag_days"),
            )
        return ModelOutput(
            logits=logits,
            patient_repr=torch.zeros(
                logits.size(0), self._d_model, device=logits.device
            ),
        )

    def predict(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        self.eval()
        with torch.no_grad():
            return self.forward(batch)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.train()
        out = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(out.logits, batch["labels"])
        return {"loss": loss, "logits": out.logits}

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "checkpoint.pt")

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        state = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)

    @property
    def temporal(self):
        """TemporalGate-compatible facade (theta0 / beta / lambda_of)."""
        if self._uses_content_persistence:
            return self._model.gate
        return self._model.temporal

    def age_parameters(self):
        return self._model.age_parameters()

    def zero_all_betas_(self):
        return self._model.zero_all_betas_()

    def restore_betas_(self, saved):
        return self._model.restore_betas_(saved)
