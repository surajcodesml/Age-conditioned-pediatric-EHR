"""DTR adapter — wraps the existing BenchmarkModel to expose BaselineModel interface.

The DTR model is the existing synthetic_age_temporal.model.BenchmarkModel.
This adapter simply wraps it so it plugs into the common evaluation framework.
It does NOT re-implement DTR.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import register_baseline


@register_baseline("dtr")
class DTRAdapter(BaselineModel, nn.Module):
    """Wraps synthetic_age_temporal.model.BenchmarkModel."""

    def __init__(
        self,
        arm: str,
        n_codes: int,
        n_types: int,
        n_targets: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        # Import here to avoid circular dependencies at module level
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "synthetic_age_temporal"))
        from model import BenchmarkModel
        self._model = BenchmarkModel(
            arm=arm, n_codes=n_codes, n_types=n_types, n_targets=n_targets,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )
        self.arm = arm
        self._n_codes = n_codes
        self._n_targets = n_targets
        self._d_model = d_model

    @property
    def name(self) -> str:
        return f"dtr_{self.arm}"

    @property
    def has_age_input(self) -> bool:
        return self.arm in ("age_only", "age_temporal", "age_temporal_per_head", "historical_age")

    @property
    def has_time_input(self) -> bool:
        return self.arm in ("temporal_only", "age_temporal",
                            "temporal_only_per_head", "age_temporal_per_head", "historical_age")

    @property
    def model_card(self) -> dict[str, Any]:
        tp = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return {
            "trainable_params": tp,
            "embedding_params": sum(p.numel() for p in self._model.code_emb.parameters()),
            "layers": self._model.n_layers,
            "heads": self._model.n_heads,
            "hidden_size": self._d_model,
            "ffn_size": self._model.layers[0]["ff"][0].in_features if self._model.layers else "N/A",
            "max_seq_len": "N/A",
            "arm": self.arm,
        }

    def parameters(self, recurse=True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
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
            patient_repr=torch.zeros(logits.size(0), self._d_model, device=logits.device),
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
        """Access the temporal module for parameter recovery."""
        return self._model.temporal

    def age_parameters(self):
        return self._model.age_parameters()

    def zero_all_betas_(self):
        return self._model.zero_all_betas_()

    def restore_betas_(self, saved):
        return self._model.restore_betas_(saved)
