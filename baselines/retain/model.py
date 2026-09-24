"""RETAIN — Reverse Time Attention (Choi et al., NeurIPS 2016).

Canonical architecture:
  - Visit-level multi-hot → embedding layer
  - GRU_α (reverse time) → scalar visit attention α_t
  - GRU_β (reverse time) → vector variable attention β_t
  - context = Σ_t α_t (β_t ⊙ v_t)
  - output = W_out · context

Key properties:
  - Reverse-time recurrent attention (NOT a Transformer)
  - Visit-level, not token-level
  - No explicit age or time embedding (demographics can enter final predictor)

Reference: Choi et al., "RETAIN: An Interpretable Predictive Model for Healthcare
using Reverse Time Attention Mechanism", NeurIPS 2016.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import register_baseline


@register_baseline("retain")
class RETAINModel(BaselineModel, nn.Module):
    """Canonical RETAIN with reverse-time GRU attention."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_emb: int = 128,
        d_rnn: int = 128,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self.n_codes = n_codes
        self.n_targets = n_targets
        self.d_emb = d_emb
        self.d_rnn = d_rnn

        # Code embedding: multi-hot → dense
        self.code_embedding = nn.Linear(n_codes, d_emb)

        # Reverse-time GRU for α (scalar visit attention)
        self.gru_alpha = nn.GRU(
            input_size=d_emb, hidden_size=d_rnn,
            batch_first=True, bidirectional=False,
        )
        self.alpha_fc = nn.Linear(d_rnn, 1)

        # Reverse-time GRU for β (variable-level vector attention)
        self.gru_beta = nn.GRU(
            input_size=d_emb, hidden_size=d_rnn,
            batch_first=True, bidirectional=False,
        )
        self.beta_fc = nn.Linear(d_rnn, d_emb)

        self.dropout = nn.Dropout(dropout)

        # Output head
        self.head = nn.Linear(d_emb, n_targets)
        nn.init.zeros_(self.head.bias)

    @property
    def name(self) -> str:
        return "retain"

    @property
    def has_age_input(self) -> bool:
        return False

    @property
    def has_time_input(self) -> bool:
        return False

    @property
    def model_card(self) -> dict[str, Any]:
        tp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "trainable_params": tp,
            "embedding_params": sum(p.numel() for p in self.code_embedding.parameters()),
            "layers": "2 GRUs",
            "heads": "N/A",
            "hidden_size": self.d_rnn,
            "ffn_size": "N/A",
            "max_seq_len": "unlimited (sequential)",
            "d_emb": self.d_emb,
        }

    def _build_visit_embeddings(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Convert event-level batch to visit-level multi-hot embeddings.

        For the synthetic benchmark we group by is_query and encounter-level structure.
        For simplicity, each event is treated as its own "micro-visit" — this is
        equivalent to event-level RETAIN but respects the canonical interface.

        Returns (visit_emb [B, T, d_emb], mask [B, T], T).
        """
        code_ids = batch["code_ids"]  # [B, L]
        B, L = code_ids.shape
        pad_mask = batch.get("padding_mask", torch.zeros(B, L, dtype=torch.bool))
        is_query = batch.get("is_query", torch.zeros(B, L, dtype=torch.bool))

        # Build multi-hot per event (treating each event as a micro-visit)
        device = code_ids.device
        multi_hot = torch.zeros(B, L, self.n_codes, device=device, dtype=torch.float32)
        for i in range(B):
            for j in range(L):
                if not pad_mask[i, j] and not is_query[i, j]:
                    cid = int(code_ids[i, j])
                    if 0 < cid < self.n_codes:
                        multi_hot[i, j, cid] = 1.0

        # valid_mask: True where there is a real event (not pad, not query)
        valid_mask = (~pad_mask) & (~is_query)  # [B, L]

        # Embed
        visit_emb = self.code_embedding(multi_hot)  # [B, L, d_emb]
        return visit_emb, valid_mask, L

    def _reverse_time_pass(
        self, visit_emb: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run reverse-time GRUs for α and β.

        Args:
            visit_emb: [B, T, d_emb] visit embeddings (chronological order)
            mask: [B, T] True = valid visit

        Returns:
            alpha: [B, T] scalar visit attention (softmaxed)
            beta:  [B, T, d_emb] vector variable attention (tanh)
            context: [B, d_emb] weighted sum
        """
        B, T, D = visit_emb.shape

        # Reverse the sequence (RETAIN processes history from most recent to oldest)
        # We flip along the time dimension, then unflip after GRU
        lengths = mask.sum(dim=1).long().clamp(min=1)
        # Reverse the visit embeddings for valid entries
        reversed_emb = torch.zeros_like(visit_emb)
        for i in range(B):
            n = int(lengths[i])
            if n > 0:
                reversed_emb[i, :n] = visit_emb[i, :n].flip(0)

        # Pack padded sequences for efficient GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            reversed_emb, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )

        # Alpha pathway
        h_alpha, _ = self.gru_alpha(packed)
        h_alpha, _ = nn.utils.rnn.pad_packed_sequence(h_alpha, batch_first=True, total_length=T)

        # Beta pathway
        h_beta, _ = self.gru_beta(packed)
        h_beta, _ = nn.utils.rnn.pad_packed_sequence(h_beta, batch_first=True, total_length=T)

        # Un-reverse to align with original time order
        h_alpha_fwd = torch.zeros_like(h_alpha)
        h_beta_fwd = torch.zeros_like(h_beta)
        for i in range(B):
            n = int(lengths[i])
            if n > 0:
                h_alpha_fwd[i, :n] = h_alpha[i, :n].flip(0)
                h_beta_fwd[i, :n] = h_beta[i, :n].flip(0)

        # Scalar attention α
        alpha_logits = self.alpha_fc(h_alpha_fwd).squeeze(-1)  # [B, T]
        alpha_logits = alpha_logits.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(alpha_logits, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Variable attention β
        beta = torch.tanh(self.beta_fc(h_beta_fwd))  # [B, T, d_emb]

        # Context: weighted sum of β ⊙ v
        context = (alpha.unsqueeze(-1) * beta * visit_emb).sum(dim=1)  # [B, d_emb]

        return alpha, beta, context

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        visit_emb, mask, T = self._build_visit_embeddings(batch)
        alpha, beta, context = self._reverse_time_pass(visit_emb, mask)
        context = self.dropout(context)
        logits = self.head(context)
        return ModelOutput(
            logits=logits,
            patient_repr=context.detach(),
            attention=alpha.detach(),
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
        torch.save(self.state_dict(), path / "checkpoint.pt")
        with (path / "config.json").open("w") as f:
            json.dump({
                "n_codes": self.n_codes, "n_targets": self.n_targets,
                "d_emb": self.d_emb, "d_rnn": self.d_rnn,
            }, f)

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        state = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.load_state_dict(state)
