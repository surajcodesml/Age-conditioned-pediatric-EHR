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
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import register_baseline


def _reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse the valid (left) prefix of a right-padded sequence batch.

    Args:
        x: [B, T, D]
        lengths: [B] valid lengths
    """
    B, T, D = x.shape
    idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
    gather_idx = (lengths.unsqueeze(1) - 1 - idx).clamp(min=0)
    out = x.gather(1, gather_idx.unsqueeze(-1).expand(B, T, D))
    valid = (idx < lengths.unsqueeze(1)).unsqueeze(-1).to(dtype=x.dtype)
    return out * valid


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

        # Vocab: PAD=0, UNK=1, real codes = v+2 (model_new.data collate).
        # Event-level "micro-visits" are single-code, so Embedding is equivalent
        # to Linear(multi-hot) without building a B×L×|V| dense tensor.
        self.vocab_size = n_codes + 2
        self.code_embedding = nn.Embedding(self.vocab_size, d_emb, padding_idx=0)

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
            "vocab_size": self.vocab_size,
        }

    def _build_visit_embeddings(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Convert event-level batch to visit-level embeddings (vectorized).

        Each event is treated as its own micro-visit (single code). Equivalent to
        one-hot → Linear, but O(B·L·d) instead of O(B·L·|V|).

        Returns (visit_emb [B, T, d_emb], mask [B, T], T).
        """
        code_ids = batch["code_ids"]  # [B, L]
        B, L = code_ids.shape
        device = code_ids.device
        pad_mask = batch.get(
            "padding_mask", torch.zeros(B, L, dtype=torch.bool, device=device)
        )
        is_query = batch.get(
            "is_query", torch.zeros(B, L, dtype=torch.bool, device=device)
        )
        valid_mask = (~pad_mask) & (~is_query)  # [B, L]

        # Clamp to vocab; zero out pads/queries/invalid ids after lookup.
        safe_ids = code_ids.clamp(0, self.vocab_size - 1)
        visit_emb = self.code_embedding(safe_ids)
        invalid = (~valid_mask) | (code_ids <= 0) | (code_ids >= self.vocab_size)
        visit_emb = visit_emb.masked_fill(invalid.unsqueeze(-1), 0.0)
        return visit_emb, valid_mask, L

    def _reverse_time_pass(
        self, visit_emb: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run reverse-time GRUs for α and β (fully vectorized reverse)."""
        B, T, _D = visit_emb.shape
        lengths = mask.sum(dim=1).long().clamp(min=1)

        reversed_emb = _reverse_padded(visit_emb, lengths)

        packed = nn.utils.rnn.pack_padded_sequence(
            reversed_emb, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )

        h_alpha, _ = self.gru_alpha(packed)
        h_alpha, _ = nn.utils.rnn.pad_packed_sequence(
            h_alpha, batch_first=True, total_length=T
        )

        h_beta, _ = self.gru_beta(packed)
        h_beta, _ = nn.utils.rnn.pad_packed_sequence(
            h_beta, batch_first=True, total_length=T
        )

        h_alpha_fwd = _reverse_padded(h_alpha, lengths)
        h_beta_fwd = _reverse_padded(h_beta, lengths)

        alpha_logits = self.alpha_fc(h_alpha_fwd).squeeze(-1)  # [B, T]
        alpha_logits = alpha_logits.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(alpha_logits, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        beta = torch.tanh(self.beta_fc(h_beta_fwd))  # [B, T, d_emb]
        context = (alpha.unsqueeze(-1) * beta * visit_emb).sum(dim=1)  # [B, d_emb]

        return alpha, beta, context

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        visit_emb, mask, _T = self._build_visit_embeddings(batch)
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
                "vocab_size": self.vocab_size,
            }, f)

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        state = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.load_state_dict(state)
