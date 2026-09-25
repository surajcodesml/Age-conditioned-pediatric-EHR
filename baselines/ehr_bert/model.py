"""Vanilla EHR-BERT — architectural control.

Properties:
  - Code embedding (shared vocabulary)
  - Standard sinusoidal positional embedding (absolute position)
  - Segment IDs: alternating per encounter to mark visit boundaries
  - [CLS] token prepended → prediction from [CLS] representation
  - Bidirectional Transformer encoder
  - NO age embedding, NO time/elapsed-time embedding

This is an architectural control, not a claim of a specific published model.
It isolates what a standard BERT architecture can learn from code sequences
alone, without any temporal or age signal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import register_baseline


class SinusoidalPositionalEmbedding(nn.Module):
    """Standard sinusoidal position embedding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len]


@register_baseline("ehr_bert")
class EHRBertModel(BaselineModel, nn.Module):
    """Vanilla BERT for EHR codes — no age, no time."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        nn.Module.__init__(self)
        self.n_codes = n_codes
        self.n_targets = n_targets
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Collate uses PAD=0, UNK=1, real=v+2 → ids in [0, n_codes+1].
        # CLS occupies the next free id.
        self.vocab_size = n_codes + 3  # PAD/UNK/codes + [CLS]
        self.cls_id = n_codes + 2
        self.code_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.position_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        self.segment_embedding = nn.Embedding(2, d_model)  # alternating A/B

        self.embed_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # [CLS] → pooler → prediction head
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.head = nn.Linear(d_model, n_targets)
        nn.init.zeros_(self.head.bias)

    @property
    def name(self) -> str:
        return "ehr_bert"

    @property
    def has_age_input(self) -> bool:
        return False

    @property
    def has_time_input(self) -> bool:
        return False

    @property
    def model_card(self) -> dict[str, Any]:
        tp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ep = sum(p.numel() for p in self.code_embedding.parameters())
        return {
            "trainable_params": tp,
            "embedding_params": ep,
            "layers": self.n_layers,
            "heads": self.n_heads,
            "hidden_size": self.d_model,
            "ffn_size": self.d_ff,
            "max_seq_len": self.max_seq_len,
        }

    def _prepare_input(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend [CLS] and build segment IDs.

        Returns (input_ids [B, L+1], src_key_padding_mask [B, L+1]).
        """
        code_ids = batch["code_ids"]  # [B, L]
        B, L = code_ids.shape
        device = code_ids.device

        pad_mask = batch.get("padding_mask", torch.zeros(B, L, dtype=torch.bool, device=device))
        is_query = batch.get("is_query", torch.zeros(B, L, dtype=torch.bool, device=device))

        # Mask query tokens (make them pads for vanilla BERT)
        effective_mask = pad_mask | is_query

        # Prepend CLS token
        cls_ids = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        input_ids = torch.cat([cls_ids, code_ids], dim=1)  # [B, L+1]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        key_pad_mask = torch.cat([cls_mask, effective_mask], dim=1)  # [B, L+1]

        return input_ids, key_pad_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        input_ids, key_pad_mask = self._prepare_input(batch)
        B, L_plus = input_ids.shape

        # Embeddings: code + position + segment (alternating)
        code_emb = self.code_embedding(input_ids.clamp(0, self.vocab_size - 1))
        pos_emb = self.position_embedding(L_plus)
        seg_ids = torch.zeros(B, L_plus, dtype=torch.long, device=input_ids.device)
        seg_emb = self.segment_embedding(seg_ids)  # uniform segment for vanilla

        emb = self.embed_norm(code_emb + pos_emb + seg_emb)
        emb = self.embed_dropout(emb)

        hidden = self.encoder(emb, src_key_padding_mask=key_pad_mask)

        # [CLS] representation
        cls_repr = hidden[:, 0]
        pooled = self.pooler(cls_repr)
        logits = self.head(pooled)

        return ModelOutput(
            logits=logits,
            patient_repr=pooled.detach(),
            token_reprs=hidden.detach(),
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
            json.dump(self.model_card, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        state = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.load_state_dict(state)
