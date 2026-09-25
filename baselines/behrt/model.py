"""BEHRT — BERT for Electronic Health Records (Li et al., Scientific Reports 2020).

Key properties (faithful to the paper):
  - Code/disease token embedding
  - Learnable AGE embedding: one embedding per discretized age bracket
    (NOT DTR's continuous age function — this is crucial)
  - Positional encoding: standard absolute position
  - Alternating visit segment embedding (A/B/A/B)
  - Bidirectional Transformer
  - [CLS] → prediction head

Age brackets (pediatric + adult extended):
  0–1, 1–2, 2–5, 5–10, 10–15, 15–18, 18–30, 30–40, 40–50, 50–60, 60–70, 70–80, 80+

Reference: Li et al., "BEHRT: Transformer for Electronic Health Records",
Scientific Reports 2020. Official: deepmedicine/BEHRT
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


# BEHRT age buckets: discretized, not continuous
AGE_BUCKETS = [0, 1, 2, 5, 10, 15, 18, 30, 40, 50, 60, 70, 80]
N_AGE_BUCKETS = len(AGE_BUCKETS) + 1  # includes 80+


def age_to_bucket(age_years: float) -> int:
    """Map continuous age to BEHRT discrete bucket index."""
    for i, boundary in enumerate(AGE_BUCKETS):
        if age_years < boundary:
            return max(0, i - 1)
    return len(AGE_BUCKETS) - 1


def ages_to_buckets(ages: torch.Tensor) -> torch.Tensor:
    """Vectorized age bucketing. ages: [B] or [B, L]."""
    buckets = torch.zeros_like(ages, dtype=torch.long)
    for i, boundary in enumerate(AGE_BUCKETS):
        buckets = torch.where(ages >= float(boundary), i, buckets)
    return buckets


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:, :seq_len]


@register_baseline("behrt")
class BEHRTModel(BaselineModel, nn.Module):
    """BEHRT with learnable age embeddings per age bracket."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 288,
        n_layers: int = 6,
        n_heads: int = 12,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        nn.Module.__init__(self)
        if d_ff is None:
            d_ff = d_model * 4
        self.n_codes = n_codes
        self.n_targets = n_targets
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Collate: PAD=0, UNK=1, real=v+2; CLS = n_codes+2
        self.vocab_size = n_codes + 3
        self.cls_id = n_codes + 2

        # Embeddings: code + age + position + segment
        self.code_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.age_embedding = nn.Embedding(N_AGE_BUCKETS, d_model)  # BEHRT learnable age
        self.position_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        self.segment_embedding = nn.Embedding(2, d_model)  # alternating A/B

        self.embed_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.head = nn.Linear(d_model, n_targets)
        nn.init.zeros_(self.head.bias)

    @property
    def name(self) -> str:
        return "behrt"

    @property
    def has_age_input(self) -> bool:
        return True  # BEHRT explicitly uses age embeddings

    @property
    def has_time_input(self) -> bool:
        return False  # no explicit τ/elapsed-time

    @property
    def model_card(self) -> dict[str, Any]:
        tp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ep = sum(p.numel() for p in self.code_embedding.parameters())
        ep += sum(p.numel() for p in self.age_embedding.parameters())
        return {
            "trainable_params": tp,
            "embedding_params": ep,
            "layers": self.n_layers,
            "heads": self.n_heads,
            "hidden_size": self.d_model,
            "ffn_size": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "age_representation": "learnable_bucket_embedding",
            "n_age_buckets": N_AGE_BUCKETS,
        }

    def _prepare_input(self, batch: dict[str, torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Prepare BEHRT inputs: code_ids, age_bucket_ids, segment_ids, padding_mask.

        Prepends [CLS] token with age = prediction age.
        """
        code_ids = batch["code_ids"]  # [B, L]
        B, L = code_ids.shape
        device = code_ids.device

        pad_mask = batch.get("padding_mask", torch.zeros(B, L, dtype=torch.bool, device=device))
        is_query = batch.get("is_query", torch.zeros(B, L, dtype=torch.bool, device=device))
        age = batch["age"]  # [B] prediction age

        # Mask query tokens
        effective_mask = pad_mask | is_query

        # Age buckets: use prediction age for all tokens in the event sequence
        # (BEHRT assigns encounter-level age; in synthetic benchmark all events
        #  share the patient's prediction age)
        age_bucket_all = ages_to_buckets(age)  # [B]
        age_ids = age_bucket_all.unsqueeze(1).expand(B, L)  # [B, L]

        # Prepend CLS
        cls_ids = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        input_ids = torch.cat([cls_ids, code_ids], dim=1)
        cls_age = age_bucket_all.unsqueeze(1)
        age_ids_full = torch.cat([cls_age, age_ids], dim=1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        key_pad_mask = torch.cat([cls_mask, effective_mask], dim=1)

        # Segment IDs: alternating 0/1 per encounter
        # In synthetic benchmark: simple uniform 0 (no explicit encounter boundaries in flat tokens)
        seg_ids = torch.zeros(B, L + 1, dtype=torch.long, device=device)

        return input_ids, age_ids_full, seg_ids, key_pad_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        input_ids, age_ids, seg_ids, key_pad_mask = self._prepare_input(batch)
        B, L_plus = input_ids.shape

        code_emb = self.code_embedding(input_ids.clamp(0, self.vocab_size - 1))
        age_emb = self.age_embedding(age_ids)
        pos_emb = self.position_embedding(L_plus)
        seg_emb = self.segment_embedding(seg_ids)

        emb = self.embed_norm(code_emb + age_emb + pos_emb + seg_emb)
        emb = self.embed_dropout(emb)

        hidden = self.encoder(emb, src_key_padding_mask=key_pad_mask)

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
