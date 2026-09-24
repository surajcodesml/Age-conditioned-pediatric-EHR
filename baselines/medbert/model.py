"""Med-BERT — Pretrained Embeddings on Structured EHR (Rasmy et al., npj Dig Med 2021).

Key properties (canonical, faithful to paper):
  - Code embedding
  - Visit embedding: learned embedding for visit index (which visit in sequence)
  - Serialization embedding: position-within-visit ordering
  - NO explicit age embedding (canonical Med-BERT)
  - NO [CLS]/[SEP] in canonical formulation
  - Bidirectional Transformer
  - Mean-pooling over valid tokens → prediction head

Paper architecture:
  6 layers, 6 attention heads, hidden=192, dropout=0.1, max_seq_len=512

Do NOT add age/time to make Med-BERT more competitive.

Reference: Rasmy et al., "Med-BERT: pretrained contextualized embeddings on
large-scale structured electronic health records for disease prediction",
npj Digital Medicine 2021. Official: ZhiGroup/Med-BERT
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


@register_baseline("medbert")
class MedBERTModel(BaselineModel, nn.Module):
    """Canonical Med-BERT with visit and serialization embeddings, no age."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 6,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        max_visits: int = 128,
        max_serial: int = 64,
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
        self.max_visits = max_visits
        self.max_serial = max_serial

        # Embeddings: code + visit index + within-visit serialization
        self.code_embedding = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.visit_embedding = nn.Embedding(max_visits, d_model)
        self.serial_embedding = nn.Embedding(max_serial, d_model)

        self.embed_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Med-BERT: mean-pool (no [CLS]) → head
        self.head = nn.Linear(d_model, n_targets)
        nn.init.zeros_(self.head.bias)

    @property
    def name(self) -> str:
        return "medbert"

    @property
    def has_age_input(self) -> bool:
        return False  # canonical Med-BERT has no age

    @property
    def has_time_input(self) -> bool:
        return False  # no explicit τ/elapsed-time

    @property
    def model_card(self) -> dict[str, Any]:
        tp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ep = sum(p.numel() for p in self.code_embedding.parameters())
        ep += sum(p.numel() for p in self.visit_embedding.parameters())
        ep += sum(p.numel() for p in self.serial_embedding.parameters())
        return {
            "trainable_params": tp,
            "embedding_params": ep,
            "layers": self.n_layers,
            "heads": self.n_heads,
            "hidden_size": self.d_model,
            "ffn_size": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "age_representation": "none",
            "time_representation": "none",
            "pooling": "mean_pool",
        }

    def _prepare_input(self, batch: dict[str, torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Prepare Med-BERT inputs: code + visit index + serial index.

        In the synthetic benchmark (flat event sequence), we assign:
          visit_id = running index (each event = micro-visit)
          serial_id = 0 (single code per micro-visit)
        For real EHR, this should be derived from encounter boundaries.
        """
        code_ids = batch["code_ids"]  # [B, L]
        B, L = code_ids.shape
        device = code_ids.device

        pad_mask = batch.get("padding_mask", torch.zeros(B, L, dtype=torch.bool, device=device))
        is_query = batch.get("is_query", torch.zeros(B, L, dtype=torch.bool, device=device))
        effective_mask = pad_mask | is_query

        # Visit IDs: running position (clamped to max_visits)
        visit_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        visit_ids = visit_ids.clamp(max=self.max_visits - 1)

        # Serialization IDs: 0 for all (single code per position in flat sequence)
        serial_ids = torch.zeros(B, L, dtype=torch.long, device=device)

        return code_ids, visit_ids, serial_ids, effective_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        code_ids, visit_ids, serial_ids, key_pad_mask = self._prepare_input(batch)
        B, L = code_ids.shape

        code_emb = self.code_embedding(code_ids)
        visit_emb = self.visit_embedding(visit_ids)
        serial_emb = self.serial_embedding(serial_ids)

        emb = self.embed_norm(code_emb + visit_emb + serial_emb)
        emb = self.embed_dropout(emb)

        hidden = self.encoder(emb, src_key_padding_mask=key_pad_mask)

        # Mean-pool over valid tokens
        valid = (~key_pad_mask).to(hidden.dtype).unsqueeze(-1)  # [B, L, 1]
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)  # [B, d]

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
