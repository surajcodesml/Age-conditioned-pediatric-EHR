"""TALE-EHR — Time-Aware Language Encoder for Electronic Health Records.

Reference:
  Yu et al., "Time-Aware Attention for Enhanced Electronic Health Records Modeling",
  arXiv:2507.14847, July 2025.

Key architectural properties faithful to the paper:
  - Continuous time-aware self-attention mechanism (Section 3.1, Eq. 4-5):
      scores[j, k] = (Q_j K_k^T / sqrt(d)) + log(w(|t_j - t_k|) + eps)
      where w(Delta t) = sigma(sum_{l=0}^5 a_l * (Delta t)^l) is a learnable
      order-5 polynomial weighting function.
  - Multi-scale hierarchical history representation (Section 3.2, Eq. 6-7):
      h_t = sum_j alpha_j(t) * V_j
      with query Q_base identifying clinically significant events modulated
      by temporal proximity.
  - Semantic code representation via embeddings, projected into task-specific
    query, key, value spaces via MLPs.
  - Downstream classification head (MLP) for multi-label next-visit prediction
    and disease forecasting.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import register_baseline


class PolynomialTimeWeighting(nn.Module):
    """Order-5 polynomial continuous temporal weighting function w(Delta t).

    From TALE-EHR paper Eq. 5:
        w(t) = sigma(sum_{l=0}^S a_l * t^l)
    with S=5 and sigma being sigmoid ensuring weights in (0, 1).
    """

    def __init__(self, order: int = 5) -> None:
        super().__init__()
        self.order = order
        init_coeffs = torch.zeros(order + 1)
        init_coeffs[0] = 1.0
        init_coeffs[1] = -0.5
        self.coeffs = nn.Parameter(init_coeffs)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Evaluate polynomial weight for time differences delta_t.

        delta_t: arbitrary shape >= 0
        """
        norm_t = torch.log1p(torch.clamp(delta_t, min=0.0) / 7.0)

        poly_val = self.coeffs[0]
        t_power = norm_t
        for l in range(1, self.order + 1):
            poly_val = poly_val + self.coeffs[l] * t_power
            if l < self.order:
                t_power = t_power * norm_t

        return torch.sigmoid(poly_val)


class TimeAwareMultiheadAttention(nn.Module):
    """Multi-head attention with continuous polynomial temporal weighting."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, poly_order: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")

        self.q_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.k_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.time_weighting = PolynomialTimeWeighting(order=poly_order)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        timestamps_days: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_mlp(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_mlp(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_mlp(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        delta_t = torch.abs(timestamps_days.unsqueeze(2) - timestamps_days.unsqueeze(1))
        w_t = self.time_weighting(delta_t).unsqueeze(1)  # [B, 1, L, L]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + torch.log(w_t + 1e-6)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class TALEEHRBlock(nn.Module):
    """TALE-EHR Transformer layer with Pre-LayerNorm and Time-Aware Attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, poly_order: int = 5) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TimeAwareMultiheadAttention(d_model, n_heads, dropout=dropout, poly_order=poly_order)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps_days: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.norm1(x), timestamps_days, key_padding_mask=key_padding_mask)
        out = h + self.ffn(self.norm2(h))
        return out


@register_baseline("tale_ehr")
class TALEEHRModel(BaselineModel, nn.Module):
    """TALE-EHR baseline with continuous time-aware attention and hierarchical history pooling."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        poly_order: int = 5,
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
        self.poly_order = poly_order

        self.vocab_size = n_codes + 3
        self.cls_id = n_codes + 2

        self.code_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TALEEHRBlock(d_model, n_heads, d_ff, dropout=dropout, poly_order=poly_order)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

        self.q_base = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.history_time_weight = PolynomialTimeWeighting(order=poly_order)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )
        nn.init.zeros_(self.head[-1].bias)

    @property
    def name(self) -> str:
        return "tale_ehr"

    @property
    def has_age_input(self) -> bool:
        return True

    @property
    def has_time_input(self) -> bool:
        return True

    @property
    def model_card(self) -> dict[str, Any]:
        tp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ep = sum(p.numel() for p in self.code_embedding.parameters())
        return {
            "model": "tale_ehr",
            "paper": "Yu et al., arXiv:2507.14847 (2025)",
            "trainable_params": tp,
            "embedding_params": ep,
            "layers": self.n_layers,
            "heads": self.n_heads,
            "hidden_size": self.d_model,
            "ffn_size": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "temporal_mechanism": f"continuous_poly{self.poly_order}_attention",
        }

    def _prepare_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        code_ids = batch.get("code_ids", batch.get("code_indices"))
        B, L = code_ids.shape
        device = code_ids.device

        if "padding_mask" in batch:
            pad_mask = batch["padding_mask"].bool()
        elif "attention_mask" in batch:
            pad_mask = ~batch["attention_mask"].bool()
        else:
            pad_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        is_query = batch.get("is_query", torch.zeros(B, L, dtype=torch.bool, device=device))
        effective_mask = pad_mask | is_query

        if "lag_days" in batch:
            lag_days = batch["lag_days"].float()
            timestamps = -lag_days
            pred_time = torch.zeros(B, device=device)
        elif "timestamps_days" in batch:
            timestamps = batch["timestamps_days"].float()
            pred_time = timestamps[:, -1]
        elif "tau" in batch:
            lag_days = 7.0 * torch.expm1(batch["tau"].float().clamp(max=20.0))
            timestamps = -lag_days
            pred_time = torch.zeros(B, device=device)
        else:
            timestamps = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(0).expand(B, L)
            pred_time = timestamps[:, -1]

        cls_tokens = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        input_ids = torch.cat([cls_tokens, code_ids], dim=1)

        cls_time = pred_time.unsqueeze(1)
        full_timestamps = torch.cat([cls_time, timestamps], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, effective_mask], dim=1)

        return input_ids, full_timestamps, pred_time, full_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        input_ids, full_timestamps, pred_time, full_mask = self._prepare_inputs(batch)
        B, L_plus = input_ids.shape

        tok_emb = self.code_embedding(input_ids.clamp(0, self.vocab_size - 1))
        x = self.embed_norm(tok_emb)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x, full_timestamps, key_padding_mask=full_mask)

        hidden = self.out_norm(x)

        q_b = self.q_base.expand(B, 1, self.d_model)
        scores = torch.matmul(q_b, hidden.transpose(-2, -1)) / math.sqrt(self.d_model)

        delta_to_pred = torch.abs(pred_time.unsqueeze(1) - full_timestamps)
        w_hist = self.history_time_weight(delta_to_pred).unsqueeze(1)
        scores = scores + torch.log(w_hist + 1e-6)

        if full_mask is not None:
            scores = scores.masked_fill(full_mask.unsqueeze(1), -1e9)

        alpha = F.softmax(scores, dim=-1)
        h_t = torch.matmul(alpha, hidden).squeeze(1)

        logits = self.head(h_t)

        return ModelOutput(
            logits=logits,
            patient_repr=h_t,
            token_reprs=hidden,
            attention=alpha.squeeze(1),
        )

    def predict(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        self.eval()
        with torch.no_grad():
            return self.forward(batch)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.train()
        out = self.forward(batch)
        labels = batch["labels"].float()
        loss = F.binary_cross_entropy_with_logits(out.logits, labels)
        return {"loss": loss, "logits": out.logits}

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "checkpoint.pt")
        with (path / "config.json").open("w") as f:
            json.dump(self.model_card, f, indent=2)

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        if path.is_file():
            ckpt = path
        elif (path / "best_checkpoint.pt").exists():
            ckpt = path / "best_checkpoint.pt"
        else:
            ckpt = path / "checkpoint.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
