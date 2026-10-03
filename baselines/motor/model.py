"""MOTOR — A Time-to-Event Foundation Model for Structured Medical Records.

Reference:
  Steinberg et al., "MOTOR: A Time-to-Event Foundation Model for Structured
  Medical Records", ICLR 2024.
  Official code: https://github.com/som-shahlab/motor_code_release / FEMR

Key properties faithful to the paper and official FEMR implementation:
  - Continuous time / age encoded via Rotary Positional Embedding (RoPE)
    with inv_freq = 1 / (10000 ** linspace(0, 2, dim // 2))
  - Pre-RMSNorm transformer architecture with GELU activation
  - Embeddings: code tokens + continuous age projection
  - Supports both:
    1. Defining time-to-event / piecewise exponential survival hazard pretraining
    2. Downstream task-matched prediction head with BCEWithLogitsLoss
  - Contract compliance: BaselineModel ABC, predict(), training_step(), save/load.
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


class MotorRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as in official FEMR/MOTOR."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.scale


def compute_motor_rope(
    ages: torch.Tensor,
    dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fixed rotary position embeddings from continuous ages/timestamps.

    Matches official FEMR fixed_pos_embedding:
        inv_freq = 1.0 / (10000 ** (linspace(0, 2, num=dim // 2)))
        t = inv_freq * ages
        sin, cos = sin(t), cos(t)
    """
    device = ages.device
    half_dim = dim // 2
    inv_freq = 1.0 / (10000.0 ** torch.linspace(0.0, 2.0, steps=half_dim, dtype=torch.float32, device=device))
    # ages: [B, L]
    t = ages.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)  # [B, L, half_dim]
    sin_val = torch.sin(t).to(dtype)
    cos_val = torch.cos(t).to(dtype)
    sin = torch.cat([sin_val, sin_val], dim=-1)  # [B, L, dim]
    cos = torch.cat([cos_val, cos_val], dim=-1)  # [B, L, dim]
    return sin, cos


def apply_motor_rotary_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensor.

    x: [B, n_heads, L, head_dim]
    sin, cos: [B, 1, L, head_dim]
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


class MotorAttention(nn.Module):
    """Multi-head self-attention with continuous age RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        sin_b = sin.unsqueeze(1)
        cos_b = cos.unsqueeze(1)
        q = apply_motor_rotary_emb(q, sin_b, cos_b)
        k = apply_motor_rotary_emb(k, sin_b, cos_b)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, n_heads, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class MotorTransformerBlock(nn.Module):
    """Transformer block with Pre-RMSNorm and GELU FFN as in FEMR/MOTOR."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = MotorRMSNorm(d_model)
        self.attn = MotorAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = MotorRMSNorm(d_model)
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
        sin: torch.Tensor,
        cos: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.norm1(x), sin, cos, key_padding_mask=key_padding_mask)
        out = h + self.ffn(self.norm2(h))
        return out


@register_baseline("motor")
class MOTORModel(BaselineModel, nn.Module):
    """MOTOR baseline implementing continuous time-to-event representation learning."""

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
        time_to_event_pretrain: bool = False,
        n_time_bins: int = 8,
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
        self.time_to_event_pretrain = time_to_event_pretrain
        self.n_time_bins = n_time_bins

        self.vocab_size = n_codes + 3
        self.cls_id = n_codes + 2

        self.code_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        self.age_projection = nn.Linear(1, d_model)

        self.embed_norm = MotorRMSNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MotorTransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = MotorRMSNorm(d_model)

        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.head = nn.Linear(d_model, n_targets)
        nn.init.zeros_(self.head.bias)

        if time_to_event_pretrain:
            self.hazard_proj = nn.Linear(d_model, n_time_bins * n_targets)
            self.register_buffer(
                "time_bins",
                torch.tensor([1.0, 7.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]),
            )

    @property
    def name(self) -> str:
        return "motor"

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
            "model": "motor",
            "paper": "Steinberg et al., ICLR 2024",
            "official_repo": "som-shahlab/motor_code_release",
            "trainable_params": tp,
            "embedding_params": ep,
            "layers": self.n_layers,
            "heads": self.n_heads,
            "hidden_size": self.d_model,
            "ffn_size": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "temporal_mechanism": "continuous_rope_on_event_ages",
            "normalization": "RMSNorm",
        }

    def _prepare_inputs(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if "age_years" in batch and torch.is_tensor(batch["age_years"]) and batch["age_years"].ndim == 2:
            event_ages = batch["age_years"].float()
            pred_age = event_ages[:, -1]
        elif "age" in batch and "lag_days" in batch:
            pred_age = batch["age"].float()
            event_ages = pred_age.unsqueeze(1) - (batch["lag_days"].float() / 365.25)
        elif "age" in batch and "timestamps_days" in batch:
            pred_age = batch["age"].float()
            ts = batch["timestamps_days"].float()
            event_ages = ts / 365.25
        elif "tau" in batch and "age" in batch:
            pred_age = batch["age"].float()
            lag_days = 7.0 * torch.expm1(batch["tau"].float().clamp(max=20.0))
            event_ages = pred_age.unsqueeze(1) - (lag_days / 365.25)
        else:
            pred_age = batch.get("age", torch.zeros(B, device=device)).float()
            event_ages = pred_age.unsqueeze(1).expand(B, L)

        cls_tokens = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        input_ids = torch.cat([cls_tokens, code_ids], dim=1)

        cls_age = pred_age.unsqueeze(1)
        full_ages = torch.cat([cls_age, event_ages], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, effective_mask], dim=1)

        return input_ids, full_ages, full_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        input_ids, full_ages, full_mask = self._prepare_inputs(batch)
        B, L_plus = input_ids.shape

        tok_emb = self.code_embedding(input_ids.clamp(0, self.vocab_size - 1))
        norm_ages = (full_ages / 50.0).unsqueeze(-1)
        age_emb = self.age_projection(norm_ages)

        x = self.embed_norm(tok_emb + age_emb)
        x = self.embed_dropout(x)

        head_dim = self.d_model // self.n_heads
        sin, cos = compute_motor_rope(full_ages, head_dim, dtype=x.dtype)

        for block in self.blocks:
            x = block(x, sin, cos, key_padding_mask=full_mask)

        hidden = self.out_norm(x)

        cls_repr = hidden[:, 0]
        pooled = self.pooler(cls_repr)
        logits = self.head(pooled)

        return ModelOutput(
            logits=logits,
            patient_repr=pooled,
            token_reprs=hidden,
        )

    def predict(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        self.eval()
        with torch.no_grad():
            return self.forward(batch)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.train()
        out = self.forward(batch)
        labels = batch["labels"].float()

        if self.time_to_event_pretrain and "event_times" in batch and "is_censor" in batch:
            B = out.patient_repr.shape[0]
            hazards = self.hazard_proj(out.patient_repr).view(B, self.n_time_bins, self.n_targets)
            times = batch["event_times"].unsqueeze(-1).unsqueeze(-1)
            bins = self.time_bins.unsqueeze(0).unsqueeze(-1)
            time_in_bin = torch.clamp(times - bins, min=0.0)
            is_event = (~batch["is_censor"]).float().unsqueeze(1).unsqueeze(-1)
            event_loss = -(hazards * is_event).sum() / (B * self.n_targets + 1e-6)
            surv_loss = torch.exp(hazards + torch.log(time_in_bin + 1e-6)).sum() / (B * self.n_targets + 1e-6)
            loss = event_loss + surv_loss
        else:
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
