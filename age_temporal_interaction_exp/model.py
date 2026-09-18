"""Minimal Transformer with optional linear age × lag attention bias.

Primary diagnostic arm
    s_ij = q_i^T k_j / sqrt(d) - (λ0 + β z(a)) τ_j

Age enters the interaction arm only through this bias. No Fourier features,
Chebyshev basis, softplus, or auxiliary loss.

Initialization is polarity-aligned so POS/NEG cannot silently flip the
learned interaction sign: POS = +e_0, NEG = -e_0, head reads dimension 0.
Content Q/K start small so the temporal bias can dominate early training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import ARMS, TAU_WEEK_SCALE, tau_max


class InteractionModel(nn.Module):
    def __init__(
        self,
        arm: str,
        n_codes: int,
        n_types: int,
        d_model: int = 64,
        n_layers: int = 1,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
        head_hidden: int = 32,
        pos_id: int | None = None,
        neg_id: int | None = None,
        query_id: int | None = None,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_layers != 1:
            raise ValueError("This diagnostic model is intentionally 1-layer.")
        self.arm = arm
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.tau_max_value = tau_max()
        self.register_buffer("tau_max", torch.tensor(self.tau_max_value, dtype=torch.float32))

        self.code_embedding = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(n_types, d_model, padding_idx=0)
        self.time_projection = nn.Linear(1, d_model)
        nn.init.zeros_(self.time_projection.weight)
        nn.init.zeros_(self.time_projection.bias)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        nn.init.zeros_(self.ff[-2].weight)
        nn.init.zeros_(self.ff[-2].bias)
        nn.init.eye_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)
        with torch.no_grad():
            self.qkv.weight.zero_()
            self.qkv.bias.zero_()
            d = d_model
            self.qkv.weight[2 * d :, :] = torch.eye(d)
            self.qkv.weight[: 2 * d].normal_(0.0, 0.02)

        self.lambda0 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        # Linear head; dim 0 is the polarity axis. Extra unit is late age.
        self.head = nn.Linear(d_model + 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.weight.data[0, 0] = 1.0
        _ = head_hidden

        if pos_id is not None and neg_id is not None:
            with torch.no_grad():
                self.code_embedding.weight.zero_()
                self.type_embedding.weight.zero_()
                self.code_embedding.weight[pos_id, 0] = 1.0
                self.code_embedding.weight[neg_id, 0] = -1.0
                if query_id is not None:
                    self.code_embedding.weight[query_id].zero_()

        self._cache: dict[str, torch.Tensor] | None = None

    def encode_events(
        self,
        code_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_norm: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.code_embedding(code_ids)
            + self.type_embedding(type_ids)
            + self.time_projection(time_norm.unsqueeze(-1))
        )

    def temporal_slope(self, z_age: torch.Tensor) -> torch.Tensor:
        """λ(a) = λ0 + β z(a) for age_temporal; λ0 for temporal_only; 0 otherwise."""
        if self.arm == "age_temporal":
            return self.lambda0 + self.beta * z_age
        if self.arm == "temporal_only":
            return self.lambda0.expand_as(z_age)
        return torch.zeros_like(z_age)

    def lambda_at_ages(
        self, ages_years: list[float] | tuple[float, ...], age_mean: float, age_std: float
    ) -> dict[str, float]:
        device = self.lambda0.device
        age = torch.tensor(list(ages_years), dtype=torch.float32, device=device)
        z = (age - age_mean) / max(age_std, 1e-6)
        lam = self.temporal_slope(z)
        return {str(a): float(v) for a, v in zip(ages_years, lam.detach().cpu().tolist())}

    def _attn_bias(
        self, days_before: torch.Tensor, time_norm: torch.Tensor, z_age: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Key-side bias b_j = -(λ0 + β z(a)) τ_j, τ_j = lag of key from index."""
        tau_key = time_norm.clamp(0.0, 1.0)
        tau = tau_key.unsqueeze(1).expand(-1, tau_key.size(1), -1)
        slope = self.temporal_slope(z_age).view(-1, 1, 1)
        if self.arm in ("temporal_only", "age_temporal"):
            bias = -slope * tau
        else:
            bias = torch.zeros_like(tau)
        return bias, tau

    def _attention(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        bias: torch.Tensor,
        is_query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq, _ = x.shape
        y = self.ln1(x)
        qkv = self.qkv(y).view(bsz, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + bias.unsqueeze(1)
        key_mask = padding_mask | is_query
        scores = scores.masked_fill(key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        out = self.attn_out(out)
        return x + self.attn_dropout(out), attn

    def forward(
        self,
        code_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_norm: torch.Tensor,
        days_before: torch.Tensor,
        padding_mask: torch.Tensor,
        z_age: torch.Tensor,
        is_query: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encode_events(code_ids, type_ids, time_norm)
        bias, tau = self._attn_bias(days_before, time_norm, z_age)
        h, attn = self._attention(x, padding_mask, bias, is_query)
        h = h + self.ff(self.ln2(h))

        query_mask = is_query & (~padding_mask)
        missing = query_mask.sum(dim=1) == 0
        if missing.any():
            last = (~padding_mask).long().sum(dim=1).clamp(min=1) - 1
            query_mask = query_mask.clone()
            query_mask[missing, last[missing]] = True
        weights = query_mask.to(h.dtype)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp(
            min=1.0
        )

        if self.arm == "late_age":
            head_age = z_age
        else:
            head_age = torch.zeros_like(z_age)
        logits = self.head(torch.cat([pooled, head_age.unsqueeze(-1)], dim=-1)).squeeze(-1)

        self._cache = {
            "attn": attn.detach(),
            "tau": tau.detach(),
            "bias": bias.detach(),
            "lambda": self.temporal_slope(z_age).detach(),
            "query_mask": query_mask.detach(),
        }
        return logits


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def temporal_param_ids(model: InteractionModel) -> set[int]:
    return {id(model.lambda0), id(model.beta)}
