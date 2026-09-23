"""Prediction-time age × temporal Transformer arms for the benchmark.

Global age_temporal:
    s_*j^(h) = q_*^(h)⊤ k_j^(h) / sqrt(d_h) - λ(a*) τ_*j
    λ(a*) = softplus(θ0 + β z(a*))

Per-head age_temporal_per_head:
    s_*j^(h) = q_*^(h)⊤ k_j^(h) / sqrt(d_h) - λ_h(a*) τ_*j
    λ_h(a*) = softplus(θ0_h + β_h z(a*))

No Fourier / Chebyshev / age MLP. No sign constraint on β / β_h.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AGE_CENTER, AGE_SCALE, ARMS


class SoftplusAgeTemporal(nn.Module):
    """Global λ(a) = softplus(θ0 + β z(a)), z(a)=(a-9)/9."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        self.arm = arm
        self.per_head = False
        self.n_heads = 1
        self.theta0 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        if arm in ("no_age", "age_only"):
            self.theta0.requires_grad_(False)
            self.beta.requires_grad_(False)
        elif arm in ("temporal_only", "temporal_only_per_head"):
            self.beta.requires_grad_(False)

    def z_of(self, age_years: torch.Tensor) -> torch.Tensor:
        return (age_years - AGE_CENTER) / AGE_SCALE

    def lambda_of(self, age_years: torch.Tensor) -> torch.Tensor:
        z = self.z_of(age_years)
        if self.arm in ("no_age", "age_only"):
            return torch.zeros_like(age_years)
        if self.arm in ("temporal_only", "temporal_only_per_head"):
            return F.softplus(self.theta0).expand_as(age_years)
        return F.softplus(self.theta0 + self.beta * z)

    def lambda_at_ages(self, ages: list[float] | tuple[float, ...]) -> dict[str, float]:
        device = self.theta0.device
        a = torch.tensor(list(ages), dtype=torch.float32, device=device)
        lam = self.lambda_of(a)
        return {str(x): float(v) for x, v in zip(ages, lam.detach().cpu().tolist())}

    def age_parameters(self) -> list[nn.Parameter]:
        return [p for p in (self.theta0, self.beta) if p.requires_grad]

    def beta_vector(self) -> torch.Tensor:
        return self.beta.detach().reshape(-1)

    def theta0_vector(self) -> torch.Tensor:
        return self.theta0.detach().reshape(-1)


class SoftplusAgeTemporalPerHead(nn.Module):
    """Per-head λ_h(a) = softplus(θ0_h + β_h z(a))."""

    def __init__(self, arm: str, n_heads: int) -> None:
        super().__init__()
        self.arm = arm
        self.per_head = True
        self.n_heads = int(n_heads)
        self.theta0 = nn.Parameter(torch.zeros(self.n_heads))
        self.beta = nn.Parameter(torch.zeros(self.n_heads))
        if arm == "temporal_only_per_head":
            self.beta.requires_grad_(False)
        elif arm != "age_temporal_per_head":
            raise ValueError(f"per-head module only for per-head arms, got {arm}")

    def z_of(self, age_years: torch.Tensor) -> torch.Tensor:
        return (age_years - AGE_CENTER) / AGE_SCALE

    def lambda_of(self, age_years: torch.Tensor) -> torch.Tensor:
        """Return [B, H] (or [H] if age is 1-D of length n_ages broadcast differently)."""
        z = self.z_of(age_years)
        if z.ndim == 0:
            z = z.view(1)
        # age_years: [B] → λ: [B, H]
        if self.arm == "temporal_only_per_head":
            # softplus(θ0_h) expanded over batch
            return F.softplus(self.theta0).unsqueeze(0).expand(z.shape[0], -1)
        return F.softplus(self.theta0.unsqueeze(0) + self.beta.unsqueeze(0) * z.unsqueeze(-1))

    def lambda_at_ages(self, ages: list[float] | tuple[float, ...]) -> dict[str, Any]:
        device = self.theta0.device
        a = torch.tensor(list(ages), dtype=torch.float32, device=device)
        lam = self.lambda_of(a)  # [n_ages, H]
        out: dict[str, Any] = {"per_head": True}
        for i, age in enumerate(ages):
            out[str(age)] = {
                f"h{h}": float(lam[i, h].detach().cpu()) for h in range(self.n_heads)
            }
        out["mean_over_heads"] = {
            str(age): float(lam[i].mean().detach().cpu()) for i, age in enumerate(ages)
        }
        return out

    def age_parameters(self) -> list[nn.Parameter]:
        return [p for p in (self.theta0, self.beta) if p.requires_grad]

    def beta_vector(self) -> torch.Tensor:
        return self.beta.detach().reshape(-1)

    def theta0_vector(self) -> torch.Tensor:
        return self.theta0.detach().reshape(-1)


def build_temporal(arm: str, n_heads: int) -> nn.Module:
    if arm in ("age_temporal_per_head", "temporal_only_per_head"):
        return SoftplusAgeTemporalPerHead(arm, n_heads)
    return SoftplusAgeTemporal(arm)


class BenchmarkModel(nn.Module):
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
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}; expected one of {ARMS}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.arm = arm
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_layers = n_layers
        self.per_head_kernel = arm in ("age_temporal_per_head", "temporal_only_per_head")

        self.code_emb = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(n_types, d_model, padding_idx=0)
        self.temporal = build_temporal(arm, n_heads)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q": nn.Linear(d_model, d_model),
                        "k": nn.Linear(d_model, d_model),
                        "v": nn.Linear(d_model, d_model),
                        "out": nn.Linear(d_model, d_model),
                        "ln1": nn.LayerNorm(d_model),
                        "ln2": nn.LayerNorm(d_model),
                        "ff": nn.Sequential(
                            nn.Linear(d_model, dim_feedforward),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim_feedforward, d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.dropout = nn.Dropout(dropout)

        age_only = arm == "age_only"
        self.query_proj = nn.Sequential(
            nn.Linear(d_model + (1 if age_only else 0), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        head_in = d_model + (1 if age_only else 0)
        self.head = nn.Linear(head_in, n_targets)
        nn.init.zeros_(self.head.bias)
        self._cache: dict[str, torch.Tensor] | None = None

    def encode(self, code_ids: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        return self.code_emb(code_ids) + self.type_emb(type_ids)

    def _content_self_attn(
        self,
        layer: nn.ModuleDict,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, l, _ = x.shape
        h = layer["ln1"](x)
        q = layer["q"](h).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        k = layer["k"](h).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = layer["v"](h).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = padding_mask[:, None, None, :]
        scores = scores.masked_fill(key_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, l, self.d_model)
        out = layer["out"](out)
        x = x + self.dropout(out)
        x = x + layer["ff"](layer["ln2"](x))
        return x

    def prediction_attention(
        self,
        h: torch.Tensor,
        padding_mask: torch.Tensor,
        is_query: torch.Tensor,
        tau: torch.Tensor,
        age: torch.Tensor,
        event_ages: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, _ = h.shape
        hist_mask = (~padding_mask) & (~is_query)
        weights = hist_mask.to(h.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * weights.unsqueeze(-1)).sum(dim=1) / denom

        if self.arm == "age_only":
            q_in = torch.cat([pooled, self.temporal.z_of(age).unsqueeze(-1)], dim=-1)
        else:
            q_in = pooled
        q_vec = self.query_proj(q_in)

        scale = 1.0 / math.sqrt(self.head_dim)
        q = q_vec.view(b, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = h.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        v = h.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,1,L]

        if self.arm == "historical_age":
            assert event_ages is not None
            lam = self.temporal.lambda_of(event_ages)  # [B, L]
            bias = -lam * tau  # [B, L]
            scores = scores + bias[:, None, None, :]
            lam_cache = self.temporal.lambda_of(age)
        elif self.per_head_kernel:
            lam = self.temporal.lambda_of(age)  # [B, H]
            bias = -lam.unsqueeze(-1) * tau.unsqueeze(1)  # [B, H, L]
            scores = scores + bias.unsqueeze(2)  # [B,H,1,L]
            lam_cache = lam
        elif self.arm in ("temporal_only", "age_temporal"):
            lam = self.temporal.lambda_of(age)  # [B]
            bias = -lam.unsqueeze(-1) * tau
            scores = scores + bias[:, None, None, :]
            lam_cache = lam
        else:
            bias = torch.zeros_like(tau)
            scores = scores + bias[:, None, None, :]
            lam_cache = torch.zeros_like(age)

        key_mask = padding_mask | is_query
        scores = scores.masked_fill(key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, self.d_model)

        self._cache = {
            "attn": attn.detach(),
            "bias": bias.detach(),
            "lambda": lam_cache.detach(),
            "tau": tau.detach(),
        }
        return ctx, attn

    def forward(
        self,
        code_ids: torch.Tensor,
        type_ids: torch.Tensor,
        tau: torch.Tensor,
        padding_mask: torch.Tensor,
        is_query: torch.Tensor,
        age: torch.Tensor,
        lag_days: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = self.encode(code_ids, type_ids)
        event_ages = None
        if self.arm == "historical_age":
            assert lag_days is not None
            event_ages = (age.unsqueeze(-1) - (lag_days / 365.25)).clamp(min=0.0)

        h = x
        for layer in self.layers:
            h = self._content_self_attn(layer, h, padding_mask)

        ctx, _ = self.prediction_attention(
            h, padding_mask, is_query, tau, age, event_ages=event_ages
        )
        if self.arm == "age_only":
            u = torch.cat([ctx, self.temporal.z_of(age).unsqueeze(-1)], dim=-1)
        else:
            u = ctx
        return self.head(u)

    @torch.no_grad()
    def extract_repr(
        self,
        code_ids: torch.Tensor,
        type_ids: torch.Tensor,
        tau: torch.Tensor,
        padding_mask: torch.Tensor,
        is_query: torch.Tensor,
        age: torch.Tensor,
        lag_days: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Frozen-model representations for age-decoding probes."""
        x = self.encode(code_ids, type_ids)
        event_ages = None
        if self.arm == "historical_age":
            assert lag_days is not None
            event_ages = (age.unsqueeze(-1) - (lag_days / 365.25)).clamp(min=0.0)
        h = x
        for layer in self.layers:
            h = self._content_self_attn(layer, h, padding_mask)
        hist_mask = (~padding_mask) & (~is_query)
        w = hist_mask.to(h.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
        pre_pool = (h * w.unsqueeze(-1)).sum(dim=1) / denom
        ctx, attn = self.prediction_attention(
            h, padding_mask, is_query, tau, age, event_ages=event_ages
        )
        # Per-head context from prediction attention: [B, H, d_head]
        # attn is [B, H, 1, L]; v from h
        b, l, _ = h.shape
        v = h.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        head_ctx = torch.matmul(attn, v).squeeze(2)  # [B, H, d_head]
        return {
            "pre_pool": pre_pool.detach(),
            "pooled": ctx.detach(),
            "head_ctx": head_ctx.detach(),
            "attn": attn.detach(),
        }

    def age_parameters(self) -> list[nn.Parameter]:
        return self.temporal.age_parameters()

    def zero_all_betas_(self) -> dict[str, torch.Tensor]:
        """Inference ablation: set all β (or β_h) to 0. Returns saved values."""
        saved = {"beta": self.temporal.beta.detach().clone()}
        with torch.no_grad():
            self.temporal.beta.zero_()
        return saved

    def restore_betas_(self, saved: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.temporal.beta.copy_(saved["beta"])

    def zero_one_head_beta_(self, head: int) -> float:
        saved = float(self.temporal.beta[head].detach().cpu())
        with torch.no_grad():
            self.temporal.beta[head] = 0.0
        return saved


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
