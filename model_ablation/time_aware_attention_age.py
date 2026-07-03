#!/usr/bin/env python3
"""Age-conditioned time-aware attention for the ablation (Chebyshev basis).

Differences from the frozen ``model/`` version:
  * Kernel injection is LOCKED to additive-logspace: ``scores + logsigmoid(poly)``.
    There is no multiplicative path.
  * The temporal polynomial is evaluated in the CHEBYSHEV basis on
    ``x = 2*log1p(dt/7)/t_max - 1`` (t_max ~= 6.5), which keeps degree 5 numerically
    well-conditioned (Gram cond ~2.6e9 monomial -> ~34 Chebyshev).
  * Age is supplied as a separate ``age_years`` tensor (never through demographics).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_ablation.age_embedding import AgeCoefficientGenerator, FourierAgeEmbedding

# Upper bound of the empirical log1p(dt/7) distribution (see diagnostics section E:
# 99th pct ~6.23, max ~6.49). Used only to rescale into the Chebyshev domain [-1, 1].
CHEB_TMAX = 6.5


def _chebyshev_powers(x: torch.Tensor, degree: int) -> list[torch.Tensor]:
    """Chebyshev polynomials T_0..T_degree evaluated at x, via the recurrence
    T_0=1, T_1=x, T_{k+1}=2x T_k - T_{k-1}."""
    terms = [torch.ones_like(x), x]
    for _ in range(degree - 1):
        terms.append(2.0 * x * terms[-1] - terms[-2])
    return terms[: degree + 1]


class ChebyshevPolynomialWeight(nn.Module):
    """Temporal decay: sigmoid of a degree-D Chebyshev polynomial in rescaled log-dt,
    with optional age-conditioned additive coefficient deltas (Delta-alpha)."""

    def __init__(
        self,
        poly_degree: int = 5,
        age_emb_dim: int = 32,
        hidden_dim: int = 64,
        age_conditioning_mode: str = "none",
        t_max: float = CHEB_TMAX,
    ) -> None:
        super().__init__()
        self.poly_degree = int(poly_degree)
        self.t_max = float(t_max)
        coeffs = torch.zeros(self.poly_degree + 1)
        coeffs[0] = 0.5  # T_0 offset -> sigmoid(0.5) ~ 0.62 at init, matches frozen init
        self.coefficients = nn.Parameter(coeffs)
        self.age_coeff_gen = AgeCoefficientGenerator(
            in_dim=age_emb_dim,
            hidden_dim=hidden_dim,
            out_dim=self.poly_degree + 1,
            mode=age_conditioning_mode,
        )

    def coefficient_delta(self, age_features: torch.Tensor) -> torch.Tensor:
        return self.age_coeff_gen(age_features)

    def _poly(self, log_delta_t: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        alpha_delta = self.age_coeff_gen(age_features)      # [..., D+1] (or broadcastable)
        alpha = self.coefficients + alpha_delta
        x = 2.0 * log_delta_t / self.t_max - 1.0            # rescale to [-1, 1] domain
        basis = _chebyshev_powers(x, self.poly_degree)      # list of [...] tensors
        poly = torch.zeros_like(log_delta_t)
        for k in range(self.poly_degree + 1):
            poly = poly + alpha[..., k : k + 1] * basis[k]
        return poly

    def poly_value(self, log_delta_t: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        return self._poly(log_delta_t, age_features)

    def forward(self, log_delta_t: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._poly(log_delta_t, age_features))


class AgeConditionedTimeAwareAttention(nn.Module):
    """Single-head time-aware attention; additive-logspace kernel injection only."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        d_model: int = 256,
        poly_degree: int = 5,
        age_emb_dim: int = 32,
        age_hidden_dim: int = 64,
        age_conditioning_mode: str = "none",
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.d_model = int(d_model)

        self.mlp_q = nn.Sequential(nn.Linear(self.embedding_dim, self.d_model), nn.GELU())
        self.mlp_k = nn.Sequential(nn.Linear(self.embedding_dim, self.d_model), nn.GELU())
        self.mlp_v = nn.Sequential(nn.Linear(self.embedding_dim, self.d_model), nn.GELU())
        self.temporal_weight = ChebyshevPolynomialWeight(
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            hidden_dim=age_hidden_dim,
            age_conditioning_mode=age_conditioning_mode,
        )
        self.age_emb = FourierAgeEmbedding(num_frequencies=age_emb_dim // 2)

    def forward(
        self,
        code_embeddings: torch.Tensor,
        delta_t: torch.Tensor,
        attention_mask: torch.Tensor,
        age_features: torch.Tensor,
    ) -> torch.Tensor:
        """``age_features`` are precomputed Fourier features [B, L, age_emb_dim]
        (the caller decides real vs constant age). ``delta_t`` is log1p(|dt|/7)."""
        q = self.mlp_q(code_embeddings)
        k = self.mlp_k(code_embeddings)
        v = self.mlp_v(code_embeddings)

        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale
        poly = self.temporal_weight.poly_value(delta_t, age_features)
        scores = scores + F.logsigmoid(poly)  # additive-logspace (locked)

        _, l, _ = scores.shape
        device = scores.device
        causal = torch.tril(torch.ones((l, l), device=device, dtype=torch.bool))
        pad_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        full_mask = pad_mask & causal.unsqueeze(0)

        scores = scores.masked_fill(~full_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(~full_mask, 0.0)
        return torch.bmm(attn, v)


class AgeConditionedMultiScaleTemporalAggregation(nn.Module):
    """Temporal aggregation to a single vector; additive-logspace kernel on relevance.

    Kept for pretraining parity. Note: fine-tuning uses ``return_repr_only`` and does
    NOT call this module (see diagnostics: aggregation-path gradient death), so the
    fine-tune age pathways live entirely in the attention module above.
    """

    def __init__(
        self,
        d_model: int = 256,
        poly_degree: int = 5,
        age_emb_dim: int = 32,
        age_hidden_dim: int = 64,
        age_conditioning_mode: str = "none",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.q_base = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.temporal_weight = ChebyshevPolynomialWeight(
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            hidden_dim=age_hidden_dim,
            age_conditioning_mode=age_conditioning_mode,
        )
        self.age_emb = FourierAgeEmbedding(num_frequencies=age_emb_dim // 2)

    def forward(
        self,
        e: torch.Tensor,
        timestamps_days: torch.Tensor,
        attention_mask: torch.Tensor,
        age_features_current: torch.Tensor,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        b = e.shape[0]
        batch_idx = torch.arange(b, device=e.device)
        t_current = timestamps_days[batch_idx, lengths - 1]

        delta_to_current = torch.abs(t_current.unsqueeze(1) - timestamps_days)
        log_delta = torch.log1p(delta_to_current / 7.0)

        relevance = torch.einsum("d, bld -> bl", self.q_base, e)
        poly = self.temporal_weight.poly_value(log_delta, age_features_current)
        scores = relevance + F.logsigmoid(poly)  # additive-logspace (locked)
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(~attention_mask, 0.0)
        return torch.einsum("bl, bld -> bd", alpha, e)
