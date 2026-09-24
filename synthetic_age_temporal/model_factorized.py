"""Factorized mechanism-isolation models: M1 (kernel-only), M2, M3.

M1 — no Transformer; additive temporal gate on event embeddings:
    λ(a*) = softplus(θ0 + β z(a*))
    g_j   = exp(-λ(a*) τ_*j)
    h_*   = Σ_j g_j e_j     (additive; optional + mass m=Σ g_j)
    ŷ     = f(h_*, z(a*))

Age×time only through g_j. Matched control freezes β=0.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AGE_CENTER, AGE_SCALE


class TemporalGate(nn.Module):
    def __init__(self, age_temporal: bool) -> None:
        super().__init__()
        self.age_temporal = bool(age_temporal)
        self.theta0 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        if not self.age_temporal:
            self.beta.requires_grad_(False)

    def z_of(self, age: torch.Tensor) -> torch.Tensor:
        return (age - AGE_CENTER) / AGE_SCALE

    def lambda_of(self, age: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        z = self.z_of(age)
        th = self.theta0 if theta is None else theta
        if th.dim() > z.dim():
            z = z.view(z.shape + (1,) * (th.dim() - z.dim()))
        if self.age_temporal:
            return F.softplus(th + self.beta * z)
        if th.dim() > age.dim():
            return F.softplus(th).expand_as(th)
        return F.softplus(th).expand_as(age)

    def gate(self, age: torch.Tensor, tau: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        """g_j = exp(-λ(a*) τ_j), shape [B, L]."""
        lam = self.lambda_of(age, theta)
        if lam.dim() < tau.dim():
            lam = lam.unsqueeze(-1)
        return torch.exp(-lam * tau)

    def age_parameters(self) -> list[nn.Parameter]:
        return [p for p in (self.theta0, self.beta) if p.requires_grad]


class M1KernelOnly(nn.Module):
    """Kernel-only additive evidence model (primary isolation architecture)."""

    def __init__(
        self,
        n_codes: int,
        n_types: int,
        n_targets: int,
        d_model: int = 64,
        age_temporal: bool = True,
        aggregation: str = "additive",  # additive | softmax | additive_mass
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if aggregation not in ("additive", "softmax", "additive_mass"):
            raise ValueError(aggregation)
        self.aggregation = aggregation
        self.age_temporal = age_temporal
        self.d_model = d_model
        self.code_emb = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(n_types, d_model, padding_idx=0)
        self.gate = TemporalGate(age_temporal)
        head_in = d_model + 1  # + z(a*)
        if aggregation == "additive_mass":
            head_in += 1
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )
        self._cache: dict[str, torch.Tensor] | None = None

    def encode(self, code_ids, type_ids):
        return self.code_emb(code_ids) + self.type_emb(type_ids)

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
        e = self.encode(code_ids, type_ids)  # [B, L, d]
        hist = (~padding_mask) & (~is_query)
        g = self.gate.gate(age, tau)  # [B, L]
        g = g * hist.to(g.dtype)
        if self.aggregation == "softmax":
            # Normalized attention over history using log-gate as score.
            scores = torch.log(g.clamp_min(1e-12))
            scores = scores.masked_fill(~hist, float("-inf"))
            alpha = torch.softmax(scores, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            h = (alpha.unsqueeze(-1) * e).sum(dim=1)
            mass = alpha.sum(dim=1, keepdim=True)  # ~1
            g_used = alpha
        else:
            h = (g.unsqueeze(-1) * e).sum(dim=1)
            mass = g.sum(dim=1, keepdim=True)
            g_used = g
        z = self.gate.z_of(age).unsqueeze(-1)
        if self.aggregation == "additive_mass":
            u = torch.cat([h, mass, z], dim=-1)
        else:
            u = torch.cat([h, z], dim=-1)
        self._cache = {
            "g": g_used.detach(),
            "lambda": self.gate.lambda_of(age).detach(),
            "mass": mass.detach(),
            "gate_mean": (g_used.sum(dim=1) / hist.to(g.dtype).sum(dim=1).clamp(min=1)).detach(),
        }
        return self.head(u)

    def age_parameters(self) -> list[nn.Parameter]:
        return self.gate.age_parameters()

    def zero_all_betas_(self):
        saved = {"beta": self.gate.beta.detach().clone()}
        with torch.no_grad():
            self.gate.beta.zero_()
        return saved

    def restore_betas_(self, saved):
        with torch.no_grad():
            self.gate.beta.copy_(saved["beta"])


class M2ContentTimesGate(nn.Module):
    """Content score u_j = q⊤k_j (no age/τ) × temporal gate g_j."""

    def __init__(
        self,
        n_codes: int,
        n_types: int,
        n_targets: int,
        d_model: int = 64,
        age_temporal: bool = True,
        aggregation: str = "additive",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.aggregation = aggregation
        self.code_emb = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(n_types, d_model, padding_idx=0)
        self.gate = TemporalGate(age_temporal)
        self.q = nn.Parameter(torch.randn(d_model) * 0.02)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        head_in = d_model + 1
        if aggregation == "additive_mass":
            head_in += 1
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )
        self._cache = None

    def forward(self, code_ids, type_ids, tau, padding_mask, is_query, age, lag_days=None, **kw):
        e = self.code_emb(code_ids) + self.type_emb(type_ids)
        hist = (~padding_mask) & (~is_query)
        k = self.W_k(e)
        v = self.W_v(e)
        u = torch.matmul(k, self.q) / math.sqrt(k.size(-1))  # [B, L] content only
        g = self.gate.gate(age, tau) * hist.to(tau.dtype)
        # w = exp(u) * g  (content cannot see age/τ)
        w = torch.exp(u.clamp(max=20.0)) * g
        w = w * hist.to(w.dtype)
        if self.aggregation == "softmax":
            scores = u + torch.log(g.clamp_min(1e-12))
            scores = scores.masked_fill(~hist, float("-inf"))
            alpha = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            h = (alpha.unsqueeze(-1) * v).sum(dim=1)
            mass = alpha.sum(dim=1, keepdim=True)
            content_mag = u.abs().mean()
            gate_mag = (-torch.log(g.clamp_min(1e-12))).mean()
        else:
            h = (w.unsqueeze(-1) * v).sum(dim=1)
            mass = w.sum(dim=1, keepdim=True)
            content_mag = u.abs().mean()
            gate_mag = (-torch.log(g.clamp_min(1e-12))).mean()
        z = self.gate.z_of(age).unsqueeze(-1)
        u_in = torch.cat([h, mass, z], dim=-1) if self.aggregation == "additive_mass" else torch.cat([h, z], dim=-1)
        self._cache = {
            "content_mag": content_mag.detach(),
            "gate_mag": gate_mag.detach(),
            "lambda": self.gate.lambda_of(age).detach(),
            "ratio_content_over_gate": (content_mag / gate_mag.clamp_min(1e-6)).detach(),
        }
        return self.head(u_in)

    def age_parameters(self):
        return self.gate.age_parameters()

    def zero_all_betas_(self):
        saved = {"beta": self.gate.beta.detach().clone()}
        with torch.no_grad():
            self.gate.beta.zero_()
        return saved

    def restore_betas_(self, saved):
        with torch.no_grad():
            self.gate.beta.copy_(saved["beta"])


class M3EncounterTemporal(nn.Module):
    """Encounter content encoder (no age/τ) + developmental temporal retrieval."""

    def __init__(
        self,
        n_codes: int,
        n_types: int,
        n_targets: int,
        d_model: int = 64,
        age_temporal: bool = True,
        aggregation: str = "additive",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # For this benchmark, "encounters" ≈ events (each event is its own block).
        # Content encoder: MLP on embedding, no age/τ.
        self.inner = M2ContentTimesGate(
            n_codes, n_types, n_targets, d_model, age_temporal, aggregation, dropout
        )

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def age_parameters(self):
        return self.inner.age_parameters()

    def zero_all_betas_(self):
        return self.inner.zero_all_betas_()

    def restore_betas_(self, saved):
        return self.inner.restore_betas_(saved)

    @property
    def gate(self):
        return self.inner.gate

    @property
    def _cache(self):
        return self.inner._cache


def build_factorized(
    family: str,
    *,
    age_temporal: bool,
    n_codes: int,
    n_types: int,
    n_targets: int,
    d_model: int = 64,
    aggregation: str = "additive",
) -> nn.Module:
    if family == "M1":
        return M1KernelOnly(n_codes, n_types, n_targets, d_model, age_temporal, aggregation)
    if family == "M2":
        return M2ContentTimesGate(n_codes, n_types, n_targets, d_model, age_temporal, aggregation)
    if family == "M3":
        return M3EncounterTemporal(n_codes, n_types, n_targets, d_model, age_temporal, aggregation)
    raise ValueError(family)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
