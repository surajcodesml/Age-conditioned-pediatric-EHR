"""Developmental Temporal Retrieval (DTR) — final real-EHR-ready architecture.

Pipeline:
  encounter content encoder (no age/τ)
    → age-independent content relevance u_m = qᵀ k_m
    → developmental temporal gate g_m = exp[-λ(a*) τ_m]
    → w_m = exp(u_m) · g_m
    → mass-preserving aggregation [h̄, log(1+M)]
    → structurally additive logits:
         ℓ = f_history(h) + f_age(z(a*)) + b

Age modifies historical relevance ONLY through g_m.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AGE_CENTER, AGE_SCALE
from model_factorized import TemporalGate

AGGREGATIONS = ("weighted_mean_plus_log_mass", "raw_additive")


class DevelopmentalTemporalRetrieval(nn.Module):
    """Encounter-level DTR model."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 64,
        age_temporal: bool = True,
        aggregation: str = "weighted_mean_plus_log_mass",
        dropout: float = 0.0,
        max_codes_per_encounter: int = 32,
        content_persistence: bool = False,
        multi_query_K: int = 1,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation}")
        self.aggregation = aggregation
        self.age_temporal = bool(age_temporal)
        self.d_model = d_model
        self.max_codes_per_encounter = max_codes_per_encounter
        self.content_persistence = content_persistence
        self.multi_query_K = multi_query_K

        self.code_emb = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.enc_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate = TemporalGate(age_temporal)
        self.q = nn.Parameter(torch.randn(self.multi_query_K, d_model) * 0.02)
        if self.content_persistence:
            self.W_r = nn.Linear(d_model, 1, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        hist_in = (d_model * self.multi_query_K) + (self.multi_query_K if aggregation == "weighted_mean_plus_log_mass" else 0)
        # Small MLP: linear readout cannot recombine [h̄, log(1+M)] into
        # mass-scaled evidence (which raw Σwv provides implicitly).
        self.f_history = nn.Sequential(
            nn.Linear(hist_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_targets),
        )
        self.f_age = nn.Linear(1, n_targets)  # main effect only; no history interaction
        self.bias = nn.Parameter(torch.zeros(n_targets))

        self._cache: dict[str, torch.Tensor] | None = None

    def encode_encounters(
        self,
        enc_code_ids: torch.Tensor,  # [B, M, K]
        enc_code_mask: torch.Tensor,  # [B, M, K] True = valid code
    ) -> torch.Tensor:
        """DeepSets mean-pool of code embeddings → MLP. No age/τ."""
        e = self.code_emb(enc_code_ids)  # [B, M, K, d]
        mask = enc_code_mask.to(e.dtype).unsqueeze(-1)
        summed = (e * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp(min=1.0)
        mean_e = summed / denom
        return self.enc_mlp(mean_e)  # [B, M, d]

    def forward(
        self,
        enc_code_ids: torch.Tensor,
        enc_code_mask: torch.Tensor,
        enc_tau: torch.Tensor,
        enc_padding_mask: torch.Tensor,  # True = pad encounter
        age: torch.Tensor,
        return_parts: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # Reject accidental age/τ leakage into encoder inputs (kwargs may carry extras).
        v = self.encode_encounters(enc_code_ids, enc_code_mask)  # [B, M, d]
        hist = ~enc_padding_mask  # [B, M]

        k = self.W_k(v)
        val = self.W_v(v)
        # Content relevance: no age, no τ
        u = torch.matmul(k, self.q.T) / math.sqrt(k.size(-1))  # [B, M, K]
        u = u.masked_fill(~hist.unsqueeze(-1), 0.0)

        theta = self.W_r(v).squeeze(-1) if self.content_persistence else None
        g = self.gate.gate(age, enc_tau, theta) * hist.to(enc_tau.dtype)  # [B, M]

        # w = exp(u) * g  (clamp u for numerical stability; meaning unchanged)
        w = torch.exp(u.clamp(max=20.0)) * g.unsqueeze(-1)
        w = w * hist.unsqueeze(-1).to(w.dtype)

        M = w.sum(dim=1)  # [B, K]
        weighted = torch.einsum("bmk,bmd->bkd", w, val)  # [B, K, d]

        if self.aggregation == "raw_additive":
            h_hist = weighted.reshape(weighted.size(0), -1)
        else:
            h_bar = weighted / (M.unsqueeze(-1) + 1e-6)
            log_mass = torch.log1p(M)
            h_hist = torch.cat([h_bar.reshape(weighted.size(0), -1), log_mass], dim=-1)

        z = ((age - AGE_CENTER) / AGE_SCALE).unsqueeze(-1)
        history_logit = self.f_history(h_hist)
        age_logit = self.f_age(z)
        total = history_logit + age_logit + self.bias

        self._cache = {
            "u": u.detach(),
            "g": g.detach(),
            "w": w.detach(),
            "M": M.detach(),
            "lambda": self.gate.lambda_of(age).detach(),
            "h_hist": h_hist.detach(),
            "history_logit": history_logit.detach(),
            "age_logit": age_logit.detach(),
            "content_mag": u.abs().mean().detach(),
            "gate_mag": (-torch.log(g.clamp_min(1e-12))).mean().detach(),
        }
        if return_parts:
            return {
                "logits": total,
                "history_logit": history_logit,
                "age_logit": age_logit,
                "h_hist": h_hist,
                "M": M,
                "g": g,
                "u": u,
                "w": w,
            }
        return total

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


def build_dtr(
    *,
    age_temporal: bool,
    n_codes: int,
    n_targets: int,
    d_model: int = 64,
    aggregation: str = "weighted_mean_plus_log_mass",
    content_persistence: bool = False,
    multi_query_K: int = 1,
) -> DevelopmentalTemporalRetrieval:
    return DevelopmentalTemporalRetrieval(
        n_codes=n_codes,
        n_targets=n_targets,
        d_model=d_model,
        age_temporal=age_temporal,
        aggregation=aggregation,
        content_persistence=content_persistence,
        multi_query_K=multi_query_K,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DTRContentPersistence(DevelopmentalTemporalRetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = nn.Linear(self.d_model, 1, bias=False)
        
    def forward(self, enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age, return_parts=False, **kwargs):
        v = self.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask

        k = self.W_k(v)
        val = self.W_v(v)
        u = torch.matmul(k, self.q) / math.sqrt(k.size(-1))
        u = u.masked_fill(~hist, 0.0)

        theta_m = self.gate.theta0 + self.r(v).squeeze(-1)
        z = self.gate.z_of(age).unsqueeze(-1)
        lam_m = F.softplus(theta_m + self.gate.beta * z)
        
        g = torch.exp(-lam_m * enc_tau) * hist.to(enc_tau.dtype)

        w = torch.exp(u.clamp(max=20.0)) * g
        w = w * hist.to(w.dtype)

        M = w.sum(dim=-1, keepdim=True)
        weighted = (w.unsqueeze(-1) * val).sum(dim=1)

        if self.aggregation == "raw_additive":
            h_hist = weighted
        else:
            h_bar = weighted / (M + 1e-6)
            log_mass = torch.log1p(M)
            h_hist = torch.cat([h_bar, log_mass], dim=-1)

        history_logit = self.f_history(h_hist)
        age_logit = self.f_age(z)
        total = history_logit + age_logit + self.bias
        
        self._cache = {
            "u": u.detach(),
            "g": g.detach(),
            "w": w.detach(),
            "M": M.detach(),
            "lambda": lam_m.detach(),
            "theta_m": theta_m.detach(),
        }
        if return_parts:
            return {"logits": total, "history_logit": history_logit, "age_logit": age_logit, "h_hist": h_hist, "M": M, "g": g, "u": u, "w": w}
        return total


class DTRMultiQuery(DevelopmentalTemporalRetrieval):
    def __init__(self, *args, num_queries=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_queries = num_queries
        self.Q = nn.Parameter(torch.randn(num_queries, self.d_model) * 0.02)
        n_targets = self.bias.size(0)
        hist_in = (self.d_model + (1 if self.aggregation == "weighted_mean_plus_log_mass" else 0)) * num_queries
        self.f_history = nn.Sequential(
            nn.Linear(hist_in, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, n_targets),
        )

    def forward(self, enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age, return_parts=False, **kwargs):
        v = self.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask

        k = self.W_k(v)
        val = self.W_v(v)
        
        u = torch.matmul(k, self.Q.T) / math.sqrt(k.size(-1))
        u = u.masked_fill(~hist.unsqueeze(-1), 0.0)
        
        g = self.gate.gate(age, enc_tau) * hist.to(enc_tau.dtype)
        
        w = torch.exp(u.clamp(max=20.0)) * g.unsqueeze(-1)
        w = w * hist.unsqueeze(-1).to(w.dtype)
        
        M = w.sum(dim=1)
        
        weighted = (w.unsqueeze(-1) * val.unsqueeze(-2)).sum(dim=1)
        
        if self.aggregation == "raw_additive":
            h_hist = weighted.view(weighted.size(0), -1)
        else:
            h_bar = weighted / (M.unsqueeze(-1) + 1e-6)
            log_mass = torch.log1p(M).unsqueeze(-1)
            h_hist = torch.cat([h_bar, log_mass], dim=-1).view(weighted.size(0), -1)

        z = self.gate.z_of(age).unsqueeze(-1)
        history_logit = self.f_history(h_hist)
        age_logit = self.f_age(z)
        total = history_logit + age_logit + self.bias
        
        self._cache = {
            "u": u.detach(),
            "g": g.detach(),
            "w": w.detach(),
            "M": M.detach(),
            "lambda": self.gate.lambda_of(age).detach(),
        }
        if return_parts:
            return {"logits": total, "history_logit": history_logit, "age_logit": age_logit, "h_hist": h_hist}
        return total
