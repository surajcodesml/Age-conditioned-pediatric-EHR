"""E01: target-conditioned unnormalized content retrieval on C01 backbone."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_dtr import CONTENT_SCORE_EXP_CLAMP, DevelopmentalTemporalRetrieval


class TargetConditionedDTR(nn.Module):
    """C01 encoder + developmental gate; target-specific content retrieval."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        *,
        age_temporal: bool,
        d_model: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.architecture = "target_conditioned"
        self.variant = "target_conditioned"
        self.age_temporal = bool(age_temporal)
        self.d_model = int(d_model)
        self.n_targets = int(n_targets)
        self.base = DevelopmentalTemporalRetrieval(
            n_codes=n_codes,
            n_targets=n_targets,
            d_model=d_model,
            age_temporal=True,
            aggregation="raw_additive",
            dropout=dropout,
            content_persistence=True,
        )
        # Replace global content path with target-conditioned projections.
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.target_queries = nn.Parameter(torch.randn(n_targets, d_model) * 0.02)
        self.target_value_queries = nn.Parameter(torch.randn(n_targets, d_model) * 0.02)
        # Freeze unused global content_query/key; history MLP unused for logits.
        self.base.content_query.requires_grad_(False)
        self.base.content_key.requires_grad_(False)
        for p in self.base.history_head.parameters():
            p.requires_grad_(False)
        self.configure_arm_()

    def configure_arm_(self) -> None:
        self.base.beta.requires_grad_(self.age_temporal)
        self.base.age_temporal = True

    def temporal_leaf_names(self) -> set[str]:
        return {"theta0", "beta"}

    def set_backbone_requires_grad(self, flag: bool) -> None:
        skip = self.temporal_leaf_names()
        for name, param in self.named_parameters():
            leaf = name.split(".")[-1]
            if leaf in skip:
                continue
            if name.endswith("content_query") or "content_key" in name:
                param.requires_grad_(False)
                continue
            if "history_head" in name:
                param.requires_grad_(False)
                continue
            param.requires_grad_(bool(flag))
        self.configure_arm_()

    def temporal_parameters(self) -> list[nn.Parameter]:
        params = [self.base.theta0]
        if self.base.beta.requires_grad:
            params.append(self.base.beta)
        return params

    def _active_beta(self) -> nn.Parameter:
        return self.base.beta

    def zero_all_betas_(self) -> dict[str, torch.Tensor]:
        return self.base.zero_all_betas_()

    def restore_betas_(self, saved: dict[str, torch.Tensor]) -> None:
        self.base.restore_betas_(saved)

    def forward(
        self,
        enc_code_ids: torch.Tensor,
        enc_code_mask: torch.Tensor,
        enc_tau: torch.Tensor,
        enc_padding_mask: torch.Tensor,
        age: torch.Tensor,
        enc_lag_days: torch.Tensor | None = None,
        return_parts: bool = False,
        gate_age: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del enc_lag_days
        base = self.base
        v = base.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        hist_f = hist.to(v.dtype)

        k = self.key_proj(v)
        val = self.value_proj(v)
        scale = math.sqrt(self.d_model)
        # r_mk = q_k^T k_m / sqrt(d)  -> [B, M, T]
        r = torch.einsum("bmd,td->bmt", k, self.target_queries) / scale
        r = r.masked_fill(~hist.unsqueeze(-1), 0.0)
        a = torch.exp(r.clamp(max=CONTENT_SCORE_EXP_CLAMP))
        # e_mk = p_k^T V(v_m)
        e = torch.einsum("bmd,td->bmt", val, self.target_value_queries)

        age_for_gate = age if gate_age is None else gate_age
        z_gate = base.z_of(age_for_gate)
        theta_content = base.persistence_offset(v) * hist_f
        theta_m = base.theta0 + theta_content
        if self.age_temporal:
            lam = F.softplus(theta_m + base.beta * z_gate.unsqueeze(-1))
        else:
            lam = F.softplus(theta_m)
        g = torch.exp(-lam * enc_tau) * hist_f
        contrib = g.unsqueeze(-1) * a * e * hist_f.unsqueeze(-1)
        history_logit = contrib.sum(dim=1)
        z = base.z_of(age).unsqueeze(-1)
        age_logit = base.age_head(z)
        total = history_logit + age_logit + base.bias
        if not return_parts:
            return total
        return {
            "logits": total,
            "history_logit": history_logit,
            "age_logit": age_logit,
            "g": g,
            "a": a,
            "e": e,
            "r": r,
            "contrib": contrib,
            "v": v,
            "u": r.mean(dim=-1),
            "lambda": F.softplus(base.theta0 + (base.beta * z_gate if self.age_temporal else 0.0)),
        }


def build_e01(cfg: dict[str, Any], n_codes: int, n_targets: int, *, age_temporal: bool) -> TargetConditionedDTR:
    return TargetConditionedDTR(
        n_codes,
        n_targets,
        age_temporal=age_temporal,
        d_model=int(cfg["d_model"]),
        dropout=float(cfg["dropout"]),
    )
