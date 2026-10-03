"""High-impact DTR variants built on Content-Persistence DTR / C01."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_dtr import CONTENT_SCORE_EXP_CLAMP, DevelopmentalTemporalRetrieval

VARIANTS = ("oracle_gate", "multihead_shared", "multihead_dev")


class HighImpactDTR(nn.Module):
    """C01 current DTR with one targeted change per variant."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        *,
        variant: str,
        age_temporal: bool,
        d_model: int = 64,
        dropout: float = 0.0,
        n_heads: int = 4,
        d_head: int = 16,
        oracle_theta0: float = 0.0,
        oracle_beta: float = 0.0,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        if int(n_heads) * int(d_head) != int(d_model):
            raise ValueError(f"H*d_head must equal d_model, got {n_heads}*{d_head}!={d_model}")
        self.variant = variant
        self.architecture = variant
        self.age_temporal = bool(age_temporal)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = int(d_head)
        self.base = DevelopmentalTemporalRetrieval(
            n_codes=n_codes,
            n_targets=n_targets,
            d_model=d_model,
            age_temporal=True,
            aggregation="raw_additive",
            dropout=dropout,
            content_persistence=True,
        )
        self.register_buffer("oracle_theta0", torch.tensor([float(oracle_theta0)]))
        self.register_buffer("oracle_beta_true", torch.tensor([float(oracle_beta)]))
        self._oracle_beta_override: float | None = None

        if variant in ("multihead_shared", "multihead_dev"):
            scale = 0.02
            self.content_queries = nn.Parameter(torch.randn(self.n_heads, self.d_head) * scale)
            self.key_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
            self.value_proj = nn.Linear(d_model, self.n_heads * self.d_head, bias=False)
            # Disable unused single-query path parameters.
            self.base.content_query.requires_grad_(False)
            self.base.content_key.requires_grad_(False)
        if variant == "multihead_dev":
            self.beta_global = nn.Parameter(torch.zeros(1))
            self.delta = nn.Parameter(torch.zeros(self.n_heads))
            self.base.beta.requires_grad_(False)
        if variant == "oracle_gate":
            self.base.theta0.requires_grad_(False)
            self.base.beta.requires_grad_(False)
            self.base.persistence_projection.requires_grad_(False)
        self.configure_arm_()

    def set_oracle_(self, theta0: float, beta_true: float) -> None:
        with torch.no_grad():
            self.oracle_theta0.fill_(float(theta0))
            self.oracle_beta_true.fill_(float(beta_true))

    def temporal_leaf_names(self) -> set[str]:
        if self.variant == "oracle_gate":
            return set()
        if self.variant == "multihead_dev":
            return {"theta0", "beta_global", "delta"}
        return {"theta0", "beta"}

    def _active_beta(self) -> nn.Parameter:
        if self.variant == "multihead_dev":
            return self.beta_global
        return self.base.beta

    def configure_arm_(self) -> None:
        if self.variant == "oracle_gate":
            self.base.theta0.requires_grad_(False)
            self.base.beta.requires_grad_(False)
            self.base.persistence_projection.requires_grad_(False)
            self.base.age_temporal = True
            return
        if self.variant == "multihead_dev":
            self.base.beta.requires_grad_(False)
            self.beta_global.requires_grad_(self.age_temporal)
            self.delta.requires_grad_(self.age_temporal)
        else:
            self.base.beta.requires_grad_(self.age_temporal)
        self.base.age_temporal = True

    def set_backbone_requires_grad(self, flag: bool) -> None:
        skip = self.temporal_leaf_names()
        for name, param in self.named_parameters():
            leaf = name.split(".")[-1]
            if leaf in skip or name in skip:
                continue
            if self.variant == "oracle_gate" and (
                leaf in {"theta0", "beta"} or "persistence_projection" in name
            ):
                param.requires_grad_(False)
                continue
            if self.variant.startswith("multihead") and (
                name.endswith("content_query") or "content_key" in name
            ):
                param.requires_grad_(False)
                continue
            param.requires_grad_(bool(flag))
        self.configure_arm_()

    def temporal_parameters(self) -> list[nn.Parameter]:
        if self.variant == "oracle_gate":
            return []
        if self.variant == "multihead_dev":
            params = [self.base.theta0]
            if self.beta_global.requires_grad:
                params.extend([self.beta_global, self.delta])
            return params
        params = [self.base.theta0]
        if self.base.beta.requires_grad:
            params.append(self.base.beta)
        return params

    def zero_all_betas_(self) -> dict[str, torch.Tensor]:
        if self.variant == "oracle_gate":
            self._oracle_beta_override = 0.0
            return {"oracle_beta_override": torch.tensor(0.0)}
        if self.variant == "multihead_dev":
            saved = {
                "beta_global": self.beta_global.detach().clone(),
                "delta": self.delta.detach().clone(),
            }
            with torch.no_grad():
                self.beta_global.zero_()
                self.delta.zero_()
            return saved
        return self.base.zero_all_betas_()

    def restore_betas_(self, saved: dict[str, torch.Tensor]) -> None:
        if self.variant == "oracle_gate":
            self._oracle_beta_override = None
            return
        if self.variant == "multihead_dev":
            with torch.no_grad():
                self.beta_global.copy_(saved["beta_global"])
                self.delta.copy_(saved["delta"])
            return
        self.base.restore_betas_(saved)

    def head_betas(self) -> torch.Tensor | None:
        if self.variant != "multihead_dev":
            return None
        if not self.age_temporal:
            return self.beta_global.expand(self.n_heads) * 0.0
        centered = self.delta - self.delta.mean()
        return self.beta_global + centered

    def _oracle_beta_value(self) -> float:
        if self._oracle_beta_override is not None:
            return float(self._oracle_beta_override)
        if not self.age_temporal:
            return 0.0
        return float(self.oracle_beta_true.item())

    def oracle_lambda(self, age: torch.Tensor) -> torch.Tensor:
        z = self.base.z_of(age)
        beta = self._oracle_beta_value()
        return F.softplus(self.oracle_theta0 + beta * z)

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
        ablate_head: int | None = None,
        **_: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del enc_lag_days
        if self.variant == "oracle_gate":
            return self._oracle_forward(
                enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age,
                return_parts=return_parts, gate_age=gate_age,
            )
        return self._multihead_forward(
            enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age,
            return_parts=return_parts, gate_age=gate_age, ablate_head=ablate_head,
        )

    def _oracle_forward(
        self,
        enc_code_ids,
        enc_code_mask,
        enc_tau,
        enc_padding_mask,
        age,
        *,
        return_parts: bool,
        gate_age: torch.Tensor | None,
    ):
        base = self.base
        v = base.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        hist_f = hist.to(v.dtype)
        k = base.content_key(v)
        scale = math.sqrt(k.size(-1))
        u = torch.einsum("bmd,d->bm", k, base.content_query) / scale
        u = u.masked_fill(~hist, 0.0)

        age_for_gate = age if gate_age is None else gate_age
        lam = self.oracle_lambda(age_for_gate).unsqueeze(-1)
        g = torch.exp(-lam * enc_tau) * hist_f
        w = torch.exp(u.clamp(max=CONTENT_SCORE_EXP_CLAMP)) * g * hist_f
        weighted = (w.unsqueeze(-1) * v).sum(dim=1)
        z = base.z_of(age).unsqueeze(-1)
        history_logit = base.history_head(weighted)
        age_logit = base.age_head(z)
        total = history_logit + age_logit + base.bias
        if not return_parts:
            return total
        return {
            "logits": total,
            "history_logit": history_logit,
            "age_logit": age_logit,
            "g": g,
            "w": w,
            "u": u,
            "v": v,
            "lambda": lam.squeeze(-1),
            "h_heads": weighted.unsqueeze(1),
        }

    def _multihead_forward(
        self,
        enc_code_ids,
        enc_code_mask,
        enc_tau,
        enc_padding_mask,
        age,
        *,
        return_parts: bool,
        gate_age: torch.Tensor | None,
        ablate_head: int | None,
    ):
        base = self.base
        v = base.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        hist_f = hist.to(v.dtype)
        bsz, n_enc, _ = v.shape

        keys = self.key_proj(v).view(bsz, n_enc, self.n_heads, self.d_head)
        values = self.value_proj(v).view(bsz, n_enc, self.n_heads, self.d_head)
        scale = math.sqrt(self.d_head)
        u = torch.einsum("bmhd,hd->bmh", keys, self.content_queries) / scale
        u = u.masked_fill(~hist.unsqueeze(-1), 0.0)

        age_for_gate = age if gate_age is None else gate_age
        z_gate = base.z_of(age_for_gate)
        theta_content = base.persistence_offset(v) * hist_f
        theta_m = base.theta0 + theta_content

        if self.variant == "multihead_shared":
            if self.age_temporal:
                lam = F.softplus(theta_m + base.beta * z_gate.unsqueeze(-1))
            else:
                lam = F.softplus(theta_m)
            g = torch.exp(-lam * enc_tau) * hist_f
            g_heads = g.unsqueeze(-1).expand(-1, -1, self.n_heads)
            lam_heads = lam.unsqueeze(-1).expand(-1, -1, self.n_heads)
        else:
            betas = self.head_betas()
            if betas is None:
                betas = torch.zeros(self.n_heads, device=v.device, dtype=v.dtype)
            # [B,M,H]
            raw = theta_m.unsqueeze(-1) + betas.view(1, 1, -1) * z_gate.view(bsz, 1, 1)
            lam_heads = F.softplus(raw)
            g_heads = torch.exp(-lam_heads * enc_tau.unsqueeze(-1)) * hist_f.unsqueeze(-1)

        w = torch.exp(u.clamp(max=CONTENT_SCORE_EXP_CLAMP)) * g_heads * hist_f.unsqueeze(-1)
        # h_h = sum_m w_mh * value_mh  -> [B, H, d_head]
        h_heads = torch.einsum("bmh,bmhd->bhd", w, values)
        if ablate_head is not None:
            h_heads = h_heads.clone()
            h_heads[:, int(ablate_head), :] = 0.0
        history = h_heads.reshape(bsz, self.d_model)
        z = base.z_of(age).unsqueeze(-1)
        history_logit = base.history_head(history)
        age_logit = base.age_head(z)
        total = history_logit + age_logit + base.bias
        if not return_parts:
            return total
        out = {
            "logits": total,
            "history_logit": history_logit,
            "age_logit": age_logit,
            "g": g_heads.mean(dim=-1),
            "g_heads": g_heads,
            "w": w,
            "u": u,
            "v": v,
            "h_heads": h_heads,
            "lam_heads_enc": lam_heads,
        }
        if self.variant == "multihead_shared":
            if self.age_temporal:
                out["lambda"] = F.softplus(base.theta0 + base.beta * z_gate)
            else:
                out["lambda"] = F.softplus(base.theta0).expand_as(z_gate)
        else:
            betas = self.head_betas()
            out["beta_h"] = betas
            out["lambda_h"] = F.softplus(
                base.theta0 + betas * z_gate.unsqueeze(-1)
            )
        return out


def build_high_impact(
    cfg: dict[str, Any],
    n_codes: int,
    n_targets: int,
    *,
    age_temporal: bool,
    oracle_theta0: float = 0.0,
    oracle_beta: float = 0.0,
) -> HighImpactDTR:
    return HighImpactDTR(
        n_codes,
        n_targets,
        variant=str(cfg["variant"]),
        age_temporal=age_temporal,
        d_model=int(cfg["d_model"]),
        dropout=float(cfg["dropout"]),
        n_heads=int(cfg.get("n_heads", 4)),
        d_head=int(cfg.get("d_head", 16)),
        oracle_theta0=oracle_theta0,
        oracle_beta=oracle_beta,
    )
