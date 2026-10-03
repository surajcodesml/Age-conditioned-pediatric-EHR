"""Current DTR plus one atomic change per variant.

C00 uses DevelopmentalTemporalRetrieval unchanged apart from an optional
gate_age input. Other variants start from that module and change one piece.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ladder.models.common import inverse_softplus_value
from model_dtr import CONTENT_SCORE_EXP_CLAMP, DevelopmentalTemporalRetrieval

VARIANTS = (
    "current",
    "mass",
    "weak_init",
    "no_persistence",
    "shared_beta_mixture",
    "component_beta_mixture",
)
_MIXTURES = {"shared_beta_mixture", "component_beta_mixture"}


def _mass_readout_(base: DevelopmentalTemporalRetrieval) -> None:
    """Grow only the first history-layer input. Copy the original columns."""
    d_model = int(base.d_model)
    old = base.history_head
    first = old[0]
    new_first = nn.Linear(d_model + 1, d_model)
    with torch.no_grad():
        new_first.weight.zero_()
        new_first.weight[:, :d_model].copy_(first.weight)
        new_first.bias.copy_(first.bias)
    base.history_head = nn.Sequential(new_first, nn.GELU(), old[2])
    base.aggregation = "weighted_mean_plus_log_mass"
    base.temporal_aggregation = base.aggregation


class AtomicDTR(nn.Module):
    """One current-DTR arm. beta is frozen when age_temporal is false."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        *,
        variant: str,
        age_temporal: bool,
        d_model: int = 64,
        dropout: float = 0.0,
        lambda_init: float = 0.1,
        n_components: int = 3,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}")
        self.variant = variant
        self.architecture = variant
        self.age_temporal = bool(age_temporal)
        self.n_components = int(n_components)
        self.lambda_init = float(lambda_init)
        self.base = DevelopmentalTemporalRetrieval(
            n_codes=n_codes,
            n_targets=n_targets,
            d_model=d_model,
            age_temporal=True,
            aggregation="raw_additive",
            dropout=dropout,
            content_persistence=True,
        )
        if variant == "mass":
            _mass_readout_(self.base)
        elif variant == "weak_init":
            self.base.theta0.data.fill_(inverse_softplus_value(self.lambda_init))
        elif variant == "no_persistence":
            self.base.content_persistence = False
            self.base.content_dependent_persistence = False
        elif variant in _MIXTURES:
            self.base.content_persistence = False
            self.base.content_dependent_persistence = False
            self.f_mix = nn.Linear(d_model, self.n_components)
            nn.init.zeros_(self.f_mix.weight)
            nn.init.zeros_(self.f_mix.bias)
            self.theta_k = nn.Parameter(torch.zeros(self.n_components))
            if variant == "shared_beta_mixture":
                self.beta = nn.Parameter(torch.zeros(1))
            else:
                self.beta_k = nn.Parameter(torch.zeros(self.n_components))
        self.configure_arm_()

    def temporal_leaf_names(self) -> set[str]:
        if self.variant == "component_beta_mixture":
            return {"theta_k", "beta_k"}
        if self.variant == "shared_beta_mixture":
            return {"theta_k", "beta"}
        return {"theta0", "beta"}

    def _active_beta(self) -> nn.Parameter:
        if self.variant == "component_beta_mixture":
            return self.beta_k
        if self.variant == "shared_beta_mixture":
            return self.beta
        return self.base.beta

    def configure_arm_(self) -> None:
        if self.variant in _MIXTURES or self.variant == "no_persistence":
            self.base.persistence_projection.requires_grad_(False)
        if self.variant in _MIXTURES:
            self.base.theta0.requires_grad_(False)
            self.base.beta.requires_grad_(False)
        self._active_beta().requires_grad_(self.age_temporal)
        self.base.age_temporal = True

    def set_backbone_requires_grad(self, flag: bool) -> None:
        skip = self.temporal_leaf_names()
        for name, param in self.named_parameters():
            if name.split(".")[-1] in skip:
                continue
            param.requires_grad_(bool(flag))
        self.configure_arm_()

    def temporal_parameters(self) -> list[nn.Parameter]:
        if self.variant in _MIXTURES:
            params: list[nn.Parameter] = [self.theta_k]
        else:
            params = [self.base.theta0]
        beta = self._active_beta()
        if beta.requires_grad:
            params.append(beta)
        return params

    def zero_all_betas_(self) -> dict[str, torch.Tensor]:
        beta = self._active_beta()
        saved = beta.detach().clone()
        with torch.no_grad():
            beta.zero_()
        return {"beta": saved}

    def restore_betas_(self, saved: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self._active_beta().copy_(saved["beta"])

    def content_free_lambda(self, age: torch.Tensor) -> torch.Tensor | None:
        """lambda(a) ignoring content. None when each component has its own slope."""
        z = self.base.z_of(age)
        if self.variant in _MIXTURES:
            return None
        theta = self.base.theta0
        if self.age_temporal:
            return F.softplus(theta + self.base.beta * z)
        return F.softplus(theta).expand_as(z)

    def component_lambda(self, age: torch.Tensor) -> torch.Tensor | None:
        if self.variant not in _MIXTURES:
            return None
        z_col = self.base.z_of(age).unsqueeze(-1)
        theta = self.theta_k.view(1, -1)
        if self.variant == "shared_beta_mixture" and self.age_temporal:
            raw = theta + self.beta * z_col
        elif self.variant == "component_beta_mixture" and self.age_temporal:
            raw = theta + self.beta_k.view(1, -1) * z_col
        else:
            raw = theta.expand(z_col.shape[0], self.n_components)
        return F.softplus(raw)

    def has_global_gate(self) -> bool:
        return self.variant == "no_persistence"

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
        if self.variant in _MIXTURES:
            return self._mixture_forward(
                enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age,
                return_parts=return_parts, gate_age=gate_age,
            )
        return self.base(
            enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age,
            return_parts=return_parts, gate_age=gate_age,
        )

    def _mixture_forward(
        self,
        enc_code_ids: torch.Tensor,
        enc_code_mask: torch.Tensor,
        enc_tau: torch.Tensor,
        enc_padding_mask: torch.Tensor,
        age: torch.Tensor,
        *,
        return_parts: bool,
        gate_age: torch.Tensor | None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        base = self.base
        v = base.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        hist_f = hist.to(v.dtype)
        tau = enc_tau.masked_fill(~hist, 0.0)
        k = base.content_key(v)
        scale = v.size(-1) ** 0.5
        u = torch.einsum("bmd,d->bm", k, base.content_query) / scale
        u = u.masked_fill(~hist, 0.0)

        pi = torch.softmax(self.f_mix(v), dim=-1) * hist_f.unsqueeze(-1)
        z = base.z_of(age)
        z_gate = z if gate_age is None else base.z_of(gate_age)
        z_col = z_gate.unsqueeze(-1)
        theta = self.theta_k.view(1, -1)
        if self.variant == "shared_beta_mixture" and self.age_temporal:
            raw = theta + self.beta * z_col
        elif self.variant == "component_beta_mixture" and self.age_temporal:
            raw = theta + self.beta_k.view(1, -1) * z_col
        else:
            raw = theta.expand(z_col.shape[0], self.n_components)
        lam_k = F.softplus(raw)
        decay = torch.exp(-lam_k.unsqueeze(1) * tau.unsqueeze(-1))
        g = (pi * decay).sum(dim=-1) * hist_f

        w = torch.exp(u.clamp(max=CONTENT_SCORE_EXP_CLAMP)) * g * hist_f
        weighted = (w.unsqueeze(-1) * v).sum(dim=1)
        z1 = z.unsqueeze(-1)
        history_logit = base.history_head(weighted)
        age_logit = base.age_head(z1)
        total = history_logit + age_logit + base.bias
        if not return_parts:
            return total
        return {
            "logits": total,
            "history_logit": history_logit,
            "age_logit": age_logit,
            "g": g,
            "w": w,
            "pi": pi,
            "lambda_k": lam_k,
            "v": v,
            "u": u,
        }


def build_atomic(cfg: dict[str, Any], n_codes: int, n_targets: int, *, age_temporal: bool) -> AtomicDTR:
    return AtomicDTR(
        n_codes,
        n_targets,
        variant=str(cfg["variant"]),
        age_temporal=age_temporal,
        d_model=int(cfg["d_model"]),
        dropout=float(cfg["dropout"]),
        lambda_init=float(cfg.get("lambda_init", 0.1)),
        n_components=int(cfg.get("n_components", 3)),
    )
