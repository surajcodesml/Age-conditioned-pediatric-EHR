"""E05 structured persistence mixture. Content-only pi, K developmental rates."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_dtr import EncounterEncoder

from ladder.models.common import LadderModule, inverse_softplus_value, masked_tau, z_of


class PersistenceMixtureDTR(LadderModule):
    """pi_m = softmax(f(v_m)); g_m = sum_k pi_mk exp(-lambda_k(a) tau_m)."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 64,
        dropout: float = 0.0,
        age_temporal: bool = True,
        lambda_init: float = 0.1,
        n_components: int = 3,
    ) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.architecture = "mixture"
        self.age_temporal = bool(age_temporal)
        self.lambda_init = float(lambda_init)
        self.n_components = int(n_components)
        self.d_model = int(d_model)

        self.encounter_encoder = EncounterEncoder(n_codes, d_model, dropout=dropout)
        self.f_persistence = nn.Linear(d_model, self.n_components)
        theta = inverse_softplus_value(self.lambda_init)
        self.theta_k = nn.Parameter(torch.full((self.n_components,), theta))
        self.beta_k = nn.Parameter(torch.zeros(self.n_components))
        self.W_history = nn.Linear(d_model, n_targets, bias=False)
        self.W_age = nn.Linear(1, n_targets, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self.configure_arm_()

    def theta_param(self) -> nn.Parameter:
        return self.theta_k

    def beta_param(self) -> nn.Parameter:
        return self.beta_k

    def mixture_weights(self, v: torch.Tensor) -> torch.Tensor:
        """pi depends only on encounter content."""
        return torch.softmax(self.f_persistence(v), dim=-1)

    def lambda_components(self, age: torch.Tensor) -> torch.Tensor:
        """lambda_k(a), shape [..., K]."""
        z = z_of(age)
        # z: [...] -> [..., 1]; theta_k/beta_k: [K]
        z_col = z.unsqueeze(-1)
        if self.age_temporal:
            raw = self.theta_k + self.beta_k * z_col
        else:
            raw = self.theta_k.expand(*z_col.shape[:-1], self.n_components)
        return F.softplus(raw)

    def forward(
        self,
        enc_code_ids: torch.Tensor,
        enc_code_mask: torch.Tensor,
        enc_tau: torch.Tensor,
        enc_padding_mask: torch.Tensor,
        age: torch.Tensor,
        enc_lag_days: torch.Tensor | None = None,
        return_parts: bool = False,
        **_: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        del enc_lag_days
        v = self.encounter_encoder(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        tau = masked_tau(enc_tau, hist)
        z = z_of(age)
        pi = self.mixture_weights(v) * hist.to(v.dtype).unsqueeze(-1)
        lam_k = self.lambda_components(age)  # [B, K]
        # g_m = sum_k pi_mk * exp(-lambda_k * tau_m)
        decay = torch.exp(-lam_k.unsqueeze(1) * tau.unsqueeze(-1))
        g = (pi * decay).sum(dim=-1) * hist.to(v.dtype)
        history = (g.unsqueeze(-1) * v).sum(dim=1)
        logits, history_logit, age_logit = self.additive_readout(history, z)
        if not return_parts:
            return logits
        return {
            "logits": logits,
            "history_logit": history_logit,
            "age_logit": age_logit,
            "g": g,
            "v": v,
            "pi": pi,
            "lambda_k": lam_k,
            "H": history,
            "z": z,
        }
