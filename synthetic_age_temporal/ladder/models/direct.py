"""E01 direct evidence and E03 composition-plus-mass. Same module, aggregation flag only."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_dtr import EncounterEncoder

from ladder.models.common import LadderModule, inverse_softplus_value, masked_tau, z_of

AGGREGATIONS = ("raw_additive", "mass")


class DirectEvidenceDTR(LadderModule):
    """Minimal developmental retrieval.

    v_m = f_enc(C_m)
    lambda(a) = softplus(theta0 + beta z(a))
    g_m = exp(-lambda(a) tau_m)
    raw:    H = sum_m g_m v_m
    mass:   H = concat(sum g v / (M+eps), log1p(M))
    logits = W_history H + W_age z + b
    """

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 64,
        dropout: float = 0.0,
        age_temporal: bool = True,
        lambda_init: float = 0.1,
        aggregation: str = "raw_additive",
        mass_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation}")
        self.architecture = "direct"
        self.aggregation = aggregation
        self.age_temporal = bool(age_temporal)
        self.lambda_init = float(lambda_init)
        self.mass_eps = float(mass_eps)
        self.d_model = int(d_model)

        self.encounter_encoder = EncounterEncoder(n_codes, d_model, dropout=dropout)
        theta = inverse_softplus_value(self.lambda_init)
        self.theta0 = nn.Parameter(torch.tensor([theta], dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1))
        hist_dim = d_model + (1 if aggregation == "mass" else 0)
        self.W_history = nn.Linear(hist_dim, n_targets, bias=False)
        self.W_age = nn.Linear(1, n_targets, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self.configure_arm_()

    def aggregate(self, g: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weighted = (g.unsqueeze(-1) * v).sum(dim=1)
        mass = g.sum(dim=1, keepdim=True)
        if self.aggregation == "raw_additive":
            return weighted, mass
        h_bar = weighted / (mass + self.mass_eps)
        history = torch.cat([h_bar, torch.log1p(mass)], dim=-1)
        return history, mass

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
        lam = self.developmental_lambda(age).unsqueeze(-1)
        g = torch.exp(-lam * tau) * hist.to(v.dtype)
        history, mass = self.aggregate(g, v)
        logits, history_logit, age_logit = self.additive_readout(history, z)
        if not return_parts:
            return logits
        return {
            "logits": logits,
            "history_logit": history_logit,
            "age_logit": age_logit,
            "g": g,
            "v": v,
            "H": history,
            "M": mass,
            "lambda": lam.squeeze(-1),
            "z": z,
        }
