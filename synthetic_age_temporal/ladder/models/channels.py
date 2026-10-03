"""E04 content channels with one shared developmental lambda."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_dtr import EncounterEncoder

from ladder.models.common import LadderModule, inverse_softplus_value, masked_tau, z_of


class ContentChannelDTR(LadderModule):
    """H content channels. q_h sees v_m only. One shared lambda(a)."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 64,
        dropout: float = 0.0,
        age_temporal: bool = True,
        lambda_init: float = 0.1,
        n_channels: int = 4,
    ) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        self.architecture = "channels"
        self.age_temporal = bool(age_temporal)
        self.lambda_init = float(lambda_init)
        self.n_channels = int(n_channels)
        self.d_model = int(d_model)

        self.encounter_encoder = EncounterEncoder(n_codes, d_model, dropout=dropout)
        self.q_h = nn.Parameter(torch.randn(self.n_channels, d_model) * 0.02)
        theta = inverse_softplus_value(self.lambda_init)
        self.theta0 = nn.Parameter(torch.tensor([theta], dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1))
        self.W_history = nn.Linear(self.n_channels * d_model, n_targets, bias=False)
        self.W_age = nn.Linear(1, n_targets, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self.configure_arm_()

    def channel_scores(self, v: torch.Tensor) -> torch.Tensor:
        """c_mh = sigmoid(q_h^T v_m). Content only."""
        return torch.sigmoid(torch.einsum("bmd,hd->bmh", v, self.q_h))

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
        c = self.channel_scores(v)
        c = c * hist.to(v.dtype).unsqueeze(-1)
        # E_h = sum_m c_mh * g_m * v_m
        evidence = torch.einsum("bmh,bm,bmd->bhd", c, g, v)
        history = evidence.reshape(evidence.size(0), -1)
        logits, history_logit, age_logit = self.additive_readout(history, z)
        if not return_parts:
            return logits
        return {
            "logits": logits,
            "history_logit": history_logit,
            "age_logit": age_logit,
            "g": g,
            "v": v,
            "c": c,
            "channel_scores": c,
            "E": evidence,
            "H": history,
            "lambda": lam.squeeze(-1),
            "z": z,
        }
