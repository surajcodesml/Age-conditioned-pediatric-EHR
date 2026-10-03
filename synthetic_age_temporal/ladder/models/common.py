"""Shared pieces for ladder models. Encoder is the existing DTR encounter encoder."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AGE_CENTER, AGE_SCALE, DAYS_PER_YEAR
from model_dtr import EncounterEncoder


def inverse_softplus_value(y: float) -> float:
    """theta such that softplus(theta) = y, for y > 0."""
    value = float(y)
    if value <= 0.0:
        raise ValueError(f"lambda_init must be positive, got {y}")
    if value > 20.0:
        return value
    return float(math.log(math.expm1(value)))


def z_of(age: torch.Tensor) -> torch.Tensor:
    return (age - AGE_CENTER) / AGE_SCALE


def masked_tau(enc_tau: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
    return enc_tau.masked_fill(~hist, 0.0)


class LadderModule(nn.Module):
    """Arm control shared by every ladder architecture."""

    age_temporal: bool
    architecture: str

    def theta_param(self) -> nn.Parameter:
        return self.theta0

    def beta_param(self) -> nn.Parameter:
        return self.beta

    def temporal_parameter_names(self) -> set[str]:
        return {"theta0", "beta", "theta_k", "beta_k"}

    def configure_arm_(self) -> None:
        self.beta_param().requires_grad_(bool(self.age_temporal))

    def temporal_parameters(self) -> list[nn.Parameter]:
        params = [self.theta_param()]
        beta = self.beta_param()
        if beta.requires_grad:
            params.append(beta)
        return params

    def set_backbone_requires_grad(self, flag: bool) -> None:
        skip = self.temporal_parameter_names()
        for name, param in self.named_parameters():
            leaf = name.split(".")[-1]
            if name in skip or leaf in skip:
                continue
            param.requires_grad_(bool(flag))
        self.configure_arm_()

    def zero_all_betas_(self) -> dict[str, torch.Tensor]:
        beta = self.beta_param()
        saved = beta.detach().clone()
        with torch.no_grad():
            beta.zero_()
        return {"beta": saved}

    def restore_betas_(self, saved: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.beta_param().copy_(saved["beta"])

    def additive_readout(
        self,
        history: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_logit = self.W_history(history)
        age_logit = self.W_age(z.unsqueeze(-1))
        total = history_logit + age_logit + self.bias
        return total, history_logit, age_logit

    def developmental_lambda(self, age: torch.Tensor) -> torch.Tensor:
        """Scalar-beta lambda(a) = softplus(theta0 + beta z(a))."""
        z = z_of(age)
        theta = self.theta_param()
        if self.age_temporal:
            return F.softplus(theta + self.beta_param() * z)
        return F.softplus(theta).expand_as(z)

    def extra_mechanism(self) -> dict[str, Any]:
        return {}
