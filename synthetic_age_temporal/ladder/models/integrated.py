"""E06 developmental integrated hazard. Small positive piecewise-linear rho."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DAYS_PER_YEAR
from model_dtr import EncounterEncoder

from ladder.models.common import LadderModule, inverse_softplus_value, z_of


def integrate_piecewise_linear(
    rho_knots: torch.Tensor,
    knots: torch.Tensor,
    a0: torch.Tensor,
    a1: torch.Tensor,
) -> torch.Tensor:
    """Exact integral of a knot-linear rho with constant extrapolation outside the knots.

    a0 and a1 broadcast. The width terms assume a1 >= a0; callers enforce that.
    """
    a0 = torch.as_tensor(a0, dtype=rho_knots.dtype, device=rho_knots.device)
    a1 = torch.as_tensor(a1, dtype=rho_knots.dtype, device=rho_knots.device)
    a0, a1 = torch.broadcast_tensors(a0, a1)
    total = torch.zeros_like(a0)
    n_knots = int(knots.numel())
    for i in range(n_knots - 1):
        lo = knots[i]
        hi = knots[i + 1]
        seg = (hi - lo).clamp_min(1e-8)
        left = torch.maximum(a0, lo)
        right = torch.minimum(a1, hi)
        width = (right - left).clamp_min(0)
        t_left = ((left - lo) / seg).clamp(0, 1)
        t_right = ((right - lo) / seg).clamp(0, 1)
        r0 = rho_knots[i]
        r1 = rho_knots[i + 1]
        rho_left = r0 + (r1 - r0) * t_left
        rho_right = r0 + (r1 - r0) * t_right
        total = total + 0.5 * (rho_left + rho_right) * width
    left_end = torch.minimum(a1, knots[0])
    total = total + rho_knots[0] * (left_end - a0).clamp_min(0)
    right_start = torch.maximum(a0, knots[-1])
    total = total + rho_knots[-1] * (a1 - right_start).clamp_min(0)
    return total


class IntegratedHazardDTR(LadderModule):
    """g_m = exp(-∫_{age_event}^{age_current} rho(s) ds).

    rho knot values are softplus(theta0 + beta_k). beta=0 makes rho constant,
    which is the temporal-only hazard.
    """

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 64,
        dropout: float = 0.0,
        age_temporal: bool = True,
        lambda_init: float = 0.1,
        knots: tuple[float, ...] | list[float] = (0.0, 6.0, 12.0, 18.0),
        integral_clip: float = 80.0,
    ) -> None:
        super().__init__()
        knot_values = [float(k) for k in knots]
        if len(knot_values) < 2 or any(b <= a for a, b in zip(knot_values, knot_values[1:])):
            raise ValueError(f"knots must be strictly increasing, got {knot_values}")
        self.architecture = "integrated_hazard"
        self.age_temporal = bool(age_temporal)
        self.lambda_init = float(lambda_init)
        self.integral_clip = float(integral_clip)
        self.d_model = int(d_model)
        self.n_knots = len(knot_values)

        self.encounter_encoder = EncounterEncoder(n_codes, d_model, dropout=dropout)
        self.register_buffer("knots", torch.tensor(knot_values, dtype=torch.float32))
        theta = inverse_softplus_value(self.lambda_init)
        self.theta0 = nn.Parameter(torch.tensor([theta], dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(self.n_knots))
        self.W_history = nn.Linear(d_model, n_targets, bias=False)
        self.W_age = nn.Linear(1, n_targets, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self.configure_arm_()

    def rho_knots(self) -> torch.Tensor:
        if self.age_temporal:
            raw = self.theta0 + self.beta
        else:
            raw = self.theta0.expand_as(self.beta)
        return F.softplus(raw)

    def rho_of(self, age: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear interpolation of rho, constant outside the knots."""
        rho = self.rho_knots()
        knots = self.knots
        age_f = age.to(dtype=rho.dtype)
        out = torch.zeros_like(age_f)
        # Left of the first knot.
        out = torch.where(age_f <= knots[0], rho[0].expand_as(age_f), out)
        for i in range(self.n_knots - 1):
            lo = knots[i]
            hi = knots[i + 1]
            t = ((age_f - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
            piece = rho[i] + (rho[i + 1] - rho[i]) * t
            inside = (age_f >= lo) & (age_f <= hi)
            out = torch.where(inside, piece, out)
        out = torch.where(age_f >= knots[-1], rho[-1].expand_as(age_f), out)
        return out

    def integrate(self, age_event: torch.Tensor, age_current: torch.Tensor) -> torch.Tensor:
        return integrate_piecewise_linear(self.rho_knots(), self.knots, age_event, age_current)

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
        del enc_tau  # E06 uses elapsed age, not tau. Accepted so the batch interface stays shared.
        if enc_lag_days is None:
            raise ValueError("integrated hazard requires enc_lag_days")
        v = self.encounter_encoder(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        lag = enc_lag_days.masked_fill(~hist, 0.0)
        age_current = age.unsqueeze(-1)
        age_event = age_current - lag / DAYS_PER_YEAR
        age_event = torch.minimum(age_event, age_current)
        integral = self.integrate(age_event, age_current).clamp(0, self.integral_clip)
        g = torch.exp(-integral) * hist.to(v.dtype)
        z = z_of(age)
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
            "H": history,
            "integral": integral,
            "age_event": age_event,
            "z": z,
            "rho_knots": self.rho_knots(),
        }
