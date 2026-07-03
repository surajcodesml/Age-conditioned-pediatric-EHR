#!/usr/bin/env python3
"""Age feature embedding and the two age-conditioning pathways.

Self-contained copy for the ablation package. Two pathways share the same fixed
``FourierAgeEmbedding``:

  * kernel pathway  -> :class:`AgeCoefficientGenerator` -> Delta-alpha on the
    temporal-kernel polynomial coefficients.
  * additive pathway -> :class:`AdditiveAgeEmbedding` -> a delta added to the
    per-event code embeddings before mlp_q/k/v.

The Fourier banding is intentionally left fine (min period 1 month) so the
0-2y range is resolved rather than collapsed into one bucket.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierAgeEmbedding(nn.Module):
    """Fixed Fourier embedding of age in YEARS.

    Frequencies are geometrically spaced between ``min_period_years`` (default
    1 month) and ``max_period_years`` (default 200 y). The 1-month min period is
    kept on purpose: the pediatric 0-2y range needs sub-year resolution.
    """

    def __init__(
        self,
        num_frequencies: int = 16,
        min_period_years: float = 1.0 / 12.0,
        max_period_years: float = 200.0,
    ) -> None:
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.min_period_years = float(min_period_years)
        self.max_period_years = float(max_period_years)
        if self.num_frequencies <= 0:
            raise ValueError("num_frequencies must be > 0")
        if self.min_period_years <= 0.0 or self.max_period_years <= 0.0:
            raise ValueError("period bounds must be > 0")
        if self.max_period_years < self.min_period_years:
            raise ValueError("max_period_years must be >= min_period_years")

        log_periods = torch.linspace(
            math.log(self.max_period_years),
            math.log(self.min_period_years),
            steps=self.num_frequencies,
            dtype=torch.float32,
        )
        periods = torch.exp(log_periods)
        frequencies = 1.0 / periods
        self.register_buffer("frequencies", frequencies)

    @property
    def embedding_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, age_years: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * age_years.unsqueeze(-1) * self.frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AgeCoefficientGenerator(nn.Module):
    """Kernel pathway: age features -> additive polynomial-coefficient deltas.

    ``mode`` reuses the frozen package's semantics so ``--arm`` can drive it:
      * ``real``            - Delta-alpha = MLP(phi(real_age)).
      * ``random_constant`` - Delta-alpha = MLP(phi(fixed_constant_age));
        architecturally identical to ``real`` but no real-age signal.
      * ``none``            - Delta-alpha identically zero (vanilla / additive arms).

    Zero-init of the final layer keeps Delta-alpha = 0 at initialization so every
    arm starts exactly at the vanilla kernel.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 6,
        mode: str = "real",
    ) -> None:
        super().__init__()
        if mode not in {"real", "random_constant", "none"}:
            raise ValueError("mode must be one of {'real', 'random_constant', 'none'}")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.mode = mode

        # Build parameters only when the pathway is live, so vanilla/additive arms
        # carry ZERO kernel-age parameters and their state_dict matches the vanilla
        # backbone exactly.
        if mode == "none":
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, age_features: torch.Tensor) -> torch.Tensor:
        if age_features.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim={self.in_dim}, got {age_features.shape[-1]}")
        if self.mode == "none" or self.mlp is None:
            out_shape = age_features.shape[:-1] + (self.out_dim,)
            return torch.zeros(out_shape, dtype=age_features.dtype, device=age_features.device)
        # For "real" and "random_constant" the *caller* controls the age fed to the
        # Fourier embedding (real age vs a fixed constant), so both use the MLP here.
        return self.mlp(age_features)

    def num_pathway_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdditiveAgeEmbedding(nn.Module):
    """Additive pathway: age features -> delta added to code embeddings.

    ``enabled=False`` returns an exact zero delta (vanilla / kernel arms) while
    still constructing the parameters, so the module's parameter count is well
    defined and comparable across arms. Zero-init of the final layer additionally
    makes the delta exactly zero at initialization even when enabled.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 1024,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.enabled = bool(enabled)

        # Parameters exist only when the additive pathway is live (additive arm),
        # so vanilla/kernel arms carry ZERO additive-pathway parameters.
        if not self.enabled:
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, age_features: torch.Tensor) -> torch.Tensor:
        if age_features.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim={self.in_dim}, got {age_features.shape[-1]}")
        if not self.enabled or self.mlp is None:
            out_shape = age_features.shape[:-1] + (self.out_dim,)
            return torch.zeros(out_shape, dtype=age_features.dtype, device=age_features.device)
        return self.mlp(age_features)

    def num_pathway_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
