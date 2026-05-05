#!/usr/bin/env python3
"""Age feature embedding and age-conditioned coefficient generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierAgeEmbedding(nn.Module):
    """Fixed Fourier embedding of age in years."""

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
    """Generate additive polynomial coefficients from age features."""

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

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        g = torch.Generator()
        g.manual_seed(0)
        random_constant = torch.randn(self.in_dim, generator=g) * 0.5
        self.register_buffer("random_constant", random_constant)

    def forward(self, age_features: torch.Tensor) -> torch.Tensor:
        if age_features.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim={self.in_dim}, got {age_features.shape[-1]}")

        if self.mode == "none":
            out_shape = age_features.shape[:-1] + (self.out_dim,)
            return torch.zeros(out_shape, dtype=age_features.dtype, device=age_features.device)

        if self.mode == "real":
            x = age_features
        elif self.mode == "random_constant":
            x = self.random_constant.expand_as(age_features)
        else:
            raise RuntimeError(f"Unsupported mode: {self.mode}")

        return self.mlp(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    emb = FourierAgeEmbedding()
    age = torch.linspace(0.0, 100.0, steps=40, dtype=torch.float32).view(4, 10)
    feat = emb(age)
    assert feat.shape == (4, 10, 32)
    age_extreme = torch.tensor([[0.0, 100.0]], dtype=torch.float32)
    feat_extreme = emb(age_extreme)
    assert torch.isfinite(feat_extreme).all()

    min_freq = float(emb.frequencies.min())
    max_freq = float(emb.frequencies.max())
    assert math.isclose(min_freq, 1.0 / 200.0, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(max_freq, 12.0, rel_tol=0.0, abs_tol=1e-6)
    assert bool(torch.all(emb.frequencies[1:] > emb.frequencies[:-1]))

    age_features = emb(torch.tensor([[2.0, 65.0]], dtype=torch.float32))
    gen_real = AgeCoefficientGenerator(mode="real")
    gen_rand = AgeCoefficientGenerator(mode="random_constant")
    gen_none = AgeCoefficientGenerator(mode="none")

    out_real_0 = gen_real(age_features)
    out_rand_0 = gen_rand(age_features)
    out_none_0 = gen_none(age_features)
    assert torch.equal(out_real_0, torch.zeros_like(out_real_0))
    assert torch.equal(out_rand_0, torch.zeros_like(out_rand_0))
    assert torch.equal(out_none_0, torch.zeros_like(out_none_0))

    with torch.no_grad():
        gen_real.mlp[-1].weight.copy_(torch.randn_like(gen_real.mlp[-1].weight) * 0.1)
        gen_real.mlp[-1].bias.copy_(torch.randn_like(gen_real.mlp[-1].bias) * 0.1)
        gen_rand.mlp[-1].weight.copy_(torch.randn_like(gen_rand.mlp[-1].weight) * 0.1)
        gen_rand.mlp[-1].bias.copy_(torch.randn_like(gen_rand.mlp[-1].bias) * 0.1)

    out_real = gen_real(age_features)
    out_rand = gen_rand(age_features)
    out_none = gen_none(age_features)
    assert not torch.allclose(out_real[0, 0], out_real[0, 1], atol=1e-6)
    assert not torch.allclose(out_real, out_rand, atol=1e-6)
    assert torch.equal(out_none, torch.zeros_like(out_none))
    assert torch.allclose(out_rand[0, 0], out_rand[0, 1], atol=1e-6)

    print("Smoke test passed.")
