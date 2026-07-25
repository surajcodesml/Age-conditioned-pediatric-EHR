#!/usr/bin/env python3
"""Chebyshev temporal basis -- the single place Chebyshev polynomials are evaluated.

    tau       = log1p(delta_t_days / 7)          # computed once, in the collate
    tau_tilde = 2 * tau / tau_max - 1            # here
    log_w     = sum_{k=1..s} alpha_k * T_k(tau_tilde)

Two corrections to the legacy kernel live here.

D1 -- basis conditioning. The legacy kernel used raw monomials ``[1, tau, ..., tau^5]``
on ``tau in [0, ~6.5]``, whose Gram matrix on the empirical lag distribution has condition
number ~2.1e9. Chebyshev polynomials on ``tau_tilde in [-1, 1]`` bring that to ~46.

D2 -- no constant term. ``T_0`` is dropped entirely. Softmax is invariant to a per-row
constant, so ``alpha_0`` cannot change any attention weight; carrying it only gives the
optimizer a free direction that does nothing. This holds at both kernel sites and in the
age-conditioned case, because within one softmax row the query age -- and therefore
``alpha_0(a)`` -- is fixed. Hence ``s = 5`` with indices 1..5, and ``T_0`` is never exposed.

Coefficients are **zero-initialised**, so ``log_w == 0`` at the start of training and the
population decay is learned rather than assumed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ChebyshevKernel", "chebyshev_basis", "DEFAULT_S"]

DEFAULT_S = 5


def chebyshev_basis(x: torch.Tensor, s: int) -> torch.Tensor:
    """``T_1 ... T_s`` evaluated at ``x``, stacked on a new trailing axis -> ``[*x.shape, s]``.

    Uses the standard recurrence ``T_0 = 1, T_1 = x, T_{k+1} = 2 x T_k - T_{k-1}``. ``T_0``
    is needed to start the recurrence but is never returned (D2).
    """
    if s < 1:
        raise ValueError(f"s must be >= 1, got {s}")
    t_prev = torch.ones_like(x)  # T_0, used only to seed the recurrence
    t_curr = x  # T_1
    terms = [t_curr]
    for _ in range(s - 1):
        t_next = 2.0 * x * t_curr - t_prev
        t_prev, t_curr = t_curr, t_next
        terms.append(t_curr)
    return torch.stack(terms, dim=-1)


class ChebyshevKernel(nn.Module):
    """Evaluate ``log_w = sum_{k=1..s} alpha_k T_k(tau_tilde)``.

    ``alpha`` broadcasts on the trailing axis, so one implementation serves both kernel
    sites:

      * encoder -- ``tau [B, L, L]`` with ``alpha [B, L, s]``; ``alpha[..., k:k+1]`` is
        ``[B, L, 1]`` and broadcasts along the **key** axis, so attention row ``i`` is
        conditioned on the query age ``a_i`` alone.
      * pooling -- ``tau [B, L]`` with ``alpha [B, s]``.
      * diagnostics -- ``tau [G]`` with ``alpha [s]``.

    ``tau_max`` is a persistent buffer so it serialises with the checkpoint and cannot be
    recomputed per dataset (D8). Clamping of ``tau_tilde`` is counted rather than silent;
    the rate is a MEASURE quantity read by ``diagnostics``.
    """

    def __init__(self, s: int = DEFAULT_S, tau_max: float = 6.5) -> None:
        super().__init__()
        self.s = int(s)
        if self.s < 1:
            raise ValueError(f"s must be >= 1, got {s}")
        if not (tau_max > 0):
            raise ValueError(f"tau_max must be > 0, got {tau_max}")
        self.register_buffer("tau_max", torch.tensor(float(tau_max), dtype=torch.float32),
                             persistent=True)
        # Running clamp counters. Non-persistent: they describe a run, not the model.
        self.register_buffer("clamp_count", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("total_count", torch.zeros((), dtype=torch.float64), persistent=False)

    def extra_repr(self) -> str:
        return f"s={self.s}, tau_max={float(self.tau_max):.6f}"

    @torch.no_grad()
    def reset_clamp_stats(self) -> None:
        self.clamp_count.zero_()
        self.total_count.zero_()

    @property
    def clamp_fraction(self) -> float:
        total = float(self.total_count)
        return float(self.clamp_count) / total if total > 0 else 0.0

    def rescale(self, tau: torch.Tensor, *, count: bool = True) -> torch.Tensor:
        """``tau -> tau_tilde``, clamped to ``[-1, 1]`` (INV-DOMAIN)."""
        tau_tilde = 2.0 * tau / self.tau_max.to(tau.dtype) - 1.0
        if count and self.total_count is not None:
            with torch.no_grad():
                out = (tau_tilde < -1.0) | (tau_tilde > 1.0)
                self.clamp_count += out.sum().to(self.clamp_count.dtype)
                self.total_count += torch.tensor(
                    float(out.numel()), dtype=self.total_count.dtype, device=self.total_count.device
                )
        return tau_tilde.clamp(-1.0, 1.0)

    def forward(self, tau: torch.Tensor, alpha: torch.Tensor, *, count: bool = True) -> torch.Tensor:
        """-> ``log_w`` with the shape of ``tau``."""
        if alpha.shape[-1] != self.s:
            raise ValueError(f"alpha must have trailing dim s={self.s}, got {tuple(alpha.shape)}")
        tau_tilde = self.rescale(tau, count=count)
        basis = chebyshev_basis(tau_tilde, self.s)  # [*tau.shape, s]
        log_w = torch.zeros_like(tau_tilde)
        for k in range(self.s):
            log_w = log_w + alpha[..., k : k + 1] * basis[..., k]
        return log_w


def _smoke() -> None:
    """Match numpy's chebval with a zeroed constant term. Routed through diagnostics."""
    import numpy as np
    from numpy.polynomial import chebyshev as npcheb

    from model_new import diagnostics

    torch.manual_seed(0)
    s, tau_max = DEFAULT_S, 6.5
    kern = ChebyshevKernel(s=s, tau_max=tau_max)
    tau = torch.linspace(0.0, tau_max, 257, dtype=torch.float64)
    alpha = torch.randn(s, dtype=torch.float64)

    got = kern(tau, alpha).numpy()
    x = (2.0 * tau.numpy() / tau_max - 1.0).clip(-1.0, 1.0)
    want = npcheb.chebval(x, np.concatenate([[0.0], alpha.numpy()]))  # c_0 == 0 (D2)
    err = float(np.max(np.abs(got - want)))

    zero = ChebyshevKernel(s=s, tau_max=tau_max)
    log_w0 = zero(tau.float(), torch.zeros(s))

    diagnostics.print_block(
        "basis.py smoke",
        [
            f"max |cheb - numpy.chebval|      : {err:.3e}   (tol 1e-6)",
            f"log_w at zero-init alpha        : max|.| = {float(log_w0.abs().max()):.3e}",
            f"basis returns T_1..T_{s}           : shape {tuple(chebyshev_basis(tau, s).shape)}",
            f"clamp fraction on [0, tau_max]  : {kern.clamp_fraction:.6f}",
        ],
    )
    if err > 1e-6:
        raise AssertionError(f"[INV-BASIS] chebval mismatch {err:.3e} > 1e-6")


if __name__ == "__main__":
    _smoke()
