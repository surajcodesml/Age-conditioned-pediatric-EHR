#!/usr/bin/env python3
"""INV-HEATMAP-ZERO -- Panel A is identically zero when Δα ≡ 0; τ̃ matches lag_to_tau.

The figure path must not invent a lag→τ or Chebyshev convention of its own. Setting the
age-conditioning residual to zero (vanilla arm, or an explicit zero Δα tensor) must leave
the softmax-valid deviation plane at machine zero, and day-lags fed through the figure's
τ̃ helper must match ``data.lag_to_tau`` followed by the same ``τ_max`` clip the kernel uses.
"""

from __future__ import annotations

import numpy as np
import torch

from model_new.basis import ChebyshevKernel
from model_new.data import lag_to_tau
from model_new.figures.fig_age_kernel_heatmap import (
    build_lag_grid_days,
    panel_a_delta,
    tau_tilde_from_days,
)
from model_new.tests.conftest import wake_generators_


TOL_ZERO = 1e-6
TOL_TAU = 1e-6


def test_panel_a_zero_when_delta_alpha_is_zero():
    """Acceptance #1: Δα ≡ 0 ⇒ Panel A is identically zero (float tol)."""
    torch.manual_seed(0)
    s, tau_max = 5, 6.5
    alpha_base = torch.randn(s)
    ages = torch.linspace(0.0, 90.0, 17)
    delta = torch.zeros(ages.numel(), s)
    days = build_lag_grid_days(64)
    tau_t = tau_tilde_from_days(days, tau_max)
    plane = panel_a_delta(alpha_base, delta, tau_t)
    assert float(np.max(np.abs(plane))) < TOL_ZERO, (
        f"[INV-HEATMAP-ZERO] nonzero Panel A with Δα≡0: max|Δ|={float(np.max(np.abs(plane))):.3e}"
    )


def test_panel_a_zero_on_vanilla_site(model_factory):
    """Vanilla arm has mode='none' generators; Δα is structurally zero at every age."""
    m = model_factory("vanilla").eval()
    site = dict(m.kernel_sites())["encoder_layer0"]
    ages = torch.linspace(0.0, 90.0, 21)
    with torch.no_grad():
        # Give alpha_base a nontrivial population kernel so the zero result is not vacuous.
        site.alpha_base.copy_(torch.randn(site.alpha_base.shape))
        da = site.age(ages)
        assert float(da.abs().max()) == 0.0
        days = build_lag_grid_days(48)
        tau_t = tau_tilde_from_days(days, float(site.kernel.tau_max))
        plane = panel_a_delta(site.alpha_base.detach(), da, tau_t)
    assert float(np.max(np.abs(plane))) < TOL_ZERO


def test_panel_a_nonzero_when_delta_alpha_awake(model_factory):
    """Sanity: a woken kernel arm yields a non-trivial (non-acceptance) plane."""
    m = model_factory("kernel").eval()
    wake_generators_(m, scale=1.0, seed=3)
    site = dict(m.kernel_sites())["encoder_layer0"]
    with torch.no_grad():
        site.alpha_base.copy_(torch.randn(site.alpha_base.shape) * 0.3)
        ages = torch.tensor([0.5, 2.0, 15.0, 40.0, 70.0])
        da = site.age(ages)
        days = build_lag_grid_days(48)
        tau_t = tau_tilde_from_days(days, float(site.kernel.tau_max))
        plane = panel_a_delta(site.alpha_base.detach(), da, tau_t)
    assert float(np.max(np.abs(plane))) > 1e-4


def test_tau_tilde_matches_lag_to_tau_then_rescale():
    """Acceptance #2: figure τ̃ equals lag_to_tau → ChebyshevKernel.rescale."""
    tau_max = 6.72380256652832
    days = np.array(
        [1.0 / 24.0, 1.0, 7.0, 30.0, 365.25, 5.0 * 365.25, 10.0 * 365.25],
        dtype=np.float64,
    )
    got = tau_tilde_from_days(days, tau_max)
    kern = ChebyshevKernel(s=5, tau_max=tau_max)
    want = kern.rescale(lag_to_tau(torch.as_tensor(days, dtype=torch.float64)), count=False)
    err = float((got.double() - want.double()).abs().max())
    assert err < TOL_TAU, f"[INV-HEATMAP-ZERO] τ̃ mismatch {err:.3e} > {TOL_TAU}"
