#!/usr/bin/env python3
"""ChebyshevKernel matches numpy.polynomial.chebyshev.chebval with a zeroed constant term."""

from __future__ import annotations

import numpy as np
import torch
from numpy.polynomial import chebyshev as npcheb

from model_new.basis import ChebyshevKernel, chebyshev_basis


def test_matches_numpy_chebval():
    torch.manual_seed(0)
    tau_max, s = 6.5, 5
    k = ChebyshevKernel(s=s, tau_max=tau_max)
    tau = torch.linspace(0.0, tau_max, 513, dtype=torch.float64)
    for trial in range(5):
        alpha = torch.randn(s, dtype=torch.float64, generator=torch.Generator().manual_seed(trial))
        got = k(tau, alpha).numpy()
        x = np.clip(2.0 * tau.numpy() / tau_max - 1.0, -1.0, 1.0)
        want = npcheb.chebval(x, np.concatenate([[0.0], alpha.numpy()]))
        assert np.max(np.abs(got - want)) < 1e-6


def test_basis_columns_match_numpy():
    x = np.linspace(-1, 1, 101)
    got = chebyshev_basis(torch.tensor(x), 5).numpy()
    for k in range(1, 6):
        coef = np.zeros(k + 1)
        coef[k] = 1.0
        assert np.max(np.abs(got[:, k - 1] - npcheb.chebval(x, coef))) < 1e-10


def test_broadcasting_shapes():
    k = ChebyshevKernel(s=5, tau_max=6.5)
    assert k(torch.rand(2, 4, 4), torch.rand(2, 4, 5)).shape == (2, 4, 4)   # encoder
    assert k(torch.rand(2, 4), torch.rand(2, 5)).shape == (2, 4)            # pooling
    assert k(torch.rand(9), torch.rand(5)).shape == (9,)                    # diagnostics
