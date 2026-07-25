#!/usr/bin/env python3
"""INV-BASIS -- no constant term exists anywhere in the parameterization."""

from __future__ import annotations

import inspect

import torch

from model_new.basis import ChebyshevKernel, chebyshev_basis
from model_new.arms import ARMS


def test_basis_returns_T1_through_Ts():
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    b = chebyshev_basis(x, 5)
    assert b.shape == (17, 5)
    # T_1 == x. If T_0 had leaked in, column 0 would be all ones.
    assert torch.allclose(b[:, 0], x)
    assert not torch.allclose(b[:, 0], torch.ones_like(x))


def test_alpha_has_exactly_s_entries_at_every_site(model_factory):
    for arm in ARMS:
        m = model_factory(arm)
        for name, site in m.kernel_sites():
            assert site.alpha_base.shape == (m.s,), f"{arm}/{name}"
            assert site.age.out_dim == m.s, f"{arm}/{name}"
            if site.age.generator.mlp is not None:
                assert site.age.generator.mlp[-1].out_features == m.s, f"{arm}/{name}"


def test_kernel_rejects_wrong_alpha_width():
    k = ChebyshevKernel(s=5)
    tau = torch.rand(3, 4)
    for bad in (4, 6):
        try:
            k(tau, torch.zeros(bad))
        except ValueError:
            continue
        raise AssertionError(f"alpha width {bad} was accepted; s=5 is the only legal width")


def test_no_module_exposes_T0():
    """A per-row constant cannot change any attention weight, so T_0 must not be reachable."""
    src = inspect.getsource(chebyshev_basis)
    assert "terms = [t_curr]" in src, "the recurrence must seed with T_1, not T_0"
    k = ChebyshevKernel(s=5)
    log_w = k(torch.rand(5, 6), torch.zeros(5))
    assert torch.equal(log_w, torch.zeros_like(log_w)), "zero alpha must give exactly zero bias"
