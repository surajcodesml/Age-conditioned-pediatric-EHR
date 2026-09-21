"""Stage-1 MIMIC-IV pretraining with minimal age × temporal attention.

Replaces the Fourier/Chebyshev DKM kernel with two shared scalars:

    s_ij = q_i^T k_j / sqrt(d) - [λ0 + β z(a_i)] τ_ij

Does not import ``model_new.basis``, ``model_new.age_encoding``, or ``DKMModel``.
Data loading, lag convention, future-visit targets, and padding masks are reused
from ``model_new.data`` / ``model_new.encoder``.
"""
from __future__ import annotations

from stage1_mimic_pretrain.config import ARMS, resolve_arm

__all__ = ["ARMS", "resolve_arm"]
