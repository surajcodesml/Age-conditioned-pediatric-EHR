"""Stage-2 pediatric adaptation on processed NCH SleepBank sequences.

Continued longitudinal pretraining from a Stage-1 MIMIC checkpoint, with a
fixed pediatric age map z_P(a)=(a-9)/9 and two matched arms that differ only
in whether beta_P is trainable.
"""
from __future__ import annotations

from stage2_nch.config import ARMS, resolve_arm

__all__ = ["ARMS", "resolve_arm"]
