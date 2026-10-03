"""Controlled DTR architecture ladder on the existing Synthea age x temporal benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

_SAT = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[2]


def _prefer(path: str) -> None:
    """Keep ``path`` at the front. Repo stays ahead of the synthetic package.

    ``synthetic_age_temporal/baselines.py`` would otherwise shadow the
    ``baselines`` package. Bare imports such as ``config`` still resolve
    because the synthetic directory remains on the path.
    """
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


_prefer(str(_SAT))
_prefer(str(_REPO))

REPO_ROOT = _REPO
SAT_ROOT = _SAT
ARTIFACT_ROOT = _REPO / "artifacts" / "synthetic_architecture_ladder"
CONFIG_DIR = _REPO / "configs" / "architecture_ladder"
