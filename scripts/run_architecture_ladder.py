#!/usr/bin/env python3
"""Entry point. Run with the ehr environment interpreter."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "synthetic_age_temporal"))

from ladder.run_ladder import main

if __name__ == "__main__":
    main()
