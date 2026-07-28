"""Offline age-route audit of finished pretraining checkpoints.

Nothing here trains or fine-tunes. Nothing outside ``model_new/audit/`` is modified.
All console / JSON output is routed through ``model_new.diagnostics`` (D11).
"""

from __future__ import annotations

__all__ = ["AUDIT_SEED", "SUPPORT_AGE_MIN", "INTERVENTION_AGES", "AGE_SHIFTS"]

# Fixed for every metric that samples; recorded in age_audit.json.
AUDIT_SEED = 0

# Youngest event age on the MIMIC pretrain corpus (paper_numbers / corpus_stats).
SUPPORT_AGE_MIN = 16.6

INTERVENTION_AGES = (1.0, 5.0, 12.0, 25.0, 50.0, 75.0, 89.0)
AGE_SHIFTS = (-20.0, -10.0, 10.0, 20.0)
