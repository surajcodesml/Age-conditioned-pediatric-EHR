"""Content bottleneck audit + E01 target-conditioned retrieval final test."""
from __future__ import annotations

import ladder

REPO_ROOT = ladder.REPO_ROOT
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "content_bottleneck_final"
C01_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "dtr_atomic_followup"
CONFIG_PATH = REPO_ROOT / "configs" / "content_bottleneck" / "base.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "content_bottleneck_final_test.md"
