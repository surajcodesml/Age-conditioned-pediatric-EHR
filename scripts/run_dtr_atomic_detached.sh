#!/usr/bin/env bash
# Detached C00–C06 run. Survives IDE disconnect.
set -euo pipefail
cd /home/suraj/Git/Age-conditioned-pediatric-EHR/synthetic_age_temporal
PY=/home/suraj/miniconda3/envs/ehr/bin/python
LOG=/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/run.log
mkdir -p "$(dirname "$LOG")"
{
  echo "=== start $(date -Is) pid $$ ==="
  echo "=== contract tests ==="
  "$PY" tests/test_atomic_followup.py
  echo "=== smoke ==="
  "$PY" -m atomic.run_atomic --mode smoke
  echo "=== full ==="
  "$PY" -m atomic.run_atomic --mode full --jobs 2
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
