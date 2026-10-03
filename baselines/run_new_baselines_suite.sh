#!/usr/bin/env bash
# =============================================================================
# Run MOTOR, TALE-EHR, and NEST sequentially across all experimental stages:
#   1. Synthetic Benchmark (S0–S3, S5) + Counterfactual Mechanism Evaluations
#   2. MIMIC Stage-1 Pretraining (30k vocab next-visit prediction)
#   3. NCH Stage-2 Pediatric Finetuning (transfer from Stage-1 checkpoints)
#
# Usage:
#   # Run directly in background with nohup:
#   nohup bash baselines/run_new_baselines_suite.sh > results/baselines/logs/run_suite_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Or inside a tmux session:
#   tmux new -s new_baselines
#   bash baselines/run_new_baselines_suite.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/suraj/miniconda3/envs/ehr/bin/python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
MODELS="${MODELS:-motor,tale_ehr,nest}"
LOG_DIR="${REPO_ROOT}/results/baselines/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="${LOG_DIR}/pipeline_${STAMP}.log"

echo "========================================================================" | tee -a "$PIPELINE_LOG"
echo "Starting Sequential Baseline Pipeline for: ${MODELS}" | tee -a "$PIPELINE_LOG"
echo "Date:   $(date)" | tee -a "$PIPELINE_LOG"
echo "Host:   $(hostname)" | tee -a "$PIPELINE_LOG"
echo "Device: ${DEVICE}" | tee -a "$PIPELINE_LOG"
echo "Log:    ${PIPELINE_LOG}" | tee -a "$PIPELINE_LOG"
echo "========================================================================" | tee -a "$PIPELINE_LOG"

# Step 1: Synthetic Benchmark (S0–S3 + S5 with CF evaluation)
echo "" | tee -a "$PIPELINE_LOG"
echo ">>> [1/3] Running Stage: SYNTHETIC BENCHMARK (S0–S3, S5)..." | tee -a "$PIPELINE_LOG"
"$PYTHON" -m baselines.run_suite \
    --models "$MODELS" \
    --stage synthetic \
    --config configs/baselines/synthetic.yaml \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume 2>&1 | tee -a "$PIPELINE_LOG"

# Step 2: MIMIC Stage-1 Pretraining
echo "" | tee -a "$PIPELINE_LOG"
echo ">>> [2/3] Running Stage: MIMIC-IV PRETRAINING (Stage-1)..." | tee -a "$PIPELINE_LOG"
"$PYTHON" -m baselines.run_suite \
    --models "$MODELS" \
    --stage mimic_pretrain \
    --config configs/baselines/mimic.yaml \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume 2>&1 | tee -a "$PIPELINE_LOG"

# Step 3: NCH Stage-2 Pediatric Finetuning
echo "" | tee -a "$PIPELINE_LOG"
echo ">>> [3/3] Running Stage: NCH PEDIATRIC FINETUNING (Stage-2)..." | tee -a "$PIPELINE_LOG"
"$PYTHON" -m baselines.run_suite \
    --models "$MODELS" \
    --stage nch_finetune \
    --config configs/baselines/nch.yaml \
    --device "$DEVICE" \
    --seed "$SEED" \
    --use-mimic-checkpoints \
    --resume 2>&1 | tee -a "$PIPELINE_LOG"

echo "" | tee -a "$PIPELINE_LOG"
echo "========================================================================" | tee -a "$PIPELINE_LOG"
echo "Pipeline finished successfully at $(date)" | tee -a "$PIPELINE_LOG"
echo "========================================================================" | tee -a "$PIPELINE_LOG"
