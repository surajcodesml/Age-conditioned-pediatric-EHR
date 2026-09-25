#!/usr/bin/env bash
# =============================================================================
# Full baseline + DTR pipeline
#
# Order:
#   0) MIMIC smoke / sanity checks (fail fast)
#   1) Synthetic: non-DTR baselines on S5 only  (S0–S3 already done)
#   2) Synthetic: DTR + ablation arms on S0–S3 + S5
#   3) Synthetic counterfactual eval (S5 for all; S0–S3 for DTR arms)
#   4) MIMIC Stage-1 pretraining — DTR arms first, then other baselines
#
# Artifacts per run dir (neural models):
#   best_checkpoint.pt  last_checkpoint.pt  checkpoint.pt(=best)
#   history.json  config.json  result.json
#
# Early stopping: patience is enabled, but each full run trains at least
#   synthetic: max(PATIENCE=5, MAX_EPOCHS//2=12) → 12 epochs minimum
#   MIMIC:     MIN_EPOCHS=5 (MAX_EPOCHS=10, PATIENCE=3)
#
# Usage:
#   bash baselines/run_full_pipeline.sh
#   DEVICE=cuda SKIP_SMOKE=1 bash baselines/run_full_pipeline.sh
#   bash baselines/run_full_pipeline.sh 2>&1 | tee logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-ehr}"
DATA_SEED="${DATA_SEED:-20260922}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_SYNTHETIC="${SKIP_SYNTHETIC:-0}"
SKIP_MIMIC="${SKIP_MIMIC:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/full_pipeline_${STAMP}.log"

# Non-DTR synthetic baselines (S5 only)
BASELINE_MODELS="count_lightgbm,retain,ehr_bert,behrt,medbert,cehrbert"
# DTR synthetic ablation arms (runner expands --models dtr → no_age / age_only /
# temporal_only / age_temporal under results/baselines/synthetic/dtr_*/)
DTR_MODEL="dtr"
SYNTH_SCENARIOS_CORE=(S0 S1 S2 S3)
SYNTH_SCENARIO_S5=S5

PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }
section() {
  log "============================================================"
  log "$*"
  log "============================================================"
}
run() {
  log "+ $*"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    log "FAILED (exit $rc): $*"
    exit "$rc"
  fi
}

require_file() {
  if [[ ! -e "$1" ]]; then
    log "MISSING required path: $1"
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    log "MISSING required directory: $1"
    exit 1
  fi
}

# -----------------------------------------------------------------------------
section "0a. Preflight: data + env"
# -----------------------------------------------------------------------------
require_dir "synthetic_age_temporal/outputs/data/seed${DATA_SEED}/controlled/S5"
require_dir "data/processed/tensorized_flat/train"
require_dir "data/processed/tensorized_flat/val"
require_dir "data/processed/tensorized_flat/test"
require_file "data/processed/code_vocab.json"
require_file "data/processed/bge_embeddings.pt"

run "${PY[@]}" - <<'PY'
import torch, sys
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: CUDA/ROCm not available — training will use CPU", file=sys.stderr)
else:
    print("device0", torch.cuda.get_device_name(0))
PY

# -----------------------------------------------------------------------------
if [[ "$SKIP_SMOKE" != "1" ]]; then
section "0b. MIMIC smoke / sanity (fail fast before long runs)"
# -----------------------------------------------------------------------------
  # Unit sanity for Minimal-DKM + tiny 1-shard MIMIC overfit for both arms
  run "${PY[@]}" -m stage1_mimic_pretrain.tests.run_all

  # baselines.mimic smoke: DTR arms first (age_temporal, no_interaction)
  run "${PY[@]}" -m baselines.mimic.runner \
      --models dtr \
      --device "$DEVICE" \
      --smoke

  # One non-DTR baseline smoke
  run "${PY[@]}" -m baselines.mimic.runner \
      --models retain \
      --device "$DEVICE" \
      --smoke

  # Synthetic S5 smoke into an isolated output dir (do not clobber real results)
  SMOKE_SYNTH_OUT="${REPO_ROOT}/results/baselines/synthetic/_smoke"
  run "${PY[@]}" -m baselines.synthetic.runner \
      --scenario S5 --models retain --device "$DEVICE" --smoke --data-seed "$DATA_SEED" \
      --output-dir "$SMOKE_SYNTH_OUT"
  run "${PY[@]}" -m baselines.synthetic.runner \
      --scenario S5 --models dtr --device "$DEVICE" --smoke --data-seed "$DATA_SEED" \
      --output-dir "$SMOKE_SYNTH_OUT"

  log "Smoke checks passed."
else
  log "SKIP_SMOKE=1 — skipping smoke checks"
fi

# -----------------------------------------------------------------------------
if [[ "$SKIP_SYNTHETIC" != "1" ]]; then
section "1. Synthetic baselines on S5 only (S0–S3 already trained)"
# -----------------------------------------------------------------------------
  run "${PY[@]}" -m baselines.synthetic.runner \
      --scenario "$SYNTH_SCENARIO_S5" \
      --models "$BASELINE_MODELS" \
      --device "$DEVICE" \
      --data-seed "$DATA_SEED"

section "2. Synthetic DTR + ablation arms on S0–S3 + S5"
  for scen in "${SYNTH_SCENARIOS_CORE[@]}" "$SYNTH_SCENARIO_S5"; do
    log "--- DTR arms | scenario $scen ---"
    run "${PY[@]}" -m baselines.synthetic.runner \
        --scenario "$scen" \
        --models "$DTR_MODEL" \
        --device "$DEVICE" \
        --data-seed "$DATA_SEED"
  done

section "3. Counterfactual evaluation"
  # S5 for every model that has a result.json
  run "${PY[@]}" -m baselines.synthetic.counterfactual_eval \
      --scenario S5 --device "$DEVICE" --data-seed "$DATA_SEED"
  # Refresh S0–S3 CF for DTR arms (and any new checkpoints)
  for scen in "${SYNTH_SCENARIOS_CORE[@]}"; do
    run "${PY[@]}" -m baselines.synthetic.counterfactual_eval \
        --scenario "$scen" --device "$DEVICE" --data-seed "$DATA_SEED"
  done

  log "Synthetic phase complete → results/baselines/synthetic/"
else
  log "SKIP_SYNTHETIC=1 — skipping synthetic training/eval"
fi

# -----------------------------------------------------------------------------
if [[ "$SKIP_MIMIC" != "1" ]]; then
section "4. MIMIC Stage-1 pretraining (DTR arms first, then baselines)"
# -----------------------------------------------------------------------------
  # Order inside runner --models all:
  #   dtr_age_temporal, dtr_no_interaction, retain, ehr_bert, behrt, medbert, cehrbert
  # Each writes under results/baselines/mimic/<name>/:
  #   best_checkpoint.pt  last_checkpoint.pt  checkpoint.pt(=best)
  #   config.json  history.json  result.json
  # Early stopping cannot fire before MIN_EPOCHS=5.
  run "${PY[@]}" -m baselines.mimic.runner \
      --models all \
      --device "$DEVICE"

  log "MIMIC phase complete → results/baselines/mimic/"
  log "Combined summary → results/baselines/mimic/all_results.json"
else
  log "SKIP_MIMIC=1 — skipping MIMIC pretraining"
fi

section "DONE"
log "Master log: $MASTER_LOG"
log "Synthetic:  results/baselines/synthetic/"
log "MIMIC:      results/baselines/mimic/"
