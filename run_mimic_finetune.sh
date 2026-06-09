#!/usr/bin/env bash
# =============================================================================
# MIMIC-IV disease fine-tuning: vanilla TALE-EHR vs age-conditioned backbones.
#
# Order of execution (one after another, sequentially):
#   1. t2d              age      -> 5 epochs   (vanilla already computed earlier)
#   2. heart_failure    age, then vanilla -> 3 epochs each
#   3. depression       age, then vanilla -> 3 epochs each
#   4. arteriosclerosis age, then vanilla -> 3 epochs each
#   5. acute_kidney     age, then vanilla -> 3 epochs each
#
# For tasks 2-5 the cohort + tensorized shards are built once (shared by the
# age and vanilla runs). t2d reuses the existing cohort/tensors from the
# completed vanilla run so the age-vs-vanilla comparison is on identical data.
#
# Pretrained backbones (variant auto-detected by finetune/model.py):
#   age     : checkpoints/age_real_202605112156/epoch_012.pt
#   vanilla : checkpoints/run_20260427_152603/best_pretrain.pt
#
# Each run writes to checkpoints/finetune/<task>_<weights>/ :
#   epoch_NNN.pt / best.pt   - checkpoints (best = highest val AUROC)
#   console.log              - full training log
#   history.json             - per-epoch + test eval metrics (AUROC/AUPRC/...)
#   test_predictions.parquet - per-example test predictions
#   decay_alpha.json         - age-kernel diagnostics (age runs only)
#
# Disease label = PheCode prefix present anywhere in the timeline; cohort uses
# the v1 quantile-matched truncation design (finetune/build_disease_cohort.py).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONDA_RUN="conda run --no-capture-output -n ehr"

# Pin a single GPU for determinism (host has 2x AMD R9700, ROCm).
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false

# --- pretrained backbones ----------------------------------------------------
AGE_CKPT="checkpoints/age_real_202605112156/epoch_012.pt"
VANILLA_CKPT="checkpoints/run_20260427_152603/best_pretrain.pt"

# --- shared data paths -------------------------------------------------------
VOCAB="data/processed/code_vocab.json"
EVENTS="data/processed/patient_events_rolled_full.parquet"
FT_DATA_ROOT="data/finetune"
CKPT_ROOT="checkpoints/finetune"

# --- loader / optimization settings ------------------------------------------
# disease_collate materializes a [B, L, L] delta_t matrix per batch (L up to
# max_seq_len=1024). batch_size 32 with 4 workers / prefetch 2 keeps peak
# host+GPU memory well under the t2d-proven batch 64 for an unattended run.
BATCH_SIZE=32
NUM_WORKERS=4
PREFETCH=2
LR_BACKBONE=1e-5
LR_HEAD=1e-3

# t2d reuses the cohort/tensors from the existing vanilla run; match its batch
# size (64) so the age run is directly comparable to the completed vanilla one.
T2D_BATCH_SIZE=64
T2D_EPOCHS=5
T2D_COHORT_DIR="$FT_DATA_ROOT/t2d"
T2D_TENSOR_DIR="$FT_DATA_ROOT/t2d_tensorized"

DISEASE_EPOCHS=3

mkdir -p "$CKPT_ROOT"
MASTER_LOG="$CKPT_ROOT/run_mimic_finetune_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# task short-name -> PheCode prefix (label = code present anywhere in timeline)
declare -A DISEASE_PREFIX=(
  [heart_failure]="PHE_428"
  [depression]="PHE_296.2"
  [arteriosclerosis]="PHE_440"
  [acute_kidney]="PHE_585.3"
)
# Build/train order requested by the user.
DISEASES=(heart_failure depression arteriosclerosis acute_kidney)

# -----------------------------------------------------------------------------
build_cohort_and_tensors() {
  # Idempotent: builds the cohort parquet + tensorized shards for $task once.
  local task="$1" prefix="$2"
  local cohort_dir="$FT_DATA_ROOT/$task"
  local tensor_dir="${cohort_dir}_tensorized"

  if [[ ! -f "$cohort_dir/train_cohort.parquet" ]]; then
    log "[$task] building cohort (prefix=$prefix) -> $cohort_dir"
    $CONDA_RUN python -m finetune.build_disease_cohort \
      --disease "$task" \
      --code_prefix "$prefix" \
      --out_dir "$cohort_dir" 2>&1 | tee -a "$MASTER_LOG"
  else
    log "[$task] cohort already exists -> skip"
  fi

  if [[ ! -f "$tensor_dir/train/shard_00000.npz" ]]; then
    log "[$task] tensorizing -> $tensor_dir"
    $CONDA_RUN python -m finetune.build_disease_tensors \
      --cohort_dir "$cohort_dir" \
      --events_parquet "$EVENTS" \
      --vocab_path "$VOCAB" \
      --out_dir "$tensor_dir" 2>&1 | tee -a "$MASTER_LOG"
  else
    log "[$task] tensors already exist -> skip"
  fi
}

run_finetune() {
  # run_finetune <task> <weights:age|vanilla> <ckpt> <cohort_dir> <tensor_dir> <epochs> <batch>
  local task="$1" weights="$2" ckpt="$3" cohort_dir="$4" tensor_dir="$5" epochs="$6" batch="$7"
  local run_dir="$CKPT_ROOT/${task}_${weights}"

  if [[ -f "$run_dir/history.json" ]]; then
    log "[$task/$weights] already complete ($run_dir/history.json) -> skip"
    return 0
  fi

  log "FINETUNE task=$task weights=$weights epochs=$epochs batch=$batch -> $run_dir"
  $CONDA_RUN python -m finetune.train \
    --disease "$task" \
    --pretrained_ckpt "$ckpt" \
    --cohort_dir "$cohort_dir" \
    --tensorized_dir "$tensor_dir" \
    --events_parquet "$EVENTS" \
    --vocab_path "$VOCAB" \
    --run_dir "$run_dir" \
    --epochs "$epochs" \
    --batch_size "$batch" \
    --lr_backbone "$LR_BACKBONE" \
    --lr_head "$LR_HEAD" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH" 2>&1 | tee -a "$MASTER_LOG"
  log "DONE task=$task weights=$weights -> $run_dir"
}

# =============================================================================
log "=== run_mimic_finetune started (master_log=$MASTER_LOG) ==="
log "age_ckpt=$AGE_CKPT  vanilla_ckpt=$VANILLA_CKPT  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"

for p in "$AGE_CKPT" "$VANILLA_CKPT" "$VOCAB" "$EVENTS" "$T2D_TENSOR_DIR/train/shard_00000.npz"; do
  [[ -e "$p" ]] || { log "FATAL: missing required path: $p"; exit 1; }
done

# --- 1. t2d (age weights, 5 epochs; vanilla already computed) -----------------
log "--- [1] t2d  weights=age ---"
run_finetune "t2d" "age" "$AGE_CKPT" "$T2D_COHORT_DIR" "$T2D_TENSOR_DIR" "$T2D_EPOCHS" "$T2D_BATCH_SIZE"

# --- 2..5. heart_failure, depression, arteriosclerosis, acute_kidney ----------
#           build data once, then age then vanilla (3 epochs each).
for task in "${DISEASES[@]}"; do
  prefix="${DISEASE_PREFIX[$task]}"
  cohort_dir="$FT_DATA_ROOT/$task"
  tensor_dir="${cohort_dir}_tensorized"

  log "--- $task (prefix=$prefix) ---"
  build_cohort_and_tensors "$task" "$prefix"

  run_finetune "$task" "age"     "$AGE_CKPT"     "$cohort_dir" "$tensor_dir" "$DISEASE_EPOCHS" "$BATCH_SIZE"
  run_finetune "$task" "vanilla" "$VANILLA_CKPT" "$cohort_dir" "$tensor_dir" "$DISEASE_EPOCHS" "$BATCH_SIZE"
done

log "=== run_mimic_finetune complete -> $CKPT_ROOT ==="
