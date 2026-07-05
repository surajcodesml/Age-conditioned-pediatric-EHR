#!/usr/bin/env bash
# Four-arm CHD (heart_malformations) fine-tune ablation, end to end:
#   1) build the flat/mmap tensorized shards once (idempotent)
#   2) fine-tune all four arms sequentially from the SAME pretrained backbone
#
# Only --arm varies; seed and every hyperparameter are identical across arms.
# Arm semantics (resolved inside the model):
#   vanilla          - no age conditioning (control)
#   random_constant  - kernel architecture, age = fixed 7y (capacity-matched control)
#   additive         - age delta added to code embeddings (real age)
#   kernel           - age -> Delta polynomial coeffs on the temporal kernel (real age)
#
# Run after pretraining has finished (fine-tune uses the GPU).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# ---- shared backbone: best-val_bce epoch of the completed 10-epoch pretrain ----
PRETRAINED_CKPT="checkpoints/ablation_pretrain/run20260703_194128/epoch_010.pt"

# ---- data ----
COHORT_DIR="data/processed/pic/finetune/heart_malformations/cohort"
EVENTS_PARQUET="data/processed/pic/finetune/heart_malformations/events.parquet"
VOCAB="data/processed/code_vocab.json"
TENSORIZED_DIR="data/finetune/heart_malformations"

# ---- hyperparameters (shared across all arms) ----
SEED=42
EPOCHS=5
BATCH_SIZE=64
LR_BACKBONE=1e-5
LR_HEAD=1e-3
NUM_WORKERS=6
GPU=0

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="checkpoints/finetune/ablation/chd_run${STAMP}"
mkdir -p "$RUN_ROOT"

test -f "$PRETRAINED_CKPT" || { echo "ABORT: backbone not found: $PRETRAINED_CKPT"; exit 1; }
echo "[backbone] $PRETRAINED_CKPT" | tee "$RUN_ROOT/BACKBONE.txt"

# ---- 1) build tensorized shards (idempotent) ----
if ls "$TENSORIZED_DIR"/train/shard_*.npz >/dev/null 2>&1; then
  echo "[tensorize] shards already present at $TENSORIZED_DIR (skipping build)"
else
  echo "[tensorize] building $TENSORIZED_DIR from cohort + events ..."
  conda run --no-capture-output -n ehr python finetune/build_disease_tensors.py \
    --cohort_dir "$COHORT_DIR" \
    --events_parquet "$EVENTS_PARQUET" \
    --vocab_path "$VOCAB" \
    --out_dir "$TENSORIZED_DIR"
fi

# ---- 2) run all four arms in sequence (only --arm varies) ----
for ARM in vanilla random_constant additive kernel; do
  ARM_DIR="$RUN_ROOT/$ARM"
  mkdir -p "$ARM_DIR"
  echo "=================================================================="
  echo "[arm=$ARM] start $(date '+%H:%M:%S')  ->  $ARM_DIR"
  echo "=================================================================="
  HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 \
  conda run --no-capture-output -n ehr python model_ablation/train_finetune.py \
      --arm "$ARM" \
      --pretrained_ckpt "$PRETRAINED_CKPT" \
      --tensorized_dir "$TENSORIZED_DIR" \
      --vocab_path "$VOCAB" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr_backbone "$LR_BACKBONE" \
      --lr_head "$LR_HEAD" \
      --num_workers "$NUM_WORKERS" \
      --seed "$SEED" \
      --run_dir "$ARM_DIR" \
      2>&1 | tee "$ARM_DIR/train.log"
  echo "[arm=$ARM] done $(date '+%H:%M:%S')"
done

echo "=================================================================="
echo "ALL ARMS DONE -> $RUN_ROOT"
for ARM in vanilla random_constant additive kernel; do
  echo "  ${ARM}: $RUN_ROOT/$ARM/history.json"
done
