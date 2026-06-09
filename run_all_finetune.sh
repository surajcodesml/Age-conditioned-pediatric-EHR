#!/usr/bin/env bash
# =============================================================================
# Run all fine-tuning jobs sequentially in the `ehr` conda environment.
#
#   1. MIMIC-IV  los_gt7            (vanilla TALE-EHR backbone)
#   2. PIC setup + tensorization    (per-task cohort dirs + event shards)
#   3. PIC tasks x {age, vanilla}   (8 runs -> checkpoints/finetune/PIC/<task>_<weights>)
#
# Pretrained weights
#   age     : checkpoints/age_real_202605112156/epoch_012.pt
#   vanilla : checkpoints/run_20260427_152603/best_pretrain.pt
#
# Hardware: 2x AMD R9700 (32GB, ROCm), 16 CPU, 30GB RAM.
# Batch/worker sizes are chosen to keep one GPU busy without OOM. PIC sequences
# are short (1-day observation window, mostly single-admission) so batch_size
# 128 is safe; the large MIMIC los_gt7 cohort uses the user-specified 64.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONDA_RUN="conda run --no-capture-output -n ehr"

# Pin a single GPU for determinism (avoids spreading across both R9700s).
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false

AGE_CKPT="checkpoints/age_real_202605112156/epoch_012.pt"
VANILLA_CKPT="checkpoints/run_20260427_152603/best_pretrain.pt"

PIC_VOCAB="data/processed/pic/code_vocab_pic.json"
PIC_EMB="data/processed/pic/bge_embeddings_pic.pt"
PIC_FT_ROOT="data/processed/pic/finetune"
PIC_TENSOR_ROOT="data/tensorized/pic"
PIC_CKPT_ROOT="checkpoints/finetune/PIC"

PIC_TASKS=(mortality los_gt7 pneumonia heart_malformations)

# PIC loader/optimization settings (short sequences -> large batch is safe).
PIC_BATCH_SIZE=64
PIC_NUM_WORKERS=4
PIC_PREFETCH=2
PIC_EPOCHS=10
PIC_LR_KERNEL=1e-3
PIC_LR_BACKBONE=1e-5
PIC_LR_HEAD=1e-3

MASTER_LOG="$PIC_CKPT_ROOT/run_all_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PIC_CKPT_ROOT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "=== run_all_finetune started ==="
log "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES  master_log=$MASTER_LOG"

# -----------------------------------------------------------------------------
# 1. MIMIC-IV los_gt7 (exact user-specified command)
# -----------------------------------------------------------------------------
# MIMIC los_gt7 has long sequences (up to max_seq_len=1024). disease_collate
# materializes a [B, L, L] delta_t matrix, and at startup num_workers x
# prefetch_factor of them are built at once alongside ~0.48GB shard loads, which
# OOM-kills workers at batch_size 32/64 on this 30GB host. batch_size 16 is the
# proven-safe setting from prior successful runs.

# -----------------------------------------------------------------------------
# 2. PIC setup + tensorization (once per task; shared by age & vanilla runs)
# -----------------------------------------------------------------------------
log "--- [2/3] PIC setup + tensorization ---"
$CONDA_RUN python -m finetune.PIC.setup_finetune_dirs --out_root "$PIC_FT_ROOT" 2>&1 | tee -a "$MASTER_LOG"

for task in "${PIC_TASKS[@]}"; do
  out_dir="$PIC_TENSOR_ROOT/$task"
  if [[ -f "$out_dir/train/shard_00000.npz" ]]; then
    log "tensorized shards already exist for $task -> skip"
    continue
  fi
  log "tensorizing $task ..."
  $CONDA_RUN python -m finetune.PIC.build_disease_tensors_pic \
    --cohort_dir "$PIC_FT_ROOT/$task/cohort" \
    --events_parquet "$PIC_FT_ROOT/$task/events.parquet" \
    --vocab_path "$PIC_VOCAB" \
    --out_dir "$out_dir" 2>&1 | tee -a "$MASTER_LOG"
done

# -----------------------------------------------------------------------------
# 3. PIC fine-tuning: all tasks for age weights, then all tasks for vanilla
# -----------------------------------------------------------------------------
run_pic_task() {
  local task="$1" weights="$2" ckpt="$3"
  local run_dir="$PIC_CKPT_ROOT/${task}_${weights}_lr_10eph"
  log "FINETUNE pic/$task weights=$weights -> $run_dir"
  $CONDA_RUN python -m finetune.PIC.train_pic \
    --disease "$task" \
    --pretrained_ckpt "$ckpt" \
    --cohort_dir "$PIC_FT_ROOT/$task/cohort" \
    --tensorized_dir "$PIC_TENSOR_ROOT/$task" \
    --vocab_path "$PIC_VOCAB" \
    --embedding_path "$PIC_EMB" \
    --run_dir "$run_dir" \
    --epochs "$PIC_EPOCHS" \
    --lr_kernel "$PIC_LR_KERNEL" \
    --lr_backbone "$PIC_LR_BACKBONE" \
    --lr_head "$PIC_LR_HEAD" \
    --batch_size "$PIC_BATCH_SIZE" \
    --num_workers "$PIC_NUM_WORKERS" \
    --prefetch_factor "$PIC_PREFETCH" 2>&1 | tee -a "$MASTER_LOG"
  log "DONE pic/$task weights=$weights"
}

log "--- [3/3] PIC fine-tuning (age weights) ---"
for task in "${PIC_TASKS[@]}"; do
  run_pic_task "$task" "age" "$AGE_CKPT"
done

log "--- [3/3] PIC fine-tuning (vanilla weights) ---"
for task in "${PIC_TASKS[@]}"; do
  run_pic_task "$task" "vanilla" "$VANILLA_CKPT"
done

log "=== run_all_finetune complete -> $PIC_CKPT_ROOT ==="
