#!/usr/bin/env bash
# Fine-tune ablation MATRIX with the dedicated age optimizer group + longer training.
#
#   arms   : vanilla random_constant additive kernel
#   tasks  : pneumonia heart_malformations mortality los_gt7   (PIC cohorts)
#   lr_age : swept over {1e-3, 3e-3}   (Tier 2 only)
#   seed   : 42   (SHARED)             backbone: ONE shared pretrained checkpoint
#
# Only --arm / --task / --lr_age vary; everything else is identical across cells.
# Age-injection params (kernel age_coeff_gen, additive additive_age_emb) train in a
# dedicated AdamW group at --lr_age; the rest of the backbone stays at --lr_backbone.
#
# Smoke tiers (gate before the full sweep) — set TIER=0|1|2 (default 2):
#   TIER=0  --dry_run_one_epoch, all arms on the smoke task (seconds; plumbing check)
#   TIER=1  --max_rows 5000 --epochs 2, all arms on the smoke task
#           (confirm age-group grad norms are now O(1) in kernel + additive)
#   TIER=2  full data, full arm x task x lr_age matrix (the real run)
#
# Usage:
#   TIER=0 bash model_ablation/run_finetune_matrix.sh
#   TIER=1 bash model_ablation/run_finetune_matrix.sh
#   bash model_ablation/run_finetune_matrix.sh            # TIER=2 full sweep
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

TIER="${TIER:-2}"

# ---- shared backbone (identical hash across every cell) --------------------------
PRETRAINED_CKPT="checkpoints/ablation_pretrain/run20260703_194128/epoch_010.pt"
VOCAB="data/processed/code_vocab.json"
test -f "$PRETRAINED_CKPT" || { echo "ABORT: backbone not found: $PRETRAINED_CKPT"; exit 1; }

# ---- shared hyperparameters -----------------------------------------------------
SEED=42
BATCH_SIZE=64
LR_BACKBONE=1e-5
LR_HEAD=1e-3
NUM_WORKERS=6
GPU=0

ARMS="vanilla random_constant additive kernel"
TASKS="pneumonia heart_malformations mortality los_gt7"
SMOKE_TASK="heart_malformations"

# per-task data locations (uniform PIC finetune layout)
cohort_dir() { echo "data/processed/pic/finetune/$1/cohort"; }
events_pq()  { echo "data/processed/pic/finetune/$1/events.parquet"; }
tensor_dir() { echo "data/finetune/$1"; }

# ---- tier settings --------------------------------------------------------------
case "$TIER" in
  0) EXTRA="--dry_run_one_epoch";        LR_AGES="1e-3"; RUN_TASKS="$SMOKE_TASK";;
  1) EXTRA="--max_rows 5000 --epochs 2"; LR_AGES="1e-3"; RUN_TASKS="$SMOKE_TASK";;
  2) EXTRA="--epochs 30 --patience 6";   LR_AGES="1e-3 3e-3"; RUN_TASKS="$TASKS";;
  *) echo "ABORT: TIER must be 0, 1, or 2 (got '$TIER')"; exit 1;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="checkpoints/finetune/ablation/matrix_tier${TIER}_${STAMP}"
mkdir -p "$RUN_ROOT"
echo "[matrix] TIER=$TIER  root=$RUN_ROOT"
echo "[backbone] $PRETRAINED_CKPT" | tee "$RUN_ROOT/BACKBONE.txt"

# ---- 1) build tensorized shards for the tasks we will run (idempotent) -----------
for T in $RUN_TASKS; do
  TD="$(tensor_dir "$T")"
  if ls "$TD"/train/shard_*.npz >/dev/null 2>&1; then
    echo "[tensorize] $T: shards present at $TD (skip)"
  else
    echo "[tensorize] $T: building $TD ..."
    conda run --no-capture-output -n ehr python finetune/build_disease_tensors.py \
      --cohort_dir "$(cohort_dir "$T")" \
      --events_parquet "$(events_pq "$T")" \
      --vocab_path "$VOCAB" \
      --out_dir "$TD"
  fi
done

# ---- 2) run the matrix ----------------------------------------------------------
for T in $RUN_TASKS; do
  TD="$(tensor_dir "$T")"
  for LRA in $LR_AGES; do
    for ARM in $ARMS; do
      ARM_DIR="$RUN_ROOT/$T/lr_age_${LRA}/$ARM"
      mkdir -p "$ARM_DIR"
      echo "=================================================================="
      echo "[task=$T lr_age=$LRA arm=$ARM] start $(date '+%H:%M:%S') -> $ARM_DIR"
      echo "=================================================================="
      HIP_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 \
      conda run --no-capture-output -n ehr python model_ablation/train_finetune.py \
          --arm "$ARM" \
          --task_name "$T" \
          --pretrained_ckpt "$PRETRAINED_CKPT" \
          --tensorized_dir "$TD" \
          --vocab_path "$VOCAB" \
          --batch_size "$BATCH_SIZE" \
          --lr_backbone "$LR_BACKBONE" \
          --lr_head "$LR_HEAD" \
          --lr_age "$LRA" \
          --num_workers "$NUM_WORKERS" \
          --seed "$SEED" \
          --run_dir "$ARM_DIR" \
          $EXTRA \
          2>&1 | tee "$ARM_DIR/train.log"
      echo "[task=$T lr_age=$LRA arm=$ARM] done $(date '+%H:%M:%S')"
    done
  done
done

echo "=================================================================="
echo "MATRIX (TIER=$TIER) DONE -> $RUN_ROOT"
if [ "$TIER" != "2" ]; then
  echo "[tier $TIER] age-pathway drift is the fix signal (grad norm stays small for kernel by"
  echo "             construction; Adam step ~= lr, so drift should be non-trivial for kernel+additive):"
  for T in $RUN_TASKS; do for ARM in $ARMS; do
    H="$RUN_ROOT/$T/lr_age_1e-3/$ARM/history.json"
    [ -f "$H" ] && printf "  %-10s %-16s " "$T" "$ARM" && \
      conda run -n ehr python -c "import json; h=json.load(open('$H'))['history']; e=h[-1]; g=e['age_pathway_grad_norm_preclip']; print('age_drift=%.3e base_coeff_drift=%.3e | age_gradnorm(mean)=%.3e' % (e.get('age_pathway_drift_l2',0.0), e.get('base_coeff_drift_l2',0.0), g['mean']))"
  done; done
fi
