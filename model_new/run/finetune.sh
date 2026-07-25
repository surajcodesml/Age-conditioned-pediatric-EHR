#!/usr/bin/env bash
# Four-arm fine-tune from ONE shared pretrained backbone.
#
#   CKPT=model_new/run/vanilla_s0/epoch_008.pt ./model_new/run/finetune.sh
#   TASK_DIR=data/finetune/heart_malformations_tensorized CKPT=... ./model_new/run/finetune.sh
#
# tau_max is read from the checkpoint and reused bit-for-bit (INV-TMAX). It is never
# re-derived from the fine-tune corpus.
set -euo pipefail
cd "$(dirname "$0")/../.."

# MODE selects which pretraining design is being fine-tuned:
#
#   matched — each arm fine-tunes from ITS OWN pretrained backbone.
#             Matches the Method section: "The age pathway is trained here
#             [pretraining], rather than being bolted on at fine-tuning."
#
#   shared  — all arms fine-tune from ONE vanilla backbone.
#             Matches the Table 1 caption: "all fine-tuned from a single shared
#             backbone." Tests age introduced only at fine-tuning.
#
# Run both. The pair separates "age helps when pretrained" from "age helps when
# bolted on", which is the claim the Method section actually makes.
MODE="${MODE:-matched}"
CKPT_ROOT="${CKPT_ROOT:-model_new/run}"
TASK_DIR="${TASK_DIR:-data/finetune/heart_malformations_tensorized}"
ARMS="${ARMS:-vanilla kernel random_constant additive}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
RUN_ROOT="${RUN_ROOT:-model_new/run_finetune_${MODE}}"

COMMON=(
  --tensorized_dir  "$TASK_DIR"
  --embedding_path  data/processed/bge_embeddings.pt
  --epochs "$EPOCHS" --batch_size 16 --num_workers 4
  --lr_backbone 1e-5 --lr_age 1e-3 --lr_head 1e-3
  --device cuda --run_root "$RUN_ROOT"
)
FAILED=()

for seed in $SEEDS; do
  for arm in $ARMS; do
    name="${arm}_s${seed}"
    dir="${RUN_ROOT}/${name}"

    case "$MODE" in
      matched) ckpt="${CKPT_ROOT}/${arm}_s${seed}/best.pt" ;;
      shared)  ckpt="${CKPT_ROOT}/vanilla_s${seed}/best.pt" ;;
      *) echo "MODE must be 'matched' or 'shared', got '${MODE}'"; exit 1 ;;
    esac

    if [ ! -e "$ckpt" ]; then
      echo "=== ${name}: missing checkpoint ${ckpt}, skipping"
      FAILED+=("${name} (no ckpt)")
      continue
    fi

    mkdir -p "$dir"
    echo "=== ${name} [${MODE}] <- ${ckpt}"

    set +e
    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
      conda run --no-capture-output -n ehr python -m model_new.train_finetune \
        --arm "$arm" --seed "$seed" --run_name "$name" \
        --pretrained_ckpt "$ckpt" "${COMMON[@]}" \
        > "${dir}/train.log" 2>&1
    rc=$?
    set -e

    [ $rc -ne 0 ] && { echo "    FAILED (rc=${rc})"; FAILED+=("$name"); continue; }
    echo "    done"
  done
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo; echo "FAILED RUNS: ${FAILED[*]}"; exit 1
fi