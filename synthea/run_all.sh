#!/usr/bin/env bash
# Fine-tune all four model_new pretrained arms on Synthea disease-onset tasks.
#
# Outputs: model_new/run/synthea/{arm}_{task}_{timestamp}/
#
# Prerequisite: synthea/run_pipeline.sh has built cohorts + tensors under
#   data/synthea/finetune/tensorized/{obesity,t2d,osa,asthma}/
#
# Usage:
#   ./synthea/run_all.sh
#   EPOCHS=5 TASKS="obesity t2d" ./synthea/run_all.sh
#   ARMS="vanilla kernel" ./synthea/run_all.sh
set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP="${TIMESTAMP:-$(date +%m%d%Y%H%M)}"
TASKS="${TASKS:-obesity t2d osa asthma}"
ARMS="${ARMS:-vanilla kernel random_constant additive}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-64}"
BAND_TABLE="${BAND_TABLE:-pediatric}"

TASK_ROOT="data/synthea/finetune/tensorized"
EMBEDDINGS="data/synthea/processed/bge_embeddings.pt"
RUN_ROOT="model_new/run/synthea"

# Matched-mode: each arm fine-tunes from its own pretrained backbone.
declare -A CKPT=(
  [vanilla]="model_new/run/vanilla_s0/epoch_008.pt"
  [kernel]="model_new/run/kernel_s0_072420260946/epoch_008.pt"
  [random_constant]="model_new/run/random_constant_s0_072420261750/epoch_008.pt"
  [additive]="model_new/run/additive_s0_072520260143/epoch_008.pt"
)

FAILED=()

echo "=============================================================="
echo "[synthea finetune] timestamp=${TIMESTAMP}"
echo "[synthea finetune] arms=${ARMS}"
echo "[synthea finetune] tasks=${TASKS}"
echo "[synthea finetune] run_root=${RUN_ROOT}/{arm}_{task}_${TIMESTAMP}/"
echo "=============================================================="

for arm in $ARMS; do
  ckpt="${CKPT[$arm]:-}"
  if [ -z "$ckpt" ]; then
    echo "=== ${arm}: unknown arm (no CKPT entry), skipping"
    FAILED+=("${arm} (unknown)")
    continue
  fi
  if [ ! -e "$ckpt" ]; then
    echo "=== ${arm}: missing checkpoint ${ckpt}, skipping"
    FAILED+=("${arm} (no ckpt)")
    continue
  fi

  for task in $TASKS; do
    task_dir="${TASK_ROOT}/${task}"
    name="${arm}_${task}_${TIMESTAMP}"
    dir="${RUN_ROOT}/${name}"

    if [ ! -d "$task_dir" ]; then
      echo "=== ${name}: missing task dir ${task_dir}, skipping"
      FAILED+=("${name} (no task dir)")
      continue
    fi

    mkdir -p "$dir"
    echo "=== ${name} <- ${ckpt}"

    set +e
    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
      conda run --no-capture-output -n ehr python -m model_new.train_finetune \
        --arm "$arm" \
        --seed "$SEED" \
        --run_name "$name" \
        --run_root "$RUN_ROOT" \
        --pretrained_ckpt "$ckpt" \
        --tensorized_dir "$task_dir" \
        --embedding_path "$EMBEDDINGS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH" \
        --num_workers 8 \
        --lr_backbone 1e-5 --lr_age 1e-3 --lr_head 1e-3 \
        --device cuda \
        --band_table "$BAND_TABLE" \
        --task_name "$task" \
        --primary_task "$task" \
        --primary_endpoint val_auprc \
        --vocab_choice synthea_bge_table \
        --deviation "Synthea pediatric synthetic finetune; backbone=${ckpt}" \
        --deviation "DECISION D3: vocab_choice=synthea_bge_table, embedding_path=${EMBEDDINGS}" \
        > "${dir}/train.log" 2>&1
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
      echo "    FAILED (rc=${rc}) — see ${dir}/train.log"
      FAILED+=("${name}")
      continue
    fi
    echo "    done: ${dir}/{config.json,pic_config.json,train.json,best.pt,epoch_*.pt}"
  done
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "FAILED RUNS: ${FAILED[*]}"
  exit 1
fi

echo
echo "All runs finished under ${RUN_ROOT}/ (*_${TIMESTAMP})"
