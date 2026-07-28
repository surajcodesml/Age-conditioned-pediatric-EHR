#!/usr/bin/env bash
# Four-arm fine-tune from a PER-ARM checkpoint map.
#
#   ./model_new/run/finetune.sh                                   # MODE=matched, one task
#   MODE=shared ./model_new/run/finetune.sh
#   TASK=heart_malformations BAND_TABLE=pediatric ./model_new/run/finetune.sh
#   TASKS="heart_malformations los_gt7 mortality pneumonia" ./model_new/run/finetune.sh
#   # MIMIC (remaining disease tasks; t2d already in run_finetune_matched):
#   MODE=matched TASK_ROOT=data/finetune TASK_DIR_SUFFIX=_tensorized \
#     RUN_ROOT_BASE=model_new/run/MIMIC BAND_TABLE=adult \
#     EMBEDDINGS=data/processed/bge_embeddings.pt VOCAB_CHOICE=mimic_bge_table \
#     TASKS="acute_kidney arteriosclerosis depression heart_failure" \
#     ./model_new/run/finetune.sh
#   CKPT_MAP="vanilla=a/best.pt kernel=b/best.pt" ./model_new/run/finetune.sh
#
# Outputs land under RUN_ROOT_BASE/<task>/<arm>_s<seed>/ (default: model_new/run/pic/).
#
# tau_max, the age standardization constants, the Fourier buffers, the race one-hot
# ordering and s are read from the checkpoint and reused bit-for-bit (INV-TMAX,
# INV-AGESTD, INV-FT-FROZEN). The ARM is read from the checkpoint too (INV-FT-ARM):
# --arm is passed only so a mismatch is loud. Nothing is re-derived from PIC.
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# DECISION D2 -- which backbone each arm fine-tunes from. NOT decided here.
#
#   matched — each arm fine-tunes from ITS OWN pretrained backbone. Measures end-to-end
#             OOD transfer of the mechanism, which is the paper's claim, but confounds
#             pretraining differences with fine-tune-time behaviour.
#
#   shared  — every arm fine-tunes from ONE vanilla backbone. Isolates the fine-tune-time
#             mechanism on an identical backbone, but supports no claim about what
#             pretraining with the kernel bought.
#
#   CKPT_MAP — an explicit "arm=path" map, which overrides MODE entirely. Both designs
#             need this plumbing, so it exists regardless of which one is chosen.
#
# Under MODE=shared every arm loads the vanilla checkpoint, so INV-FT-ARM would refuse a
# non-vanilla --arm. That refusal is correct -- "arm" is a property of the weights -- and
# shared mode is exactly the case where the mismatch is deliberate, so it passes
# --allow_arm_mismatch. Both arms then land in pic_config.json and the mismatch is listed
# under deviations_from_pretrain. It is declared, not tolerated.
# ---------------------------------------------------------------------------
MODE="${MODE:-matched}"
CKPT_ROOT="${CKPT_ROOT:-model_new/run_selected}"
CKPT_MAP="${CKPT_MAP:-}"

TASK_ROOT="${TASK_ROOT:-data/tensorized/pic}"
# MIMIC shards live at data/finetune/<task>_tensorized; set TASK_DIR_SUFFIX=_tensorized.
TASK_DIR_SUFFIX="${TASK_DIR_SUFFIX:-}"
EMBEDDINGS="${EMBEDDINGS:-data/processed/pic/bge_embeddings_pic.pt}"
VOCAB_CHOICE="${VOCAB_CHOICE:-pic_bge_table}"
BAND_TABLE="${BAND_TABLE:-pediatric}"

# TASKS loops over every task; TASK alone is the single-task shortcut.
TASKS="${TASKS:-${TASK:-heart_malformations}}"

PRIMARY_ENDPOINT="${PRIMARY_ENDPOINT:-val_auprc}"

ARMS="${ARMS:-vanilla kernel random_constant additive}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-64}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-model_new/run/pic}"

# ckpt_for <arm> <seed> -> the checkpoint path for that arm.
ckpt_for() {
  local arm="$1" seed="$2" entry
  if [ -n "$CKPT_MAP" ]; then
    for entry in $CKPT_MAP; do
      case "$entry" in "${arm}="*) echo "${entry#*=}"; return 0 ;; esac
    done
    echo ""      # an arm absent from an explicit map is skipped, not silently defaulted
    return 0
  fi
  case "$MODE" in
    matched) echo "${CKPT_ROOT}/${arm}_s${seed}/best.pt" ;;
    shared)  echo "${CKPT_ROOT}/vanilla_s${seed}/best.pt" ;;
    *) echo "MODE must be 'matched' or 'shared', got '${MODE}'" >&2; return 1 ;;
  esac
}

FAILED=()

for TASK in $TASKS; do
  TASK_DIR="${TASK_ROOT}/${TASK}${TASK_DIR_SUFFIX}"
  # PRIMARY_TASK defaults to the current task (declared before the run). Override with
  # PRIMARY_TASK=... only when deliberately selecting on a different task name.
  PRIMARY="${PRIMARY_TASK:-$TASK}"
  RUN_ROOT="${RUN_ROOT_BASE}/${TASK}"

  COMMON=(
    --tensorized_dir  "$TASK_DIR"
    --embedding_path  "$EMBEDDINGS"
    --epochs "$EPOCHS" --batch_size "$BATCH" --num_workers 8
    --lr_backbone 1e-5 --lr_age 1e-3 --lr_head 1e-3
    --device cuda --run_root "$RUN_ROOT"
    --band_table "$BAND_TABLE"
    --task_name "$TASK"
    --primary_task "$PRIMARY" --primary_endpoint "$PRIMARY_ENDPOINT"
    --vocab_choice "$VOCAB_CHOICE"
  )

  for seed in $SEEDS; do
    for arm in $ARMS; do
      name="${arm}_s${seed}"
      dir="${RUN_ROOT}/${name}"
      ckpt="$(ckpt_for "$arm" "$seed")"

      if [ -z "$ckpt" ]; then
        echo "=== ${TASK}/${name}: no entry in CKPT_MAP, skipping"
        FAILED+=("${TASK}/${name} (not in CKPT_MAP)")
        continue
      fi
      if [ ! -e "$ckpt" ]; then
        echo "=== ${TASK}/${name}: missing checkpoint ${ckpt}, skipping"
        FAILED+=("${TASK}/${name} (no ckpt)")
        continue
      fi
      if [ ! -d "$TASK_DIR" ]; then
        echo "=== ${TASK}/${name}: missing task dir ${TASK_DIR}, skipping"
        FAILED+=("${TASK}/${name} (no task dir)")
        continue
      fi

      # matched: the checkpoint's arm and the requested arm must agree; --arm makes a
      # mismatch loud. shared: the mismatch is the design, so it is declared explicitly.
      ARM_FLAG=(--arm "$arm")
      [ "$MODE" = "shared" ] && ARM_FLAG+=(--allow_arm_mismatch)

      mkdir -p "$dir"
      echo "=== ${TASK}/${name} [${MODE}] <- ${ckpt}"

      set +e
      HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
        conda run --no-capture-output -n ehr python -m model_new.train_finetune \
          "${ARM_FLAG[@]}" --seed "$seed" --run_name "$name" \
          --pretrained_ckpt "$ckpt" "${COMMON[@]}" \
          --deviation "backbone selection: MODE=${MODE}, checkpoint ${ckpt}" \
          --deviation "DECISION D3: vocab_choice=${VOCAB_CHOICE}, embedding_path=${EMBEDDINGS}" \
          > "${dir}/train.log" 2>&1
      rc=$?
      set -e

      [ $rc -ne 0 ] && { echo "    FAILED (rc=${rc}) — see ${dir}/train.log"; FAILED+=("${TASK}/${name}"); continue; }
      echo "    done: ${dir}/{config.json,pic_config.json,train.json,best.pt,epoch_*.pt}"
    done
  done
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo; echo "FAILED RUNS: ${FAILED[*]}"; exit 1
fi
