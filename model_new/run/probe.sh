#!/usr/bin/env bash
# Frozen-representation linear probe across the four pretrained arms.
#
#   ./model_new/run/probe.sh
#   CKPT_MAP="vanilla=a/epoch_008.pt kernel=b/epoch_006.pt ..." ./model_new/run/probe.sh
#   RUN_NAME=probe_s0 SEED=0 ./model_new/run/probe.sh
#
# Checkpoints are pretraining epoch_NNN.pt files (best val loss in train.json when
# pointing at a run dir). Nothing is fine-tuned; the encoder stays frozen.
set -euo pipefail
cd "$(dirname "$0")/../.."

CKPT_ROOT="${CKPT_ROOT:-model_new/run_selected}"
CKPT_MAP="${CKPT_MAP:-}"
RUN_ROOT="${RUN_ROOT:-model_new/run}"
RUN_NAME="${RUN_NAME:-probe_s0}"
SEED="${SEED:-0}"
ARMS="${ARMS:-vanilla kernel random_constant additive}"
BATCH="${BATCH:-128}"
WORKERS="${WORKERS:-8}"
TRAIN_SUBSAMPLE="${TRAIN_SUBSAMPLE:-50000}"
DEVICE="${DEVICE:-cuda}"

INDOMAIN_DIR="${INDOMAIN_DIR:-data/finetune/t2d_tensorized}"
INDOMAIN_TASK="${INDOMAIN_TASK:-t2d}"
OOD_DIR="${OOD_DIR:-data/tensorized/pic/pneumonia}"
OOD_TASK="${OOD_TASK:-pneumonia}"
OOD_EMBEDDINGS="${OOD_EMBEDDINGS:-data/processed/pic/bge_embeddings_pic.pt}"

# Selection.json-backed run dirs (used to pick epoch by val_loss when CKPT_MAP is empty
# or points at best.pt copies under CKPT_ROOT).
RUNS_MAP="${RUNS_MAP:-vanilla=model_new/run/vanilla_s0 kernel=model_new/run/kernel_s0_072420260946 random_constant=model_new/run/random_constant_s0_072420261750 additive=model_new/run/additive_s0_072520260143}"

ckpt_for() {
  local arm="$1" entry
  if [ -n "$CKPT_MAP" ]; then
    for entry in $CKPT_MAP; do
      case "$entry" in "${arm}="*) echo "${entry#*=}"; return 0 ;; esac
    done
    echo ""
    return 0
  fi
  # Prefer epoch selected via RUNS_MAP + train.json; fall back to CKPT_ROOT/best.pt.
  local run_dir=""
  for entry in $RUNS_MAP; do
    case "$entry" in "${arm}="*) run_dir="${entry#*=}" ;; esac
  done
  if [ -n "$run_dir" ] && [ -f "${run_dir}/train.json" ]; then
    python - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
hist = json.load(open(d / "train.json"))
best = min(hist, key=lambda e: float(e["val_loss"]))
print(d / f"epoch_{int(best['epoch']):03d}.pt")
PY
    return 0
  fi
  echo "${CKPT_ROOT}/${arm}_s${SEED}/best.pt"
}

CKPT_ARGS=()
RUN_ARGS=()
for arm in $ARMS; do
  ckpt="$(ckpt_for "$arm")"
  if [ -z "$ckpt" ] || [ ! -e "$ckpt" ]; then
    echo "=== probe: missing checkpoint for ${arm}: ${ckpt:-none}" >&2
    exit 1
  fi
  CKPT_ARGS+=("${arm}=${ckpt}")
done
for entry in $RUNS_MAP; do
  RUN_ARGS+=("$entry")
done

dir="${RUN_ROOT}/${RUN_NAME}"
mkdir -p "$dir"
echo "=== probe ${RUN_NAME} -> ${dir}/probe.log"
echo "    checkpoints: ${CKPT_ARGS[*]}"

HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 \
  conda run --no-capture-output -n ehr python -m model_new.probe \
    --run_name "$RUN_NAME" --run_root "$RUN_ROOT" --seed "$SEED" \
    --arms $ARMS \
    --ckpt_map "${CKPT_ARGS[@]}" \
    --runs "${RUN_ARGS[@]}" \
    --train_subsample "$TRAIN_SUBSAMPLE" --batch_size "$BATCH" --num_workers "$WORKERS" --device "$DEVICE" \
    --indomain_tensorized "$INDOMAIN_DIR" --indomain_task "$INDOMAIN_TASK" \
    --ood_tensorized "$OOD_DIR" --ood_task "$OOD_TASK" \
    --ood_embedding_path "$OOD_EMBEDDINGS" \
    > "${dir}/probe.log" 2>&1

echo "    done: ${dir}/{probe.json,paper_numbers.json,repr_cache/}"
