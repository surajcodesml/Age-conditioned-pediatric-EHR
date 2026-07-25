#!/usr/bin/env bash
# Four-arm pretrain. Identical flags apart from --arm and --run_name.
#
#   ./model_new/run/pretrain.sh            # all four arms, seed 0, sequential
#   SEEDS="0 1 2" ./model_new/run/pretrain.sh
#   ARMS="kernel vanilla" ./model_new/run/pretrain.sh
#
# Extra seeds are a loop, not an edit.
set -euo pipefail
cd "$(dirname "$0")/../.."

ARMS="${ARMS:-vanilla kernel random_constant additive}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
RUN_ROOT="${RUN_ROOT:-model_new/run}"

# LRs scaled by sqrt(BATCH/16) from the setting validated by the step-200 probe.
# B=64 -> sqrt(4) = 2x. Group ratios (1:10:10) are preserved deliberately.
# If you change BATCH, change these together and re-run the step-200 probe.
LR_BACKBONE="${LR_BACKBONE:-2e-4}"
LR_AGE="${LR_AGE:-2e-3}"
LR_HEAD="${LR_HEAD:-2e-3}"

# Every arm gets exactly these. Nothing below is tuned per arm.
COMMON=(
  --tensorized_dir data/processed/tensorized_flat
  --embedding_path data/processed/bge_embeddings.pt
  --vocab_path     data/processed/code_vocab.json
  --epochs "$EPOCHS" --batch_size "$BATCH" --num_workers "$WORKERS"
  --d_model 256 --n_layers 1 --n_heads 1 --s 5
  --lr_backbone "$LR_BACKBONE" --lr_age "$LR_AGE" --lr_head "$LR_HEAD"
  --race_encoding one_hot --device cuda --run_root "$RUN_ROOT"
)

FAILED=()

for seed in $SEEDS; do
  for arm in $ARMS; do
    # Unique per-run folder: e.g. kernel_s0_072420260943 (mmddyyyyHHMM).
    # Every invocation writes a fresh directory; no prior run is reused or
    # overwritten. --run_name still records the timestamped name.
    ts="$(date +%m%d%Y%H%M)"
    name="${arm}_s${seed}_${ts}"
    dir="${RUN_ROOT}/${name}"

    mkdir -p "$dir"
    echo "=== ${name} -> ${dir}/train.log"

    # set -e would abort every remaining arm on one failure. Over a multi-day
    # sequential run that is the difference between losing 20 minutes and two days.
    set +e
    HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 \
      conda run --no-capture-output -n ehr python -m model_new.train \
        --arm "$arm" --seed "$seed" --run_name "$name" "${COMMON[@]}" \
        > "${dir}/train.log" 2>&1
    rc=$?
    set -e

    if [ $rc -ne 0 ]; then
      echo "    FAILED (rc=${rc}) — see ${dir}/train.log"
      FAILED+=("$name")
      continue
    fi

    # Stable pointer to the checkpoint chosen by validation. Ideally train.py
    # writes this itself; this is the fallback. Adjust the metric key to match
    # your train.json schema.
    python - "$dir" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
hist = json.load(open(os.path.join(d, "train.json")))
best = max(hist, key=lambda e: e["recall@20"])   # was e["val"]["recall@20"]
src  = f"epoch_{best['epoch']:03d}.pt"
if os.path.exists(os.path.join(d, src)):
    link = os.path.join(d, "best.pt")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(src, link)
    print(f"    best.pt -> {src}  (val recall@20 {best['val']['recall@20']:.4f})")
PYEOF

    echo "    done: ${dir}/{config.json,train.json,paper_numbers.json,best.pt}"
  done
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "FAILED RUNS: ${FAILED[*]}"
  exit 1
fi
