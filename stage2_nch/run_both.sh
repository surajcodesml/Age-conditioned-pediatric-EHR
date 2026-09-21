#!/usr/bin/env bash
# Sequential Stage-2 NCH arms. age_temporal first, then no_interaction.
#
#   bash stage2_nch/run_both.sh
#   DEVICE=cpu bash stage2_nch/run_both.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
RUN_ROOT="${RUN_ROOT:-stage2_nch/run}"

run_arm() {
  local arm="$1"
  local name="$2"
  local extra=("${@:3}")
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "$dir"
  echo "=== ${arm} -> ${dir}/train.log  $(date -Iseconds)"
  python -u -m stage2_nch.train \
    --arm "$arm" \
    --run_name "$name" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --run_root "$RUN_ROOT" \
    "${extra[@]}" \
    2>&1 | tee "${dir}/train.log"
  echo "=== done ${name}  $(date -Iseconds)"
}

run_arm age_temporal "adkm_nch_s${SEED}"
run_arm no_interaction "nint_nch_s${SEED}" --skip_tensorize
echo "both arms finished"
