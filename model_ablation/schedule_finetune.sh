#!/usr/bin/env bash
# Schedule the four-arm fine-tune ablation to launch later, unattended.
#
#   Mode A (default): wait until the running pretrain(s) finish, then run.
#   Mode B:           wait a fixed delay, then run.
#
# Usage (launch DETACHED so it survives terminal/logout):
#   nohup bash model_ablation/schedule_finetune.sh              > checkpoints/finetune_scheduled.log 2>&1 &
#   nohup bash model_ablation/schedule_finetune.sh --after 4h   > checkpoints/finetune_scheduled.log 2>&1 &
#
# Delay accepts s / m / h suffixes (e.g. 90m, 4h, 3600).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

MODE="pretrain"
DELAY=""
if [ "${1:-}" = "--after" ]; then
  MODE="delay"; DELAY="${2:?usage: --after <delay like 4h>}"
fi

to_secs() {
  local v="$1"
  case "$v" in
    *h) echo $(( ${v%h} * 3600 )) ;;
    *m) echo $(( ${v%m} * 60 )) ;;
    *s) echo "${v%s}" ;;
    *)  echo "$v" ;;
  esac
}

echo "[schedule] $(date '+%F %T') mode=$MODE"
if [ "$MODE" = "delay" ]; then
  secs=$(to_secs "$DELAY")
  echo "[schedule] sleeping ${secs}s (~$DELAY), will launch around $(date -d "+${secs} seconds" '+%F %T' 2>/dev/null || echo '+'"$DELAY")"
  sleep "$secs"
else
  echo "[schedule] waiting for pretrain (model_ablation/train.py) to finish ..."
  while pgrep -f "model_ablation/train.py" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[schedule] pretrain finished at $(date '+%F %T')."
  sleep 30   # let GPU memory settle before the fine-tune claims it
fi

echo "[schedule] launching fine-tune ablation at $(date '+%F %T')"
exec bash model_ablation/run_finetune_ablation.sh
