#!/usr/bin/env bash
# Wait for the current MIMIC Stage-1 remaining-arm run to finish, sleep DELAY_HOURS,
# then launch NCH Stage-2 finetuning for all arms using Stage-1 checkpoints
# (including dtr_* and retain-backup/).
#
# Usage:
#   bash baselines/run_nch_stage2_after_mimic.sh
#   DELAY_HOURS=0.25 BATCH_SIZE=128 bash baselines/run_nch_stage2_after_mimic.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DELAY_HOURS="${DELAY_HOURS:-0}"  # 15 minutes after MIMIC finishes
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-dtr,retain,ehr_bert,behrt,medbert,cehrbert}"
POLL_SEC="${POLL_SEC:-120}"

MIMIC_DIR="$REPO_ROOT/results/baselines/mimic"
LOG_DIR="$MIMIC_DIR/logs"
NCH_LOG_DIR="$REPO_ROOT/results/baselines/nch/logs"
mkdir -p "$LOG_DIR" "$NCH_LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
WATCH_LOG="$NCH_LOG_DIR/wait_then_nch_${STAMP}.log"
NCH_RUN_LOG="$NCH_LOG_DIR/run_nch_stage2_${STAMP}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"; }

mimic_runner_alive() {
  pgrep -f 'python -m baselines.mimic.runner' >/dev/null 2>&1
}

# Remaining MIMIC arms that the current job is producing (exclude DTR / retain-backup).
required_mimic_results=(
  "$MIMIC_DIR/ehr_bert/result.json"
  "$MIMIC_DIR/behrt/result.json"
  "$MIMIC_DIR/medbert/result.json"
  "$MIMIC_DIR/cehrbert/result.json"
)

mimic_results_ready() {
  local f
  for f in "${required_mimic_results[@]}"; do
    [[ -f "$f" ]] || return 1
  done
  return 0
}

check_stage1_ckpts() {
  local missing=0
  # DTR arms
  for arm in dtr_age_temporal dtr_no_interaction; do
    if [[ ! -f "$MIMIC_DIR/$arm/checkpoint.pt" && ! -f "$MIMIC_DIR/$arm/best_checkpoint.pt" ]]; then
      log "MISSING Stage-1 ckpt: $arm"
      missing=1
    else
      log "OK Stage-1 ckpt: $arm"
    fi
  done
  # retain (prefer retain/, else retain-backup/)
  if [[ -f "$MIMIC_DIR/retain/checkpoint.pt" || -f "$MIMIC_DIR/retain/best_checkpoint.pt" ]]; then
    log "OK Stage-1 ckpt: retain"
  elif [[ -f "$MIMIC_DIR/retain-backup/checkpoint.pt" || -f "$MIMIC_DIR/retain-backup/best_checkpoint.pt" ]]; then
    log "OK Stage-1 ckpt: retain-backup (alias)"
  else
    log "MISSING Stage-1 ckpt: retain / retain-backup"
    missing=1
  fi
  for m in ehr_bert behrt medbert cehrbert; do
    if [[ ! -f "$MIMIC_DIR/$m/checkpoint.pt" && ! -f "$MIMIC_DIR/$m/best_checkpoint.pt" ]]; then
      log "MISSING Stage-1 ckpt: $m"
      missing=1
    else
      log "OK Stage-1 ckpt: $m"
    fi
  done
  return "$missing"
}

log "Watcher started. Log: $WATCH_LOG"
log "Will wait for MIMIC runner to exit AND required result.json files, then sleep ${DELAY_HOURS}h."
log "NCH models: $MODELS  batch_size=$BATCH_SIZE  workers=$NUM_WORKERS"

# ---- phase 1: wait for MIMIC ----
while true; do
  alive=0
  mimic_runner_alive && alive=1
  ready=0
  mimic_results_ready && ready=1
  log "poll: mimic_alive=$alive results_ready=$ready"
  if [[ "$alive" -eq 0 && "$ready" -eq 1 ]]; then
    log "MIMIC remaining arms finished."
    break
  fi
  if [[ "$alive" -eq 0 && "$ready" -eq 0 ]]; then
    log "WARNING: no mimic.runner process, but required results missing — still waiting."
  fi
  sleep "$POLL_SEC"
done

# ---- phase 2: delay ----
delay_sec="$(python3 -c "print(int(float('$DELAY_HOURS') * 3600))")"
log "Sleeping ${DELAY_HOURS}h (${delay_sec}s) before NCH Stage-2..."
sleep "$delay_sec"

# ---- phase 3: preflight ----
log "Preflight Stage-1 checkpoints..."
if ! check_stage1_ckpts; then
  log "ERROR: missing Stage-1 checkpoints; aborting NCH Stage-2."
  exit 1
fi

# ---- phase 4: launch NCH ----
log "Starting NCH Stage-2 → $NCH_RUN_LOG"
# shellcheck disable=SC2086
nohup conda run --no-capture-output -n ehr python -m baselines.nch.runner \
  --models "$MODELS" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --val_max_batches 50 \
  --test_max_batches 100 \
  --clear_smoke \
  > "$NCH_RUN_LOG" 2>&1 &
nch_pid=$!
log "NCH Stage-2 launched pid=$nch_pid"
log "Tail: tail -f $NCH_RUN_LOG"
wait "$nch_pid"
rc=$?
log "NCH Stage-2 exited rc=$rc"
exit "$rc"
