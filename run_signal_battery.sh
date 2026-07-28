#!/usr/bin/env bash
# Signal battery v2 — data/objective diagnostics on DKM pretraining checkpoints.
#
#   ./run_signal_battery.sh              # full run (~85–110 min wall)
#   ./run_signal_battery.sh --smoke      # tiny slice
#   ./run_signal_battery.sh --force      # re-run even if JSON exists
#   ./run_signal_battery.sh --out DIR
#
# Phase schedule (v2):
#   t=0  D2/D5 from train.json (before fork)
#   t=0  fork CPU || GPU
#   CPU: D8.1 → D7
#   GPU: D1 → D8.2 (waits on D8.1) → D4 → D10
#   last: report.py
#
# set -uo pipefail — deliberately NOT set -e.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SMOKE=0
FORCE=0
OUT="model_new/audit/signal/out"
PYTHON=(conda run --no-capture-output -n ehr python)

# Revised estimates (v2): packed cache + trimmed D1 + SQL D9. Target ~85–110 min wall.
EST_D2_D5=60
EST_D8_1=120
EST_D7=2400
EST_D1=2400
EST_D8_2=1500
EST_D4=1500
EST_D10=1200
EST_REPORT=15

usage() {
  cat <<EOF
Usage: $0 [--smoke] [--force] [--out DIR]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --force) FORCE=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$OUT/logs" "$OUT/figures" "$OUT/status"
MASTER="$OUT/battery.log"
FAILED=()
COMPLETED=()
SKIPPED=()
CURRENT=""

ts() { date -Iseconds; }

log() {
  local msg="[$(ts)] $*"
  echo "$msg" | tee -a "$MASTER"
}

ingest_status_dir() {
  local f name st
  for f in "$OUT/status"/*.status; do
    [[ -e "$f" ]] || continue
    name=$(basename "$f" .status)
    st=$(cat "$f")
    case "$st" in
      ok)   [[ " ${COMPLETED[*]:-} " == *" $name "* ]] || COMPLETED+=("$name") ;;
      fail) [[ " ${FAILED[*]:-} " == *" $name "* ]] || FAILED+=("$name") ;;
      skip) [[ " ${SKIPPED[*]:-} " == *" $name "* ]] || SKIPPED+=("$name") ;;
    esac
  done
}

on_signal() {
  log "INT/TERM received while running=${CURRENT:-none}"
  ingest_status_dir
  log "Finished before signal: ${COMPLETED[*]:-none}"
  log "Skipped: ${SKIPPED[*]:-none}"
  log "Failed so far: ${FAILED[*]:-none}"
  exit 130
}
trap on_signal INT TERM

SMOKE_FLAG=()
FORCE_FLAG=()
[[ "$SMOKE" == 1 ]] && SMOKE_FLAG=(--smoke)
[[ "$FORCE" == 1 ]] && FORCE_FLAG=(--force)
common_args=(--out "$OUT" "${SMOKE_FLAG[@]}" "${FORCE_FLAG[@]}")

run_test() {
  local name=$1; shift
  local label=$1; shift
  local est=$1; shift
  CURRENT="$name"

  if [[ -f "$OUT/${name}.json" && "$FORCE" != 1 ]]; then
    log "SKIP  $name (json exists)"
    echo skip > "$OUT/status/${name}.status"
    SKIPPED+=("$name")
    CURRENT=""
    return 0
  fi

  log "START $name  ($label)  est≈${est}s"
  local t0=$SECONDS
  "$@" 2>&1 | tee "$OUT/logs/${name}.log"
  local rc=${PIPESTATUS[0]}
  local dt=$((SECONDS - t0))
  if [[ $rc -eq 0 ]]; then
    log "OK    $name  ${dt}s  (est ${est}s)"
    echo ok > "$OUT/status/${name}.status"
    COMPLETED+=("$name")
  else
    log "FAIL  $name  ${dt}s  rc=$rc  (continuing)"
    echo fail > "$OUT/status/${name}.status"
    FAILED+=("$name")
  fi
  CURRENT=""
  return 0
}

wait_for_json() {
  # wait_for_json <name> <timeout_s> — poll until JSON exists or timeout.
  local name=$1
  local timeout=$2
  local t0=$SECONDS
  while [[ ! -f "$OUT/${name}.json" ]]; do
    if (( SECONDS - t0 > timeout )); then
      log "WAIT  $name timed out after ${timeout}s"
      return 1
    fi
    sleep 5
  done
  return 0
}

log "================================================================"
log "signal battery v2 start  smoke=$SMOKE force=$FORCE out=$OUT"
log "estimates (s): D2/D5≈${EST_D2_D5} D8.1≈${EST_D8_1} D7≈${EST_D7} D1≈${EST_D1} D8.2≈${EST_D8_2} D4≈${EST_D4} D10≈${EST_D10}"
log "target wall ≈85–110 min (GPU critical path)"
if [[ "$SMOKE" == 1 ]]; then
  log "SMOKE mode: 2 batches, 1% patients, 50 bootstrap, 5 perm, top-20 codes"
fi
log "================================================================"

# ---------------------------------------------------------------------------
# Phase 0 — D2/D5 first (user reads immediately; no GPU/CPU contention)
# ---------------------------------------------------------------------------
log "PHASE 0  D2/D5 from train.json  (est ${EST_D2_D5}s)"
run_test d2_d5_logs "phase0" "$EST_D2_D5" \
  "${PYTHON[@]}" -m model_new.audit.signal.d2_d5_logs "${common_args[@]}"

# ---------------------------------------------------------------------------
# Phase 1 — fork CPU || GPU
# ---------------------------------------------------------------------------
log "PHASE 1  fork CPU (D8.1→D7) || GPU (D1→D8.2→D4→D10)"

(
  run_test d8_horizon_hist "cpu" "$EST_D8_1" \
    "${PYTHON[@]}" -m model_new.audit.signal.d8_horizon --part 1 "${common_args[@]}"
  run_test d7_halflife "cpu" "$EST_D7" \
    "${PYTHON[@]}" -m model_new.audit.signal.d7_halflife "${common_args[@]}"
) &
cpu_pid=$!

(
  run_test d1_timestamps "gpu" "$EST_D1" \
    "${PYTHON[@]}" -m model_new.audit.signal.d1_timestamps "${common_args[@]}"
  # D8.2 needs D8.1 JSON; wait up to ~10 min (est 2 min).
  if wait_for_json d8_horizon_hist 600; then
    run_test d8_horizon_recall "gpu" "$EST_D8_2" \
      "${PYTHON[@]}" -m model_new.audit.signal.d8_horizon --part 2 "${common_args[@]}"
  else
    log "FAIL  d8_horizon_recall  (D8.1 missing)"
    echo fail > "$OUT/status/d8_horizon_recall.status"
  fi
  run_test d4_lossmass "gpu" "$EST_D4" \
    "${PYTHON[@]}" -m model_new.audit.signal.d4_lossmass "${common_args[@]}"
  run_test d10_head_align "gpu" "$EST_D10" \
    "${PYTHON[@]}" -m model_new.audit.signal.d10_head_align "${common_args[@]}"
) &
gpu_pid=$!

wait "$cpu_pid" || true
wait "$gpu_pid" || true
ingest_status_dir

# ---------------------------------------------------------------------------
# Phase 2 — report (always)
# ---------------------------------------------------------------------------
log "PHASE 2  report  (est ${EST_REPORT}s)"
CURRENT="report"
t0=$SECONDS
"${PYTHON[@]}" -m model_new.audit.signal.report --out "$OUT" 2>&1 | tee "$OUT/logs/report.log"
rc=${PIPESTATUS[0]}
dt=$((SECONDS - t0))
if [[ $rc -eq 0 ]]; then
  log "OK    report  ${dt}s"
  echo ok > "$OUT/status/report.status"
  COMPLETED+=("report")
else
  log "FAIL  report  ${dt}s  rc=$rc"
  echo fail > "$OUT/status/report.status"
  FAILED+=("report")
fi
CURRENT=""

ingest_status_dir
log "================================================================"
log "SUMMARY"
log "  completed: ${COMPLETED[*]:-none}"
log "  skipped:   ${SKIPPED[*]:-none}"
log "  failed:    ${FAILED[*]:-none}"
log "  report:    $OUT/SIGNAL_REPORT.md"
log "================================================================"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
exit 0
