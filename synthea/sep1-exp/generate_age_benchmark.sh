#!/usr/bin/env bash
# Generate stock-Synthea pediatric CSVs for the age-signal benchmark.
#
# Four separate runs (one per developmental stratum) so coverage is balanced.
# Does NOT load custom modules in synthea/modules/.
# Does NOT write to synthea/output/ (the prior 0-25 custom-module cohort).
#
# Usage:
#   synthea/sep1-exp/generate_age_benchmark.sh pilot
#   synthea/sep1-exp/generate_age_benchmark.sh full
set -euo pipefail

MODE="${1:-pilot}"
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "Usage: $0 [pilot|full]" >&2
  exit 2
fi

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${EXP_DIR}/age_benchmark_config.json"
PROPS="${EXP_DIR}/age_benchmark.properties"

read_cfg() {
  python3 - "$CONFIG" "$1" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
path = sys.argv[2].split(".")
cur = cfg
for p in path:
    if p.isdigit():
        cur = cur[int(p)]
    else:
        cur = cur[p]
if isinstance(cur, (dict, list)):
    print(json.dumps(cur))
else:
    print(cur)
PY
}

ENGINE_DIR="$(read_cfg engine_dir)"
GEOGRAPHY="$(read_cfg geography)"
REF_DATE="$(read_cfg reference_date)"
N="$(read_cfg "n_per_stratum.${MODE}")"
OUT_ROOT="${EXP_DIR}/$(read_cfg output_root)/${MODE}/raw"

mkdir -p "${OUT_ROOT}"

echo "=============================================================="
echo "[age-benchmark] mode          : ${MODE}"
echo "[age-benchmark] engine        : ${ENGINE_DIR}"
echo "[age-benchmark] properties    : ${PROPS}"
echo "[age-benchmark] geography     : ${GEOGRAPHY}"
echo "[age-benchmark] reference     : ${REF_DATE}"
echo "[age-benchmark] end date      : ${REF_DATE}"
echo "[age-benchmark] n per stratum : ${N}"
echo "[age-benchmark] output        : ${OUT_ROOT}"
echo "[age-benchmark] custom modules: NOT loaded"
echo "=============================================================="

n_strata="$(python3 -c "import json; print(len(json.load(open('${CONFIG}'))['strata']))")"
for i in $(seq 0 $((n_strata - 1))); do
  NAME="$(read_cfg "strata.${i}.name")"
  AGE="$(read_cfg "strata.${i}.age_flag")"
  SEED="$(read_cfg "strata.${i}.seed")"
  STRATUM_DIR="${OUT_ROOT}/${NAME}"
  mkdir -p "${STRATUM_DIR}"

  echo
  echo "---------- stratum ${NAME}  -a ${AGE}  -p ${N}  -s ${SEED} ----------"
  (
    cd "${ENGINE_DIR}"
    ./run_synthea \
      -p "${N}" \
      -a "${AGE}" \
      -s "${SEED}" \
      -cs "${SEED}" \
      -r "${REF_DATE}" \
      -e "${REF_DATE}" \
      -o false \
      -c "${PROPS}" \
      --exporter.baseDirectory="${STRATUM_DIR}" \
      --exporter.years_of_history=0 \
      "${GEOGRAPHY}"
  )
  echo "[age-benchmark] wrote ${STRATUM_DIR}/csv/"
  ls -la "${STRATUM_DIR}/csv/" || true
done

echo
echo "[age-benchmark] generation complete (${MODE})."
echo "Next: conda run -n ehr python synthea/sep1-exp/build_age_benchmark.py --mode ${MODE}"
