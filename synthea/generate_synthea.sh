#!/usr/bin/env bash
# Generate synthetic longitudinal pediatric EHR (ages 0-25) with Synthea, loading
# the custom age-varying obesity / T2D / OSA / asthma modules in synthea/modules/.
#
# Usage: synthea/generate_synthea.sh [POPULATION] [SEED]
#   POPULATION : number of patients to attempt (over-generate; heavy dropout from
#                the downstream <2-encounter / <5-event filters is expected).
#   SEED       : RNG + clinician seed for reproducibility.
set -euo pipefail

ENGINE_DIR="/home/suraj/Git/synthea"
REPO_DIR="/home/suraj/Git/Age-conditioned-pediatric-EHR"
MODULES_DIR="${REPO_DIR}/synthea/modules"
CONFIG_FILE="${REPO_DIR}/synthea/synthea_peds.properties"
OUT_DIR="${REPO_DIR}/synthea/output"

POPULATION="${1:-2000}"
SEED="${2:-20260608}"

mkdir -p "${OUT_DIR}"

echo "=============================================================="
echo "[synthea] engine     : ${ENGINE_DIR}"
echo "[synthea] modules    : ${MODULES_DIR}"
echo "[synthea] config     : ${CONFIG_FILE}"
echo "[synthea] output     : ${OUT_DIR}"
echo "[synthea] population : ${POPULATION}  (ages 0-25)"
echo "[synthea] seed       : ${SEED}"
echo "=============================================================="

cd "${ENGINE_DIR}"
# -p population  -a ageRange  -s seed  -cs clinicianSeed  -d localModulesDir  -c overrideConfig
./run_synthea \
  -p "${POPULATION}" \
  -a 0-25 \
  -s "${SEED}" \
  -cs "${SEED}" \
  -d "${MODULES_DIR}" \
  -c "${CONFIG_FILE}"

echo "[synthea] generation complete. CSVs in ${OUT_DIR}/csv/"
ls -la "${OUT_DIR}/csv/" || true
