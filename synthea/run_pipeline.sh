#!/usr/bin/env bash
# STEPS 1-4 for the Synthea comparison: generate -> rolled events -> splits ->
# fresh BGE -> tensorize -> per-task disease cohorts + tensors, with sanity gates
# at each stage. All Synthea artifacts are isolated under data/synthea/ so the
# MIMIC artifacts in data/processed are never touched.
#
# Usage: synthea/run_pipeline.sh [POPULATION] [SEED]
#   If POPULATION is given (>0), (re)generates that many patients first.
#   If omitted/0, uses whatever is already in synthea/output/csv.
set -euo pipefail

source /home/suraj/miniconda3/etc/profile.d/conda.sh
conda activate ehr

REPO="/home/suraj/Git/Age-conditioned-pediatric-EHR"
cd "${REPO}"

PROC="data/synthea/processed"
FT="data/synthea/finetune"
VOCAB="${PROC}/code_vocab.json"
EMB="${PROC}/bge_embeddings.pt"
ROLLED="${PROC}/patient_events_rolled_full.parquet"

POPULATION="${1:-0}"
SEED="${2:-20260608}"

# task -> target code_id (also used as build_disease_cohort --code_prefix)
TASKS=(obesity t2d osa asthma)
declare -A CODE=(
  [obesity]="COND_414916001"
  [t2d]="COND_44054006"
  [osa]="COND_78275009"
  [asthma]="COND_195967001"
)

hr() { echo "=============================================================="; }

if [ "${POPULATION}" -gt 0 ]; then
  hr; echo "[STEP 1] Synthea generation (pop=${POPULATION}, ages 0-25, seed=${SEED})"; hr
  bash synthea/generate_synthea.sh "${POPULATION}" "${SEED}"
fi

hr; echo "[STEP 1c] Onset-age distribution gate (must NOT be flat across bands)"; hr
python synthea/check_onset_distribution.py

hr; echo "[STEP 2] Build rolled event table + code_descriptions.json"; hr
python synthea/build_synthea_events.py --out_dir "${PROC}"

hr; echo "[STEP 3a] Patient-level splits"; hr
python synthea/build_splits_synthea.py --data_dir "${PROC}"

hr; echo "[STEP 3b] Fresh Synthea BGE embeddings + vocab"; hr
python preprocessing/compute_bge_embeddings.py \
  --input "${PROC}/code_descriptions.json" \
  --embeddings_out "${EMB}" --vocab_out "${VOCAB}" --force

hr; echo "[STEP 3c] Tensorize pretraining-style shards (pipeline parity)"; hr
python preprocessing/tensorize.py --data_dir "${PROC}" \
  --out_dir "data/synthea/tensorized" --vocab_path "${VOCAB}"

hr; echo "[STEP 4] Per-task prevalence gate"; hr
python synthea/check_prevalence.py --rolled "${ROLLED}"

LEAK_FAIL=0
for TASK in "${TASKS[@]}"; do
  hr; echo "[STEP 4] Cohort + tensors + leakage gate: ${TASK} (${CODE[$TASK]})"; hr
  python finetune/build_disease_cohort.py \
    --disease "${TASK}" --code_prefix "${CODE[$TASK]}" \
    --data_dir "${PROC}" --out_dir "${FT}/cohorts/${TASK}"
  python finetune/build_disease_tensors.py \
    --cohort_dir "${FT}/cohorts/${TASK}" \
    --events_parquet "${ROLLED}" --vocab_path "${VOCAB}" \
    --out_dir "${FT}/tensorized/${TASK}"
  if ! python synthea/test_cohort_leakage_synthea.py --cohort_dir "${FT}/cohorts/${TASK}"; then
    echo "[LEAKAGE] FAIL for ${TASK} -- length alone predicts the label; any model delta is meaningless."
    LEAK_FAIL=1
  fi
done

hr
if [ "${LEAK_FAIL}" -ne 0 ]; then
  echo "[STEP 4] WARNING: one or more cohorts FAILED the length-leakage gate."
  echo "         Increase the generated population (larger negative-matching pool) and re-run."
else
  echo "[STEP 4] All cohorts passed the length-leakage gate."
fi
hr
echo "Pipeline complete. Next: synthea/run_all.sh"
