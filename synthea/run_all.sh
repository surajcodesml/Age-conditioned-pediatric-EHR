#!/usr/bin/env bash
# STEP 6: runnable comparison of age-conditioned vs vanilla TALE-EHR backbones on
# the Synthea disease-onset tasks. SEQUENTIAL nested loop (one run at a time):
#     for CKPT in vanilla age:
#       for TASK in <tasks>:
#         finetune the same auto-detecting classifier, tagged run_dir by ckpt+task
# Then collect every history.json and print the task x ckpt x band summary table.
#
# Prerequisite: synthea/run_pipeline.sh has built the cohorts + tensorized shards
# under data/synthea/finetune/.
set -euo pipefail

source /home/suraj/miniconda3/etc/profile.d/conda.sh
conda activate ehr

REPO="/home/suraj/Git/Age-conditioned-pediatric-EHR"
cd "${REPO}"

PROC="data/synthea/processed"
FT="data/synthea/finetune"
VOCAB="${PROC}/code_vocab.json"
EMB="${PROC}/bge_embeddings.pt"
RUNS_ROOT="${REPO}/checkpoints/synthea_compare"

SEED=42
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TASKS=(obesity t2d osa asthma)

run_one() {
  local ckpt_name="$1" ckpt_path="$2" task="$3"
  local run_dir="${RUNS_ROOT}/${task}_${ckpt_name}"
  echo "######################################################################"
  echo "# RUN  ckpt=${ckpt_name}  task=${task}"
  echo "#   pretrained_ckpt = ${ckpt_path}"
  echo "#   cohort_dir      = ${FT}/cohorts/${task}"
  echo "#   tensorized_dir  = ${FT}/tensorized/${task}"
  echo "#   run_dir         = ${run_dir}"
  echo "######################################################################"
  python synthea/train_synthea.py \
    --disease "${task}" \
    --pretrained_ckpt "${ckpt_path}" \
    --cohort_dir "${FT}/cohorts/${task}" \
    --tensorized_dir "${FT}/tensorized/${task}" \
    --vocab_path "${VOCAB}" \
    --embedding_path "${EMB}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --run_dir "${run_dir}"
}

main() {
  mkdir -p "${RUNS_ROOT}"
  # Nested loop: ckpt (outer) x task (inner), strictly sequential, one at a time.
  for CKPT_NAME in vanilla age; do
    if [ "${CKPT_NAME}" = "vanilla" ]; then CKPT_PATH="${VANILLA_CKPT}"; else CKPT_PATH="${AGE_CKPT}"; fi
    for TASK in "${TASKS[@]}"; do
      run_one "${CKPT_NAME}" "${CKPT_PATH}" "${TASK}"
    done
  done

  echo "######################################################################"
  echo "# SUMMARY: task x ckpt x developmental band"
  echo "######################################################################"
  python synthea/summarize_runs.py --runs_root "${RUNS_ROOT}" --tasks "${TASKS[@]}"
}

# ---------------------------------------------------------------------------
# Pretrained backbone checkpoints (paths provided with the task).
# Edit these two lines to point at different backbones.
# ---------------------------------------------------------------------------
VANILLA_CKPT="${REPO}/checkpoints/run_20260427_152603/best_pretrain.pt"
AGE_CKPT="${REPO}/checkpoints/age_real_202605112156/epoch_010.pt"

main
