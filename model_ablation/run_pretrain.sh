#!/usr/bin/env bash
# Full shared-vanilla pretrain for the ablation, with config + per-epoch metrics JSON.
# Writes to a fresh timestamped save_dir so it never clobbers a previous run.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

STAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="checkpoints/ablation_pretrain/run${STAMP}"
LOG="${SAVE_DIR}/train.log"
mkdir -p "$SAVE_DIR"

# Single GPU, single-threaded BLAS in the parent (workers set their own via worker_init).
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 \
nohup conda run --no-capture-output -n ehr python model_ablation/train.py \
    --tensorized_dir data/processed/tensorized_flat \
    --embedding_path data/processed/bge_embeddings.pt \
    --vocab_path data/processed/code_vocab.json \
    --epochs 8 \
    --batch_size 16 \
    --lr 1e-4 \
    --d_model 256 \
    --poly_degree 5 \
    --num_workers 6 \
    --seed 42 \
    --val_max_batches 50 \
    --device cuda \
    --save_dir "$SAVE_DIR" \
    > "$LOG" 2>&1 &

PID=$!
echo "pretrain started: PID ${PID}"
echo "  save_dir : ${SAVE_DIR}"
echo "  log      : ${LOG}"
echo "  config   : ${SAVE_DIR}/config.json"
echo "  metrics  : ${SAVE_DIR}/pretrain_metrics.json   (rewritten after every epoch)"
echo
echo "monitor with:  tail -f ${LOG}"
