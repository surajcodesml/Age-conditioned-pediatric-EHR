#!/usr/bin/env bash
# Finish remaining MIMIC Stage-1 arms (behrt finalize if needed, then medbert+cehrbert),
# then start NCH Stage-2 immediately (no delay).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NCH_NUM_WORKERS="${NCH_NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/results/baselines/mimic/logs"
NCH_LOG_DIR="$REPO_ROOT/results/baselines/nch/logs"
mkdir -p "$LOG_DIR" "$NCH_LOG_DIR"
CHAIN_LOG="$LOG_DIR/chain_mimic_then_nch_${STAMP}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

log "Chain started → $CHAIN_LOG"

# ---- 1) Finalize behrt if training finished but result.json missing (OOM-kill after last epoch) ----
if [[ ! -f results/baselines/mimic/behrt/result.json \
   && -f results/baselines/mimic/behrt/best_checkpoint.pt \
   && -f results/baselines/mimic/behrt/history.json ]]; then
  log "Finalizing behrt from existing checkpoints (skip retrain)..."
  conda run --no-capture-output -n ehr python - <<'PY' 2>&1 | tee -a "$CHAIN_LOG"
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/suraj/Git/Age-conditioned-pediatric-EHR")
sys.path.insert(0, str(REPO))

from model_new.data import TensorizedPretrainDataset, make_collate, dataloader_worker_init
from baselines.behrt.model import BEHRTModel
from baselines.common.training import evaluate_loader, get_device
from baselines.common.capacity_report import count_parameters
from baselines.mimic.runner import RenameLoader, MAX_SEQ_LEN, BASELINE_CONFIGS

run_dir = REPO / "results/baselines/mimic/behrt"
history = json.loads((run_dir / "history.json").read_text())
best_epoch = min(history, key=lambda r: r["val_bce"])["epoch"]
best_val = min(r["val_bce"] for r in history)

tdir = REPO / "data/processed/tensorized_flat"
vocab = REPO / "data/processed/code_vocab.json"
test_ds = TensorizedPretrainDataset(tdir / "test", vocab, max_seq_len=MAX_SEQ_LEN)
n_codes = test_ds.num_codes

model = BEHRTModel(
    n_codes=n_codes, n_targets=n_codes,
    **{**BASELINE_CONFIGS["behrt"], "max_seq_len": MAX_SEQ_LEN + 1},
)
ckpt = run_dir / "best_checkpoint.pt"
state = torch.load(ckpt, map_location="cpu", weights_only=True)
model.load_state_dict(state)
device = get_device("cuda")
model.to(device)

collate = make_collate("one_hot")
loader = RenameLoader(DataLoader(
    test_ds, batch_size=64, shuffle=False, collate_fn=collate,
    num_workers=2, pin_memory=True, worker_init_fn=dataloader_worker_init,
))
print("Running capped test eval for behrt (bs=64, max_batches=100)...", flush=True)
test_metrics = evaluate_loader(model, model.predict, loader, device, max_batches=100)
print("test_metrics:", test_metrics, flush=True)

result = {
    "model": "behrt",
    "seed": 0,
    "smoke": False,
    "n_codes": n_codes,
    "batch_size": 128,
    "val_max_batches": 50,
    "test_max_batches": 100,
    "param_counts": count_parameters(model),
    "train": {
        "best_val_bce": best_val,
        "best_epoch": best_epoch,
        "last_epoch": len(history),
        "epochs_trained": len(history),
        "max_epochs": 10,
        "min_epochs": 5,
        "patience": 3,
        "val_max_batches": 50,
        "stopped_early": True,
        "history": history,
        "total_time_s": history[-1].get("time_s"),
        "finalized_after_kill": True,
    },
    "test_metrics": test_metrics,
    "model_card": model.model_card,
    "AUROC": test_metrics.get("micro_auroc"),
    "AUPRC": test_metrics.get("micro_auprc"),
    "BCE": test_metrics.get("bce"),
    "best_val_bce": best_val,
    "epochs_trained": len(history),
}
(run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
model.save_checkpoint(run_dir)
print(f"Wrote {run_dir / 'result.json'}", flush=True)
PY
  log "behrt finalized."
else
  log "behrt already has result.json or missing ckpts — skip finalize."
fi

# ---- 2) Remaining MIMIC models (runner skips any with result.json) ----
MIMIC_LOG="$LOG_DIR/run_remaining_medbert_cehrbert_${STAMP}.log"
log "Starting MIMIC medbert,cehrbert → $MIMIC_LOG"
conda run --no-capture-output -n ehr python -m baselines.mimic.runner \
  --models medbert,cehrbert \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --val_max_batches 50 \
  --test_max_batches 100 \
  2>&1 | tee -a "$MIMIC_LOG"
log "MIMIC remaining arms done."

# ---- 3) NCH Stage-2 immediately (no delay) ----
NCH_LOG="$NCH_LOG_DIR/run_nch_stage2_${STAMP}.log"
log "Starting NCH Stage-2 immediately → $NCH_LOG"
conda run --no-capture-output -n ehr python -m baselines.nch.runner \
  --models dtr,retain,ehr_bert,behrt,medbert,cehrbert \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NCH_NUM_WORKERS" \
  2>&1 | tee -a "$NCH_LOG"
log "NCH Stage-2 finished."
