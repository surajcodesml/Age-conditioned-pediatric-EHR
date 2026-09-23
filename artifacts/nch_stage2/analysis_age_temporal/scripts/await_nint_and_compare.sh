#!/usr/bin/env bash
# Wait until no-interaction Stage-2 training fully finishes, then refresh paired J*.
# Usage: bash artifacts/nch_stage2/analysis_age_temporal/scripts/await_nint_and_compare.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
NINT_DIR="stage2_nch/run/nint_nch_s0"
FINAL="$NINT_DIR/checkpoint_final.pt"
BEST="$NINT_DIR/checkpoint_best_auprc.pt"
LOG="$NINT_DIR/train.log"

echo "[await] Waiting for nint training to finish (need $FINAL) ..."
while true; do
  if [[ -f "$FINAL" ]]; then
    echo "[await] Found $FINAL"
    break
  fi
  # also exit wait if train.log says done and best exists
  if [[ -f "$LOG" ]] && grep -qE '^done run_dir=.*nint_nch_s0' "$LOG" && [[ -f "$BEST" ]]; then
    echo "[await] train.log reports done; using $BEST"
    break
  fi
  # if no training process and best exists for >2 min, proceed
  if [[ -f "$BEST" ]] && ! pgrep -f 'stage2_nch.train --arm no_interaction --run_name nint_nch_s0' >/dev/null 2>&1; then
    sleep 30
    if ! pgrep -f 'stage2_nch.train --arm no_interaction --run_name nint_nch_s0' >/dev/null 2>&1; then
      echo "[await] No nint train process; proceeding with $BEST"
      break
    fi
  fi
  sleep 60
done

OUT="artifacts/nch_stage2/analysis_age_temporal"
mkdir -p "$OUT/raw" "$OUT/predictions" "$OUT/tables" "$OUT/figures"
echo "[await] Refreshing paired J* against $BEST on cuda:0"
PYTHONPATH=. /home/suraj/miniconda3/envs/ehr/bin/python -u <<'PY' 2>&1 | tee artifacts/nch_stage2/analysis_age_temporal/raw/eval_with_nint_final.log
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from stage2_nch.analysis import ANALYSIS, NINT_DIR, PRIMARY_CKPT_NAME, ADKM_DIR
from stage2_nch.analysis.fast_eval import eval_windows, HORIZONS_DAYS, HORIZON_LABELS
from stage2_nch.analysis.eval_protocol import build_model_from_ckpt
from stage2_nch.analysis.run_analysis import paired_compare, write_report, run_abc
from stage2_nch.dataset import NCHForecastDataset, make_nch_collate
from stage2_nch.config import VOCAB_PATH

out = ANALYSIS
device = torch.device("cuda:0")
cfg = json.loads((ADKM_DIR / "config.json").read_text())
ds = NCHForecastDataset(Path(cfg["data"]["paths"]["tensorized_dir"]) / "test", VOCAB_PATH, max_seq_len=1024)
loader = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2,
    collate_fn=make_nch_collate(assert_horizon=False), pin_memory=True,
)

ck = torch.load(NINT_DIR / PRIMARY_CKPT_NAME, map_location="cpu", weights_only=False)
nint_meta = {k: ck.get(k) for k in ("epoch", "val_micro_auprc", "val_bce")}
print("nint primary ckpt", nint_meta, flush=True)

model = build_model_from_ckpt(NINT_DIR / PRIMARY_CKPT_NAME, device)

# full
print("=== nint full ===", flush=True)
dfn, overall, _ = eval_windows(model, loader, device, horizon_days=None)
dfn.to_parquet(out / "predictions" / "nint_windows_full.parquet", index=False)
(out / "raw" / "nint_overall_fast.json").write_text(json.dumps(overall, indent=2) + "\n")

dfa = pd.read_parquet(out / "predictions" / "adkm_windows_full.parquet")
delta = paired_compare(dfa, dfn, "recall@5")
delta.update({
    "nint_epoch": nint_meta.get("epoch"),
    "nint_val_auprc": nint_meta.get("val_micro_auprc"),
    "note": "Final paired compare vs nint checkpoint_best_auprc after training finished",
})
(out / "raw" / "model_delta_overall.json").write_text(json.dumps(delta, indent=2) + "\n")
print("overall Δ", delta, flush=True)

age_rows = []
for band in ["<1", "1-5", "6-11", "12-17"]:
    d = paired_compare(dfa[dfa.age_band == band], dfn[dfn.age_band == band], "recall@5")
    d["age_band"] = band
    age_rows.append(d)
pd.DataFrame(age_rows).to_csv(out / "tables" / "model_delta_by_age.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([r["age_band"] for r in age_rows], [r["delta_point"] for r in age_rows], color="#1f4e79")
ax.errorbar(
    range(len(age_rows)), [r["delta_point"] for r in age_rows],
    yerr=[[r["delta_point"] - r["ci_lo"] for r in age_rows],
          [r["ci_hi"] - r["delta_point"] for r in age_rows]],
    fmt="none", ecolor="k", capsize=3,
)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_ylabel(r"Δ Recall@5 (age-temporal − no-interaction)")
ax.set_title("Paired patient-bootstrap model difference by age")
fig.tight_layout()
fig.savefig(out / "figures" / "fig_model_delta_by_age.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "figures" / "fig_model_delta_by_age.pdf", bbox_inches="tight")
plt.close(fig)

# horizons
rows = []
for horizon, label in zip(HORIZONS_DAYS, HORIZON_LABELS):
    print("=== nint horizon", label, "===", flush=True)
    path = out / "predictions" / f"nint_windows_horizon_{label}.parquet"
    dfn_h, _, _ = eval_windows(model, loader, device, horizon_days=horizon)
    dfn_h.to_parquet(path, index=False)
    dfa_h = pd.read_parquet(out / "predictions" / f"adkm_windows_horizon_{label}.parquet")
    d = paired_compare(dfa_h, dfn_h, "recall@5")
    d["horizon"] = label
    for band in ["<1", "1-5", "6-11", "12-17"]:
        db = paired_compare(
            dfa_h[dfa_h.age_band == band], dfn_h[dfn_h.age_band == band], "recall@5"
        )
        d[f"delta_{band}"] = db["delta_point"]
    rows.append(d)
    print(" ", label, d["delta_point"], flush=True)

pd.DataFrame(rows).to_csv(out / "tables" / "model_delta_by_horizon.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.errorbar(
    range(len(rows)), [r["delta_point"] for r in rows],
    yerr=[[r["delta_point"] - r["ci_lo"] for r in rows],
          [r["ci_hi"] - r["delta_point"] for r in rows]],
    fmt="o-", color="#1f4e79", capsize=3,
)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r["horizon"] for r in rows])
ax.set_ylabel("Δ Recall@5")
ax.set_title("Paired model difference by history horizon")
fig.tight_layout()
fig.savefig(out / "figures" / "fig_model_delta_by_history.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "figures" / "fig_model_delta_by_history.pdf", bbox_inches="tight")
plt.close(fig)

mat = np.array([[r[f"delta_{b}"] for r in rows] for b in ["<1", "1-5", "6-11", "12-17"]], float)
fig, ax = plt.subplots(figsize=(8, 3.8))
vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
ax.set_yticks(range(4))
ax.set_yticklabels(["<1", "1-5", "6-11", "12-17"])
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r["horizon"] for r in rows])
ax.set_title("Δ Recall@5 age×horizon (age-temporal − no-interaction)")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(out / "figures" / "fig_model_delta_age_history.png", dpi=300, bbox_inches="tight")
fig.savefig(out / "figures" / "fig_model_delta_age_history.pdf", bbox_inches="tight")
plt.close(fig)

# refresh report sections via existing writer + overlay
abc = run_abc(primary_name=PRIMARY_CKPT_NAME)
adkm_overall = json.loads((out / "raw" / "adkm_overall_fast.json").read_text())
abl = json.loads((out / "raw" / "adkm_interaction_ablations.json").read_text())
nint_status = {
    "available": True,
    "path": str(NINT_DIR / PRIMARY_CKPT_NAME),
    "paired_overall": delta,
    "nint_ckpt_meta": nint_meta,
    "note": "Final refresh after nint training finished",
}
write_report(out, abc, {"overall": {"recall@5": adkm_overall.get("recall@5")}, "ablations": abl}, nint_status)
print("[await] Paired comparison complete.", flush=True)
PY
echo "[await] Done."
