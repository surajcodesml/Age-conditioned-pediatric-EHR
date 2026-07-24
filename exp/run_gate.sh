#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Go/no-go gate. CPU only, ~25 min. Facts confirmed by the agent, baked in:
#   CHEB_TMAX = 6.5  (from within-row pairwise gaps -- the favorable answer)
#   operator  = logsigmoid   scores + F.logsigmoid(poly)
#   masking   = causal       tril & pad
#   two role-distinct kernels: attention site + aggregation site
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-$HOME/Git/Age-conditioned-pediatric-EHR}"
OUT="${OUT:-$REPO/results/gate}"
CKPT="${CKPT:?best CHD fine-tune checkpoint}"
PIC="${PIC:?CHD tensorized test split dir}"
MIMIC="${MIMIC:?MIMIC-IV tensorized val split dir}"
TMAX=6.5

mkdir -p "$OUT"; cd "$REPO"

# --- E2: attention site is primary. Aggregation is inert at fine-tune time
#     (return_repr_only returns first), so expect it at or near zero-init.
python exp/e2_alpha_radius.py --ckpt "$CKPT" --site attention   --out_dir "$OUT"
python exp/e2_alpha_radius.py --ckpt "$CKPT" --site aggregation --out_dir "$OUT" || true
RADIUS=$(python -c "import json;print(json.load(open('$OUT/alpha_attention.json'))['recommended_radius_for_E1'])")
echo "### empirical radius = $RADIUS"

# --- E1 main grid. logsigmoid is what ships; identity is the unbounded upper
#     bound. The gap between them is the operator's share of the ceiling.
for SPLIT in "PIC:$PIC" "MIMIC:$MIMIC"; do
  NAME="${SPLIT%%:*}"; DIR="${SPLIT#*:}"
  for OP in logsigmoid identity; do
    python exp/e1_kernel_headroom.py --split_dir "$DIR" --label "${NAME}_causal_${OP}" \
      --basis chebyshev --cheb_tmax $TMAX --operator "$OP" --masking causal \
      --radius "$RADIUS" --offsets_npz "$OUT/alpha_attention.npz" \
      --max_patients 400 --rows_per_patient 32 --out_dir "$OUT"
  done
done

# --- counterfactual: what would dropping causal masking buy?
python exp/e1_kernel_headroom.py --split_dir "$PIC" --label "PIC_bidir_logsigmoid" \
  --basis chebyshev --cheb_tmax $TMAX --operator logsigmoid --masking bidirectional \
  --radius "$RADIUS" --offsets_npz "$OUT/alpha_attention.npz" --out_dir "$OUT"

# --- degree sweep. Free here, and it settles degree 2/3 vs 5 empirically
#     instead of from coefficient magnitudes.
BASE5=$(python -c "import json;b=json.load(open('$OUT/alpha_attention.json'))['base_coefficients'] or [0.5,0,0,0,0,0];print(' '.join(map(str,b)))")
for D in 2 3 5; do
  BASE=$(python -c "print(' '.join('$BASE5'.split()[:$D+1]))")
  python exp/e1_kernel_headroom.py --split_dir "$PIC" --label "PIC_deg${D}" \
    --basis chebyshev --cheb_tmax $TMAX --operator logsigmoid --masking causal \
    --degree "$D" --base_coeffs $BASE --radius "$RADIUS" --out_dir "$OUT"
done

echo
echo "### GATE ###"
echo "Primary: PIC row-centered max|dlogit|, logsigmoid, causal, learned offsets."
echo "  >= 0.30   launch the 3-arm backbone ablation"
echo "  0.05-0.30 marginal; n=1 seed will not resolve it"
echo "  <  0.05   ceiling confirmed; the PIC-vs-MIMIC gap is the paper"
echo "Also read: domain occupancy x on PIC, and the logsigmoid-vs-identity gap."
