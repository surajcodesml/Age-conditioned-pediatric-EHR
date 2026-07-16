#!/usr/bin/env python3
"""Positive control for the kernel temporal-weight mechanism (gate before trusting metrics).

Question: can the kernel *mechanism* move the model's outputs at all on this data? If
forcing two wildly different temporal-weight shapes (fast-decay vs slow-decay) barely
changes the logits, then the kernel cannot matter here regardless of what age feeds it,
and the ablation is measuring nothing — we stop.

Procedure:
  1. Build a kernel classifier from the shared backbone; optionally overlay a trained
     kernel fine-tune checkpoint. Freeze everything.
  2. Take ONE fixed batch (first N test samples; deterministic).
  3. Hand-set the encoder-attention temporal-weight base coefficients to two very
     different Chebyshev vectors:
        fast-decay:  high weight at short lag, ~0 at long lag
        slow-decay:  ~flat high weight at all lags
     (Only the encoder-attention stage matters here: the fine-tune graph runs
     return_repr_only=True, so the aggregation stage is never called.)
  4. Report max|Delta w| over a canonical tau grid (sanity: the shape space is
     expressive) and max|Delta logit| over the fixed batch between the two shapes
     (the real test: can it move outputs).

Large max|Delta logit|  => mechanism CAN move outputs -> proceed.
Near-zero               => mechanism structurally cannot on this data -> STOP.

Usage:
  conda run -n ehr python model_ablation/positive_control.py \
      --pretrained_ckpt checkpoints/ablation_pretrain/run20260703_194128/epoch_010.pt \
      --tensorized_dir data/finetune/heart_malformations \
      [--finetune_ckpt checkpoints/finetune/ablation/<run>/kernel/best.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_ablation.dataset_finetune import TensorizedDiseaseClassificationDataset, disease_collate
from model_ablation.model_finetune import TALEEHRAblationClassifier
from model_ablation.time_aware_attention_age import CHEB_TMAX, _chebyshev_powers

# Logit-shift threshold below which we declare the mechanism structurally inert.
STOP_THRESHOLD = 1e-3
# Canonical lag grid (days) -> log1p(dt/7), matching the collate transform.
TAU_DAYS = np.array([0, 1, 3, 7, 14, 30, 90, 180, 365, 730, 1825, 3650], dtype=np.float64)
PROBE_AGES = (0.5, 2.0, 10.0)  # years


def _fixed_batch(tensorized_dir: Path, split: str, batch_size: int, max_seq_len: int) -> dict:
    ds = TensorizedDiseaseClassificationDataset(tensorized_dir / split, max_seq_len=max_seq_len)
    items = [ds[i] for i in range(min(batch_size, len(ds)))]
    return disease_collate(items)


@torch.no_grad()
def _w_curve(tw, age_emb, forced_coeffs: torch.Tensor, age_years: float, device) -> np.ndarray:
    """w(tau | age) on TAU_DAYS for a single age, given forced base coefficients.
    Includes the trained age_coeff_gen delta so this is the true conditioned kernel."""
    deg = tw.poly_degree
    log_delta = torch.log1p(torch.tensor(TAU_DAYS / 7.0, dtype=torch.float32, device=device))
    x = 2.0 * log_delta / CHEB_TMAX - 1.0
    basis = _chebyshev_powers(x, deg)                                  # list of [G]
    af = age_emb(torch.tensor([age_years], dtype=torch.float32, device=device))  # [1, age_dim]
    dalpha = tw.age_coeff_gen(af)[0]                                   # [D+1]
    alpha = forced_coeffs.to(device) + dalpha
    poly = torch.zeros_like(log_delta)
    for k in range(deg + 1):
        poly = poly + alpha[k] * basis[k]
    return torch.sigmoid(poly).cpu().numpy()


@torch.no_grad()
def _logits(model, tw, forced_coeffs: torch.Tensor, batch: dict, device) -> np.ndarray:
    saved = tw.coefficients.data.clone()
    try:
        tw.coefficients.data.copy_(forced_coeffs.to(device))
        out = model(batch)
    finally:
        tw.coefficients.data.copy_(saved)
    return out.detach().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Kernel mechanism positive control.")
    ap.add_argument("--pretrained_ckpt", type=Path, required=True,
                    help="shared backbone checkpoint (defines the architecture)")
    ap.add_argument("--finetune_ckpt", type=Path, default=None,
                    help="optional kernel-arm best.pt to overlay trained weights")
    ap.add_argument("--tensorized_dir", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)

    model = TALEEHRAblationClassifier("kernel", args.pretrained_ckpt, freeze_backbone=True).to(device)
    if args.finetune_ckpt is not None:
        ck = torch.load(args.finetune_ckpt, map_location="cpu")
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        load = model.load_state_dict(sd, strict=False)
        print(f"[load] finetune_ckpt overlaid (missing={len(load.missing_keys)}, "
              f"unexpected={len(load.unexpected_keys)})", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tw = model.backbone.time_aware_attention.temporal_weight
    age_emb = model.backbone.time_aware_attention.age_emb
    deg = tw.poly_degree
    # Two MAXIMALLY different Chebyshev shapes. x = -1 <-> tau=0, x = +1 <-> long lag.
    # The kernel enters attention as scores += logsigmoid(poly); softmax is shift-invariant,
    # so only ACROSS-LAG variation can move outputs. Recency vs anti-recency are opposite
    # shapes -> they suppress opposite ends of the lag axis, the strongest fair contrast.
    fast = torch.zeros(deg + 1); fast[1] = -6.0   # poly = -6 x -> w high at tau~0, ~0 at long (recency)
    slow = torch.zeros(deg + 1); slow[1] = +6.0   # poly = +6 x -> w ~0 at tau~0, high at long (anti-recency)

    batch = _fixed_batch(args.tensorized_dir, args.split, args.batch_size, args.max_seq_len)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    # ---- max|Delta w| over the tau grid and probe ages (shape-space expressiveness) ----
    dw_max = 0.0
    w_report: dict[str, dict[str, list[float]]] = {}
    for age in PROBE_AGES:
        wf = _w_curve(tw, age_emb, fast, age, device)
        ws = _w_curve(tw, age_emb, slow, age, device)
        dw_max = max(dw_max, float(np.max(np.abs(wf - ws))))
        w_report[f"age_{age}"] = {"w_fast": [round(float(x), 4) for x in wf],
                                  "w_slow": [round(float(x), 4) for x in ws]}

    # ---- max|Delta logit| over the fixed batch (can the mechanism move outputs?) ----
    lf = _logits(model, tw, fast, batch, device)
    ls = _logits(model, tw, slow, batch, device)
    dlogit = np.abs(lf - ls)
    dlogit_max = float(np.max(dlogit))
    dlogit_mean = float(np.mean(dlogit))

    print("=" * 70)
    print("POSITIVE CONTROL — kernel temporal-weight mechanism")
    print(f"  tau grid (days): {TAU_DAYS.astype(int).tolist()}")
    print(f"  batch: n={lf.shape[0]} from {args.tensorized_dir}/{args.split}")
    print(f"  max|Delta w|      (fast vs slow, over tau x ages) = {dw_max:.4f}")
    print(f"  max|Delta logit|  (fast vs slow, over batch)      = {dlogit_max:.6f}")
    print(f"  mean|Delta logit| (over batch)                    = {dlogit_mean:.6f}")
    verdict = "PROCEED" if dlogit_max >= STOP_THRESHOLD else "STOP"
    if verdict == "PROCEED":
        print(f"  VERDICT: PROCEED — mechanism can move outputs (max|Delta logit| >= {STOP_THRESHOLD:g}).")
    else:
        print(f"  VERDICT: *** STOP *** — mechanism is structurally inert on this data "
              f"(max|Delta logit| < {STOP_THRESHOLD:g}). Kernel age-conditioning cannot matter here.")
    print("=" * 70)

    report = {
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "finetune_ckpt": str(args.finetune_ckpt) if args.finetune_ckpt else None,
        "tensorized_dir": str(args.tensorized_dir), "split": args.split,
        "batch_n": int(lf.shape[0]), "poly_degree": int(deg),
        "coeffs_fast": fast.tolist(), "coeffs_slow": slow.tolist(),
        "tau_days": TAU_DAYS.astype(int).tolist(), "probe_ages_years": list(PROBE_AGES),
        "max_abs_delta_w": dw_max,
        "max_abs_delta_logit": dlogit_max, "mean_abs_delta_logit": dlogit_mean,
        "stop_threshold": STOP_THRESHOLD, "verdict": verdict,
        "w_curves": w_report,
    }
    out = args.out or (args.tensorized_dir / "positive_control.json")
    with Path(out).open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] {out}", flush=True)
    return 0 if verdict == "PROCEED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
