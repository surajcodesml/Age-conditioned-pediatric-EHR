#!/usr/bin/env python3
"""Static analysis of the degree-5 age-conditioned temporal kernel.

Question: do degrees 4-5 of the polynomial actually do work for AGE conditioning
in the pediatric range, or are c4/c5 cosmetic / droppable?

No training. Loads a PIC-finetuned age checkpoint (backbone is frozen during PIC
finetuning, so its temporal kernel == the age pretrain kernel), then:

  1. Age-varying coefficient mass: std of alpha_k(age) across an age grid,
     separately for pediatric [0,18]y and full [0,100]y. This is the age-driven
     variation of each coefficient, reported alongside (but distinct from) the
     base coefficient magnitude |c_k|.

  2. Functional reconstruction: evaluate w(dt; age) on a (dt x age) grid at
     degree 5, least-squares project the degree-5 logit onto a degree-3 basis
     (in x = log1p(dt/7)), and report max/mean |w5 - w3| deviation by age band.
     Also reports the pure truncation variant (drop c4,c5 -> 0) as a cross-check.

The kernel variable is x = log1p(dt_days / 7). Basis is MONOMIAL (no Chebyshev
basis is present in the model code).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.PIC.pic_age_eval_common import DEV_BANDS, PIC_CKPT_ROOT, load_finetuned_classifier


def _dt_grid() -> np.ndarray:
    """dt in days, 0..730: 0 plus log-spaced positives (pediatric-relevant horizon)."""
    pos = np.geomspace(0.5, 730.0, num=200)
    return np.unique(np.concatenate([[0.0], pos]))


def _x_of_days(dt_days: np.ndarray) -> np.ndarray:
    return np.log1p(dt_days / 7.0)


@torch.no_grad()
def _alpha_at_ages(tw, age_emb, ages_yr: np.ndarray) -> np.ndarray:
    """alpha_k(age) = base_k + age_coeff_gen(fourier(age)).  Returns [N_age, deg+1]."""
    age_t = torch.as_tensor(ages_yr, dtype=torch.float32)
    feat = age_emb(age_t.clamp(min=0.0))
    delta = tw.age_coeff_gen(feat)
    alpha = tw.coefficients.unsqueeze(0) + delta
    return alpha.cpu().numpy().astype(np.float64)


def _poly_logit(alpha_row: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate sum_k alpha_k * x^k at monomial degree len(alpha)-1."""
    # np.polyval wants highest-order first
    return np.polyval(alpha_row[::-1], x)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ section 1
def coefficient_mass(tw, age_emb, deg: int) -> dict:
    base = tw.coefficients.detach().cpu().numpy().astype(np.float64)
    ped = np.linspace(0.0, 18.0, 721)      # 0-18y, ~9-day spacing
    full = np.linspace(0.0, 100.0, 2001)   # 0-100y
    a_ped = _alpha_at_ages(tw, age_emb, ped)
    a_full = _alpha_at_ages(tw, age_emb, full)
    return {
        "base": base,
        "std_ped": a_ped.std(axis=0),
        "std_full": a_full.std(axis=0),
        "range_ped": a_ped.max(axis=0) - a_ped.min(axis=0),
        "alpha_ped": a_ped,
        "ages_ped": ped,
    }


# ------------------------------------------------------------------ section 2
def functional_reconstruction(tw, age_emb, deg: int, dt_days: np.ndarray) -> dict:
    """For each age, find the best deg-3 approximation to the deg-5 kernel w(dt).

    The fit minimizes error in w (the kernel value that actually feeds attention),
    not raw logit error: we solve a sensitivity-weighted linear LS in logit space
    with weights s = w5*(1-w5) = d w / d logit, i.e. a first-order linearization of
    w-error. This is the honest 'can degree-3 reproduce this curve where it matters'
    test; unweighted logit LS is ill-conditioned and inflates deviation in the
    saturated region. One Gauss-Newton refinement step tightens the linearization.
    Also reports the pure truncation variant (c4,c5 -> 0, no refit) as a bound."""
    x = _x_of_days(dt_days)
    V5 = np.vander(x, N=deg + 1, increasing=True)     # [G, 6]
    V3 = np.vander(x, N=4, increasing=True)           # [G, 4]

    def weighted_deg3_fit(logit5: np.ndarray) -> np.ndarray:
        w5 = _sigmoid(logit5)
        c3 = np.zeros(4)
        c3[: min(4, deg + 1)] = 0.0  # start from truncation is fine; refine below
        for _ in range(6):  # Gauss-Newton on w-space objective
            z = V3 @ c3
            wz = _sigmoid(z)
            s = np.clip(wz * (1.0 - wz), 1e-6, None)   # d w / d z at current fit
            # solve weighted LS: min || sqrt(s) (z_target - V3 c) ||, with residual
            # target chosen so linearized w matches: z + (w5 - wz)/s
            target = z + (w5 - wz) / s
            sw = np.sqrt(s)
            A = sw[:, None] * V3
            b = sw * target
            c3, *_ = np.linalg.lstsq(A, b, rcond=None)
        return c3

    ages = np.array([b.center_yr for b in DEV_BANDS] +
                    [0.05, 0.25, 1.0, 3.0, 8.0, 15.0, 30.0, 65.0])
    alpha = _alpha_at_ages(tw, age_emb, ages)  # [A, 6]

    rows = []
    for i, a in enumerate(ages):
        logit5 = V5 @ alpha[i]
        w5 = _sigmoid(logit5)
        c3 = weighted_deg3_fit(logit5)
        w3_proj = _sigmoid(V3 @ c3)
        alpha_trunc = alpha[i].copy()
        alpha_trunc[4:] = 0.0
        w3_trunc = _sigmoid(V5 @ alpha_trunc)
        rows.append({
            "age": a,
            "proj_max": float(np.max(np.abs(w5 - w3_proj))),
            "proj_mean": float(np.mean(np.abs(w5 - w3_proj))),
            "trunc_max": float(np.max(np.abs(w5 - w3_trunc))),
            "trunc_mean": float(np.mean(np.abs(w5 - w3_trunc))),
        })
    return {"rows": rows, "dt_days": dt_days}


# ------------------------------------------------------------------ driver
def analyze_module(name: str, tw, age_emb, deg: int) -> None:
    print(f"\n{'='*72}\nMODULE: {name}   (monomial degree {deg}, x = log1p(dt_days/7))\n{'='*72}")

    cm = coefficient_mass(tw, age_emb, deg)
    print("\n[1] AGE-VARYING COEFFICIENT MASS")
    print("     k |    base c_k |  |base c_k| | age-std[0,18y] | age-std[full] | age-range[0,18y]")
    print("    ---+-------------+------------+----------------+---------------+-----------------")
    for k in range(deg + 1):
        print(f"     {k} | {cm['base'][k]:11.4f} | {abs(cm['base'][k]):10.4f} |"
              f" {cm['std_ped'][k]:14.5f} | {cm['std_full'][k]:13.5f} | {cm['range_ped'][k]:15.5f}")
    sp, sf = cm["std_ped"], cm["std_full"]
    tot_ped, tot_full = sp.sum(), sf.sum()
    hi_ped = sp[4:].sum()
    lo_ped = sp[1:3].sum()
    print(f"\n    age-varying std total (ped): {tot_ped:.5f}   full: {tot_full:.5f}")
    if tot_ped > 0:
        print(f"    share of pediatric age-variation in c4+c5: {100*hi_ped/tot_ped:5.1f}%"
              f"   vs c1+c2: {100*lo_ped/tot_ped:5.1f}%")
    print(f"    (c0..c5 ped std share: "
          f"{[f'{100*v/tot_ped:.0f}%' for v in sp] if tot_ped>0 else 'n/a'})")

    fr = functional_reconstruction(tw, age_emb, deg, _dt_grid())
    print("\n[2] FUNCTIONAL RECONSTRUCTION  w5 vs deg-3  (dt in [0,730]d)")
    print("    deviation in w-units (kernel value, 0..1)")
    print("      age(y) | proj max | proj mean | trunc max | trunc mean | band")
    print("    ---------+----------+-----------+-----------+------------+---------")
    band_names = {b.center_yr: b.name for b in DEV_BANDS}
    for r in fr["rows"]:
        ped_flag = "ped" if r["age"] <= 18.0 else "adult"
        bn = band_names.get(r["age"], ped_flag)
        print(f"    {r['age']:8.3f} | {r['proj_max']:8.4f} | {r['proj_mean']:9.4f} |"
              f" {r['trunc_max']:9.4f} | {r['trunc_mean']:10.4f} | {bn}")

    ped_rows = [r for r in fr["rows"] if r["age"] <= 18.0]
    proj_max_ped = max(r["proj_max"] for r in ped_rows)
    proj_mean_ped = np.mean([r["proj_mean"] for r in ped_rows])
    return {
        "std_ped": sp, "std_full": sf, "base": cm["base"],
        "hi_share_ped": (hi_ped / tot_ped) if tot_ped > 0 else 0.0,
        "proj_max_ped": proj_max_ped, "proj_mean_ped": proj_mean_ped,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pneumonia",
                   help="PIC task whose _age backbone to read (kernel is frozen -> "
                        "identical across PIC tasks)")
    p.add_argument("--ckpt_root", type=Path, default=PIC_CKPT_ROOT)
    args = p.parse_args()

    age_dir = args.ckpt_root / f"{args.task}_age"
    print(f"[load] PIC-finetuned age backbone from {age_dir} "
          f"(backbone frozen in finetune -> kernel == age pretrain kernel)")
    model = load_finetuned_classifier(age_dir, "age", torch.device("cpu"))
    bb = model.backbone

    deg = bb.time_aware_attention.temporal_weight.poly_degree
    summ = {}
    summ["attention"] = analyze_module(
        "attention", bb.time_aware_attention.temporal_weight,
        bb.time_aware_attention.age_emb, deg)
    summ["aggregation"] = analyze_module(
        "aggregation", bb.temporal_aggregation.temporal_weight,
        bb.temporal_aggregation.age_emb, deg)

    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    for mod in ("attention", "aggregation"):
        s = summ[mod]
        load_bearing = (s["hi_share_ped"] > 0.15) and (s["proj_max_ped"] > 0.02)
        tag = "LOAD-BEARING" if load_bearing else "DROPPABLE"
        print(f"  [{mod:11s}] c4+c5 carry {100*s['hi_share_ped']:.1f}% of pediatric age-variation; "
              f"deg-3 projection error max={s['proj_max_ped']:.4f} mean={s['proj_mean_ped']:.4f} "
              f"w-units -> degrees 4-5 are {tag} for age conditioning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
