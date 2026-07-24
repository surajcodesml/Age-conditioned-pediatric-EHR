#!/usr/bin/env python3
"""
E2 -- What Delta-alpha(age) did the model actually learn?

Reads a checkpoint's state_dict and reconstructs the age pathway by hand:

    age (years) -> FourierAgeEmbedding (fixed sin/cos, frequencies are a buffer
                   and therefore live in the state_dict)
                -> Linear(32,64) -> GELU -> Linear(64,6)
                -> Delta-alpha in R^6

No model class import, no repo on the path, no GPU.  This matters because it
keeps the diagnostic decoupled from whatever the code looks like today.

Two things come out of it:

  1. ||Delta-alpha(a)||_2 across the lifespan and per age band -- the empirical
     RADIUS.  Feed this to e1_kernel_headroom.py as --radius, otherwise you are
     measuring the headroom of a kernel offset the model never produces.

  2. The actual Delta-alpha vectors, saved to .npz, so E1 can compute headroom
     with the LEARNED offsets rather than hand-set extremes.

USAGE
  python e2_alpha_radius.py --ckpt checkpoints/<run>/best.pt --out_dir results/e1
  python e2_alpha_radius.py --ckpt ... --site aggregation   # the pooling kernel
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


def find_age_pathway(sd: dict, site: str) -> dict:
    """Locate the Fourier frequencies + generator MLP for one conditioning site."""
    site_pat = {
        "attention": r"(time_aware_attention|encoder|attn)",
        "aggregation": r"(temporal_aggregation|aggregation|pool)",
    }[site]

    keys = list(sd.keys())
    freq_keys = [k for k in keys if k.endswith("age_emb.frequencies") and re.search(site_pat, k)]
    w0_keys = [k for k in keys if re.search(r"age_coeff_gen\.mlp\.0\.weight$", k) and re.search(site_pat, k)]
    w2_keys = [k for k in keys if re.search(r"age_coeff_gen\.mlp\.2\.weight$", k) and re.search(site_pat, k)]
    base_keys = [k for k in keys if re.search(r"temporal_weight\.coefficients$", k) and re.search(site_pat, k)]

    if not (freq_keys and w0_keys and w2_keys):
        raise SystemExit(
            f"Could not locate the '{site}' age pathway in this checkpoint.\n"
            f"Looked for *age_emb.frequencies / *age_coeff_gen.mlp.{{0,2}}.weight matching /{site_pat}/.\n"
            f"Matching keys present:\n  " +
            "\n  ".join([k for k in keys if "age" in k or "coeff" in k][:40])
        )

    prefix = w0_keys[0].rsplit("mlp.0.weight", 1)[0]
    def np_(t):
        return np.asarray(t.detach().float().cpu().numpy(), dtype=np.float64)

    return {
        "frequencies": np_(sd[freq_keys[0]]),
        "w0": np_(sd[prefix + "mlp.0.weight"]),
        "b0": np_(sd[prefix + "mlp.0.bias"]),
        "w2": np_(sd[prefix + "mlp.2.weight"]),
        "b2": np_(sd[prefix + "mlp.2.bias"]),
        "base": np_(sd[base_keys[0]]) if base_keys else None,
        "prefix": prefix,
    }


def _gelu(x: np.ndarray) -> np.ndarray:
    # exact erf gelu, matching torch.nn.GELU() default (approximate='none')
    from math import sqrt
    from scipy.special import erf  # noqa: F401  (only if available)
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def _gelu_noscipy(x: np.ndarray) -> np.ndarray:
    # erf via numpy-only rational approximation (Abramowitz & Stegun 7.1.26)
    z = x / np.sqrt(2.0)
    sign = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1.0 + sign * y)


def delta_alpha(ages_years: np.ndarray, pw: dict) -> np.ndarray:
    angles = 2.0 * math.pi * ages_years[:, None] * pw["frequencies"][None, :]
    feat = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    pre = feat @ pw["w0"].T + pw["b0"]
    try:
        h = _gelu(pre)
    except Exception:
        h = _gelu_noscipy(pre)
    return h @ pw["w2"].T + pw["b2"]


AGE_BANDS = [
    ("neonate", 0.0, 1.0 / 12.0), ("infant", 1.0 / 12.0, 2.0),
    ("child", 2.0, 12.0), ("adolescent", 12.0, 18.0),
    ("adult", 18.0, 65.0), ("older_adult", 65.0, 95.0),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--site", choices=["attention", "aggregation"], default="attention")
    p.add_argument("--out_dir", default="results/e1")
    p.add_argument("--max_age", type=float, default=95.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # only needed to unpickle the checkpoint
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = obj
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    if not isinstance(sd, dict):
        raise SystemExit("Could not find a state_dict inside the checkpoint.")

    pw = find_age_pathway(sd, args.site)
    grid = np.linspace(0.0, args.max_age, 951)
    da = delta_alpha(grid, pw)
    norms = np.linalg.norm(da, axis=1)

    # Is the generator even alive?  Zero-init means an untrained pathway gives
    # exactly zero for every age.
    dead = float(np.max(norms)) < 1e-8

    report = {
        "ckpt": args.ckpt,
        "site": args.site,
        "param_prefix": pw["prefix"],
        "generator_is_dead_zero_init": dead,
        "base_coefficients": (pw["base"].tolist() if pw["base"] is not None else None),
        "delta_alpha_norm": {
            "mean": float(norms.mean()), "p50": float(np.percentile(norms, 50)),
            "p95": float(np.percentile(norms, 95)), "max": float(norms.max()),
            "argmax_age_years": float(grid[int(np.argmax(norms))]),
        },
        "by_band": {},
        "recommended_radius_for_E1": float(np.percentile(norms, 95)),
    }
    for name, lo, hi in AGE_BANDS:
        m = (grid >= lo) & (grid < hi)
        if m.sum() == 0:
            continue
        report["by_band"][name] = {
            "norm_mean": float(norms[m].mean()),
            "norm_max": float(norms[m].max()),
        }

    # Pediatric-vs-adult contrast: how different are the kernels the model
    # generates for a 1-year-old and a 60-year-old?  This is the quantity the
    # whole thesis rests on.
    peds = da[(grid >= 0.0) & (grid < 18.0)]
    adults = da[(grid >= 18.0)]
    if peds.size and adults.size:
        report["peds_vs_adult_mean_offset_gap_L2"] = float(
            np.linalg.norm(peds.mean(axis=0) - adults.mean(axis=0))
        )

    np.savez(out_dir / f"alpha_{args.site}.npz",
             ages=grid, delta_alpha=da, norms=norms,
             base=(pw["base"] if pw["base"] is not None else np.zeros(da.shape[1])))
    (out_dir / f"alpha_{args.site}.json").write_text(json.dumps(report, indent=2))

    print(f"\n=== {args.site} age pathway from {args.ckpt} ===")
    if dead:
        print("!! Delta-alpha == 0 for every age. The generator never left zero-init.")
        print("   Headroom analysis is moot until this trains. Check the optimizer group.")
    print(f"||Delta-alpha||  mean={report['delta_alpha_norm']['mean']:.5f}  "
          f"p95={report['delta_alpha_norm']['p95']:.5f}  "
          f"max={report['delta_alpha_norm']['max']:.5f} at age "
          f"{report['delta_alpha_norm']['argmax_age_years']:.1f}y")
    for b, v in report["by_band"].items():
        print(f"  {b:<12} mean={v['norm_mean']:.5f}  max={v['norm_max']:.5f}")
    if "peds_vs_adult_mean_offset_gap_L2" in report:
        print(f"peds-vs-adult mean offset gap: {report['peds_vs_adult_mean_offset_gap_L2']:.5f}")
    print(f"\n--> pass --radius {report['recommended_radius_for_E1']:.4f} to e1_kernel_headroom.py")
    print(f"--> or pass --offsets_npz {out_dir}/alpha_{args.site}.npz to use the learned offsets")


if __name__ == "__main__":
    main()
