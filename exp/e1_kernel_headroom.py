#!/usr/bin/env python3
"""
E1/E2 -- Within-row tau spread and kernel headroom.

Answers, without training anything and without loading the model graph:

  Q1  How much spread is there in tau_tilde = log1p(|t_i - t_j| / 7) *within a
      single attention row*?  Softmax is shift-invariant, so a kernel that adds
      a near-constant bias across a row has literally zero effect on attention.
      The marginal tau histogram does not answer this; only the within-row
      spread does.

  Q2  Given that spread, what is the largest attention-logit change any
      admissible kernel coefficient vector can produce?  This is the honest
      "headroom" number -- the ceiling on the mechanism, independent of
      training quality.

WHY THIS NEEDS NO MODEL
-----------------------
Under additive log-space injection the score is

    s_ij = (q_i . k_j) / sqrt(d)  +  g(tau_ij ; alpha)

so for two coefficient vectors alpha_A, alpha_B the logit *difference* is

    delta_ij = g(tau_ij ; alpha_A) - g(tau_ij ; alpha_B)

and the q.k term cancels exactly.  The trained weights are irrelevant to the
headroom.  Only tau and alpha matter.  (This is not true for multiplicative
injection, which is one more reason additive is the right operator.)

Softmax over row i is invariant to adding a constant to the whole row, so the
quantity that actually reaches the attention weights is the ROW-CENTERED
difference:  delta_ij - mean_j(delta_ij).  Reporting uncentered max|delta| --
which is probably what the earlier 0.0059 figure was -- overstates the effect.
This script reports both.

USAGE
-----
  python e1_kernel_headroom.py \
      --split_dir data/finetune/heart_malformations_tensorized/test \
      --label PIC_CHD --basis chebyshev --cheb_tmax 4.0 \
      --out_dir results/e1

  # add a second corpus to get the contrast that makes the figure work
  python e1_kernel_headroom.py \
      --split_dir data/processed/tensorized_flat/val \
      --label MIMIC_IV --basis chebyshev --cheb_tmax 4.0 \
      --out_dir results/e1

Outputs <out_dir>/<label>_headroom.json and three PNGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Age bands (patient-level, taken at the last event = the aggregation query)
# --------------------------------------------------------------------------
AGE_BANDS = [
    ("neonate", 0.0, 1.0 / 12.0),
    ("infant", 1.0 / 12.0, 2.0),
    ("child", 2.0, 12.0),
    ("adolescent", 12.0, 18.0),
    ("adult", 18.0, 65.0),
    ("older_adult", 65.0, 1e9),
]


def age_band(age_years: float) -> str:
    for name, lo, hi in AGE_BANDS:
        if lo <= age_years < hi:
            return name
    return "unknown"


# --------------------------------------------------------------------------
# Polynomial bases
# --------------------------------------------------------------------------
def design_matrix(tau: np.ndarray, degree: int, basis: str, cheb_tmax: float,
                  clip_domain: bool = False) -> np.ndarray:
    """Return B with shape (..., degree+1) such that p = B @ alpha.

    clip_domain=False matches the shipped model, which does NOT clamp x to
    [-1,1]. Outside the domain the Chebyshev recurrence grows like (2x)^n, so
    a pair separated by more than 7*(exp(cheb_tmax)-1) days produces a very
    large |p|. That is a live numerical concern, not a formality -- see the
    out_of_domain_frac field in the output.
    """
    if basis == "chebyshev":
        x = 2.0 * (tau / cheb_tmax) - 1.0
        if clip_domain:
            x = np.clip(x, -1.0, 1.0)
        cols = [np.ones_like(x), x]
        for _ in range(degree - 1):
            cols.append(2.0 * x * cols[-1] - cols[-2])
        return np.stack(cols[: degree + 1], axis=-1)
    if basis == "monomial":
        cols = [np.ones_like(tau)]
        cur = tau.copy()
        for _ in range(degree):
            cols.append(cur.copy())
            cur = cur * tau
        return np.stack(cols[: degree + 1], axis=-1)
    raise ValueError(f"unknown basis {basis!r}")


def apply_operator(p: np.ndarray, operator: str) -> np.ndarray:
    """Map raw polynomial value p to the additive log-space bias g."""
    if operator == "logsigmoid":
        # log sigmoid(p) = -softplus(-p); bounded above by 0, saturates near 0
        return -np.logaddexp(0.0, -p)
    if operator == "neg_softplus":
        # g = -softplus(p); unbounded below, no upper saturation trap
        return -np.logaddexp(0.0, p)
    if operator == "identity":
        return p
    raise ValueError(f"unknown operator {operator!r}")


# --------------------------------------------------------------------------
# Data loading -- reads the tensorized shards directly, no torch, no Dataset
# --------------------------------------------------------------------------
def iter_patients(split_dir: Path, max_seq_len: int, rng: np.random.Generator, max_patients: int):
    shard_paths = sorted(split_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz under {split_dir}")

    yielded = 0
    for sp in shard_paths:
        npz = np.load(sp, mmap_mode="r", allow_pickle=False)
        # Two tensorizers ship different names for the same per-patient index
        # into the flat event arrays: PIC finetune shards use "offsets", the
        # MIMIC pretrain shards use "event_offsets". Same dtype, same n+1
        # length, same semantics.
        off_key = next((k for k in ("offsets", "event_offsets") if k in npz.files), None)
        if off_key is None:
            raise RuntimeError(f"{sp} is legacy object-array format; re-tensorize.")
        off = npz[off_key]
        n = int(len(npz["subject_id"]))
        order = rng.permutation(n)
        for pos in order:
            if yielded >= max_patients:
                npz.close()
                return
            start, end = int(off[pos]), int(off[pos + 1])
            t = np.asarray(npz["timestamps_days"][start:end], dtype=np.float64)
            a = np.asarray(npz["age_days"][start:end], dtype=np.float64)
            if t.shape[0] > max_seq_len:
                t = t[-max_seq_len:]
                a = a[-max_seq_len:]
            if t.shape[0] < 2:
                continue
            yield {
                "subject_id": int(npz["subject_id"][pos]),
                "t": t,
                "age_years_last": float(a[-1] / 365.25),
                "L": int(t.shape[0]),
            }
            yielded += 1
        npz.close()


# --------------------------------------------------------------------------
# Row extraction
# --------------------------------------------------------------------------
def sample_rows(t: np.ndarray, masking: str, n_rows: int, max_keys: int,
                rng: np.random.Generator) -> list[np.ndarray]:
    """Return a list of 1-D tau_tilde arrays, one per sampled attention row."""
    L = t.shape[0]
    if masking == "causal":
        # row i sees keys 0..i; rows with <2 keys carry no spread by definition
        candidates = np.arange(1, L)
    elif masking == "bidirectional":
        candidates = np.arange(0, L)
    else:
        raise ValueError(masking)
    if candidates.size == 0:
        return []
    if candidates.size > n_rows:
        candidates = rng.choice(candidates, size=n_rows, replace=False)

    rows = []
    for i in candidates:
        keys = np.arange(0, i + 1) if masking == "causal" else np.arange(0, L)
        if keys.size > max_keys:
            keys = rng.choice(keys, size=max_keys, replace=False)
        tau = np.log1p(np.abs(t[i] - t[keys]) / 7.0)
        rows.append(tau)
    return rows


# --------------------------------------------------------------------------
# Candidate kernel shapes
# --------------------------------------------------------------------------
def canonical_shapes(degree: int, basis: str, radius: float) -> dict[str, np.ndarray]:
    """Hand-set extreme coefficient offsets, all with L2 norm == radius."""
    d = degree + 1
    shapes: dict[str, np.ndarray] = {}

    def norm_to(v):
        v = np.asarray(v, dtype=np.float64)[:d]
        v = np.pad(v, (0, max(0, d - v.size)))
        nrm = np.linalg.norm(v)
        return v * (radius / nrm) if nrm > 0 else v

    # linear term dominates -> monotone decay in tau (recency preference)
    shapes["recency"] = norm_to([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    # linear term flipped -> monotone increase in tau (anti-recency)
    shapes["anti_recency"] = norm_to([0.0, +1.0, 0.0, 0.0, 0.0, 0.0])
    # curvature only -> U / inverted-U, no net slope
    shapes["curvature"] = norm_to([0.0, 0.0, +1.0, 0.0, 0.0, 0.0])
    shapes["curvature_neg"] = norm_to([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    # high-order wiggle -> the shape flexibility degree 5 is supposed to buy
    shapes["high_order"] = norm_to([0.0, 0.0, 0.0, 0.0, 0.0, +1.0])
    # pure level shift -> the null case; softmax must cancel this exactly
    shapes["level_only"] = norm_to([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return shapes


def random_offsets(degree: int, radius: float, n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, degree + 1))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


# --------------------------------------------------------------------------
# Core measurement
# --------------------------------------------------------------------------
def row_headroom(tau_row: np.ndarray, base: np.ndarray, da_A: np.ndarray, da_B: np.ndarray,
                 degree: int, basis: str, cheb_tmax: float, operator: str,
                 clip_domain: bool = False) -> tuple[float, float]:
    """Return (uncentered max|delta|, row-centered max|delta|) for one row."""
    B = design_matrix(tau_row, degree, basis, cheb_tmax, clip_domain)   # (K, d)
    gA = apply_operator(B @ (base + da_A), operator)              # (K,)
    gB = apply_operator(B @ (base + da_B), operator)
    delta = gA - gB
    return float(np.max(np.abs(delta))), float(np.max(np.abs(delta - delta.mean())))


def load_learned(args, degree: int):
    """Optional: real Delta-alpha(age) curve produced by e2_alpha_radius.py."""
    if not args.offsets_npz:
        return None
    z = np.load(args.offsets_npz)
    da = z["delta_alpha"].astype(np.float64)
    if da.shape[1] != degree + 1:
        raise SystemExit(f"offsets npz has degree {da.shape[1]-1}, --degree says {degree}")
    return {"ages": z["ages"].astype(np.float64), "da": da,
            "base": z["base"].astype(np.float64)}


def analyse(args) -> dict:
    rng = np.random.default_rng(args.seed)
    degree = args.degree
    base = np.array(args.base_coeffs, dtype=np.float64)
    if base.size != degree + 1:
        raise SystemExit(f"--base_coeffs needs {degree + 1} values, got {base.size}")

    learned = load_learned(args, degree)
    if learned is not None:
        base = learned["base"]
        print(f"[info] using base coefficients and learned offsets from {args.offsets_npz}")

    shapes = canonical_shapes(degree, args.basis, args.radius)
    rand = random_offsets(degree, args.radius, args.n_random, rng)

    records: list[dict] = []
    n_pat = 0
    for pat in iter_patients(Path(args.split_dir), args.max_seq_len, rng, args.max_patients):
        n_pat += 1
        rows = sample_rows(pat["t"], args.masking, args.rows_per_patient, args.max_keys, rng)
        band = age_band(pat["age_years_last"])
        for tau_row in rows:
            if tau_row.size < 2:
                continue
            spread_std = float(np.std(tau_row))
            spread_iqr = float(np.subtract(*np.percentile(tau_row, [75, 25])))
            spread_ptp = float(np.ptp(tau_row))

            # canonical extreme pair: recency vs anti-recency
            unc, cen = row_headroom(tau_row, base, shapes["recency"], shapes["anti_recency"],
                                    degree, args.basis, args.cheb_tmax, args.operator,
                                    args.clip_domain)
            # Level-only offset vs zero. Under --operator identity this MUST be
            # ~1e-16 (softmax shift-invariance). Under the nonlinear operators a
            # small nonzero value is expected and is itself informative: it is
            # the fraction of a pure level shift that the nonlinearity converts
            # into shape. If it is comparable to the recency/anti-recency number,
            # most of your "kernel effect" is nonlinearity, not shape.
            _, cen_level = row_headroom(tau_row, base, shapes["level_only"], np.zeros(degree + 1),
                                        degree, args.basis, args.cheb_tmax, args.operator,
                                    args.clip_domain)
            # random search for the true achievable max at this radius
            best_cen = cen
            Bm = design_matrix(tau_row, degree, args.basis, args.cheb_tmax, args.clip_domain)
            g0 = apply_operator(Bm @ base, args.operator)
            for da in rand:
                g = apply_operator(Bm @ (base + da), args.operator)
                dl = g - g0
                c = float(np.max(np.abs(dl - dl.mean())))
                if c > best_cen:
                    best_cen = c

            # Headroom of the offset the model ACTUALLY produces at this
            # patient's age, relative to the unconditioned kernel.
            cen_learned = float("nan")
            if learned is not None:
                j = int(np.argmin(np.abs(learned["ages"] - pat["age_years_last"])))
                _, cen_learned = row_headroom(
                    tau_row, base, learned["da"][j], np.zeros(degree + 1),
                    degree, args.basis, args.cheb_tmax, args.operator, args.clip_domain)

            records.append({
                "band": band,
                "age_years": pat["age_years_last"],
                "dlogit_learned_centered": cen_learned,
                "L": pat["L"],
                "n_keys": int(tau_row.size),
                "tau_mean": float(np.mean(tau_row)),
                "tau_max": float(np.max(tau_row)),
                "oo_domain_frac": float(np.mean(np.abs(2.0 * (tau_row / args.cheb_tmax) - 1.0) > 1.0)),
                "spread_std": spread_std,
                "spread_iqr": spread_iqr,
                "spread_ptp": spread_ptp,
                "dlogit_uncentered": unc,
                "dlogit_centered": cen,
                "dlogit_centered_max": best_cen,
                "dlogit_level_only_centered": cen_level,
            })

    if not records:
        raise SystemExit("No rows collected -- check --split_dir and --masking.")

    arr = {k: np.array([r[k] for r in records], dtype=np.float64)
           for k in records[0] if k != "band"}
    bands = np.array([r["band"] for r in records])

    def summ(x: np.ndarray) -> dict:
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "p05": float(np.percentile(x, 5)),
            "p25": float(np.percentile(x, 25)),
            "p50": float(np.percentile(x, 50)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)),
        }

    # Chebyshev domain occupancy: where does tau actually land in [-1, 1]?
    tau_all = np.concatenate([np.array([r["tau_mean"]]) for r in records])
    x_all = np.clip(2.0 * (arr["tau_mean"] / args.cheb_tmax) - 1.0, -1.0, 1.0)

    out = {
        "label": args.label,
        "out_of_domain": {
            "frac_rows_touching": float(np.mean(arr["oo_domain_frac"] > 0)),
            "frac_pairs_outside": float(np.mean(arr["oo_domain_frac"])),
            "tau_max_p99": float(np.percentile(arr["tau_max"], 99)),
            "cheb_tmax": args.cheb_tmax,
        },
        "split_dir": args.split_dir,
        "config": {
            "basis": args.basis, "operator": args.operator, "degree": degree,
            "cheb_tmax": args.cheb_tmax, "radius": args.radius,
            "masking": args.masking, "base_coeffs": list(map(float, base)),
            "n_random": args.n_random, "seed": args.seed,
        },
        "n_patients": n_pat,
        "n_rows": len(records),
        "within_row_spread": {
            "std": summ(arr["spread_std"]),
            "iqr": summ(arr["spread_iqr"]),
            "range": summ(arr["spread_ptp"]),
        },
        "cheb_domain_occupancy_x": summ(x_all),
        "headroom": {
            "dlogit_uncentered": summ(arr["dlogit_uncentered"]),
            "dlogit_centered": summ(arr["dlogit_centered"]),
            "dlogit_centered_randmax": summ(arr["dlogit_centered_max"]),
            "level_only_centered_leak": summ(arr["dlogit_level_only_centered"]),
            "dlogit_learned_centered": (
                summ(arr["dlogit_learned_centered"])
                if np.isfinite(arr["dlogit_learned_centered"]).all() else None),
        },
        "by_band": {},
        "by_spread_quartile": {},
    }

    for b in sorted(set(bands)):
        m = bands == b
        if m.sum() < 10:
            continue
        out["by_band"][b] = {
            "n_rows": int(m.sum()),
            "spread_std_p50": float(np.percentile(arr["spread_std"][m], 50)),
            "dlogit_centered_p50": float(np.percentile(arr["dlogit_centered"][m], 50)),
            "dlogit_centered_randmax_p50": float(np.percentile(arr["dlogit_centered_max"][m], 50)),
        }

    # E3 sensitivity curve, model-free version: headroom vs within-row spread
    qs = np.percentile(arr["spread_std"], [0, 25, 50, 75, 100])
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = (arr["spread_std"] >= lo) & (arr["spread_std"] <= hi)
        if m.sum() < 10:
            continue
        out["by_spread_quartile"][f"Q{i + 1}"] = {
            "spread_std_range": [float(lo), float(hi)],
            "n_rows": int(m.sum()),
            "dlogit_centered_p50": float(np.percentile(arr["dlogit_centered"][m], 50)),
            "dlogit_centered_randmax_p50": float(np.percentile(arr["dlogit_centered_max"][m], 50)),
        }

    return out, arr, bands


# --------------------------------------------------------------------------
def make_figures(out: dict, arr: dict, out_dir: Path, label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping figures")
        return

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(arr["spread_std"], bins=60, color="#2b6cb0")
    ax.set_xlabel(r"within-row std of $\tilde\tau=\log(1+\Delta t/7)$")
    ax.set_ylabel("attention rows")
    ax.set_title(f"{label}: within-row $\\tau$ spread")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_spread_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(arr["spread_std"], arr["dlogit_centered_max"], s=3, alpha=0.25, color="#c53030")
    ax.set_xlabel("within-row std of $\\tilde\\tau$")
    ax.set_ylabel("row-centered max$|\\Delta$logit$|$")
    ax.set_title(f"{label}: kernel headroom vs. row spread")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_headroom_vs_spread.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(arr["dlogit_centered_max"], bins=60, color="#2f855a")
    ax.axvline(0.0059, ls="--", c="k", lw=1, label="prior 0.0059")
    ax.set_xlabel("row-centered max$|\\Delta$logit$|$")
    ax.set_ylabel("attention rows")
    ax.legend()
    ax.set_title(f"{label}: achievable logit change")
    fig.tight_layout()
    fig.savefig(out_dir / f"{label}_headroom_hist.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split_dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out_dir", default="results/e1")
    p.add_argument("--basis", choices=["chebyshev", "monomial"], default="chebyshev")
    p.add_argument("--operator", choices=["logsigmoid", "neg_softplus", "identity"],
                   default="neg_softplus")
    p.add_argument("--degree", type=int, default=5)
    p.add_argument("--cheb_tmax", type=float, default=4.0,
                   help="tau_tilde value mapped to x=+1. MUST match the model's constant.")
    p.add_argument("--base_coeffs", type=float, nargs="+", default=[0.5, 0, 0, 0, 0, 0])
    p.add_argument("--radius", type=float, default=1.0,
                   help="L2 norm of the coefficient offset. Set from observed ||Delta alpha||.")
    p.add_argument("--masking", choices=["causal", "bidirectional"], default="bidirectional")
    p.add_argument("--max_patients", type=int, default=400)
    p.add_argument("--rows_per_patient", type=int, default=32)
    p.add_argument("--max_keys", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--n_random", type=int, default=64)
    p.add_argument("--offsets_npz", default=None,
                   help="alpha_*.npz from e2_alpha_radius.py; uses the LEARNED offsets.")
    p.add_argument("--clip_domain", action="store_true",
                   help="Clamp x to [-1,1]. OFF by default to match the model.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out, arr, _ = analyse(args)
    (out_dir / f"{args.label}_headroom.json").write_text(json.dumps(out, indent=2))
    make_figures(out, arr, out_dir, args.label)

    s = out["within_row_spread"]["std"]
    h = out["headroom"]
    print(f"\n=== {args.label} ({out['n_patients']} patients, {out['n_rows']} rows) ===")
    print(f"within-row std of tau_tilde : p25={s['p25']:.4f} p50={s['p50']:.4f} p75={s['p75']:.4f}")
    print(f"max|dlogit| uncentered      : p50={h['dlogit_uncentered']['p50']:.4f} "
          f"p95={h['dlogit_uncentered']['p95']:.4f}")
    print(f"max|dlogit| ROW-CENTERED    : p50={h['dlogit_centered']['p50']:.4f} "
          f"p95={h['dlogit_centered']['p95']:.4f}")
    print(f"max|dlogit| centered (rand) : p50={h['dlogit_centered_randmax']['p50']:.4f} "
          f"p95={h['dlogit_centered_randmax']['p95']:.4f}")
    print(f"level-only leak (0 iff identity op): max={h['level_only_centered_leak']['max']:.2e}")
    od = out["out_of_domain"]
    print(f"outside Chebyshev domain    : {od['frac_pairs_outside']*100:.2f}% of pairs, "
          f"{od['frac_rows_touching']*100:.2f}% of rows touch it (tau_max p99={od['tau_max_p99']:.2f} "
          f"vs tmax={od['cheb_tmax']})")
    dom = out["cheb_domain_occupancy_x"]
    print(f"domain occupancy x          : p05={dom['p05']:.3f} p50={dom['p50']:.3f} p95={dom['p95']:.3f} "
          f"(full domain is [-1,+1])")
    if h.get("dlogit_learned_centered"):
        L = h["dlogit_learned_centered"]
        print(f"LEARNED offset, row-centered: p50={L['p50']:.4f} p95={L['p95']:.4f} max={L['max']:.4f}")
    print("\nby spread quartile (this is the E3 curve):")
    for k, v in out["by_spread_quartile"].items():
        print(f"  {k} spread[{v['spread_std_range'][0]:.3f},{v['spread_std_range'][1]:.3f}] "
              f"-> centered p50 {v['dlogit_centered_p50']:.4f}, "
              f"rand-max p50 {v['dlogit_centered_randmax_p50']:.4f}")
    print(f"\nwrote {out_dir}/{args.label}_headroom.json + 3 PNGs")


if __name__ == "__main__":
    main()
