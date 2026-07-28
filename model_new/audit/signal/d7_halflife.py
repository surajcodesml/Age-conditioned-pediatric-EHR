"""D7 — Does age-dependent memory exist in MIMIC? (CPU, DuckDB)

Model-free self-recurrence half-lives by age band for the top-N codes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model_new import diagnostics as D
from model_new.audit.common import REPO_ROOT
from model_new.audit.signal import D7_AGE_BANDS, D7_LAG_BINS
from model_new.audit.signal.common import (
    add_common_args,
    ensure_batches,
    open_duckdb,
    write_json_atomic,
)


def _lag_midpoints() -> np.ndarray:
    mids = []
    for _, lo, hi in D7_LAG_BINS:
        if np.isinf(hi):
            mids.append(lo + 365.25 * 2)  # ~5y representative for >3y
        else:
            mids.append(0.5 * (lo + hi))
    return np.asarray(mids, dtype=np.float64)


def _fit_half_life(p: np.ndarray, tau: np.ndarray) -> float:
    """Fit lift(τ) ≈ A·exp(−τ/h) in log space; return h (days). NaN if undefined."""
    p = np.asarray(p, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    ok = np.isfinite(p) & (p > 0) & np.isfinite(tau) & (tau > 0)
    if ok.sum() < 2:
        return float("nan")
    y = np.log(p[ok])
    x = tau[ok]
    # y = log A - x/h  →  slope = -1/h
    X = np.column_stack([np.ones(x.size), x])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    slope = float(coef[1])
    if slope >= 0:
        return float("nan")  # growing with lag — not a decay half-life
    return float(-np.log(2.0) / slope)


def _band_label_sql() -> str:
    parts = []
    for name, lo, hi in D7_AGE_BANDS:
        if np.isinf(hi):
            parts.append(f"WHEN age_years >= {lo} THEN '{name}'")
        else:
            parts.append(f"WHEN age_years >= {lo} AND age_years < {hi} THEN '{name}'")
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def _lag_bin_sql(alias_dt: str = "dt") -> str:
    parts = []
    for name, lo, hi in D7_LAG_BINS:
        if np.isinf(hi):
            parts.append(f"WHEN {alias_dt} >= {lo} THEN '{name}'")
        else:
            parts.append(
                f"WHEN {alias_dt} >= {lo} AND {alias_dt} < {hi} THEN '{name}'"
            )
    return "CASE " + " ".join(parts) + " ELSE NULL END"


def run_d7(ctx: dict, store: dict, *, events_parquet: Path) -> dict:
    flags = ctx["flags"]
    top_n = int(flags["top_codes"])
    n_boot = int(flags["n_boot"])
    n_perm = int(flags["n_perm"])
    seed = int(ctx.get("seed", 0))
    patient_frac = float(flags["patient_frac"])

    con = open_duckdb()
    path = str(events_parquet).replace("'", "''")
    # Sample subjects for smoke / frac.
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_events AS
        SELECT subject_id,
               code_id,
               timestamp_days,
               age_at_event_days / 365.25 AS age_years
        FROM read_parquet('{path}')
        WHERE code_id IS NOT NULL
    """)
    if patient_frac < 1.0:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE keep_subj AS
            SELECT subject_id FROM (
                SELECT DISTINCT subject_id FROM raw_events
            ) t USING SAMPLE {max(1, int(100 * patient_frac))}%
        """)
        con.execute("""
            CREATE OR REPLACE TEMP TABLE events AS
            SELECT e.* FROM raw_events e
            INNER JOIN keep_subj k USING (subject_id)
        """)
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE events AS SELECT * FROM raw_events")

    # Top-N codes by event count.
    top = con.execute(f"""
        SELECT code_id, COUNT(*) AS n
        FROM events GROUP BY 1 ORDER BY n DESC LIMIT {top_n}
    """).fetchall()
    top_codes = [r[0] for r in top]
    con.execute("CREATE OR REPLACE TEMP TABLE top_codes (code_id VARCHAR)")
    for c in top_codes:
        con.execute("INSERT INTO top_codes VALUES (?)", [c])

    band_sql = _band_label_sql()
    lag_sql = _lag_bin_sql("dt")

    # Self-recurrence: for each occurrence of c, look forward for next c.
    # Restrict to top codes; explode pairs within subject ordered by time.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE code_events AS
        SELECT e.subject_id, e.code_id, e.timestamp_days, e.age_years,
               {band_sql} AS band
        FROM events e
        INNER JOIN top_codes t USING (code_id)
        WHERE ({band_sql}) IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE recurrence AS
        SELECT a.code_id, a.band, a.subject_id,
               (b.timestamp_days - a.timestamp_days) AS dt
        FROM code_events a
        INNER JOIN code_events b
          ON a.subject_id = b.subject_id
         AND a.code_id = b.code_id
         AND b.timestamp_days > a.timestamp_days
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY a.subject_id, a.code_id, a.timestamp_days
            ORDER BY b.timestamp_days
        ) = 1
    """)
    # Denominator: occurrences of c in band (with any future opportunity — all of them).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE denom AS
        SELECT code_id, band, COUNT(*) AS n_obs
        FROM code_events GROUP BY 1, 2
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE by_bin AS
        SELECT code_id, band, ({lag_sql}) AS lag_bin, COUNT(*) AS n_rec
        FROM recurrence
        WHERE dt > 0 AND ({lag_sql}) IS NOT NULL
        GROUP BY 1, 2, 3
    """)

    bin_names = [n for n, _, _ in D7_LAG_BINS]
    band_names = [n for n, _, _ in D7_AGE_BANDS]
    mid = _lag_midpoints()
    mid_map = {n: float(m) for n, m in zip(bin_names, mid)}

    rows = con.execute("""
        SELECT d.code_id, d.band, d.n_obs, b.lag_bin, COALESCE(b.n_rec, 0) AS n_rec
        FROM denom d
        LEFT JOIN by_bin b USING (code_id, band)
    """).fetchall()

    # P(recurs in bin | observed) ≈ n_rec_bin / n_obs  (competing bins; not exclusive).
    # Build per (code, band) vector over bins.
    from collections import defaultdict
    store_p: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    n_obs_map: dict[tuple[str, str], int] = {}
    for code, band, n_obs, lag_bin, n_rec in rows:
        key = (str(code), str(band))
        n_obs_map[key] = int(n_obs)
        if lag_bin is not None:
            store_p[key][str(lag_bin)] = float(n_rec) / float(max(1, n_obs))

    half_lives: dict[str, list[float]] = {b: [] for b in band_names}
    per_code = []
    for (code, band), probs in store_p.items():
        pvec = np.array([probs.get(bn, 0.0) for bn in bin_names], dtype=np.float64)
        # Use relative lift vs first bin if available, else raw p.
        if pvec[0] > 0:
            lift = pvec / pvec[0]
        else:
            lift = pvec
        h = _fit_half_life(lift, mid)
        if band in half_lives and np.isfinite(h):
            half_lives[band].append(h)
        per_code.append({
            "code_id": code, "band": band, "half_life_days": h,
            "p_by_bin": {bn: float(probs.get(bn, 0.0)) for bn in bin_names},
            "n_obs": n_obs_map.get((code, band), 0),
        })

    def _summarize(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"n": 0, "median": float("nan"), "mean": float("nan"),
                    "ci_lo": float("nan"), "ci_hi": float("nan")}
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            samp = rng.choice(a, size=a.size, replace=True)
            boots.append(float(np.median(samp)))
        boots_a = np.asarray(boots)
        return {
            "n": int(a.size),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "ci_lo": float(np.percentile(boots_a, 2.5)),
            "ci_hi": float(np.percentile(boots_a, 97.5)),
        }

    by_band = {b: _summarize(half_lives[b]) for b in band_names}

    # Monotonicity: median h should increase with age if memory lengthens (or decrease).
    # Test: Spearman-like — correlation of band index with median h; permute band labels.
    medians = np.array([by_band[b]["median"] for b in band_names], dtype=np.float64)
    x = np.arange(len(band_names), dtype=np.float64)
    obs_slope = float(np.polyfit(x[np.isfinite(medians)], medians[np.isfinite(medians)], 1)[0]) \
        if np.isfinite(medians).sum() >= 2 else float("nan")

    # Flatten code-level h with band labels for permutation.
    code_h = []
    code_band_idx = []
    band_index = {b: i for i, b in enumerate(band_names)}
    for rec in per_code:
        if np.isfinite(rec["half_life_days"]) and rec["band"] in band_index:
            code_h.append(rec["half_life_days"])
            code_band_idx.append(band_index[rec["band"]])
    code_h = np.asarray(code_h, dtype=np.float64)
    code_band_idx = np.asarray(code_band_idx, dtype=np.int64)

    def _slope_from_labels(labels: np.ndarray) -> float:
        meds = []
        xs = []
        for i in range(len(band_names)):
            sel = labels == i
            if sel.sum() == 0:
                continue
            meds.append(float(np.median(code_h[sel])))
            xs.append(float(i))
        if len(meds) < 2:
            return float("nan")
        return float(np.polyfit(xs, meds, 1)[0])

    rng = np.random.default_rng(seed + 7)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(code_band_idx)
        null.append(_slope_from_labels(perm))
    null_a = np.asarray(null, dtype=np.float64)
    null_a = null_a[np.isfinite(null_a)]
    if null_a.size and np.isfinite(obs_slope):
        # Two-sided.
        p_perm = float(np.mean(np.abs(null_a) >= abs(obs_slope)))
    else:
        p_perm = float("nan")

    # CI overlap across bands?
    cis = [(by_band[b]["ci_lo"], by_band[b]["ci_hi"]) for b in band_names]
    # Conservative: if every pairwise CI overlaps, say overlap.
    overlap = True
    for i in range(len(cis)):
        for j in range(i + 1, len(cis)):
            lo_i, hi_i = cis[i]
            lo_j, hi_j = cis[j]
            if not (np.isfinite(lo_i) and np.isfinite(hi_i)
                    and np.isfinite(lo_j) and np.isfinite(hi_j)):
                continue
            if hi_i < lo_j or hi_j < lo_i:
                overlap = False

    out = {
        "batch_list_hash": store["batch_list_hash"],
        "seed": seed,
        "smoke": bool(flags["smoke"]),
        "top_codes": top_n,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "events_parquet": str(events_parquet),
        "patient_frac": patient_frac,
        "lag_bins": [{"name": n, "lo": lo, "hi": (None if np.isinf(hi) else hi),
                      "mid": mid_map[n]} for n, lo, hi in D7_LAG_BINS],
        "age_bands": [{"name": n, "lo": lo, "hi": (None if np.isinf(hi) else hi)}
                      for n, lo, hi in D7_AGE_BANDS],
        "by_band": by_band,
        "median_half_life_days": {b: by_band[b]["median"] for b in band_names},
        "monotonicity": {
            "obs_slope_median_h_vs_band_index": obs_slope,
            "permutation_p": p_perm,
            "n_perm": n_perm,
        },
        "ci_overlap_all_bands": bool(overlap),
        "verdict": {
            "route": (
                "no age×lag in MIMIC → reframe, don't rebuild"
                if overlap else
                "signal exists, model failing → multi-head + TTE"
            ),
        },
        "n_code_band_fits": len(per_code),
        # Keep a thin sample of per-code fits for debugging (not all in smoke JSON bloat).
        "per_code_sample": per_code[: min(50, len(per_code))],
    }
    con.close()
    return out


def main(argv: list[str] | None = None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument(
        "--events",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "train_events.parquet",
    )
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    D.print_block("D7 age-dependent memory", [
        f"out={out_dir}  smoke={args.smoke}  events={args.events}",
    ])
    ctx, store = ensure_batches(
        out_dir, smoke=args.smoke, batch_size=args.batch_size,
        force=args.force, run_root=args.run_root,
    )
    ctx["seed"] = args.seed
    result = run_d7(ctx, store, events_parquet=args.events)
    write_json_atomic(out_dir / "d7_halflife.json", result)

    # Figure: median half-life vs age band.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bands = [n for n, _, _ in D7_AGE_BANDS]
        med = [result["by_band"][b]["median"] for b in bands]
        lo = [result["by_band"][b]["ci_lo"] for b in bands]
        hi = [result["by_band"][b]["ci_hi"] for b in bands]
        x = np.arange(len(bands))
        yerr = np.array([
            [m - l if np.isfinite(m) and np.isfinite(l) else 0 for m, l in zip(med, lo)],
            [h - m if np.isfinite(m) and np.isfinite(h) else 0 for m, h in zip(med, hi)],
        ])
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.errorbar(x, med, yerr=yerr, fmt="o-", capsize=4, color="#1f4e79")
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.set_ylabel("Median predictive half-life (days)")
        ax.set_xlabel("Age band")
        ax.set_title("D7: self-recurrence half-life vs age")
        fig.tight_layout()
        fig.savefig(fig_dir / "d7_halflife_vs_age.png", dpi=140)
        plt.close(fig)
        result["figure"] = str(fig_dir / "d7_halflife_vs_age.png")
        write_json_atomic(out_dir / "d7_halflife.json", result)
    except Exception as e:
        D.print_block("D7 figure", [f"skipped: {e}"])

    lines = [
        f"batch_list_hash={result['batch_list_hash']}",
        f"ci_overlap_all_bands={result['ci_overlap_all_bands']}",
        f"slope={result['monotonicity']['obs_slope_median_h_vs_band_index']:.4g}  "
        f"perm_p={result['monotonicity']['permutation_p']:.4g}",
        f"verdict: {result['verdict']['route']}",
    ]
    for b, s in result["by_band"].items():
        lines.append(
            f"{b}: median={s['median']:.4g} CI[{s['ci_lo']:.4g},{s['ci_hi']:.4g}] n={s['n']}"
        )
    D.print_block("D7 results", lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
