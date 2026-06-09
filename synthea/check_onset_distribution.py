#!/usr/bin/env python3
"""STEP 1 gate: confirm condition onset-age distributions are NOT flat across
developmental bands.

Reads Synthea ``patients.csv`` (BIRTHDATE) and ``conditions.csv`` (START, CODE),
computes onset age in years for each target condition, and prints the per-band
histogram (<1, 1-5, 6-11, 12-17, 18-25). Fails (exit 1) if any target condition
is effectively flat / concentrated in a single band, which would kill the
age-conditioned vs vanilla comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

# code -> human label, for reporting only.
TARGETS = {
    "414916001": "obesity",
    "44054006": "t2d",
    "78275009": "osa",
    "195967001": "asthma",
}

BANDS = [
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.0),
    ("18-25", 18.0, 26.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Synthea onset-age distributions.")
    p.add_argument(
        "--csv_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "csv",
        help="Directory containing Synthea patients.csv and conditions.csv.",
    )
    p.add_argument(
        "--min_band_frac",
        type=float,
        default=0.05,
        help="A condition is 'not flat' if >=2 bands each hold this fraction of onsets.",
    )
    p.add_argument("--min_onsets", type=int, default=10, help="Minimum onsets to assess a condition.")
    return p.parse_args()


def _esc(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


def main() -> int:
    args = parse_args()
    patients = args.csv_dir / "patients.csv"
    conditions = args.csv_dir / "conditions.csv"
    for f in (patients, conditions):
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}")

    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT
            c.CODE AS code,
            (epoch(CAST(c.START AS TIMESTAMP)) - epoch(CAST(p.BIRTHDATE AS TIMESTAMP)))
                / 86400.0 / 365.25 AS onset_age_years
        FROM read_csv_auto('{_esc(conditions)}') c
        JOIN read_csv_auto('{_esc(patients)}') p ON c.PATIENT = p.Id
        WHERE CAST(c.CODE AS VARCHAR) IN ({','.join(f"'{k}'" for k in TARGETS)})
        """
    ).df()
    con.close()

    print("=== Onset-age distribution by developmental band ===")
    all_ok = True
    for code, label in TARGETS.items():
        sub = df[df["code"].astype(str) == code]
        ages = sub["onset_age_years"].to_numpy()
        ages = ages[(ages >= 0.0) & (ages < 26.0)]
        n = int(ages.size)
        print(f"\n[{label}] code={code}  total_onsets={n}")
        if n == 0:
            print("  WARNING: no onsets recorded -- module may not have fired.")
            all_ok = False
            continue
        band_fracs = []
        for name, lo, hi in BANDS:
            c = int(((ages >= lo) & (ages < hi)).sum())
            frac = c / n
            band_fracs.append(frac)
            bar = "#" * int(round(frac * 40))
            print(f"  {name:>6}: {c:6d}  {frac:6.1%}  {bar}")
        if n < args.min_onsets:
            print(f"  NOTE: only {n} onsets (<{args.min_onsets}); distribution assessment is weak.")
            continue
        n_bands_populated = sum(1 for f in band_fracs if f >= args.min_band_frac)
        flat = n_bands_populated < 2
        verdict = "FLAT (FAIL)" if flat else f"OK ({n_bands_populated} bands populated)"
        print(f"  -> {verdict}")
        if flat:
            all_ok = False

    print("\n=== GATE:", "PASS" if all_ok else "FAIL", "===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
