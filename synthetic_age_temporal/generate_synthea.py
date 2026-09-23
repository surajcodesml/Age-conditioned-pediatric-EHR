#!/usr/bin/env python3
"""Locate or (optionally) regenerate the pediatric Synthea base cohort.

By default this reuses the existing sep1-exp full cohort (~10k patients) and
writes a provenance manifest. It does **not** regenerate Synthea unless
``--force-regenerate`` is passed (requires a local Synthea build).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    DEFAULT_SYNTHEA_ENGINE,
    DEFAULT_SYNTHEA_PROCESSED,
    DEFAULT_OUTPUT_DIR,
    SYNTHEA_COMMIT,
    SYNTHEA_GEOGRAPHY,
    SYNTHEA_REFERENCE_DATE,
    SYNTHEA_STRATUM_SEEDS,
    SYNTHEA_VERSION,
    DAYS_PER_YEAR,
)


def load_background_tables(
    synthea_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patients + background events (exclude prior synthetic signals)."""
    synthea_dir = Path(synthea_dir or DEFAULT_SYNTHEA_PROCESSED)
    patients = pd.read_parquet(synthea_dir / "patients.parquet")
    events = pd.read_parquet(synthea_dir / "events.parquet")

    # Drop labels / signals from the prior sep1-exp benchmark.
    keep_cols = [
        c
        for c in patients.columns
        if c
        in {
            "patient_id",
            "date_of_birth",
            "index_date",
            "age_at_index",
            "developmental_age_group",
            "generation_stratum",
        }
    ]
    patients = patients[keep_cols].copy()
    patients["date_of_birth"] = pd.to_datetime(patients["date_of_birth"])
    patients["index_date"] = pd.to_datetime(patients["index_date"])
    patients["cutoff_time"] = patients["index_date"]
    patients["age_at_cutoff"] = patients["age_at_index"].astype(np.float64)

    # Background clinical history only.
    if "source" in events.columns:
        events = events[events["source"] == "synthea"].copy()
    else:
        events = events[~events["event_type"].astype(str).str.contains("signal")].copy()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"])
    return patients.reset_index(drop=True), events.reset_index(drop=True)


def cohort_stats(patients: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    ages = patients["age_at_cutoff"].to_numpy(dtype=np.float64)
    type_counts = events["event_type"].value_counts().to_dict()
    return {
        "n_patients": int(len(patients)),
        "n_events": int(len(events)),
        "age_distribution": {
            "min": float(ages.min()),
            "max": float(ages.max()),
            "mean": float(ages.mean()),
            "std": float(ages.std()),
            "p25": float(np.percentile(ages, 25)),
            "p50": float(np.percentile(ages, 50)),
            "p75": float(np.percentile(ages, 75)),
            "hist_edges_years": list(range(0, 19)),
            "hist_counts": np.histogram(ages, bins=range(0, 20))[0].tolist(),
        },
        "event_counts": {str(k): int(v) for k, v in type_counts.items()},
        "events_per_patient": {
            "mean": float(events.groupby("patient_id").size().mean()),
            "median": float(events.groupby("patient_id").size().median()),
        },
    }


def write_manifest(
    out_dir: Path,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    *,
    reused: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = cohort_stats(patients, events)
    manifest = {
        "synthea_version": SYNTHEA_VERSION,
        "synthea_commit": SYNTHEA_COMMIT,
        "generation_seed": SYNTHEA_STRATUM_SEEDS,
        "geography": SYNTHEA_GEOGRAPHY,
        "reference_date": SYNTHEA_REFERENCE_DATE,
        "age_range": "0-18",
        "reused_existing_cohort": reused,
        "source_path": str(DEFAULT_SYNTHEA_PROCESSED),
        "n_patients": stats["n_patients"],
        "age_distribution": stats["age_distribution"],
        "event_counts": stats["event_counts"],
        "events_per_patient": stats["events_per_patient"],
        "n_events": stats["n_events"],
        "notes": [
            "Background events only (source=synthea).",
            "Prior sep1-exp SIGNAL_A/B and S0/S1/S2 labels are discarded.",
            "Controlled age×temporal signals are injected later by build_benchmark.py.",
        ],
    }
    with (out_dir / "synthea_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    patients.to_parquet(out_dir / "patients_background.parquet", index=False)
    # Store a lightweight event index pointer rather than duplicating 400MB unless asked.
    with (out_dir / "events_source.json").open("w") as f:
        json.dump(
            {
                "parquet": str(DEFAULT_SYNTHEA_PROCESSED / "events.parquet"),
                "filter": "source == 'synthea'",
            },
            f,
            indent=2,
        )
    return manifest


def force_regenerate(engine_dir: Path, n_patients: int, seed: int, out_csv: Path) -> None:
    """Optional full regenerations via Synthea CLI (slow; not used by default)."""
    engine_dir = Path(engine_dir)
    jar_candidates = list(engine_dir.glob("**/synthea-with-dependencies.jar"))
    if not jar_candidates:
        raise FileNotFoundError(
            f"No Synthea jar under {engine_dir}; build Synthea first or reuse existing cohort."
        )
    jar = jar_candidates[0]
    out_csv.mkdir(parents=True, exist_ok=True)
    cmd = [
        "java",
        "-jar",
        str(jar),
        "-p",
        str(n_patients),
        "-s",
        str(seed),
        "-a",
        "0-18",
        "--exporter.csv.export=true",
        f"--exporter.baseDirectory={out_csv}",
        SYNTHEA_GEOGRAPHY,
    ]
    subprocess.check_call(cmd, cwd=str(engine_dir))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthea-dir", type=Path, default=DEFAULT_SYNTHEA_PROCESSED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "synthea")
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--n-patients", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--engine-dir", type=Path, default=DEFAULT_SYNTHEA_ENGINE)
    args = p.parse_args()

    if args.force_regenerate:
        force_regenerate(args.engine_dir, args.n_patients, args.seed, args.output_dir / "raw")
        print("Regeneration requested; convert CSVs with sep1-exp build before continuing.")
        return

    patients, events = load_background_tables(args.synthea_dir)
    manifest = write_manifest(args.output_dir, patients, events, reused=True)
    print(json.dumps({k: manifest[k] for k in ("n_patients", "n_events", "synthea_commit")}, indent=2))
    print("Wrote", args.output_dir / "synthea_manifest.json")


if __name__ == "__main__":
    main()
