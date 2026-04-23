#!/usr/bin/env python3
"""Build code-to-description mapping for unified ED events."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_descriptions(event_table_path: Path) -> pd.DataFrame:
    events = pd.read_parquet(event_table_path)

    static_feature_descriptions = {
        "temperature": "Body temperature",
        "heartrate": "Heart rate",
        "resprate": "Respiratory rate",
        "o2sat": "Oxygen saturation",
        "sbp": "Systolic blood pressure",
        "dbp": "Diastolic blood pressure",
        "pain": "Pain score",
        "acuity": "Triage acuity",
        "chiefcomplaint": "Chief complaint",
        "rhythm": "Cardiac rhythm",
    }

    features = pd.DataFrame(
        {
            "event_source": "triage_or_vitalsign",
            "event_code": list(static_feature_descriptions.keys()),
            "description": list(static_feature_descriptions.values()),
        }
    )

    diagnosis = (
        events.loc[events["event_source"] == "diagnosis", ["event_source", "event_code", "event_value"]]
        .rename(columns={"event_value": "description"})
        .dropna(subset=["event_code", "description"])
        .drop_duplicates(subset=["event_source", "event_code"], keep="first")
    )

    med_like = (
        events.loc[events["event_source"].isin(["medrecon", "pyxis"]), ["event_source", "event_code"]]
        .dropna()
        .drop_duplicates()
    )
    med_like["description"] = med_like["event_code"].astype(str)

    out = pd.concat([features, diagnosis, med_like], ignore_index=True).drop_duplicates()
    return out.sort_values(["event_source", "event_code"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ED code descriptions.")
    parser.add_argument(
        "--event-table",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "ed_event_table.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "ed_code_descriptions.csv",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = build_descriptions(args.event_table)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} code descriptions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
