#!/usr/bin/env python3
"""Build a unified event table from MIMIC-IV-ED v2.2 CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _make_events(df: pd.DataFrame, source: str, code_col: str, value_col: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    if time_col not in out.columns:
        out[time_col] = pd.NaT
    out["event_time"] = pd.to_datetime(out[time_col], errors="coerce")
    out["event_source"] = source
    out["event_code"] = out[code_col].astype(str)
    out["event_value"] = out[value_col].astype(str)
    keep_cols = ["subject_id", "stay_id", "event_time", "event_source", "event_code", "event_value"]
    return out[keep_cols]


def build_event_table(ed_dir: Path) -> pd.DataFrame:
    triage = pd.read_csv(ed_dir / "triage.csv.gz")
    vitals = pd.read_csv(ed_dir / "vitalsign.csv.gz")
    diagnosis = pd.read_csv(ed_dir / "diagnosis.csv.gz")
    medrecon = pd.read_csv(ed_dir / "medrecon.csv.gz")
    pyxis = pd.read_csv(ed_dir / "pyxis.csv.gz")
    edstays = pd.read_csv(ed_dir / "edstays.csv.gz")

    triage["charttime"] = pd.NaT
    triage_long = triage.melt(
        id_vars=["subject_id", "stay_id", "charttime"],
        value_vars=["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"],
        var_name="feature",
        value_name="value",
    )
    triage_events = _make_events(triage_long, "triage", "feature", "value", "charttime")
    vital_long = vitals.melt(
        id_vars=["subject_id", "stay_id", "charttime"],
        value_vars=["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "rhythm", "pain"],
        var_name="feature",
        value_name="value",
    )
    vital_events = _make_events(vital_long, "vitalsign", "feature", "value", "charttime")
    diagnosis["charttime"] = pd.NaT
    dx_events = _make_events(diagnosis, "diagnosis", "icd_code", "icd_title", "charttime")
    med_events = _make_events(medrecon, "medrecon", "name", "etcdescription", "charttime")
    pyxis_events = _make_events(pyxis, "pyxis", "name", "med_rn", "charttime")

    all_events = pd.concat([triage_events, vital_events, dx_events, med_events, pyxis_events], ignore_index=True)
    all_events = all_events.merge(edstays[["subject_id", "stay_id", "intime", "outtime", "hadm_id"]], on=["subject_id", "stay_id"], how="left")
    all_events["intime"] = pd.to_datetime(all_events["intime"], errors="coerce")
    all_events["outtime"] = pd.to_datetime(all_events["outtime"], errors="coerce")
    all_events["hours_from_intime"] = (all_events["event_time"] - all_events["intime"]).dt.total_seconds() / 3600.0
    all_events = all_events.sort_values(["subject_id", "stay_id", "event_time", "event_source"], na_position="last")
    return all_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified MIMIC-IV-ED event table.")
    parser.add_argument(
        "--ed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "mimic-iv-ed" / "physionet.org" / "files" / "mimic-iv-ed" / "2.2" / "ed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "ed_event_table.parquet",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = build_event_table(args.ed_dir)
    table.to_parquet(args.output, index=False)
    print(f"Wrote {len(table):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
