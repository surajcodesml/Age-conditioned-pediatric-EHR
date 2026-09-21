"""v2 cleanup: sentinel/impossible events, PSG-encounter exclusion, pediatric cohort."""
from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.NCH.v2 import paths as P

SENTINEL_YEARS = {1899, 1900}
MAX_AGE_YEARS = 120.0
HISTORY_SLACK_DAYS = 365.25

OSA_ICD10_N = {"G4733"}
OSA_ICD9_N = {"32723"}
SNORING_ICD10_N = {"R0683"}
PSG_CPT = {"95810", "95811", "95782", "95783", "G0398", "G0399", "G0400"}


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper().str.replace(".", "", regex=False)


def clean_canonical_events(canonical: pd.DataFrame, sleep: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = canonical.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    years = df["event_time"].dt.year
    age = df["age_at_event_days"]

    reason = pd.Series(pd.NA, index=df.index, dtype="object")
    reason = reason.mask(df["event_time"].isna(), "missing_event_time")
    reason = reason.mask(reason.isna() & years.isin(SENTINEL_YEARS), "sentinel_year")
    reason = reason.mask(reason.isna() & (years < 1990), "pre1990_timestamp")
    reason = reason.mask(reason.isna() & age.notna() & (age < 0), "age_before_dob")
    reason = reason.mask(
        reason.isna() & age.notna() & (age > MAX_AGE_YEARS * 365.25), "age_gt_120y"
    )
    df["removal_reason"] = reason
    n_before = len(df)
    removed = df[df["removal_reason"].notna()].copy()
    kept = df[df["removal_reason"].isna()].copy()

    sleep = sleep.copy()
    sleep["index_time"] = pd.to_datetime(sleep["index_time"], errors="coerce")
    sleep_enc = sleep.dropna(subset=["study_enc_id"]).copy()
    sleep_enc["study_enc_id"] = sleep_enc["study_enc_id"].astype("int64")
    sleep_enc["patient_id"] = sleep_enc["patient_id"].astype("int64")
    enc_keys = set(zip(sleep_enc["patient_id"], sleep_enc["study_enc_id"]))

    enc_ok = kept["encounter_id"].notna()
    pair = pd.Series(False, index=kept.index)
    if enc_ok.any():
        pid_i = kept.loc[enc_ok, "patient_id"].astype("int64")
        enc_i = kept.loc[enc_ok, "encounter_id"].astype("int64")
        pair.loc[enc_ok] = [
            (int(p), int(e)) in enc_keys for p, e in zip(pid_i.to_numpy(), enc_i.to_numpy())
        ]
    kept["is_psg_encounter_event"] = pair.to_numpy()
    psg_enc_events = kept[kept["is_psg_encounter_event"]].copy()
    kept_no_psg = kept[~kept["is_psg_encounter_event"]].copy()

    raw_n = _norm(psg_enc_events["raw_code"]) if len(psg_enc_events) else pd.Series(dtype=str)
    tok = psg_enc_events["mimic_token"].fillna("").astype(str) if len(psg_enc_events) else pd.Series(dtype=str)
    et = psg_enc_events["event_type"] if len(psg_enc_events) else pd.Series(dtype=str)
    sleep_rel = pd.Series([None] * len(psg_enc_events), index=psg_enc_events.index, dtype=object)
    if len(psg_enc_events):
        sleep_rel = sleep_rel.mask(
            (et == "diagnosis") & raw_n.isin(OSA_ICD10_N | OSA_ICD9_N), "osa"
        )
        sleep_rel = sleep_rel.mask(
            sleep_rel.isna() & (et == "diagnosis") & raw_n.isin(SNORING_ICD10_N), "snoring"
        )
        sleep_rel = sleep_rel.mask(
            sleep_rel.isna() & (et == "diagnosis") & tok.str.startswith("PHE_327"), "osa_related_token"
        )
        sleep_rel = sleep_rel.mask(
            sleep_rel.isna() & (et == "procedure") & raw_n.isin(PSG_CPT), "psg_cpt"
        )
    psg_enc_events["sleep_related"] = sleep_rel

    report = {
        "rules": {
            "sentinel_years": sorted(SENTINEL_YEARS),
            "pre1990_timestamp": True,
            "age_before_dob": True,
            "age_gt_120y": True,
            "missing_event_time": True,
            "exclude_entire_psg_encounter": True,
            "note": "Invalid events removed; patients retained if remaining valid events exist.",
        },
        "n_events_before": int(n_before),
        "n_events_after_sentinel_filter": int(len(kept)),
        "n_events_after_psg_encounter_exclusion": int(len(kept_no_psg)),
        "removed_by_reason": removed["removal_reason"].value_counts().to_dict() if len(removed) else {},
        "psg_encounter_removed": {
            "n_events": int(len(psg_enc_events)),
            "by_event_type": psg_enc_events["event_type"].value_counts().to_dict() if len(psg_enc_events) else {},
            "sleep_related": psg_enc_events["sleep_related"].value_counts(dropna=False).to_dict()
            if len(psg_enc_events) else {},
            "osa": int((psg_enc_events["sleep_related"] == "osa").sum()) if len(psg_enc_events) else 0,
            "snoring": int((psg_enc_events["sleep_related"] == "snoring").sum()) if len(psg_enc_events) else 0,
            "psg_cpt": int((psg_enc_events["sleep_related"] == "psg_cpt").sum()) if len(psg_enc_events) else 0,
        },
        "patients_with_any_valid_event": int(kept_no_psg["patient_id"].nunique()),
    }

    kept_no_psg.to_parquet(P.DIRS["processed"] / "canonical_events_clean.parquet", index=False)
    removed.to_parquet(P.DIRS["cleanup"] / "removed_events.parquet", index=False)
    psg_enc_events.to_parquet(P.DIRS["cleanup"] / "psg_encounter_events.parquet", index=False)
    P.write_json(P.DIRS["cleanup"] / "cleanup_report.json", report)
    return kept_no_psg, report


def preindex_washout_report(events: pd.DataFrame, sleep: pd.DataFrame) -> dict:
    sleep = sleep.copy()
    sleep["index_time"] = pd.to_datetime(sleep["index_time"], errors="coerce")
    first = (
        sleep.sort_values(["patient_id", "index_time", "sleep_study_id"])
        .groupby("patient_id", as_index=False)
        .first()
    )
    ev = events.copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    raw_n = _norm(ev["raw_code"])
    tok = ev["mimic_token"].fillna("").astype(str)
    is_rel = (
        ((ev["event_type"] == "diagnosis") & raw_n.isin(OSA_ICD10_N | OSA_ICD9_N | SNORING_ICD10_N))
        | ((ev["event_type"] == "diagnosis") & tok.str.startswith("PHE_327"))
        | ((ev["event_type"] == "procedure") & raw_n.isin(PSG_CPT))
    )
    ev = ev.loc[is_rel, ["patient_id", "event_time", "event_type", "raw_code"]].copy()

    merged = ev.merge(first[["patient_id", "index_time"]], on="patient_id", how="inner")
    merged = merged[merged["event_time"] < merged["index_time"]]
    dt = (merged["index_time"] - merged["event_time"]).dt.total_seconds() / 86400.0
    out = {
        "window_24h": {
            "n_events": int((dt <= 1.0).sum()),
            "n_patients": int(merged.loc[dt <= 1.0, "patient_id"].nunique()),
        },
        "window_7d": {
            "n_events": int((dt <= 7.0).sum()),
            "n_patients": int(merged.loc[dt <= 7.0, "patient_id"].nunique()),
        },
        "note": "Analysis only; these windows are NOT removed in v2 primary sequences.",
    }
    P.write_json(P.DIRS["cleanup"] / "preindex_washout_report.json", out)
    return out


def build_pediatric_cohort(sleep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sleep = sleep.copy()
    sleep["index_time"] = pd.to_datetime(sleep["index_time"], errors="coerce")
    sleep["index_age_years"] = sleep["index_age_days"] / P.DAYS_PER_YEAR
    ped = sleep[sleep["index_age_years"] < 18].copy()
    adult = sleep[sleep["index_age_years"] >= 18].copy()
    ped_first = (
        ped.sort_values(["patient_id", "index_time", "sleep_study_id"])
        .groupby("patient_id", as_index=False)
        .first()
    )
    adult_first = (
        adult.sort_values(["patient_id", "index_time", "sleep_study_id"])
        .groupby("patient_id", as_index=False)
        .first()
    )
    report = {
        "primary_rule": "index_age_years < 18",
        "pediatric_all_studies": {
            "n_studies": int(len(ped)),
            "n_patients": int(ped["patient_id"].nunique()),
        },
        "pediatric_first_study": {
            "n_studies": int(len(ped_first)),
            "n_patients": int(len(ped_first)),
        },
        "adult_all_studies": {
            "n_studies": int(len(adult)),
            "n_patients": int(adult["patient_id"].nunique()),
        },
        "adult_first_study": {
            "n_studies": int(len(adult_first)),
            "n_patients": int(len(adult_first)),
        },
    }
    ped.to_parquet(P.DIRS["processed"] / "sleep_studies_pediatric_all.parquet", index=False)
    ped_first.to_parquet(P.DIRS["processed"] / "sleep_studies_pediatric_first.parquet", index=False)
    adult.to_parquet(P.DIRS["processed"] / "sleep_studies_adult_all.parquet", index=False)
    adult_first.to_parquet(P.DIRS["processed"] / "sleep_studies_adult_first.parquet", index=False)
    P.write_json(P.DIRS["cleanup"] / "cohort_counts.json", report)
    return ped_first, ped, report
