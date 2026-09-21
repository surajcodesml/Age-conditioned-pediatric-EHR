"""Incident OSA labels + AHI extraction from available SleepBank files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.NCH.v2 import paths as P

# Reproducible OSA code set for NCH coding systems
OSA_CODES = {
    "ICD10CM": {"G4733"},
    "ICD9CM": {"32723"},
}
# Related (not used for primary incident label positivity alone)
OSA_RELATED = {
    "ICD10CM": {"G4730", "G4731", "G4739", "G4700", "G4710"},  # sleep apnea NOS etc.
    "ICD9CM": {"32720", "32721", "78057"},
}
SNORING = {"R0683", "78609"}

# Pediatric AHI severity (AASM / common pediatric clinical thresholds)
# Retain continuous AHI so thresholds can change without re-extraction.
PED_AHI_BINS = [
    ("normal", 0.0, 1.0),
    ("mild", 1.0, 5.0),
    ("moderate", 5.0, 10.0),
    ("severe", 10.0, float("inf")),
]


def _norm(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code or "").upper())


def osa_code_set() -> dict:
    return {
        "primary_positive_codes": OSA_CODES,
        "related_not_required_for_positive": OSA_RELATED,
        "snoring_codes": sorted(SNORING),
        "definition": {
            "osa_at_index": "OSA code on linked PSG/index encounter",
            "prior_osa": "OSA code in any pre-index event (excluding index encounter)",
            "incident_osa": "osa_at_index AND NOT prior_osa",
        },
        "limitation": (
            "Negatives reflect absence of coded OSA, not confirmed physiological absence. "
            "Coding incompleteness may undercall prior OSA and overcall incident cases."
        ),
    }


def build_incident_osa(
    clean_events: pd.DataFrame,
    indexes: pd.DataFrame,
    *,
    include_psg_encounter_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build incident OSA labels for each index study.

    Index-encounter diagnoses come from the saved PSG-encounter extraction (removed from
    model input) when provided; otherwise from raw DIAGNOSIS joined via study_enc_id.
    """
    osa_all = set()
    for v in OSA_CODES.values():
        osa_all |= v

    # Prior OSA from cleaned pre-index events (PSG enc already removed)
    dx = clean_events[clean_events["event_type"] == "diagnosis"].copy()
    dx["event_time"] = pd.to_datetime(dx["event_time"], errors="coerce")
    dx["code_n"] = dx["raw_code"].map(_norm)
    dx_osa = dx[dx["code_n"].isin(osa_all)]

    # Index-encounter OSA from psg encounter dump or DIAGNOSIS CSV
    if include_psg_encounter_events is not None and len(include_psg_encounter_events):
        psg = include_psg_encounter_events.copy()
        psg = psg[psg["event_type"] == "diagnosis"]
        psg["code_n"] = psg["raw_code"].map(_norm)
        psg_osa = psg[psg["code_n"].isin(osa_all)]
    else:
        # load diagnosis for study encounters
        diag = pd.read_csv(
            P.NCH_CSVS["diagnosis"],
            usecols=["STUDY_PAT_ID", "STUDY_ENC_ID", "DX_CODE", "DX_CODE_TYPE"],
            low_memory=False,
        )
        diag["code_n"] = diag["DX_CODE"].map(_norm)
        diag = diag[diag["code_n"].isin(osa_all)]
        psg_osa = diag.rename(columns={
            "STUDY_PAT_ID": "patient_id",
            "STUDY_ENC_ID": "encounter_id",
            "DX_CODE": "raw_code",
        })

    rows = []
    indexes = indexes.copy()
    indexes["index_time"] = pd.to_datetime(indexes["index_time"], errors="coerce")
    for r in indexes.itertuples(index=False):
        pid = int(r.patient_id)
        enc = int(r.study_enc_id) if pd.notna(r.study_enc_id) else None
        idx_t = r.index_time
        # osa at index
        if enc is not None and len(psg_osa):
            if "encounter_id" in psg_osa.columns:
                at = psg_osa[
                    (psg_osa["patient_id"].astype("int64") == pid)
                    & (psg_osa["encounter_id"].astype("float").fillna(-1).astype("int64") == enc)
                ]
            else:
                at = psg_osa[(psg_osa["patient_id"].astype("int64") == pid)]
            osa_at_index = int(len(at) > 0)
        else:
            osa_at_index = 0
        # prior
        prior = dx_osa[
            (dx_osa["patient_id"] == pid)
            & (dx_osa["event_time"] < idx_t)
        ]
        prior_osa = int(len(prior) > 0)
        incident = int(osa_at_index == 1 and prior_osa == 0)
        rows.append({
            "patient_id": pid,
            "sleep_study_id": int(r.sleep_study_id),
            "index_time": str(idx_t),
            "index_age_days": float(r.index_age_days) if pd.notna(r.index_age_days) else np.nan,
            "index_age_years": float(r.index_age_days) / P.DAYS_PER_YEAR if pd.notna(r.index_age_days) else np.nan,
            "osa_at_index": osa_at_index,
            "prior_osa": prior_osa,
            "incident_osa": incident,
            "label_valid": 1,
        })
    return pd.DataFrame(rows)


def _severity(ahi: float) -> str:
    if ahi != ahi:
        return "missing"
    for name, lo, hi in PED_AHI_BINS:
        if lo <= ahi < hi:
            return name
    return "severe"


def extract_ahi_from_local_tsvs() -> pd.DataFrame:
    """Derive AHI from locally available scored annotation TSVs.

    Only ~20 TSV files exist locally — not cohort-wide. Derivation:
      AHI = (obstructive+central+mixed apnea + hypopnea counts) / (total sleep hours)
    Sleep time estimated from sleep-stage annotation durations when present; else
    recording span of sleep-related annotations.
    """
    sd = P.NCH_SLEEP_DATA
    rows = []
    apnea_pat = re.compile(r"(obstructive|central|mixed)?\s*apnea|hypopnea", re.I)
    stage_pat = re.compile(r"\b(N1|N2|N3|REM|Wake|W|Stage)\b", re.I)
    sleep_stage_pat = re.compile(r"\b(N1|N2|N3|REM|Stage\s*[123]|S[123])\b", re.I)

    for tsv in sorted(sd.glob("*.tsv")):
        # filename: {patient}_{study}.tsv
        stem = tsv.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        try:
            pat_id, study_id = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        df = pd.read_csv(tsv, sep="\t")
        if "description" not in df.columns:
            rows.append({
                "patient_id": pat_id, "sleep_study_id": study_id,
                "ahi": np.nan, "ahi_valid": 0, "provenance": "tsv_missing_description",
                "n_apnea_hypopnea": 0, "sleep_hours": np.nan,
            })
            continue
        desc = df["description"].astype(str)
        is_ah = desc.str.contains(apnea_pat, na=False) & ~desc.str.contains("arousal", case=False, na=False)
        n_ah = int(is_ah.sum())
        # sleep time
        sleep_hours = np.nan
        if "duration" in df.columns:
            sleep_mask = desc.str.contains(sleep_stage_pat, na=False)
            if sleep_mask.any():
                sleep_hours = float(df.loc[sleep_mask, "duration"].sum()) / 3600.0
            elif desc.str.contains(stage_pat, na=False).any():
                # fallback: total annotated duration excluding technical markers
                tech = desc.str.contains(r"Montage|Recording|Analyzer|Video|Lights", case=False, na=False)
                sleep_hours = float(df.loc[~tech, "duration"].sum()) / 3600.0
        if not (sleep_hours == sleep_hours and sleep_hours > 0.05):
            # cannot compute reliable AHI
            rows.append({
                "patient_id": pat_id, "sleep_study_id": study_id,
                "ahi": np.nan, "ahi_valid": 0,
                "provenance": "tsv_insufficient_sleep_time",
                "n_apnea_hypopnea": n_ah, "sleep_hours": sleep_hours,
            })
            continue
        ahi = n_ah / sleep_hours
        rows.append({
            "patient_id": pat_id, "sleep_study_id": study_id,
            "ahi": float(ahi), "ahi_valid": 1,
            "provenance": "derived_from_tsv_annotations",
            "n_apnea_hypopnea": n_ah, "sleep_hours": float(sleep_hours),
            "severity_class": _severity(ahi),
        })

    out = pd.DataFrame(rows)
    summary = {
        "n_local_tsv_files": len(list(sd.glob("*.tsv"))),
        "n_local_edf_files": len(list(sd.glob("*.edf"))),
        "n_studies_with_valid_ahi": int(out["ahi_valid"].sum()) if len(out) else 0,
        "n_studies_attempted": int(len(out)),
        "missingness_note": (
            "SleepBank Sleep_Data locally contains only ~20 EDF/TSV pairs; "
            "cohort-wide AHI is NOT available from Health_Data tables. "
            "Do not fabricate labels for the remaining studies."
        ),
        "derivation": (
            "AHI = count(apnea|hypopnea annotations) / sleep_hours; "
            "sleep_hours from stage annotation durations when present."
        ),
        "pediatric_severity_bins": [
            {"name": n, "lo": lo, "hi": hi} for n, lo, hi in PED_AHI_BINS
        ],
    }
    return out, summary


def build_labels(
    clean_events: pd.DataFrame,
    ped_first: pd.DataFrame,
    psg_enc_events: pd.DataFrame | None,
) -> dict:
    osa = build_incident_osa(clean_events, ped_first, include_psg_encounter_events=psg_enc_events)
    osa["age_band"] = pd.cut(
        osa["index_age_years"],
        bins=[-0.01, 1, 6, 12, 18],
        labels=["<1", "1-5", "6-11", "12-17"],
    ).astype(str)

    ahi_df, ahi_sum = extract_ahi_from_local_tsvs()
    labels = osa.merge(ahi_df, on=["patient_id", "sleep_study_id"], how="left")
    labels["ahi_valid"] = labels["ahi_valid"].fillna(0).astype(int)
    labels["severity_class"] = labels.apply(
        lambda r: _severity(r["ahi"]) if r.get("ahi_valid", 0) == 1 else "missing", axis=1
    )

    prev = (
        labels.groupby("age_band")
        .agg(
            n=("patient_id", "size"),
            incident_osa_prev=("incident_osa", "mean"),
            osa_at_index_prev=("osa_at_index", "mean"),
            prior_osa_prev=("prior_osa", "mean"),
            n_ahi_valid=("ahi_valid", "sum"),
        )
        .reset_index()
    )
    report = {
        "osa_code_set": osa_code_set(),
        "n_patients": int(labels["patient_id"].nunique()),
        "n_studies": int(len(labels)),
        "incident_osa_n": int(labels["incident_osa"].sum()),
        "incident_osa_prevalence": float(labels["incident_osa"].mean()),
        "osa_at_index_n": int(labels["osa_at_index"].sum()),
        "prior_osa_n": int(labels["prior_osa"].sum()),
        "by_age_band": prev.to_dict(orient="records"),
        "ahi": ahi_sum,
        "ahi_distribution_valid": (
            labels.loc[labels["ahi_valid"] == 1, "ahi"].describe().to_dict()
            if (labels["ahi_valid"] == 1).any() else {}
        ),
        "ahi_severity_counts": labels.loc[labels["ahi_valid"] == 1, "severity_class"].value_counts().to_dict()
        if (labels["ahi_valid"] == 1).any() else {},
    }
    out = P.DIRS["labels"]
    labels.to_parquet(out / "stage2_labels_pediatric_first.parquet", index=False)
    P.write_json(out / "osa_code_set.json", osa_code_set())
    P.write_json(out / "labels_report.json", report)
    if len(ahi_df):
        ahi_df.to_csv(out / "ahi_local_tsv_derived.csv", index=False)
    return report
