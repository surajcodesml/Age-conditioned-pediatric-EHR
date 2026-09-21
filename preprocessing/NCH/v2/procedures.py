"""Procedure OOV taxonomy + CPT chapter grouping (CCS Svcs/Proc unavailable)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.NCH.v2 import paths as P

# AMA CPT section ranges (standard public codebook section boundaries).
# NOT AHRQ CCS for Services and Procedures (AMA-gated; not available locally).
CPT_CHAPTERS: list[tuple[str, str, int | None, int | None, str]] = [
    ("CPT_CHAPTER_ANES", "Anesthesia", 100, 1999, r"^\d{4,5}$"),
    ("CPT_CHAPTER_SURG", "Surgery", 10004, 69990, r"^\d{5}$"),
    ("CPT_CHAPTER_RAD", "Radiology", 70010, 79999, r"^\d{5}$"),
    ("CPT_CHAPTER_PATH", "Pathology and Laboratory", 80047, 89398, r"^\d{5}$"),
    ("CPT_CHAPTER_MED", "Medicine", 90281, 99607, r"^\d{5}$"),
    ("CPT_CHAPTER_EAM", "Evaluation and Management", 99201, 99499, r"^\d{5}$"),
]

PROVENANCE = {
    "grouping_scheme": "AMA CPT codebook section ranges (public section boundaries)",
    "not_used": "AHRQ CCS for Services and Procedures (AMA license gate; download 404/unavailable)",
    "stage1_CCS_namespace": "CCS_* from ICD-9-PCS / ICD-10-PCS Single-Level CCS — NOT interchangeable with CPT Svcs/Proc CCS",
    "new_namespace": "CPT_CHAPTER_* (distinct from Stage-1 CCS_*)",
}


def classify_proc_family(code_system: str, raw_code: str) -> str:
    cs = (code_system or "").upper()
    raw = (raw_code or "").strip().upper().replace(".", "")
    if cs in {"ICD9PCS", "ICD-9-CM VOLUME 3", "ICD9"}:
        return "ICD9_PCS"
    if cs in {"ICD10PCS", "ICD-10-PCS", "ICD10"}:
        return "ICD10_PCS"
    if cs in {"NCH_LOCAL"} or raw.startswith("SHX"):
        return "NCH_LOCAL_SHX"
    if not raw or raw in {"REDACTED", "NI", "UN", "UNKNOWN"}:
        return "malformed_unknown"
    # HCPCS Level II: letter + digits (or letter-digit patterns)
    if re.fullmatch(r"[A-Z]\d{4}", raw) or re.fullmatch(r"[A-Z]\d{3}[A-Z]", raw):
        return "HCPCS_LEVEL_II"
    if re.fullmatch(r"\d{4}T", raw) or re.fullmatch(r"\d{4}F", raw):
        return "CPT_CATEGORY_II_III"
    if re.fullmatch(r"\d{4,5}", raw) or cs in {"HCPCS", "CPT"}:
        return "CPT_HCPCS_LEVEL_I"
    return "malformed_unknown"


def cpt_chapter_token(raw_code: str) -> str | None:
    raw = (raw_code or "").strip().upper().replace(".", "")
    if re.fullmatch(r"[A-Z]\d{4}", raw):
        return "CPT_CHAPTER_HCPCS2"
    if re.fullmatch(r"\d{4}T", raw):
        return "CPT_CHAPTER_CAT3"
    if re.fullmatch(r"\d{4}F", raw):
        return "CPT_CHAPTER_CAT2"
    if not re.fullmatch(r"\d{4,5}", raw):
        return None
    n = int(raw)
    # E/M overlaps numeric Medicine range; check first
    if 99201 <= n <= 99499:
        return "CPT_CHAPTER_EAM"
    if 100 <= n <= 1999:
        return "CPT_CHAPTER_ANES"
    if 10004 <= n <= 69990:
        return "CPT_CHAPTER_SURG"
    if 70010 <= n <= 79999:
        return "CPT_CHAPTER_RAD"
    if 80047 <= n <= 89398:
        return "CPT_CHAPTER_PATH"
    if 90281 <= n <= 99607:
        return "CPT_CHAPTER_MED"
    return "CPT_CHAPTER_OTHER"


def _cpt_chapter_coverage_block(proc: pd.DataFrame, chapters: list) -> dict:
    mask = proc["family"].str.startswith("CPT") | proc["family"].str.startswith("HCPCS")
    out = {
        "note": "CPT chapter grouping recovers category labels but does NOT map to Stage-1 CCS_* tokens",
        "top_chapters": Counter([c for c in chapters if c]).most_common(15),
    }
    if mask.any():
        ch_ok = pd.Series(chapters, index=proc.index).notna() & mask
        sub = proc.loc[mask]
        out.update(_coverage(ch_ok.loc[mask], sub, sub["patient_id"]))
    return out


def _coverage(mapped: pd.Series, events: pd.DataFrame, patients: pd.Series) -> dict:
    ok = mapped.reindex(events.index).fillna(False).astype(bool)
    return {
        "unique_code_coverage_pct": (
            100.0 * events.loc[ok, "raw_code"].nunique() / max(1, events["raw_code"].nunique())
        ),
        "event_weighted_coverage_pct": 100.0 * float(ok.mean()) if len(ok) else 0.0,
        "patient_coverage_pct": (
            100.0 * patients.loc[ok].nunique() / max(1, patients.nunique())
        ),
        "n_events_covered": int(ok.sum()),
        "n_events_oov": int((~ok).sum()),
    }


def analyze_procedures(clean_events: pd.DataFrame, vocab: dict[str, int]) -> dict:
    proc = clean_events[clean_events["event_type"] == "procedure"].copy()
    if proc.empty:
        return {"error": "no procedure events"}

    proc["family"] = [
        classify_proc_family(cs, rc)
        for cs, rc in zip(proc["code_system"].astype(str), proc["raw_code"].astype(str))
    ]
    family_counts = proc["family"].value_counts().to_dict()

    # Exact / token-level: already in mimic_token_id
    exact_ok = proc["mimic_token_id"].notna()
    # Formatting recovery: try HCPCS_ zero-pad / strip for still-OOV CPT
    recovered_fmt = []
    for r in proc.itertuples(index=False):
        if pd.notna(r.mimic_token_id):
            recovered_fmt.append(r.mimic_token)
            continue
        raw = str(r.raw_code or "").strip().upper().replace(".", "")
        cands = []
        if re.fullmatch(r"\d+", raw):
            cands = [f"HCPCS_{raw}", f"HCPCS_{raw.lstrip('0')}", f"HCPCS_{raw.zfill(5)}"]
        elif raw:
            cands = [f"HCPCS_{raw}"]
        hit = None
        for t in cands:
            if t in vocab:
                hit = t
                break
        recovered_fmt.append(hit)
    proc["fmt_token"] = recovered_fmt
    fmt_ok = exact_ok | pd.Series(recovered_fmt, index=proc.index).notna()

    # Chapter grouping for CPT/HCPCS Level I/II
    chapters = [cpt_chapter_token(str(rc)) for rc in proc["raw_code"]]
    proc["cpt_chapter"] = chapters
    # ICD PCS already roll to CCS_* when mapped
    chapter_ok = pd.Series(
        [
            (ch is not None) if fam.startswith("CPT") or fam.startswith("HCPCS") else bool(pd.notna(mid))
            for ch, fam, mid in zip(chapters, proc["family"], proc["mimic_token_id"])
        ],
        index=proc.index,
    )

    # For ICD already mapped to CCS — count as grouped under Stage-1 CCS
    icd_ccs_ok = proc["mimic_token"].fillna("").astype(str).str.startswith("CCS_")

    # Representation tokens for normalized_transfer (only existing Stage-1 tokens)
    # CPT chapters are NEW namespace — not Stage-1 compatible → extended only
    norm_token = []
    for r in proc.itertuples(index=False):
        if pd.notna(r.mimic_token_id):
            norm_token.append(r.mimic_token)
        elif r.fmt_token:
            norm_token.append(r.fmt_token)
        elif str(r.mimic_token or "").startswith("CCS_"):
            norm_token.append(r.mimic_token)
        else:
            norm_token.append(None)
    proc["normalized_stage1_token"] = norm_token

    # Extended token: prefer Stage-1, else CPT_CHAPTER_*, else None
    ext_token = []
    for r in proc.itertuples(index=False):
        if pd.notna(r.mimic_token_id) or r.fmt_token:
            ext_token.append(r.fmt_token or r.mimic_token)
        elif r.cpt_chapter:
            ext_token.append(r.cpt_chapter)
        elif str(r.mimic_token or "").startswith("CCS_"):
            ext_token.append(r.mimic_token)
        else:
            ext_token.append(None)
    proc["extended_token"] = ext_token

    report = {
        "provenance": PROVENANCE,
        "n_events": int(len(proc)),
        "n_unique_codes": int(proc["raw_code"].nunique()),
        "family_counts": family_counts,
        "exact_token_level": _coverage(exact_ok, proc, proc["patient_id"]),
        "after_format_normalization": _coverage(fmt_ok, proc, proc["patient_id"]),
        "cpt_chapter_grouping": _cpt_chapter_coverage_block(proc, chapters),
        "icd_pcs_ccs_mapped_events": int(icd_ccs_ok.sum()),
        "compatible_with_stage1_CCS": {
            "verdict": "NO — Stage-1 CCS_* are ICD procedure CCS categories; CPT chapters / SvcsProc CCS are a different taxonomy",
            "do_not_map_cpt_chapter_to_CCS": True,
        },
        "normalized_transfer_to_existing_stage1": _coverage(
            pd.Series(norm_token, index=proc.index).notna(), proc, proc["patient_id"]
        ),
        "extended_chapter_coverage": _coverage(
            pd.Series(ext_token, index=proc.index).notna(), proc, proc["patient_id"]
        ),
    }

    # Top recovered / remaining OOV
    oov = proc[~fmt_ok]
    report["top_remaining_oov_codes"] = (
        oov.groupby(["family", "raw_code"]).size().sort_values(ascending=False).head(30)
        .reset_index(name="n").to_dict(orient="records")
    )
    report["top_exact_mapped"] = (
        proc[exact_ok].groupby("mimic_token").size().sort_values(ascending=False).head(20)
        .reset_index(name="n").to_dict(orient="records")
    )

    out = P.DIRS["procedure_mapping"]
    proc[
        [
            "patient_id", "raw_code", "code_system", "family", "mimic_token", "mimic_token_id",
            "fmt_token", "cpt_chapter", "normalized_stage1_token", "extended_token", "mapping_status",
        ]
    ].to_parquet(out / "procedure_mapping_detail.parquet", index=False)
    P.write_json(out / "procedure_compatibility_report.json", report)
    return report


def mimic_vs_nch_chapter_matrix(clean_nch: pd.DataFrame) -> dict:
    """Apply CPT chapter grouping to MIMIC HCPCS events for analysis only."""
    # MIMIC train events: HCPCS_* tokens
    frames = []
    for path in (P.TRAIN_EVENTS, P.VAL_EVENTS):
        if path.exists():
            df = pd.read_parquet(path, columns=["code_id", "code_type"] if False else None)
            # detect columns
            cols = list(pd.read_parquet(path).head(0).columns)
            use = [c for c in ("code_id", "rolled_code", "code", "event_type", "code_type") if c in cols]
            frames.append(pd.read_parquet(path, columns=use))
    if not frames:
        return {"error": "MIMIC events not found"}
    mim = pd.concat(frames, ignore_index=True)
    code_col = "code_id" if "code_id" in mim.columns else ("rolled_code" if "rolled_code" in mim.columns else mim.columns[0])
    codes = mim[code_col].astype(str)
    hcpcs = codes[codes.str.startswith("HCPCS_")]
    raws = hcpcs.str.replace("HCPCS_", "", regex=False)
    mim_chapters = Counter(cpt_chapter_token(r) or "NONE" for r in raws)

    nch = clean_nch[clean_nch["event_type"] == "procedure"].copy()
    nch_h = nch[nch["code_system"].astype(str).str.upper().isin({"HCPCS", "CPT"})
                | nch["raw_code"].astype(str).str.match(r"^\d{4,5}$|^[A-Z]\d{4}$", na=False)]
    nch_chapters = Counter(cpt_chapter_token(str(r)) or "NONE" for r in nch_h["raw_code"])

    all_ch = sorted(set(mim_chapters) | set(nch_chapters))
    matrix = {
        ch: {
            "mimic_events": int(mim_chapters.get(ch, 0)),
            "nch_events": int(nch_chapters.get(ch, 0)),
            "both_nonzero": bool(mim_chapters.get(ch, 0) and nch_chapters.get(ch, 0)),
        }
        for ch in all_ch
    }
    shared = sum(1 for ch, v in matrix.items() if v["both_nonzero"] and ch != "NONE")
    out = {
        "note": "Same CPT chapter scheme on MIMIC leftover HCPCS_* and NCH CPT/HCPCS events. Analysis only.",
        "n_mimic_hcpcs_events": int(len(hcpcs)),
        "n_nch_cpt_hcpcs_events": int(len(nch_h)),
        "shared_chapters": shared,
        "matrix": matrix,
        "interpretation": (
            "If many chapters are shared, exact-code OOV is largely granularity/code-set mismatch, "
            "not absence of overlapping clinical procedure domains."
        ),
    }
    P.write_json(P.DIRS["procedure_mapping"] / "mimic_nch_chapter_matrix.json", out)
    return out
