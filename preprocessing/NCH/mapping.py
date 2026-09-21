"""Deterministic NCH → Stage-1 MIMIC token mapping.

Reuses the Stage-1 conventions from ``preprocessing/rollup_and_describe.py``:

* ICD codes are matched after ``strip/upper/remove-dots`` (``normalize_code``)
* Diagnoses roll up through the PheWAS ICD-9 / ICD-10 → PheCode maps
* Unmapped diagnoses keep ``ICD9_<nodot>`` / ``ICD10_<nodot>``
* Medications use ``RXN_<RxCUI>`` (NCH already stores RxNorm CUIs)
* CPT/HCPCS keep ``HCPCS_<code>`` (the Stage-1 HCPCS leftover namespace)
* ICD procedures, if present, roll up through the same CCS maps as MIMIC

No NCH-local vocabulary is created. Unjustified name-matching is not applied.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import paths

# Same transform as preprocessing.rollup_and_describe.normalize_code
def normalize_code(code: str) -> str:
    return str(code).strip().upper().replace(".", "")


REDACTED_EXACT = {
    "",
    "REDACTED",
    "REDACT",
    "*REDACTED*",
    "[REDACTED]",
    "NI",
    "UN",
    "UNKNOWN",
    "NULL",
    "NA",
    "N/A",
    "NONE",
    "MISSING",
    ".",
    "-",
    "*",
    "UNABLE TO OBTAIN",
}


@dataclass(frozen=True)
class MappedCode:
    event_type: str
    code_system: str
    raw_code: str
    normalized_code: str
    mimic_token: str | None
    mimic_token_id: int | None  # vocab index in [0, V); None if OOV/not retained
    mapping_status: str
    oov_cause: str | None
    needed_normalization: bool
    description: str | None = None


def is_redacted(code: object) -> bool:
    if code is None:
        return True
    s = str(code).strip()
    if not s:
        return True
    u = s.upper()
    if u in REDACTED_EXACT:
        return True
    if "REDACT" in u:
        return True
    return False


def classify_icd_system(code_type: object) -> str | None:
    """Return 'ICD9' or 'ICD10' from NCH DX_CODE_TYPE / PROC_CODE_TYPE, else None."""
    if code_type is None:
        return None
    s = re.sub(r"[^A-Z0-9]", "", str(code_type).strip().upper())
    if not s:
        return None
    if s.startswith("ICD10") or s in {"I10", "CM10", "PCS10"}:
        return "ICD10"
    if s.startswith("ICD9") or s in {"I9", "CM9", "PCS9"}:
        return "ICD9"
    return None


def classify_procedure_system(code_type: object, raw_code: str) -> str:
    """Return a Stage-1-aligned procedure namespace tag."""
    icd = classify_icd_system(code_type)
    if icd == "ICD9":
        return "ICD9PCS"
    if icd == "ICD10":
        return "ICD10PCS"
    t = re.sub(r"[^A-Z0-9]", "", str(code_type or "").strip().upper())
    if t in {"CPT", "CPT4", "HCPCS", "HCPC", "CPTII", "CPT2"}:
        return "HCPCS"
    # Numeric 5-digit CPT / alphanumeric HCPCS often arrive with empty type.
    code = normalize_code(raw_code)
    if re.fullmatch(r"\d{4}[A-Z0-9]", code) or re.fullmatch(r"[A-Z]\d{4}", code):
        return "HCPCS"
    if re.fullmatch(r"\d{5}", code):
        return "HCPCS"
    if t.startswith("SHX") or code.startswith("SHX"):
        return "NCH_LOCAL"
    if t in {"", "NAN", "NONE"}:
        return "UNKNOWN"
    return "NCH_LOCAL"


def nch_sex_to_mimic(gender: object) -> int:
    """MIMIC convention from build_event_table.py: M → 1, else 0."""
    if gender is None:
        return 0
    s = str(gender).strip().upper()
    if s in {"M", "MALE", "1"}:
        return 1
    return 0


def nch_race_to_mimic_string(
    race_descr: object,
    hispanic_cd: object,
    ethnicity_descr: object,
    pcori_race_cd: object = None,
) -> str:
    """Build a race string that ``model_new.data.encode_race`` understands.

    Hispanic ethnicity is promoted to HISPANIC, matching MIMIC's combined
    race/ethnicity field rather than leaving Hispanic White as WHITE.
    """
    hisp = str(hispanic_cd or "").strip().upper()
    eth = str(ethnicity_descr or "").strip().upper()
    # "Not Hispanic or Latino" contains the substring LATINO — check negation first.
    not_hisp = hisp in {"N", "NO", "0", "FALSE"} or eth.startswith("NOT HISPANIC")
    is_hisp = hisp in {"Y", "YES", "1", "TRUE"} or (
        ("HISPANIC" in eth or eth == "LATINO") and not not_hisp
    )
    if is_hisp and not not_hisp:
        return "HISPANIC"
    descr = str(race_descr or "").strip()
    if descr:
        return descr
    cd = str(pcori_race_cd or "").strip()
    # PCORI CDM race codes (01 American Indian, 02 Asian, 03 Black, 04 Hawaiian,
    # 05 White, 06 Multiple, 07 Refuse, UN Unknown).
    pcori = {
        "01": "AMERICAN INDIAN",
        "02": "ASIAN",
        "03": "BLACK",
        "04": "OTHER",
        "05": "WHITE",
        "06": "OTHER",
        "07": "UNKNOWN",
        "UN": "UNKNOWN",
        "OT": "OTHER",
    }
    return pcori.get(cd, "UNKNOWN")


def _load_phecode_csv(path: Path, code_col: str) -> dict[str, str]:
    import csv

    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(code_col) or row.get(code_col.upper()) or row.get(code_col.lower())
            phe = row.get("PheCode") or row.get("phecode") or row.get("PHECODE")
            if raw is None or phe is None:
                continue
            key = normalize_code(raw)
            phe_s = str(phe).strip()
            if not key or not phe_s:
                continue
            # Same conflict rule as create_phecode_maps: MIN(phecode) per ICD.
            if key not in out or phe_s < out[key]:
                out[key] = phe_s
    return out


def _load_ccs_proc_maps() -> tuple[dict[str, str], dict[str, str]]:
    """ICD-9-CM / ICD-10-PCS procedure → CCS, matching ``create_ccs_maps``."""
    import sys

    sys.path.insert(0, str(paths.REPO_ROOT / "preprocessing"))
    from rollup_and_describe import _parse_hcup_ccs_file  # type: ignore  # noqa: E402

    d = paths.MAPPINGS_DIR / "ccs_procedure_downloads"
    icd10 = d / "ccs_pr_icd10pcs_2019_1.csv"
    icd9 = next((p for p in d.glob("*prref*") if p.is_file()), None)
    proc10: dict[str, str] = {}
    proc9: dict[str, str] = {}
    if icd10.exists():
        proc10, _ = _parse_hcup_ccs_file(
            icd10,
            code_key="ICD-10-PCS CODE",
            ccs_key="CCS CATEGORY",
            desc_key="CCS CATEGORY DESCRIPTION",
        )
    if icd9 is not None:
        proc9, _ = _parse_hcup_ccs_file(
            icd9,
            code_key="ICD-9-CM CODE",
            ccs_key="CCS CATEGORY",
            desc_key="CCS CATEGORY DESCRIPTION",
        )
    return proc9, proc10


def load_mapping_resources(vocab: dict[str, int] | None = None) -> dict:
    if vocab is None:
        with paths.VOCAB_PATH.open("r", encoding="utf-8") as f:
            vocab = {str(k): int(v) for k, v in json.load(f).items()}
    phe9 = _load_phecode_csv(paths.MAPPINGS_DIR / "phecode_icd9_rolled.csv", "ICD9")
    phe10 = _load_phecode_csv(paths.MAPPINGS_DIR / "phecode_icd10.csv", "ICD10")
    try:
        proc9, proc10 = _load_ccs_proc_maps()
    except Exception:
        proc9, proc10 = {}, {}
    descriptions: dict[str, str] = {}
    if paths.DESCRIPTIONS_PATH.exists():
        with paths.DESCRIPTIONS_PATH.open("r", encoding="utf-8") as f:
            descriptions = {str(k): str(v) for k, v in json.load(f).items()}
    return {
        "vocab": vocab,
        "phe9": phe9,
        "phe10": phe10,
        "proc9": proc9,
        "proc10": proc10,
        "descriptions": descriptions,
        "unk_index": len(vocab),
    }


def _vocab_get(vocab: dict[str, int], token: str) -> int | None:
    if token in vocab:
        return int(vocab[token])
    return None


def _phe_token(phe: str) -> str:
    return "PHE_" + str(phe).strip()


def map_diagnosis(raw_code: object, code_type: object, res: dict) -> MappedCode:
    event_type = "diagnosis"
    raw = "" if raw_code is None else str(raw_code)
    if is_redacted(raw):
        return MappedCode(
            event_type, str(code_type or ""), raw, "", None, None,
            "redacted", "redacted", False,
        )
    system = classify_icd_system(code_type) or _guess_icd_version(raw)
    norm = normalize_code(raw)
    needed = norm != raw.strip().upper() and norm != raw.strip()
    # Also true if dots/case changed
    needed = normalize_code(raw) != str(raw).strip()
    vocab = res["vocab"]
    phe_maps = {"ICD9": res["phe9"], "ICD10": res["phe10"]}
    prefix = system if system in {"ICD9", "ICD10"} else None
    if prefix is None:
        return MappedCode(
            event_type, str(code_type or "UNKNOWN"), raw, norm, None, None,
            "oov", "coding_system_mismatch", needed,
        )
    phe = phe_maps[prefix].get(norm)
    if phe:
        token = _phe_token(phe)
        vid = _vocab_get(vocab, token)
        if vid is not None:
            return MappedCode(
                event_type, prefix, raw, norm, token, vid,
                "mapped_phecode", None, needed,
                res["descriptions"].get(token),
            )
        # Valid PheCode that adult MIMIC never kept in the frozen vocab.
        return MappedCode(
            event_type, prefix, raw, norm, token, None,
            "oov", "pediatric_absent", needed,
        )
    # Unmapped ICD kept in the Stage-1 leftover namespace (no dots).
    token = f"{prefix}_{norm}"
    vid = _vocab_get(vocab, token)
    if vid is not None:
        return MappedCode(
            event_type, prefix, raw, norm, token, vid,
            "unmapped_icd_in_vocab", None, needed,
            res["descriptions"].get(token),
        )
    # Try dotted leftover in case MIMIC stored dots (it does not, but check).
    dotted = f"{prefix}_{str(raw).strip().upper()}"
    vid2 = _vocab_get(vocab, dotted)
    if vid2 is not None:
        return MappedCode(
            event_type, prefix, raw, str(raw).strip().upper(), dotted, vid2,
            "unmapped_icd_in_vocab", None, False,
            res["descriptions"].get(dotted),
        )
    oov_cause = "code_version_mismatch" if prefix == "ICD9" else "unknown_unmappable"
    # If the other ICD version would map, call it version mismatch.
    other = "ICD10" if prefix == "ICD9" else "ICD9"
    if phe_maps[other].get(norm) or _vocab_get(vocab, f"{other}_{norm}") is not None:
        oov_cause = "code_version_mismatch"
    return MappedCode(
        event_type, prefix, raw, norm, token, None,
        "oov", oov_cause, needed,
    )


def _guess_icd_version(raw: str) -> str | None:
    s = normalize_code(raw)
    if not s:
        return None
    # ICD-10-CM: letter + digits. ICD-9-CM: digits (or E/V).
    if re.match(r"^[A-Z][0-9]", s):
        return "ICD10"
    if re.match(r"^[0-9EV]", s):
        return "ICD9"
    return None


def map_procedure(raw_code: object, code_type: object, res: dict, *,
                  nch_local_id: object = None) -> MappedCode:
    event_type = "procedure"
    raw = "" if raw_code is None else str(raw_code)
    if is_redacted(raw) and (nch_local_id is None or is_redacted(nch_local_id)):
        return MappedCode(
            event_type, str(code_type or ""), raw, "", None, None,
            "redacted", "redacted", False,
        )
    if is_redacted(raw):
        return MappedCode(
            event_type, "NCH_LOCAL", str(nch_local_id),
            str(nch_local_id).strip(), None, None,
            "oov", "nch_local", False,
        )
    system = classify_procedure_system(code_type, raw)
    vocab = res["vocab"]
    norm = normalize_code(raw)
    needed = norm != str(raw).strip()

    if system in {"ICD9PCS", "ICD10PCS"}:
        ccs_map = res["proc9"] if system == "ICD9PCS" else res["proc10"]
        ccs = ccs_map.get(norm)
        if ccs:
            token = "CCS_" + str(ccs).strip()
            vid = _vocab_get(vocab, token)
            if vid is not None:
                return MappedCode(
                    event_type, system, raw, norm, token, vid,
                    "mapped_ccs", None, needed, res["descriptions"].get(token),
                )
            leftover = ("PROC9_" if system == "ICD9PCS" else "PROC10_") + norm
            vid2 = _vocab_get(vocab, leftover)
            if vid2 is not None:
                return MappedCode(
                    event_type, system, raw, norm, leftover, vid2,
                    "unmapped_proc_in_vocab", None, needed,
                    res["descriptions"].get(leftover),
                )
            return MappedCode(
                event_type, system, raw, norm, token, None,
                "oov", "unknown_unmappable", needed,
            )
        leftover = ("PROC9_" if system == "ICD9PCS" else "PROC10_") + norm
        vid = _vocab_get(vocab, leftover)
        if vid is not None:
            return MappedCode(
                event_type, system, raw, norm, leftover, vid,
                "unmapped_proc_in_vocab", None, needed,
                res["descriptions"].get(leftover),
            )
        return MappedCode(
            event_type, system, raw, norm, leftover, None,
            "oov", "coding_system_mismatch", needed,
        )

    if system == "HCPCS":
        # Stage-1 leftover namespace for unmapped HCPCS/CPT. Try several formats.
        candidates = []
        for c in (norm, str(raw).strip().upper(), str(raw).strip()):
            if c:
                candidates.append("HCPCS_" + c)
        # Numeric CPT zero-stripped / zero-padded variants are formatting, not new concepts.
        if re.fullmatch(r"\d+", norm):
            candidates.append("HCPCS_" + norm.lstrip("0"))
            candidates.append("HCPCS_" + norm.zfill(5))
        seen = set()
        for token in candidates:
            if token in seen:
                continue
            seen.add(token)
            vid = _vocab_get(vocab, token)
            if vid is not None:
                needed_fmt = token != "HCPCS_" + str(raw).strip()
                return MappedCode(
                    event_type, "HCPCS", raw, norm, token, vid,
                    "hcpcs_exact", None, needed_fmt,
                    res["descriptions"].get(token),
                )
        return MappedCode(
            event_type, "HCPCS", raw, norm, "HCPCS_" + norm, None,
            "oov", "unknown_unmappable" if norm else "redacted", needed,
        )

    return MappedCode(
        event_type, system, raw, norm, None, None,
        "oov", "nch_local" if system in {"NCH_LOCAL", "UNKNOWN"} else "coding_system_mismatch",
        needed,
    )


def map_medication(rxnorm: object, generic: object, descr: object, res: dict) -> MappedCode:
    event_type = "medication"
    raw = "" if rxnorm is None else str(rxnorm).strip()
    if is_redacted(raw):
        # Do not invent a token from the free-text name.
        label = str(generic or descr or "")
        return MappedCode(
            event_type, "RXNORM", raw, "", None, None,
            "oov", "unknown_unmappable" if not label else "unknown_unmappable",
            False, label or None,
        )
    # CSV may parse RxCUI as float.
    try:
        cui = str(int(float(raw)))
    except (TypeError, ValueError):
        cui = normalize_code(raw)
    if not cui or cui == "0":
        return MappedCode(
            event_type, "RXNORM", raw, cui, None, None,
            "oov", "unknown_unmappable", False,
        )
    token = "RXN_" + cui
    vid = _vocab_get(res["vocab"], token)
    if vid is not None:
        return MappedCode(
            event_type, "RXNORM", raw, cui, token, vid,
            "rxcui_exact", None, False,
            res["descriptions"].get(token),
        )
    # Token is a well-formed RxCUI that adult MIMIC never observed.
    return MappedCode(
        event_type, "RXNORM", raw, cui, token, None,
        "oov", "pediatric_absent", False,
        str(generic or descr or "") or None,
    )


def map_drg(raw_code: object, res: dict) -> MappedCode:
    event_type = "drg"
    raw = "" if raw_code is None else str(raw_code).strip()
    if is_redacted(raw):
        return MappedCode(
            event_type, "DRG", raw, "", None, None, "redacted", "redacted", False,
        )
    candidates = []
    try:
        n = int(float(raw))
        candidates.extend([f"DRG_{n}", f"DRG_{n:03d}"])
    except (TypeError, ValueError):
        pass
    candidates.append("DRG_" + raw)
    candidates.append("DRG_" + normalize_code(raw))
    seen = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        vid = _vocab_get(res["vocab"], token)
        if vid is not None:
            return MappedCode(
                event_type, "DRG", raw, token[4:], token, vid,
                "drg_exact", None, token != "DRG_" + raw,
                res["descriptions"].get(token),
            )
    return MappedCode(
        event_type, "DRG", raw, normalize_code(raw), "DRG_" + normalize_code(raw),
        None, "oov", "unknown_unmappable", False,
    )


def map_row(event_type: str, raw_code: object, code_system: object, res: dict,
            **kwargs) -> MappedCode:
    et = str(event_type).strip().lower()
    if et == "diagnosis":
        return map_diagnosis(raw_code, code_system, res)
    if et == "procedure":
        return map_procedure(raw_code, code_system, res, nch_local_id=kwargs.get("nch_local_id"))
    if et == "medication":
        return map_medication(raw_code, kwargs.get("generic"), kwargs.get("description"), res)
    if et == "drg":
        return map_drg(raw_code, res)
    return MappedCode(
        et, str(code_system or ""), str(raw_code or ""),
        normalize_code(raw_code or ""), None, None,
        "excluded", "coding_system_mismatch", False,
    )


def mapped_to_dict(m: MappedCode) -> dict:
    return {
        "event_type": m.event_type,
        "code_system": m.code_system,
        "raw_code": m.raw_code,
        "normalized_code": m.normalized_code,
        "mimic_token": m.mimic_token,
        "mimic_token_id": m.mimic_token_id,
        "mapping_status": m.mapping_status,
        "oov_cause": m.oov_cause,
        "needed_normalization": bool(m.needed_normalization),
        "description": m.description,
    }


def iter_unique_maps(rows: Iterable[tuple], res: dict) -> list[dict]:
    """rows: (event_type, code_system, raw_code, extra_dict)."""
    out = []
    seen = set()
    for event_type, code_system, raw_code, extra in rows:
        key = (str(event_type), str(code_system), str(raw_code))
        if key in seen:
            continue
        seen.add(key)
        extra = extra or {}
        m = map_row(event_type, raw_code, code_system, res, **extra)
        d = mapped_to_dict(m)
        out.append(d)
    return out
