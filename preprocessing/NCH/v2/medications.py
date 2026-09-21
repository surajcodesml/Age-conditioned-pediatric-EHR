"""Medication normalization ladder using RxNorm relationships + NCH class fields."""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from preprocessing.NCH.v2 import paths as P

RXNORM_ZIP = P.MAPPINGS_DIR / "RxNorm_full_prescribe_04062026.zip"
RXCUI_NAME_CACHE = P.MAPPINGS_DIR / "rxcui_name_cache.json"

# RxNorm RELA values for brand→generic / ingredient collapse (TTY-aware)
INGREDIENT_RELAS = {
    "has_ingredient",
    "has_precise_ingredient",
    "has_tradename",  # brand SCD -> tradename; inverse used carefully
    "tradename_of",
    "has_form",
    "form_of",
    "consists_of",
    "constitutes",
    "has_dose_form",
    "dose_form_of",
}


def _load_rxcui_names() -> dict[str, str]:
    if RXCUI_NAME_CACHE.exists():
        return {str(k): str(v) for k, v in json.loads(RXCUI_NAME_CACHE.read_text()).items()}
    return {}


def _extract_rxn_tables(work: Path) -> tuple[Path | None, Path | None]:
    """Extract RXNCONSO + RXNREL from the pinned RxNorm zip if needed."""
    work.mkdir(parents=True, exist_ok=True)
    conso = work / "RXNCONSO.RRF"
    rel = work / "RXNREL.RRF"
    if conso.exists() and rel.exists():
        return conso, rel
    if not RXNORM_ZIP.exists():
        return None, None
    with zipfile.ZipFile(RXNORM_ZIP, "r") as zf:
        names = zf.namelist()
        for target, dest in (("RXNCONSO.RRF", conso), ("RXNREL.RRF", rel)):
            match = next((n for n in names if n.endswith(target)), None)
            if match:
                with zf.open(match) as src, dest.open("wb") as out:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
    return (conso if conso.exists() else None), (rel if rel.exists() else None)


def _build_rxnorm_maps(conso: Path, rel: Path, needed_cuis: set[str]) -> dict:
    """Build TTY map and ingredient collapse for needed CUIs (+ their neighbors)."""
    tty: dict[str, str] = {}
    name: dict[str, str] = {}
    # Pass 1: CONSO for needed + collect all for name lookups of ingredients
    with conso.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 15:
                continue
            cui, sab, tty_v, suppress, str_v = parts[0], parts[11], parts[12], parts[16] if len(parts) > 16 else "", parts[14]
            if sab != "RXNORM":
                continue
            if suppress == "Y":
                continue
            tty[cui] = tty_v
            name[cui] = str_v

    # Pass 2: REL edges involving needed cuis
    # RXNREL: rxcui1|...|rxcui2|...|rela|...
    to_in: dict[str, set[str]] = defaultdict(set)  # cui -> ingredient cuis
    to_scd: dict[str, str] = {}  # brand/pack -> SCD if available
    related = set(needed_cuis)
    with rel.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8:
                continue
            c1, c2, rela = parts[0], parts[4], parts[7]
            if not c1 or not c2:
                continue
            if c1 not in related and c2 not in related:
                continue
            if rela in {"has_ingredient", "has_precise_ingredient"}:
                to_in[c1].add(c2)
            elif rela == "has_ingredients":
                to_in[c1].add(c2)
            elif rela == "tradename_of":
                # c1 brand tradename of c2 generic
                to_scd[c1] = c2
            elif rela == "has_tradename":
                to_scd[c2] = c1  # careful — prefer not for collapse

    # Ingredient set key for combination drugs
    def ingredient_key(cui: str) -> str | None:
        ings = sorted(to_in.get(cui) or [])
        if not ings:
            # if already an ingredient TTY
            t = tty.get(cui, "")
            if t in {"IN", "PIN", "MIN"}:
                return f"INGSET_{cui}"
            # try via tradename_of then ingredients
            base = to_scd.get(cui)
            if base:
                ings = sorted(to_in.get(base) or [])
                if ings:
                    return "INGSET_" + "+".join(ings)
                if tty.get(base) in {"IN", "PIN", "MIN"}:
                    return f"INGSET_{base}"
            return None
        return "INGSET_" + "+".join(ings)

    def brand_neutral(cui: str) -> str | None:
        # Prefer SCD/GPCK/BPCK collapse via tradename_of to clinical drug
        t = tty.get(cui, "")
        if t in {"SCD", "SBD", "GPCK", "BPCK", "SCDG", "SBDG"}:
            base = to_scd.get(cui, cui)
            # If SBD -> tradename_of SCD
            if t == "SBD" and cui in to_scd:
                return f"RXN_{to_scd[cui]}"
            if t in {"SCD", "GPCK"}:
                return f"RXN_{cui}"
            if t == "BPCK" and cui in to_scd:
                return f"RXN_{to_scd[cui]}"
            return f"RXN_{cui}"
        if cui in to_scd:
            return f"RXN_{to_scd[cui]}"
        return None

    return {
        "tty": tty,
        "name": name,
        "to_ingredients": {k: sorted(v) for k, v in to_in.items()},
        "ingredient_key_fn": ingredient_key,
        "brand_neutral_fn": brand_neutral,
        "provenance": {
            "rxnorm_zip": str(RXNORM_ZIP),
            "rxnorm_version_pin": "RxNorm_full_prescribe_04062026",
            "conso": str(conso),
            "rel": str(rel),
        },
    }


def _norm_name(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 +/\-]", "", s)
    return s


def analyze_medications(clean_events: pd.DataFrame, vocab: dict[str, int]) -> dict:
    med = clean_events[clean_events["event_type"] == "medication"].copy()
    # Enrich from source CSV for class / term type / names
    src = pd.read_csv(
        P.NCH_CSVS["medication"],
        usecols=[
            "STUDY_PAT_ID", "STUDY_ENC_ID", "RXNORM_CODE", "RXNORM_TERM_TYPE",
            "MEDICATION_DESCR", "GENERIC_DRUG_DESCR", "PHARM_CLASS", "THERA_CLASS",
            "ROUTE", "DRUG_DOSE_UNIT",
        ],
        low_memory=False,
    )
    # Build lookup by patient+rxcui for class fields (many-to-many; take first non-null)
    src["RXCUI"] = pd.to_numeric(src["RXNORM_CODE"], errors="coerce")
    class_by_cui = (
        src.dropna(subset=["RXCUI"])
        .assign(RXCUI=lambda d: d["RXCUI"].astype(int).astype(str))
        .groupby("RXCUI")
        .agg({
            "RXNORM_TERM_TYPE": "first",
            "GENERIC_DRUG_DESCR": "first",
            "MEDICATION_DESCR": "first",
            "PHARM_CLASS": "first",
            "THERA_CLASS": "first",
        })
    )
    rxcui_names = _load_rxcui_names()

    med["rxcui"] = med["raw_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    med.loc[med["rxcui"].isin({"", "nan", "None", "0"}), "rxcui"] = pd.NA
    med = med.merge(class_by_cui, left_on="rxcui", right_index=True, how="left")

    needed = set(med["rxcui"].dropna().astype(str).unique())
    work = P.DIRS["work"] / "rxnorm"
    conso, rel = _extract_rxn_tables(work)
    maps = None
    if conso and rel:
        maps = _build_rxnorm_maps(conso, rel, needed)

    # Level 0: exact Stage-1 RXN_*
    level0 = med["mimic_token_id"].notna()

    # Level 1: brand-neutral clinical drug -> existing Stage-1 if possible
    l1_token = []
    l2_token = []
    l3_token = []
    tty_list = []
    for r in med.itertuples(index=False):
        cui = None if pd.isna(r.rxcui) else str(r.rxcui)
        tty_v = getattr(r, "RXNORM_TERM_TYPE", None) or (maps["tty"].get(cui) if maps and cui else None)
        tty_list.append(tty_v)
        if pd.notna(r.mimic_token_id):
            l1_token.append(r.mimic_token)
            l2_token.append(r.mimic_token)
        elif maps and cui:
            bn = maps["brand_neutral_fn"](cui)
            if bn and bn in vocab:
                l1_token.append(bn)
            else:
                l1_token.append(None)
            ik = maps["ingredient_key_fn"](cui)
            if ik:
                # Map ingredient set to Stage-1 only if single ingredient and RXN_ing in vocab
                ings = ik.replace("INGSET_", "").split("+")
                if len(ings) == 1 and f"RXN_{ings[0]}" in vocab:
                    l2_token.append(f"RXN_{ings[0]}")
                else:
                    # extended namespace candidate
                    l2_token.append(ik if ik else None)
            else:
                l2_token.append(None)
        else:
            l1_token.append(None)
            l2_token.append(None)

        th = getattr(r, "THERA_CLASS", None)
        ph = getattr(r, "PHARM_CLASS", None)
        klass = th if (th is not None and not (isinstance(th, float) and th != th) and str(th) != "nan") else ph
        if klass is None or (isinstance(klass, float) and klass != klass) or str(klass).strip().lower() in {
            "", "nan", "none", "ni", "un", "unknown", "redacted"
        }:
            l3_token.append(None)
        else:
            slug = re.sub(r"[^A-Z0-9]+", "_", str(klass).upper()).strip("_")[:80]
            l3_token.append(f"THERA_{slug}" if slug else None)

    med["tty"] = tty_list
    med["level1_token"] = l1_token
    med["level2_token"] = l2_token
    med["level3_token"] = l3_token

    # Missing RxCUI: exact name match against source GENERIC/MEDICATION_DESCR
    name_to_cui = {_norm_name(v): k for k, v in rxcui_names.items()}
    # Also index by source rows lacking RxCUI
    miss_src = src[src["RXNORM_CODE"].isna()].copy()
    miss_src["g_norm"] = miss_src["GENERIC_DRUG_DESCR"].map(lambda x: _norm_name(str(x or "")))
    miss_src["d_norm"] = miss_src["MEDICATION_DESCR"].map(lambda x: _norm_name(str(x or "")))

    missing = med["rxcui"].isna()
    name_map_rows = []
    recovered = 0
    # Prefer mapping via normalized generic/descr on the enriched med frame
    for idx, r in med.loc[missing].iterrows():
        g = str(r.get("GENERIC_DRUG_DESCR") or "")
        d = str(r.get("MEDICATION_DESCR") or r.get("description") or "")
        hit = None
        conf = None
        for label, field in ((g, "generic"), (d, "medication_descr")):
            nn = _norm_name(label)
            if nn and nn in name_to_cui:
                hit = name_to_cui[nn]
                conf = f"exact_norm_name:{field}"
                break
        if hit:
            recovered += 1
            tok = f"RXN_{hit}"
            name_map_rows.append({
                "patient_id": int(r["patient_id"]),
                "mapped_rxcui": hit,
                "token": tok,
                "in_stage1_vocab": tok in vocab,
                "provenance": conf,
                "source_generic": g,
                "source_descr": d,
            })
            if tok in vocab:
                med.at[idx, "level1_token"] = tok
                med.at[idx, "mimic_token"] = tok
                med.at[idx, "mimic_token_id"] = vocab[tok]

    # Report how many distinct missing-RxCUI source names could match in principle
    uniq_g = miss_src["g_norm"].replace("", pd.NA).dropna().unique()
    uniq_d = miss_src["d_norm"].replace("", pd.NA).dropna().unique()
    n_g_hit = sum(1 for n in uniq_g if n in name_to_cui)
    n_d_hit = sum(1 for n in uniq_d if n in name_to_cui)

    def cov(series_ok: pd.Series) -> dict:
        return {
            "event_weighted_pct": 100.0 * float(series_ok.mean()),
            "unique_concept_n": int(med.loc[series_ok, "rxcui"].nunique()) if "rxcui" in med else 0,
            "patient_coverage_pct": 100.0 * med.loc[series_ok, "patient_id"].nunique() / max(1, med["patient_id"].nunique()),
            "n_events_recovered": int(series_ok.sum()),
            "n_events_remaining_oov": int((~series_ok).sum()),
        }

    l0_ok = med["mimic_token_id"].notna()
    l1_ok = pd.Series(med["level1_token"]).notna()
    # Level 2 Stage-1 compatible = token starts with RXN_ and in vocab
    l2_stage1 = med["level2_token"].apply(lambda t: isinstance(t, str) and t.startswith("RXN_") and t in vocab)
    l2_any = med["level2_token"].notna()
    l3_any = med["level3_token"].notna()

    # Characterize OOV: pediatric vs formulation
    oov = med[~l0_ok]
    oov_cui = oov["rxcui"].dropna().astype(str)
    tty_oov = Counter(oov["tty"].fillna("missing").astype(str))
    # If ingredient of OOV drug exists in Stage-1 → formulation mismatch
    form_mismatch = 0
    pediatric_absent = 0
    if maps:
        for cui in oov_cui.unique():
            ik = maps["ingredient_key_fn"](cui)
            if not ik:
                pediatric_absent += 1
                continue
            ings = ik.replace("INGSET_", "").split("+")
            if any(f"RXN_{i}" in vocab for i in ings):
                form_mismatch += 1
            else:
                pediatric_absent += 1

    report = {
        "stage1_rxn_token_semantics": {
            "token_format": "RXN_<RxCUI>",
            "source": "MIMIC NDC→RxCUI rollup via RxNorm_full_prescribe_04062026; leftover NDC_* rare",
            "observed_nch_term_types": med["tty"].value_counts(dropna=False).head(20).to_dict(),
            "note": "Stage-1 tokens are RxCUI IDs as observed after NDC mapping (typically SCD/IN mix depending on RxNorm map target).",
        },
        "nch_medication_fields": {
            "columns": [
                "RXNORM_CODE", "RXNORM_TERM_TYPE", "MEDICATION_DESCR", "GENERIC_DRUG_DESCR",
                "PHARM_CLASS", "PHARM_SUBCLASS", "THERA_CLASS", "THERA_SUBCLASS", "ROUTE",
                "EFFECTIVE_DRUG_DOSE", "DRUG_DOSE_UNIT",
            ],
            "missing_rxcui_event_fraction_in_source_sample": float(src["RXNORM_CODE"].isna().mean()),
        },
        "levels": {
            "L0_exact_stage1_rxcui": cov(l0_ok),
            "L1_brand_neutral_to_existing_stage1": cov(l1_ok),
            "L2_ingredient_to_existing_stage1": cov(l2_stage1),
            "L2_ingredient_any_including_new_ingset": cov(l2_any),
            "L3_therapeutic_class_nch_fields": cov(l3_any),
        },
        "missing_rxcui_name_mapping": {
            "method": "exact normalized name match to rxcui_name_cache.json only (no fuzzy)",
            "n_missing_rxcui_events": int(missing.sum()),
            "n_exact_name_mapped": recovered,
            "n_mapped_into_stage1_vocab": int(sum(1 for r in name_map_rows if r["in_stage1_vocab"])),
            "unique_missing_generic_names": int(len(uniq_g)),
            "unique_missing_descr_names": int(len(uniq_d)),
            "unique_generics_exact_matchable_to_rxnorm_cache": int(n_g_hit),
            "unique_descr_exact_matchable_to_rxnorm_cache": int(n_d_hit),
            "examples": name_map_rows[:20],
            "note": (
                "Many missing-RxCUI rows are NCH compound/custom strings (e.g. LET gel, "
                "human milk) that have no exact RxNorm string match; fuzzy matching is disallowed."
            ),
        },
        "oov_characterization": {
            "unique_oov_cuis": int(oov_cui.nunique()),
            "formulation_mismatch_cuis_ingredient_in_stage1": form_mismatch,
            "genuinely_absent_or_unresolvable_cuis": pediatric_absent,
            "tty_among_oov": dict(tty_oov.most_common(15)),
        },
        "rxnorm_provenance": maps["provenance"] if maps else {"error": "RxNorm extract unavailable"},
        "examples_l1": (
            med.loc[l1_ok & ~l0_ok, ["rxcui", "level1_token", "GENERIC_DRUG_DESCR"]]
            .drop_duplicates("level1_token").head(15).to_dict(orient="records")
        ),
        "information_lost_notes": {
            "L1": "Collapses brand variants to clinical drug when RxNorm tradename_of supports it; loses brand identity.",
            "L2": "Collapses strengths/dose forms to ingredient set; loses formulation/route/strength.",
            "L3": "Uses NCH PHARM/THERA class strings; coarsest; not a Stage-1 token namespace.",
        },
    }

    out = P.DIRS["medication_mapping"]
    med[
        [
            "patient_id", "rxcui", "mimic_token", "mimic_token_id", "tty",
            "level1_token", "level2_token", "level3_token",
            "GENERIC_DRUG_DESCR", "PHARM_CLASS", "THERA_CLASS",
        ]
    ].to_parquet(out / "medication_mapping_detail.parquet", index=False)
    if name_map_rows:
        pd.DataFrame(name_map_rows).to_csv(out / "missing_rxcui_name_matches.csv", index=False)
    P.write_json(out / "medication_ladder_report.json", report)
    # Crosswalk for reproducibility
    cross = med.loc[med["rxcui"].notna(), ["rxcui", "level1_token", "level2_token", "level3_token", "tty"]].drop_duplicates()
    cross.to_csv(out / "rxnorm_normalization_crosswalk.csv", index=False)
    return report
