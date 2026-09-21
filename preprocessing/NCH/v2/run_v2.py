"""NCH Stage-2 v2 orchestration: cleanup → recovery → labels → splits → report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from preprocessing.NCH.v2 import paths as P
from preprocessing.NCH.v2.cleanup import (
    build_pediatric_cohort,
    clean_canonical_events,
    preindex_washout_report,
)
from preprocessing.NCH.v2.sequences import (
    assert_no_index_encounter_leakage,
    build_indexed_sequences,
    write_sequence_npz,
)
from preprocessing.NCH.v2.procedures import analyze_procedures, mimic_vs_nch_chapter_matrix, cpt_chapter_token
from preprocessing.NCH.v2.medications import analyze_medications
from preprocessing.NCH.v2.labels import build_labels
from preprocessing.NCH.v2.splits import make_splits
from preprocessing.NCH.v2.age_extrapolation import analyze_age_extrapolation
from preprocessing.NCH.v2.bge_extension import verify_frozen_bge, build_extended_vocab_sample
from collections import Counter


def _load_vocab() -> dict[str, int]:
    return {str(k): int(v) for k, v in json.loads(P.VOCAB_PATH.read_text()).items()}


def _truncation_report(meta: pd.DataFrame) -> dict:
    if meta.empty:
        return {}
    lens = meta["sequence_length_before_truncation"]
    qs = lens.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    by_age = []
    for band in ["<1", "1-5", "6-11", "12-17"]:
        sub = meta[meta["age_band"] == band]
        if sub.empty:
            continue
        by_age.append({
            "age_band": band,
            "n": int(len(sub)),
            "pct_gt_1024": float((sub["sequence_length_before_truncation"] > P.MAX_SEQ_LEN).mean() * 100),
            "median_len": float(sub["sequence_length_before_truncation"].median()),
            "median_duration_before": float(sub["history_duration"].median()),
            "median_duration_retained": float(sub["retained_duration"].median()),
            "median_frac_retained": float(sub["frac_history_retained"].median()),
            "median_earliest_age_before_y": float(sub["earliest_age_before"].median() / P.DAYS_PER_YEAR),
            "median_earliest_age_after_y": float(sub["earliest_age_after"].median() / P.DAYS_PER_YEAR),
        })
    # Adolescent early-childhood loss flag
    ado = meta[meta["age_band"] == "12-17"]
    ado_flag = {}
    if len(ado):
        ado_flag = {
            "pct_truncated": float((ado["sequence_length_before_truncation"] > P.MAX_SEQ_LEN).mean() * 100),
            "median_earliest_age_before_y": float(ado["earliest_age_before"].median() / P.DAYS_PER_YEAR),
            "median_earliest_age_after_y": float(ado["earliest_age_after"].median() / P.DAYS_PER_YEAR),
            "highlight": (
                "Truncation disproportionately drops early-childhood history among adolescents"
                if (ado["earliest_age_after"].median() - ado["earliest_age_before"].median()) > 365
                else "Adolescent truncation does not strongly shift earliest retained age"
            ),
        }
    report = {
        "percentiles_raw_length": {str(k): float(v) for k, v in qs.items()},
        "n_gt_1024": int((lens > P.MAX_SEQ_LEN).sum()),
        "pct_gt_1024": float((lens > P.MAX_SEQ_LEN).mean() * 100),
        "by_age_band": by_age,
        "adolescent_early_history": ado_flag,
    }
    # plots
    fig_dir = P.DIRS["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(meta["index_age_years"], meta["sequence_length_before_truncation"], s=8, alpha=0.35)
    ax.axhline(P.MAX_SEQ_LEN, color="C1", ls="--", label="max_seq_len=1024")
    ax.set_xlabel("index age (years)")
    ax.set_ylabel("raw sequence length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "seq_len_by_index_age.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(meta["index_age_years"], meta["retained_duration"] / P.DAYS_PER_YEAR, s=8, alpha=0.35)
    ax.set_xlabel("index age (years)")
    ax.set_ylabel("retained history duration (years)")
    fig.tight_layout()
    fig.savefig(fig_dir / "retained_duration_by_index_age.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(meta["index_age_years"], meta["frac_history_retained"], s=8, alpha=0.35)
    ax.set_xlabel("index age (years)")
    ax.set_ylabel("fraction of history retained")
    fig.tight_layout()
    fig.savefig(fig_dir / "frac_history_retained_by_index_age.png", dpi=120)
    plt.close(fig)

    P.write_json(P.DIRS["validation"] / "truncation_report.json", report)
    return report


def _fix_procedures_mimic_matrix_via_vocab(nch_proc: pd.DataFrame) -> dict:
    vocab = _load_vocab()
    h = [k.replace("HCPCS_", "") for k in vocab if k.startswith("HCPCS_")]
    mim = Counter(cpt_chapter_token(x) or "NONE" for x in h)
    nch_h = nch_proc[
        nch_proc["code_system"].astype(str).str.upper().isin({"HCPCS", "CPT"})
        | nch_proc["raw_code"].astype(str).str.match(r"^\d{4,5}$|^[A-Z]\d{4}$", na=False)
    ]
    nch = Counter(cpt_chapter_token(str(r)) or "NONE" for r in nch_h["raw_code"])
    all_ch = sorted(set(mim) | set(nch))
    matrix = {
        ch: {
            "mimic_unique_hcpcs_codes": int(mim.get(ch, 0)),
            "nch_events": int(nch.get(ch, 0)),
            "both_nonzero": bool(mim.get(ch, 0) and nch.get(ch, 0)),
        }
        for ch in all_ch
    }
    out = {
        "note": "MIMIC side uses unique HCPCS_* vocab codes (event table too large to scan). NCH side event-weighted.",
        "shared_chapters": sum(1 for ch, v in matrix.items() if v["both_nonzero"] and ch != "NONE"),
        "matrix": matrix,
    }
    P.write_json(P.DIRS["procedure_mapping"] / "mimic_nch_chapter_matrix.json", out)
    return out


def _temporal_stats(events: pd.DataFrame, indexes: pd.DataFrame) -> dict:
    ev = events.copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    idx = indexes.copy()
    idx["index_time"] = pd.to_datetime(idx["index_time"], errors="coerce")
    rows = []
    by = {p: g for p, g in ev.groupby("patient_id")}
    for r in idx.itertuples(index=False):
        g = by.get(int(r.patient_id))
        if g is None:
            continue
        sub = g[g["event_time"] < r.index_time]
        if sub.empty:
            continue
        dur = (r.index_time - sub["event_time"].min()).total_seconds() / 86400.0
        rows.append(dur)
    arr = np.asarray(rows, dtype=float)
    return {
        "n": int(len(arr)),
        "history_duration_days": {
            "min": float(arr.min()) if len(arr) else None,
            "p50": float(np.median(arr)) if len(arr) else None,
            "p95": float(np.percentile(arr, 95)) if len(arr) else None,
            "p99": float(np.percentile(arr, 99)) if len(arr) else None,
            "max": float(arr.max()) if len(arr) else None,
        },
        "max_years": float(arr.max() / P.DAYS_PER_YEAR) if len(arr) else None,
        "extreme_120y_gone": bool(len(arr) == 0 or arr.max() < 120 * P.DAYS_PER_YEAR),
    }


def _attach_representation_tokens(events: pd.DataFrame, proc_detail: Path | None, med_detail: Path | None,
                                  mode: str, vocab: dict) -> pd.DataFrame:
    """mode: strict | normalized | extended"""
    ev = events.copy()
    if mode == "strict":
        return ev
    if proc_detail and proc_detail.exists():
        pd_ = pd.read_parquet(proc_detail)
        col = "normalized_stage1_token" if mode == "normalized" else "extended_token"
        m = pd_.set_index(["patient_id", "raw_code"])[col].to_dict()
        # apply to procedures
        is_p = ev["event_type"] == "procedure"
        new_tok, new_id = [], []
        for r in ev.itertuples(index=False):
            if r.event_type != "procedure":
                new_tok.append(r.mimic_token)
                new_id.append(r.mimic_token_id)
                continue
            t = m.get((r.patient_id, r.raw_code))
            if t and t in vocab:
                new_tok.append(t)
                new_id.append(vocab[t])
            elif mode == "extended" and t and str(t).startswith("CPT_CHAPTER_"):
                new_tok.append(t)
                new_id.append(pd.NA)  # will be UNK unless extended vocab loaded
            else:
                new_tok.append(r.mimic_token)
                new_id.append(r.mimic_token_id)
        ev["mimic_token"] = new_tok
        ev["mimic_token_id"] = new_id
    if med_detail and med_detail.exists() and mode in {"normalized", "extended"}:
        md = pd.read_parquet(med_detail)
        col = "level1_token" if mode == "normalized" else "level2_token"
        # build rxcui map
        sub = md.dropna(subset=[col]).drop_duplicates("rxcui")
        mp = {
            str(getattr(r, "rxcui")): getattr(r, col)
            for r in sub.itertuples(index=False)
        }
        toks, ids = [], []
        for r in ev.itertuples(index=False):
            if r.event_type != "medication":
                toks.append(r.mimic_token)
                ids.append(r.mimic_token_id)
                continue
            if pd.notna(r.mimic_token_id):
                toks.append(r.mimic_token)
                ids.append(r.mimic_token_id)
                continue
            t = mp.get(str(r.raw_code).replace(".0", ""))
            if t and isinstance(t, str) and t.startswith("RXN_") and t in vocab:
                toks.append(t)
                ids.append(vocab[t])
            else:
                toks.append(r.mimic_token)
                ids.append(r.mimic_token_id)
        ev["mimic_token"] = toks
        ev["mimic_token_id"] = ids
    return ev


def main() -> int:
    P.ensure_dirs()
    vocab = _load_vocab()
    print("Loading v1 canonical / sleep / patients…")
    canonical = pd.read_parquet(P.V1_CANONICAL)
    sleep = pd.read_parquet(P.V1_SLEEP)
    patients = pd.read_parquet(P.V1_PATIENTS)

    print("A. Cleanup…")
    clean, cleanup_report = clean_canonical_events(canonical, sleep)
    wash = preindex_washout_report(clean, sleep)
    ped_first, ped_all, cohort = build_pediatric_cohort(sleep)
    leak = assert_no_index_encounter_leakage(sleep, clean)
    temporal = _temporal_stats(clean, ped_first)
    P.write_json(P.DIRS["cleanup"] / "temporal_stats_after_cleanup.json", temporal)
    P.write_json(P.DIRS["validation"] / "leakage_test.json", leak)
    print("  cleaned events", len(clean), "leakage passed", leak["passed"], "max hist y", temporal.get("max_years"))

    psg_path = P.DIRS["cleanup"] / "psg_encounter_events.parquet"
    psg_enc = pd.read_parquet(psg_path) if psg_path.exists() else None

    print("B. Procedures…")
    proc_report = analyze_procedures(clean, vocab)
    proc_matrix = _fix_procedures_mimic_matrix_via_vocab(clean[clean.event_type == "procedure"])
    print("  exact coverage", proc_report["exact_token_level"]["event_weighted_coverage_pct"])

    print("C. Medications…")
    med_report = analyze_medications(clean, vocab)
    print("  L0", med_report["levels"]["L0_exact_stage1_rxcui"]["event_weighted_pct"])

    print("D. BGE freeze + sample extension…")
    freeze = verify_frozen_bge(P.STAGE1_BEST)
    # candidate concepts from OOV procedures/meds with descriptions
    candidates = []
    for ch, desc in [
        ("CPT_CHAPTER_ANES", "Anesthesia procedure (CPT chapter)"),
        ("CPT_CHAPTER_SURG", "Surgical procedure (CPT chapter)"),
        ("CPT_CHAPTER_RAD", "Radiology procedure (CPT chapter)"),
        ("CPT_CHAPTER_PATH", "Pathology and laboratory procedure (CPT chapter)"),
        ("CPT_CHAPTER_MED", "Medicine procedure (CPT chapter)"),
        ("CPT_CHAPTER_EAM", "Evaluation and management service (CPT chapter)"),
        ("CPT_CHAPTER_HCPCS2", "HCPCS Level II procedure/service"),
    ]:
        candidates.append({"token": ch, "description": desc, "source": "cpt_chapter"})
    # a few pediatric-absent RXNs from med detail if available
    md_path = P.DIRS["medication_mapping"] / "medication_mapping_detail.parquet"
    if md_path.exists():
        md = pd.read_parquet(md_path)
        oov = md[md["mimic_token_id"].isna() & md["rxcui"].notna()].drop_duplicates("rxcui").head(15)
        names = {}
        try:
            names = json.loads((P.MAPPINGS_DIR / "rxcui_name_cache.json").read_text())
        except Exception:
            pass
        for r in oov.itertuples(index=False):
            nm = names.get(str(r.rxcui)) or r.GENERIC_DRUG_DESCR or f"RxNorm concept {r.rxcui}"
            candidates.append({
                "token": f"RXN_{r.rxcui}",
                "description": f"{nm} (medication)",
                "source": "pediatric_or_unseen_rxcui",
            })
    # Encode sample on CPU (small)
    ext = build_extended_vocab_sample(candidates, max_encode=25, run_encode=True)

    print("E+F. Sequences (UNK retention) + truncation…")
    variants = []
    # 1 diagnoses only
    for name, mods, mode in [
        ("diagnoses_only", ("diagnosis",), "strict"),
        ("dx_proc_strict", ("diagnosis", "procedure"), "strict"),
        ("dx_proc_normalized", ("diagnosis", "procedure"), "normalized"),
        ("dx_med_normalized", ("diagnosis", "medication"), "normalized"),
        ("dx_proc_med_strict", ("diagnosis", "procedure", "medication"), "strict"),
        ("dx_proc_med_normalized", ("diagnosis", "procedure", "medication"), "normalized"),
    ]:
        ev = _attach_representation_tokens(
            clean,
            P.DIRS["procedure_mapping"] / "procedure_mapping_detail.parquet",
            P.DIRS["medication_mapping"] / "medication_mapping_detail.parquet",
            mode,
            vocab,
        )
        samples, meta, summary = build_indexed_sequences(
            ev, ped_first, vocab, modalities=mods, representation=name, retain_oov_as_unk=True,
        )
        write_sequence_npz(P.DIRS["processed"] / f"{name}_sequences.npz", samples, len(vocab))
        meta.to_parquet(P.DIRS["processed"] / f"{name}_meta.parquet", index=False)
        P.write_json(P.DIRS["processed"] / f"{name}_summary.json", summary)
        # modality composition
        if samples:
            # approximate from clean events in cohort patients
            pass
        variants.append(summary)
        print(f"  {name}: n={summary['n_samples']} unk%={summary['unk_event_pct']:.1f} trunc%={summary['truncation_pct']:.1f}")

    # Primary meta for truncation / splits: dx_proc_med_normalized
    primary_meta = pd.read_parquet(P.DIRS["processed"] / "dx_proc_med_normalized_meta.parquet")
    trunc = _truncation_report(primary_meta)

    # UNK by modality on strict full
    strict_sum = json.loads((P.DIRS["processed"] / "dx_proc_med_strict_summary.json").read_text())
    unk_report = {
        "stage1_oov_contract": (
            "OOV clinical codes are retained as vocab index |V| → model token id 1 (UNK). "
            "They are NOT dropped. Confirmed in model_new/data.py collate (unk_id → 1)."
        ),
        "v1_bug": "v1 NCH preprocessing silently dropped OOV events; v2 preserves Stage-1 UNK contract.",
        "strict_dx_proc_med": strict_sum,
        "unk_by_age_band": primary_meta.groupby("age_band")["frac_unk"].mean().to_dict(),
    }
    P.write_json(P.DIRS["validation"] / "unk_behavior.json", unk_report)

    print("G. Age extrapolation…")
    age_rep = analyze_age_extrapolation(P.STAGE1_BEST)

    print("H. Labels…")
    lab_rep = build_labels(clean, ped_first, psg_enc)
    labels = pd.read_parquet(P.DIRS["labels"] / "stage2_labels_pediatric_first.parquet")
    # attach demographics from ped_first
    labels = labels.merge(ped_first[["patient_id", "sex", "race"]], on="patient_id", how="left")

    print("I. Splits…")
    split_rep = make_splits(labels, primary_meta)

    print("J. Representation comparison…")
    # modality composition helper
    def modality_comp(mods):
        sub = clean[clean["event_type"].isin(mods)]
        return sub["event_type"].value_counts(normalize=True).mul(100).round(2).to_dict()

    comparison = []
    for name, mods in [
        ("diagnoses_only", ("diagnosis",)),
        ("dx_proc_strict", ("diagnosis", "procedure")),
        ("dx_proc_normalized", ("diagnosis", "procedure")),
        ("dx_med_normalized", ("diagnosis", "medication")),
        ("dx_proc_med_strict", ("diagnosis", "procedure", "medication")),
        ("dx_proc_med_normalized", ("diagnosis", "procedure", "medication")),
        ("extended_bge_candidate", ("diagnosis", "procedure", "medication")),
    ]:
        path = P.DIRS["processed"] / f"{name}_summary.json"
        if name == "extended_bge_candidate":
            # coverage if extended tokens were in vocab — report from proc/med extended stats
            comparison.append({
                "representation": name,
                "note": "Offline BGE append valid; full tensorized extended sequences not finetuned yet",
                "procedure_extended_coverage_pct": proc_report["extended_chapter_coverage"]["event_weighted_coverage_pct"],
                "medication_L2_any_pct": med_report["levels"]["L2_ingredient_any_including_new_ingset"]["event_weighted_pct"],
                "bge_extension_status": ext.get("status"),
                "technically_justified": freeze["model_contract"]["technically_valid_to_append"],
            })
            continue
        s = json.loads(path.read_text())
        s["modality_composition_pct"] = modality_comp(mods)
        comparison.append(s)

    P.write_json(P.DIRS["reports"] / "representation_comparison.json", comparison)

    # Final answers report
    final = {
        "version": "nch_stage2_v2",
        "cleanup": cleanup_report,
        "washout_preindex_analysis_only": wash,
        "cohort": cohort,
        "leakage_test": leak,
        "temporal_after_cleanup": temporal,
        "procedures": {
            "exact_event_weighted_pct": proc_report["exact_token_level"]["event_weighted_coverage_pct"],
            "format_normalized_pct": proc_report["after_format_normalization"]["event_weighted_coverage_pct"],
            "chapter_extended_pct": proc_report["extended_chapter_coverage"]["event_weighted_coverage_pct"],
            "compatible_with_stage1_CCS": proc_report["compatible_with_stage1_CCS"],
            "family_counts": proc_report["family_counts"],
            "mimic_nch_shared_chapters": proc_matrix.get("shared_chapters"),
            "ccs_svcsproc_available": False,
        },
        "medications": med_report["levels"],
        "medication_oov_char": med_report["oov_characterization"],
        "missing_rxcui": med_report["missing_rxcui_name_mapping"],
        "bge_extension": {"freeze": freeze, "sample": {k: ext.get(k) for k in ("status", "extended_shape", "defensible", "n_selected_for_encode")}},
        "truncation": trunc,
        "age_extrapolation": {
            "status": age_rep.get("status"),
            "lambda0": (age_rep.get("checkpoint") or {}).get("lambda0"),
            "beta": (age_rep.get("checkpoint") or {}).get("beta"),
            "flags": age_rep.get("flags"),
        },
        "labels": lab_rep,
        "splits": {k: split_rep["by_split"][k] for k in ("train", "val", "test")},
        "representation_variants_ready": [
            "diagnoses_only",
            "dx_proc_strict",
            "dx_proc_normalized",
            "dx_med_normalized",
            "dx_proc_med_strict",
            "dx_proc_med_normalized",
            "extended_bge_sample_embeddings (offline only; not yet a full Stage-2 tensor set)",
        ],
        "answers": _answers(
            cleanup_report, wash, cohort, leak, temporal, proc_report, proc_matrix,
            med_report, freeze, ext, trunc, age_rep, lab_rep, split_rep, comparison,
        ),
    }
    P.write_json(P.DIRS["reports"] / "v2_final_report.json", final)
    _write_markdown(final)
    print("Done. Report:", P.DIRS["reports"] / "v2_final_report.md")
    return 0


def _answers(cleanup, wash, cohort, leak, temporal, proc, matrix, med, freeze, ext,
             trunc, age, lab, splits, comparison):
    return {
        "1_procedure_mismatch_ontology": (
            f"Most OOV is CPT/HCPCS Level I ({proc['family_counts']}); ICD-PCS already maps to CCS_*. "
            f"Exact event-weighted coverage {proc['exact_token_level']['event_weighted_coverage_pct']:.1f}% → "
            f"format-normalized {proc['after_format_normalization']['event_weighted_coverage_pct']:.1f}%."
        ),
        "2_procedure_grouping_recovery": (
            f"CPT chapter grouping (not AMA CCS Svcs/Proc — unavailable) covers "
            f"{proc['extended_chapter_coverage']['event_weighted_coverage_pct']:.1f}% events as CPT_CHAPTER_* "
            f"(new namespace). Stage-1-compatible normalized recovery remains "
            f"{proc['normalized_transfer_to_existing_stage1']['event_weighted_coverage_pct']:.1f}%."
        ),
        "3_groups_compatible_with_stage1": (
            "NO for CPT chapters vs CCS_*: Stage-1 CCS_* are ICD procedure CCS. "
            f"MIMIC/NCH share {matrix.get('shared_chapters')} CPT chapters at coarse level."
        ),
        "4_med_coverage_ladder": med["levels"],
        "5_med_oov_pediatric_vs_formulation": med["oov_characterization"],
        "6_missing_rxcui_mappable": med["missing_rxcui_name_mapping"],
        "7_extended_bge_valid": freeze["model_contract"]["technically_valid_to_append"] and not freeze.get("bge_vectors_updated_during_pretraining", True),
        "8_extended_preserves": {
            "proc_chapter_pct": proc["extended_chapter_coverage"]["event_weighted_coverage_pct"],
            "med_L2_any_pct": med["levels"]["L2_ingredient_any_including_new_ingset"]["event_weighted_pct"],
            "sample_encode_status": ext.get("status"),
        },
        "9_truncation": trunc,
        "10_index_leakage": leak,
        "11_12_age_function": age.get("flags"),
        "13_incident_osa": {
            "n": lab.get("incident_osa_n"),
            "prevalence": lab.get("incident_osa_prevalence"),
        },
        "14_ahi": lab.get("ahi"),
        "15_splits": {k: splits["by_split"][k]["n_patients"] for k in ("train", "val", "test")},
        "16_ready_variants": [
            "diagnoses_only", "dx_proc_strict", "dx_proc_normalized",
            "dx_med_normalized", "dx_proc_med_strict", "dx_proc_med_normalized",
            "extended_bge_offline_sample",
        ],
        "washout_24h_7d": wash,
        "cohort_counts": cohort,
        "temporal_cleaned": temporal,
    }


def _write_markdown(final: dict) -> None:
    a = final["answers"]
    lines = [
        "# NCH Stage-2 preprocessing v2 report",
        "",
        "Versioned under `artifacts/nch_stage2/v2/` (v1 audit untouched).",
        "",
        "## Answers",
        "",
        f"1. **Procedure mismatch:** {a['1_procedure_mismatch_ontology']}",
        f"2. **Grouping recovery:** {a['2_procedure_grouping_recovery']}",
        f"3. **Stage-1 CCS compatible?** {a['3_groups_compatible_with_stage1']}",
        f"4. **Medication ladder:** see `medication_mapping/medication_ladder_report.json`",
        f"5. **Med OOV pediatric vs formulation:** `{json.dumps(a['5_med_oov_pediatric_vs_formulation'])}`",
        f"6. **Missing RxCUI:** `{json.dumps(a['6_missing_rxcui_mappable'])}`",
        f"7. **Extended frozen-BGE valid?** `{a['7_extended_bge_valid']}`",
        f"8. **Additional data preservable:** `{json.dumps(a['8_extended_preserves'])}`",
        f"9. **Truncation:** `{json.dumps(a['9_truncation'], default=str)[:500]}…`",
        f"10. **Index leakage:** `{json.dumps(a['10_index_leakage'])}`",
        f"11–12. **Age extrapolation:** `{json.dumps(a['11_12_age_function'])}`",
        f"13. **Incident OSA:** `{json.dumps(a['13_incident_osa'])}`",
        f"14. **AHI:** `{json.dumps(a['14_ahi'])}`",
        f"15. **Splits:** `{json.dumps(a['15_splits'])}`",
        f"16. **Ready variants:** {a['16_ready_variants']}",
        "",
        "## Notes",
        "",
        "- AHRQ CCS for Services and Procedures was **not** available (AMA gate); used CPT chapters with distinct `CPT_CHAPTER_*` namespace.",
        "- Stage-1 OOV contract restored: retain as UNK (id 1), do not drop.",
        "- No Stage-2 finetuning launched.",
        "",
    ]
    (P.DIRS["reports"] / "v2_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
