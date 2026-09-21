#!/usr/bin/env python3
"""NCH Sleep DataBank → Stage-1 MIMIC contract (CPU-only, no training).

    python -m preprocessing.NCH.run_all

Does not touch the live ``adkm_s0`` run, MIMIC processed shards, or the GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Hide GPUs *before* importing torch (via model_new.data). Does not affect the
# already-running Stage-1 process, whose device visibility is already set.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

# Repo root on sys.path so `python -m preprocessing.NCH.run_all` and
# `python preprocessing/NCH/run_all.py` both work.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from preprocessing.NCH import paths  # noqa: E402
from preprocessing.NCH.contract import write_contract  # noqa: E402
from preprocessing.NCH.mapping import load_mapping_resources  # noqa: E402
from preprocessing.NCH.pipeline import (  # noqa: E402
    build_canonical_events,
    build_code_mapping,
    build_patients_and_studies,
    build_raw_events,
    build_sequences,
    compatibility_tables,
    connect,
    inventory_nch,
    load_frozen_constants,
    load_vocab,
    make_figures,
    temporal_audit,
    write_compatibility_markdown,
)
from preprocessing.NCH.validate import run_validation  # noqa: E402


def recommended_stage2_config(compat: dict, temporal: dict, cohort: dict, constants: dict) -> dict:
    families = {r["event_type"]: r for r in compat.get("by_family") or []}
    dx = families.get("diagnosis") or {}
    pr = families.get("procedure") or {}
    med = families.get("medication") or {}
    dx_cov = float(dx.get("event_weighted_coverage_pct") or 0)
    pr_cov = float(pr.get("event_weighted_coverage_pct") or 0)
    med_cov = float(med.get("event_weighted_coverage_pct") or 0)
    include_meds = med_cov >= 50.0
    nch_t = (temporal.get("nch_first_study_dx_proc") or {})
    return {
        "do_not_start_finetuning": True,
        "encoder_inputs": {
            "vocab_path": str(paths.VOCAB_PATH),
            "embedding_path": str(paths.EMBEDDING_PATH),
            "max_seq_len": constants["max_seq_len"],
            "pad_id": 0,
            "unk_model_id": 1,
            "unk_vocab_index": None,
            "week_days": constants["week_days"],
            "days_per_year": constants["days_per_year"],
            "age_mean": constants["age_mean"],
            "age_sd": constants["age_sd"],
            "tau_max_diagnostic_only": constants["tau_max"],
            "tau_max_used_by_stage1_kernel": False,
            "race_encoding": "one_hot",
            "truncation": "newest",
            "index": "first eligible sleep study; events with event_time < t_index",
        },
        "modalities_first_run": ["diagnosis", "procedure"],
        "modalities_optional": ["medication"] if include_meds else [],
        "exclude_from_tokens": ["measurement", "lab", "chart", "nch_local_procedure_id"],
        "coverage": {
            "diagnosis_event_pct": dx_cov,
            "procedure_event_pct": pr_cov,
            "medication_event_pct": med_cov,
            "include_medications_in_first_run": include_meds,
            "reason": (
                "Include medications in the first Stage-2 run only if event-weighted "
                f"RXN_ coverage is ≥50% (observed {med_cov:.1f}%)."
            ),
        },
        "cohort": "first_study (one index per patient)",
        "splits": "not assigned; when created, split on patient_id even for all-study sequences",
        "scientific_flags": {
            "age_z_is_adult_standardized": True,
            "nch_mean_z_age": (nch_t.get("z_age_using_mimic_moments") or {}).get("mean"),
            "tau_not_rescaled": True,
            "nch_span_tau_over_tau_max": nch_t.get("mean_span_tau_over_tau_max"),
            "same_calendar_day_diagnoses_before_psg_are_included": True,
        },
        "sequence_artifacts": {
            "first_study_dx_proc": str(paths.PROCESSED_DIR / "first_study_sequences.npz"),
            "all_study_dx_proc": str(paths.PROCESSED_DIR / "all_study_sequences.npz"),
        },
        "usable": cohort.get("usable_histories"),
    }


def main() -> int:
    paths.ensure_output_dirs()
    print("=== Stage-1 contract (read-only) ===", flush=True)
    write_contract()
    print(f"wrote {paths.CONTRACT_DIR}", flush=True)

    constants = load_frozen_constants()
    vocab = load_vocab()
    res = load_mapping_resources(vocab)
    print(f"vocab |V|={len(vocab)} phe9={len(res['phe9'])} phe10={len(res['phe10'])}", flush=True)

    con = connect(mem="6GB", threads=4)
    try:
        print("=== NCH inventory ===", flush=True)
        inv = inventory_nch(con)
        print(
            f"patients={inv['patients']['n_patients']} "
            f"studies={inv['sleep_studies']['n_studies']} "
            f"dx_rows={inv['diagnoses']['n_rows']}",
            flush=True,
        )

        print("=== patients + sleep studies ===", flush=True)
        build_patients_and_studies(con)

        print("=== raw events ===", flush=True)
        build_raw_events(con)
        n_raw = int(con.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0])
        print(f"raw_events={n_raw:,}", flush=True)

        print("=== code mapping (frozen MIMIC vocab) ===", flush=True)
        build_code_mapping(con, res)
        print("=== canonical events ===", flush=True)
        build_canonical_events(con)

        print("=== compatibility ===", flush=True)
        compat = compatibility_tables(con)
        write_compatibility_markdown(compat, inv)

        print("=== sequences (first/all × dx+proc / dx+proc+med) ===", flush=True)
        first_dx = build_sequences(con, vocab, constants, ("diagnosis", "procedure"), "first")
        all_dx = build_sequences(con, vocab, constants, ("diagnosis", "procedure"), "all")
        first_med = build_sequences(
            con, vocab, constants, ("diagnosis", "procedure", "medication"), "first"
        )
        all_med = build_sequences(
            con, vocab, constants, ("diagnosis", "procedure", "medication"), "all"
        )
        # Drop in-memory event arrays from the optional packs before audits.
        for pack in (all_dx, first_med, all_med):
            pack.pop("samples", None)
            pack.pop("meta_rows", None)
            print(
                f"{pack['cohort']} {pack['modalities']}: "
                f"{pack['n_samples']} samples / {pack['n_patients']} patients",
                flush=True,
            )
        print(
            f"first dx+proc: {first_dx['n_samples']} samples / {first_dx['n_patients']} patients",
            flush=True,
        )

        print("=== temporal audit + figures ===", flush=True)
        temporal = temporal_audit(con, first_dx, constants)
        figs = make_figures(temporal, first_dx, constants)
        print(f"figures: {len(figs)}", flush=True)

        print("=== validation ===", flush=True)
        from preprocessing.NCH.validate import run_validation

        val = run_validation(con, first_dx, vocab, constants)

        rec = recommended_stage2_config(compat, temporal, val["cohort"], constants)
        rec["sequence_counts"] = {
            "first_dx_proc": {k: first_dx[k] for k in ("n_samples", "n_patients", "npz", "meta")},
            "all_dx_proc": {k: all_dx[k] for k in ("n_samples", "n_patients", "npz", "meta")},
            "first_dx_proc_med": {k: first_med[k] for k in ("n_samples", "n_patients", "npz", "meta")},
            "all_dx_proc_med": {k: all_med[k] for k in ("n_samples", "n_patients", "npz", "meta")},
        }
        from preprocessing.NCH.pipeline import _write_json

        _write_json(paths.PROCESSED_DIR / "stage2_recommended_config.json", rec)
        print("done", flush=True)
        print(f"artifacts: {paths.ARTIFACT_ROOT}", flush=True)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
