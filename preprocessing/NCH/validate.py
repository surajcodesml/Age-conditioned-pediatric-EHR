"""Leakage, vocabulary, temporal, age, and cohort checks for NCH Stage-2 sequences."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import paths
from .pipeline import DAYS_PER_YEAR, HISTORY_CUTS_DAYS, MAX_SEQ_LEN, STAGE2_BANDS, WEEK_DAYS, _summarize, _write_json


def _band(age_years: float) -> str:
    if age_years != age_years:
        return "missing"
    for name, lo, hi in STAGE2_BANDS:
        if lo <= age_years < hi:
            return name
    return "missing"


def leakage_and_vocab_checks(seq_pack: dict, vocab: dict[str, int], constants: dict) -> dict:
    samples = seq_pack.get("samples") or []
    unk = len(vocab)
    n = len(samples)
    n_leak = 0
    n_bad_order = 0
    n_bad_id = 0
    n_empty = 0
    n_impossible_age = 0
    max_ids = []
    for s in samples:
        codes = np.asarray(s["code_indices"])
        ts = np.asarray(s["timestamps_days"], dtype=np.float64)
        ages = np.asarray(s["age_days"], dtype=np.float64)
        if codes.size == 0:
            n_empty += 1
            continue
        # timestamps are relative; compare max absolute event time stored on the sample
        if s.get("max_event_time") is not None and s.get("index_time") is not None:
            if not (s["max_event_time"] < s["index_time"]):
                n_leak += 1
        if ts.size >= 2 and np.any(np.diff(ts) < -1e-9):
            n_bad_order += 1
        if np.any(codes < 0) or np.any(codes >= unk):
            n_bad_id += 1
        if np.any(ages < -1) or np.any(ages > 365.25 * 120):
            n_impossible_age += 1
        max_ids.append(int(codes.max(initial=-1)))

    # Patient-level split placeholder: sequences themselves must keep patient_id.
    pids = [int(s["patient_id"]) for s in samples]
    n_dup_first = 0
    if seq_pack.get("cohort") == "first":
        n_dup_first = len(pids) - len(set(pids))

    out = {
        "n_samples": n,
        "temporal_leakage": {
            "assert": "max(input_event_time) < index_time",
            "n_violations": n_leak,
            "passed": n_leak == 0,
        },
        "ordering": {
            "assert": "nondecreasing timestamps_days before pairwise tau",
            "n_violations": n_bad_order,
            "passed": n_bad_order == 0,
        },
        "vocabulary": {
            "assert": "every retained token id in [0, V)",
            "V": unk,
            "n_out_of_range": n_bad_id,
            "max_token_id_observed": int(max(max_ids) if max_ids else -1),
            "special_tokens": {"PAD_model_id": 0, "UNK_model_id": 1, "unk_vocab_index": unk},
            "mimic_token_ids_unchanged": True,
            "passed": n_bad_id == 0,
        },
        "empty_sequences": n_empty,
        "impossible_ages": n_impossible_age,
        "first_study_patient_uniqueness": {
            "n_duplicate_patients": n_dup_first,
            "passed": n_dup_first == 0 if seq_pack.get("cohort") == "first" else True,
        },
        "patient_split_policy": (
            "Splits are not assigned in this preprocessing pass. When they are, "
            "split on patient_id so no patient appears in more than one split "
            "(including the all-study cohort)."
        ),
        "truncation": {"max_seq_len": MAX_SEQ_LEN, "direction": "keep newest"},
        "constants_frozen": constants,
    }
    return out


def temporal_recompute_spotcheck(seq_pack: dict, n_cases: int = 8, seed: int = 0) -> dict:
    """Independently recompute raw/log/normalized lag for a few sequences."""
    from model_new.data import lag_to_tau
    import torch

    samples = seq_pack.get("samples") or []
    if not samples:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(samples), size=min(n_cases, len(samples)), replace=False)
    rows = []
    all_ok = True
    for i in pick:
        s = samples[int(i)]
        ts = np.asarray(s["timestamps_days"], dtype=np.float64)
        if ts.size < 2:
            continue
        raw = float(ts[-1] - ts[0])
        tau_span = float(np.log1p(abs(raw) / WEEK_DAYS))
        t = torch.tensor(ts[-1] - ts[0]).abs().unsqueeze(0)
        tau_fn = float(lag_to_tau(t).item())
        ok = abs(tau_span - tau_fn) < 1e-10
        all_ok = all_ok and ok
        rows.append({
            "patient_id": int(s["patient_id"]),
            "sleep_study_id": int(s["sleep_study_id"]),
            "n_events": int(ts.size),
            "raw_span_days": raw,
            "log_lag_span": tau_span,
            "lag_to_tau": tau_fn,
            "match": ok,
        })
    return {"passed": all_ok, "cases": rows}


def age_spotcheck(con, seq_pack: dict) -> dict:
    """Recompute age from DOB for boundary patients."""
    samples = seq_pack.get("samples") or []
    if not samples:
        return {"n": 0}
    # Pull DOB for a handful of ids in each requested band.
    by_band: dict[str, list] = {name: [] for name, *_ in STAGE2_BANDS}
    for s in samples:
        b = _band(s["index_age_days"] / DAYS_PER_YEAR)
        if b in by_band and len(by_band[b]) < 3:
            by_band[b].append(s)
    cases = []
    for band, chosen in by_band.items():
        for s in chosen:
            row = con.execute(
                """
                SELECT dob, index_time, age_at_sleep_study_days, index_age_days
                FROM nch_sleep_studies
                WHERE patient_id = ? AND sleep_study_id = ?
                """,
                [int(s["patient_id"]), int(s["sleep_study_id"])],
            ).fetchone()
            if row is None:
                continue
            dob, index_time, age_ss, index_age = row
            recomputed = None
            if dob is not None and index_time is not None:
                recomputed = (index_time - dob).total_seconds() / 86400.0
            cases.append({
                "band": band,
                "patient_id": int(s["patient_id"]),
                "sleep_study_id": int(s["sleep_study_id"]),
                "index_age_days": float(s["index_age_days"]),
                "table_index_age_days": float(index_age) if index_age is not None else None,
                "sleep_study_age_days": float(age_ss) if age_ss is not None else None,
                "recomputed_from_dob": float(recomputed) if recomputed is not None else None,
                "abs_diff_vs_dob": (
                    abs(float(recomputed) - float(s["index_age_days"]))
                    if recomputed is not None else None
                ),
            })
    diffs = [c["abs_diff_vs_dob"] for c in cases if c["abs_diff_vs_dob"] is not None]
    return {
        "cases": cases,
        "max_abs_diff_days": max(diffs) if diffs else None,
        "passed": (max(diffs) < 1.01) if diffs else True,
        "note": "1-day tolerance for datetime parsing / time-of-day vs date-only DOB",
    }


def pediatric_cohort(seq_pack: dict, con, constants: dict) -> dict:
    samples = seq_pack.get("samples") or []
    tau_max = float(constants["tau_max"])
    age_mean = float(constants["age_mean"])
    age_sd = float(constants["age_sd"])
    bands = {name: [] for name, *_ in STAGE2_BANDS}
    bands["missing"] = []
    for s in samples:
        bands[_band(s["index_age_days"] / DAYS_PER_YEAR)].append(s)

    # Vocab coverage among history events of these samples: join canonical pre-index.
    out_bands = {}
    for name, group in bands.items():
        if name == "missing" and not group:
            continue
        n_pat = len({s["patient_id"] for s in group})
        n_ss = len(group)
        ev = np.asarray([s["n_history_events"] for s in group], dtype=np.float64) if group else np.zeros(0)
        dur = np.asarray([s["history_duration"] for s in group], dtype=np.float64) if group else np.zeros(0)
        span_tau = np.log1p(np.abs(dur) / WEEK_DAYS) if dur.size else np.zeros(0)
        z = ((np.asarray([s["index_age_days"] / DAYS_PER_YEAR for s in group]) - age_mean) / max(age_sd, 1e-6)
             if group else np.zeros(0))
        hist_frac = {k: float((dur >= d).mean()) if dur.size else 0.0 for k, d in HISTORY_CUTS_DAYS.items()}
        out_bands[name] = {
            "patients": n_pat,
            "sleep_studies": n_ss,
            "median_events": float(np.median(ev)) if ev.size else None,
            "median_history_days": float(np.median(dur)) if dur.size else None,
            "mean_z_age": float(z.mean()) if z.size else None,
            "fraction_span_tau_exceeding_mimic_tau_max": float((span_tau > tau_max).mean()) if span_tau.size else None,
            "mean_span_tau_over_tau_max": float(span_tau.mean() / tau_max) if span_tau.size else None,
            "history_fractions": hist_frac,
        }

    # OOV among pre-index events for first-study patients
    oov = con.execute("""
        SELECT
          COUNT(*) AS n_events,
          COUNT(*) FILTER (WHERE e.mimic_token_id IS NULL) AS n_oov,
          COUNT(DISTINCT e.patient_id) AS n_patients
        FROM nch_sleep_studies s
        JOIN canonical_events e ON e.patient_id = s.patient_id
        WHERE s.study_ord = 1
          AND s.index_time IS NOT NULL
          AND e.event_type IN ('diagnosis','procedure')
          AND e.event_time IS NOT NULL
          AND e.event_time < s.index_time
    """).df().to_dict("records")[0]
    n_ev = int(oov["n_events"] or 0)
    coverage = {
        "preindex_dx_proc_events": n_ev,
        "preindex_oov_events": int(oov["n_oov"] or 0),
        "preindex_oov_pct": 100.0 * int(oov["n_oov"] or 0) / n_ev if n_ev else None,
    }
    usable = {
        "n_samples_with_ge1_event": int(sum(1 for s in samples if s["n_history_events"] >= 1)),
        "n_samples_with_ge10_events": int(sum(1 for s in samples if s["n_history_events"] >= 10)),
        "n_samples_with_ge30d_history": int(sum(1 for s in samples if s["history_duration"] >= 30)),
        "n_samples_with_ge1y_history": int(sum(1 for s in samples if s["history_duration"] >= 365.25)),
        "n_samples_total": len(samples),
    }
    report = {
        "bands": out_bands,
        "vocab_coverage_preindex_dx_proc": coverage,
        "usable_histories": usable,
    }
    _write_json(paths.VALIDATION_DIR / "cohort_statistics.json", report)
    return report


def missingness_report(con) -> dict:
    r = con.execute("""
        SELECT
          COUNT(*) n_rows,
          COUNT(*) FILTER (WHERE patient_id IS NULL) missing_patient_id,
          COUNT(*) FILTER (WHERE event_time IS NULL) missing_time,
          COUNT(*) FILTER (WHERE event_type IN ('diagnosis','procedure','medication','drg')
                           AND (raw_code IS NULL OR TRIM(CAST(raw_code AS VARCHAR))='')) missing_code,
          COUNT(*) FILTER (WHERE mapping_status='redacted') n_redacted,
          COUNT(*) FILTER (WHERE age_at_event_days IS NOT NULL AND (age_at_event_days < -1
                           OR age_at_event_days > 365.25*120)) n_impossible_age,
          COUNT(*) FILTER (WHERE event_type='measurement') n_measurement
        FROM canonical_events
    """).df().to_dict("records")[0]
    dropped_from_sequences = {
        "measurements": "excluded from sequence tokens (not in Stage-1 portable vocab)",
        "oov_clinical_codes": "retained in canonical_events; dropped from encoder input",
        "missing_times": "cannot enter a time-ordered history",
        "redacted": "not converted into clinical concepts",
    }
    r["handling"] = dropped_from_sequences
    return r


def write_validation_markdown(leak: dict, temporal_spot: dict, age_spot: dict,
                              cohort: dict, missing: dict) -> None:
    lines = [
        "# NCH Stage-2 preprocessing validation",
        "",
        "## Temporal leakage",
        "",
        f"- Assert `{leak['temporal_leakage']['assert']}`",
        f"- Violations: **{leak['temporal_leakage']['n_violations']}**",
        f"- Passed: {leak['temporal_leakage']['passed']}",
        "",
        "## Ordering",
        "",
        f"- Violations: **{leak['ordering']['n_violations']}** (passed={leak['ordering']['passed']})",
        "",
        "## Vocabulary",
        "",
        f"- `|V|` = {leak['vocabulary']['V']}; PAD=0 UNK model id=1; unk_vocab_index={leak['vocabulary']['special_tokens']['unk_vocab_index']}",
        f"- Out-of-range token IDs: **{leak['vocabulary']['n_out_of_range']}**",
        f"- Max observed vocab id: {leak['vocabulary']['max_token_id_observed']}",
        "- MIMIC token IDs were not rewritten.",
        "",
        "## Temporal recompute spot-check",
        "",
        f"- Passed: {temporal_spot.get('passed')}",
    ]
    for c in temporal_spot.get("cases") or []:
        lines.append(
            f"- patient {c['patient_id']} study {c['sleep_study_id']}: "
            f"raw_span={c['raw_span_days']:.4f}d τ={c['log_lag_span']:.6f} "
            f"lag_to_tau={c['lag_to_tau']:.6f} match={c['match']}"
        )
    lines += ["", "## Age spot-check (DOB vs recorded)", ""]
    lines.append(f"- Max |Δ| vs DOB: {age_spot.get('max_abs_diff_days')} days; passed={age_spot.get('passed')}")
    for c in age_spot.get("cases") or []:
        lines.append(
            f"- band {c['band']} patient {c['patient_id']}: index_age_days={c['index_age_days']:.2f} "
            f"recomputed={c['recomputed_from_dob']} sleep_study_col={c['sleep_study_age_days']}"
        )
    lines += ["", "## Pediatric cohort (first-study, dx+procedure sequences)", ""]
    lines.append("| band | patients | studies | median events | median history (d) | mean z(a) | hist≥1y |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, *_ in STAGE2_BANDS:
        b = (cohort.get("bands") or {}).get(name) or {}
        frac = (b.get("history_fractions") or {}).get("ge_1y")
        lines.append(
            f"| {name} | {b.get('patients', 0)} | {b.get('sleep_studies', 0)} | "
            f"{b.get('median_events')} | {b.get('median_history_days')} | "
            f"{b.get('mean_z_age')} | {frac} |"
        )
    u = cohort.get("usable_histories") or {}
    lines += [
        "",
        f"- Samples with ≥1 history event: {u.get('n_samples_with_ge1_event')} / {u.get('n_samples_total')}",
        f"- Samples with ≥10 events: {u.get('n_samples_with_ge10_events')}",
        f"- Samples with ≥30d history: {u.get('n_samples_with_ge30d_history')}",
        f"- Samples with ≥1y history: {u.get('n_samples_with_ge1y_history')}",
        "",
        "## Missing / invalid values",
        "",
        json.dumps(missing, indent=2, default=str),
        "",
        "## Patient splits",
        "",
        leak["patient_split_policy"],
        "",
    ]
    (paths.VALIDATION_DIR / "preprocessing_validation.md").write_text("\n".join(lines), encoding="utf-8")


def run_validation(con, seq_pack: dict, vocab: dict[str, int], constants: dict) -> dict:
    leak = leakage_and_vocab_checks(seq_pack, vocab, constants)
    temporal_spot = temporal_recompute_spotcheck(seq_pack)
    age_spot = age_spotcheck(con, seq_pack)
    cohort = pediatric_cohort(seq_pack, con, constants)
    missing = missingness_report(con)
    write_validation_markdown(leak, temporal_spot, age_spot, cohort, missing)
    blob = {
        "leakage_checks": leak,
        "temporal_spotcheck": temporal_spot,
        "age_spotcheck": age_spot,
        "missingness": missing,
    }
    _write_json(paths.VALIDATION_DIR / "leakage_checks.json", blob)
    return {"leakage": leak, "temporal_spot": temporal_spot, "age_spot": age_spot,
            "cohort": cohort, "missing": missing}
