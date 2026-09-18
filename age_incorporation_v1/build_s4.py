#!/usr/bin/env python3
"""Build S4 (age-dependent temporal decay) by post-processing the existing cohort.

Reads the existing patients.parquet / events.parquet, selects eligible patients
(≥90 days pre-index history), injects TEMP_QUERY + 6 temporal signal events,
generates y_S4 labels, writes augmented tables + diagnostics + plots.

Does NOT modify S0-S2 labels or rerun Synthea.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

YEAR = 365.25
EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "synthea" / "sep1-exp" / "output" / "full" / "processed"
AGE_GROUPS = ("<1", "1-5", "6-11", "12-17")
AGE_SCALE_YEARS = 18.0
SIGNAL_SEED = 20240904
LABEL_SEED = 20240905
N_SIGNALS = 6
GAP_MIN_DAYS = 7
GAP_MAX_DAYS = 90
NOISE_STD = 0.25
COEFF_TEMPORAL = 2.0
TARGET_PREVALENCE = 0.22


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def lambda_true(age_years):
    """True age-dependent decay rate."""
    return 0.5 + 2.5 * np.exp(-np.asarray(age_years, dtype=np.float64) / 4.0)


def tau_norm(gap_days):
    """Temporal normalization matching the model's convention."""
    denom = math.log1p(AGE_SCALE_YEARS * YEAR)
    t = np.log1p(np.asarray(gap_days, dtype=np.float64)) / denom
    return np.clip(t, 0.0, 1.0)


def find_intercept(lp, noise, target):
    lo, hi = -8.0, 6.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        mean_p = float(np.mean(sigmoid(mid + lp + noise)))
        if mean_p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def auc_score(y, scores):
    y = np.asarray(y, dtype=int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    return (float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def build_s4(data_dir: Path | None = None, output_dir: Path | None = None):
    data_dir = data_dir or DEFAULT_DATA_DIR
    output_dir = output_dir or (EXP_DIR / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_parquet(data_dir / "patients.parquet")
    events = pd.read_parquet(data_dir / "events.parquet")

    # ── Eligibility: ≥90 days pre-index history ──
    max_history = events.groupby("patient_id")["time_before_index_days"].max()
    eligible_pids = set(max_history[max_history >= 90].index)
    elig = patients[patients["patient_id"].isin(eligible_pids)].copy()
    elig = elig.sort_values("patient_id").reset_index(drop=True)
    n_elig = len(elig)
    print(f"Eligible patients (≥90d history): {n_elig} / {len(patients)}")

    # ── Generate signals ──
    rng = np.random.default_rng(SIGNAL_SEED)

    age_at_index = elig["age_at_index"].to_numpy(dtype=np.float64)
    index_dates = pd.to_datetime(elig["index_date"]).to_numpy()
    dobs = pd.to_datetime(elig["date_of_birth"]).to_numpy()

    # For each patient: 6 signals, each with polarity and gap
    polarities = rng.choice([-1, 1], size=(n_elig, N_SIGNALS))  # +1 or -1
    # Log-uniform gaps between 7 and 90 days
    log_min, log_max = math.log(GAP_MIN_DAYS), math.log(GAP_MAX_DAYS)
    gap_days = np.exp(rng.uniform(log_min, log_max, size=(n_elig, N_SIGNALS)))

    # TEMP_QUERY is 1 day before index
    query_gap_days = 1.0

    # Compute tau, lambda, weights
    query_age_years = age_at_index  # age at TEMP_QUERY ≈ age at index
    lam_true = lambda_true(query_age_years)  # (n_elig,)
    tau_vals = tau_norm(gap_days)  # (n_elig, 6) — gap from signal to TEMP_QUERY

    weights = np.exp(-lam_true[:, None] * tau_vals)  # (n_elig, 6)
    temporal_score = np.sum(polarities * weights, axis=1) / math.sqrt(N_SIGNALS)

    # Generate labels
    label_rng = np.random.default_rng(LABEL_SEED)
    noise = label_rng.normal(0.0, NOISE_STD, size=n_elig)
    lp = COEFF_TEMPORAL * temporal_score
    intercept = find_intercept(lp, noise, TARGET_PREVALENCE)
    score_s4 = intercept + lp + noise
    p_s4 = sigmoid(score_s4)
    u = label_rng.random(n_elig)
    y_s4 = (u < p_s4).astype(int)

    print(f"S4 intercept: {intercept:.4f}")
    print(f"S4 prevalence: {y_s4.mean():.4f}")

    # ── Save diagnostics on patients table ──
    elig["y_S4"] = y_s4
    elig["p_S4"] = p_s4
    elig["query_age_years"] = query_age_years
    elig["lambda_true"] = lam_true
    elig["oracle_temporal_score"] = temporal_score

    # Store per-signal diagnostics as JSON strings
    for k in range(N_SIGNALS):
        elig[f"signal_{k}_polarity"] = polarities[:, k]
        elig[f"signal_{k}_gap_days"] = gap_days[:, k]
        elig[f"signal_{k}_tau"] = tau_vals[:, k]
        elig[f"signal_{k}_weight"] = weights[:, k]

    # ── Create new events ──
    new_events = []
    for i in range(n_elig):
        pid = elig.iloc[i]["patient_id"]
        idx_date = pd.Timestamp(index_dates[i])
        dob = pd.Timestamp(dobs[i])

        # TEMP_QUERY: 1 day before index
        query_ts = idx_date - pd.Timedelta(days=query_gap_days)
        query_age_event = (query_ts - dob).total_seconds() / 86400.0 / YEAR
        new_events.append({
            "patient_id": pid,
            "event_timestamp": query_ts,
            "age_at_event": query_age_event,
            "time_before_index_days": query_gap_days,
            "event_code": "TEMP_QUERY",
            "event_type": "temporal_signal",
            "source": "s4_synthetic",
        })

        # 6 signal events: gap before TEMP_QUERY
        for k in range(N_SIGNALS):
            sig_gap = gap_days[i, k]
            sig_ts = query_ts - pd.Timedelta(days=sig_gap)
            sig_age = (sig_ts - dob).total_seconds() / 86400.0 / YEAR
            sig_tbi = (idx_date - sig_ts).total_seconds() / 86400.0
            code = "TEMP_POS" if polarities[i, k] == 1 else "TEMP_NEG"
            new_events.append({
                "patient_id": pid,
                "event_timestamp": sig_ts,
                "age_at_event": sig_age,
                "time_before_index_days": sig_tbi,
                "event_code": code,
                "event_type": "temporal_signal",
                "source": "s4_synthetic",
            })

    new_events_df = pd.DataFrame(new_events)
    # Combine with existing events
    events_s4 = pd.concat([events, new_events_df], ignore_index=True)
    events_s4 = events_s4.sort_values(["patient_id", "event_timestamp"]).reset_index(drop=True)

    # ── Write augmented tables ──
    s4_dir = output_dir / "s4_data"
    s4_dir.mkdir(parents=True, exist_ok=True)

    # Patients: keep all original columns, add S4 columns for eligible only
    patients_out = patients.merge(
        elig[["patient_id", "y_S4", "p_S4", "query_age_years", "lambda_true", "oracle_temporal_score"]],
        on="patient_id", how="left"
    )
    patients_out.to_parquet(s4_dir / "patients.parquet", index=False)

    # Events: full augmented set
    events_s4.to_parquet(s4_dir / "events.parquet", index=False)

    # Eligible patient IDs
    elig_ids = set(elig["patient_id"])

    # ── Diagnostics JSON ──
    diag = {
        "n_total": len(patients),
        "n_eligible": n_elig,
        "n_ineligible": len(patients) - n_elig,
        "eligible_by_age_group": {},
        "eligible_by_split": {},
        "prevalence_overall": float(y_s4.mean()),
        "prevalence_by_age": {},
        "prevalence_by_split": {},
        "intercept_S4": intercept,
        "noise_std": NOISE_STD,
        "coeff_temporal": COEFF_TEMPORAL,
        "target_prevalence": TARGET_PREVALENCE,
        "n_signals": N_SIGNALS,
        "gap_range_days": [GAP_MIN_DAYS, GAP_MAX_DAYS],
    }

    for g in AGE_GROUPS:
        mask = elig["developmental_age_group"] == g
        diag["eligible_by_age_group"][g] = int(mask.sum())
        if mask.any():
            diag["prevalence_by_age"][g] = float(y_s4[mask.to_numpy()].mean())

    for s in ("train", "val", "test"):
        mask = elig["split"] == s
        diag["eligible_by_split"][s] = int(mask.sum())
        if mask.any():
            diag["prevalence_by_split"][s] = float(y_s4[mask.to_numpy()].mean())

    # TEMP_POS/NEG balance
    n_pos_events = int((polarities == 1).sum())
    n_neg_events = int((polarities == -1).sum())
    diag["temp_pos_events"] = n_pos_events
    diag["temp_neg_events"] = n_neg_events
    diag["temp_pos_frac"] = n_pos_events / (n_pos_events + n_neg_events)

    # Gap distribution
    all_gaps = gap_days.ravel()
    diag["gap_days_median"] = float(np.median(all_gaps))
    diag["gap_days_p10"] = float(np.percentile(all_gaps, 10))
    diag["gap_days_p90"] = float(np.percentile(all_gaps, 90))
    diag["gap_days_by_age"] = {}
    for g in AGE_GROUPS:
        mask = (elig["developmental_age_group"] == g).to_numpy()
        if mask.any():
            gg = gap_days[mask].ravel()
            diag["gap_days_by_age"][g] = {
                "median": float(np.median(gg)),
                "p10": float(np.percentile(gg, 10)),
                "p90": float(np.percentile(gg, 90)),
            }

    # ── Truncation loss estimate ──
    # Check how many injected signals survive 1024-event truncation
    s4_event_counts = events_s4.groupby("patient_id").size()
    s4_sig_events = events_s4[events_s4["source"] == "s4_synthetic"]
    n_injected = len(s4_sig_events)
    # For patients over 1024, check how many signals are in the last 1024
    n_lost = 0
    for pid in elig_ids:
        pev = events_s4[events_s4["patient_id"] == pid]
        if len(pev) > 1024:
            kept = pev.tail(1024)
            kept_sig = kept[kept["source"] == "s4_synthetic"]
            total_sig = pev[pev["source"] == "s4_synthetic"]
            n_lost += len(total_sig) - len(kept_sig)
    diag["n_injected_events"] = n_injected
    diag["n_injected_lost_truncation"] = n_lost
    diag["frac_injected_lost"] = n_lost / n_injected if n_injected else 0.0

    # ── Simple diagnostic AUROCs ──
    diag["diagnostic_auroc"] = {}
    # 1. age-only
    diag["diagnostic_auroc"]["age_only"] = auc_score(y_s4, query_age_years)
    # 2. polarity-count-only (sum of polarities)
    pol_count = polarities.sum(axis=1).astype(float)
    diag["diagnostic_auroc"]["polarity_count_only"] = auc_score(y_s4, pol_count)
    # 3. gap-only (mean gap)
    mean_gap = gap_days.mean(axis=1)
    diag["diagnostic_auroc"]["gap_only"] = auc_score(y_s4, -mean_gap)
    # 4. unweighted polarity sum
    unwt_sum = polarities.sum(axis=1).astype(float) / math.sqrt(N_SIGNALS)
    diag["diagnostic_auroc"]["unweighted_polarity_sum"] = auc_score(y_s4, unwt_sum)
    # 5. oracle temporal score
    diag["diagnostic_auroc"]["oracle_temporal_score"] = auc_score(y_s4, temporal_score)

    (s4_dir / "s4_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    print("\n=== S4 Diagnostics ===")
    for k, v in diag.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")
    print("\n  diagnostic AUROCs:")
    for k, v in diag["diagnostic_auroc"].items():
        print(f"    {k}: {v:.4f}")

    # ── Plots ──
    # 1. age → lambda_true(age)
    ages_plot = np.linspace(0, 18, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ages_plot, lambda_true(ages_plot), color="#1f4e79", linewidth=2)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\lambda_{\mathrm{true}}(a)$")
    ax.set_title(r"True age-dependent decay rate $\lambda_{\mathrm{true}}(a) = 0.5 + 2.5 \, e^{-a/4}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(s4_dir / "lambda_true_curve.png", dpi=150)
    plt.close(fig)

    # 2. temporal weight vs gap at ages 1, 5, 10, 15
    fig, ax = plt.subplots(figsize=(6.5, 4))
    gap_plot = np.linspace(1, 90, 200)
    tau_plot = tau_norm(gap_plot)
    for age in [1, 5, 10, 15]:
        lam = lambda_true(age)
        w = np.exp(-lam * tau_plot)
        ax.plot(gap_plot, w, label=f"age={age}y (λ={lam:.2f})")
    ax.set_xlabel("Gap (days)")
    ax.set_ylabel("Temporal weight")
    ax.set_title("Temporal weight vs gap at different ages")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(s4_dir / "temporal_weight_by_age.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote S4 data to {s4_dir}")
    print(f"  patients: {s4_dir / 'patients.parquet'}")
    print(f"  events: {s4_dir / 'events.parquet'}")
    print(f"  diagnostics: {s4_dir / 's4_diagnostics.json'}")
    print(f"  plots: lambda_true_curve.png, temporal_weight_by_age.png")

    # Save full per-patient diagnostics (not model input)
    diag_cols = ["patient_id", "split", "developmental_age_group",
                 "query_age_years", "lambda_true", "oracle_temporal_score",
                 "y_S4", "p_S4"]
    for k in range(N_SIGNALS):
        diag_cols.extend([f"signal_{k}_polarity", f"signal_{k}_gap_days",
                          f"signal_{k}_tau", f"signal_{k}_weight"])
    elig[diag_cols].to_parquet(s4_dir / "s4_patient_diagnostics.parquet", index=False)

    return diag


if __name__ == "__main__":
    build_s4()
