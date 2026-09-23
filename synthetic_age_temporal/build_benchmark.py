#!/usr/bin/env python3
"""Build the semi-synthetic age × temporal benchmark from Synthea backgrounds.

Pipeline
--------
1. Load background Synthea trajectories (no regeneration by default).
2. Patient-level split (70/15/15) with a fixed seed.
3. Construct prediction cutoffs (index date) with history strictly before t*.
4. Inject SYN_SIGNAL_* events at controlled lags.
5. Generate multi-label targets from the known softplus mechanism.
6. Save model inputs and ground-truth metadata separately.
7. Run oracle validation; refuse to mark the build ready if S2/S3 fail.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (
    CONTROLLED_MAX_SIGNAL_LAG_DAYS,
    CONTROLLED_MIN_AGE,
    DATA_SEED,
    DEFAULT_OUTPUT_DIR,
    INTERACTION_STRENGTHS,
    NOISE_STD,
    SCENARIO_SPECS,
    SCENARIOS,
    SIGNAL_LAGS_DAYS,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    TARGET_PREVALENCE,
    THETA0_DEFAULT,
    all_signal_codes,
    tau_from_days,
    z_age,
)
from generate_synthea import load_background_tables, write_manifest
from ground_truth import (
    ExampleSignals,
    build_target_specs,
    calibrate_parameters,
    calibrate_target_biases,
    compute_target_logits,
    oracle_predict,
    sample_targets,
    save_calibration,
    sigmoid,
)


def _bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _auroc_micro_macro(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    out = {"micro_auroc": float("nan"), "macro_auroc": float("nan"),
           "micro_auprc": float("nan"), "macro_auprc": float("nan")}
    try:
        out["micro_auroc"] = float(roc_auc_score(y.ravel(), p.ravel()))
        out["micro_auprc"] = float(average_precision_score(y.ravel(), p.ravel()))
    except ValueError:
        pass
    aurocs, auprcs = [], []
    for k in range(y.shape[1]):
        if y[:, k].sum() == 0 or y[:, k].sum() == len(y):
            continue
        try:
            aurocs.append(roc_auc_score(y[:, k], p[:, k]))
            auprcs.append(average_precision_score(y[:, k], p[:, k]))
        except ValueError:
            continue
    if aurocs:
        out["macro_auroc"] = float(np.mean(aurocs))
        out["macro_auprc"] = float(np.mean(auprcs))
    return out


def patient_split(
    patient_ids: np.ndarray, seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(patient_ids.astype(str)))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(SPLIT_FRACTIONS["train"] * n))
    n_val = int(round(SPLIT_FRACTIONS["val"] * n))
    train = ids[:n_train].tolist()
    val = ids[n_train : n_train + n_val].tolist()
    test = ids[n_train + n_val :].tolist()
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    return {"train": train, "val": val, "test": test}


def inject_signals_for_patient(
    *,
    age: float,
    cutoff: pd.Timestamp,
    dob: pd.Timestamp,
    rng: np.random.Generator,
    max_lag_days: float,
    n_signals: int | None = None,
) -> ExampleSignals:
    """Place SYN_SIGNAL_* events at controlled historical lags."""
    codes_pool = all_signal_codes()
    # Lifetime available before cutoff.
    life_days = max(0.0, float((cutoff - dob) / np.timedelta64(1, "D")))
    allowed = [d for d in SIGNAL_LAGS_DAYS if d <= max_lag_days and d < life_days - 1e-6]
    if not allowed:
        # Fall back to short lags still inside lifetime.
        allowed = [d for d in (7.0, 30.0, 90.0) if d < life_days - 1e-6]
    if not allowed:
        return ExampleSignals(
            codes=np.array([], dtype=object),
            lag_days=np.zeros(0),
            tau=np.zeros(0),
            times=np.array([], dtype="datetime64[ns]"),
        )

    k = n_signals if n_signals is not None else int(rng.integers(4, 9))
    k = min(k, len(allowed) * 2)
    lag_days = rng.choice(np.asarray(allowed, dtype=np.float64), size=k, replace=True)
    # Jitter ±10% so lags are not discrete one-hot shortcuts.
    jitter = rng.uniform(0.9, 1.1, size=k)
    lag_days = np.clip(lag_days * jitter, 1.0, min(max_lag_days, life_days - 0.5))
    code_ids = rng.choice(len(codes_pool), size=k, replace=True)
    codes = np.array([codes_pool[i] for i in code_ids], dtype=object)
    times = np.array(
        [cutoff - pd.Timedelta(days=float(d)) for d in lag_days],
        dtype="datetime64[ns]",
    )
    # Enforce t_j < t*.
    assert np.all(times < np.datetime64(cutoff))
    tau = tau_from_days(lag_days)
    return ExampleSignals(codes=codes, lag_days=lag_days, tau=tau, times=times)


def _resolve_beta(scenario: str, strength: str) -> float:
    spec = SCENARIO_SPECS[scenario]
    if scenario == "S2":
        return -abs(INTERACTION_STRENGTHS.get(strength, abs(spec.beta_true)))
    if scenario == "S3":
        return abs(INTERACTION_STRENGTHS.get(strength, abs(spec.beta_true)))
    return float(spec.beta_true)


def build_scenario(
    *,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    splits: dict[str, list[str]],
    scenario: str,
    strength: str,
    cohort: str,
    out_dir: Path,
    data_seed: int,
    theta0: float = THETA0_DEFAULT,
) -> dict[str, Any]:
    beta = _resolve_beta(scenario, strength)
    rng = np.random.default_rng(data_seed + hash(scenario + strength + cohort) % 10_000_007)

    # Cohort filter.
    df = patients.copy()
    max_lag = CONTROLLED_MAX_SIGNAL_LAG_DAYS if cohort == "controlled" else max(SIGNAL_LAGS_DAYS)
    if cohort == "controlled":
        df = df[df["age_at_cutoff"] >= CONTROLLED_MIN_AGE].copy()
    df = df.sort_values("patient_id").reset_index(drop=True)

    # Restrict splits to patients remaining in this cohort.
    id_set = set(df["patient_id"].astype(str))
    split_map = {}
    for sp, ids in splits.items():
        split_map[sp] = [i for i in ids if i in id_set]
    pid_to_split = {p: s for s, ids in split_map.items() for p in ids}
    df = df[df["patient_id"].astype(str).isin(pid_to_split)].reset_index(drop=True)

    # Inject signals.
    signal_list: list[ExampleSignals] = []
    for row in df.itertuples(index=False):
        sig = inject_signals_for_patient(
            age=float(row.age_at_cutoff),
            cutoff=pd.Timestamp(row.cutoff_time),
            dob=pd.Timestamp(row.date_of_birth),
            rng=rng,
            max_lag_days=float(max_lag),
        )
        signal_list.append(sig)

    ages = df["age_at_cutoff"].to_numpy(dtype=np.float64)
    train_mask = df["patient_id"].astype(str).map(pid_to_split).to_numpy() == "train"

    # Target specs + bias calibration on train only.
    specs = build_target_specs(rng)
    specs = calibrate_target_biases(
        ages=ages[train_mask],
        signal_list=[signal_list[i] for i in range(len(df)) if train_mask[i]],
        specs=specs,
        scenario=scenario,
        theta0=theta0,
        beta=beta,
        rng=rng,
        target_p=TARGET_PREVALENCE,
        noise_std=NOISE_STD,
    )

    # Sample labels for all examples; store rich ground truth.
    n = len(df)
    n_t = len(specs)
    Y = np.zeros((n, n_t), dtype=np.float32)
    logits_all = np.zeros((n, n_t), dtype=np.float64)
    probs_all = np.zeros((n, n_t), dtype=np.float64)
    true_lambdas = np.zeros(n, dtype=np.float64)
    noise = rng.normal(0.0, NOISE_STD, size=(n, n_t))

    gt_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    # Background events grouped for model input construction.
    bg = events[events["patient_id"].isin(df["patient_id"])].copy()
    bg_groups = {pid: g for pid, g in bg.groupby("patient_id", sort=False)}

    for i, row in enumerate(df.itertuples(index=False)):
        pid = str(row.patient_id)
        split = pid_to_split[pid]
        age = float(row.age_at_cutoff)
        cutoff = pd.Timestamp(row.cutoff_time)
        sig = signal_list[i]
        logits, probs, R_int, lam = compute_target_logits(
            age=age,
            signals=sig,
            specs=specs,
            scenario=scenario,
            theta0=theta0,
            beta=beta,
            noise=noise[i],
        )
        y = sample_targets(probs, rng)
        Y[i] = y
        logits_all[i] = logits
        probs_all[i] = probs
        true_lambdas[i] = lam

        # Model-visible history: background (strictly before cutoff) + signals.
        hist_codes: list[str] = []
        hist_times: list[pd.Timestamp] = []
        hist_types: list[str] = []
        hist_lags: list[float] = []
        g = bg_groups.get(row.patient_id)
        if g is not None:
            g = g[g["event_timestamp"] < cutoff].sort_values("event_timestamp")
            # Keep newest background events only (cap).
            g = g.tail(64)
            for er in g.itertuples(index=False):
                ts = pd.Timestamp(er.event_timestamp)
                lag = float((cutoff - ts) / np.timedelta64(1, "D"))
                hist_codes.append(str(er.event_code))
                hist_times.append(ts)
                hist_types.append(str(er.event_type))
                hist_lags.append(lag)
        for j in range(sig.codes.size):
            hist_codes.append(str(sig.codes[j]))
            hist_times.append(pd.Timestamp(sig.times[j]))
            hist_types.append("signal")
            hist_lags.append(float(sig.lag_days[j]))

        model_rows.append(
            {
                "example_id": i,
                "patient_id": pid,
                "split": split,
                "cutoff_time": str(cutoff),
                "age_at_cutoff": age,
                "z_age": float(z_age(age)),
                "history_codes": hist_codes,
                "history_types": hist_types,
                "history_lag_days": hist_lags,
                "history_tau": tau_from_days(hist_lags).tolist() if hist_lags else [],
                "labels": y.tolist(),
            }
        )

        # Ground-truth metadata (never fed to the model).
        for j in range(sig.codes.size):
            gt_rows.append(
                {
                    "example_id": i,
                    "patient_id": pid,
                    "split": split,
                    "cutoff_time": str(cutoff),
                    "age_at_cutoff": age,
                    "z_age": float(z_age(age)),
                    "signal_event_code": str(sig.codes[j]),
                    "signal_event_time": str(pd.Timestamp(sig.times[j])),
                    "lag_days": float(sig.lag_days[j]),
                    "tau": float(sig.tau[j]),
                    "true_lambda": float(lam),
                    "true_event_relevance": float(R_int[j]) if R_int.size else float("nan"),
                    "scenario": scenario,
                    "generation_seed": data_seed,
                    "strength": strength,
                    "cohort": cohort,
                }
            )
        # Per-target ground truth (one row block).
        for k, sp in enumerate(specs):
            gt_rows.append(
                {
                    "example_id": i,
                    "patient_id": pid,
                    "split": split,
                    "cutoff_time": str(cutoff),
                    "age_at_cutoff": age,
                    "z_age": float(z_age(age)),
                    "signal_event_code": None,
                    "signal_event_time": None,
                    "lag_days": None,
                    "tau": None,
                    "true_lambda": float(lam),
                    "true_event_relevance": None,
                    "true_target_logit": float(logits[k]),
                    "true_target_probability": float(probs[k]),
                    "sampled_target": float(y[k]),
                    "target_id": k,
                    "target_name": sp["name"],
                    "target_mechanism_type": sp["mechanism"],
                    "scenario": scenario,
                    "generation_seed": data_seed,
                    "strength": strength,
                    "cohort": cohort,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist artifacts.
    pd.DataFrame(model_rows).to_parquet(out_dir / "examples.parquet", index=False)
    pd.DataFrame(gt_rows).to_parquet(out_dir / "ground_truth.parquet", index=False)
    np.savez_compressed(
        out_dir / "labels.npz",
        Y=Y,
        logits=logits_all,
        probs=probs_all,
        ages=ages,
        true_lambda=true_lambdas,
    )
    with (out_dir / "target_specs.json").open("w") as f:
        json.dump(specs, f, indent=2)
    with (out_dir / "splits.json").open("w") as f:
        json.dump(split_map, f, indent=2)

    cal = calibrate_parameters(theta0=theta0, beta=beta)
    save_calibration(out_dir / "calibration.json", cal)

    meta = {
        "scenario": scenario,
        "strength": strength,
        "cohort": cohort,
        "theta0": theta0,
        "beta_true": beta,
        "n_examples": n,
        "n_targets": n_t,
        "split_sizes": {s: len(v) for s, v in split_map.items()},
        "prevalence_mean": float(Y.mean()),
        "prevalence_per_target": Y.mean(axis=0).tolist(),
        "controlled_min_age": CONTROLLED_MIN_AGE if cohort == "controlled" else 0.0,
        "max_signal_lag_days": float(max_lag),
        "data_seed": data_seed,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    # Oracle validation.
    oracle = run_oracle_validation(
        ages=ages,
        signal_list=signal_list,
        specs=specs,
        Y=Y,
        scenario=scenario,
        theta0=theta0,
        beta=beta,
        seed=data_seed,
    )
    with (out_dir / "oracle_metrics.json").open("w") as f:
        json.dump(oracle, f, indent=2)

    ready = oracle["ready_for_training"]
    ready_path = out_dir / "READY"
    not_ready_path = out_dir / "NOT_READY"
    if ready:
        not_ready_path.unlink(missing_ok=True)
        ready_path.write_text(json.dumps(oracle["readiness_reasons"], indent=2))
    else:
        ready_path.unlink(missing_ok=True)
        not_ready_path.write_text(json.dumps(oracle["readiness_reasons"], indent=2))

    print(
        f"[{cohort}/{scenario}/{strength}] n={n} prev={Y.mean():.3f} "
        f"oracle_auroc={oracle['correct']['micro_auroc']:.3f} ready={ready}"
    )
    return {"meta": meta, "oracle": oracle, "out_dir": str(out_dir)}


def run_oracle_validation(
    *,
    ages: np.ndarray,
    signal_list: list[ExampleSignals],
    specs: list[dict[str, Any]],
    Y: np.ndarray,
    scenario: str,
    theta0: float,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 99)
    inter_idx = [i for i, sp in enumerate(specs) if sp["mechanism"] == "interaction"]

    def pack(mode: str, idx: list[int] | None = None) -> dict[str, float]:
        p = oracle_predict(
            ages=ages,
            signal_list=signal_list,
            specs=specs,
            scenario=scenario,
            theta0=theta0,
            beta=beta,
            mode=mode,
            shuffle_rng=rng,
        )
        if idx is not None:
            m = _auroc_micro_macro(Y[:, idx], p[:, idx])
            m["bce"] = _bce(Y[:, idx], p[:, idx])
        else:
            m = _auroc_micro_macro(Y, p)
            m["bce"] = _bce(Y, p)
        return m

    correct = pack("correct")
    sh_age = pack("shuffle_age")
    sh_lag = pack("shuffle_lag")
    no_int = pack("no_interaction")
    rem_age = pack("remove_age")

    # Mechanism-focused metrics (interaction targets only) for S2/S3 gating.
    correct_i = pack("correct", inter_idx) if inter_idx else correct
    sh_age_i = pack("shuffle_age", inter_idx) if inter_idx else sh_age
    sh_lag_i = pack("shuffle_lag", inter_idx) if inter_idx else sh_lag
    no_int_i = pack("no_interaction", inter_idx) if inter_idx else no_int

    d_age = sh_age["bce"] - correct["bce"]
    d_lag = sh_lag["bce"] - correct["bce"]
    d_int = no_int["bce"] - correct["bce"]
    d_rem = rem_age["bce"] - correct["bce"]
    d_age_i = sh_age_i["bce"] - correct_i["bce"]
    d_lag_i = sh_lag_i["bce"] - correct_i["bce"]
    d_int_i = no_int_i["bce"] - correct_i["bce"]

    reasons: list[str] = []
    ready = True
    if scenario == "S0":
        if abs(d_age) > 0.02:
            ready = False
            reasons.append(f"S0 expected ~0 age-shuffle BCE delta, got {d_age:.4f}")
        else:
            reasons.append(f"S0 age-shuffle BCE delta ok ({d_age:.4f})")
    elif scenario == "S1":
        if d_rem < 0.005:
            ready = False
            reasons.append(f"S1 remove-age should hurt; delta={d_rem:.4f}")
        if abs(d_int) > 0.02:
            reasons.append(f"WARN S1 no-interaction delta={d_int:.4f}")
        reasons.append(f"S1 remove-age delta={d_rem:.4f}, no-int delta={d_int:.4f}")
    elif scenario in ("S2", "S3"):
        # Gate on interaction-label metrics so null/content targets cannot dilute.
        if d_age_i < 0.015:
            ready = False
            reasons.append(f"{scenario} age-shuffle (interaction) should hurt; delta={d_age_i:.4f}")
        if d_lag_i < 0.010:
            ready = False
            reasons.append(f"{scenario} lag-shuffle (interaction) should hurt; delta={d_lag_i:.4f}")
        if d_int_i < 0.010:
            ready = False
            reasons.append(f"{scenario} no-interaction (interaction) should hurt; delta={d_int_i:.4f}")
        if correct_i["micro_auroc"] < 0.62:
            ready = False
            reasons.append(
                f"{scenario} interaction-oracle AUROC too weak: {correct_i['micro_auroc']:.3f}"
            )
        reasons.append(
            f"{scenario} interaction deltas age={d_age_i:.4f} lag={d_lag_i:.4f} "
            f"noint={d_int_i:.4f} auroc={correct_i['micro_auroc']:.3f}"
        )

    return {
        "correct": correct,
        "shuffle_age": sh_age,
        "shuffle_lag": sh_lag,
        "no_interaction": no_int,
        "remove_age": rem_age,
        "correct_interaction": correct_i,
        "shuffle_age_interaction": sh_age_i,
        "shuffle_lag_interaction": sh_lag_i,
        "no_interaction_interaction": no_int_i,
        "delta_bce_shuffle_age": d_age,
        "delta_bce_shuffle_lag": d_lag,
        "delta_bce_no_interaction": d_int,
        "delta_bce_remove_age": d_rem,
        "delta_bce_shuffle_age_interaction": d_age_i,
        "delta_bce_shuffle_lag_interaction": d_lag_i,
        "delta_bce_no_interaction_interaction": d_int_i,
        "ready_for_training": ready,
        "readiness_reasons": reasons,
        "beta_true": beta,
        "theta0": theta0,
        "scenario": scenario,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--data-seed", type=int, default=DATA_SEED)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--cohorts", nargs="+", default=["controlled", "full"])
    ap.add_argument(
        "--strengths",
        nargs="+",
        default=["medium"],
        help="S2/S3 interaction strengths to build (weak/medium/strong).",
    )
    ap.add_argument("--skip-oracle-gate", action="store_true")
    args = ap.parse_args()

    patients, events = load_background_tables()
    synth_dir = args.output_dir / "synthea"
    write_manifest(synth_dir, patients, events, reused=True)

    splits = patient_split(patients["patient_id"].to_numpy(), seed=SPLIT_SEED)
    split_path = args.output_dir / "data" / f"seed{args.data_seed}"
    split_path.mkdir(parents=True, exist_ok=True)
    with (split_path / "patient_splits.json").open("w") as f:
        json.dump(splits, f, indent=2)

    # Global calibration report for default S2 params.
    save_calibration(
        split_path / "calibration_S2_default.json",
        calibrate_parameters(theta0=THETA0_DEFAULT, beta=-2.0),
    )

    results = []
    blocked = False
    for cohort in args.cohorts:
        for scenario in args.scenarios:
            strengths = args.strengths if scenario in ("S2", "S3") else ["medium"]
            for strength in strengths:
                out_dir = (
                    split_path
                    / cohort
                    / (scenario if strength == "medium" else f"{scenario}_{strength}")
                )
                r = build_scenario(
                    patients=patients,
                    events=events,
                    splits=splits,
                    scenario=scenario,
                    strength=strength,
                    cohort=cohort,
                    out_dir=out_dir,
                    data_seed=args.data_seed,
                )
                results.append(r)
                if (
                    not args.skip_oracle_gate
                    and scenario in ("S2", "S3")
                    and strength == "medium"
                    and cohort == "controlled"
                    and not r["oracle"]["ready_for_training"]
                ):
                    blocked = True

    summary_path = split_path / "build_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", summary_path)
    if blocked:
        raise SystemExit(
            "Oracle gate failed for S2/S3 controlled medium — fix generator before training."
        )


if __name__ == "__main__":
    main()
