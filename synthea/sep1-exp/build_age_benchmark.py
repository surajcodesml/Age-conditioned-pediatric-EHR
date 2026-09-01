#!/usr/bin/env python3
"""Build the age-signal pediatric benchmark from stock Synthea CSVs.

Reads four age-stratum CSV folders produced by ``generate_age_benchmark.sh``,
injects SIGNAL_A / SIGNAL_B, assigns S0/S1/S2 labels on the same patients,
writes patient/event tables, splits, config, and a validation report.

Does not train models.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXP_DIR / "age_benchmark_config.json"
YEAR = 365.25


def _esc(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def age_group(age_years: float, groups: list[dict]) -> str | None:
    for g in groups:
        if g["min_inclusive"] <= age_years < g["max_exclusive"]:
            return g["name"]
    return None


def find_intercept(lp: np.ndarray, noise: np.ndarray, target: float) -> float:
    lo, hi = -8.0, 6.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        mean_p = float(np.mean(sigmoid(mid + lp + noise)))
        if mean_p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def iqr(x: np.ndarray) -> tuple[float, float, float]:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(q1), float(q3)


def fmt_iqr(x: np.ndarray) -> str:
    med, q1, q3 = iqr(x)
    return f"{med:.2f} (IQR {q1:.2f}–{q3:.2f})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the age-signal pediatric benchmark.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    p.add_argument(
        "--verify-history-only",
        action="store_true",
        help="Only check that adolescent records contain early-childhood events.",
    )
    return p.parse_args()


def raw_root(cfg: dict, mode: str) -> Path:
    return EXP_DIR / cfg["output_root"] / mode / "raw"


def processed_dir(cfg: dict, mode: str) -> Path:
    return EXP_DIR / cfg["output_root"] / mode / "processed"


def load_unioned_tables(cfg: dict, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = raw_root(cfg, mode)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='12GB'")
    patient_parts: list[str] = []
    event_parts: list[str] = []
    missing: list[str] = []

    for stratum in cfg["strata"]:
        name = stratum["name"]
        csv_dir = root / name / "csv"
        patients_csv = csv_dir / "patients.csv"
        if not patients_csv.exists():
            missing.append(str(patients_csv))
            continue
        patient_parts.append(
            f"""
            SELECT
                CAST(Id AS VARCHAR) AS patient_id,
                CAST(BIRTHDATE AS TIMESTAMP) AS date_of_birth,
                TRY_CAST(DEATHDATE AS TIMESTAMP) AS death_date,
                CAST(GENDER AS VARCHAR) AS gender,
                CAST(RACE AS VARCHAR) AS race,
                '{name}' AS generation_stratum
            FROM read_csv_auto('{_esc(patients_csv)}')
            """
        )

        sources = [
            (
                "encounters.csv",
                "START",
                "ENC_",
                "encounter",
                "CODE",
                "DESCRIPTION",
            ),
            (
                "conditions.csv",
                "START",
                "COND_",
                "condition",
                "CODE",
                "DESCRIPTION",
            ),
            (
                "medications.csv",
                "START",
                "MED_",
                "medication",
                "CODE",
                "DESCRIPTION",
            ),
            (
                "observations.csv",
                "DATE",
                "OBS_",
                "observation",
                "CODE",
                "DESCRIPTION",
            ),
            (
                "procedures.csv",
                "START",
                "PROC_",
                "procedure",
                "CODE",
                "DESCRIPTION",
            ),
            (
                "immunizations.csv",
                "DATE",
                "IMM_",
                "immunization",
                "CODE",
                "DESCRIPTION",
            ),
        ]
        for fname, time_col, prefix, etype, code_col, desc_col in sources:
            path = csv_dir / fname
            if not path.exists():
                continue
            event_parts.append(
                f"""
                SELECT
                    CAST(PATIENT AS VARCHAR) AS patient_id,
                    CAST({time_col} AS TIMESTAMP) AS event_timestamp,
                    '{prefix}' || CAST({code_col} AS VARCHAR) AS event_code,
                    '{etype}' AS event_type,
                    CAST({desc_col} AS VARCHAR) AS event_description,
                    'synthea' AS source
                FROM read_csv_auto('{_esc(path)}')
                WHERE {code_col} IS NOT NULL AND {time_col} IS NOT NULL
                """
            )

    if missing:
        raise FileNotFoundError(
            "Missing Synthea CSVs:\n  " + "\n  ".join(missing)
            + "\nRun: bash synthea/sep1-exp/generate_age_benchmark.sh " + mode
        )

    patients = con.execute(" UNION ALL ".join(patient_parts)).df()
    events = con.execute(" UNION ALL ".join(event_parts)).df()
    con.close()
    patients["date_of_birth"] = pd.to_datetime(patients["date_of_birth"], utc=True).dt.tz_localize(None)
    if "death_date" in patients.columns:
        patients["death_date"] = pd.to_datetime(patients["death_date"], utc=True, errors="coerce").dt.tz_localize(None)
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True).dt.tz_localize(None)
    return patients, events


def _utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def verify_history(cfg: dict, patients: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    if "age_at_event" not in events.columns:
        tmp = events.merge(patients[["patient_id", "date_of_birth"]], on="patient_id")
        age = (tmp["event_timestamp"] - tmp["date_of_birth"]).dt.total_seconds() / 86400.0 / YEAR
        events = tmp.assign(age_at_event=age)
    ado_ids = set(patients.loc[patients["generation_stratum"] == "adolescent", "patient_id"])
    ado = events.loc[events["patient_id"].isin(ado_ids)].copy()
    ado["age_at_event_years"] = ado["age_at_event"]
    n_ado_patients = int(patients.loc[patients["generation_stratum"] == "adolescent", "patient_id"].nunique())
    n_with_infant_event = int(
        ado.loc[ado["age_at_event_years"] < 1.0, "patient_id"].nunique()
    ) if len(ado) else 0
    n_with_early = int(
        ado.loc[ado["age_at_event_years"] < 6.0, "patient_id"].nunique()
    ) if len(ado) else 0
    min_age = float(ado["age_at_event_years"].min()) if len(ado) else float("nan")
    max_age = float(ado["age_at_event_years"].max()) if len(ado) else float("nan")
    span = (
        ado.groupby("patient_id")["age_at_event_years"].agg(["min", "max"])
        if len(ado) else pd.DataFrame()
    )
    median_span = float((span["max"] - span["min"]).median()) if len(span) else float("nan")
    result = {
        "reference_date": cfg["reference_date"],
        "n_adolescent_patients": n_ado_patients,
        "n_adolescents_with_event_age_lt_1": n_with_infant_event,
        "n_adolescents_with_event_age_lt_6": n_with_early,
        "frac_adolescents_with_infant_history": (
            n_with_infant_event / n_ado_patients if n_ado_patients else float("nan")
        ),
        "adolescent_event_age_min": min_age,
        "adolescent_event_age_max": max_age,
        "adolescent_median_event_age_span_years": median_span,
        "history_complete": bool(n_ado_patients > 0 and (n_with_infant_event / n_ado_patients) >= 0.8),
    }
    return result


def build_patients(cfg: dict, patients: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ref = _utc_ts(cfg["reference_date"])
    out = patients.drop_duplicates("patient_id").copy()
    out["index_date"] = ref
    if "death_date" in out.columns:
        died = out["death_date"].notna() & (out["death_date"] < out["index_date"])
        out.loc[died, "index_date"] = out.loc[died, "death_date"]
    out["age_at_index_days"] = (out["index_date"] - out["date_of_birth"]).dt.total_seconds() / 86400.0
    out["age_at_index"] = out["age_at_index_days"] / YEAR
    out["developmental_age_group"] = out["age_at_index"].apply(lambda a: age_group(float(a), cfg["age_groups"]))
    out = out.loc[out["developmental_age_group"].notna()].copy()

    ev_stats = (
        events.groupby("patient_id", as_index=False)
        .agg(n_raw_events=("event_timestamp", "size"), first_event=("event_timestamp", "min"), last_event=("event_timestamp", "max"))
    )
    out = out.merge(ev_stats, on="patient_id", how="left")
    return out.reset_index(drop=True)


def filter_preindex_events(events: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    idx = patients[["patient_id", "index_date", "date_of_birth", "age_at_index"]]
    ev = events.merge(idx, on="patient_id", how="inner")
    ev = ev.loc[ev["event_timestamp"] < ev["index_date"]].copy()
    ev["age_at_event_days"] = (ev["event_timestamp"] - ev["date_of_birth"]).dt.total_seconds() / 86400.0
    ev["age_at_event"] = ev["age_at_event_days"] / YEAR
    ev["time_before_index_days"] = (ev["index_date"] - ev["event_timestamp"]).dt.total_seconds() / 86400.0
    return ev


def inject_signals(
    cfg: dict,
    patients: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(cfg["signal_seed"]))
    patients = patients.sort_values("patient_id").reset_index(drop=True)
    n = len(patients)
    p = float(cfg["signal_probability"])
    patients["has_SIGNAL_A"] = rng.random(n) < p
    patients["has_SIGNAL_B"] = rng.random(n) < p

    pre = events[["patient_id", "event_timestamp"]].drop_duplicates()
    pre = pre.sort_values(["patient_id", "event_timestamp"]).reset_index(drop=True)
    pre["u_a"] = rng.random(len(pre))
    pre["u_b"] = rng.random(len(pre))
    pick_a = pre.loc[pre.groupby("patient_id")["u_a"].idxmin(), ["patient_id", "event_timestamp"]]
    pick_b = pre.loc[pre.groupby("patient_id")["u_b"].idxmin(), ["patient_id", "event_timestamp"]]
    pick_a = pick_a.rename(columns={"event_timestamp": "ts_A"})
    pick_b = pick_b.rename(columns={"event_timestamp": "ts_B"})
    patients = patients.merge(pick_a, on="patient_id", how="left").merge(pick_b, on="patient_id", how="left")

    mid = patients["date_of_birth"] + (patients["index_date"] - patients["date_of_birth"]) / 2
    need_a = patients["has_SIGNAL_A"] & patients["ts_A"].isna()
    need_b = patients["has_SIGNAL_B"] & patients["ts_B"].isna()
    patients.loc[need_a, "ts_A"] = mid.loc[need_a]
    patients.loc[need_b, "ts_B"] = mid.loc[need_b]
    too_late_a = patients["has_SIGNAL_A"] & (patients["ts_A"] >= patients["index_date"])
    too_late_b = patients["has_SIGNAL_B"] & (patients["ts_B"] >= patients["index_date"])
    patients.loc[too_late_a, "ts_A"] = mid.loc[too_late_a]
    patients.loc[too_late_b, "ts_B"] = mid.loc[too_late_b]

    patients["age_at_SIGNAL_A"] = np.where(
        patients["has_SIGNAL_A"],
        (patients["ts_A"] - patients["date_of_birth"]).dt.total_seconds() / 86400.0 / YEAR,
        np.nan,
    )
    patients["age_at_SIGNAL_B"] = np.where(
        patients["has_SIGNAL_B"],
        (patients["ts_B"] - patients["date_of_birth"]).dt.total_seconds() / 86400.0 / YEAR,
        np.nan,
    )

    signal_rows = []
    for flag, code, ts_col, age_col in (
        ("has_SIGNAL_A", "SIGNAL_A", "ts_A", "age_at_SIGNAL_A"),
        ("has_SIGNAL_B", "SIGNAL_B", "ts_B", "age_at_SIGNAL_B"),
    ):
        sub = patients.loc[patients[flag]].copy()
        if sub.empty:
            continue
        signal_rows.append(
            pd.DataFrame(
                {
                    "patient_id": sub["patient_id"].to_numpy(),
                    "event_timestamp": pd.to_datetime(sub[ts_col], utc=True).dt.tz_localize(None).to_numpy(),
                    "event_code": code,
                    "event_type": "synthetic_signal",
                    "event_description": code,
                    "source": "synthetic_signal",
                    "index_date": pd.to_datetime(sub["index_date"], utc=True).dt.tz_localize(None).to_numpy(),
                    "date_of_birth": pd.to_datetime(sub["date_of_birth"], utc=True).dt.tz_localize(None).to_numpy(),
                    "age_at_index": sub["age_at_index"].to_numpy(),
                    "age_at_event": sub[age_col].to_numpy(),
                    "age_at_event_days": (sub[age_col] * YEAR).to_numpy(),
                    "time_before_index_days": (
                        (pd.to_datetime(sub["index_date"], utc=True).dt.tz_localize(None) - pd.to_datetime(sub[ts_col], utc=True).dt.tz_localize(None)).dt.total_seconds() / 86400.0
                    ).to_numpy(),
                }
            )
        )
    if signal_rows:
        events = pd.concat([events, pd.concat(signal_rows, ignore_index=True)], ignore_index=True)
    patients = patients.drop(columns=["ts_A", "ts_B"])
    return patients, events


def assign_labels(cfg: dict, patients: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(cfg["label_seed"]))
    coef = cfg["coefficients"]
    a = patients["has_SIGNAL_A"].astype(float).to_numpy()
    b = patients["has_SIGNAL_B"].astype(float).to_numpy()
    age = patients["age_at_index"].to_numpy(dtype=float)
    age_mean = float(np.mean(age))
    age_std = float(np.std(age, ddof=0))
    if age_std < 1e-8:
        age_std = 1.0
    z_age = (age - age_mean) / age_std

    age_a = np.where(patients["has_SIGNAL_A"].to_numpy(), patients["age_at_SIGNAL_A"].to_numpy(dtype=float), np.nan)
    age_b = np.where(patients["has_SIGNAL_B"].to_numpy(), patients["age_at_SIGNAL_B"].to_numpy(dtype=float), np.nan)
    g_a = sigmoid((age_a - coef["g_center_years"]) / coef["g_scale_years"])
    g_b = sigmoid((age_b - coef["g_center_years"]) / coef["g_scale_years"])
    int_a = np.where(np.isnan(age_a), 0.0, a * (1.0 - 2.0 * g_a))
    int_b = np.where(np.isnan(age_b), 0.0, -b * (1.0 - 2.0 * g_b))

    noise = rng.normal(0.0, float(cfg["noise_std"]), size=len(patients))
    target = float(cfg["target_prevalence"])

    lp0 = coef["beta_A"] * a + coef["beta_B"] * b
    lp1 = lp0 + coef["beta_age"] * z_age
    lp2 = coef["interaction_strength"] * (int_a + int_b)

    intercepts = {
        "S0": find_intercept(lp0, noise, target),
        "S1": find_intercept(lp1, noise, target),
        "S2": find_intercept(lp2, noise, target),
    }
    p0 = sigmoid(intercepts["S0"] + lp0 + noise)
    p1 = sigmoid(intercepts["S1"] + lp1 + noise)
    p2 = sigmoid(intercepts["S2"] + lp2 + noise)
    u = rng.random(len(patients))
    patients = patients.copy()
    patients["p_S0"] = p0
    patients["p_S1"] = p1
    patients["p_S2"] = p2
    patients["y_S0"] = (u < p0).astype(int)
    u1 = rng.random(len(patients))
    u2 = rng.random(len(patients))
    patients["y_S1"] = (u1 < p1).astype(int)
    patients["y_S2"] = (u2 < p2).astype(int)
    patients["z_age"] = z_age
    meta = {
        "intercepts": intercepts,
        "age_mean": age_mean,
        "age_std": age_std,
        "noise_std": float(cfg["noise_std"]),
        "target_prevalence": target,
        "realized_prevalence": {
            "S0": float(patients["y_S0"].mean()),
            "S1": float(patients["y_S1"].mean()),
            "S2": float(patients["y_S2"].mean()),
        },
    }
    return patients, meta


def assign_splits(cfg: dict, patients: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["split_seed"]))
    frac = cfg["split_fractions"]
    patients = patients.copy()
    patients["split"] = ""
    for group, idx in patients.groupby("developmental_age_group").groups.items():
        ids = patients.loc[idx, "patient_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(np.floor(frac["train"] * n))
        n_val = int(np.floor(frac["val"] * n))
        train_ids = set(ids[:n_train])
        val_ids = set(ids[n_train : n_train + n_val])
        test_ids = set(ids[n_train + n_val :])
        patients.loc[patients["patient_id"].isin(train_ids), "split"] = "train"
        patients.loc[patients["patient_id"].isin(val_ids), "split"] = "val"
        patients.loc[patients["patient_id"].isin(test_ids), "split"] = "test"
    return patients


def ground_truth_curves(cfg: dict) -> pd.DataFrame:
    coef = cfg["coefficients"]
    ages = np.linspace(0.0, 17.0, 171)
    g = sigmoid((ages - coef["g_center_years"]) / coef["g_scale_years"])
    effect_a = coef["interaction_strength"] * (1.0 - 2.0 * g)
    effect_b = -coef["interaction_strength"] * (1.0 - 2.0 * g)
    return pd.DataFrame({"age_years": ages, "effect_SIGNAL_A": effect_a, "effect_SIGNAL_B": effect_b, "g": g})


def write_s2_plot(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(curves["age_years"], curves["effect_SIGNAL_A"], label="SIGNAL_A effect")
    ax.plot(curves["age_years"], curves["effect_SIGNAL_B"], label="SIGNAL_B effect")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(8.5, color="grey", linestyle="--", linewidth=0.8, label="crossover 8.5y")
    ax.set_xlabel("Age at signal occurrence (years)")
    ax.set_ylabel("Contribution to S2 logit")
    ax.set_title("Ground-truth S2 age × event interaction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def empirical_tables(patients: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = ["<1", "1-5", "6-11", "12-17"]
    rows = []
    for g in groups:
        sub = patients.loc[patients["developmental_age_group"] == g]
        rows.append(
            {
                "age_group": g,
                "n": int(len(sub)),
                "rate_SIGNAL_A": float(sub["has_SIGNAL_A"].mean()) if len(sub) else float("nan"),
                "rate_SIGNAL_B": float(sub["has_SIGNAL_B"].mean()) if len(sub) else float("nan"),
                "prev_S0": float(sub["y_S0"].mean()) if len(sub) else float("nan"),
                "prev_S1": float(sub["y_S1"].mean()) if len(sub) else float("nan"),
                "prev_S2": float(sub["y_S2"].mean()) if len(sub) else float("nan"),
            }
        )
    by_age = pd.DataFrame(rows)

    combo_rows = []
    for task in ("S0", "S1", "S2"):
        ycol = f"y_{task}"
        for g in groups:
            for ha in (False, True):
                for hb in (False, True):
                    sub = patients.loc[
                        (patients["developmental_age_group"] == g)
                        & (patients["has_SIGNAL_A"] == ha)
                        & (patients["has_SIGNAL_B"] == hb)
                    ]
                    combo_rows.append(
                        {
                            "task": task,
                            "age_group": g,
                            "SIGNAL_A": int(ha),
                            "SIGNAL_B": int(hb),
                            "n": int(len(sub)),
                            "prevalence": float(sub[ycol].mean()) if len(sub) else float("nan"),
                        }
                    )
    by_combo = pd.DataFrame(combo_rows)
    return {"by_age": by_age, "by_combo": by_combo}


def event_effect_by_group(patients: pd.DataFrame, ycol: str) -> pd.DataFrame:
    rows = []
    for g in ["<1", "1-5", "6-11", "12-17"]:
        sub = patients.loc[patients["developmental_age_group"] == g]
        p_a1 = float(sub.loc[sub["has_SIGNAL_A"], ycol].mean()) if sub["has_SIGNAL_A"].any() else float("nan")
        p_a0 = float(sub.loc[~sub["has_SIGNAL_A"], ycol].mean()) if (~sub["has_SIGNAL_A"]).any() else float("nan")
        p_b1 = float(sub.loc[sub["has_SIGNAL_B"], ycol].mean()) if sub["has_SIGNAL_B"].any() else float("nan")
        p_b0 = float(sub.loc[~sub["has_SIGNAL_B"], ycol].mean()) if (~sub["has_SIGNAL_B"]).any() else float("nan")
        rows.append(
            {
                "age_group": g,
                "n": int(len(sub)),
                "delta_A": p_a1 - p_a0,
                "delta_B": p_b1 - p_b0,
                "prev_A1": p_a1,
                "prev_A0": p_a0,
                "prev_B1": p_b1,
                "prev_B0": p_b0,
            }
        )
    return pd.DataFrame(rows)


def leakage_checks(patients: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    future = events.loc[events["event_timestamp"] >= events["index_date"]]
    splits = {s: set(patients.loc[patients["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    codes = set(events["event_code"].astype(str).unique())
    label_like = [c for c in codes if any(x in c.lower() for x in ("y_s0", "y_s1", "y_s2", "label", "outcome"))]
    return {
        "n_events_on_or_after_index": int(len(future)),
        "train_val_overlap": int(len(splits["train"] & splits["val"])),
        "train_test_overlap": int(len(splits["train"] & splits["test"])),
        "val_test_overlap": int(len(splits["val"] & splits["test"])),
        "n_patients_in_multiple_splits": int(
            sum(1 for pid in patients["patient_id"] if sum(pid in splits[s] for s in splits) != 1)
        ),
        "synthetic_codes": sorted(c for c in codes if c in {"SIGNAL_A", "SIGNAL_B"}),
        "label_like_event_codes": label_like,
        "signal_names_contain_label": any("y_s" in c.lower() for c in ("SIGNAL_A", "SIGNAL_B")),
        "empty_split": [s for s, ids in splits.items() if len(ids) == 0],
    }


def validate(
    cfg: dict,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    history: dict[str, Any],
    label_meta: dict[str, Any],
) -> dict[str, Any]:
    groups = ["<1", "1-5", "6-11", "12-17"]
    counts = {g: int((patients["developmental_age_group"] == g).sum()) for g in groups}
    n = len(patients)
    per_pt = events.groupby("patient_id").size().reindex(patients["patient_id"], fill_value=0).to_numpy()
    duration = (
        events.groupby("patient_id")["age_at_event"].agg(lambda s: float(s.max() - s.min()))
        .reindex(patients["patient_id"])
        .fillna(0.0)
        .to_numpy()
    )
    duration_by_g = {}
    events_by_g = {}
    for g in groups:
        pids = patients.loc[patients["developmental_age_group"] == g, "patient_id"]
        ev_g = events.groupby("patient_id").size().reindex(pids, fill_value=0).to_numpy()
        dur_g = (
            events.groupby("patient_id")["age_at_event"].agg(lambda s: float(s.max() - s.min()))
            .reindex(pids)
            .fillna(0.0)
            .to_numpy()
        )
        events_by_g[g] = {"n": int(len(pids)), "median_iqr": fmt_iqr(ev_g.astype(float))}
        duration_by_g[g] = {"n": int(len(pids)), "median_iqr_years": fmt_iqr(dur_g.astype(float))}

    tables = empirical_tables(patients)
    leak = leakage_checks(patients, events)
    y = {t: patients[f"y_{t}"].to_numpy() for t in ("S0", "S1", "S2")}
    age = patients["age_at_index"].to_numpy()
    a = patients["has_SIGNAL_A"].astype(int).to_numpy()
    b = patients["has_SIGNAL_B"].astype(int).to_numpy()
    ab_score = a.astype(float) - b.astype(float)

    aucs = {}
    for t in ("S0", "S1", "S2"):
        aucs[t] = {
            "age_only": auc_score(y[t], age),
            "A_only": auc_score(y[t], a.astype(float)),
            "B_only": auc_score(y[t], b.astype(float)),
            "AB_linear": auc_score(y[t], ab_score),
        }

    s0_spread = tables["by_age"]["prev_S0"].max() - tables["by_age"]["prev_S0"].min()
    s1_age_order = tables["by_age"]["prev_S1"].to_numpy()
    s1_increasing = bool(np.all(np.diff(s1_age_order) > -0.02)) and (s1_age_order[-1] - s1_age_order[0] > 0.04)
    deltas0 = event_effect_by_group(patients, "y_S0")
    deltas1 = event_effect_by_group(patients, "y_S1")
    deltas2 = event_effect_by_group(patients, "y_S2")

    combo0 = tables["by_combo"].loc[tables["by_combo"]["task"] == "S0"]
    within_cell_ranges = []
    for ha in (0, 1):
        for hb in (0, 1):
            cell = combo0.loc[(combo0["SIGNAL_A"] == ha) & (combo0["SIGNAL_B"] == hb), "prevalence"]
            within_cell_ranges.append(float(cell.max() - cell.min()) if cell.notna().any() else float("nan"))

    a_delta_s2 = deltas2["delta_A"].to_numpy()
    s2_a_flips = bool(a_delta_s2[0] > 0.02 and a_delta_s2[-1] < -0.02)
    b_delta_s2 = deltas2["delta_B"].to_numpy()
    s2_b_flips = bool(b_delta_s2[0] < -0.02 and b_delta_s2[-1] > 0.02)

    prev = {t: float(patients[f"y_{t}"].mean()) for t in ("S0", "S1", "S2")}
    stochastic = {
        t: bool(
            0.0 < prev[t] < 1.0
            and float(patients[f"p_{t}"].min()) < 0.95
            and float(patients[f"p_{t}"].max()) > 0.05
            and patients[f"p_{t}"].nunique() > 10
        )
        for t in ("S0", "S1", "S2")
    }

    signal_rates = {
        "overall_A": float(patients["has_SIGNAL_A"].mean()),
        "overall_B": float(patients["has_SIGNAL_B"].mean()),
        "by_group_A": {g: float(tables["by_age"].loc[tables["by_age"]["age_group"] == g, "rate_SIGNAL_A"].iloc[0]) for g in groups},
        "by_group_B": {g: float(tables["by_age"].loc[tables["by_age"]["age_group"] == g, "rate_SIGNAL_B"].iloc[0]) for g in groups},
    }
    a_rates = list(signal_rates["by_group_A"].values())
    b_rates = list(signal_rates["by_group_B"].values())

    criteria = {
        "balanced_age_groups": all(c >= 0.7 * (n / 4) for c in counts.values()) and max(counts.values()) / max(min(counts.values()), 1) < 1.5,
        "complete_history": bool(history.get("history_complete")),
        "signal_independent_of_age": (max(a_rates) - min(a_rates) < 0.08) and (max(b_rates) - min(b_rates) < 0.08),
        "S0_no_age_effect": float(s0_spread) < 0.08 and float(np.nanmax(within_cell_ranges)) < 0.12,
        "S1_age_main_effect": bool(s1_increasing) and bool(np.all(deltas1["delta_A"] > 0) or np.all(np.sign(deltas1["delta_A"]) == np.sign(deltas1["delta_A"].iloc[0]))),
        "S2_interaction": bool(s2_a_flips or (a_delta_s2[0] - a_delta_s2[-1] > 0.08)),
        "S2_not_age_only": bool(aucs["S2"]["age_only"] < 0.70),
        "S2_not_AB_only": bool(aucs["S2"]["AB_linear"] < 0.70),
        "labels_stochastic": all(stochastic.values()),
        "prevalence_ok": all(0.15 <= prev[t] <= 0.30 for t in prev),
        "splits_leak_free": leak["train_val_overlap"] == 0
        and leak["train_test_overlap"] == 0
        and leak["val_test_overlap"] == 0
        and leak["n_events_on_or_after_index"] == 0
        and not leak["label_like_event_codes"]
        and not leak["signal_names_contain_label"],
    }
    return {
        "n_patients": n,
        "counts_by_age_group": counts,
        "events_per_patient": fmt_iqr(per_pt.astype(float)),
        "history_duration_years": fmt_iqr(duration.astype(float)),
        "events_per_patient_by_group": events_by_g,
        "history_duration_by_group": duration_by_g,
        "signal_rates": signal_rates,
        "prevalence": prev,
        "prevalence_by_age": tables["by_age"].to_dict(orient="records"),
        "split_counts": {s: int((patients["split"] == s).sum()) for s in ("train", "val", "test")},
        "auc": aucs,
        "s0_prevalence_spread_across_age": float(s0_spread),
        "s0_within_AB_cell_age_range_max": float(np.nanmax(within_cell_ranges)),
        "s1_prevalence_by_age": tables["by_age"]["prev_S1"].tolist(),
        "event_effects_S0": deltas0.to_dict(orient="records"),
        "event_effects_S1": deltas1.to_dict(orient="records"),
        "event_effects_S2": deltas2.to_dict(orient="records"),
        "history": history,
        "leakage": leak,
        "stochastic": stochastic,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
        "tables": tables,
    }


def write_report(path: Path, cfg: dict, mode: str, label_meta: dict, val: dict, out_files: dict) -> None:
    c = val["criteria"]
    lines = [
        "# Age-signal pediatric benchmark — validation report",
        "",
        f"- mode: `{mode}`",
        f"- reference date: `{cfg['reference_date']}`",
        f"- geography: `{cfg['geography']}`",
        f"- patients: **{val['n_patients']:,}**",
        f"- all acceptance criteria passed: **{val['all_criteria_passed']}**",
        "",
        "## Files",
        "",
    ]
    for k, v in out_files.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## A. Age coverage",
        "",
        "| group | n |",
        "|---|---:|",
    ]
    for g, n in val["counts_by_age_group"].items():
        lines.append(f"| {g} | {n:,} |")
    lines += [
        "",
        "## B. Longitudinal history",
        "",
        f"- events per patient: {val['events_per_patient']}",
        f"- history duration (years, max−min event age): {val['history_duration_years']}",
        "",
        "| group | events/patient | duration (years) |",
        "|---|---|---|",
    ]
    for g in val["counts_by_age_group"]:
        ev = val["events_per_patient_by_group"][g]["median_iqr"]
        du = val["history_duration_by_group"][g]["median_iqr_years"]
        lines.append(f"| {g} | {ev} | {du} |")
    h = val["history"]
    lines += [
        "",
        "### Complete-history check (adolescents)",
        "",
        f"- adolescents: {h['n_adolescent_patients']}",
        f"- with an event at age < 1y: {h['n_adolescents_with_event_age_lt_1']} "
        f"({h['frac_adolescents_with_infant_history']:.3f})",
        f"- with an event at age < 6y: {h['n_adolescents_with_event_age_lt_6']}",
        f"- adolescent event-age min/max: {h['adolescent_event_age_min']:.3f} / {h['adolescent_event_age_max']:.3f}",
        f"- median event-age span (years): {h['adolescent_median_event_age_span_years']:.3f}",
        f"- history_complete: {h['history_complete']}",
        "",
        "## C. SIGNAL_A / SIGNAL_B rates",
        "",
        f"- overall A: {val['signal_rates']['overall_A']:.3f}",
        f"- overall B: {val['signal_rates']['overall_B']:.3f}",
        "",
        "| group | SIGNAL_A | SIGNAL_B |",
        "|---|---:|---:|",
    ]
    for g in val["counts_by_age_group"]:
        lines.append(
            f"| {g} | {val['signal_rates']['by_group_A'][g]:.3f} | {val['signal_rates']['by_group_B'][g]:.3f} |"
        )
    lines += [
        "",
        "## D. Label prevalence",
        "",
        f"- intercepts: S0={label_meta['intercepts']['S0']:.4f}, "
        f"S1={label_meta['intercepts']['S1']:.4f}, S2={label_meta['intercepts']['S2']:.4f}",
        f"- target prevalence: {label_meta['target_prevalence']}",
        "",
        "| | overall | <1 | 1–5 | 6–11 | 12–17 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_age = {row["age_group"]: row for row in val["prevalence_by_age"]}
    for t, key in (("S0", "prev_S0"), ("S1", "prev_S1"), ("S2", "prev_S2")):
        lines.append(
            f"| {t} | {val['prevalence'][t]:.3f} | "
            + " | ".join(f"{by_age[g][key]:.3f}" for g in val["counts_by_age_group"])
            + " |"
        )
    lines += [
        "",
        "## E. H0 / H1 / H2 sanity",
        "",
        "AUROC of simple scores vs labels (not a trained model):",
        "",
        "| task | age only | A only | B only | A−B |",
        "|---|---:|---:|---:|---:|",
    ]
    for t in ("S0", "S1", "S2"):
        a = val["auc"][t]
        lines.append(
            f"| {t} | {a['age_only']:.3f} | {a['A_only']:.3f} | {a['B_only']:.3f} | {a['AB_linear']:.3f} |"
        )
    lines += [
        "",
        f"- S0 prevalence spread across age groups: {val['s0_prevalence_spread_across_age']:.3f}",
        f"- S0 max within-(A,B) cell age-group prevalence range: {val['s0_within_AB_cell_age_range_max']:.3f}",
        "",
        "S1 prevalence by age group: " + ", ".join(f"{x:.3f}" for x in val["s1_prevalence_by_age"]),
        "",
        "### Event risk differences Δ = P(y=1|signal) − P(y=1|no signal)",
        "",
    ]
    for task, key in (("S0", "event_effects_S0"), ("S1", "event_effects_S1"), ("S2", "event_effects_S2")):
        lines.append(f"**{task}**")
        lines.append("")
        lines.append("| group | ΔA | ΔB |")
        lines.append("|---|---:|---:|")
        for row in val[key]:
            lines.append(f"| {row['age_group']} | {row['delta_A']:+.3f} | {row['delta_B']:+.3f} |")
        lines.append("")
    lines += [
        "## F. Leakage",
        "",
        f"- events on/after index: {val['leakage']['n_events_on_or_after_index']}",
        f"- train/val overlap: {val['leakage']['train_val_overlap']}",
        f"- train/test overlap: {val['leakage']['train_test_overlap']}",
        f"- val/test overlap: {val['leakage']['val_test_overlap']}",
        f"- synthetic codes: {val['leakage']['synthetic_codes']}",
        f"- label-like event codes: {val['leakage']['label_like_event_codes']}",
        "",
        f"Split counts: {val['split_counts']}",
        "",
        "## Ground-truth S2 curves",
        "",
        "`g(a) = sigmoid((a − 8.5) / 2.5)`; effect_A = +(1−2g), effect_B = −(1−2g).",
        "See `s2_ground_truth_curves.csv` / `.png`.",
        "",
        "## Acceptance criteria",
        "",
        "| criterion | pass |",
        "|---|---|",
    ]
    for k, v in c.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Regeneration",
        "",
        "```bash",
        f"bash synthea/sep1-exp/generate_age_benchmark.sh {mode}",
        f"conda run -n ehr python synthea/sep1-exp/build_age_benchmark.py --mode {mode}",
        "```",
        "",
        "Coefficients (not searched): "
        + json.dumps(cfg["coefficients"]),
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    out_dir: Path,
    patients: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
    mode: str,
    label_meta: dict,
    val: dict,
    curves: pd.DataFrame,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_cols = [
        "patient_id",
        "date_of_birth",
        "index_date",
        "age_at_index",
        "developmental_age_group",
        "generation_stratum",
        "split",
        "has_SIGNAL_A",
        "has_SIGNAL_B",
        "age_at_SIGNAL_A",
        "age_at_SIGNAL_B",
        "y_S0",
        "y_S1",
        "y_S2",
        "p_S0",
        "p_S1",
        "p_S2",
    ]
    event_cols = [
        "patient_id",
        "event_timestamp",
        "age_at_event",
        "time_before_index_days",
        "event_code",
        "event_type",
        "source",
    ]
    patients_out = patients[patient_cols].sort_values(["split", "patient_id"])
    events_out = events[event_cols].sort_values(["patient_id", "event_timestamp", "event_code"])

    patients_parquet = out_dir / "patients.parquet"
    events_parquet = out_dir / "events.parquet"
    patients_csv = out_dir / "patients.csv"
    events_csv = out_dir / "events.csv"
    patients_out.to_parquet(patients_parquet, index=False)
    events_out.to_parquet(events_parquet, index=False)
    patients_out.to_csv(patients_csv, index=False)
    events_out.to_csv(events_csv, index=False)

    curves_csv = out_dir / "s2_ground_truth_curves.csv"
    curves_png = out_dir / "s2_ground_truth_curves.png"
    curves.to_csv(curves_csv, index=False)
    write_s2_plot(curves, curves_png)
    val["tables"]["by_age"].to_csv(out_dir / "prevalence_by_age.csv", index=False)
    val["tables"]["by_combo"].to_csv(out_dir / "prevalence_by_age_and_signals.csv", index=False)

    written_cfg = json.loads(json.dumps(cfg))
    written_cfg["mode"] = mode
    written_cfg["calibrated_intercepts"] = label_meta["intercepts"]
    written_cfg["age_standardization"] = {"mean": label_meta["age_mean"], "std": label_meta["age_std"]}
    written_cfg["realized_prevalence"] = label_meta["realized_prevalence"]
    written_cfg["n_patients"] = val["n_patients"]
    written_cfg["n_events"] = int(len(events_out))
    cfg_path = out_dir / "benchmark_config.json"
    cfg_path.write_text(json.dumps(written_cfg, indent=2, default=str) + "\n")

    out_files = {
        "patients_parquet": str(patients_parquet),
        "patients_csv": str(patients_csv),
        "events_parquet": str(events_parquet),
        "events_csv": str(events_csv),
        "config": str(cfg_path),
        "s2_curves_csv": str(curves_csv),
        "s2_curves_png": str(curves_png),
    }
    report_path = out_dir / "validation_report.md"
    write_report(report_path, cfg, mode, label_meta, val, {**out_files, "report": str(report_path)})
    out_files["report"] = str(report_path)
    (out_dir / "validation_summary.json").write_text(
        json.dumps({k: v for k, v in val.items() if k != "tables"}, indent=2, default=str) + "\n"
    )
    return out_files


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.config)
    print(f"Loading Synthea CSVs for mode={args.mode} ...")
    patients_raw, events_raw = load_unioned_tables(cfg, args.mode)
    print(f"  raw patients={len(patients_raw):,}  raw events={len(events_raw):,}")
    patients = build_patients(cfg, patients_raw, events_raw)
    events = filter_preindex_events(events_raw, patients)
    keep = set(events["patient_id"].unique())
    dropped = int((~patients["patient_id"].isin(keep)).sum())
    patients = patients.loc[patients["patient_id"].isin(keep)].copy()
    print(f"  kept patients with pre-index events={len(patients):,}  dropped={dropped}")

    history = verify_history(cfg, patients, events)
    print("History check (adolescents):")
    for k, v in history.items():
        print(f"  {k}: {v}")
    if args.verify_history_only:
        if not history["history_complete"]:
            print("FAIL: adolescent records do not retain early-childhood events.")
            return 1
        print("PASS: complete longitudinal history looks intact.")
        return 0

    patients, events = inject_signals(cfg, patients, events)
    patients, label_meta = assign_labels(cfg, patients)
    patients = assign_splits(cfg, patients)
    events = events.merge(patients[["patient_id", "index_date"]], on="patient_id", suffixes=("", "_p"), how="inner")
    if "index_date_p" in events.columns:
        events = events.drop(columns=["index_date_p"])

    val = validate(cfg, patients, events, history, label_meta)
    curves = ground_truth_curves(cfg)
    out_dir = processed_dir(cfg, args.mode)
    out_files = write_outputs(out_dir, patients, events, cfg, args.mode, label_meta, val, curves)

    print("=== prevalence ===")
    print(label_meta["realized_prevalence"])
    print("=== intercepts ===")
    print(label_meta["intercepts"])
    print("=== criteria ===")
    for k, v in val["criteria"].items():
        print(f"  {k}: {v}")
    print(f"all passed: {val['all_criteria_passed']}")
    print("wrote:")
    for k, v in out_files.items():
        print(f"  {k}: {v}")
    return 0 if val["all_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
