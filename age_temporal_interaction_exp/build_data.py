#!/usr/bin/env python3
"""Inject age-independent signal events into Synthea histories and plant T0/T1/T2 labels.

Does not modify the original sep1-exp tables or production DKM code.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    AGE_GROUPS,
    BETA_TRUE,
    DATA_SEED,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SYNTHEA_DIR,
    GAP_FAR,
    GAP_MAX_DAYS,
    GAP_MIN_DAYS,
    GAP_NEAR,
    LAMBDA0_TRUE,
    MIN_HISTORY_DAYS,
    NEG_CODE,
    N_SIG_MAX,
    N_SIG_MIN,
    POS_CODE,
    QUERY_CODE,
    QUERY_TYPE,
    SIGNAL_TYPE,
    TAU_WEEK_SCALE,
    tau_from_days,
    tau_max,
)

YEAR = 365.25


def _softmax_masked(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    fill = np.where(valid, logits, -1.0e9)
    m = fill.max(axis=1, keepdims=True)
    e = np.exp(fill - m) * valid.astype(np.float64)
    z = e.sum(axis=1, keepdims=True)
    z = np.clip(z, 1e-12, None)
    return e / z


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _fit_auroc(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    pipe.fit(x, y)
    p = pipe.predict_proba(x)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": _auroc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "coef": [float(c) for c in pipe.named_steps["lr"].coef_.ravel()],
        "intercept": float(pipe.named_steps["lr"].intercept_[0]),
    }


def build_dataset(
    synthea_dir: Path | None = None,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
    seed: int = DATA_SEED,
) -> dict:
    synthea_dir = Path(synthea_dir or DEFAULT_SYNTHEA_DIR)
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    results_dir = Path(results_dir or DEFAULT_RESULTS_DIR)
    data_dir = output_dir / "data"
    fig_dir = results_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    patients = pd.read_parquet(synthea_dir / "patients.parquet")
    events = pd.read_parquet(synthea_dir / "events.parquet")

    hist = events.groupby("patient_id")["time_before_index_days"].max()
    elig_ids = set(hist[hist >= MIN_HISTORY_DAYS].index)
    elig = patients[patients["patient_id"].isin(elig_ids)].copy()
    elig = elig.sort_values("patient_id").reset_index(drop=True)
    n = len(elig)
    print(f"Eligible patients (history >= {MIN_HISTORY_DAYS:.0f}d): {n} / {len(patients)}")

    rng = np.random.default_rng(seed)
    # Even signal counts keep n_pos = n_neg so polarity counts cannot solve the task.
    n_signals = rng.choice(np.array([4, 6, 8], dtype=np.int64), size=n)
    max_k = int(N_SIG_MAX)
    valid = np.arange(max_k)[None, :] < n_signals[:, None]
    polarities = np.zeros((n, max_k), dtype=np.int8)
    for i in range(n):
        k_i = int(n_signals[i])
        half = k_i // 2
        pol = np.concatenate(
            [np.ones(half, dtype=np.int8), -np.ones(half, dtype=np.int8)]
        )
        rng.shuffle(pol)
        polarities[i, :k_i] = pol
    # Bimodal lags, shuffled independently of polarity and age: half near, half far.
    gaps = np.zeros((n, max_k), dtype=np.float64)
    for i in range(n):
        k_i = int(n_signals[i])
        half = k_i // 2
        near = rng.uniform(GAP_NEAR[0], GAP_NEAR[1], size=half)
        far = rng.uniform(GAP_FAR[0], GAP_FAR[1], size=k_i - half)
        g = np.concatenate([near, far])
        rng.shuffle(g)
        gaps[i, :k_i] = g

    ages = elig["age_at_index"].to_numpy(dtype=np.float64)
    train_mask = (elig["split"] == "train").to_numpy()
    age_mean = float(ages[train_mask].mean())
    age_std = float(ages[train_mask].std(ddof=0))
    age_std = max(age_std, 1e-6)
    z_age = (ages - age_mean) / age_std

    tau = tau_from_days(gaps)
    tau = np.where(valid, tau, 0.0)

    labels = {}
    scores = {}
    weights = {}
    lambdas = {}
    for task, beta in BETA_TRUE.items():
        lam = LAMBDA0_TRUE + beta * z_age
        logits = -lam[:, None] * tau
        w = _softmax_masked(logits, valid)
        r = (w * polarities.astype(np.float64)).sum(axis=1)
        y = (r > 0.0).astype(np.int64)
        labels[task] = y
        scores[task] = r
        weights[task] = w
        lambdas[task] = lam
        print(
            f"  {task}: beta*={beta:+.1f}  prevalence={y.mean():.4f}  "
            f"lambda range [{lam.min():.3f}, {lam.max():.3f}]"
        )
    print(
        "  label agreement  T0=T1 {:.3f}  T0=T2 {:.3f}  T1=T2 {:.3f}".format(
            float((labels["T0"] == labels["T1"]).mean()),
            float((labels["T0"] == labels["T2"]).mean()),
            float((labels["T1"] == labels["T2"]).mean()),
        )
    )

    elig["z_age"] = z_age
    elig["n_signals"] = n_signals
    elig["n_pos"] = (polarities == 1).sum(axis=1)
    elig["n_neg"] = (polarities == -1).sum(axis=1)
    elig["age_mean_train"] = age_mean
    elig["age_std_train"] = age_std
    for task in BETA_TRUE:
        elig[f"y_{task}"] = labels[task]
        elig[f"r_{task}"] = scores[task]
        elig[f"lambda_{task}"] = lambdas[task]

    # Injected events
    index_dates = pd.to_datetime(elig["index_date"]).to_numpy()
    dobs = pd.to_datetime(elig["date_of_birth"]).to_numpy()
    rows = []
    sig_rows = []
    for i in range(n):
        pid = elig.iloc[i]["patient_id"]
        idx_date = pd.Timestamp(index_dates[i])
        dob = pd.Timestamp(dobs[i])
        rows.append(
            {
                "patient_id": pid,
                "event_timestamp": idx_date,
                "age_at_event": float(ages[i]),
                "time_before_index_days": 0.0,
                "event_code": QUERY_CODE,
                "event_type": QUERY_TYPE,
                "source": "age_temporal_injected",
                "polarity": 0,
                "is_query": True,
                "is_signal": False,
            }
        )
        k_i = int(n_signals[i])
        for k in range(k_i):
            gap = float(gaps[i, k])
            pol = int(polarities[i, k])
            ts = idx_date - pd.Timedelta(days=gap)
            age_e = (ts - dob).total_seconds() / 86400.0 / YEAR
            code = POS_CODE if pol == 1 else NEG_CODE
            rows.append(
                {
                    "patient_id": pid,
                    "event_timestamp": ts,
                    "age_at_event": float(age_e),
                    "time_before_index_days": gap,
                    "event_code": code,
                    "event_type": SIGNAL_TYPE,
                    "source": "age_temporal_injected",
                    "polarity": pol,
                    "is_query": False,
                    "is_signal": True,
                }
            )
            rec = {
                "patient_id": pid,
                "signal_idx": k,
                "polarity": pol,
                "gap_days": gap,
                "tau": float(tau[i, k]),
            }
            for task in BETA_TRUE:
                rec[f"w_{task}"] = float(weights[task][i, k])
            sig_rows.append(rec)

    inj = pd.DataFrame(rows)
    signals = pd.DataFrame(sig_rows)

    keep_cols = [
        "patient_id",
        "event_timestamp",
        "age_at_event",
        "time_before_index_days",
        "event_code",
        "event_type",
        "source",
    ]
    bg = events[keep_cols].copy()
    bg["polarity"] = 0
    bg["is_query"] = False
    bg["is_signal"] = False
    elig_id_set = set(elig["patient_id"])
    bg = bg[bg["patient_id"].isin(elig_id_set)]
    events_out = pd.concat([bg, inj], ignore_index=True)
    events_out = events_out.sort_values(
        ["patient_id", "event_timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    patients_out = elig.copy()
    patients_out.to_parquet(data_dir / "patients.parquet", index=False)
    events_out.to_parquet(data_dir / "events.parquet", index=False)
    signals.to_parquet(data_dir / "signals.parquet", index=False)

    sanity = compute_sanity(patients_out, signals)
    (results_dir / "sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")
    pd.DataFrame(sanity["prevalence_rows"]).to_csv(
        results_dir / "sanity_prevalence.csv", index=False
    )

    generator = {
        "seed": seed,
        "n_total_synthea": int(len(patients)),
        "n_eligible": n,
        "min_history_days": MIN_HISTORY_DAYS,
        "gap_min_days": GAP_MIN_DAYS,
        "gap_max_days": GAP_MAX_DAYS,
        "gap_near_days": list(GAP_NEAR),
        "gap_far_days": list(GAP_FAR),
        "gap_sampling": "half Uniform[7,14], half Uniform[60,90], then shuffled independently of polarity and age",
        "tau_week_scale": TAU_WEEK_SCALE,
        "tau_max": tau_max(),
        "tau_formula": "tau = clip( log(1 + dt_days / 7) / log(1 + 90 / 7), 0, 1 )",
        "lambda0_true": LAMBDA0_TRUE,
        "beta_true": BETA_TRUE,
        "n_signals_range": [4, 6, 8],
        "n_signals_note": "Even counts only, with n_pos = n_neg, so polarity counts cannot label the task.",
        "age_mean_train": age_mean,
        "age_std_train": age_std,
        "split_counts": {s: int((elig["split"] == s).sum()) for s in ("train", "val", "test")},
        "age_group_counts": {
            g: int((elig["developmental_age_group"] == g).sum()) for g in AGE_GROUPS
        },
        "eligible_by_split_age": {
            s: {
                g: int(
                    ((elig["split"] == s) & (elig["developmental_age_group"] == g)).sum()
                )
                for g in AGE_GROUPS
            }
            for s in ("train", "val", "test")
        },
        "label_rule": "y = 1[sum_j softmax(-lambda(a) * tau_j) * x_j > 0], x in {+1,-1}",
        "injection": "POS_SIGNAL/NEG_SIGNAL sampled independently of age; Synthea events kept as distractors",
    }
    (data_dir / "generator_config.json").write_text(json.dumps(generator, indent=2) + "\n")
    (results_dir / "generator_config.json").write_text(json.dumps(generator, indent=2) + "\n")

    _plot_lambda_true(fig_dir, age_mean, age_std)
    _plot_sanity(fig_dir, patients_out, signals)
    print(f"Wrote data to {data_dir}")
    print(f"Wrote sanity to {results_dir / 'sanity.json'}")
    return {"generator": generator, "sanity": sanity, "data_dir": str(data_dir)}


def compute_sanity(patients: pd.DataFrame, signals: pd.DataFrame) -> dict:
    out: dict = {"prevalence_rows": [], "shortcuts": {}, "notes": []}
    for task in BETA_TRUE:
        y = patients[f"y_{task}"].to_numpy(dtype=int)
        age = patients["age_at_index"].to_numpy(dtype=float)
        z = patients["z_age"].to_numpy(dtype=float)
        n_pos = patients["n_pos"].to_numpy(dtype=float)
        n_neg = patients["n_neg"].to_numpy(dtype=float)
        n_sig = patients["n_signals"].to_numpy(dtype=float)
        r = patients[f"r_{task}"].to_numpy(dtype=float)

        row = {
            "task": task,
            "n": int(len(y)),
            "prevalence": float(y.mean()),
            "corr_age_label": float(np.corrcoef(age, y)[0, 1]),
            "corr_z_label": float(np.corrcoef(z, y)[0, 1]),
            "corr_npos_label": float(np.corrcoef(n_pos, y)[0, 1]),
            "corr_nneg_label": float(np.corrcoef(n_neg, y)[0, 1]),
            "corr_nsig_label": float(np.corrcoef(n_sig, y)[0, 1]),
        }
        for g in AGE_GROUPS:
            m = patients["developmental_age_group"] == g
            row[f"prevalence_{g}"] = float(y[m.to_numpy()].mean()) if m.any() else float("nan")
            row[f"n_{g}"] = int(m.sum())
        out["prevalence_rows"].append(row)

        merged = signals.merge(
            patients[["patient_id", "developmental_age_group", "split", f"y_{task}"]],
            on="patient_id",
            how="left",
        )
        lag_by_age = {}
        for g in AGE_GROUPS:
            sub = merged[merged["developmental_age_group"] == g]
            lag_by_age[g] = {
                "n": int(len(sub)),
                "gap_mean": float(sub["gap_days"].mean()) if len(sub) else float("nan"),
                "gap_median": float(sub["gap_days"].median()) if len(sub) else float("nan"),
                "tau_mean": float(sub["tau"].mean()) if len(sub) else float("nan"),
            }

        y_pos = y == 1
        npos_by_y = {
            "y0_mean_n_pos": float(n_pos[~y_pos].mean()) if (~y_pos).any() else float("nan"),
            "y1_mean_n_pos": float(n_pos[y_pos].mean()) if y_pos.any() else float("nan"),
            "y0_mean_n_neg": float(n_neg[~y_pos].mean()) if (~y_pos).any() else float("nan"),
            "y1_mean_n_neg": float(n_neg[y_pos].mean()) if y_pos.any() else float("nan"),
            "y0_mean_n_sig": float(n_sig[~y_pos].mean()) if (~y_pos).any() else float("nan"),
            "y1_mean_n_sig": float(n_sig[y_pos].mean()) if y_pos.any() else float("nan"),
        }

        counts = np.stack([n_pos, n_neg], axis=1)
        out["shortcuts"][task] = {
            "age_only": _fit_auroc(age, y),
            "z_only": _fit_auroc(z, y),
            "signal_count_only": _fit_auroc(counts, y),
            "n_signals_only": _fit_auroc(n_sig, y),
            "n_pos_minus_n_neg": _fit_auroc(n_pos - n_neg, y),
            "oracle_r": {
                "auroc": _auroc(y, r),
                "accuracy": float(((r > 0) == (y == 1)).mean()),
            },
            "lag_by_age": lag_by_age,
            "counts_by_label": npos_by_y,
            "prevalence_by_split": {
                s: float(y[(patients["split"] == s).to_numpy()].mean())
                for s in ("train", "val", "test")
            },
        }

        age_auc = out["shortcuts"][task]["age_only"]["auroc"]
        count_auc = out["shortcuts"][task]["signal_count_only"]["auroc"]
        if task in ("T1", "T2") and age_auc > 0.60:
            out["notes"].append(
                f"{task}: age-only AUROC={age_auc:.3f} is higher than expected; check leakage."
            )
        if abs(row["corr_age_label"]) > 0.15:
            out["notes"].append(
                f"{task}: corr(age, y)={row['corr_age_label']:.3f} is larger than the target ~0."
            )
        if count_auc > 0.65:
            out["notes"].append(
                f"{task}: POS/NEG count AUROC={count_auc:.3f} is a shortcut; rebalance polarities."
            )
    if not out["notes"]:
        out["notes"].append("No obvious age or count shortcut detected by the configured thresholds.")
    out["label_agreement"] = {
        "T0_eq_T1": float((patients["y_T0"] == patients["y_T1"]).mean()),
        "T0_eq_T2": float((patients["y_T0"] == patients["y_T2"]).mean()),
        "T1_eq_T2": float((patients["y_T1"] == patients["y_T2"]).mean()),
    }
    return out


def _plot_lambda_true(fig_dir: Path, age_mean: float, age_std: float) -> None:
    ages = np.linspace(0.0, 18.0, 400)
    z = (ages - age_mean) / age_std
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for task, beta, color in (
        ("T0", 0.0, "#4c4c4c"),
        ("T1", 1.0, "#1f4e79"),
        ("T2", -1.0, "#b35c1e"),
    ):
        ax.plot(ages, LAMBDA0_TRUE + beta * z, color=color, lw=2.0, label=fr"{task}: $\beta^*={beta:+.0f}$")
    ax.set_xlabel("Age at index (years)")
    ax.set_ylabel(r"$\lambda^*(a)=\lambda_0+\beta^* z(a)$")
    ax.set_title("Planted age-dependent temporal slopes")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "lambda_true.png", dpi=150)
    plt.close(fig)

    tau = np.linspace(0.0, 1.0, 200)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, task, beta in zip(axes, ("T0", "T1", "T2"), (0.0, 1.0, -1.0)):
        for age, ls in ((2.0, "-"), (10.0, "--"), (17.0, ":")):
            z_a = (age - age_mean) / age_std
            lam = LAMBDA0_TRUE + beta * z_a
            ax.plot(tau, -lam * tau, ls=ls, label=f"age={age:.0f} (λ={lam:.2f})")
        ax.set_title(f"{task}  β*={beta:+.0f}")
        ax.set_xlabel(r"$\tau$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$b^*(a,\tau)=-\lambda^*(a)\,\tau$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Planted temporal kernels", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "kernel_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sanity(fig_dir: Path, patients: pd.DataFrame, signals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, task in zip(axes, BETA_TRUE):
        rows = []
        for g in AGE_GROUPS:
            m = patients["developmental_age_group"] == g
            rows.append((g, float(patients.loc[m, f"y_{task}"].mean())))
        ax.bar([r[0] for r in rows], [r[1] for r in rows], color="#1f4e79")
        ax.axhline(0.5, color="0.6", ls="--", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{task} prevalence by age")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "prevalence_by_age.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for g in AGE_GROUPS:
        pids = set(patients.loc[patients["developmental_age_group"] == g, "patient_id"])
        gaps = signals.loc[signals["patient_id"].isin(pids), "gap_days"]
        ax.hist(gaps, bins=30, histtype="step", density=True, label=g, lw=1.6)
    ax.set_xlabel("Injected signal lag (days)")
    ax.set_ylabel("Density")
    ax.set_title("Signal lag distribution is age-independent")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "signal_lag_by_age.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    build_dataset()
