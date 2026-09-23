#!/usr/bin/env python3
"""Statistical identifiability baselines: age+lag vs age+lag+age×lag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import all_signal_codes, tau_from_days, z_age
from dataset import load_scenario_dir


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(int)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    out = {"bce": float(log_loss(y, p, labels=[0, 1]))}
    if len(np.unique(y)) < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    return out


def build_features(
    examples: pd.DataFrame, gt: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Code-aggregated temporal kernels ± age×lag interaction features.

    Base model features (per example):
        [z, Σ_j exp(-λ0 τ_j) · 1[code=c] for each signal code c]

    Interaction model adds:
        [Σ_j z·τ_j·exp(-λ0 τ_j) · 1[code=c] for each c]
    which is a first-order expansion of age-dependent softplus decay.
    """
    codes = list(all_signal_codes())
    code_to_i = {c: i for i, c in enumerate(codes)}
    n_c = len(codes)
    lam0 = 0.7

    sig = gt[gt["signal_event_code"].notna()].copy()
    ex = examples.sort_values("example_id").reset_index(drop=True)
    n = len(ex)
    X_base = np.zeros((n, 1 + n_c), dtype=np.float64)
    X_int = np.zeros((n, 1 + 2 * n_c), dtype=np.float64)
    eid_to_row = {int(e): i for i, e in enumerate(ex["example_id"].to_numpy())}

    for eid, g in sig.groupby("example_id"):
        i = eid_to_row.get(int(eid))
        if i is None:
            continue
        age = float(g["age_at_cutoff"].iloc[0])
        z = float(z_age(age))
        X_base[i, 0] = z
        X_int[i, 0] = z
        for row in g.itertuples(index=False):
            ci = code_to_i.get(str(row.signal_event_code))
            if ci is None:
                continue
            t = float(row.tau) if row.tau is not None else float(tau_from_days(row.lag_days))
            w = float(np.exp(-lam0 * t))
            X_base[i, 1 + ci] += w
            X_int[i, 1 + ci] += w
            X_int[i, 1 + n_c + ci] += z * t * w

    # Examples with no signals keep zeros except z from examples table.
    for i, row in enumerate(ex.itertuples(index=False)):
        if X_base[i, 0] == 0.0 and X_int[i, 0] == 0.0:
            z = float(row.z_age)
            X_base[i, 0] = z
            X_int[i, 0] = z

    splits = ex["split"].to_numpy()
    return X_base, X_int, splits


def fit_compare(
    X_base: np.ndarray,
    X_int: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
) -> dict[str, Any]:
    train = splits == "train"
    test = splits == "test"
    results: dict[str, Any] = {"targets": [], "summary": {}}
    deltas = []
    coefs = []
    auroc_base = []
    auroc_int = []

    for k in range(y.shape[1]):
        yk = y[:, k]
        if len(np.unique(yk[train])) < 2 or len(np.unique(yk[test])) < 2:
            continue

        def fit(X):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(max_iter=4000, solver="lbfgs", C=1.0),
                    ),
                ]
            )
            pipe.fit(X[train], yk[train])
            p = pipe.predict_proba(X[test])[:, 1]
            m = _metrics(yk[test], p)
            coef = pipe.named_steps["lr"].coef_.ravel()
            return m, coef

        m0, c0 = fit(X_base)
        m1, c1 = fit(X_int)
        delta = m1["auroc"] - m0["auroc"]
        deltas.append(delta)
        auroc_base.append(m0["auroc"])
        auroc_int.append(m1["auroc"])
        # Interaction block = second half of code features.
        n_inter = (X_int.shape[1] - 1) // 2
        interaction_coef = float(np.mean(c1[-n_inter:])) if n_inter > 0 else float("nan")
        coefs.append(interaction_coef)
        results["targets"].append(
            {
                "target": k,
                "base": m0,
                "interaction": m1,
                "delta_auroc": delta,
                "mean_interaction_block_coef": interaction_coef,
            }
        )

    results["summary"] = {
        "n_targets_fit": len(deltas),
        "mean_auroc_base": float(np.mean(auroc_base)) if auroc_base else float("nan"),
        "mean_auroc_interaction": float(np.mean(auroc_int)) if auroc_int else float("nan"),
        "mean_delta_auroc": float(np.mean(deltas)) if deltas else float("nan"),
        "median_delta_auroc": float(np.median(deltas)) if deltas else float("nan"),
        "mean_interaction_coef": float(np.mean(coefs)) if coefs else float("nan"),
        "frac_positive_delta": float(np.mean(np.array(deltas) > 0)) if deltas else float("nan"),
    }
    return results


def run_baselines(scenario_dir: Path) -> dict[str, Any]:
    examples, labels, meta, specs = load_scenario_dir(scenario_dir)
    gt = pd.read_parquet(scenario_dir / "ground_truth.parquet")
    X_base, X_int, splits = build_features(examples, gt)
    inter_idx = [i for i, s in enumerate(specs) if s["mechanism"] == "interaction"]
    y = labels[:, inter_idx] if inter_idx else labels
    res = fit_compare(X_base, X_int, y, splits)
    res["meta"] = {
        "scenario": meta["scenario"],
        "beta_true": meta["beta_true"],
        "n_interaction_targets": len(inter_idx),
        "feature_note": "base=z+exp(-λ0τ) code sums; interaction adds z·τ·exp(-λ0τ) code sums",
    }
    out = scenario_dir / "baseline_stats.json"
    with out.open("w") as f:
        json.dump(res, f, indent=2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", type=Path, required=True)
    args = ap.parse_args()
    res = run_baselines(args.scenario_dir)
    print(json.dumps(res["summary"], indent=2))
    print(json.dumps(res["meta"], indent=2))


if __name__ == "__main__":
    main()
