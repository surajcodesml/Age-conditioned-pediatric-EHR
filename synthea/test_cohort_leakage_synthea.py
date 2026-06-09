#!/usr/bin/env python3
"""STEP 4 gate: length-confound leakage check for a disease cohort.

``finetune/test_cohort_leakage.py`` targets the LANDMARK cohort schema
(``n_events_in_window`` / ``t_landmark_days``), which the disease cohorts written
by ``build_disease_cohort.py`` (columns: subject_id, label, last_event_idx) do not
have. This applies the SAME diagnostic and the SAME PASS/WARN/FAIL thresholds
(<=0.55 PASS, >0.60 FAIL) but on the disease-cohort context length
(``last_event_idx + 1``) -- the very quantity ``build_disease_cohort`` quantile-
matches between positives and negatives as its length-leakage control.

If length alone predicts the label (FAIL), any model delta on this cohort is
meaningless -- STOP and fix the cohort.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Length-confound leakage check for a disease cohort.")
    p.add_argument("--cohort_dir", type=Path, required=True)
    return p.parse_args()


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort file: {path}")
    df = pd.read_parquet(path, columns=["label", "last_event_idx"])
    if df.empty:
        raise RuntimeError(f"Empty cohort file: {path}")
    df["context_len"] = df["last_event_idx"].astype("float32") + 1.0
    return df


def main() -> int:
    args = parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    train_df = _load(args.cohort_dir / "train_cohort.parquet")
    val_df = _load(args.cohort_dir / "val_cohort.parquet")
    test_df = _load(args.cohort_dir / "test_cohort.parquet")

    y_train = train_df["label"].astype("int8")
    if y_train.nunique() < 2:
        raise RuntimeError("Train cohort has only one class; cannot fit logistic regression.")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_df[["context_len"]].astype("float32"), y_train)

    worst = 0.0
    for name, df in (("val", val_df), ("test", test_df)):
        y = df["label"].astype("int8")
        if y.nunique() < 2:
            print(f"Length-only AUROC ({name}): n/a (single class)")
            continue
        probs = clf.predict_proba(df[["context_len"]].astype("float32"))[:, 1]
        auroc = float(roc_auc_score(y, probs))
        worst = max(worst, auroc)
        print(f"Length-only AUROC ({name}): {auroc:.4f}")

    print(f"[cohort_dir] {args.cohort_dir}")
    if worst <= 0.55:
        print(f"PASS: length-only AUROC <= 0.55 (max={worst:.4f})")
        return 0
    if worst > 0.60:
        print(f"FAIL: length alone predicts the label (max={worst:.4f} > 0.60) -- STOP, fix cohort.")
        return 1
    print(f"WARN: length-only AUROC in (0.55, 0.60] (max={worst:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
