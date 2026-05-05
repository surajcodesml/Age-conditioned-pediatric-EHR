#!/usr/bin/env python3
"""Diagnostic for length leakage in LOS fine-tuning cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LOS leakage using n_events_in_window only.")
    parser.add_argument("--cohort_dir", type=Path, required=True)
    return parser.parse_args()


def _load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort file: {path}")
    df = pd.read_parquet(path, columns=["label", "n_events_in_window"])
    if df.empty:
        raise RuntimeError(f"Empty cohort file: {path}")
    return df


def main() -> int:
    args = parse_args()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for this diagnostic. Install it with `pip install scikit-learn`."
        ) from exc

    train_df = _load_split(args.cohort_dir / "train_cohort.parquet")
    val_df = _load_split(args.cohort_dir / "val_cohort.parquet")

    y_train = train_df["label"].astype("int8")
    y_val = val_df["label"].astype("int8")
    if y_train.nunique() < 2:
        raise RuntimeError("Train cohort has only one class; cannot fit logistic regression.")
    if y_val.nunique() < 2:
        raise RuntimeError("Validation cohort has only one class; cannot compute AUROC.")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_df[["n_events_in_window"]].astype("float32"), y_train)
    val_probs = clf.predict_proba(val_df[["n_events_in_window"]].astype("float32"))[:, 1]
    auroc = float(roc_auc_score(y_val, val_probs))

    print(f"Validation AUROC using only n_events_in_window: {auroc:.4f}")
    if auroc <= 0.55:
        print("PASS: AUROC <= 0.55")
    elif auroc > 0.6:
        print("FAIL: AUROC > 0.6")
    else:
        print("WARN: AUROC in (0.55, 0.6]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
