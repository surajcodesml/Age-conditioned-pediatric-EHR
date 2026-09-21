"""Patient-level leakage-safe splits for Stage-2 pediatric first-study cohort."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from preprocessing.NCH.v2 import paths as P


def _strat_key(df: pd.DataFrame) -> pd.Series:
    # Stratify on incident OSA + age band; for AHI use quantile bin when valid else 'no_ahi'
    age = df["age_band"].fillna("missing").astype(str)
    y = df["incident_osa"].fillna(0).astype(int).astype(str)
    if "ahi_valid" in df.columns and df["ahi_valid"].sum() > 20:
        ahi = df["ahi"].where(df["ahi_valid"] == 1, np.nan)
        try:
            q = pd.qcut(ahi, q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
            ahi_bin = q.astype(str).fillna("no_ahi")
        except ValueError:
            ahi_bin = pd.Series(["no_ahi"] * len(df), index=df.index)
    else:
        ahi_bin = pd.Series(["no_ahi"] * len(df), index=df.index)
    return age + "|" + y + "|" + ahi_bin.astype(str)


def make_splits(
    labels: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    *,
    seed: int = P.SPLIT_SEED,
    ratios: tuple[float, float, float] = P.SPLIT_RATIOS,
) -> dict:
    df = labels.copy()
    assert df["patient_id"].nunique() == len(df), "first-study cohort must be 1 row/patient"
    df["strat"] = _strat_key(df)
    rng = np.random.default_rng(seed)

    train_ids, val_ids, test_ids = [], [], []
    for _, g in df.groupby("strat", sort=False):
        ids = g["patient_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        # ensure at least empty-ok; fix rounding to sum to n
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        # tiny strata: assign deterministically
        if n == 1:
            train_ids.extend(ids.tolist())
        elif n == 2:
            train_ids.append(int(ids[0]))
            test_ids.append(int(ids[1]))
        else:
            train_ids.extend(ids[:n_train].tolist())
            val_ids.extend(ids[n_train:n_train + n_val].tolist())
            test_ids.extend(ids[n_train + n_val:].tolist())

    split_map = {int(i): "train" for i in train_ids}
    split_map.update({int(i): "val" for i in val_ids})
    split_map.update({int(i): "test" for i in test_ids})
    df["split"] = df["patient_id"].map(split_map)

    # Assert isolation
    sets = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert len(sets["train"] | sets["val"] | sets["test"]) == len(df)

    if meta is not None and len(meta):
        df = df.merge(
            meta[[c for c in meta.columns if c in {
                "patient_id", "history_duration", "sequence_length_before_truncation",
                "sequence_length_after_truncation", "n_unk", "frac_unk", "sex", "race",
            }]],
            on="patient_id",
            how="left",
            suffixes=("", "_meta"),
        )

    def summarize(sub: pd.DataFrame) -> dict:
        out = {
            "n_patients": int(len(sub)),
            "incident_osa_prevalence": float(sub["incident_osa"].mean()) if len(sub) else None,
            "age_years_mean": float(sub["index_age_years"].mean()) if len(sub) else None,
            "age_band_counts": sub["age_band"].value_counts().to_dict() if len(sub) else {},
            "sex_counts": sub["sex"].value_counts().to_dict() if "sex" in sub and len(sub) else {},
            "race_counts": sub["race"].value_counts().to_dict() if "race" in sub and len(sub) else {},
        }
        if "history_duration" in sub:
            out["history_duration_median"] = float(sub["history_duration"].median())
        if "sequence_length_before_truncation" in sub:
            out["seq_len_median"] = float(sub["sequence_length_before_truncation"].median())
            out["truncation_pct"] = float((sub["sequence_length_before_truncation"] > P.MAX_SEQ_LEN).mean() * 100)
        if "frac_unk" in sub:
            out["mean_frac_unk"] = float(sub["frac_unk"].mean())
        if "ahi_valid" in sub:
            out["n_ahi_valid"] = int(sub["ahi_valid"].sum())
        return out

    report = {
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "stratification": "age_band | incident_osa | ahi_quartile(if enough valid)",
        "zero_patient_overlap": True,
        "by_split": {s: summarize(df[df["split"] == s]) for s in ("train", "val", "test")},
        "patient_ids": {
            s: sorted(int(x) for x in df.loc[df["split"] == s, "patient_id"])
            for s in ("train", "val", "test")
        },
    }
    out = P.DIRS["splits"]
    df.to_parquet(out / "pediatric_first_splits.parquet", index=False)
    for s in ("train", "val", "test"):
        ids = sorted(int(x) for x in df.loc[df["split"] == s, "patient_id"])
        (out / f"{s}_patient_ids.json").write_text(json.dumps(ids) + "\n")
    # report without huge ID lists duplicated
    slim = {k: v for k, v in report.items() if k != "patient_ids"}
    slim["n_ids"] = {s: len(report["patient_ids"][s]) for s in ("train", "val", "test")}
    P.write_json(out / "splits_report.json", slim)
    P.write_json(out / "splits_patient_ids.json", report["patient_ids"])
    return report
