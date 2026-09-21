"""Matched-pair dataset: identical (x, τ), ages that flip the planted label.

Does not modify the original Synthea injection tables or previous result files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import (
    ARMS,
    BETA_TRUE,
    DEFAULT_MATCHED_DATA_DIR,
    DEFAULT_MATCHED_RESULTS_DIR,
    LAMBDA0_TRUE,
    MATCHED_AGE_OLD,
    MATCHED_AGE_YOUNG,
    MATCHED_MARGIN,
    MATCHED_SEED,
    Config,
)
from dataset import InteractionBenchmark, InteractionDataset, PackedPatient, collate_batch


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - float(np.max(logits))
    e = np.exp(x)
    return e / max(float(e.sum()), 1e-12)


def _age_group(age: float) -> str:
    if age < 1:
        return "<1"
    if age < 6:
        return "1-5"
    if age < 12:
        return "6-11"
    return "12-17"


def _planted_score(polarity: np.ndarray, tau: np.ndarray, z: float, beta: float) -> tuple[float, np.ndarray]:
    lam = LAMBDA0_TRUE + beta * z
    w = _softmax(-lam * tau)
    r = float((w * polarity.astype(np.float64)).sum())
    return r, w


def _copy_with_age(
    src: PackedPatient,
    *,
    z: float,
    age: float,
    labels: dict[str, float],
    true_w: dict[str, np.ndarray],
    pair_id: str,
    variant: str,
) -> PackedPatient:
    return PackedPatient(
        code_ids=src.code_ids,
        type_ids=src.type_ids,
        days_before=src.days_before,
        time_norm=src.time_norm,
        is_query=src.is_query,
        is_signal=src.is_signal,
        polarity=src.polarity,
        z_age=float(z),
        age_years=float(age),
        age_group=_age_group(age),
        patient_id=f"{pair_id}__{variant}",
        labels=labels,
        true_w=true_w,
        pair_id=str(pair_id),
        variant=variant,
    )


@dataclass
class MatchedPairBenchmark:
    src: InteractionBenchmark
    patients: pd.DataFrame
    rows_by_id: dict[str, PackedPatient]
    split_rows: dict[str, list[PackedPatient]]
    age_mean: float
    age_std: float
    generator: dict[str, Any]
    sanity: dict[str, Any]

    @property
    def code_vocab(self):
        return self.src.code_vocab

    @property
    def type_vocab(self):
        return self.src.type_vocab

    def make_loader(self, split: str, task: str, shuffle: bool, batch_size: int | None = None) -> DataLoader:
        ds = InteractionDataset(self.split_rows[split], task)
        cfg = self.src.cfg
        return DataLoader(
            ds,
            batch_size=batch_size or cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=collate_batch,
            drop_last=False,
        )


def build_matched_benchmark(
    src: InteractionBenchmark,
    *,
    age_young: float = MATCHED_AGE_YOUNG,
    age_old: float = MATCHED_AGE_OLD,
    margin: float = MATCHED_MARGIN,
    seed: int = MATCHED_SEED,
    results_dir: Path | None = None,
    data_dir: Path | None = None,
) -> MatchedPairBenchmark:
    results_dir = Path(results_dir or DEFAULT_MATCHED_RESULTS_DIR)
    data_dir = Path(data_dir or DEFAULT_MATCHED_DATA_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Two-point ages → z = ±1 after train standardization.
    age_mean = 0.5 * (age_young + age_old)
    age_std = 0.5 * abs(age_old - age_young)
    z_young = (age_young - age_mean) / age_std
    z_old = (age_old - age_mean) / age_std
    assert abs(z_young + 1.0) < 1e-8 and abs(z_old - 1.0) < 1e-8

    candidates: list[dict[str, Any]] = []
    for pid, src_row in src._rows.items():
        sig = src_row.is_signal
        if int(sig.sum()) < 2:
            continue
        pol = src_row.polarity[sig].astype(np.float64)
        tau = src_row.time_norm[sig].astype(np.float64)
        rec: dict[str, Any] = {"pair_id": pid, "src": src_row}
        keep = True
        for task, beta in (("T1", 1.0), ("T2", -1.0)):
            r_y, w_y = _planted_score(pol, tau, z_young, beta)
            r_o, w_o = _planted_score(pol, tau, z_old, beta)
            y_y = int(r_y > 0.0)
            y_o = int(r_o > 0.0)
            rec[f"y_{task}_young"] = y_y
            rec[f"y_{task}_old"] = y_o
            rec[f"r_{task}_young"] = r_y
            rec[f"r_{task}_old"] = r_o
            rec[f"w_{task}_young"] = w_y
            rec[f"w_{task}_old"] = w_o
            if y_y == y_o or abs(r_y) < margin or abs(r_o) < margin:
                keep = False
        if keep:
            rec["direction_T1"] = "young1_old0" if rec["y_T1_young"] == 1 else "young0_old1"
            candidates.append(rec)

    rng = np.random.default_rng(seed)
    by_dir = {"young1_old0": [], "young0_old1": []}
    for rec in candidates:
        by_dir[rec["direction_T1"]].append(rec)
    n_bal = min(len(by_dir["young1_old0"]), len(by_dir["young0_old1"]))
    chosen: list[dict[str, Any]] = []
    for key in by_dir:
        pool = by_dir[key]
        rng.shuffle(pool)
        chosen.extend(pool[:n_bal])
    rng.shuffle(chosen)

    # Split by base history, stratified on T1 flip direction.
    split_of: dict[str, str] = {}
    for key in ("young1_old0", "young0_old1"):
        ids = [r["pair_id"] for r in chosen if r["direction_T1"] == key]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        n_test = n - n_train - n_val
        if n_test < 1 and n > 2:
            n_test = 1
            n_train = n - n_val - n_test
        parts = (
            [("train", n_train), ("val", n_val), ("test", n_test)]
        )
        i = 0
        for name, k in parts:
            for pid in ids[i : i + k]:
                split_of[pid] = name
            i += k

    rows_by_id: dict[str, PackedPatient] = {}
    split_rows: dict[str, list[PackedPatient]] = {"train": [], "val": [], "test": []}
    meta_rows = []
    for rec in chosen:
        src_row: PackedPatient = rec["src"]
        pair_id = rec["pair_id"]
        split = split_of[pair_id]
        sig = src_row.is_signal
        for variant, age, z in (
            ("young", age_young, z_young),
            ("old", age_old, z_old),
        ):
            labels = {
                "T0": float(rec[f"y_T1_{variant}"]),
                "T1": float(rec[f"y_T1_{variant}"]),
                "T2": float(rec[f"y_T2_{variant}"]),
            }
            true_w = {}
            for task in ("T0", "T1", "T2"):
                w_full = np.zeros(len(src_row.time_norm), dtype=np.float32)
                src_task = "T1" if task == "T0" else task
                w_full[sig] = rec[f"w_{src_task}_{variant}"].astype(np.float32)
                true_w[task] = w_full
            packed = _copy_with_age(
                src_row,
                z=z,
                age=age,
                labels=labels,
                true_w=true_w,
                pair_id=pair_id,
                variant=variant,
            )
            rows_by_id[packed.patient_id] = packed
            split_rows[split].append(packed)
            meta_rows.append(
                {
                    "patient_id": packed.patient_id,
                    "pair_id": pair_id,
                    "variant": variant,
                    "split": split,
                    "age_at_index": age,
                    "z_age": z,
                    "age_mean_train": age_mean,
                    "age_std_train": age_std,
                    "y_T0": int(labels["T0"]),
                    "y_T1": int(labels["T1"]),
                    "y_T2": int(labels["T2"]),
                    "r_T1": rec[f"r_T1_{variant}"],
                    "r_T2": rec[f"r_T2_{variant}"],
                    "direction_T1": rec["direction_T1"],
                    "n_signals": int(sig.sum()),
                    "n_pos": int((src_row.polarity[sig] == 1).sum()),
                    "n_neg": int((src_row.polarity[sig] == -1).sum()),
                    "developmental_age_group": packed.age_group,
                }
            )

    patients = pd.DataFrame(meta_rows)
    # Empirical train z must match the planted ±1 scale.
    train_ages = patients.loc[patients["split"] == "train", "age_at_index"].to_numpy(dtype=float)
    emp_mean = float(train_ages.mean())
    emp_std = float(train_ages.std(ddof=0))
    if abs(emp_mean - age_mean) > 1e-6 or abs(emp_std - age_std) > 1e-4:
        raise RuntimeError(
            f"Train age stats drifted from planted z-scale: mean={emp_mean}, std={emp_std}"
        )

    sanity = _matched_sanity(patients, split_rows)
    generator = {
        "experiment": "matched_age_temporal_pairs",
        "seed": seed,
        "age_young": age_young,
        "age_old": age_old,
        "z_young": z_young,
        "z_old": z_old,
        "age_mean_train": age_mean,
        "age_std_train": age_std,
        "lambda0_true": LAMBDA0_TRUE,
        "beta_true": {"T1": 1.0, "T2": -1.0},
        "margin": margin,
        "n_source_histories": int(len(src._rows)),
        "n_flip_pre_balance": int(len(candidates)),
        "n_pairs": int(len(chosen)),
        "n_examples": int(len(patients)),
        "n_direction_young1_old0": int(n_bal),
        "n_direction_young0_old1": int(n_bal),
        "split_pairs": {
            s: int(patients.loc[patients["split"] == s, "pair_id"].nunique())
            for s in ("train", "val", "test")
        },
        "split_examples": {s: int((patients["split"] == s).sum()) for s in ("train", "val", "test")},
        "construction": (
            "For each Synthea+signal history, copy (x, τ) at ages 2y and 16y. "
            "Keep only pairs whose planted labels disagree, with |r| > margin on both ages. "
            "Balance the two age→label directions. Split by pair_id so both copies stay together. "
            "No age-dependent injection; only index age changes."
        ),
        "label_rule": "y = 1[sum_j softmax(-(λ0 + β z(a)) τ_j) x_j > 0], z(young)=-1, z(old)=+1",
    }
    patients.to_parquet(data_dir / "patients.parquet", index=False)
    (data_dir / "generator_config.json").write_text(json.dumps(generator, indent=2) + "\n")
    (results_dir / "generator_config.json").write_text(json.dumps(generator, indent=2) + "\n")
    (results_dir / "sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")
    print(
        f"Matched pairs: {len(chosen)} base histories × 2 ages = {len(patients)} examples "
        f"(train/val/test pairs {generator['split_pairs']})",
        flush=True,
    )
    print(
        f"T1 prevalence={patients['y_T1'].mean():.3f}  "
        f"corr(age,y_T1)={sanity['shortcuts']['T1']['corr_age_label']:.4f}  "
        f"age-only AUROC={sanity['shortcuts']['T1']['age_only_auroc']:.3f}",
        flush=True,
    )
    return MatchedPairBenchmark(
        src=src,
        patients=patients,
        rows_by_id=rows_by_id,
        split_rows=split_rows,
        age_mean=age_mean,
        age_std=age_std,
        generator=generator,
        sanity=sanity,
    )


def _matched_sanity(patients: pd.DataFrame, split_rows: dict[str, list[PackedPatient]]) -> dict[str, Any]:
    out: dict[str, Any] = {"shortcuts": {}, "pair_checks": {}, "notes": []}
    # Within-pair identity of codes/lags.
    by_pair: dict[str, list[PackedPatient]] = {}
    for rows in split_rows.values():
        for r in rows:
            by_pair.setdefault(r.pair_id, []).append(r)
    n_pairs = 0
    n_code_eq = 0
    n_tau_eq = 0
    n_label_diff = 0
    n_age_diff = 0
    for pid, members in by_pair.items():
        if len(members) != 2:
            continue
        n_pairs += 1
        a, b = members
        n_code_eq += int(np.array_equal(a.code_ids, b.code_ids) and np.array_equal(a.polarity, b.polarity))
        n_tau_eq += int(np.array_equal(a.time_norm, b.time_norm) and np.array_equal(a.days_before, b.days_before))
        n_label_diff += int(a.labels["T1"] != b.labels["T1"] and a.labels["T2"] != b.labels["T2"])
        n_age_diff += int(a.age_years != b.age_years)
    out["pair_checks"] = {
        "n_pairs": n_pairs,
        "frac_identical_codes": n_code_eq / max(n_pairs, 1),
        "frac_identical_lags": n_tau_eq / max(n_pairs, 1),
        "frac_label_disagreement": n_label_diff / max(n_pairs, 1),
        "frac_age_differs": n_age_diff / max(n_pairs, 1),
    }
    if n_code_eq != n_pairs or n_tau_eq != n_pairs:
        out["notes"].append("Some pairs do not share identical events/lags.")
    if n_label_diff != n_pairs:
        out["notes"].append("Some pairs do not disagree on labels.")

    for task in ("T1", "T2"):
        y = patients[f"y_{task}"].to_numpy(dtype=int)
        age = patients["age_at_index"].to_numpy(dtype=float)
        z = patients["z_age"].to_numpy(dtype=float)
        n_pos = patients["n_pos"].to_numpy(dtype=float)
        n_neg = patients["n_neg"].to_numpy(dtype=float)
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, solver="lbfgs"))]
        )
        pipe.fit(age.reshape(-1, 1), y)
        p_age = pipe.predict_proba(age.reshape(-1, 1))[:, 1]
        counts = np.stack([n_pos, n_neg], axis=1)
        pipe_c = Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, solver="lbfgs"))]
        )
        pipe_c.fit(counts, y)
        p_c = pipe_c.predict_proba(counts)[:, 1]
        out["shortcuts"][task] = {
            "n": int(len(y)),
            "prevalence": float(y.mean()),
            "corr_age_label": float(np.corrcoef(age, y)[0, 1]),
            "corr_z_label": float(np.corrcoef(z, y)[0, 1]),
            "age_only_auroc": float(roc_auc_score(y, p_age)),
            "age_only_accuracy": float(accuracy_score(y, (p_age >= 0.5).astype(int))),
            "count_only_auroc": float(roc_auc_score(y, p_c)),
            "prevalence_by_split": {
                s: float(y[(patients["split"] == s).to_numpy()].mean()) for s in ("train", "val", "test")
            },
            "prevalence_young": float(y[(patients["variant"] == "young").to_numpy()].mean()),
            "prevalence_old": float(y[(patients["variant"] == "old").to_numpy()].mean()),
        }
        if abs(out["shortcuts"][task]["corr_age_label"]) > 0.05:
            out["notes"].append(f"{task}: corr(age,y)={out['shortcuts'][task]['corr_age_label']:.3f}")
        if out["shortcuts"][task]["age_only_auroc"] > 0.55:
            out["notes"].append(f"{task}: age-only AUROC={out['shortcuts'][task]['age_only_auroc']:.3f}")
    if not out["notes"]:
        out["notes"].append(
            "Pairs share events/lags, labels disagree, and age/count shortcuts are at chance."
        )
    return out


def flatten_matched(result: dict[str, Any]) -> dict[str, Any]:
    rec = result.get("recovered", {})
    test = result.get("test", {})
    pair = test.get("pair", {})
    inter = result.get("intervention", {})
    d_s = inter.get("delta_shuffle", {})
    d_k = inter.get("delta_constant", {})
    pair_shuf = inter.get("pair_shuffled", {})
    pair_ok = inter.get("pair_correct", {})
    arm = result["arm"]
    return {
        "task": result["task"],
        "model": arm,
        "seed": result["seed"],
        "BCE": test.get("bce"),
        "AUROC": test.get("auroc"),
        "AUPRC": test.get("auprc"),
        "accuracy": test.get("accuracy"),
        "pair_accuracy": pair.get("pair_accuracy"),
        "same_prediction_rate": pair.get("same_prediction_rate"),
        "n_pairs": pair.get("n_pairs"),
        "beta_true": BETA_TRUE.get(result["task"]),
        "beta_hat": rec.get("beta_hat") if arm == "age_temporal" else None,
        "lambda0_hat": rec.get("lambda0_hat") if arm in ("age_temporal", "temporal_only") else None,
        "lambda0_true": LAMBDA0_TRUE,
        "n_params": result.get("n_params"),
        "best_epoch": result.get("best_epoch"),
        "correct_BCE": inter.get("correct_bce"),
        "constant_age_BCE": inter.get("constant_bce"),
        "shuffled_age_BCE": inter.get("shuffled_bce"),
        "delta_constant_mean": d_k.get("mean"),
        "delta_shuffle_mean": d_s.get("mean"),
        "pair_accuracy_correct_age": pair_ok.get("pair_accuracy"),
        "pair_accuracy_shuffled_age": pair_shuf.get("pair_accuracy"),
    }


def write_matched_tables(flat_rows: list[dict[str, Any]], results_dir: Path) -> dict[str, Any]:
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(flat_rows)
    df.to_csv(results_dir / "main_results.csv", index=False)
    (results_dir / "main_results.json").write_text(df.to_json(orient="records", indent=2) + "\n")

    agg_cols = [
        "BCE",
        "AUROC",
        "AUPRC",
        "accuracy",
        "pair_accuracy",
        "same_prediction_rate",
        "beta_hat",
        "lambda0_hat",
        "delta_shuffle_mean",
        "pair_accuracy_shuffled_age",
    ]
    agg_rows = []
    for (task, model), g in df.groupby(["task", "model"]):
        row: dict[str, Any] = {"task": task, "model": model, "n_seeds": int(len(g))}
        for c in agg_cols:
            v = g[c].astype(float)
            row[f"{c}_mean"] = float(v.mean()) if v.notna().any() else float("nan")
            row[f"{c}_std"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else float("nan")
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(results_dir / "main_results_aggregated.csv", index=False)

    inter_cols = [
        "task",
        "model",
        "seed",
        "correct_BCE",
        "shuffled_age_BCE",
        "delta_shuffle_mean",
        "pair_accuracy",
        "pair_accuracy_shuffled_age",
    ]
    df[inter_cols].to_csv(results_dir / "intervention.csv", index=False)

    plot_matched_bars(df, fig_dir / "matched_accuracy.png", "accuracy", "Test accuracy")
    plot_matched_bars(df, fig_dir / "matched_pair_accuracy.png", "pair_accuracy", "Matched-pair accuracy")
    plot_matched_bars(df, fig_dir / "matched_auroc.png", "AUROC", "Test AUROC")
    plot_matched_beta(df, fig_dir / "matched_beta.png")
    plot_matched_shuffle(df, fig_dir / "matched_shuffle.png")

    verdicts = make_matched_verdicts(df)
    (results_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    return verdicts


def _mean_std(df: pd.DataFrame, task: str, model: str, col: str) -> tuple[float, float]:
    v = df.loc[(df["task"] == task) & (df["model"] == model), col].astype(float)
    if v.empty:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)


def make_matched_verdicts(df: pd.DataFrame) -> dict[str, Any]:
    def acc(task, model):
        return _mean_std(df, task, model, "accuracy")

    def pair(task, model):
        return _mean_std(df, task, model, "pair_accuracy")

    def auroc(task, model):
        return _mean_std(df, task, model, "AUROC")

    def beta(task):
        return _mean_std(df, task, "age_temporal", "beta_hat")

    def dsh(task, model="age_temporal"):
        return _mean_std(df, task, model, "delta_shuffle_mean")

    q: dict[str, Any] = {}
    tasks = [t for t in ("T1", "T2") if t in set(df["task"])]
    primary = tasks[0] if tasks else "T1"

    no_auroc = auroc(primary, "no_age")
    tmp_auroc = auroc(primary, "temporal_only")
    late_auroc = auroc(primary, "late_age")
    at_auroc = auroc(primary, "age_temporal")
    no_pair = pair(primary, "no_age")
    tmp_pair = pair(primary, "temporal_only")
    late_pair = pair(primary, "late_age")
    at_pair = pair(primary, "age_temporal")
    at_acc = acc(primary, "age_temporal")
    tmp_acc = acc(primary, "temporal_only")
    late_acc = acc(primary, "late_age")
    t1_b = beta("T1") if "T1" in tasks else (float("nan"), float("nan"))
    t2_b = beta("T2") if "T2" in tasks else (float("nan"), float("nan"))
    ds = dsh(primary)

    def _chance(auc):
        return "PASS" if auc[0] < 0.60 else ("PARTIAL" if auc[0] < 0.70 else "FAIL")

    q["1_interaction_necessary"] = {
        "verdict": (
            "PASS"
            if (no_pair[0] < 0.05 and tmp_pair[0] < 0.05 and no_auroc[0] < 0.60 and tmp_auroc[0] < 0.60)
            else (
                "PARTIAL"
                if (no_pair[0] < 0.25 and tmp_pair[0] < 0.25)
                else "FAIL"
            )
        ),
        "detail": (
            f"no_age AUROC={no_auroc[0]:.3f} pair_acc={no_pair[0]:.3f}; "
            f"temporal_only AUROC={tmp_auroc[0]:.3f} pair_acc={tmp_pair[0]:.3f}. "
            "Chance AUROC/pair-acc is required if age×time is strictly necessary."
        ),
    }
    q["2_age_temporal_solves"] = {
        "verdict": (
            "PASS"
            if (at_pair[0] >= 0.80 and at_auroc[0] >= 0.90)
            else ("PARTIAL" if at_pair[0] >= 0.50 and at_auroc[0] >= 0.75 else "FAIL")
        ),
        "detail": (
            f"age_temporal acc={at_acc[0]:.3f}±{at_acc[1]:.3f}, "
            f"AUROC={at_auroc[0]:.3f}±{at_auroc[1]:.3f}, "
            f"pair_acc={at_pair[0]:.3f}±{at_pair[1]:.3f}."
        ),
    }
    q["3_temporal_only_cannot"] = {
        "verdict": _chance(tmp_auroc) if tmp_pair[0] < 0.10 else "FAIL",
        "detail": (
            f"temporal_only acc={tmp_acc[0]:.3f}, AUROC={tmp_auroc[0]:.3f}, "
            f"pair_acc={tmp_pair[0]:.3f} (expected ~chance / pair_acc≈0)."
        ),
    }
    q["4_late_age_indirect"] = {
        "verdict": "REPORT",
        "detail": (
            f"late_age acc={late_acc[0]:.3f}±{late_acc[1]:.3f}, "
            f"AUROC={late_auroc[0]:.3f}±{late_auroc[1]:.3f}, "
            f"pair_acc={late_pair[0]:.3f}±{late_pair[1]:.3f}. "
            "Additive age at the head cannot implement a history-dependent flip; "
            "this is the measured result, not an assumed failure."
        ),
    }
    sign_ok = True
    sign_detail = []
    if "T1" in tasks:
        ok = t1_b[0] > 0.2
        sign_ok = sign_ok and ok
        sign_detail.append(f"T1 β̂={t1_b[0]:+.3f}±{t1_b[1]:.3f} (true +1)")
    if "T2" in tasks:
        ok = t2_b[0] < -0.2
        sign_ok = sign_ok and ok
        sign_detail.append(f"T2 β̂={t2_b[0]:+.3f}±{t2_b[1]:.3f} (true −1)")
    q["5_beta_sign"] = {
        "verdict": "PASS" if sign_ok else "FAIL",
        "detail": "; ".join(sign_detail) if sign_detail else "no age_temporal rows",
    }
    q["6_shuffle_breaks_age_temporal"] = {
        "verdict": "PASS" if ds[0] > 0.2 else ("PARTIAL" if ds[0] > 0.05 else "FAIL"),
        "detail": (
            f"age_temporal ΔBCE_shuffle={ds[0]:+.3f}±{ds[1]:.3f}; "
            f"pair_acc shuffled="
            f"{_mean_std(df, primary, 'age_temporal', 'pair_accuracy_shuffled_age')[0]:.3f}."
        ),
    }
    q["7_beats_temporal_only"] = {
        "verdict": (
            "PASS"
            if (at_pair[0] > tmp_pair[0] + 0.50 and at_acc[0] > tmp_acc[0] + 0.20)
            else ("PARTIAL" if at_acc[0] > tmp_acc[0] else "FAIL")
        ),
        "detail": (
            f"age_temporal acc={at_acc[0]:.3f} pair_acc={at_pair[0]:.3f} vs "
            f"temporal_only acc={tmp_acc[0]:.3f} pair_acc={tmp_pair[0]:.3f}."
        ),
    }
    return q


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_matched_bars(df: pd.DataFrame, path: Path, metric: str, title: str) -> None:
    tasks = [t for t in ("T1", "T2") if t in set(df["task"])]
    fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(4.2 * max(len(tasks), 1) + 1.5, 3.8), squeeze=False)
    arms = list(ARMS)
    colors = ["#7a7a7a", "#5b8fa8", "#c47b3b", "#1f4e79"]
    for ax, task in zip(axes[0], tasks):
        sub = df[df["task"] == task]
        means, stds = [], []
        for arm in arms:
            v = sub.loc[sub["model"] == arm, metric].astype(float)
            means.append(float(v.mean()) if v.notna().any() else np.nan)
            stds.append(float(v.std(ddof=1)) if v.notna().sum() > 1 else 0.0)
        ax.bar(np.arange(len(arms)), means, yerr=stds, color=colors, capsize=3)
        ax.set_xticks(np.arange(len(arms)), ["no age", "temporal", "late age", "age×time"], rotation=25, ha="right")
        ax.set_title(f"{task}  β*={BETA_TRUE[task]:+.0f}")
        ax.set_ylim(0.0, 1.05)
        ax.axhline(0.5, color="0.7", ls="--", lw=1)
        _style(ax)
    axes[0][0].set_ylabel(title)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_matched_beta(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["model"] == "age_temporal"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    rng = np.random.default_rng(0)
    colors = {"T1": "#1f4e79", "T2": "#b35c1e"}
    xticks, labels = [], []
    for i, task in enumerate(("T1", "T2")):
        vals = sub.loc[sub["task"] == task, "beta_hat"].astype(float).dropna().to_numpy()
        if vals.size == 0:
            continue
        x = np.full(vals.size, i, dtype=float) + rng.uniform(-0.08, 0.08, size=vals.size)
        ax.scatter(x, vals, color=colors[task], s=42, zorder=3)
        ax.hlines(BETA_TRUE[task], i - 0.25, i + 0.25, colors=colors[task], lw=2)
        xticks.append(i)
        labels.append(f"{task} (β*={BETA_TRUE[task]:+.0f})")
    ax.axhline(0.0, color="0.7", lw=1)
    ax.set_xticks(xticks, labels)
    ax.set_ylabel(r"learned $\hat\beta$")
    ax.set_title("Matched-pair interaction coefficient")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_matched_shuffle(df: pd.DataFrame, path: Path) -> None:
    tasks = [t for t in ("T1", "T2") if t in set(df["task"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    arms = list(ARMS)
    colors = ["#7a7a7a", "#5b8fa8", "#c47b3b", "#1f4e79"]
    ax = axes[0]
    width = 0.35
    x = np.arange(len(arms))
    for j, task in enumerate(tasks[:2] or ["T1"]):
        sub = df[df["task"] == task]
        means = [
            float(sub.loc[sub["model"] == arm, "delta_shuffle_mean"].astype(float).mean())
            if (sub["model"] == arm).any()
            else np.nan
            for arm in arms
        ]
        ax.bar(x + (j - 0.5) * width, means, width=width, color=colors if len(tasks) == 1 else None, label=task, alpha=0.9)
    ax.axhline(0.0, color="0.5", lw=1)
    ax.set_xticks(x, ["no age", "temporal", "late age", "age×time"], rotation=25, ha="right")
    ax.set_ylabel("ΔBCE shuffled age")
    ax.set_title("Cost of shuffling age")
    if len(tasks) > 1:
        ax.legend(frameon=False)
    _style(ax)

    ax = axes[1]
    for j, task in enumerate(tasks[:2] or ["T1"]):
        sub = df[df["task"] == task]
        means = [
            float(sub.loc[sub["model"] == arm, "pair_accuracy_shuffled_age"].astype(float).mean())
            if (sub["model"] == arm).any()
            else np.nan
            for arm in arms
        ]
        ax.bar(x + (j - 0.5) * width, means, width=width, label=task, alpha=0.9)
    ax.set_xticks(x, ["no age", "temporal", "late age", "age×time"], rotation=25, ha="right")
    ax.set_ylabel("Pair accuracy after shuffle")
    ax.set_title("Matched-pair accuracy, shuffled age")
    ax.set_ylim(0.0, 1.05)
    if len(tasks) > 1:
        ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
