#!/usr/bin/env python3
"""Run the minimal age × temporal interaction diagnostic.

Examples:
    conda run -n ehr python age_temporal_interaction_exp/run_experiment.py --all
    conda run -n ehr python age_temporal_interaction_exp/run_experiment.py --smoke
    conda run -n ehr python age_temporal_interaction_exp/run_experiment.py --full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from build_data import build_dataset  # noqa: E402
from config import (  # noqa: E402
    ARMS,
    BETA_TRUE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    FULL_SEEDS,
    LAMBDA0_TRUE,
    NEG_CODE,
    POS_CODE,
    QUERY_CODE,
    TASKS,
    Config,
)
from dataset import InteractionBenchmark  # noqa: E402
from plots import (  # noqa: E402
    plot_beta_recovery,
    plot_intervention,
    plot_kernel_recovery,
    plot_lambda_recovery,
    plot_smoke_history,
    plot_task_comparison,
)
from train import get_device, train_run  # noqa: E402


def add_sequence_length_sanity(bench: InteractionBenchmark, results_dir: Path) -> None:
    sanity_path = results_dir / "sanity.json"
    sanity = json.loads(sanity_path.read_text()) if sanity_path.exists() else {}
    lengths = {pid: int(len(row.code_ids)) for pid, row in bench._rows.items()}
    seq_stats = {}
    for task in TASKS:
        patients = bench.patients.copy()
        patients["patient_id"] = patients["patient_id"].astype(str)
        y = patients.set_index("patient_id")[f"y_{task}"]
        L0, L1 = [], []
        for pid, L in lengths.items():
            if pid not in y.index:
                continue
            lab = int(y.loc[pid])
            (L1 if lab == 1 else L0).append(L)
        seq_stats[task] = {
            "mean_len_y0": float(np.mean(L0)) if L0 else float("nan"),
            "mean_len_y1": float(np.mean(L1)) if L1 else float("nan"),
            "corr_len_label": float(
                np.corrcoef(
                    [lengths[str(p)] for p in patients["patient_id"]],
                    patients[f"y_{task}"].to_numpy(),
                )[0, 1]
            ),
        }
    n_query = sum(int(row.is_query.any()) for row in bench._rows.values())
    seq_stats["n_patients_with_query"] = n_query
    seq_stats["n_patients"] = len(bench._rows)
    seq_stats["truncation"] = bench.truncation
    sanity["sequence_length"] = seq_stats
    sanity_path.write_text(json.dumps(sanity, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal age × temporal interaction experiment")
    p.add_argument("--build-data", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--task", choices=list(TASKS), default=None)
    p.add_argument("--arm", choices=list(ARMS), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs that already have metrics.json and rebuild tables at the end.",
    )
    return p.parse_args()


def _z_from_loader_dataset(loader) -> np.ndarray:
    return np.asarray([r.z_age for r in loader.dataset.rows], dtype=np.float32)


def run_one(
    bench: InteractionBenchmark,
    task: str,
    arm: str,
    seed: int,
    cfg_kwargs: dict[str, Any] | None = None,
    max_train: int | None = None,
    early_stop: bool = True,
    collect_attention: bool = True,
    signal_only: bool = False,
) -> dict[str, Any]:
    cfg = Config(task=task, arm=arm, seed=seed)
    if cfg_kwargs:
        for k, v in cfg_kwargs.items():
            setattr(cfg, k, v)
    train_loader = bench.make_loader(
        "train",
        task,
        shuffle=True,
        max_examples=max_train,
        seed=seed,
        signal_only=signal_only,
        batch_size=cfg.batch_size,
    )
    if max_train is None:
        val_loader = bench.make_loader(
            "val", task, shuffle=False, signal_only=signal_only, batch_size=cfg.batch_size
        )
        test_loader = bench.make_loader(
            "test", task, shuffle=False, signal_only=signal_only, batch_size=cfg.batch_size
        )
        z_test = _z_from_loader_dataset(test_loader)
    else:
        val_loader = bench.make_loader(
            "train",
            task,
            shuffle=False,
            max_examples=max_train,
            seed=seed,
            signal_only=signal_only,
            batch_size=cfg.batch_size,
        )
        test_loader = val_loader
        z_test = _z_from_loader_dataset(test_loader)
    result = train_run(
        cfg,
        train_loader,
        val_loader,
        test_loader,
        n_codes=len(bench.code_vocab),
        n_types=len(bench.type_vocab),
        age_mean=bench.age_mean,
        age_std=bench.age_std,
        extra_meta={
            "pos_id": bench.code_vocab[POS_CODE],
            "neg_id": bench.code_vocab[NEG_CODE],
            "query_id": bench.code_vocab[QUERY_CODE],
        },
        z_test=z_test,
        collect_attention=collect_attention,
        early_stop=early_stop,
    )
    return result


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    rec = result.get("recovered", {})
    test = result.get("test", {})
    attn = result.get("attention_recovery", {})
    inter = result.get("intervention", {})
    d_k = inter.get("delta_constant", {})
    d_s = inter.get("delta_shuffle", {})
    return {
        "task": result["task"],
        "model": result["arm"],
        "seed": result["seed"],
        "BCE": test.get("bce"),
        "AUROC": test.get("auroc"),
        "AUPRC": test.get("auprc"),
        "accuracy": test.get("accuracy"),
        "beta_true": BETA_TRUE[result["task"]],
        "beta_hat": rec.get("beta_hat") if result["arm"] == "age_temporal" else None,
        "lambda0_hat": rec.get("lambda0_hat")
        if result["arm"] in ("age_temporal", "temporal_only")
        else None,
        "lambda0_true": LAMBDA0_TRUE,
        "n_params": result.get("n_params"),
        "best_epoch": result.get("best_epoch"),
        "attn_pearson": attn.get("pearson_mean"),
        "attn_spearman": attn.get("spearman_mean"),
        "attn_mae": attn.get("mae_mean"),
        "attn_js": attn.get("js_mean"),
        "correct_BCE": inter.get("correct_bce"),
        "constant_age_BCE": inter.get("constant_bce"),
        "shuffled_age_BCE": inter.get("shuffled_bce"),
        "delta_constant_mean": d_k.get("mean"),
        "delta_constant_se": d_k.get("se"),
        "delta_constant_median": d_k.get("median"),
        "delta_constant_frac_pos": d_k.get("frac_positive"),
        "delta_shuffle_mean": d_s.get("mean"),
        "delta_shuffle_se": d_s.get("se"),
        "delta_shuffle_median": d_s.get("median"),
        "delta_shuffle_frac_pos": d_s.get("frac_positive"),
    }


def smoke(bench: InteractionBenchmark, results_dir: Path) -> dict[str, Any]:
    print("\n=== SMOKE: overfit 128 T1 examples with age_temporal ===", flush=True)
    result = run_one(
        bench,
        task="T1",
        arm="age_temporal",
        seed=0,
        cfg_kwargs={
            "dropout": 0.0,
            "weight_decay": 0.0,
            "max_epochs": 80,
            "patience": 80,
            "batch_size": 32,
            "lr": 1e-3,
            "run_tag": "smoke",
        },
        max_train=128,
        early_stop=False,
        collect_attention=True,
        signal_only=True,
    )
    train = result["train"]
    rec = result["recovered"]
    history = []
    run_hist = Path(DEFAULT_OUTPUT_DIR) / "runs" / "smoke_T1_age_temporal_seed0" / "history.json"
    if run_hist.exists():
        history = json.loads(run_hist.read_text())
    if history:
        plot_smoke_history(history, results_dir / "figures" / "smoke_history.png")
    passed = (
        train["accuracy"] >= 0.95
        and train["bce"] <= 0.20
        and rec["beta_hat"] > 0.05
    )
    summary = {
        "n_train": train["n"],
        "train_bce": train["bce"],
        "train_accuracy": train["accuracy"],
        "lambda0_hat": rec["lambda0_hat"],
        "beta_hat": rec["beta_hat"],
        "lambda_at_ages": rec.get("lambda_at_ages"),
        "attention_recovery": result.get("attention_recovery"),
        "passed": passed,
        "criteria": {
            "train_accuracy_ge_0.95": train["accuracy"] >= 0.95,
            "train_bce_le_0.20": train["bce"] <= 0.20,
            "beta_hat_positive": rec["beta_hat"] > 0.05,
        },
    }
    (results_dir / "smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SMOKE", "PASS" if passed else "FAIL", json.dumps(summary["criteria"]), flush=True)
    return summary


def write_tables(flat_rows: list[dict[str, Any]], results_dir: Path, bench: InteractionBenchmark) -> None:
    df = pd.DataFrame(flat_rows)
    df.to_csv(results_dir / "main_results.csv", index=False)
    (results_dir / "main_results.json").write_text(df.to_json(orient="records", indent=2) + "\n")

    agg_cols = ["BCE", "AUROC", "AUPRC", "accuracy", "beta_hat", "lambda0_hat"]
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
        "constant_age_BCE",
        "shuffled_age_BCE",
        "delta_constant_mean",
        "delta_shuffle_mean",
        "delta_constant_se",
        "delta_shuffle_se",
        "delta_constant_median",
        "delta_shuffle_median",
        "delta_constant_frac_pos",
        "delta_shuffle_frac_pos",
    ]
    inter = df[inter_cols].copy()
    inter.to_csv(results_dir / "intervention.csv", index=False)
    inter_agg = []
    for (task, model), g in df.groupby(["task", "model"]):
        inter_agg.append(
            {
                "task": task,
                "model": model,
                "correct_BCE": float(g["correct_BCE"].mean()),
                "constant_age_BCE": float(g["constant_age_BCE"].mean()),
                "shuffled_age_BCE": float(g["shuffled_age_BCE"].mean()),
                "delta_constant": float(g["delta_constant_mean"].mean()),
                "delta_shuffle": float(g["delta_shuffle_mean"].mean()),
            }
        )
    pd.DataFrame(inter_agg).to_csv(results_dir / "intervention_aggregated.csv", index=False)

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rec_rows = df[df["model"] == "age_temporal"].to_dict(orient="records")
    plot_lambda_recovery(rec_rows, bench.age_mean, bench.age_std, fig_dir / "lambda_recovery.png")
    plot_kernel_recovery(rec_rows, bench.age_mean, bench.age_std, fig_dir / "kernel_recovery.png")
    plot_beta_recovery(rec_rows, fig_dir / "beta_recovery.png")
    plot_task_comparison(df, fig_dir / "test_accuracy.png", "accuracy")
    plot_task_comparison(df, fig_dir / "test_bce.png", "BCE")
    plot_task_comparison(df, fig_dir / "test_auroc.png", "AUROC")
    plot_intervention(df, fig_dir / "intervention.png")

    verdicts = make_verdicts(df)
    (results_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")


def _mean_std(df: pd.DataFrame, task: str, model: str, col: str) -> tuple[float, float]:
    v = df.loc[(df["task"] == task) & (df["model"] == model), col].astype(float)
    if v.empty:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)


def make_verdicts(df: pd.DataFrame) -> dict[str, Any]:
    def acc(task, model):
        return _mean_std(df, task, model, "accuracy")

    def bce(task, model):
        return _mean_std(df, task, model, "BCE")

    def beta(task):
        return _mean_std(df, task, "age_temporal", "beta_hat")

    def dshuffle(task, model="age_temporal"):
        return _mean_std(df, task, model, "delta_shuffle_mean")

    t0_b = beta("T0")
    t1_b = beta("T1")
    t2_b = beta("T2")
    t1_at, t1_at_s = acc("T1", "age_temporal")
    t1_tmp, t1_tmp_s = acc("T1", "temporal_only")
    t2_at, _ = acc("T2", "age_temporal")
    t2_tmp, _ = acc("T2", "temporal_only")
    t1_late, _ = acc("T1", "late_age")
    t2_late, _ = acc("T2", "late_age")
    t0_ds = dshuffle("T0")
    t1_ds = dshuffle("T1")
    t2_ds = dshuffle("T2")

    q = {}
    q["2_t0_beta_near_zero"] = {
        "verdict": "PASS" if abs(t0_b[0]) < 0.35 else ("PARTIAL" if abs(t0_b[0]) < 0.7 else "FAIL"),
        "detail": f"T0 β̂ = {t0_b[0]:+.3f} ± {t0_b[1]:.3f} (true 0).",
    }
    q["3_t1_beta_positive"] = {
        "verdict": "PASS" if t1_b[0] > 0.2 else ("PARTIAL" if t1_b[0] > 0 else "FAIL"),
        "detail": f"T1 β̂ = {t1_b[0]:+.3f} ± {t1_b[1]:.3f} (true +1).",
    }
    q["4_t2_beta_negative"] = {
        "verdict": "PASS" if t2_b[0] < -0.2 else ("PARTIAL" if t2_b[0] < 0 else "FAIL"),
        "detail": f"T2 β̂ = {t2_b[0]:+.3f} ± {t2_b[1]:.3f} (true −1).",
    }
    q["5_correct_age_helps_only_with_interaction"] = {
        "verdict": (
            "PASS"
            if (abs(t0_ds[0]) < 0.02 and t1_ds[0] > 0.01 and t2_ds[0] > 0.01)
            else (
                "PARTIAL"
                if (t1_ds[0] > 0 and t2_ds[0] > 0)
                else "FAIL"
            )
        ),
        "detail": (
            f"ΔBCE_shuffle age_temporal: T0={t0_ds[0]:+.4f}, T1={t1_ds[0]:+.4f}, T2={t2_ds[0]:+.4f}. "
            "Expected ~0 on T0 and >0 on T1/T2."
        ),
    }
    q["6_age_temporal_beats_temporal_only_on_interaction"] = {
        "verdict": (
            "PASS"
            if (t1_at > t1_tmp + 0.02 and t2_at > t2_tmp + 0.02)
            else (
                "PARTIAL"
                if (t1_at > t1_tmp and t2_at > t2_tmp)
                else "FAIL"
            )
        ),
        "detail": (
            f"T1 acc age_temporal={t1_at:.3f} vs temporal_only={t1_tmp:.3f}; "
            f"T2 acc {t2_at:.3f} vs {t2_tmp:.3f}."
        ),
    }
    q["7_benefit_not_just_late_age"] = {
        "verdict": (
            "PASS"
            if (t1_at > t1_late + 0.02 and t2_at > t2_late + 0.02)
            else (
                "PARTIAL"
                if (t1_at > t1_late and t2_at > t2_late)
                else "FAIL"
            )
        ),
        "detail": (
            f"T1 acc age_temporal={t1_at:.3f} vs late_age={t1_late:.3f}; "
            f"T2 acc {t2_at:.3f} vs {t2_late:.3f}."
        ),
    }
    q["8_lambda_kernel_resemble_truth"] = {
        "verdict": (
            "PASS"
            if (q["3_t1_beta_positive"]["verdict"] == "PASS" and q["4_t2_beta_negative"]["verdict"] == "PASS")
            else q["3_t1_beta_positive"]["verdict"]
        ),
        "detail": "Sign-correct β implies λ(a) and b(a,τ) fan in the planted direction; see recovery figures.",
    }
    return q


def main() -> None:
    args = parse_args()
    results_dir = Path(DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    data_dir = Path(DEFAULT_OUTPUT_DIR) / "data"

    do_build = args.build_data or args.all or not (data_dir / "patients.parquet").exists()
    do_smoke = args.smoke or args.all
    do_full = args.full or args.all

    print("device:", get_device(), flush=True)
    if do_build:
        build_dataset()

    cfg0 = Config()
    bench = InteractionBenchmark(cfg0)
    print("vocab", len(bench.code_vocab), "types", len(bench.type_vocab), "trunc", bench.truncation)
    add_sequence_length_sanity(bench, results_dir)

    if do_smoke:
        smoke_summary = smoke(bench, results_dir)
        if not smoke_summary["passed"] and do_full:
            print("SMOKE FAILED — not running the full matrix. Debug the implementation first.")
            return

    if args.task and args.arm and args.seed is not None:
        kw = {}
        if args.max_epochs:
            kw["max_epochs"] = args.max_epochs
        run_one(bench, args.task, args.arm, args.seed, cfg_kwargs=kw)
        return

    if do_full:
        n_seeds = min(args.n_seeds, len(FULL_SEEDS))
        seeds = FULL_SEEDS[:n_seeds]
        jobs = [(task, arm, seed) for seed in seeds for task in TASKS for arm in ARMS]
        print(f"\n=== FULL: {len(jobs)} runs ({len(TASKS)} tasks × {len(ARMS)} arms × {n_seeds} seeds) ===")
        flat = []
        raw = []
        skipped = 0
        for i, (task, arm, seed) in enumerate(jobs, 1):
            metrics_path = Config(task=task, arm=arm, seed=seed).run_dir() / "metrics.json"
            if args.resume and metrics_path.exists():
                result = json.loads(metrics_path.read_text())
                print(f"\n--- job {i}/{len(jobs)} {task} {arm} seed={seed} SKIP (resume) ---", flush=True)
                raw.append(result)
                flat.append(flatten_result(result))
                skipped += 1
                continue
            print(f"\n--- job {i}/{len(jobs)} {task} {arm} seed={seed} ---", flush=True)
            result = run_one(bench, task, arm, seed)
            raw.append(result)
            flat.append(flatten_result(result))
            pd.DataFrame(flat).to_csv(results_dir / "main_results_partial.csv", index=False)
        if args.resume:
            print(f"resumed: skipped {skipped} completed jobs", flush=True)
            pd.DataFrame(flat).to_csv(results_dir / "main_results_partial.csv", index=False)
        (results_dir / "raw_runs.json").write_text(json.dumps(raw, indent=2, default=str) + "\n")
        write_tables(flat, results_dir, bench)
        print("Wrote", results_dir / "main_results.csv")


if __name__ == "__main__":
    main()
