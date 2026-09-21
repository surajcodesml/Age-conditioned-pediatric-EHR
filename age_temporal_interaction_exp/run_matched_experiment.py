#!/usr/bin/env python3
"""Matched-pair age × temporal experiment.

Identical (x, τ); only age changes; labels disagree. Previous experiment
artifacts under results/*.csv are not written or overwritten.
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

from config import (  # noqa: E402
    ARMS,
    DEFAULT_MATCHED_RESULTS_DIR,
    FULL_SEEDS,
    MATCHED_TASKS,
    NEG_CODE,
    POS_CODE,
    QUERY_CODE,
    Config,
)
from dataset import InteractionBenchmark  # noqa: E402
from matched_pairs import (  # noqa: E402
    MatchedPairBenchmark,
    build_matched_benchmark,
    flatten_matched,
    write_matched_tables,
)
from train import get_device, train_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched-pair age × temporal experiment")
    p.add_argument("--build-data", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--task", choices=list(MATCHED_TASKS), default=None)
    p.add_argument("--arm", choices=list(ARMS), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _z_from_loader(loader) -> np.ndarray:
    return np.asarray([r.z_age for r in loader.dataset.rows], dtype=np.float32)


def run_one_matched(
    bench: MatchedPairBenchmark,
    task: str,
    arm: str,
    seed: int,
    cfg_kwargs: dict[str, Any] | None = None,
    max_train_pairs: int | None = None,
    early_stop: bool = True,
    collect_attention: bool = True,
) -> dict[str, Any]:
    cfg = Config(task=task, arm=arm, seed=seed, run_tag="matched")
    if cfg_kwargs:
        for k, v in cfg_kwargs.items():
            setattr(cfg, k, v)
        if "run_tag" not in (cfg_kwargs or {}):
            cfg.run_tag = "matched"

    if max_train_pairs is None:
        train_loader = bench.make_loader("train", task, shuffle=True, batch_size=cfg.batch_size)
        val_loader = bench.make_loader("val", task, shuffle=False, batch_size=cfg.batch_size)
        test_loader = bench.make_loader("test", task, shuffle=False, batch_size=cfg.batch_size)
    else:
        rng = np.random.default_rng(seed)
        pair_ids = sorted({r.pair_id for r in bench.split_rows["train"]})
        rng.shuffle(pair_ids)
        keep = set(pair_ids[:max_train_pairs])
        rows = [r for r in bench.split_rows["train"] if r.pair_id in keep]
        train_loader = bench.src.loader_from_rows(rows, task, shuffle=True, batch_size=cfg.batch_size)
        val_loader = bench.src.loader_from_rows(rows, task, shuffle=False, batch_size=cfg.batch_size)
        test_loader = val_loader

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
            "experiment": "matched_age_temporal_pairs",
        },
        z_test=_z_from_loader(test_loader),
        collect_attention=collect_attention,
        early_stop=early_stop,
    )
    return result


def smoke(bench: MatchedPairBenchmark, results_dir: Path) -> dict[str, Any]:
    print("\n=== MATCHED SMOKE: overfit 64 T1 pairs with age_temporal ===", flush=True)
    result = run_one_matched(
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
            "run_tag": "matched_smoke",
        },
        max_train_pairs=64,
        early_stop=False,
        collect_attention=True,
    )
    train = result["train"]
    rec = result["recovered"]
    pair_acc = float((train.get("pair") or {}).get("pair_accuracy") or 0.0)
    passed = train["accuracy"] >= 0.95 and train["bce"] <= 0.20 and rec["beta_hat"] > 0.05 and pair_acc >= 0.90
    summary = {
        "n_train": train["n"],
        "train_bce": train["bce"],
        "train_accuracy": train["accuracy"],
        "train_pair_accuracy": pair_acc,
        "lambda0_hat": rec["lambda0_hat"],
        "beta_hat": rec["beta_hat"],
        "passed": passed,
        "criteria": {
            "train_accuracy_ge_0.95": train["accuracy"] >= 0.95,
            "train_bce_le_0.20": train["bce"] <= 0.20,
            "beta_hat_positive": rec["beta_hat"] > 0.05,
            "pair_accuracy_ge_0.90": pair_acc >= 0.90,
        },
    }
    (results_dir / "smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MATCHED SMOKE", "PASS" if passed else "FAIL", json.dumps(summary["criteria"]), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    results_dir = Path(DEFAULT_MATCHED_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    do_build = args.build_data or args.all or args.smoke or args.full
    do_smoke = args.smoke or args.all
    do_full = args.full or args.all
    if args.task and args.arm and args.seed is not None:
        do_build = True

    print("device:", get_device(), flush=True)
    src = InteractionBenchmark(Config())
    print("source vocab", len(src.code_vocab), "types", len(src.type_vocab), flush=True)
    bench = build_matched_benchmark(src, results_dir=results_dir)

    if do_smoke:
        summary = smoke(bench, results_dir)
        if not summary["passed"] and do_full:
            print("MATCHED SMOKE FAILED — not running the full matrix.")
            return

    if args.task and args.arm and args.seed is not None:
        run_one_matched(bench, args.task, args.arm, args.seed)
        return

    if do_full:
        n_seeds = min(args.n_seeds, len(FULL_SEEDS))
        seeds = FULL_SEEDS[:n_seeds]
        tasks = list(MATCHED_TASKS)
        jobs = [(task, arm, seed) for seed in seeds for task in tasks for arm in ARMS]
        print(
            f"\n=== MATCHED FULL: {len(jobs)} runs "
            f"({len(tasks)} tasks × {len(ARMS)} arms × {n_seeds} seeds) ===",
            flush=True,
        )
        flat = []
        raw = []
        skipped = 0
        for i, (task, arm, seed) in enumerate(jobs, 1):
            metrics_path = Config(task=task, arm=arm, seed=seed, run_tag="matched").run_dir() / "metrics.json"
            if args.resume and metrics_path.exists():
                result = json.loads(metrics_path.read_text())
                print(f"\n--- job {i}/{len(jobs)} {task} {arm} seed={seed} SKIP (resume) ---", flush=True)
                raw.append(result)
                flat.append(flatten_matched(result))
                skipped += 1
                continue
            print(f"\n--- job {i}/{len(jobs)} {task} {arm} seed={seed} ---", flush=True)
            result = run_one_matched(bench, task, arm, seed)
            raw.append(result)
            flat.append(flatten_matched(result))
            pd.DataFrame(flat).to_csv(results_dir / "main_results_partial.csv", index=False)
        if args.resume:
            print(f"resumed: skipped {skipped} completed jobs", flush=True)
        (results_dir / "raw_runs.json").write_text(json.dumps(raw, indent=2, default=str) + "\n")
        write_matched_tables(flat, results_dir)
        print("Wrote", results_dir / "main_results.csv")


if __name__ == "__main__":
    main()
