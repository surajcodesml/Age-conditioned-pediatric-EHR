#!/usr/bin/env python3
"""Launch Developmental Age Incorporation Benchmark v1.

Smoke (implementation check, not a result):
    conda run -n ehr python age_incorporation_v1/run_experiment.py --smoke

Full 3 tasks × 4 arms × 5 seeds:
    conda run -n ehr python age_incorporation_v1/run_experiment.py --full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from config import ARMS, FULL_SEEDS, TASKS, Config  # noqa: E402
from dataset import SyntheaBenchmark  # noqa: E402
from model import AgeIncorporationModel, count_parameters  # noqa: E402
from train import get_device, set_seed, train_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Age incorporation v1 experiment runner")
    p.add_argument("--smoke", action="store_true", help="S2 × 4 arms × seed 0 × 2 epochs")
    p.add_argument("--full", action="store_true", help="3 tasks × 4 arms × 5 seeds")
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--arm", choices=ARMS, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def jobs_from_args(args: argparse.Namespace) -> tuple[list[tuple[str, str, int]], int]:
    if args.smoke:
        return [(("S2", arm, 0)) for arm in ARMS], 2
    if args.full:
        jobs = [(task, arm, seed) for task in TASKS for arm in ARMS for seed in FULL_SEEDS]
        return jobs, 30
    if args.task and args.arm and args.seed is not None:
        epochs = args.max_epochs if args.max_epochs is not None else 30
        return [(args.task, args.arm, args.seed)], epochs
    raise SystemExit("Specify --smoke, --full, or --task/--arm/--seed.")


def assert_shared_parameter_count(n_codes: int, n_types: int, cfg: Config) -> dict[str, int]:
    counts: dict[str, int] = {}
    ref_keys = None
    for arm in ARMS:
        set_seed(0)
        m = AgeIncorporationModel(
            arm=arm,
            n_codes=n_codes,
            n_types=n_types,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            age_hidden=cfg.age_hidden,
            head_hidden=cfg.head_hidden,
        )
        counts[arm] = count_parameters(m)
        keys = tuple(m.state_dict().keys())
        if ref_keys is None:
            ref_keys = keys
        elif keys != ref_keys:
            raise RuntimeError(f"state_dict keys differ for arm {arm}")
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"parameter counts differ across arms: {counts}")
    return counts


def write_summary(rows: list[dict], path: Path) -> None:
    flat = []
    for r in rows:
        item = {
            "task": r["task"],
            "arm": r["arm"],
            "seed": r["seed"],
            "n_params": r["n_params"],
            "best_epoch": r["best_epoch"],
            "val_auprc": r["val_auprc"],
            "test_auprc": r["test_auprc"],
            "test_auroc": r["test_auroc"],
            "test_bce": r["test_bce"],
        }
        for g, m in r.get("test_by_age_group", {}).items():
            item[f"test_auprc_{g}"] = m.get("auprc")
            item[f"test_auroc_{g}"] = m.get("auroc")
        flat.append(item)
    df = pd.DataFrame(flat)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    (path.with_suffix(".json")).write_text(json.dumps(flat, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    jobs, default_epochs = jobs_from_args(args)
    max_epochs = args.max_epochs if args.max_epochs is not None else default_epochs

    cfg0 = Config()
    if args.data_dir:
        cfg0.data_dir = args.data_dir
    if args.output_dir:
        cfg0.output_dir = args.output_dir

    print("Loading benchmark from", cfg0.data_dir, flush=True)
    bench = SyntheaBenchmark(cfg0)
    trunc = bench.truncation.to_dict()
    print("=== truncation ===")
    for k, v in trunc.items():
        print(f"  {k}: {v}")
    print("=== splits (unchanged from benchmark) ===")
    print(bench.split_counts)
    print(f"dropped non-preindex events: {bench.n_dropped_non_preindex}")
    print(f"code vocab size: {len(bench.code_vocab)}  type vocab size: {len(bench.type_vocab)}")

    n_codes = len(bench.code_vocab)
    n_types = len(bench.type_vocab)
    param_counts = assert_shared_parameter_count(n_codes, n_types, cfg0)
    print("=== parameter counts (identical) ===")
    print(param_counts)

    out_root = Path(cfg0.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "truncation_stats.json").write_text(json.dumps(trunc, indent=2) + "\n")
    (out_root / "vocab_code.json").write_text(json.dumps(bench.code_vocab, indent=2) + "\n")
    (out_root / "vocab_type.json").write_text(json.dumps(bench.type_vocab, indent=2) + "\n")
    (out_root / "param_counts.json").write_text(json.dumps(param_counts, indent=2) + "\n")

    device = get_device()
    print("device:", device)

    results: list[dict] = []
    loaders_by_task: dict[str, dict[str, object]] = {}
    extra = {
        "split_counts": bench.split_counts,
        "truncation": trunc,
        "param_counts": param_counts,
    }
    for task, arm, seed in jobs:
        if task not in loaders_by_task:
            loaders_by_task[task] = {
                "train": bench.make_loader("train", task, shuffle=True),
                "val": bench.make_loader("val", task, shuffle=False),
                "test": bench.make_loader("test", task, shuffle=False),
            }
            tr = loaders_by_task[task]["train"].dataset
            va = loaders_by_task[task]["val"].dataset
            te = loaders_by_task[task]["test"].dataset
            extra_task_counts = {"train": len(tr), "val": len(va), "test": len(te)}
            print(f"loaders for {task}: {extra_task_counts}")
        cfg = Config(
            data_dir=cfg0.data_dir,
            output_dir=cfg0.output_dir,
            task=task,
            arm=arm,
            seed=seed,
            max_epochs=max_epochs,
        )
        print(f"\n======== {task} / {arm} / seed={seed} / epochs={max_epochs} ========", flush=True)
        loaders = loaders_by_task[task]
        result = train_run(
            cfg,
            loaders["train"],
            loaders["val"],
            loaders["test"],
            n_codes=n_codes,
            n_types=n_types,
            extra_meta={
                **extra,
                "train_n": len(loaders["train"].dataset),
                "val_n": len(loaders["val"].dataset),
                "test_n": len(loaders["test"].dataset),
            },
        )
        results.append(result)

    tag = "smoke" if args.smoke else ("full" if args.full else "custom")
    summary_path = out_root / f"summary_{tag}.csv"
    write_summary(results, summary_path)
    print("\nWrote", summary_path)
    print(pd.read_csv(summary_path).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
