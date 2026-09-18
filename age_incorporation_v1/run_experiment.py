#!/usr/bin/env python3
"""Launch Developmental Age Incorporation Benchmark v1.

Smoke (implementation check, not a result):
    conda run -n ehr python age_incorporation_v1/run_experiment.py --smoke

Full baseline matrix (3 tasks × 4 arms × 5 seeds):
    conda run -n ehr python age_incorporation_v1/run_experiment.py --full

DKM smoke (S2, dkm_age, seed 0, 2 epochs + correctness checks):
    conda run -n ehr python age_incorporation_v1/run_experiment.py --smoke-dkm

DKM arm only (3 tasks × 5 seeds):
    conda run -n ehr python age_incorporation_v1/run_experiment.py --dkm
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

from config import (  # noqa: E402
    ARMS,
    BASELINE_ARMS,
    DKM_PROBE_AGES,
    FULL_SEEDS,
    S4_ARMS,
    TASKS,
    Config,
)
from dataset import S4Benchmark, SyntheaBenchmark  # noqa: E402
from model import AgeIncorporationModel, count_parameters, verify_dkm_batch  # noqa: E402
from train import get_device, set_seed, train_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Age incorporation v1 experiment runner")
    p.add_argument("--smoke", action="store_true", help="S2 × 4 baseline arms × seed 0 × 2 epochs")
    p.add_argument("--full", action="store_true", help="3 tasks × 4 baseline arms × 5 seeds")
    p.add_argument("--smoke-dkm", action="store_true", help="S2 × dkm_age × seed 0 × 2 epochs")
    p.add_argument("--dkm", action="store_true", help="3 tasks × dkm_age × 5 seeds")
    p.add_argument("--smoke-s4", action="store_true", help="S4 × 6 arms × seed 0 × 2 epochs")
    p.add_argument("--s4", action="store_true", help="S4 × 6 arms × 5 seeds (seed-major order)")
    p.add_argument("--build-s4", action="store_true", help="Build S4 data only (no training)")
    p.add_argument("--task", choices=list(TASKS) + ["S4"], default=None)
    p.add_argument("--arm", choices=ARMS, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def jobs_from_args(args: argparse.Namespace) -> tuple[list[tuple[str, str, int]], int]:
    if args.smoke:
        return [(("S2", arm, 0)) for arm in BASELINE_ARMS], 2
    if args.full:
        jobs = [(task, arm, seed) for task in TASKS for arm in BASELINE_ARMS for seed in FULL_SEEDS]
        return jobs, 30
    if args.smoke_dkm:
        return [("S2", "dkm_age", 0)], 2
    if args.dkm:
        jobs = [(task, "dkm_age", seed) for task in TASKS for seed in FULL_SEEDS]
        return jobs, 30
    if args.smoke_s4:
        return [("S4", arm, 0) for arm in S4_ARMS], 2
    if args.s4:
        # Seed-major order
        jobs = [("S4", arm, seed) for seed in FULL_SEEDS for arm in S4_ARMS]
        return jobs, 30
    if getattr(args, "build_s4", False):
        return [], 0
    if args.task and args.arm and args.seed is not None:
        epochs = args.max_epochs if args.max_epochs is not None else 30
        return [(args.task, args.arm, args.seed)], epochs
    raise SystemExit("Specify --smoke, --full, --smoke-dkm, --dkm, --smoke-s4, --s4, --build-s4, or --task/--arm/--seed.")


def assert_shared_parameter_count(n_codes: int, n_types: int, cfg: Config, arms: tuple[str, ...] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    ref_keys = None
    for arm in (arms or ARMS):
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
            age_scale_years=cfg.age_scale_years,
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


def flatten_result(r: dict) -> dict:
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
    return item


def write_summary(rows: list[dict], path: Path) -> None:
    flat = [flatten_result(r) for r in rows]
    df = pd.DataFrame(flat)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    (path.with_suffix(".json")).write_text(json.dumps(flat, indent=2) + "\n")


def merge_dkm_into_full_summary(dkm_rows: list[dict], out_root: Path) -> Path:
    full_csv = out_root / "summary_full.csv"
    if not full_csv.exists():
        write_summary(dkm_rows, full_csv)
        return full_csv
    existing = pd.read_csv(full_csv)
    existing = existing[existing["arm"] != "dkm_age"]
    added = pd.DataFrame([flatten_result(r) for r in dkm_rows])
    merged = pd.concat([existing, added], ignore_index=True)
    merged.to_csv(full_csv, index=False)
    (full_csv.with_suffix(".json")).write_text(
        json.dumps(merged.to_dict(orient="records"), indent=2) + "\n"
    )
    return full_csv


def plot_lambda_curves(out_root: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping lambda(a) plots")
        return []
    ages = list(DKM_PROBE_AGES)
    written: list[Path] = []
    for task in TASKS:
        curves = []
        for seed in FULL_SEEDS:
            path = out_root / f"{task}_dkm_age_seed{seed}" / "dkm_diagnostics.json"
            if not path.exists():
                continue
            blob = json.loads(path.read_text())
            lam = blob["best_checkpoint"]["lambda_at_ages"]
            curves.append([lam[str(a)] for a in ages])
        if not curves:
            continue
        arr = pd.DataFrame(curves, columns=ages)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if len(curves) > 1 else arr.iloc[0] * 0
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(ages, mean.values, marker="o", color="#1f4e79", label="mean across seeds")
        ax.fill_between(
            ages,
            (mean - sd).values,
            (mean + sd).values,
            color="#1f4e79",
            alpha=0.18,
            label="±1 SD",
        )
        ax.set_xlabel("age (years)")
        ax.set_ylabel(r"$\lambda(a)$")
        ax.set_title(f"{task}: age → λ(a)  (dkm_age, n={len(curves)} seeds)")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = out_root / f"dkm_lambda_curve_{task}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
    return written


def run_dkm_correctness_check(
    bench: SyntheaBenchmark,
    n_codes: int,
    n_types: int,
    cfg0: Config,
    device,
) -> dict:
    set_seed(0)
    model = AgeIncorporationModel(
        arm="dkm_age",
        n_codes=n_codes,
        n_types=n_types,
        d_model=cfg0.d_model,
        n_layers=cfg0.n_layers,
        n_heads=cfg0.n_heads,
        dim_feedforward=cfg0.dim_feedforward,
        dropout=cfg0.dropout,
        age_hidden=cfg0.age_hidden,
        head_hidden=cfg0.head_hidden,
        age_scale_years=cfg0.age_scale_years,
    ).to(device)
    loader = bench.make_loader("train", "S2", shuffle=False)
    batch = next(iter(loader))
    checks = verify_dkm_batch(model, batch, device)
    print("=== dkm_age correctness smoke ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    required = (
        "bias_shape_ok",
        "mask_shape_ok",
        "tau_in_01",
        "tau_diag_zero",
        "lambda_positive",
        "age_generator_has_grad",
        "lambda_base_has_grad",
        "lambda_curve_nearly_constant",
    )
    failed = [k for k in required if checks.get(k) is not True]
    if failed:
        raise RuntimeError(f"dkm_age correctness checks failed: {failed}")
    print("dkm_age correctness checks passed")
    return checks


def run_s4_smoke_checks(
    bench: S4Benchmark,
    n_codes: int,
    n_types: int,
    cfg0: Config,
    device,
) -> dict[str, dict]:
    """Smoke checks for both dkm_age and shared_decay on S4."""
    results = {}
    for arm in ("dkm_age", "shared_decay"):
        set_seed(0)
        model = AgeIncorporationModel(
            arm=arm,
            n_codes=n_codes,
            n_types=n_types,
            d_model=cfg0.d_model,
            n_layers=cfg0.n_layers,
            n_heads=cfg0.n_heads,
            dim_feedforward=cfg0.dim_feedforward,
            dropout=cfg0.dropout,
            age_hidden=cfg0.age_hidden,
            head_hidden=cfg0.head_hidden,
            age_scale_years=cfg0.age_scale_years,
        ).to(device)
        loader = bench.make_loader("train", "S4", shuffle=False)
        batch = next(iter(loader))
        checks = verify_dkm_batch(model, batch, device)
        # shared_decay: verify lambda is constant across ages
        if arm == "shared_decay":
            curve = checks.get("lambda_curve_init", {})
            vals = list(curve.values()) if isinstance(curve, dict) else []
            checks["shared_decay_lambda_constant"] = (
                (max(vals) - min(vals) < 1e-8) if vals else False
            )
        print(f"=== {arm} correctness smoke ===")
        for k, v in checks.items():
            print(f"  {k}: {v}")
        results[arm] = checks
    return results


def write_s4_summary(results: list[dict], out_root: Path) -> Path:
    """Write S4 results, appending to existing summary if present."""
    s4_csv = out_root / "summary_s4.csv"
    flat = [flatten_result(r) for r in results]
    df = pd.DataFrame(flat)
    df.to_csv(s4_csv, index=False)
    (s4_csv.with_suffix(".json")).write_text(json.dumps(flat, indent=2) + "\n")

    # Also append to summary_full if it exists
    full_csv = out_root / "summary_full.csv"
    if full_csv.exists():
        existing = pd.read_csv(full_csv)
        existing = existing[existing["task"] != "S4"]
        merged = pd.concat([existing, df], ignore_index=True)
        merged.to_csv(full_csv, index=False)
        (full_csv.with_suffix(".json")).write_text(
            json.dumps(merged.to_dict(orient="records"), indent=2) + "\n"
        )
    return s4_csv


def plot_s4_lambda_curves(out_root: Path) -> list[Path]:
    """Plot learned lambda(a) curves for S4 dkm_age runs vs lambda_true."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping lambda(a) plots")
        return []
    import math
    ages = list(DKM_PROBE_AGES)
    written: list[Path] = []

    # lambda_true for comparison
    def _lambda_true(a):
        return 0.5 + 2.5 * math.exp(-a / 4.0)
    lambda_true_vals = [_lambda_true(a) for a in ages]

    for arm in ("dkm_age", "shared_decay"):
        curves = []
        for seed in FULL_SEEDS:
            path = out_root / f"S4_{arm}_seed{seed}" / "dkm_diagnostics.json"
            if not path.exists():
                continue
            blob = json.loads(path.read_text())
            lam = blob["best_checkpoint"]["lambda_at_ages"]
            curves.append([lam[str(a)] for a in ages])
        if not curves:
            continue
        arr = pd.DataFrame(curves, columns=ages)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if len(curves) > 1 else arr.iloc[0] * 0
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(ages, mean.values, marker="o", color="#1f4e79", label=f"{arm} learned (mean)")
        ax.fill_between(ages, (mean - sd).values, (mean + sd).values,
                        color="#1f4e79", alpha=0.18, label="±1 SD")
        ax.plot(ages, lambda_true_vals, "--", color="#c44e52", linewidth=2, label=r"$\lambda_{\mathrm{true}}(a)$")
        ax.set_xlabel("age (years)")
        ax.set_ylabel(r"$\lambda(a)$")
        ax.set_title(f"S4: {arm} learned λ(a) vs ground truth (n={len(curves)} seeds)")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out = out_root / f"s4_lambda_curve_{arm}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)
    return written


def main() -> int:
    args = parse_args()

    # Handle --build-s4 early
    if getattr(args, "build_s4", False):
        from build_s4 import build_s4 as _build_s4  # noqa: E402
        _build_s4()
        return 0

    jobs, default_epochs = jobs_from_args(args)
    max_epochs = args.max_epochs if args.max_epochs is not None else default_epochs
    is_s4 = getattr(args, "smoke_s4", False) or getattr(args, "s4", False) or (
        args.task == "S4" if args.task else False
    )

    cfg0 = Config()
    if args.data_dir:
        cfg0.data_dir = args.data_dir
    if args.output_dir:
        cfg0.output_dir = args.output_dir

    if is_s4:
        print("Loading S4 benchmark ...", flush=True)
        bench = S4Benchmark(cfg0)
    else:
        print("Loading benchmark from", cfg0.data_dir, flush=True)
        bench = SyntheaBenchmark(cfg0)

    trunc = bench.truncation.to_dict()
    # Add TEMP stats if present
    if hasattr(bench.truncation, "n_temp_events"):
        trunc["n_temp_events"] = bench.truncation.n_temp_events
        trunc["n_temp_events_kept"] = bench.truncation.n_temp_events_kept
        trunc["frac_temp_events_lost"] = bench.truncation.frac_temp_events_lost
    print("=== truncation ===")
    for k, v in trunc.items():
        print(f"  {k}: {v}")
    print("=== splits ===")
    print(bench.split_counts)
    print(f"dropped non-preindex events: {bench.n_dropped_non_preindex}")
    print(f"code vocab size: {len(bench.code_vocab)}  type vocab size: {len(bench.type_vocab)}")

    n_codes = len(bench.code_vocab)
    n_types = len(bench.type_vocab)
    check_arms = S4_ARMS if is_s4 else ARMS
    param_counts = assert_shared_parameter_count(n_codes, n_types, cfg0, arms=check_arms)
    print("=== parameter counts (identical) ===")
    print(param_counts)

    out_root = Path(cfg0.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tag_prefix = "s4_" if is_s4 else ""
    (out_root / f"{tag_prefix}truncation_stats.json").write_text(json.dumps(trunc, indent=2) + "\n")
    if not is_s4:
        (out_root / "vocab_code.json").write_text(json.dumps(bench.code_vocab, indent=2) + "\n")
        (out_root / "vocab_type.json").write_text(json.dumps(bench.type_vocab, indent=2) + "\n")
    else:
        (out_root / "s4_vocab_code.json").write_text(json.dumps(bench.code_vocab, indent=2) + "\n")
        (out_root / "s4_vocab_type.json").write_text(json.dumps(bench.type_vocab, indent=2) + "\n")
    (out_root / f"{tag_prefix}param_counts.json").write_text(json.dumps(param_counts, indent=2) + "\n")

    device = get_device()
    print("device:", device)

    if args.smoke_dkm or args.dkm:
        checks = run_dkm_correctness_check(bench, n_codes, n_types, cfg0, device)
        (out_root / "dkm_correctness_smoke.json").write_text(
            json.dumps(checks, indent=2, default=str) + "\n"
        )
        if args.dkm:
            print("correctness smoke passed; launching 15 full dkm_age runs", flush=True)

    if is_s4 and (getattr(args, "smoke_s4", False) or getattr(args, "s4", False)):
        s4_checks = run_s4_smoke_checks(bench, n_codes, n_types, cfg0, device)
        (out_root / "s4_correctness_smoke.json").write_text(
            json.dumps(s4_checks, indent=2, default=str) + "\n"
        )
        # Verify shared_decay lambda is constant
        sd_checks = s4_checks.get("shared_decay", {})
        if sd_checks.get("shared_decay_lambda_constant") is False:
            raise RuntimeError("shared_decay lambda is not constant across ages!")
        # Verify dkm_age age generator has gradient
        dkm_checks = s4_checks.get("dkm_age", {})
        if not dkm_checks.get("age_generator_has_grad"):
            raise RuntimeError("dkm_age age_generator has no gradient!")

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

    if not results:
        return 0

    if is_s4:
        if getattr(args, "smoke_s4", False):
            tag = "smoke_s4"
        elif getattr(args, "s4", False):
            tag = "s4"
        else:
            tag = "custom"
    elif args.smoke:
        tag = "smoke"
    elif args.full:
        tag = "full"
    elif args.smoke_dkm:
        tag = "smoke_dkm"
    elif args.dkm:
        tag = "dkm"
    else:
        tag = "custom"
    summary_path = out_root / f"summary_{tag}.csv"
    write_summary(results, summary_path)
    print("\nWrote", summary_path)
    print(pd.read_csv(summary_path).to_string(index=False))

    if args.dkm:
        merged = merge_dkm_into_full_summary(results, out_root)
        print("Appended dkm_age rows to", merged)
        plots = plot_lambda_curves(out_root)
        for p in plots:
            print("Wrote", p)
    elif args.smoke_dkm:
        plot_lambda_curves(out_root)
    elif getattr(args, "s4", False):
        s4_csv = write_s4_summary(results, out_root)
        print("Wrote S4 summary to", s4_csv)
        plots = plot_s4_lambda_curves(out_root)
        for p in plots:
            print("Wrote", p)
    elif getattr(args, "smoke_s4", False):
        pass  # No plots for smoke
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
