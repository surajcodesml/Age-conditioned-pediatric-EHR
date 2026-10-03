#!/usr/bin/env python3
"""Unified baseline runner suite for all experimental stages.

Stages supported:
    1. synthetic      - S0–S3 (core age × temporal) and S5 (heterogeneous persistence)
                        benchmark with counterfactual RMSE evaluations.
    2. mimic_pretrain - Stage-1 MIMIC-IV next-visit prediction (multilabel BCE).
    3. nch_finetune   - Stage-2 NCH pediatric finetuning (optionally initialized
                        from MIMIC Stage-1 checkpoints).
    4. all            - Runs synthetic -> mimic_pretrain -> nch_finetune sequentially.

Usage examples:
    python -m baselines.run_suite --models motor,tale_ehr,nest --stage synthetic \\
        --config configs/baselines/synthetic.yaml

    python -m baselines.run_suite --models motor,tale_ehr,nest --stage mimic_pretrain \\
        --config configs/baselines/mimic.yaml

    python -m baselines.run_suite --models motor,tale_ehr,nest --stage nch_finetune \\
        --config configs/baselines/nch.yaml --use-mimic-checkpoints

    python -m baselines.run_suite --models motor,tale_ehr,nest --stage all \\
        --config-dir configs/baselines --resume

    python -m baselines.run_suite --models motor,tale_ehr,nest --stage synthetic --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Ensure baselines are registered
from baselines.common.registry import REGISTRY
from baselines.motor.model import MOTORModel  # noqa: F401
from baselines.tale_ehr.model import TALEEHRModel  # noqa: F401
from baselines.nest.model import NESTModel, check_nest_feasibility, FEASIBILITY_STATUS  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_suite")


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def handle_nest_feasibility(models: list[str], strict: bool = False) -> list[str]:
    """Inspect NEST feasibility and return active models.

    If NEST is requested, reports official code/weight release status.
    If strict mode is enabled and official code is unreleased, excludes NEST.
    """
    if "nest" not in models:
        return models

    is_feasible, note = check_nest_feasibility()
    status = FEASIBILITY_STATUS
    logger.info("=" * 70)
    logger.info("NEST Feasibility Gate Check:")
    logger.info(f"  - Official code available: {status['official_code_available']}")
    logger.info(f"  - Official weights available: {status['official_weights_available']}")
    logger.info(f"  - Faithful paper implementation available: {is_feasible}")
    logger.info(f"  - Note: {note}")
    logger.info("=" * 70)

    if strict and not status["official_code_available"]:
        logger.warning(
            "Strict feasibility requested: Excluding 'nest' because official "
            "repository has not yet been released by the authors."
        )
        return [m for m in models if m != "nest"]

    logger.info("Proceeding with paper-faithful NEST architecture (SWE + CSE + RoPE + SwiGLU).")
    return models


def run_stage_synthetic(
    models: list[str],
    cfg: dict[str, Any],
    *,
    smoke: bool = False,
    device: str = "cuda",
    seed: int = 0,
    resume: bool = False,
    overwrite: bool = False,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute synthetic benchmark stage (training + counterfactual eval)."""
    logger.info(">>> Starting Stage: SYNTHETIC BENCHMARK")
    from baselines.synthetic.runner import (
        ALL_MODELS,
        BENCHMARK_SCENARIOS,
        run_one_model,
        resolve_scenarios,
    )
    from baselines.synthetic.counterfactual_eval import main as run_cf_eval

    scenarios = ["S0"] if smoke else cfg.get("scenarios", ["S0", "S1", "S2", "S3", "S5"])
    data_seed = cfg.get("data_seed", 20260922)
    max_epochs = cfg.get("max_epochs", 40)
    out_dir = output_dir or (REPO_ROOT / cfg.get("output_dir", "results/baselines/synthetic"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from synthetic_age_temporal.config import Config as SynConfig
    data_base = SynConfig(data_seed=data_seed).data_dir() / "controlled"

    results = {}
    for scenario in scenarios:
        scenario_dir = data_base / scenario
        if not scenario_dir.exists():
            logger.warning(f"Scenario directory {scenario_dir} not found; skipping {scenario}")
            continue

        for model_name in models:
            run_key = f"{model_name}_{scenario}"
            model_run_dir = out_dir / model_name / scenario
            if (model_run_dir / "result.json").exists() and resume and not overwrite and not smoke:
                logger.info(f"Skipping {run_key} (already completed and resume=True)")
                with (model_run_dir / "result.json").open() as f:
                    results[run_key] = json.load(f)
                continue

            logger.info(f"Running synthetic {model_name} on {scenario}...")
            res = run_one_model(
                model_name=model_name,
                scenario=scenario,
                scenario_dir=scenario_dir,
                output_dir=out_dir,
                seed=seed,
                max_epochs=max_epochs,
                device=device,
                smoke=smoke,
                data_seed=data_seed,
            )
            results[run_key] = res

    # Run counterfactual evaluation on the trained models
    logger.info("Running counterfactual evaluation for synthetic models...")
    for scenario in scenarios:
        try:
            cf_args = [
                "--scenario", scenario,
                "--models", ",".join(models),
                "--results-dir", str(out_dir),
                "--data-seed", str(data_seed),
                "--device", device,
            ]
            sys_argv_orig = sys.argv
            sys.argv = ["counterfactual_eval.py"] + cf_args
            run_cf_eval()
            sys.argv = sys_argv_orig
        except Exception as e:
            logger.error(f"Error during counterfactual eval for {scenario}: {e}")

    return results


def run_stage_mimic(
    models: list[str],
    cfg: dict[str, Any],
    *,
    smoke: bool = False,
    device: str = "cuda",
    seed: int = 0,
    resume: bool = False,
    overwrite: bool = False,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute MIMIC Stage-1 pretraining stage."""
    logger.info(">>> Starting Stage: MIMIC PRETRAINING (STAGE-1)")
    from baselines.mimic.runner import (
        TensorizedPretrainDataset,
        run_one_model,
        THROUGHPUT_BATCH_SIZES,
        BATCH_SIZE as DEFAULT_BS,
    )

    out_dir = output_dir or (REPO_ROOT / cfg.get("output_dir", "results/baselines/mimic"))
    if smoke:
        out_dir = out_dir / "_smoke"
    elif seed != 0:
        out_dir = out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tensorized_dir = REPO_ROOT / cfg.get("tensorized_dir", "data/processed/tensorized_flat")
    vocab_path = REPO_ROOT / cfg.get("vocab_path", "data/processed/code_vocab.json")

    if smoke:
        from stage1_mimic_pretrain.train import maybe_restrict_tensorized
        tensorized_dir = maybe_restrict_tensorized(tensorized_dir, out_dir / "data_subset", max_shards=1, seed=0)

    train_ds = TensorizedPretrainDataset(tensorized_dir / "train", vocab_path, max_seq_len=256)
    val_ds = TensorizedPretrainDataset(tensorized_dir / "val", vocab_path, max_seq_len=256)
    test_ds = TensorizedPretrainDataset(tensorized_dir / "test", vocab_path, max_seq_len=256)

    if smoke:
        import torch
        train_ds = torch.utils.data.Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(64, len(val_ds))))
        test_ds = torch.utils.data.Subset(test_ds, range(min(64, len(test_ds))))

    throughput_opt = cfg.get("throughput_optimized", True)
    amp = cfg.get("amp", "bf16")
    default_bs = cfg.get("batch_size", DEFAULT_BS)
    num_workers = 0 if smoke else cfg.get("num_workers", 4)
    val_max = cfg.get("val_max_batches", 50)
    test_max = cfg.get("test_max_batches", 100)

    batch_sizes_map = {**THROUGHPUT_BATCH_SIZES, **cfg.get("throughput_batch_sizes", {})}
    results = {}
    for model_name in models:
        m_dir = out_dir / model_name
        if (m_dir / "result.json").exists() and resume and not overwrite and not smoke:
            logger.info(f"Skipping MIMIC {model_name} (already completed and resume=True)")
            with (m_dir / "result.json").open() as f:
                results[model_name] = json.load(f)
            continue

        bs = batch_sizes_map.get(model_name, default_bs) if throughput_opt else default_bs
        if model_name == "nest" and bs > 32:
            bs = 32
        if smoke:
            bs = 8

        logger.info(f"Training MIMIC model: {model_name} (batch_size={bs}, amp={amp})...")
        res = run_one_model(
            model_name=model_name,
            output_dir=out_dir,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            batch_size=bs,
            smoke=smoke,
            device=device,
            arm="age_temporal",
            val_max_batches=val_max,
            test_max_batches=test_max,
            num_workers=num_workers,
            seed=seed,
            amp=amp,
        )
        results[model_name] = res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_stage_nch(
    models: list[str],
    cfg: dict[str, Any],
    *,
    smoke: bool = False,
    device: str = "cuda",
    seed: int = 0,
    resume: bool = False,
    overwrite: bool = False,
    use_mimic_checkpoints: bool = True,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute NCH Stage-2 finetuning stage."""
    logger.info(">>> Starting Stage: NCH FINETUNING (STAGE-2)")
    from baselines.nch.runner import (
        TensorizedPretrainDataset,
        run_one_model,
        BATCH_SIZE as DEFAULT_BS,
    )

    out_dir = output_dir or (REPO_ROOT / cfg.get("output_dir", "results/baselines/nch"))
    if smoke:
        out_dir = out_dir / "_smoke"
    elif seed != 0:
        out_dir = out_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    nch_dir = REPO_ROOT / cfg.get("nch_dir", "data/processed/nch")
    vocab_path = REPO_ROOT / cfg.get("vocab_path", "data/processed/code_vocab.json")

    # If NCH data is not present on disk, check if dummy / smoke subset is needed
    if not (nch_dir / "train.pt").exists() and not (nch_dir / "train").exists():
        logger.warning(
            f"NCH dataset not found at {nch_dir}. Falling back to MIMIC subset for contract validation."
        )
        from model_new.data import TensorizedPretrainDataset as FallbackDS
        mimic_dir = REPO_ROOT / "data/processed/tensorized_flat"
        train_ds = FallbackDS(mimic_dir / "train", vocab_path, max_seq_len=256)
        val_ds = FallbackDS(mimic_dir / "val", vocab_path, max_seq_len=256)
        test_ds = FallbackDS(mimic_dir / "test", vocab_path, max_seq_len=256)
    else:
        train_ds = TensorizedPretrainDataset(nch_dir / "train", vocab_path, max_seq_len=256)
        val_ds = TensorizedPretrainDataset(nch_dir / "val", vocab_path, max_seq_len=256)
        test_ds = TensorizedPretrainDataset(nch_dir / "test", vocab_path, max_seq_len=256)

    if smoke:
        import torch
        train_ds = torch.utils.data.Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(64, len(val_ds))))
        test_ds = torch.utils.data.Subset(test_ds, range(min(64, len(test_ds))))

    from baselines.mimic.runner import THROUGHPUT_BATCH_SIZES
    throughput_opt = cfg.get("throughput_optimized", True)
    default_bs = cfg.get("batch_size", DEFAULT_BS)
    batch_sizes_map = {**THROUGHPUT_BATCH_SIZES, **cfg.get("throughput_batch_sizes", {})}
    amp = cfg.get("amp", "bf16")
    num_workers = 0 if smoke else cfg.get("num_workers", 0)
    val_max = cfg.get("val_max_batches", 50)
    test_max = cfg.get("test_max_batches", 100)

    results = {}
    for model_name in models:
        m_dir = out_dir / model_name
        if (m_dir / "result.json").exists() and resume and not overwrite and not smoke:
            logger.info(f"Skipping NCH {model_name} (already completed and resume=True)")
            with (m_dir / "result.json").open() as f:
                results[model_name] = json.load(f)
            continue

        bs = batch_sizes_map.get(model_name, default_bs) if throughput_opt else default_bs
        if model_name == "nest" and bs > 32:
            bs = 32
        if smoke:
            bs = 8

        logger.info(f"Finetuning NCH model: {model_name} (batch_size={bs}, use_mimic_checkpoints={use_mimic_checkpoints})...")
        res = run_one_model(
            model_name=model_name,
            output_dir=out_dir,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            batch_size=bs,
            smoke=smoke,
            device=device,
            arm="age_temporal",
            val_max_batches=val_max,
            test_max_batches=test_max,
            num_workers=num_workers,
            seed=seed,
            amp=amp,
        )
        results[model_name] = res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Baseline Runner Suite")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["synthetic", "mimic_pretrain", "nch_finetune", "all"],
        help="Stage to execute: synthetic, mimic_pretrain, nch_finetune, or all",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names (e.g., 'motor,tale_ehr,nest') or 'all'",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file for the stage",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/baselines",
        help="Directory containing default stage configs (synthetic.yaml, mimic.yaml, nch.yaml)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke run (2 epochs, tiny batches)")
    parser.add_argument("--resume", action="store_true", help="Skip models/scenarios with existing result.json")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runs")
    parser.add_argument(
        "--use-mimic-checkpoints",
        action="store_true",
        default=True,
        help="When running nch_finetune, load corresponding MIMIC Stage-1 weights",
    )
    parser.add_argument(
        "--no-mimic-checkpoints",
        action="store_false",
        dest="use_mimic_checkpoints",
        help="Train NCH models from scratch without MIMIC initialization",
    )
    parser.add_argument(
        "--strict-feasibility",
        action="store_true",
        help="Strictly exclude models whose official code has not been released (e.g. NEST)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    # Resolve models
    if args.models == "all":
        selected_models = [
            "count_lightgbm", "retain", "ehr_bert", "behrt", "medbert", "cehrbert",
            "motor", "tale_ehr", "nest", "dtr",
        ]
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Feasibility gate for NEST
    active_models = handle_nest_feasibility(selected_models, strict=args.strict_feasibility)

    cfg_dir = REPO_ROOT / args.config_dir
    suite_summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": args.stage,
        "models_requested": selected_models,
        "models_executed": active_models,
        "seed": args.seed,
        "smoke": args.smoke,
        "device": args.device,
        "stages_run": {},
    }

    t0_suite = time.time()

    # Stage execution
    if args.stage in ("synthetic", "all"):
        cfg_path = Path(args.config) if args.config and args.stage == "synthetic" else cfg_dir / "synthetic.yaml"
        cfg_synth = load_yaml_config(cfg_path) if cfg_path.exists() else {}
        res_synth = run_stage_synthetic(
            active_models,
            cfg_synth,
            smoke=args.smoke,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
            overwrite=args.overwrite,
            output_dir=Path(args.output_dir) if args.output_dir and args.stage == "synthetic" else None,
        )
        suite_summary["stages_run"]["synthetic"] = {
            "num_runs": len(res_synth),
            "keys": list(res_synth.keys()),
        }

    if args.stage in ("mimic_pretrain", "all"):
        cfg_path = Path(args.config) if args.config and args.stage == "mimic_pretrain" else cfg_dir / "mimic.yaml"
        cfg_mimic = load_yaml_config(cfg_path) if cfg_path.exists() else {}
        res_mimic = run_stage_mimic(
            active_models,
            cfg_mimic,
            smoke=args.smoke,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
            overwrite=args.overwrite,
            output_dir=Path(args.output_dir) if args.output_dir and args.stage == "mimic_pretrain" else None,
        )
        suite_summary["stages_run"]["mimic_pretrain"] = {
            "num_runs": len(res_mimic),
            "keys": list(res_mimic.keys()),
        }

    if args.stage in ("nch_finetune", "all"):
        cfg_path = Path(args.config) if args.config and args.stage == "nch_finetune" else cfg_dir / "nch.yaml"
        cfg_nch = load_yaml_config(cfg_path) if cfg_path.exists() else {}
        res_nch = run_stage_nch(
            active_models,
            cfg_nch,
            smoke=args.smoke,
            device=args.device,
            seed=args.seed,
            resume=args.resume,
            overwrite=args.overwrite,
            use_mimic_checkpoints=args.use_mimic_checkpoints,
            output_dir=Path(args.output_dir) if args.output_dir and args.stage == "nch_finetune" else None,
        )
        suite_summary["stages_run"]["nch_finetune"] = {
            "num_runs": len(res_nch),
            "keys": list(res_nch.keys()),
        }

    suite_summary["total_elapsed_sec"] = time.time() - t0_suite

    # Save summary report
    summary_dir = REPO_ROOT / "results" / "baselines"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "run_suite_summary.json"
    with summary_file.open("w") as f:
        json.dump(suite_summary, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info(f"Baseline suite execution completed in {suite_summary['total_elapsed_sec']:.2f}s")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
