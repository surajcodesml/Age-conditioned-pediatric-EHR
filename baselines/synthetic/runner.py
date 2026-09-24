#!/usr/bin/env python3
"""Synthetic benchmark runner — train and evaluate all baselines on S0–S3.

Usage:
    python -m baselines.synthetic.runner --scenario S2 --models all
    python -m baselines.synthetic.runner --scenario S2 --models retain,behrt
    python -m baselines.synthetic.runner --scenario S2 --models count_lightgbm --smoke

All neural baselines train from scratch on the synthetic benchmark prediction task.
No MIMIC pretraining. Same splits, targets, optimizer budget, early stopping.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "synthetic_age_temporal"))

from synthetic_age_temporal.config import (
    BATCH_SIZE, D_MODEL, DROPOUT, GRAD_CLIP, LR, MAX_EPOCHS, MAX_SEQ_LEN,
    MODEL_SEED, N_HEADS, N_LAYERS, PATIENCE, WEIGHT_DECAY, Config,
)
from synthetic_age_temporal.dataset import make_loaders, collate_batch
from baselines.common.metrics import multilabel_metrics
from baselines.common.training import (
    evaluate_loader, set_seed, get_device, train_neural_baseline,
)
from baselines.common.capacity_report import count_parameters

# Import all baselines to register them
from baselines.lightgbm.model import LightGBMBaseline, build_features_from_batch
from baselines.retain.model import RETAINModel
from baselines.ehr_bert.model import EHRBertModel
from baselines.behrt.model import BEHRTModel
from baselines.medbert.model import MedBERTModel
from baselines.cehrbert_adapter.adapter import CEHRBertAdapter
from baselines.dtr_adapter.adapter import DTRAdapter

from baselines.common.registry import REGISTRY


BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "count_lightgbm": {},
    "retain": {"d_emb": 128, "d_rnn": 128, "dropout": DROPOUT},
    "ehr_bert": {"d_model": D_MODEL, "n_layers": 4, "n_heads": 4,
                 "d_ff": D_MODEL * 4, "dropout": DROPOUT, "max_seq_len": MAX_SEQ_LEN + 16},
    "behrt": {"d_model": 288, "n_layers": 6, "n_heads": 12, "dropout": DROPOUT,
              "max_seq_len": MAX_SEQ_LEN + 16},
    "medbert": {"d_model": 192, "n_layers": 6, "n_heads": 6, "dropout": DROPOUT,
                "max_seq_len": MAX_SEQ_LEN + 16},
    "cehrbert": {"d_model": 128, "n_layers": 5, "n_heads": 8, "dropout": DROPOUT,
                 "max_seq_len": MAX_SEQ_LEN + 16},
}

# DTR arms to evaluate
DTR_ARMS = ("no_age", "age_only", "temporal_only", "age_temporal")


def build_model(
    name: str,
    n_codes: int,
    n_types: int,
    n_targets: int,
    arm: str = "age_temporal",
) -> Any:
    """Instantiate a baseline model by name."""
    if name == "count_lightgbm":
        return LightGBMBaseline(n_codes=n_codes, n_targets=n_targets)
    elif name == "retain":
        cfg = BASELINE_CONFIGS["retain"]
        return RETAINModel(n_codes=n_codes, n_targets=n_targets, **cfg)
    elif name == "ehr_bert":
        cfg = BASELINE_CONFIGS["ehr_bert"]
        return EHRBertModel(n_codes=n_codes, n_targets=n_targets, **cfg)
    elif name == "behrt":
        cfg = BASELINE_CONFIGS["behrt"]
        return BEHRTModel(n_codes=n_codes, n_targets=n_targets, **cfg)
    elif name == "medbert":
        cfg = BASELINE_CONFIGS["medbert"]
        return MedBERTModel(n_codes=n_codes, n_targets=n_targets, **cfg)
    elif name == "cehrbert":
        cfg = BASELINE_CONFIGS["cehrbert"]
        return CEHRBertAdapter(n_codes=n_codes, n_targets=n_targets, **cfg)
    elif name == "dtr":
        return DTRAdapter(
            arm=arm, n_codes=n_codes, n_types=n_types, n_targets=n_targets,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            dim_feedforward=D_MODEL * 2, dropout=DROPOUT,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train_lightgbm(
    model: LightGBMBaseline,
    train_loader,
    val_loader,
    test_loader,
    n_codes: int,
) -> dict[str, Any]:
    """Train LightGBM on extracted features."""
    # Extract features and labels from all loaders
    def extract(loader):
        Xs, Ys = [], []
        for batch in loader:
            X = build_features_from_batch(batch, n_codes)
            Xs.append(X)
            y = batch["labels"]
            if isinstance(y, torch.Tensor):
                y = y.numpy()
            Ys.append(y)
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

    X_train, y_train = extract(train_loader)
    X_val, y_val = extract(val_loader)
    X_test, y_test = extract(test_loader)

    t0 = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0

    # Evaluate
    logits_test = model.predict_logits(X_test)
    test_metrics = multilabel_metrics(y_test, logits_test)

    logits_val = model.predict_logits(X_val)
    val_metrics = multilabel_metrics(y_val, logits_val)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_time_s": train_time,
        "n_features": X_train.shape[1],
    }


def train_neural(
    model,
    train_loader,
    val_loader,
    *,
    run_dir: Path,
    seed: int = MODEL_SEED,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train a neural baseline using the common training loop."""
    return train_neural_baseline(
        model=model,
        train_fn=model.training_step,
        predict_fn=model.predict,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=WEIGHT_DECAY,
        max_epochs=max_epochs,
        patience=PATIENCE,
        grad_clip=GRAD_CLIP,
        device=device,
        run_dir=run_dir,
        seed=seed,
    )


def evaluate_test(model, test_loader, device) -> dict[str, float]:
    """Evaluate on test set."""
    dev = get_device(device)
    if hasattr(model, 'to'):
        model.to(dev)
    return evaluate_loader(model, model.predict, test_loader, dev)


def run_one_model(
    model_name: str,
    scenario: str,
    scenario_dir: Path,
    output_dir: Path,
    *,
    seed: int = MODEL_SEED,
    max_epochs: int = MAX_EPOCHS,
    device: str = "cuda",
    smoke: bool = False,
) -> dict[str, Any]:
    """Train and evaluate one model on one scenario."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Scenario: {scenario}")
    print(f"{'='*60}")

    # Load data
    batch_size = 8 if smoke else BATCH_SIZE
    max_ep = 2 if smoke else max_epochs

    train_loader, val_loader, test_loader, vocab, info = make_loaders(
        scenario_dir, batch_size=batch_size, max_seq_len=MAX_SEQ_LEN,
    )
    n_codes = info["n_codes"]
    n_types = info["n_types"]
    n_targets = info["n_targets"]

    run_dir = output_dir / model_name / scenario
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "result.json").exists() and not smoke:
        print(f"  Already trained, skipping {model_name} {scenario}")
        with (run_dir / "result.json").open() as f:
            return json.load(f)

    set_seed(seed)

    result: dict[str, Any] = {
        "model": model_name,
        "scenario": scenario,
        "seed": seed,
        "n_codes": n_codes,
        "n_targets": n_targets,
        "smoke": smoke,
    }

    if model_name == "count_lightgbm":
        model = build_model(model_name, n_codes, n_types, n_targets)
        lgb_result = train_lightgbm(model, train_loader, val_loader, test_loader, n_codes)
        result.update(lgb_result)
        result["model_card"] = model.model_card
        model.save_checkpoint(run_dir)
    elif model_name == "dtr":
        # Run all DTR arms
        for arm in DTR_ARMS:
            arm_name = f"dtr_{arm}"
            arm_dir = output_dir / arm_name / scenario
            arm_dir.mkdir(parents=True, exist_ok=True)
            
            if (arm_dir / "result.json").exists() and not smoke:
                print(f"  Already trained, skipping {arm_name} {scenario}")
                continue
                
            model = build_model("dtr", n_codes, n_types, n_targets, arm=arm)
            train_result = train_neural(
                model, train_loader, val_loader,
                run_dir=arm_dir, seed=seed, max_epochs=max_ep, device=device,
            )
            test_metrics = evaluate_test(model, test_loader, device)
            arm_result = {
                "model": arm_name,
                "arm": arm,
                "scenario": scenario,
                "train": train_result,
                "test_metrics": test_metrics,
                "model_card": model.model_card,
            }
            with (arm_dir / "result.json").open("w") as f:
                json.dump(arm_result, f, indent=2, default=str)
            print(f"  {arm_name}: test AUROC={test_metrics.get('micro_auroc', 'N/A'):.4f}")
        return result  # DTR results saved per-arm
    else:
        model = build_model(model_name, n_codes, n_types, n_targets)
        if hasattr(model, 'to'):
            model.to(get_device(device))
        if isinstance(model, torch.nn.Module):
            result["param_counts"] = count_parameters(model)
        train_result = train_neural(
            model, train_loader, val_loader,
            run_dir=run_dir, seed=seed, max_epochs=max_ep, device=device,
        )
        test_metrics = evaluate_test(model, test_loader, device)
        result["train"] = train_result
        result["test_metrics"] = test_metrics
        result["model_card"] = model.model_card
        model.save_checkpoint(run_dir)

    # Save result
    with (run_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)

    if "test_metrics" in result:
        auroc = result["test_metrics"].get("micro_auroc", "N/A")
        bce = result["test_metrics"].get("bce", "N/A")
        if isinstance(auroc, float):
            print(f"  Test: AUROC={auroc:.4f}  BCE={bce:.4f}")

    return result


ALL_MODELS = ["count_lightgbm", "retain", "ehr_bert", "behrt", "medbert", "cehrbert", "dtr"]


def main():
    parser = argparse.ArgumentParser(description="Synthetic baseline runner")
    parser.add_argument("--scenario", default="S2", choices=["S0", "S1", "S2", "S3", "all"])
    parser.add_argument("--models", default="all",
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 epochs)")
    parser.add_argument("--data-seed", type=int, default=20260922)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = Config(data_seed=args.data_seed)
    data_base = cfg.data_dir() / "controlled"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "results" / "baselines" / "synthetic"

    scenarios = ["S0", "S1", "S2", "S3"] if args.scenario == "all" else [args.scenario]
    models = ALL_MODELS if args.models == "all" else args.models.split(",")

    all_results: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        scenario_dir = data_base / scenario
        if not scenario_dir.exists():
            print(f"WARNING: scenario dir not found: {scenario_dir}")
            continue

        for model_name in models:
            key = f"{model_name}_{scenario}"
            try:
                result = run_one_model(
                    model_name, scenario, scenario_dir, output_dir,
                    seed=args.seed, device=args.device, smoke=args.smoke,
                )
                all_results[key] = result
            except Exception as e:
                print(f"ERROR: {model_name} on {scenario}: {e}")
                import traceback
                traceback.print_exc()
                all_results[key] = {"error": str(e)}

    # Save combined results
    combined_path = output_dir / "all_results.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {combined_path}")


if __name__ == "__main__":
    main()
