#!/usr/bin/env python3
"""Evaluate all trained baselines on counterfactual mechanism tests.

Computes CF-RMSE_age, CF-RMSE_lag, and Surface RMSE against the oracle,
and performs mechanism classification (Functional / Partial / None).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "synthetic_age_temporal"))

from synthetic_age_temporal.config import (
    SURFACE_AGES, SURFACE_LAGS_DAYS,
    Config, SCENARIO_SPECS,
)
from synthetic_age_temporal.dataset import make_loaders
from synthetic_age_temporal.evaluate import quick_ablation_deltas

from baselines.synthetic.runner import build_model
from baselines.common.counterfactual import full_counterfactual_report
from baselines.common.training import get_device

# For LightGBM feature builder
from baselines.lightgbm.model import build_features_from_batch

def make_predict_fns(
    model: Any,
    batch_template: dict[str, torch.Tensor],
    device: torch.device,
    n_codes: int,
) -> tuple[Callable, Callable, Callable]:
    """Create prediction functions varying age and lag."""
    is_lgb = getattr(model, "name", "") == "count_lightgbm"

    def _predict(a: float | None = None, lag: float | None = None) -> np.ndarray:
        b = {k: v.clone() for k, v in batch_template.items()}
        if a is not None:
            b["age"] = torch.full_like(b["age"], float(a))
            b["z_age"] = (b["age"] - 9.0) / 9.0
        if lag is not None:
            # We must map lag_days to tau for neural models.
            b["lag_days"] = torch.full_like(b["lag_days"], float(lag))
            # Tau = log1p(|lag| / 7)
            b["tau"] = torch.log1p(b["lag_days"] / 7.0)

        if is_lgb:
            X = build_features_from_batch(b, n_codes)
            # return proba
            return model.predict_proba(X)[0]
        else:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.no_grad():
                out = model.predict(b)
            logits = out.logits[0].cpu().numpy()
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    return (
        lambda a: _predict(a=a, lag=None),
        lambda lag: _predict(a=None, lag=lag),
        lambda a, lag: _predict(a=a, lag=lag),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="S2", choices=["S0", "S1", "S2", "S3"])
    parser.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "results" / "baselines" / "synthetic"))
    parser.add_argument("--data-seed", type=int, default=20260922)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Config(data_seed=args.data_seed)
    scenario_dir = cfg.data_dir() / "controlled" / args.scenario
    results_dir = Path(args.results_dir)

    print(f"Loading data from {scenario_dir}")
    _, _, test_loader, _, info = make_loaders(scenario_dir, batch_size=1)
    n_codes = info["n_codes"]
    n_types = info["n_types"]
    n_targets = info["n_targets"]
    target_idx = info.get("target_idx", list(range(n_targets)))

    # Get a batch template (first patient with a signal event)
    template = None
    for batch in test_loader:
        if batch["is_signal"].any():
            template = {k: v for k, v in batch.items()}
            break
    if template is None:
        raise ValueError("No signal event found in test set.")

    device = get_device(args.device)

    # 1. Oracle functions
    from synthetic_age_temporal.ground_truth import compute_target_logits, ExampleSignals
    
    specs = json.loads((scenario_dir / "target_specs.json").read_text())
    meta = json.loads((scenario_dir / "meta.json").read_text())
    theta0 = float(meta["theta0"])
    beta = float(meta["beta_true"])

    def _oracle_predict(a: float | None = None, lag: float | None = None) -> np.ndarray:
        b = {k: v.clone() for k, v in template.items()}
        curr_a = float(a) if a is not None else float(b["age"][0])
        types = b["type_ids"][0].cpu().numpy()
        codes = b["code_ids"][0].cpu().numpy()
        lags = b["lag_days"][0].cpu().numpy()
        
        sig_codes = []
        sig_lags = []
        for i, t in enumerate(types):
            if t == 1:  # SIGNAL_TYPE
                # reconstruct signal code
                sig_codes.append(f"SYN_SIGNAL_{chr(ord('A') + (codes[i] - 1) % 12)}")
                sig_lags.append(float(lag) if lag is not None else lags[i])

        curr_sig = ExampleSignals(
            codes=np.array(sig_codes, dtype=object),
            lag_days=np.array(sig_lags, dtype=np.float64),
            tau=np.log1p(np.array(sig_lags, dtype=np.float64) / 7.0),
            times=np.array([], dtype="datetime64[ns]")
        )

        logits, probs, _, _ = compute_target_logits(
            age=curr_a,
            signals=curr_sig,
            specs=specs,
            scenario=args.scenario,
            theta0=theta0,
            beta=beta,
            noise=np.zeros(len(specs))
        )
        return probs

    o_age = lambda a: _oracle_predict(a=a, lag=None)
    o_lag = lambda lag: _oracle_predict(a=None, lag=lag)
    o_surf = lambda a, lag: _oracle_predict(a=a, lag=lag)

    # 2. Iterate models
    reports = {}
    from baselines.synthetic.runner import ALL_MODELS, DTR_ARMS

    for model_name in ALL_MODELS:
        if model_name == "dtr":
            arms = [f"dtr_{arm}" for arm in DTR_ARMS]
        else:
            arms = [model_name]

        for arm_name in arms:
            model_dir = results_dir / arm_name / args.scenario
            if not (model_dir / "result.json").exists():
                continue

            print(f"Evaluating {arm_name} on {args.scenario}...")
            # S0 check for false interaction
            s0_rmse = None
            if args.scenario != "S0":
                s0_dir = results_dir / arm_name / "S0"
                if (s0_dir / "cf_report.json").exists():
                    with (s0_dir / "cf_report.json").open() as f:
                        s0_rmse = json.load(f)["cf_rmse_age"]

            # Load model
            if model_name == "dtr":
                arm_type = arm_name.replace("dtr_", "")
                model = build_model("dtr", n_codes, n_types, n_targets, arm=arm_type)
            else:
                model = build_model(model_name, n_codes, n_types, n_targets)

            try:
                model.load_checkpoint(model_dir)
            except Exception as e:
                print(f"Failed to load checkpoint for {arm_name}: {e}")
                continue

            if hasattr(model, 'to'):
                model.to(device)

            p_age, p_lag, p_surf = make_predict_fns(model, template, device, n_codes)

            report = full_counterfactual_report(
                p_age, p_lag, p_surf, o_age, o_lag, o_surf, cf_age_rmse_s0=s0_rmse
            )
            report["model"] = arm_name
            report["scenario"] = args.scenario
            reports[arm_name] = report

            # Save report
            with (model_dir / "cf_report.json").open("w") as f:
                json.dump(report, f, indent=2)

    # Save summary
    with (results_dir / f"cf_summary_{args.scenario}.json").open("w") as f:
        json.dump(reports, f, indent=2)

if __name__ == "__main__":
    main()
