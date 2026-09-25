#!/usr/bin/env python3
"""Evaluate trained baselines on counterfactual mechanism tests.

Computes CF-RMSE_age, CF-RMSE_lag, and Surface RMSE against the oracle,
plus S5-specific persistence-group surface RMSE and ordering checks.

S0–S3 use the existing age×temporal mechanism classification.
S5 uses heterogeneous-persistence classification (separate labels).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "synthetic_age_temporal"))

from synthetic_age_temporal.config import Config
from synthetic_age_temporal.dataset import make_loaders

from baselines.synthetic.runner import ALL_MODELS, DTR_ARMS, build_model
from baselines.synthetic.data_adapter import (
    BENCHMARK_SCENARIOS,
    CORE_SCENARIOS,
    model_batch,
    resolve_scenarios,
)
from baselines.synthetic.result_schema import from_train_and_cf
from baselines.synthetic.s5_eval import find_multigroup_template, full_s5_counterfactual_report
from baselines.common.counterfactual import full_counterfactual_report
from baselines.common.training import get_device
from baselines.lightgbm.model import build_features_from_batch


def make_predict_fns(
    model: Any,
    batch_template: dict[str, torch.Tensor],
    device: torch.device,
    n_codes: int,
) -> tuple[Callable, Callable, Callable, Callable]:
    """Create prediction functions varying age and lag.

    Returns (predict_age, predict_lag, predict_surface, predict_batch).
    ``predict_batch`` is used by S5 group evaluation (receives a full batch).
    """
    is_lgb = getattr(model, "name", "") == "count_lightgbm"

    def _predict_from_batch(b_in: dict[str, torch.Tensor]) -> np.ndarray:
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b_in.items()}
        # Models never see eval-only keys
        b_model = model_batch(b)
        if is_lgb:
            X = build_features_from_batch(b_model, n_codes)
            return model.predict_proba(X)[0]
        b_model = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in b_model.items()
        }
        with torch.no_grad():
            out = model.predict(b_model)
        logits = out.logits[0].cpu().numpy()
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    def _predict(a: float | None = None, lag: float | None = None) -> np.ndarray:
        b = {k: v.clone() for k, v in batch_template.items() if torch.is_tensor(v)}
        for k, v in batch_template.items():
            if k not in b:
                b[k] = v
        if a is not None:
            b["age"] = torch.full_like(b["age"], float(a))
            b["z_age"] = (b["age"] - 9.0) / 9.0
        if lag is not None:
            b["lag_days"] = torch.full_like(b["lag_days"], float(lag))
            b["tau"] = torch.log1p(b["lag_days"] / 7.0)
        return _predict_from_batch(b)

    return (
        lambda a: _predict(a=a, lag=None),
        lambda lag: _predict(a=None, lag=lag),
        lambda a, lag: _predict(a=a, lag=lag),
        _predict_from_batch,
    )


def _build_oracle_fns(
    template: dict[str, torch.Tensor],
    itos: dict[int, str],
    specs: list[dict[str, Any]],
    scenario: str,
    theta0: float,
    beta: float,
):
    from synthetic_age_temporal.ground_truth import ExampleSignals, compute_target_logits

    def _oracle_predict(a: float | None = None, lag: float | None = None) -> np.ndarray:
        b = {k: v.clone() for k, v in template.items() if torch.is_tensor(v)}
        curr_a = float(a) if a is not None else float(template["age"][0])
        types = template["type_ids"][0].cpu().numpy()
        codes = template["code_ids"][0].cpu().numpy()
        lags = template["lag_days"][0].cpu().numpy()
        pad = template["padding_mask"][0].cpu().numpy()
        is_sig = template["is_signal"][0].cpu().numpy()

        sig_codes, sig_lags = [], []
        for i in range(len(types)):
            if pad[i] or not is_sig[i]:
                continue
            name = itos.get(int(codes[i]))
            if name is None or not str(name).startswith("SYN_SIGNAL_"):
                # Fallback for robustness
                name = f"SYN_SIGNAL_{chr(ord('A') + (int(codes[i]) - 3) % 12)}"
            sig_codes.append(str(name))
            sig_lags.append(float(lag) if lag is not None else float(lags[i]))

        curr_sig = ExampleSignals(
            codes=np.array(sig_codes, dtype=object),
            lag_days=np.array(sig_lags, dtype=np.float64),
            tau=np.log1p(np.array(sig_lags, dtype=np.float64) / 7.0),
            times=np.array([], dtype="datetime64[ns]"),
        )
        _, probs, _, _ = compute_target_logits(
            age=curr_a,
            signals=curr_sig,
            specs=specs,
            scenario=scenario,
            theta0=theta0,
            beta=beta,
            noise=np.zeros(len(specs)),
        )
        return probs

    return (
        lambda a: _oracle_predict(a=a, lag=None),
        lambda lag: _oracle_predict(a=None, lag=lag),
        lambda a, lag: _oracle_predict(a=a, lag=lag),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="S2",
        choices=list(BENCHMARK_SCENARIOS) + ["all", "core"],
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "baselines" / "synthetic"),
    )
    parser.add_argument("--data-seed", type=int, default=20260922)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scenarios = resolve_scenarios(args.scenario)
    results_dir = Path(args.results_dir)
    cfg = Config(data_seed=args.data_seed)
    device = get_device(args.device)

    for scenario in scenarios:
        scenario_dir = cfg.data_dir() / "controlled" / scenario
        print(f"\n=== Counterfactual eval: {scenario} ===")
        print(f"Loading data from {scenario_dir}")
        _, _, test_loader, vocab, info = make_loaders(scenario_dir, batch_size=1)
        n_codes = info["n_codes"]
        n_types = info["n_types"]
        n_targets = info["n_targets"]
        itos = dict(vocab.itos)

        template = None
        if scenario == "S5":
            template = find_multigroup_template(test_loader, itos, min_groups=3)
        if template is None:
            for batch in test_loader:
                if batch["is_signal"].any():
                    template = {k: v for k, v in batch.items()}
                    break
        if template is None:
            raise ValueError(f"No signal event found in test set for {scenario}.")

        specs = json.loads((scenario_dir / "target_specs.json").read_text())
        meta = json.loads((scenario_dir / "meta.json").read_text())
        theta0 = float(meta["theta0"])
        beta = float(meta["beta_true"])

        o_age, o_lag, o_surf = _build_oracle_fns(
            template, itos, specs, scenario, theta0, beta,
        )

        reports: dict[str, Any] = {}

        for model_name in ALL_MODELS:
            arms = [f"dtr_{arm}" for arm in DTR_ARMS] if model_name == "dtr" else [model_name]

            for arm_name in arms:
                model_dir = results_dir / arm_name / scenario
                if not (model_dir / "result.json").exists():
                    continue

                print(f"Evaluating {arm_name} on {scenario}...")
                s0_rmse = None
                if scenario in CORE_SCENARIOS and scenario != "S0":
                    s0_dir = results_dir / arm_name / "S0"
                    if (s0_dir / "cf_report.json").exists():
                        with (s0_dir / "cf_report.json").open() as f:
                            s0_rmse = json.load(f)["cf_rmse_age"]

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

                if hasattr(model, "to"):
                    model.to(device)

                p_age, p_lag, p_surf, p_batch = make_predict_fns(
                    model, template, device, n_codes,
                )

                try:
                    if scenario == "S5":
                        # Universal CF on full history
                        base = full_counterfactual_report(
                            p_age, p_lag, p_surf, o_age, o_lag, o_surf, cf_age_rmse_s0=None,
                        )
                        # S5 group metrics + heterogeneous classification
                        s5 = full_s5_counterfactual_report(
                            p_batch, template, itos, specs, theta0, beta,
                            cf_rmse_age=base["cf_rmse_age"],
                            cf_rmse_lag=base["cf_rmse_lag"],
                            surface_rmse_full=base["surface_rmse"],
                        )
                        report = {
                            **base,
                            **s5,
                            # Override classification with S5-specific label
                            "mechanism_classification": s5["mechanism_classification"],
                        }
                    else:
                        report = full_counterfactual_report(
                            p_age, p_lag, p_surf, o_age, o_lag, o_surf, cf_age_rmse_s0=s0_rmse,
                        )
                except Exception as e:
                    print(f"ERROR evaluating {arm_name} on {scenario}: {e}")
                    import traceback
                    traceback.print_exc()
                    reports[arm_name] = {
                        "model": arm_name,
                        "scenario": scenario,
                        "error": str(e),
                    }
                    continue

                report["model"] = arm_name
                report["scenario"] = scenario
                reports[arm_name] = report

                with (model_dir / "cf_report.json").open("w") as f:
                    json.dump(report, f, indent=2)

                # Merge into result.json if present
                result_path = model_dir / "result.json"
                if result_path.exists():
                    with result_path.open() as f:
                        result = json.load(f)
                    result["cf_report"] = report
                    schema = from_train_and_cf(
                        scenario=scenario,
                        model=arm_name,
                        test_metrics=result.get("test_metrics"),
                        cf_report=report,
                    )
                    result["benchmark_record"] = schema
                    for k, v in schema.items():
                        if k not in ("scenario", "model"):
                            result[k] = v
                    with result_path.open("w") as f:
                        json.dump(result, f, indent=2, default=str)

        with (results_dir / f"cf_summary_{scenario}.json").open("w") as f:
            json.dump(reports, f, indent=2)
        print(f"Wrote cf_summary_{scenario}.json ({len(reports)} models)")


if __name__ == "__main__":
    main()
