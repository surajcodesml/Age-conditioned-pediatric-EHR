#!/usr/bin/env python3
"""Mechanism ablations + summary for Content-Persistence DTR ``*_new`` runs.

Computes β=0 / age-shuffle ΔBCE on the validation set and merges with
predictive + counterfactual metrics into ``dtr_new_mechanism_summary.json``.

Never reads or writes legacy unsuffixed ``dtr_temporal_only`` / ``dtr_age_temporal``
directories except for optional delta-vs-legacy reporting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "synthetic_age_temporal"))

from baselines.common.training import get_device
from baselines.synthetic.data_adapter import (
    BENCHMARK_SCENARIOS,
    make_dtr_baseline_loaders,
    resolve_scenarios,
)
from baselines.synthetic.runner import DTR_ARMS, build_model, dtr_arm_dirname
from train_dtr import ablations, evaluate


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -40, 40)))


def surface_metrics(
    model,
    device: torch.device,
    *,
    beta_true: float,
    theta0_true: float = 0.0,
) -> dict[str, float]:
    ages = np.arange(0, 19, 1.0)
    z = (ages - 9.0) / 9.0
    lam_true = _softplus(theta0_true + beta_true * z)
    with torch.no_grad():
        lam_hat = (
            model.temporal.lambda_of(
                torch.tensor(ages, dtype=torch.float32, device=device)
            )
            .detach()
            .cpu()
            .numpy()
        )
    lags = np.array([7.0, 30.0, 90.0, 180.0, 365.0])
    tau = np.log1p(lags / 7.0)
    g_true = np.exp(-lam_true[:, None] * tau[None, :])
    g_hat = np.exp(-lam_hat[:, None] * tau[None, :])
    return {
        "Surface_RMSE": float(np.sqrt(((g_true - g_hat) ** 2).mean())),
        "lambda_RMSE": float(np.sqrt(((lam_true - lam_hat) ** 2).mean())),
        "lambda_corr": float(np.corrcoef(lam_true, lam_hat)[0, 1])
        if np.std(lam_hat) > 1e-12
        else float("nan"),
    }


def eval_arm(
    *,
    results_dir: Path,
    arm: str,
    scenario: str,
    name_suffix: str,
    data_seed: int,
    device: torch.device,
) -> dict[str, Any] | None:
    arm_name = dtr_arm_dirname(arm, name_suffix)
    arm_dir = results_dir / arm_name / scenario
    result_path = arm_dir / "result.json"
    if not result_path.exists():
        print(f"  skip missing {arm_name}/{scenario}")
        return None

    with result_path.open() as f:
        result = json.load(f)

    _, val_raw, test_raw, _, info = make_dtr_baseline_loaders(
        scenario, data_seed=data_seed, batch_size=32
    )
    model = build_model(
        "dtr",
        info["n_codes"],
        info.get("n_types", 11),
        info["n_targets"],
        arm=arm,
    )
    model.load_checkpoint(arm_dir)
    model.to(device)

    test_m = evaluate(model._model, test_raw, device)
    theta0_hat = float(model.temporal.theta0.detach().cpu().reshape(-1)[0])
    beta_hat = float(model.temporal.beta.detach().cpu().reshape(-1)[0])

    out: dict[str, Any] = {
        "model": arm_name,
        "arm": arm,
        "scenario": scenario,
        "seed": result.get("seed", 0),
        "AUROC": float(test_m["micro_auroc"]),
        "AUPRC": float(test_m["micro_auprc"]),
        "BCE": float(test_m["bce"]),
        "theta0_hat": theta0_hat,
        "beta_hat": beta_hat,
        "delta_BCE_beta0": None,
        "delta_BCE_age_shuffle": None,
        "CF_RMSE_age": result.get("CF_RMSE_age")
        or (result.get("cf_report") or {}).get("cf_rmse_age"),
        "CF_RMSE_lag": result.get("CF_RMSE_lag")
        or (result.get("cf_report") or {}).get("cf_rmse_lag"),
        "Surface_RMSE": result.get("Surface_RMSE")
        or (result.get("cf_report") or {}).get("surface_rmse"),
        "mechanism_classification": result.get("mechanism_classification")
        or (result.get("cf_report") or {}).get("mechanism_classification"),
        "S5_Surface_RMSE_acute": result.get("S5_Surface_RMSE_acute"),
        "S5_Surface_RMSE_intermediate": result.get("S5_Surface_RMSE_intermediate"),
        "S5_Surface_RMSE_chronic": result.get("S5_Surface_RMSE_chronic"),
        "S5_Surface_RMSE_mean": result.get("S5_Surface_RMSE_mean"),
        "persistence_order_correct": result.get("persistence_order_correct"),
    }

    if arm == "age_temporal":
        abl = ablations(model._model, val_raw, device)
        out["delta_BCE_beta0"] = float(abl["delta_bce_beta0"])
        out["delta_BCE_age_shuffle"] = float(abl["delta_bce_shuffle_age"])
        meta = json.loads(
            (
                REPO_ROOT
                / "synthetic_age_temporal"
                / "outputs"
                / "data"
                / f"seed{data_seed}"
                / "controlled"
                / scenario
                / "meta.json"
            ).read_text()
        )
        surf = surface_metrics(
            model,
            device,
            beta_true=float(meta["beta_true"]),
            theta0_true=float(meta.get("theta0", 0.0)),
        )
        out["param_Surface_RMSE"] = surf["Surface_RMSE"]
        out["lambda_RMSE"] = surf["lambda_RMSE"]
        out["lambda_corr"] = surf["lambda_corr"]
        # Prefer universal CF surface when present; keep parametric as backup key.
        if out["Surface_RMSE"] is None:
            out["Surface_RMSE"] = surf["Surface_RMSE"]

    # Persist mechanism fields into result.json (same _new dir only).
    result.update(
        {
            "theta0_hat": theta0_hat,
            "beta_hat": beta_hat,
            "delta_BCE_beta0": out["delta_BCE_beta0"],
            "delta_BCE_age_shuffle": out["delta_BCE_age_shuffle"],
            "AUROC": out["AUROC"],
            "AUPRC": out["AUPRC"],
            "BCE": out["BCE"],
        }
    )
    for k in (
        "param_Surface_RMSE",
        "lambda_RMSE",
        "lambda_corr",
    ):
        if k in out and out[k] is not None:
            result[k] = out[k]
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=list(BENCHMARK_SCENARIOS) + ["all", "core"])
    ap.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "baselines" / "synthetic"),
    )
    ap.add_argument("--name-suffix", type=str, default="_new")
    ap.add_argument("--data-seed", type=int, default=20260922)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    device = get_device(args.device)
    scenarios = resolve_scenarios(args.scenario)
    suffix = args.name_suffix

    rows: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        print(f"\n=== Mechanism eval {scenario} ===", flush=True)
        for arm in DTR_ARMS:
            row = eval_arm(
                results_dir=results_dir,
                arm=arm,
                scenario=scenario,
                name_suffix=suffix,
                data_seed=args.data_seed,
                device=device,
            )
            if row is None:
                continue
            key = f"{dtr_arm_dirname(arm, suffix)}_{scenario}"
            rows[key] = row
            print(
                f"  {key}: AUPRC={row['AUPRC']:.4f} β={row['beta_hat']:+.4f} "
                f"Δβ0={row['delta_BCE_beta0']} shuffle={row['delta_BCE_age_shuffle']}",
                flush=True,
            )

    # Pairwise age_temporal_new − temporal_only_new
    deltas: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        at = rows.get(f"{dtr_arm_dirname('age_temporal', suffix)}_{scenario}")
        to = rows.get(f"{dtr_arm_dirname('temporal_only', suffix)}_{scenario}")
        if not at or not to:
            continue
        deltas[scenario] = {
            "delta_AUROC": at["AUROC"] - to["AUROC"],
            "delta_AUPRC": at["AUPRC"] - to["AUPRC"],
            "delta_BCE": at["BCE"] - to["BCE"],
            "age_temporal_beta_hat": at["beta_hat"],
            "temporal_only_beta_hat": to["beta_hat"],
            "delta_BCE_beta0": at["delta_BCE_beta0"],
            "delta_BCE_age_shuffle": at["delta_BCE_age_shuffle"],
            "CF_RMSE_age": at.get("CF_RMSE_age"),
            "CF_RMSE_lag": at.get("CF_RMSE_lag"),
            "Surface_RMSE": at.get("Surface_RMSE"),
        }

    summary = {
        "name_suffix": suffix,
        "architecture": "Content-Persistence DTR",
        "seed": 0,
        "arms": rows,
        "age_temporal_minus_temporal_only": deltas,
    }
    out_path = results_dir / f"dtr{suffix}_mechanism_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
