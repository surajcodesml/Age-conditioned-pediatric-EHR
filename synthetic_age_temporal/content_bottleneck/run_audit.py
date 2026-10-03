"""Part 1: content information audit over C01 checkpoints (no retrain)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.common.training import get_device
from content_bottleneck import C01_ARTIFACT_ROOT
import content_bottleneck as cb
from content_bottleneck.extract import extract_encounter_table
from content_bottleneck.oracle_substitution import run_oracle_substitution
from content_bottleneck.probes import (
    background_leakage,
    probe_oracle_content_vector,
    probe_signal_identity,
    retrieval_alignment,
)
from ladder.artifacts import write_json


def _mean_seed(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def audit_scenario(scenario: str, seeds: list[int], data_seed: int, device, max_batches) -> dict[str, Any]:
    seed_reports = []
    for seed in seeds:
        print(f"audit extract {scenario} seed {seed}", flush=True)
        table = extract_encounter_table(
            scenario=scenario, seed=seed, data_seed=data_seed,
            device=device, max_batches=max_batches,
        )
        out_dir = cb.ARTIFACT_ROOT / "audit" / scenario.lower() / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "encounter_table.npz", **{
            k: v for k, v in table.items() if isinstance(v, np.ndarray)
        })

        mem = table["signal_membership"]
        a1_v = probe_signal_identity(table["v"], mem, n_codes=table["n_codes"], seed=seed)
        a1_pre = probe_signal_identity(table["pre_pool"], mem, n_codes=table["n_codes"], seed=seed)
        a2_v = probe_oracle_content_vector(table["v"], table["w_true"], mem, seed=seed)
        a2_pre = probe_oracle_content_vector(table["pre_pool"], table["w_true"], mem, seed=seed)
        a2_raw = probe_oracle_content_vector(mem, table["w_true"], mem, seed=seed)
        a3 = background_leakage(table["u"], mem, table["oracle_pre_decay"])
        b = retrieval_alignment(table["u"], table["oracle_pre_decay"], mem)

        print(f"audit oracle subst {scenario} seed {seed}", flush=True)
        subst = run_oracle_substitution(
            scenario=scenario, seed=seed, data_seed=data_seed,
            device=device, max_batches=max_batches,
        )
        report = {
            "scenario": scenario,
            "seed": seed,
            "A1_signal_identity_v": a1_v,
            "A1_signal_identity_pre_pool": a1_pre,
            "A2_oracle_w_from_v": a2_v,
            "A2_oracle_w_from_pre_pool": a2_pre,
            "A2_oracle_w_from_membership_upper_bound": a2_raw,
            "A3_background_leakage": a3,
            "B_retrieval_alignment": b,
            "C_oracle_substitution": subst,
            "W_has_both_signs": bool((table["W"] > 0).any() and (table["W"] < 0).any()),
            "n_encounters": int(table["u"].shape[0]),
        }
        write_json(out_dir / "audit_seed.json", report)
        seed_reports.append(report)
    return {"scenario": scenario, "seeds": seed_reports}


def _flatten_audit(all_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for scen in all_reports:
        for r in scen["seeds"]:
            base = {"scenario": r["scenario"], "seed": r["seed"]}
            rows.append({**base, "metric": "A1_v_mean_auroc", "value": r["A1_signal_identity_v"]["mean_auroc"]})
            rows.append({**base, "metric": "A1_pre_mean_auroc", "value": r["A1_signal_identity_pre_pool"]["mean_auroc"]})
            rows.append({**base, "metric": "A2_v_rmse", "value": r["A2_oracle_w_from_v"]["rmse"]})
            rows.append({**base, "metric": "A2_v_corr", "value": r["A2_oracle_w_from_v"]["corr"]})
            rows.append({**base, "metric": "A2_v_r2", "value": r["A2_oracle_w_from_v"]["r2"]})
            rows.append({**base, "metric": "A2_raw_rmse", "value": r["A2_oracle_w_from_membership_upper_bound"]["rmse"]})
            rows.append({**base, "metric": "A2_raw_corr", "value": r["A2_oracle_w_from_membership_upper_bound"]["corr"]})
            rows.append({**base, "metric": "A3_u_mean_bg", "value": r["A3_background_leakage"]["u_mean_background"]})
            rows.append({**base, "metric": "A3_u_mean_sig", "value": r["A3_background_leakage"]["u_mean_signal"]})
            rows.append({**base, "metric": "B_corr_u_mean_abs", "value": r["B_retrieval_alignment"]["corr_u_mean_abs_oracle"]})
            rows.append({**base, "metric": "B_corr_u_max_abs", "value": r["B_retrieval_alignment"]["corr_u_max_abs_oracle"]})
            c = r["C_oracle_substitution"]
            for arm in ("C1", "C2"):
                for m in ("bce", "auroc", "auprc", "surface_rmse"):
                    rows.append({**base, "metric": f"{arm}_{m}", "value": c[arm][m]})
            for m, v in c["delta_C1_minus_C2"].items():
                rows.append({**base, "metric": f"delta_C1_C2_{m}", "value": v})
    return rows


def write_audit_outputs(all_reports: list[dict[str, Any]], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = _flatten_audit(all_reports)
    csv_path = root / "content_information_audit.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "seed", "metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_json(root / "content_information_audit.json", all_reports)

    # Oracle decomposition table (mean over seeds)
    decomp_rows = []
    for scen in all_reports:
        scenario = scen["scenario"]
        for arm in ("C1", "C2"):
            for metric in ("bce", "auroc", "auprc", "surface_rmse"):
                vals = [r["C_oracle_substitution"][arm][metric] for r in scen["seeds"]]
                decomp_rows.append({
                    "scenario": scenario, "arm": arm, "metric": metric,
                    "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                })
    with (root / "oracle_content_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "arm", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(decomp_rows)
    # Also load C01/D00 for decomposition context if present
    write_json(root / "oracle_content_decomposition.json", decomp_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scenarios", nargs="+", default=["S0", "S1", "S2", "S3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-seed", type=int, default=20260922)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    root = Path(args.artifact_root) if args.artifact_root is not None else cb.ARTIFACT_ROOT
    cb.ARTIFACT_ROOT = root
    device = get_device(args.device)
    reports = []
    for scenario in args.scenarios:
        reports.append(audit_scenario(scenario, args.seeds, args.data_seed, device, args.max_batches))
    write_audit_outputs(reports, root)
    print(f"wrote audit under {root}", flush=True)


if __name__ == "__main__":
    main()
