#!/usr/bin/env python3
"""End-to-end orchestration for the synthetic age × temporal benchmark."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import ARMS, DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_DIR, INTERACTION_STRENGTHS, SCENARIOS
from plots import generate_all

PKG = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(PKG))


def classify_scenario(metrics: dict) -> str:
    """MECHANISM RECOVERED / PARTIALLY RECOVERED / NOT RECOVERED."""
    scen = metrics.get("scenario")
    arm = metrics.get("arm")
    if arm != "age_temporal":
        return "N/A"
    abl = metrics.get("ablations", {})
    rec = metrics.get("recovery", {})
    d_shuf = abl.get("delta_bce_shuffle_age", 0.0)
    d_b0 = abl.get("delta_bce_beta0", 0.0)
    sign_ok = rec.get("sign_match")
    corr = rec.get("corr_lambda")
    corr = -1.0 if corr is None or (isinstance(corr, float) and corr != corr) else corr

    if scen in ("S0", "S1"):
        # Should remain inert: small shuffle effect, beta near 0.
        beta = abs(metrics.get("beta_hat", 99))
        if abs(d_shuf) < 0.02 and beta < 0.5:
            return "MECHANISM RECOVERED"
        if abs(d_shuf) < 0.05:
            return "PARTIALLY RECOVERED"
        return "NOT RECOVERED"

    # S2 / S3
    checks = [
        bool(sign_ok),
        d_shuf > 0.01,
        d_b0 > 0.005,
        corr > 0.5,
    ]
    n = sum(checks)
    if n >= 3:
        return "MECHANISM RECOVERED"
    if n >= 2:
        return "PARTIALLY RECOVERED"
    return "NOT RECOVERED"


def write_report(output_dir: Path, results_dir: Path) -> None:
    data_root = output_dir / "data"
    seeds = sorted(data_root.glob("seed*")) if data_root.exists() else []
    lines = [
        "# Synthetic age × temporal benchmark report",
        "",
        "Semi-synthetic pediatric EHR benchmark using Synthea trajectories with a",
        "known softplus age × temporal ground-truth mechanism.",
        "",
        "## Core success criterion",
        "",
        "> The model recovers and functionally uses the known age × temporal",
        "> mechanism when it exists, and remains inert when it does not.",
        "",
    ]
    # Manifest
    man = output_dir / "synthea" / "synthea_manifest.json"
    if man.exists():
        with man.open() as f:
            m = json.load(f)
        lines += [
            "## Synthea cohort",
            "",
            f"- version: `{m.get('synthea_version')}`",
            f"- commit: `{m.get('synthea_commit')}`",
            f"- n_patients: {m.get('n_patients')}",
            f"- n_events (background): {m.get('n_events')}",
            f"- reused_existing_cohort: {m.get('reused_existing_cohort')}",
            f"- generation seeds: `{json.dumps(m.get('generation_seed'))}`",
            "",
            "### Age distribution",
            "",
            "```json",
            json.dumps(m.get("age_distribution"), indent=2),
            "```",
            "",
            "### Event counts",
            "",
            "```json",
            json.dumps(m.get("event_counts"), indent=2),
            "```",
            "",
        ]

    lines += [
        "## Target-generation equations",
        "",
        r"- $z(a)=(a-9)/9$",
        r"- $\tau=\log(1+\Delta t/7)$",
        r"- $\lambda_{\mathrm{true}}(a)=\mathrm{softplus}(\theta_0+\beta_{\mathrm{true}} z(a))$",
        r"- $R(a,\tau)=\exp[-\lambda_{\mathrm{true}}(a)\tau]$",
        r"- Interaction: $\eta_k=b_k+\gamma_k z(a_*)+\sum_j w_{kj} R(a_*,\tau_j)+\epsilon$",
        r"- $Y_k\sim\mathrm{Bernoulli}(\sigma(\eta_k))$",
        "",
        "## Scenario parameters",
        "",
        "| Scenario | β_true | notes |",
        "|---|---|---|",
        "| S0 | 0 | temporal only, no interaction |",
        "| S1 | 0 | age main effect only |",
        "| S2 | <0 (default −2) | developmental interaction |",
        "| S3 | >0 (default +2) | reversed interaction |",
        "",
    ]

    classifications = {}
    run_root = output_dir / "runs"
    if run_root.exists():
        lines += ["## Neural results", ""]
        for metrics_path in sorted(run_root.rglob("metrics.json")):
            with metrics_path.open() as f:
                met = json.load(f)
            label = classify_scenario(met)
            key = f"{met.get('cohort')}/{met.get('scenario')}/{met.get('arm')}"
            classifications[key] = label
            lines += [
                f"### {key} (seed d={met.get('data_seed')} m={met.get('model_seed')})",
                "",
                f"- classification: **{label}**",
                f"- β̂={met.get('beta_hat'):.4f} (true {met.get('beta_true')})",
                f"- test micro AUROC={met.get('test', {}).get('micro_auroc')}",
                f"- shuffle-age ΔBCE={met.get('ablations', {}).get('delta_bce_shuffle_age')}",
                f"- β=0 ΔBCE={met.get('ablations', {}).get('delta_bce_beta0')}",
                f"- λ corr={met.get('recovery', {}).get('corr_lambda')}",
                f"- surface RMSE={met.get('recovery', {}).get('RMSE_surface')}",
                "",
            ]

    lines += [
        "## Scenario recovery summary",
        "",
        "| Setting | Classification |",
        "|---|---|",
    ]
    for k, v in sorted(classifications.items()):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## Figures",
        "",
        f"See `{results_dir / 'figures'}` for fig1–fig6 (PNG + SVG).",
        "",
        "## Notes",
        "",
        "- Production MIMIC/NCH preprocessing was not modified.",
        "- Ground-truth mechanism fields are stored separately from model inputs.",
        "- Prediction-time attention uses cutoff age/time only (no future timestamps).",
        "",
    ]
    report_path = PKG / "report_runs.md"
    report_path.write_text("\n".join(lines))
    print("Wrote", report_path)
    print("Curated scientific report: ", PKG / "report.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--data-seed", type=int, default=20260922)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Controlled cohort only; fewer epochs.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["no_age", "age_only", "temporal_only", "age_temporal"],
    )
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--strength-sweep", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        cohorts = ["controlled"] if args.quick else ["controlled", "full"]
        strengths = ["medium"]
        if args.strength_sweep:
            strengths = list(INTERACTION_STRENGTHS.keys())
        run(
            [
                PY,
                "generate_synthea.py",
                "--output-dir",
                str(args.output_dir / "synthea"),
            ]
        )
        cmd = [
            PY,
            "build_benchmark.py",
            "--output-dir",
            str(args.output_dir),
            "--data-seed",
            str(args.data_seed),
            "--cohorts",
            *cohorts,
            "--scenarios",
            *args.scenarios,
            "--strengths",
            *strengths,
        ]
        run(cmd)

    # Unit tests (always).
    data_root = args.output_dir / "data" / f"seed{args.data_seed}"
    run([PY, "tests/test_sanity.py", str(data_root)])

    # Baselines.
    for scen in args.scenarios:
        sdir = data_root / "controlled" / scen
        if sdir.exists():
            run([PY, "baselines.py", "--scenario-dir", str(sdir)])

    if not args.skip_train:
        epochs = args.epochs if args.epochs is not None else (5 if args.quick else 25)
        for scen in args.scenarios:
            for arm in args.arms:
                run(
                    [
                        PY,
                        "train.py",
                        "--scenario",
                        scen,
                        "--arm",
                        arm,
                        "--cohort",
                        "controlled",
                        "--data-seed",
                        str(args.data_seed),
                        "--model-seed",
                        str(args.model_seed),
                        "--epochs",
                        str(epochs),
                        "--device",
                        args.device,
                        "--scenario-dir",
                        str(data_root / "controlled" / scen),
                    ]
                )
        if args.strength_sweep:
            for strength in INTERACTION_STRENGTHS:
                if strength == "medium":
                    continue
                sdir = data_root / "controlled" / f"S2_{strength}"
                if not sdir.exists():
                    continue
                run(
                    [
                        PY,
                        "train.py",
                        "--scenario",
                        "S2",
                        "--arm",
                        "age_temporal",
                        "--strength",
                        strength,
                        "--cohort",
                        "controlled",
                        "--data-seed",
                        str(args.data_seed),
                        "--model-seed",
                        str(args.model_seed),
                        "--epochs",
                        str(epochs),
                        "--device",
                        args.device,
                        "--scenario-dir",
                        str(sdir),
                    ]
                )

    if not args.skip_plots:
        generate_all(
            args.results_dir / "figures",
            metrics_root=args.output_dir / "runs",
        )

    write_report(args.output_dir, args.results_dir)


if __name__ == "__main__":
    main()
