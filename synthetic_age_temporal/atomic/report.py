"""Narrative report from saved summary and paired-delta tables."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from atomic import REPO_ROOT
from atomic.config import EXPERIMENT_ORDER, experiment_configs

LOWER_BETTER = {
    "bce", "surface_rmse", "cf_rmse_age", "cf_rmse_lag",
    "gate_signal_rmse", "gate_signal_mae", "gate_surface_rmse", "gate_surface_mae",
}
HIGHER_BETTER = {
    "auroc", "auprc", "delta_bce_beta0", "delta_bce_gate_age_shuffle",
    "delta_bce_full_age_shuffle", "gate_signal_correlation", "gate_surface_correlation",
    "delta_auroc_at_minus_to", "delta_auprc_at_minus_to",
}


def _load(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _paired_stats(rows: list[dict[str, str]], experiment: str, scenario: str, arm: str, metric: str) -> tuple[float | None, float | None, int]:
    values = [
        float(row["delta"])
        for row in rows
        if row["experiment"] == experiment and row["scenario"] == scenario and row["arm"] == arm and row["metric"] == metric
    ]
    if not values:
        return None, None, 0
    import numpy as np
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, int(arr.size)


def _fmt(mean: float | None, std: float | None, n: int) -> str:
    if mean is None:
        return "—"
    if n <= 1:
        return f"{mean:+.4f} (n=1)"
    return f"{mean:+.4f} ± {std:.4f}"


def _judge(metric: str, mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "missing"
    if metric in LOWER_BETTER:
        direction = "favorable" if mean < 0 else "unfavorable"
    elif metric in HIGHER_BETTER:
        direction = "favorable" if mean > 0 else "unfavorable"
    else:
        direction = "descriptive"
    if abs(mean) > std:
        return f"{direction}; |mean paired difference| exceeds the seed standard deviation"
    return f"{direction}; unresolved because the seed standard deviation is at least as large as the mean paired difference"


def write_report(root: Path, report_path: Path | None = None) -> Path:
    root = Path(root)
    paired = _load(root / "paired_delta_vs_C00.csv")
    summary = _load(root / "atomic_followup_summary.csv")
    cards = {cfg["experiment_id"]: cfg for cfg in experiment_configs()}
    out = Path(report_path) if report_path else REPO_ROOT / "reports" / "dtr_atomic_followup.md"
    lines = [
        "# Atomic DTR follow-up",
        "",
        "Comparisons are paired by seed against C00. Tables were computed from saved predictions and mechanism arrays.",
        "",
        "## Protocol",
        "",
        "- C00 retrains the current Content-Persistence DTR on seeds 0–4 with matched initialization and the same data order in both arms.",
        "- Accept/reject decisions use C00, not the earlier single-seed dtr_age_temporal_new result.",
        "- Training budget matches that reference: AdamW, lr 3e-4, weight decay 0.01, batch 32, 25 epochs, patience 5, minimum 12 epochs.",
        "- C01 is the only optimizer change. Stage B trains theta0/beta for 5 epochs at 10× learning rate and zero weight decay, with the rest frozen.",
        "- delta_bce_gate_age_shuffle shuffles age only inside the temporal gate. The additive age head keeps the true age.",
        "- delta_bce_full_age_shuffle is the previous whole-model age shuffle.",
        "- Gate recovery for content-dependent models is the forward gate on signal encounters versus the oracle gate at the same age and tau. Mixture models use g_eff = sum_k pi_k exp(-lambda_k tau), not the mean of lambda_k.",
        "- C04 also stores a global gate surface because its lambda does not depend on content.",
        "- Matched-arm deltas are BCE(temporal_only) − BCE(age_temporal), and age_temporal minus temporal_only for AUROC and AUPRC.",
        "",
        f"Artifact root: `{root}`",
        "",
    ]
    failures_path = root / "failures.json"
    if failures_path.exists():
        failures = json.loads(failures_path.read_text())
        lines.append("## Recorded failures")
        lines.append("")
        for item in failures:
            lines.append(f"- {item.get('experiment')} {item.get('scenario')} {item.get('arm')} seed {item.get('seed')}: {item.get('error')}")
        lines.append("")

    def cell(experiment: str, scenario: str, arm: str, metric: str) -> str:
        hits = [
            row for row in summary
            if row["experiment"] == experiment and row["scenario"] == scenario and row["arm"] == arm and row["metric"] == metric
        ]
        if not hits:
            return "—"
        row = hits[0]
        mean = float(row["mean"])
        std = float(row["std"])
        n = int(row["n"])
        if n <= 1:
            return f"{mean:.4f}"
        return f"{mean:.4f} ± {std:.4f}"

    focus = (
        ("surface_rmse", "S2 surface RMSE"),
        ("gate_signal_rmse", "S2 gate RMSE"),
        ("delta_bce_gate_age_shuffle", "S2 gate-shuffle ΔBCE"),
        ("delta_bce_beta0", "S2 β=0 ΔBCE"),
        ("bce", "S2 BCE"),
        ("auprc", "S2 AUPRC"),
        ("delta_bce_to_minus_at", "S2 matched BCE TO−AT"),
    )
    lines.append("## Paired differences versus C00")
    lines.append("")
    lines.append("Age-temporal arm unless the metric is a matched-arm delta. Cells are mean ± standard deviation of seed-paired (candidate − C00).")
    lines.append("")
    header = "| Experiment | " + " | ".join(label for _, label in focus) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in focus) + " |")
    for experiment in EXPERIMENT_ORDER:
        if experiment == "C00_current_dtr":
            continue
        cells = []
        for metric, _label in focus:
            arm = "matched" if metric.startswith("delta_bce_to") or metric.startswith("delta_auroc") or metric.startswith("delta_auprc") else "age_temporal"
            mean, std, n = _paired_stats(paired, experiment, "S2", arm, metric)
            cells.append(_fmt(mean, std, n))
        lines.append(f"| {experiment} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Levels on C00 and each candidate")
    lines.append("")
    lines.append("| Experiment | S2 surface | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 β=0 ΔBCE | S0 gate-shuffle ΔBCE | S0 β=0 ΔBCE | S2 β mean | S3 β mean | S2 AUROC | S2 AUPRC |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for experiment in EXPERIMENT_ORDER:
        lines.append("| " + " | ".join([
            experiment,
            cell(experiment, "S2", "age_temporal", "surface_rmse"),
            cell(experiment, "S2", "age_temporal", "gate_signal_rmse"),
            cell(experiment, "S2", "age_temporal", "delta_bce_gate_age_shuffle"),
            cell(experiment, "S2", "age_temporal", "delta_bce_beta0"),
            cell(experiment, "S0", "age_temporal", "delta_bce_gate_age_shuffle"),
            cell(experiment, "S0", "age_temporal", "delta_bce_beta0"),
            cell(experiment, "S2", "age_temporal", "beta_mean"),
            cell(experiment, "S3", "age_temporal", "beta_mean"),
            cell(experiment, "S2", "age_temporal", "auroc"),
            cell(experiment, "S2", "age_temporal", "auprc"),
        ]) + " |")
    lines.append("")

    carry = []
    for experiment in EXPERIMENT_ORDER:
        if experiment == "C00_current_dtr":
            continue
        card = cards[experiment]
        lines.append(f"## {experiment}")
        lines.append("")
        lines.append(f"Hypothesis: {card['hypothesis']}")
        lines.append("")
        lines.append(f"Single change: {card['change']}")
        lines.append("")
        lines.append(f"Artifacts: `{root / experiment}`")
        lines.append("")
        observed_support = []
        for scenario, arm, metric in (
            ("S2", "age_temporal", "surface_rmse"),
            ("S2", "age_temporal", "gate_signal_rmse"),
            ("S2", "age_temporal", "cf_rmse_age"),
            ("S2", "age_temporal", "cf_rmse_lag"),
            ("S2", "age_temporal", "delta_bce_gate_age_shuffle"),
            ("S2", "age_temporal", "delta_bce_beta0"),
            ("S2", "age_temporal", "bce"),
            ("S2", "age_temporal", "auroc"),
            ("S2", "age_temporal", "auprc"),
            ("S2", "matched", "delta_bce_to_minus_at"),
            ("S0", "age_temporal", "delta_bce_gate_age_shuffle"),
            ("S0", "age_temporal", "delta_bce_beta0"),
            ("S0", "age_temporal", "gate_signal_rmse"),
            ("S3", "age_temporal", "surface_rmse"),
            ("S3", "age_temporal", "gate_signal_rmse"),
        ):
            mean, std, n = _paired_stats(paired, experiment, scenario, arm, metric)
            judgment = _judge(metric, mean, std)
            lines.append(f"- {scenario} {arm} {metric}: {_fmt(mean, std, n)}. {judgment}.")
            if judgment.startswith("favorable; |mean"):
                observed_support.append((scenario, metric))
        s2_beta = cell(experiment, "S2", "age_temporal", "beta_mean")
        s3_beta = cell(experiment, "S3", "age_temporal", "beta_mean")
        lines.append(f"- Observed S2 β mean {s2_beta}; S3 β mean {s3_beta}. S3 should reverse the S2 developmental sign.")
        lines.append("")
        recovery = any(metric in {"surface_rmse", "gate_signal_rmse"} and scenario in {"S2", "S3"} for scenario, metric in observed_support)
        predictive_harm = False
        for metric in ("bce", "auprc"):
            mean, std, _n = _paired_stats(paired, experiment, "S2", "age_temporal", metric)
            judgment = _judge(metric, mean, std)
            if judgment.startswith("unfavorable; |mean"):
                predictive_harm = True
        if recovery and not predictive_harm:
            carry.append(experiment)
            lines.append("Supported interpretation: at least one S2/S3 gate or prediction-surface comparison improves beyond seed noise, and S2 BCE/AUPRC do not worsen beyond seed noise.")
        elif recovery and predictive_harm:
            lines.append("Supported interpretation: a recovery metric moves beyond seed noise, and S2 predictive BCE or AUPRC also worsens beyond seed noise.")
        else:
            lines.append("Supported interpretation: no S2/S3 gate or prediction-surface gain is larger than the paired seed noise.")
        lines.append("Unresolved where the paired seed standard deviation is at least as large as the mean difference.")
        lines.append("")

    lines.append("## Changes with evidence for a later combination stage")
    lines.append("")
    lines.append("This list is not a license to combine the changes yet. A name appears when a paired S2 or S3 surface or gate RMSE improvement exceeds seed noise and S2 BCE/AUPRC do not show a supported worsening.")
    lines.append("")
    if carry:
        for name in carry:
            lines.append(f"- {name}")
    else:
        lines.append("- None of C01–C06 met that paired rule.")
    lines.append("")
    lines.append("## Figure paths")
    lines.append("")
    fig = root / "figures"
    for name in (
        "prediction_surface_S2.png",
        "prediction_surface_S3.png",
        "gate_signal_rmse_s2.png",
        "lambda_curves.png",
        "negative_controls_s0.png",
    ):
        lines.append(f"- `{fig / name}`")
    lines.append(f"- `{root / 'atomic_followup_metrics.csv'}`")
    lines.append(f"- `{root / 'atomic_followup_summary.csv'}`")
    lines.append(f"- `{root / 'atomic_followup_summary.json'}`")
    lines.append(f"- `{root / 'paired_delta_vs_C00.csv'}`")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
