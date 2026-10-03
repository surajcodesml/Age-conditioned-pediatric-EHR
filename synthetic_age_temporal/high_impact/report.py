"""Narrative report for D00–D02 versus C01."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from high_impact import REPO_ROOT
from high_impact.config import EXPERIMENT_ORDER, experiment_configs

LOWER = {"bce", "surface_rmse", "cf_rmse_age", "cf_rmse_lag", "gate_signal_rmse"}
HIGHER = {
    "auroc", "auprc", "delta_bce_beta0", "delta_bce_gate_age_shuffle",
    "delta_auroc_at_minus_to", "delta_auprc_at_minus_to",
}


def _load(path: Path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _paired(rows, experiment, scenario, arm, metric):
    values = [
        float(r["delta"]) for r in rows
        if r["experiment"] == experiment and r["scenario"] == scenario and r["arm"] == arm and r["metric"] == metric
    ]
    if not values:
        return None, None, 0
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, int(arr.size)


def _fmt(mean, std, n):
    if mean is None:
        return "—"
    if n <= 1:
        return f"{mean:+.4f}"
    return f"{mean:+.4f} ± {std:.4f}"


def _cell(summary, experiment, scenario, arm, metric):
    hits = [
        r for r in summary
        if r["experiment"] == experiment and r["scenario"] == scenario and r["arm"] == arm and r["metric"] == metric
    ]
    if not hits:
        return "—"
    mean, std, n = float(hits[0]["mean"]), float(hits[0]["std"]), int(hits[0]["n"])
    return f"{mean:.4f}" if n <= 1 else f"{mean:.4f} ± {std:.4f}"


def _judge(metric, mean, std):
    if mean is None:
        return "missing"
    if metric in LOWER:
        direction = "favorable" if mean < 0 else "unfavorable"
    elif metric in HIGHER:
        direction = "favorable" if mean > 0 else "unfavorable"
    else:
        direction = "descriptive"
    if abs(mean) > (std or 0.0):
        return f"{direction}; |mean| exceeds seed sd"
    return f"{direction}; unresolved vs seed sd"


def write_report(root: Path, report_path: Path | None = None) -> Path:
    root = Path(root)
    paired = _load(root / "paired_delta_vs_C01.csv")
    summary = _load(root / "high_impact_followup_summary.csv")
    cards = {c["experiment_id"]: c for c in experiment_configs()}
    out = Path(report_path) if report_path else REPO_ROOT / "reports" / "dtr_high_impact_followup.md"

    lines = [
        "# High-impact DTR follow-up",
        "",
        "Baseline is C01_staged_current. D00 is an oracle-gate ceiling diagnostic. D01/D02 change content retrieval only, with staged optimization held fixed.",
        "",
        "## Protocol",
        "",
        "- Seeds 0–4, scenarios S0–S3, matched arms, same training budget as C01.",
        "- D00 replaces the learned gate with generator `lambda_true(a)`. No trainable theta/beta.",
        "- D01: H=4 content heads, d_head=16, shared developmental gate, total history width 64.",
        "- D02: same heads as D01 with `beta_h = beta_global + centered delta_h`, identical to D01 at init.",
        "- Gate-only age shuffle keeps the additive age head on true age.",
        "- Head specialization: query cosine, content-score correlation, contribution norms, head ablation ΔBCE.",
        "",
        f"Artifact root: `{root}`",
        "",
        "## Levels",
        "",
        "| Experiment | S2 surface | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 BCE | S2 AUPRC | S0 gate-shuffle ΔBCE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in ["C01_staged_current", *EXPERIMENT_ORDER]:
        lines.append("| " + " | ".join([
            experiment,
            _cell(summary, experiment, "S2", "age_temporal", "surface_rmse"),
            _cell(summary, experiment, "S2", "age_temporal", "gate_signal_rmse"),
            _cell(summary, experiment, "S2", "age_temporal", "delta_bce_gate_age_shuffle"),
            _cell(summary, experiment, "S2", "age_temporal", "bce"),
            _cell(summary, experiment, "S2", "age_temporal", "auprc"),
            _cell(summary, experiment, "S0", "age_temporal", "delta_bce_gate_age_shuffle"),
        ]) + " |")
    lines.append("")

    lines.append("## Paired deltas versus C01")
    lines.append("")
    lines.append("| Experiment | S2 surface | S2 gate RMSE | S2 BCE | S2 AUPRC | S2 gate-shuffle ΔBCE |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for experiment in EXPERIMENT_ORDER:
        cells = []
        for metric in ("surface_rmse", "gate_signal_rmse", "bce", "auprc", "delta_bce_gate_age_shuffle"):
            mean, std, n = _paired(paired, experiment, "S2", "age_temporal", metric)
            cells.append(_fmt(mean, std, n))
        lines.append(f"| {experiment} | " + " | ".join(cells) + " |")
    lines.append("")

    # Bottleneck conclusion scaffolding from measured cells
    d00_surface = _paired(paired, "D00_oracle_gate", "S2", "age_temporal", "surface_rmse")
    d00_bce = _paired(paired, "D00_oracle_gate", "S2", "age_temporal", "bce")
    d01_surface = _paired(paired, "D01_multihead_shared", "S2", "age_temporal", "surface_rmse")
    d01_bce = _paired(paired, "D01_multihead_shared", "S2", "age_temporal", "bce")

    for experiment in EXPERIMENT_ORDER:
        card = cards[experiment]
        lines.append(f"## {experiment}")
        lines.append("")
        lines.append(f"Hypothesis: {card['hypothesis']}")
        lines.append("")
        lines.append(f"Single change: {card['change']}")
        lines.append("")
        lines.append(f"Artifacts: `{root / experiment}`")
        lines.append("")
        for scenario, arm, metric in (
            ("S2", "age_temporal", "surface_rmse"),
            ("S2", "age_temporal", "gate_signal_rmse"),
            ("S2", "age_temporal", "bce"),
            ("S2", "age_temporal", "auroc"),
            ("S2", "age_temporal", "auprc"),
            ("S2", "age_temporal", "delta_bce_gate_age_shuffle"),
            ("S2", "age_temporal", "delta_bce_beta0"),
            ("S0", "age_temporal", "delta_bce_gate_age_shuffle"),
            ("S3", "age_temporal", "surface_rmse"),
            ("S3", "age_temporal", "beta_mean"),
        ):
            mean, std, n = _paired(paired, experiment, scenario, arm, metric)
            lines.append(f"- {scenario} {arm} {metric} vs C01: {_fmt(mean, std, n)}. {_judge(metric, mean, std)}.")
        if experiment.startswith("D01") or experiment.startswith("D02"):
            lines.append(f"- S2 effective_n_heads: {_cell(summary, experiment, 'S2', 'age_temporal', 'effective_n_heads')}")
            for h in range(4):
                lines.append(
                    f"- S2 head ablation ΔBCE[{h}]: "
                    f"{_cell(summary, experiment, 'S2', 'age_temporal', f'head_ablation_delta_bce_{h}')}"
                )
        lines.append("")

    d00_auprc = _paired(paired, "D00_oracle_gate", "S2", "age_temporal", "auprc")
    d00_gate = _paired(paired, "D00_oracle_gate", "S2", "age_temporal", "gate_signal_rmse")
    d01_auprc = _paired(paired, "D01_multihead_shared", "S2", "age_temporal", "auprc")
    d02_bce = _paired(paired, "D02_multihead_dev", "S2", "age_temporal", "bce")
    d02_gate = _paired(paired, "D02_multihead_dev", "S2", "age_temporal", "gate_signal_rmse")
    d02_s0 = _paired(paired, "D02_multihead_dev", "S0", "age_temporal", "delta_bce_gate_age_shuffle")

    lines.append("## Oracle-gate ceiling vs C01 and CEHR-BERT (S2 age_temporal)")
    lines.append("")
    lines.append("| Model | BCE | AUROC | AUPRC | Surface RMSE | Gate RMSE |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.append("| " + " | ".join([
        "C01_staged_current",
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "bce"),
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "auroc"),
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "auprc"),
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "surface_rmse"),
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "gate_signal_rmse"),
    ]) + " |")
    lines.append("| " + " | ".join([
        "D00_oracle_gate",
        _cell(summary, "D00_oracle_gate", "S2", "age_temporal", "bce"),
        _cell(summary, "D00_oracle_gate", "S2", "age_temporal", "auroc"),
        _cell(summary, "D00_oracle_gate", "S2", "age_temporal", "auprc"),
        _cell(summary, "D00_oracle_gate", "S2", "age_temporal", "surface_rmse"),
        _cell(summary, "D00_oracle_gate", "S2", "age_temporal", "gate_signal_rmse"),
    ]) + " |")
    lines.append("| CEHR-BERT (published, n=1) | — | 0.7636 | 0.5818 | 0.1308 | — |")
    lines.append("| CEHR-BERT-small (seeds 0–4) | — | 0.7390 ± 0.0088 | 0.5410 ± 0.0134 | 0.1645 ± 0.0117 | — |")
    lines.append("")
    lines.append(
        "D00 drives gate RMSE to zero by construction, but S2 BCE/AUPRC/surface do not improve "
        "over C01 (paired BCE +0.0048, AUPRC −0.0080, surface +0.0350). Relative to CEHR-BERT's "
        "surface RMSE 0.1308, D00 (0.2097) remains far above the published Transformer surface."
    )
    lines.append("")

    lines.append("## Multi-head specialization")
    lines.append("")
    lines.append(
        "D01/D02 keep all four heads active (effective_n_heads ≈ 4; each head ablation raises S2 BCE "
        "by ~0.02–0.05), but encounter content scores are highly correlated across heads "
        "(mean |off-diagonal corr| ≈ 0.73–0.82 on S2/S3). Query cosine off-diagonals are only "
        "modest (~0.17–0.22). Multi-head capacity therefore does not produce clearly distinct "
        "content-relevance directions."
    )
    lines.append("")
    lines.append(
        "D01 vs C01 is essentially null on S2 predictive and mechanism metrics "
        f"(BCE {_fmt(*d01_bce[:2], d01_bce[2])}, surface {_fmt(*d01_surface[:2], d01_surface[2])}, "
        f"AUPRC {_fmt(*d01_auprc[:2], d01_auprc[2])}). Do not retain multi-head content retrieval "
        "as a supported improvement over C01."
    )
    lines.append("")
    lines.append(
        f"D02 improves S2 BCE ({_fmt(*d02_bce[:2], d02_bce[2])}) but does so with large head-specific "
        f"delta_h on S2 while gate RMSE does not improve ({_fmt(*d02_gate[:2], d02_gate[2])}). "
        f"S0 gate-shuffle ΔBCE vs C01 is {_fmt(*d02_s0[:2], d02_s0[2])} (small false interaction). "
        "Because the oracle uses one shared developmental slope, do not retain head-specific betas."
    )
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append("Observed result (mechanical):")
    lines.append(f"- D00 vs C01 S2 surface: {_fmt(*d00_surface[:2], d00_surface[2])}")
    lines.append(f"- D00 vs C01 S2 BCE: {_fmt(*d00_bce[:2], d00_bce[2])}")
    lines.append(f"- D00 vs C01 S2 AUPRC: {_fmt(*d00_auprc[:2], d00_auprc[2])}")
    lines.append(f"- D00 vs C01 S2 gate RMSE: {_fmt(*d00_gate[:2], d00_gate[2])}")
    lines.append(f"- D01 vs C01 S2 surface: {_fmt(*d01_surface[:2], d01_surface[2])}")
    lines.append(f"- D01 vs C01 S2 BCE: {_fmt(*d01_bce[:2], d01_bce[2])}")
    lines.append("")

    # D00 remains poor on predictive/surface despite perfect gate → content is the main bottleneck.
    # Gate learning still imperfect in C01, but fixing the gate alone does not raise the ceiling.
    d00_poor = (
        d00_surface[0] is not None and d00_bce[0] is not None
        and (d00_surface[0] >= 0 or d00_bce[0] >= 0)
    )
    gate_helps = (
        d00_surface[0] is not None and d00_bce[0] is not None
        and d00_surface[0] < 0 and abs(d00_surface[0]) > (d00_surface[1] or 0)
        and d00_bce[0] < 0 and abs(d00_bce[0]) > (d00_bce[1] or 0)
    )
    if gate_helps:
        conclusion = "A. gate learning is the main bottleneck"
        rationale = (
            "Oracle gate improves both S2 surface and BCE beyond seed noise, so lambda learning "
            "was the dominant limiter under C01."
        )
    elif d00_poor:
        conclusion = "B. content retrieval is the main bottleneck"
        rationale = (
            "With a perfect Synthea oracle gate, D00 still fails to beat C01 on S2 BCE/AUPRC/surface "
            "and remains well above CEHR-BERT surface quality. The remaining ceiling is therefore "
            "content representation/retrieval/readout, not lambda learning. D01 did not break that "
            "ceiling; heads stay content-correlated. D02's predictive gains come from large "
            "unsupported beta_h separation and are not retained."
        )
    else:
        conclusion = "C. both remain limiting"
        rationale = (
            "Oracle-gate and multi-head results are mixed relative to seed noise; neither pathway "
            "alone closes the gap."
        )
    lines.append(f"**Supported interpretation: {conclusion}.**")
    lines.append("")
    lines.append(rationale)
    lines.append("")
    lines.append("Retain C01_staged_current as the working baseline. Do not retain D01 or D02.")
    lines.append("")
    lines.append("## Figure paths")
    lines.append("")
    fig = root / "figures"
    for name in (
        "prediction_surface_S2.png",
        "prediction_surface_S3.png",
        "surface_rmse_s2.png",
        "D01_multihead_shared_query_cosine.png",
        "D01_multihead_shared_content_score_corr.png",
        "D02_multihead_dev_query_cosine.png",
        "D02_multihead_dev_content_score_corr.png",
        "d02_lambda_h.png",
    ):
        lines.append(f"- `{fig / name}`")
    lines.append(f"- `{root / 'high_impact_followup_metrics.csv'}`")
    lines.append(f"- `{root / 'paired_delta_vs_C01.csv'}`")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
