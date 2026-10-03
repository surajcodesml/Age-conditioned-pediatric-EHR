"""Write the architecture-ladder report from saved summary tables only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ladder import REPO_ROOT

EXPERIMENTS = (
    "E01_direct",
    "E02_staged",
    "E03_mass",
    "E04_channels",
    "E05_mixture",
    "E06_integrated_hazard",
)

CARD = {
    "E01_direct": {
        "change": "Remove content query, exp(u), content-dependent persistence, and the nonlinear history MLP. Direct linear readout. Weak lambda initialization.",
        "training": "Same trainer as the current DTR baseline: one AdamW group, learning rate 3e-4, weight decay 1e-2, 25 epochs, patience 5, minimum 12 epochs, batch size 32, gradient clip 1.",
        "added": "none beyond theta0, beta, linear W_history, linear W_age, and bias",
        "removed": "content query/key, exp(u), persistence projection r·v + b_r, history MLP",
    },
    "E02_staged": {
        "change": "No architectural change relative to E01.",
        "training": "Stage A trains beta=0. The checkpoint is cloned into both arms. Stage B freezes the encoder and readout and trains theta0/beta for 5 epochs at 10x learning rate and zero weight decay. Stage C unfreezes all parameters and fine-tunes with the same temporal parameter group.",
        "added": "none",
        "removed": "none",
    },
    "E03_mass": {
        "change": "Replace only the sum g·v aggregation by normalized composition plus log1p(evidence mass).",
        "training": "Identical to E01.",
        "added": "log1p(M) feature on the linear readout",
        "removed": "raw unnormalized sum as the sole history vector",
    },
    "E04_channels": {
        "change": "Add 4 content channels c=sigmoid(q_h^T v) with one shared lambda(a). Queries do not receive age or lag.",
        "training": "Identical to E01.",
        "added": "4 content query vectors",
        "removed": "none from the E01 mechanism",
    },
    "E05_mixture": {
        "change": "K=3 content-only mixture of developmental rates. First model with content-specific timescales.",
        "training": "Identical to E01.",
        "added": "linear content mixture, theta_k, beta_k",
        "removed": "single global theta0/beta",
    },
    "E06_integrated_hazard": {
        "change": "Replace tau-multiplied lambda(a) by the integral of a 4-knot positive piecewise-linear hazard from event age to prediction age. Temporal-only is a constant hazard.",
        "training": "Identical to E01.",
        "added": "knot coefficients of rho(a)",
        "removed": "softplus lambda(a) times tau",
    },
}

HYPOTHESIS = {
    "E01_direct": "Retrieval and readout complexity in the current DTR is the main bottleneck for explicit age×lag recovery.",
    "E02_staged": "A staged optimizer, with architecture held fixed at E01, improves mechanism recovery.",
    "E03_mass": "Separating history composition from surviving evidence mass improves stability and recovery.",
    "E04_channels": "Extra content channels help while the developmental decay stays a single shared lambda(a).",
    "E05_mixture": "A small content-dependent mixture of timescales recovers the age×lag surface better than one global decay.",
    "E06_integrated_hazard": "An integrated hazard from event age to prediction age is a separate developmental parameterization.",
}


def _load_summary(root: Path) -> list[dict[str, str]]:
    with (root / "architecture_ladder_summary.csv").open() as handle:
        return list(csv.DictReader(handle))


def _get(rows, experiment, scenario, arm, metric) -> dict[str, str] | None:
    for row in rows:
        if (row["experiment"], row["scenario"], row["arm"], row["metric"]) == (
            experiment, scenario, arm, metric,
        ):
            return row
    return None


def _fmt(row: dict[str, str] | None, digits: int = 4) -> str:
    if row is None or row.get("mean", "") == "":
        return "—"
    mean = float(row["mean"])
    std = float(row["std"]) if row.get("std") not in ("", None) else 0.0
    n = int(float(row["n"]))
    if n <= 1:
        return f"{mean:.{digits}f} (n=1)"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _mean(row: dict[str, str] | None) -> float | None:
    if row is None or row.get("mean", "") == "":
        return None
    return float(row["mean"])


def _std(row: dict[str, str] | None) -> float:
    if row is None or row.get("std") in ("", None):
        return 0.0
    return float(row["std"])


def _judgment(new: dict[str, str] | None, old: dict[str, str] | None, *, lower_better: bool) -> str:
    a, b = _mean(new), _mean(old)
    if a is None or b is None:
        return "Not compared; a value is missing."
    delta = a - b
    favorable = delta < 0 if lower_better else delta > 0
    scale = max(_std(new), _std(old))
    direction = "favorable" if favorable else "unfavorable"
    if int(float(new["n"])) <= 1 and int(float(old["n"])) <= 1:
        return (
            f"Observed difference {delta:+.4f} ({direction} if "
            f"{'lower' if lower_better else 'higher'} is better). "
            "Both sides are single-seed, so variability across seeds is not estimated."
        )
    if abs(delta) > scale:
        return (
            f"Observed difference {delta:+.4f} ({direction}). "
            "The absolute difference exceeds the larger seed standard deviation."
        )
    return (
        f"Observed difference {delta:+.4f} ({direction}). "
        "The absolute difference does not exceed the larger seed standard deviation."
    )


def write_report(root: Path, report_path: Path | None = None) -> Path:
    root = Path(root)
    rows = _load_summary(root)
    failures_path = root / "failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else []
    out = report_path or (REPO_ROOT / "reports" / "synthetic_architecture_ladder.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Synthetic architecture ladder")
    lines.append("")
    lines.append("Metrics and figures were computed from saved predictions and mechanism arrays.")
    lines.append("Models were not rerun for the tables or plots.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Dataset: existing controlled Synthea scenarios S0–S3, patient split seed 20260922.")
    lines.append("- Seeds, fixed before fitting: 0, 1, 2, 3, 4. This is the existing `cehrbert_small` seed list. The current DTR reference itself was fit only at seed 0.")
    lines.append("- Training budget matches `dtr_age_temporal_new`: AdamW, lr 3e-4, weight decay 0.01, batch 32, 25 epochs, patience 5, minimum 12 epochs, gradient clip 1, d_model 64, dropout 0.")
    lines.append("- Encounter tensors are precomputed and padded to the split maximum. Pad positions are masked, and the shuffle still comes from the trainer seed. Independent seeds run two at a time on one GPU. Batch size, learning rate, epoch budget, and seeds are unchanged.")
    lines.append("- E02 is the only run that changes the optimizer.")
    lines.append("- E06 is an extension. The S2 oracle was generated with current-age λ(a), not an integrated hazard, so E06 is not expected to beat E01 on oracle recovery by construction.")
    lines.append("- Surface RMSE, CF-RMSE_age, and CF-RMSE_lag use `baselines.common.counterfactual` on the saved grids.")
    lines.append("- β=0 ΔBCE and age-shuffle ΔBCE are test-set BCE(counterfactual) − BCE(original). The age shuffle uses NumPy seed 0, the same seed as the existing DTR ablation.")
    lines.append("- For mixture models, λ(a) compared with the oracle is the unweighted mean of λ_k(a). That reduction was fixed in the metric code before the runs.")
    lines.append("- Seed-level intervals in the summary CSV are normal approximations mean ± 1.96·sd/√n. Each run's `metrics.json` also stores a 200-draw patient bootstrap of the BCE deltas.")
    lines.append("")
    lines.append(f"Artifact root: `{root}`")
    lines.append("")
    if failures:
        lines.append("## Recorded failures")
        lines.append("")
        for item in failures:
            lines.append(f"- {item.get('experiment')} {item.get('scenario')} {item.get('arm')} seed {item.get('seed')}: {item.get('error')}")
        lines.append("")

    def block(experiment: str, scenario: str, arm: str) -> list[str]:
        bits = []
        for metric, label in (
            ("bce", "BCE"),
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("surface_rmse", "Surface RMSE"),
            ("cf_rmse_age", "CF-RMSE_age"),
            ("cf_rmse_lag", "CF-RMSE_lag"),
            ("delta_bce_beta0", "β=0 ΔBCE"),
            ("delta_bce_age_shuffle", "age-shuffle ΔBCE"),
            ("beta", "β"),
            ("abs_beta_mean", "|β|"),
            ("lambda_rmse", "λ RMSE"),
            ("lambda_corr", "λ correlation"),
        ):
            bits.append(f"{label} {_fmt(_get(rows, experiment, scenario, arm, metric))}")
        return bits

    for experiment in EXPERIMENTS:
        card = CARD[experiment]
        lines.append(f"## {experiment}")
        lines.append("")
        lines.append(f"Hypothesis: {HYPOTHESIS[experiment]}")
        lines.append("")
        lines.append(f"Single change: {card['change']}")
        lines.append("")
        lines.append(f"Parameters added: {card['added']}. Parameters removed: {card['removed']}.")
        lines.append("")
        lines.append(f"Training: {card['training']}")
        lines.append("")
        lines.append(f"Artifacts: `{root / experiment}`")
        lines.append("")
        for scenario in ("S0", "S1", "S2", "S3"):
            for arm in ("age_temporal", "temporal_only"):
                lines.append(f"- {scenario} {arm}: " + "; ".join(block(experiment, scenario, arm)))
        lines.append("")
        lines.append("### Comparison with E01 and the current DTR")
        lines.append("")
        ref_arch = "E01_direct" if experiment != "E01_direct" else "E00_dtr_age_temporal_new"
        ref_label = "E01" if experiment != "E01_direct" else "E00 current DTR"
        for metric, lower in (
            ("surface_rmse", True),
            ("cf_rmse_age", True),
            ("cf_rmse_lag", True),
            ("bce", True),
            ("auroc", False),
            ("auprc", False),
            ("delta_bce_beta0", False),
            ("delta_bce_age_shuffle", False),
        ):
            new = _get(rows, experiment, "S2", "age_temporal", metric)
            old = _get(rows, ref_arch, "S2", "age_temporal", metric)
            lines.append(f"- S2 age-temporal {metric} versus {ref_label}: {_judgment(new, old, lower_better=lower)}")
        s0_new = _get(rows, experiment, "S0", "age_temporal", "cf_rmse_age")
        s0_old = _get(rows, ref_arch, "S0", "age_temporal", "cf_rmse_age")
        lines.append(f"- S0 CF-RMSE_age versus {ref_label}: {_judgment(s0_new, s0_old, lower_better=True)}")
        beta_s0 = _get(rows, experiment, "S0", "age_temporal", "abs_beta_mean")
        lines.append(f"- S0 |β| is {_fmt(beta_s0)}.")
        lines.append("")
        lines.append("Supported interpretation is limited to the comparisons above.")
        lines.append("Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.")
        if experiment == "E06_integrated_hazard":
            lines.append("")
            lines.append("E06-specific constraint: a worse S2 surface than E01 is not evidence against the integrated hazard as a model of a different mechanism. The oracle labels were not generated from that integral.")
        lines.append("")

    lines.append("## Compact comparison")
    lines.append("")
    lines.append("AUROC and AUPRC are S2 age-temporal micro averages. S0 false interaction is S0 age-temporal CF-RMSE_age. Cells are seed mean ± sample standard deviation.")
    lines.append("")
    lines.append("| Experiment | Single change tested | S2 Surface RMSE | S2 β=0 ΔBCE | S2 shuffle ΔBCE | S0 false interaction | AUROC | AUPRC |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    table_order = (
        ("E00_dtr_age_temporal_new", "age_temporal", "current DTR age_temporal (seed 0 reference)"),
        ("E00_dtr_temporal_only_new", "temporal_only", "current DTR temporal_only (seed 0 reference)"),
        ("E00_legacy_dtr_age_temporal", "age_temporal", "legacy unsuffixed dtr_age_temporal"),
        ("E00_legacy_dtr_temporal_only", "temporal_only", "legacy unsuffixed dtr_temporal_only"),
        ("CEHR-BERT", "cehrbert", "published CEHR-BERT, all 32 targets"),
        ("CEHR-BERT-small", "cehrbert_small", "published CEHR-BERT-small, S2 seeds 0–4"),
        ("E01_direct", "age_temporal", CARD["E01_direct"]["change"]),
        ("E02_staged", "age_temporal", CARD["E02_staged"]["change"]),
        ("E03_mass", "age_temporal", CARD["E03_mass"]["change"]),
        ("E04_channels", "age_temporal", CARD["E04_channels"]["change"]),
        ("E05_mixture", "age_temporal", CARD["E05_mixture"]["change"]),
        ("E06_integrated_hazard", "age_temporal", CARD["E06_integrated_hazard"]["change"]),
    )
    for experiment, arm, change in table_order:
        lines.append("| " + " | ".join([
            experiment,
            change.replace("|", "/"),
            _fmt(_get(rows, experiment, "S2", arm, "surface_rmse")),
            _fmt(_get(rows, experiment, "S2", arm, "delta_bce_beta0")),
            _fmt(_get(rows, experiment, "S2", arm, "delta_bce_age_shuffle")),
            _fmt(_get(rows, experiment, "S0", arm, "cf_rmse_age")),
            _fmt(_get(rows, experiment, "S2", arm, "auroc")),
            _fmt(_get(rows, experiment, "S2", arm, "auprc")),
        ]) + " |")
    lines.append("")
    lines.append("## Figure paths")
    lines.append("")
    fig = root / "figures"
    for name in (
        "heatmap_S2.png",
        "heatmap_S3.png",
        "lambda_age.png",
        "e06_rho_age.png",
        "mechanism_metrics_s2.png",
        "predictive_metrics_s2.png",
        "negative_controls_s0_s1.png",
    ):
        lines.append(f"- `{fig / name}`")
    lines.append(f"- `{root / 'architecture_ladder_metrics.csv'}`")
    lines.append(f"- `{root / 'architecture_ladder_summary.csv'}`")
    lines.append(f"- `{root / 'architecture_ladder_summary.json'}`")
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    return out
