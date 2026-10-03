"""Final report: content bottleneck audit + E01 decision A/B/C/D."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from content_bottleneck import ARTIFACT_ROOT, C01_ARTIFACT_ROOT, REPORT_PATH


def _load_csv(path: Path):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _cell(summary, experiment, scenario, arm, metric):
    hits = [
        r for r in summary
        if r["experiment"] == experiment and r["scenario"] == scenario
        and r["arm"] == arm and r["metric"] == metric
    ]
    if not hits:
        return "—"
    mean, std, n = float(hits[0]["mean"]), float(hits[0]["std"]), int(hits[0]["n"])
    return f"{mean:.4f}" if n <= 1 else f"{mean:.4f} ± {std:.4f}"


def _paired_mean(paired, experiment, scenario, arm, metric):
    vals = [
        float(r["delta"]) for r in paired
        if r["experiment"] == experiment and r["scenario"] == scenario
        and r["arm"] == arm and r["metric"] == metric
    ]
    if not vals:
        return None, None, 0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size)


def _fmt(mean, std, n):
    if mean is None:
        return "—"
    if n <= 1:
        return f"{mean:+.4f}"
    return f"{mean:+.4f} ± {std:.4f}"


def _audit_metric(audit_rows, scenario, metric):
    vals = [
        float(r["value"]) for r in audit_rows
        if r["scenario"] == scenario and r["metric"] == metric and r["value"] not in ("", None)
    ]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return "—"
    arr = np.asarray(vals)
    if arr.size == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def write_report(root: Path | None = None, report_path: Path | None = None) -> Path:
    root = Path(root or ARTIFACT_ROOT)
    out = Path(report_path or REPORT_PATH)
    summary = _load_csv(root / "e01_vs_c01_summary.csv")
    paired = _load_csv(root / "paired_delta_vs_C01.csv")
    audit = _load_csv(root / "content_information_audit.csv")
    decomp = _load_csv(root / "oracle_content_decomposition.csv")
    content = _load_csv(root / "target_signal_content_recovery.csv")

    # Load D00 from high_impact if available for decomposition table
    d00_summary = _load_csv(C01_ARTIFACT_ROOT.parent / "dtr_high_impact_followup" / "high_impact_followup_summary.csv")

    lines = [
        "# Content bottleneck final test",
        "",
        "Working baseline remains `C01_staged_current`. Part 1 audits frozen C01 content path;",
        "Part 2 is the single predefined correction `E01_target_conditioned_retrieval`.",
        "",
        "## Part 1 — Content information audit (C01, no retrain)",
        "",
        "### A. Representation sufficiency (S2)",
        "",
        f"- A1 signal-identity AUROC from post-encoder `v_m`: {_audit_metric(audit, 'S2', 'A1_v_mean_auroc')}",
        f"- A1 signal-identity AUROC from pre-encoder pool: {_audit_metric(audit, 'S2', 'A1_pre_mean_auroc')}",
        f"- A2 oracle `w` probe RMSE from `v_m`: {_audit_metric(audit, 'S2', 'A2_v_rmse')} (corr {_audit_metric(audit, 'S2', 'A2_v_corr')}, R² {_audit_metric(audit, 'S2', 'A2_v_r2')})",
        f"- A2 upper bound from raw signal membership: RMSE {_audit_metric(audit, 'S2', 'A2_raw_rmse')} (corr {_audit_metric(audit, 'S2', 'A2_raw_corr')})",
        f"- A3 mean content score `u` background vs signal: {_audit_metric(audit, 'S2', 'A3_u_mean_bg')} vs {_audit_metric(audit, 'S2', 'A3_u_mean_sig')}",
        "",
        "### B. Retrieval sufficiency (S2)",
        "",
        f"- corr(`u`, mean |oracle target relevance|): {_audit_metric(audit, 'S2', 'B_corr_u_mean_abs')}",
        f"- corr(`u`, max |oracle target relevance|): {_audit_metric(audit, 'S2', 'B_corr_u_max_abs')}",
        "",
        "### C. Oracle content substitution decomposition (S2)",
        "",
        "| Arm | BCE | AUROC | AUPRC | Surface RMSE |",
        "| --- | --- | --- | --- | --- |",
    ]

    def _decomp(arm, metric):
        hits = [r for r in decomp if r["scenario"] == "S2" and r["arm"] == arm and r["metric"] == metric]
        if not hits:
            return "—"
        mean, std, n = float(hits[0]["mean"]), float(hits[0]["std"]), int(hits[0]["n"])
        return f"{mean:.4f}" if n <= 1 else f"{mean:.4f} ± {std:.4f}"

    for arm in ("C1", "C2"):
        lines.append("| " + " | ".join([
            f"oracle content + {'C01 gate' if arm == 'C1' else 'oracle gate'} ({arm})",
            _decomp(arm, "bce"), _decomp(arm, "auroc"), _decomp(arm, "auprc"), _decomp(arm, "surface_rmse"),
        ]) + " |")
    lines.append("| " + " | ".join([
        "C01 (learned content + learned gate)",
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "bce"),
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "auroc"),
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "auprc"),
        _cell(summary, "C01_staged_current", "S2", "age_temporal", "surface_rmse"),
    ]) + " |")
    lines.append("| " + " | ".join([
        "D00 (learned content + oracle gate)",
        _cell(d00_summary, "D00_oracle_gate", "S2", "age_temporal", "bce") if d00_summary else "—",
        _cell(d00_summary, "D00_oracle_gate", "S2", "age_temporal", "auroc") if d00_summary else "—",
        _cell(d00_summary, "D00_oracle_gate", "S2", "age_temporal", "auprc") if d00_summary else "—",
        _cell(d00_summary, "D00_oracle_gate", "S2", "age_temporal", "surface_rmse") if d00_summary else "—",
    ]) + " |")
    lines.append("")

    c1_bce = _audit_metric(audit, "S2", "C1_bce")
    c2_bce = _audit_metric(audit, "S2", "C2_bce")
    delta_bce = _audit_metric(audit, "S2", "delta_C1_C2_bce")
    delta_surf = _audit_metric(audit, "S2", "delta_C1_C2_surface_rmse")
    lines.append(
        f"C1 vs C2 (oracle content; gate differs): ΔBCE={delta_bce}, Δsurface={delta_surf}. "
        "If C1≈C2, C01 temporal learning is already adequate and learned content is the limiter."
    )
    lines.append("")

    lines.append("## Part 2 — E01 target-conditioned retrieval")
    lines.append("")
    lines.append("| Model | S2 BCE | S2 AUPRC | S2 surface | S2 gate RMSE | S0 gate-shuffle ΔBCE |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for exp in ("C01_staged_current", "E01_target_conditioned_retrieval"):
        lines.append("| " + " | ".join([
            exp,
            _cell(summary, exp, "S2", "age_temporal", "bce"),
            _cell(summary, exp, "S2", "age_temporal", "auprc"),
            _cell(summary, exp, "S2", "age_temporal", "surface_rmse"),
            _cell(summary, exp, "S2", "age_temporal", "gate_signal_rmse"),
            _cell(summary, exp, "S0", "age_temporal", "delta_bce_gate_age_shuffle"),
        ]) + " |")
    lines.append("")
    lines.append("Paired E01 − C01 (S2 age_temporal):")
    for metric in ("bce", "auprc", "surface_rmse", "gate_signal_rmse", "cf_rmse_age", "delta_bce_gate_age_shuffle", "delta_bce_beta0"):
        mean, std, n = _paired_mean(paired, "E01_target_conditioned_retrieval", "S2", "age_temporal", metric)
        lines.append(f"- {metric}: {_fmt(mean, std, n)}")
    lines.append("")
    if content:
        s2c = [r for r in content if r["scenario"] == "S2"]
        if s2c:
            rmse = np.mean([float(r["matrix_rmse"]) for r in s2c if r["matrix_rmse"]])
            pear = np.mean([float(r["pearson"]) for r in s2c if r["pearson"]])
            lines.append(f"S2 content matrix recovery: RMSE={rmse:.4f}, Pearson={pear:.4f}.")
    lines.append("")

    # Decision logic
    a1_v = np.mean([float(r["value"]) for r in audit if r["scenario"] == "S2" and r["metric"] == "A1_v_mean_auroc" and r["value"]]) if audit else None
    a2_gap = None
    if audit:
        v_rmse = [float(r["value"]) for r in audit if r["scenario"] == "S2" and r["metric"] == "A2_v_rmse" and r["value"]]
        raw_rmse = [float(r["value"]) for r in audit if r["scenario"] == "S2" and r["metric"] == "A2_raw_rmse" and r["value"]]
        if v_rmse and raw_rmse:
            a2_gap = float(np.mean(v_rmse) - np.mean(raw_rmse))
    b_corr = np.mean([float(r["value"]) for r in audit if r["scenario"] == "S2" and r["metric"] == "B_corr_u_mean_abs" and r["value"]]) if audit else None
    d_bce = [float(r["value"]) for r in audit if r["scenario"] == "S2" and r["metric"] == "delta_C1_C2_bce" and r["value"]]
    d_surf = [float(r["value"]) for r in audit if r["scenario"] == "S2" and r["metric"] == "delta_C1_C2_surface_rmse" and r["value"]]
    c1_close_c2 = bool(d_bce) and abs(float(np.mean(d_bce))) < 0.01 and (not d_surf or abs(float(np.mean(d_surf))) < 0.05)

    e01_bce_d, e01_bce_s, e01_bce_n = _paired_mean(paired, "E01_target_conditioned_retrieval", "S2", "age_temporal", "bce")
    e01_surf_d, e01_surf_s, _ = _paired_mean(paired, "E01_target_conditioned_retrieval", "S2", "age_temporal", "surface_rmse")
    e01_s0_d, e01_s0_s, _ = _paired_mean(paired, "E01_target_conditioned_retrieval", "S0", "age_temporal", "delta_bce_gate_age_shuffle")
    e01_improves = (
        e01_bce_d is not None and e01_bce_d < 0 and abs(e01_bce_d) > (e01_bce_s or 0)
    ) or (
        e01_surf_d is not None and e01_surf_d < 0 and abs(e01_surf_d) > (e01_surf_s or 0)
    )
    s0_ok = e01_s0_d is None or abs(e01_s0_d) < 0.005

    repr_ok = a1_v is not None and a1_v >= 0.85
    repr_lossy = a2_gap is not None and a2_gap > 0.2
    retrieval_weak = b_corr is not None and b_corr < 0.4

    if e01_improves and s0_ok:
        conclusion = "B. global target-agnostic retrieval is the primary bottleneck"
        retain = "Retain E01_target_conditioned_retrieval over C01."
    elif repr_lossy and not retrieval_weak and not e01_improves:
        conclusion = "A. encounter representation is the primary content bottleneck"
        retain = "Retain C01_staged_current. Stop synthetic architecture search."
    elif retrieval_weak and repr_ok and not e01_improves:
        conclusion = "B. global target-agnostic retrieval is the primary bottleneck"
        retain = "Retain C01_staged_current (E01 did not convert the diagnosis into a gain). Stop synthetic architecture search."
    elif (repr_lossy or not repr_ok) and retrieval_weak and not e01_improves:
        conclusion = "C. both contribute"
        retain = "Retain C01_staged_current. Stop synthetic architecture search."
    else:
        conclusion = "D. neither explains enough of the remaining gap; stop synthetic architecture search and retain C01"
        retain = "Retain C01_staged_current. Document residual mismatch; proceed to full-realism Synthea and MIMIC/NCH."

    # Prefer D when E01 fails and C1≈C2 already said content limits but E01 didn't fix it,
    # and representation probes are strong (encoder not the issue) while retrieval diagnosis
    # was suggestive but the correction failed.
    if (not e01_improves) and c1_close_c2 and repr_ok and retrieval_weak:
        conclusion = "D. neither explains enough of the remaining gap; stop synthetic architecture search and retain C01"
        retain = (
            "Oracle-content diagnostics implicate content modeling, and global-u retrieval is poorly "
            "aligned with target-specific relevance, but E01 did not improve C01. Retain C01; stop "
            "synthetic architecture development."
        )
    if (not e01_improves) and repr_ok and (a2_gap is not None and a2_gap <= 0.2) and (b_corr is not None and b_corr >= 0.4):
        conclusion = "D. neither explains enough of the remaining gap; stop synthetic architecture search and retain C01"
        retain = "Representation and retrieval probes do not isolate a clear fix; E01 did not help. Retain C01."

    lines.append("## Decision")
    lines.append("")
    lines.append(f"C1≈C2 (temporal adequate under oracle content): {c1_close_c2}.")
    lines.append(f"E01 improves C01 (paired S2): {e01_improves}.")
    lines.append(f"A1 v AUROC≈{a1_v}; A2 RMSE gap(v−raw)≈{a2_gap}; B corr(u, oracle)≈{b_corr}.")
    lines.append("")
    lines.append(retain)
    lines.append("")
    lines.append(conclusion)
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{root / 'content_information_audit.csv'}`")
    lines.append(f"- `{root / 'oracle_content_decomposition.csv'}`")
    lines.append(f"- `{root / 'target_signal_content_recovery.csv'}`")
    lines.append(f"- `{root / 'paired_delta_vs_C01.csv'}`")
    lines.append(f"- `{root / 'figures' / 'true_vs_learned_content_S2.png'}`")
    lines.append(f"- `{root / 'figures' / 'true_vs_learned_content_S3.png'}`")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out
