"""Assemble SIGNAL_REPORT.md from whatever JSONs exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_new import diagnostics as D
from model_new.audit.common import REPO_ROOT


TESTS = (
    ("d9_baselines.json", "D9"),
    ("d7_halflife.json", "D7"),
    ("d1_timestamps.json", "D1"),
    ("d4_lossmass.json", "D4"),
    ("d8_horizon_hist.json", "D8.1"),
    ("d8_horizon_recall.json", "D8.2"),
    ("d2_d5_logs.json", "D2/D5"),
    ("d10_head_align.json", "D10"),
)


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, digits=4):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "nan"
        return f"{float(x):.{digits}g}"
    except (TypeError, ValueError):
        return str(x)


def _ci(blob) -> str:
    if not blob:
        return "—"
    if "ci" in blob:
        return f"{_fmt(blob.get('point'))} [{_fmt(blob['ci'].get('lo'))}, {_fmt(blob['ci'].get('hi'))}]"
    return _fmt(blob)


def build_report(out_dir: Path) -> str:
    data = {name: _load(out_dir / fname) for fname, name in TESTS}
    present = [n for n, v in data.items() if v is not None]
    missing = [n for n, v in data.items() if v is None]

    lines: list[str] = []
    lines.append("# SIGNAL REPORT")
    lines.append("")
    lines.append("Do not soften the verdict. Missing tests are marked explicitly.")
    lines.append("")
    lines.append(f"Present: {', '.join(present) if present else '(none)'}")
    lines.append(f"Missing: {', '.join(missing) if missing else '(none)'}")
    lines.append("")
    lines.append(
        "**Standing note.** Persistence ≈0.083 and global prior ≈0.105 against arms at "
        "~0.137 means neither crossed the escalation threshold; co-occurrence is the last "
        "baseline that could."
    )
    lines.append("")
    lines.append(
        "**D3 cut.** Superseded by T5: 25-vs-75 pool TV = 0.0991, ‖Δh‖/‖h‖ = 0.1129 "
        "(in-support). T4 max_D R = 0.19 is driven by the pediatric extrapolation region "
        "(pool TV 0.35–0.38 at ages 1–5 vs 0.030–0.042 at 50–89) against a corpus that is "
        "99.95% ≥18 with median 68. σ_content = 3.2929 reused from T4."
    )
    lines.append("")

    # Routing table first.
    lines.append("## Routing table")
    lines.append("")
    lines.append("| Test | Number | Threshold | Route |")
    lines.append("|---|---|---|---|")

    d9 = data.get("D9")
    if d9:
        r10p = d9["baselines"]["persistence"]["recall@10"]
        r10c = d9["baselines"]["cooccurrence"]["recall@10"]
        lines.append(
            f"| D9 persistence recall@10 | {_fmt(r10p)} | ≈ arms (0.134–0.138) | "
            f"{d9['verdict']['route']} |"
        )
        lines.append(
            f"| D9 co-occurrence recall@10 | {_fmt(r10c)} | ≈ arms | "
            f"{'backbone near-baseline → stop, escalate' if max(r10p, r10c) >= 0.12 else 'below arms'} |"
        )
    else:
        lines.append("| D9 persistence recall@10 | — | ≈ arms | MISSING |")
        lines.append("| D9 co-occurrence recall@10 | — | ≈ arms | MISSING |")

    d7 = data.get("D7")
    if d7:
        lines.append(
            f"| D7 half-life vs age | CI overlap={d7.get('ci_overlap_all_bands')} | "
            f"CIs overlap / separate | {d7['verdict']['route']} |"
        )
    else:
        lines.append("| D7 half-life vs age | — | CIs overlap | MISSING |")

    d1 = data.get("D1")
    if d1:
        v = d1["headlines"]["vanilla_constant_delta_recall@10"]["point"]
        lines.append(
            f"| D1 constant Δrecall@10 (vanilla) | {_fmt(v)} | <1% / >5% | "
            f"{d1['verdict']['timing_route']} |"
        )
        onset = d1["headlines"]["jitter_onset_days"].get("vanilla")
        lines.append(
            f"| D1 jitter onset | {onset if onset is not None else 'none'} | ≥ ±365d | "
            f"{d1['verdict']['jitter_onset_route']} |"
        )
        if d1["verdict"].get("kernel_temporal_pathway_inert"):
            lines.append(
                "| D1 kernel vs vanilla constant | inert | kernel ≥ vanilla degrade | "
                "kernel temporal pathway inert |"
            )
    else:
        lines.append("| D1 constant Δrecall@10 (vanilla) | — | <1% / >5% | MISSING |")
        lines.append("| D1 jitter onset | — | ≥ ±365d | MISSING |")

    d82 = data.get("D8.2")
    if d82:
        lines.append(
            f"| D8 recall vs gap | flat={d82['verdict'].get('flat')} | flat / steep | "
            f"{d82['verdict']['route']} |"
        )
    else:
        lines.append("| D8 recall vs gap | — | flat / steep | MISSING |")

    d4 = data.get("D4")
    if d4:
        lines.append(
            f"| D4 positive loss mass | {_fmt(d4['positive_loss_mass_fraction'], 4)} | <5% | "
            f"{d4['verdict']['route']} |"
        )
    else:
        lines.append("| D4 positive loss mass | — | <5% | MISSING |")

    d25 = data.get("D2/D5")
    if d25 and d25.get("d2"):
        r = d25["d2"]["mean_consecutive_r_across_sites"]
        lines.append(
            f"| D2 Δα epoch correlation | {_fmt(r)} | <0.3 | "
            f"{d25['d2']['verdict']['route']} |"
        )
    else:
        lines.append("| D2 Δα epoch correlation | — | <0.3 | MISSING |")

    d10 = data.get("D10")
    if d10:
        lines.append(
            f"| D10 cos(Δh, ∇_h L) | {_fmt(d10['cos_dh_grad']['mean'])} "
            f"(rand {_fmt(d10['cos_random_control']['mean'])}) | ≈ random control | "
            f"{d10['verdict']['route']} |"
        )
    else:
        lines.append(
            "| D10 cos(Δh, ∇_h L) | — | ≈ random control | MISSING |"
        )

    lines.append("")

    # Detail tables.
    if d9:
        lines.append("## D9 baselines")
        lines.append("")
        lines.append("| Baseline | recall@5 | recall@10 | recall@20 |")
        lines.append("|---|---|---|---|")
        for name, row in d9["baselines"].items():
            lines.append(
                f"| {name} | {_fmt(row['recall@5'])} | {_fmt(row['recall@10'])} | "
                f"{_fmt(row['recall@20'])} |"
            )
        lines.append("")
        lines.append(f"Hand-check match: {d9['handcheck']['match']}")
        lines.append("")

    if d7:
        lines.append("## D7 half-life by age band")
        lines.append("")
        lines.append("| Band | median h (days) | CI lo | CI hi | n |")
        lines.append("|---|---|---|---|---|")
        for b, s in d7["by_band"].items():
            lines.append(
                f"| {b} | {_fmt(s['median'])} | {_fmt(s['ci_lo'])} | "
                f"{_fmt(s['ci_hi'])} | {s['n']} |"
            )
        lines.append("")
        lines.append(
            f"Monotonicity slope={_fmt(d7['monotonicity']['obs_slope_median_h_vs_band_index'])} "
            f"perm_p={_fmt(d7['monotonicity']['permutation_p'])}"
        )
        lines.append("")

    if d1:
        lines.append("## D1 timestamp conditions")
        lines.append("")
        lines.append(
            f"Vanilla constant Δr@10: {_ci(d1['headlines']['vanilla_constant_delta_recall@10'])}"
        )
        lines.append(
            f"Kernel constant Δr@10: {_ci(d1['headlines']['kernel_constant_delta_recall@10'])}"
        )
        lines.append(
            f"Assertions: {json.dumps(d1['assertions'])}"
        )
        lines.append("")
        lines.append("| Arm | jitter days | Δr@10 | CI |")
        lines.append("|---|---|---|---|")
        jitter = d1["headlines"]["jitter_curve_delta_recall@10"]
        # v2: jitter only on vanilla; tolerate legacy two-arm layout.
        if "vanilla" in jitter and isinstance(jitter["vanilla"], dict):
            items = [("vanilla", jitter["vanilla"])]
        else:
            items = [("vanilla", jitter)]
        for arm, curve in items:
            for k, blob in curve.items():
                lines.append(
                    f"| {arm} | {k} | {_fmt(blob['point'])} | "
                    f"[{_fmt(blob['ci']['lo'])}, {_fmt(blob['ci']['hi'])}] |"
                )
        lines.append("")

    d10 = data.get("D10")
    if d10:
        lines.append("## D10 head sensitivity alignment")
        lines.append("")
        lines.append(
            f"cos(Δh, ∇_h L) mean={_fmt(d10['cos_dh_grad']['mean'])} "
            f"p50={_fmt(d10['cos_dh_grad']['p50'])}; "
            f"random control mean={_fmt(d10['cos_random_control']['mean'])}"
        )
        lines.append(
            f"frac ‖Δh‖ in top-10 ∇ PCs mean={_fmt(d10['frac_dh_in_top10_grad_pcs']['mean'])}"
        )
        lines.append(f"Verdict: {d10['verdict']['route']}")
        lines.append("")

    if d4:
        lines.append("## D4 loss mass")
        lines.append("")
        lines.append(
            f"Positive mass = {_fmt(d4['positive_loss_mass_fraction'], 4)}; "
            f"mean positives/example = {_fmt(d4['mean_positives_per_example'])}"
        )
        if d4.get("grad_split"):
            lines.append(f"Grad split: {json.dumps(d4['grad_split'])}")
        lines.append("")

    d81 = data.get("D8.1")
    if d81:
        lines.append("## D8.1 gap histogram")
        lines.append("")
        lines.append(
            f"Signed median={_fmt(d81['signed']['median'])} "
            f"IQR={d81['signed']['iqr']} "
            f"p10/p90={_fmt(d81['signed']['p10'])}/{_fmt(d81['signed']['p90'])}; "
            f"frac_negative={_fmt(d81['frac_negative'])}"
        )
        lines.append("")

    if d82:
        lines.append("## D8.2 recall vs gap")
        lines.append("")
        lines.append("| Gap bin | n | recall@10 |")
        lines.append("|---|---|---|")
        for e in d82["by_gap_bin"]:
            lines.append(f"| {e['bin']} | {e['n']} | {_fmt(e['recall@10'])} |")
        lines.append("")

    if d25:
        lines.append("## D2 / D5 from train.json")
        lines.append("")
        if d25.get("d2"):
            for site, s in d25["d2"]["sites"].items():
                lines.append(
                    f"- {site}: mean consecutive r={_fmt(s['mean_consecutive_r'])} "
                    f"unidentified={s['unidentified']}"
                )
            lines.append("")
        if d25.get("d5"):
            lines.append("### Selection tables")
            lines.append("")
            lines.append("| Arm | BCE-best ep | r@10 | r10-best ep | r@10 |")
            lines.append("|---|---|---|---|---|")
            for arm, row in d25["d5"]["per_arm"].items():
                a, b = row["selection_val_bce"], row["selection_recall@10"]
                lines.append(
                    f"| {arm} | {a['epoch']} | {_fmt(a['recall@10'])} | "
                    f"{b['epoch']} | {_fmt(b['recall@10'])} |"
                )
            pe = d25["d5"].get("primary_endpoint") or {}
            lines.append("")
            lines.append(
                f"Declared primary_endpoint: metric={pe.get('metric')}, "
                f"dataset={pe.get('dataset')}, comparison={pe.get('comparison')}"
            )
            lines.append("")

    # Figures.
    lines.append("## Figures")
    lines.append("")
    for fig in ("d1_jitter_curve.png", "d7_halflife_vs_age.png", "d8_recall_vs_gap.png"):
        p = out_dir / "figures" / fig
        lines.append(f"- `{fig}`: {'present' if p.is_file() else 'MISSING'}")
    lines.append("")

    # Closing: single highest-value next action. Do not soften.
    action = _next_action(data)
    lines.append("## Next action")
    lines.append("")
    lines.append(action)
    lines.append("")
    return "\n".join(lines)


def _next_action(data: dict) -> str:
    """Name the single highest-value next action from whatever evidence exists."""
    d9 = data.get("D9")
    if d9 and d9["verdict"].get("persistence_or_cooc_near_arm_recall@10"):
        return (
            "STOP. Persistence/co-occurrence already matches arm recall@10 — "
            "every arm difference is noise on a baseline none of the arms beat. "
            "Escalate: change the task (TTE / next-visit redesign), do not rebuild the age pathway."
        )
    d10 = data.get("D10")
    if d10 and d10["verdict"].get("near_random_control"):
        return (
            "Kernel Δh is orthogonal to ∇_h L (near random control). "
            "This is a head/objective failure, not an attention failure — "
            "change the head or the objective before touching the age pathway."
        )
    d1 = data.get("D1")
    if d1 and d1["verdict"].get("kernel_temporal_pathway_inert"):
        return (
            "Kernel temporal pathway is inert under constant-τ (degrades less than vanilla). "
            "Do not invest in age×lag capacity; drop or freeze the kernel route."
        )
    if d1:
        v = abs(d1["headlines"]["vanilla_constant_delta_recall@10"]["point"])
        if v < 0.01:
            return (
                "Objective is timing-blind (|Δrecall@10| under constant-τ < 1%). "
                "Build a horizon-conditioned / TTE objective before touching the age mechanism."
            )
        if v > 0.05:
            return (
                "Timing already matters under the current objective. "
                "Skip TTE; invest in multi-head age delivery and localization only."
            )
    d7 = data.get("D7")
    if d7 and d7.get("ci_overlap_all_bands"):
        return (
            "No age×lag signal in MIMIC self-recurrence (half-life CIs overlap). "
            "Reframe the scientific claim; do not rebuild the conditioner for a signal that is absent."
        )
    if d7 and not d7.get("ci_overlap_all_bands"):
        return (
            "Age×lag signal exists in the data but the model fails to use it. "
            "Next: multi-head + TTE; the mechanism audit already cleared the implementation."
        )
    d82 = data.get("D8.2")
    if d82 and d82["verdict"].get("flat"):
        return (
            "Recall is flat across target gaps — objective is horizon-marginalized. "
            "TTE redesign is justified."
        )
    d4 = data.get("D4")
    if d4 and d4["verdict"].get("pos_mass_lt_5pct"):
        return (
            "Positive BCE mass < 5%. Switch the head to sampled softmax before any further "
            "age-route work — negatives dominate the gradient."
        )
    d25 = data.get("D2/D5")
    if d25 and d25.get("d2") and d25["d2"]["mean_consecutive_r_across_sites"] is not None:
        r = d25["d2"]["mean_consecutive_r_across_sites"]
        if isinstance(r, (int, float)) and r < 0.3:
            return (
                "Δα(a) is unidentified across epochs 3→8 (r < 0.3). "
                "The age function is not merely unrewarded — it is not converging. "
                "Fix identification (stronger age supervision / constrained generator) before capacity."
            )
    if not any(data.values()):
        return "No JSON present. Run the signal battery."
    return (
        "Partial evidence only — finish missing tests, then re-read the routing table. "
        "Do not soft-pedal a null with architecture tweaks."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "model_new" / "audit" / "signal" / "out")
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    md = build_report(out_dir)
    path = out_dir / "SIGNAL_REPORT.md"
    path.write_text(md, encoding="utf-8")
    D.print_block("SIGNAL_REPORT", [
        f"wrote {path}",
        f"bytes={path.stat().st_size}",
    ])
    # Echo only the paragraph under "## Next action".
    next_lines: list[str] = []
    grab = False
    for line in md.splitlines():
        if line.strip() == "## Next action":
            grab = True
            continue
        if grab:
            if line.startswith("## "):
                break
            if line.strip():
                next_lines.append(line.strip())
    if next_lines:
        D.print_block("Next action", next_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
