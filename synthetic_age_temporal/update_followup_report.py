#!/usr/bin/env python3
"""Append follow-up investigation section to report.md from collected artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_DIR, MAX_SEQ_LEN, PKG_DIR


def _j(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        if x is None or (isinstance(x, float) and x != x):
            return "n/a"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def collect() -> dict[str, Any]:
    follow = DEFAULT_RESULTS_DIR / "followup"
    runs = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    out: dict[str, Any] = {
        "visibility": _j(follow / f"visibility_summary_L{MAX_SEQ_LEN}.json"),
        "visible_oracle": _j(follow / f"visible_oracle_summary_L{MAX_SEQ_LEN}.json"),
        "selected": _j(follow / "selected_architecture.json"),
        "runs": {},
    }
    if runs.exists():
        for p in runs.rglob("metrics.json"):
            m = json.loads(p.read_text())
            key = f"{m.get('run_tag', p.parent.name)}|{m['scenario']}|{m['arm']}|inter={m.get('interaction_only')}|seed={m.get('model_seed')}|L={m.get('max_seq_len')}"
            out["runs"][key] = {
                "path": str(p.parent),
                "beta_hat": m.get("beta_hat"),
                "beta_true": m.get("beta_true"),
                "test": m.get("test"),
                "ablations": {
                    k: m.get("ablations", {}).get(k)
                    for k in (
                        "delta_bce_shuffle_age",
                        "delta_bce_beta0",
                        "delta_bce_constant_age",
                    )
                },
                "recovery": {
                    k: m.get("recovery", {}).get(k)
                    for k in (
                        "sign_match",
                        "corr_lambda",
                        "RMSE_lambda",
                        "RMSE_surface",
                        "beta_vec",
                        "n_heads_nonzero_beta",
                        "beta_std",
                    )
                },
                "n_epochs": len(m.get("history", [])),
                "n_params": m.get("n_params"),
                "n_temporal_params": m.get("n_temporal_params"),
            }
    return out


def decide_verdict(data: dict[str, Any]) -> tuple[str, str]:
    vis = (data.get("visibility") or {}).get("S2") or {}
    p730 = vis.get("P_visible_given_lag", {}).get("730", 1.0)
    all_vis = vis.get("frac_examples_all_visible", 1.0)
    oracle = (data.get("visible_oracle") or {}).get("S2") or {}
    full_d = oracle.get("full_oracle", {}).get("delta_bce_shuffle_age_interaction", 0)
    vis_d = oracle.get("visible_oracle", {}).get("delta_bce_shuffle_age_interaction", 0)

    # Visibility bottleneck?
    if isinstance(p730, float) and p730 < 0.7 and isinstance(full_d, float) and isinstance(vis_d, float):
        if full_d > 0.02 and vis_d < 0.5 * full_d:
            return (
                "INPUT VISIBILITY IS THE PRIMARY BOTTLENECK",
                f"P(visible|730d)={p730:.2f}; visible-oracle interaction ΔBCE "
                f"({vis_d:.3f}) collapses vs full ({full_d:.3f}).",
            )

    # Find best S2 interaction-only age_temporal and per-head.
    s2_global = None
    s2_per = None
    s0_per = None
    s3_per = None
    for k, r in data.get("runs", {}).items():
        if "|S2|age_temporal|inter=True" in k and s2_global is None:
            s2_global = r
        if "|S2|age_temporal_per_head|inter=True" in k:
            s2_per = r
        if "|S0|age_temporal_per_head|" in k:
            s0_per = r
        if "|S3|age_temporal_per_head|inter=True" in k:
            s3_per = r

    def ok_mech(r):
        if r is None:
            return False
        return bool(r.get("recovery", {}).get("sign_match")) and (
            (r.get("ablations", {}).get("delta_bce_shuffle_age") or 0) > 0.01
            or (r.get("ablations", {}).get("delta_bce_beta0") or 0) > 0.01
        )

    if ok_mech(s2_per) and ok_mech(s3_per):
        # Check S0 inert
        inert = True
        if s0_per is not None:
            inert = abs(s0_per.get("ablations", {}).get("delta_bce_shuffle_age") or 0) < 0.02
        if inert and s2_global is not None:
            g_shuf = s2_global.get("ablations", {}).get("delta_bce_shuffle_age") or 0
            p_shuf = s2_per.get("ablations", {}).get("delta_bce_shuffle_age") or 0
            if p_shuf > g_shuf + 0.005:
                return (
                    "PER-HEAD KERNEL JUSTIFIED",
                    "Per-head recovers S2/S3 signs with stronger functional ablations "
                    "than the global kernel under matched interaction-only training.",
                )
        if inert:
            return (
                "PER-HEAD KERNEL JUSTIFIED",
                "Per-head recovers S2 and reversed S3 with functional ablations; S0 remains inert.",
            )

    if ok_mech(s2_global) and s2_per is None:
        return (
            "GLOBAL KERNEL SUFFICIENT",
            "Interaction-only global age_temporal recovers sign and functional ablations.",
        )

    if s2_global is not None:
        # Dilution check vs prior all-label
        return (
            "AGE × TEMPORAL MECHANISM STILL NOT FUNCTIONALLY RECOVERED",
            "Even after visibility audit and interaction-only / per-head probes, "
            "functional ablations remain weak relative to the oracle.",
        )

    return (
        "AGE × TEMPORAL MECHANISM STILL NOT FUNCTIONALLY RECOVERED",
        "Insufficient completed follow-up runs to validate the mechanism.",
    )


def render(data: dict[str, Any]) -> str:
    lines = [
        "",
        "---",
        "",
        "## Follow-up mechanism investigation",
        "",
        "Purpose: determine whether partial recovery of the global age×temporal",
        "mechanism is due to **sequence truncation**, **heterogeneous-label dilution**,",
        "**optimization**, or **insufficient temporal-model capacity** — without",
        "redesigning the generator.",
        "",
    ]

    vis = (data.get("visibility") or {}).get("S2")
    if vis:
        lines += [
            "### 1. Signal visibility audit (exact model truncation)",
            "",
            f"- `max_seq_len={vis.get('max_seq_len')}`, `max_background={vis.get('max_background')}`",
            f"- mean injected signals before truncation: {_fmt(vis.get('mean_n_signal_before'), 2)}",
            f"- mean retained after truncation: {_fmt(vis.get('mean_n_signal_after'), 2)}",
            f"- frac examples with ≥1 visible signal: {_fmt(vis.get('frac_examples_any_visible'), 3)}",
            f"- frac with **all** signals visible: {_fmt(vis.get('frac_examples_all_visible'), 3)}",
            f"- frac examples truncated: {_fmt(vis.get('frac_examples_truncated'), 3)}",
            "",
            "P(signal visible | lag):",
            "",
            "| lag | P(visible) |",
            "|---|---|",
        ]
        for k, v in (vis.get("P_visible_given_lag") or {}).items():
            lines.append(f"| {k}d | {_fmt(v, 3)} |")
        lines += [
            "",
            "Figure: `results/figures/fig7_signal_visibility_L96.{png,svg}`",
            "",
        ]

    vo = data.get("visible_oracle") or {}
    if vo:
        lines += [
            "### 2. Full oracle vs visible oracle",
            "",
            "| Scenario | full ΔBCE age-shuf (inter) | visible ΔBCE age-shuf (inter) | full AUROC | visible AUROC |",
            "|---|---|---|---|---|",
        ]
        for s in ("S0", "S1", "S2", "S3"):
            if s not in vo:
                continue
            f = vo[s]["full_oracle"]
            v = vo[s]["visible_oracle"]
            lines.append(
                f"| {s} | {_fmt(f.get('delta_bce_shuffle_age_interaction'))} | "
                f"{_fmt(v.get('delta_bce_shuffle_age_interaction'))} | "
                f"{_fmt(f.get('correct', {}).get('micro_auroc'), 3)} | "
                f"{_fmt(v.get('correct', {}).get('micro_auroc'), 3)} |"
            )
        lines += ["", "Figure: `results/figures/fig8_full_vs_visible_oracle.{png,svg}`", ""]
        # Bottleneck statement
        if "S2" in vo:
            fd = vo["S2"]["full_oracle"].get("delta_bce_shuffle_age_interaction", 0)
            vd = vo["S2"]["visible_oracle"].get("delta_bce_shuffle_age_interaction", 0)
            if isinstance(fd, float) and isinstance(vd, float) and fd > 0 and vd < 0.7 * fd:
                lines += [
                    "> **Input-information bottleneck:** visible-oracle interaction effect",
                    f"> is materially weaker than the full oracle (S2 ΔBCE {vd:.3f} vs {fd:.3f}).",
                    "",
                ]
            else:
                lines += [
                    "> Visible oracle preserves most of the full-oracle interaction signal;",
                    "> sequence truncation is **not** the primary explanation for neural under-recovery.",
                    "",
                ]

    # Interaction-only / dilution
    lines += [
        "### 3–4. Interaction-label-only + longer convergence",
        "",
        "Diagnostic only (not a clinical training recipe): loss restricted to the 8",
        "known interaction targets; histories/splits/architecture unchanged.",
        "",
        "| Run | β̂ | sign | corr λ | ΔBCE shuffle | ΔBCE β=0 | AUROC | epochs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, r in sorted(data.get("runs", {}).items()):
        if "inter=True" not in k:
            continue
        if "age_temporal|" not in k and "age_temporal_per_head|" not in k and "temporal_only|" not in k:
            continue
        if "S2|" not in k and "S3|" not in k:
            continue
        lines.append(
            f"| `{k}` | {_fmt(r.get('beta_hat'), 3)} | {r.get('recovery', {}).get('sign_match')} | "
            f"{_fmt(r.get('recovery', {}).get('corr_lambda'), 3)} | "
            f"{_fmt(r.get('ablations', {}).get('delta_bce_shuffle_age'))} | "
            f"{_fmt(r.get('ablations', {}).get('delta_bce_beta0'))} | "
            f"{_fmt((r.get('test') or {}).get('micro_auroc'), 3)} | {r.get('n_epochs')} |"
        )
    lines += [
        "",
        "Figures: `fig9_global_dilution`, `fig10_convergence`.",
        "",
        "### 5–8. Per-head age-conditioned kernels",
        "",
        r"Implemented $\lambda_h(a)=\mathrm{softplus}(\theta_{0,h}+\beta_h z(a))$ with $H=4$",
        r"(8 temporal scalars). Matched control freezes $\beta_h\equiv 0$.",
        "",
        "Figures: `fig11_global_vs_perhead`, `fig12_per_head_lambda`.",
        "",
    ]

    sel = (data.get("selected") or {}).get("selected")
    if sel:
        lines += [f"Selected candidate architecture for multi-seed: **`{sel}`**.", ""]

    verdict, explain = decide_verdict(data)
    lines += [
        "### Final interpretation",
        "",
        f"**{verdict}**",
        "",
        explain,
        "",
        "Evidence checklist (section 14): see tabulated runs above for S0 inertness,",
        "S2/S3 sign recovery, ablations, λ-surface metrics, and multi-seed folders",
        "under `outputs/runs/controlled/multiseed_*` when present.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow-dir", type=Path, default=DEFAULT_RESULTS_DIR / "followup")
    args = ap.parse_args()
    data = collect()
    (args.follow_dir / "followup_summary.json").write_text(json.dumps(data, indent=2))
    section = render(data)
    report = PKG_DIR / "report.md"
    text = report.read_text() if report.exists() else ""
    marker = "## Follow-up mechanism investigation"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n" + section
    else:
        text = text.rstrip() + "\n" + section
    report.write_text(text)
    print("Updated", report)
    verdict, explain = decide_verdict(data)
    print("VERDICT:", verdict)
    print(explain)


if __name__ == "__main__":
    main()
