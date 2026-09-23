#!/usr/bin/env python3
"""Mechanism-identifiability ladder: Phase A → M1 → (M2/M3 if gates pass)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DATA_SEED, DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_DIR, PKG_DIR
from train_factorized import m1_gate, train_factorized

PY = sys.executable
PKG = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(PKG))


def _j(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def fig15_ablation(results: dict, fig_dir: Path) -> None:
    variants = ["full", "signal_only", "bg_randomized"]
    metrics = ["delta_bce_shuffle_age", "delta_bce_beta0", "delta_auroc"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    x = np.arange(len(variants))
    width = 0.35
    for ax, key, title in zip(
        axes,
        ["delta_bce_shuffle_age", "delta_bce_beta0", "auroc_gain"],
        [r"$\Delta$BCE shuffle", r"$\Delta$BCE $\beta{=}0$", "AUROC gain"],
    ):
        to_vals, at_vals = [], []
        for v in variants:
            r = results.get(v, {})
            to = r.get("temporal_only", {})
            at = r.get("age_temporal", {})
            if key == "auroc_gain":
                to_vals.append(0.0)
                at_vals.append(
                    (at.get("test", {}).get("micro_auroc", np.nan) or np.nan)
                    - (to.get("test", {}).get("micro_auroc", np.nan) or np.nan)
                )
            else:
                to_vals.append(to.get("ablations", {}).get(key, np.nan))
                at_vals.append(at.get("ablations", {}).get(key, np.nan))
        ax.bar(x - width / 2, to_vals, width, label="temporal_only", color="#718096")
        ax.bar(x + width / 2, at_vals, width, label="age_temporal", color="#c05621")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 15 — Background-content ablation (S2)", y=1.02)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig15_background_ablation.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig15_background_ablation.svg", bbox_inches="tight")
    plt.close(fig)


def fig16_ladder(ladder: dict, fig_dir: Path) -> None:
    names = list(ladder.keys())
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    x = np.arange(len(names))
    gains = [ladder[n].get("delta_auroc", np.nan) for n in names]
    shuf = [ladder[n].get("delta_bce_shuffle", np.nan) for n in names]
    b0 = [ladder[n].get("delta_bce_beta0", np.nan) for n in names]
    for ax, vals, title in zip(
        axes,
        [gains, shuf, b0],
        ["AUROC gain vs temporal-only", r"$\Delta$BCE shuffle", r"$\Delta$BCE $\beta{=}0$"],
    ):
        ax.bar(x, vals, color="#2b6cb0")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Figure 16 — Architecture ladder", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig16_architecture_ladder.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig16_architecture_ladder.svg", bbox_inches="tight")
    plt.close(fig)


def fig17_18_lambda_surface(metrics: dict, fig_dir: Path, tag: str) -> None:
    rec = metrics.get("recovery", {})
    ages = sorted(float(a) for a in rec.get("lambda_true_by_age", {}).keys())
    if not ages:
        return
    lt = [rec["lambda_true_by_age"][str(a)] for a in ages]
    ll = [rec["lambda_learned_by_age"][str(a)] for a in ages]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ages, lt, "k--", lw=2, label=r"$\lambda_{\mathrm{true}}$")
    ax.plot(ages, ll, color="#c05621", lw=2, label=r"$\lambda_{\mathrm{learned}}$")
    ax.set_xlabel("Age")
    ax.set_ylabel(r"$\lambda(a)$")
    ax.set_title(f"Figure 17 — True vs learned λ(a) ({tag})")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / f"fig17_lambda_{tag}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig17_lambda_{tag}.svg", bbox_inches="tight")
    plt.close(fig)

    # Surface
    from config import SURFACE_LAGS_DAYS, relevance, tau_from_days, lambda_true

    beta_t = rec["beta_true"]
    theta_t = rec["theta0_true"]
    beta_h = rec["beta_hat"]
    theta_h = rec["theta0_hat"]
    lags = list(SURFACE_LAGS_DAYS)
    R_t = np.zeros((len(lags), len(ages)))
    R_l = np.zeros_like(R_t)
    for i, d in enumerate(lags):
        t = float(tau_from_days(d))
        for j, a in enumerate(ages):
            R_t[i, j] = relevance(a, t, theta_t, beta_t)
            lam = float(lambda_true(a, theta_h, beta_h))
            R_l[i, j] = np.exp(-lam * t)
    err = np.abs(R_l - R_t)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, M, title, cmap in zip(
        axes, [R_t, R_l, err], ["Truth", "Learned", "|Error|"], ["viridis", "viridis", "magma"]
    ):
        im = ax.imshow(M, aspect="auto", origin="lower", extent=[0, 18, 0, len(lags) - 1], cmap=cmap)
        ax.set_yticks(range(len(lags)))
        ax.set_yticklabels([f"{int(d)}d" for d in lags])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Figure 18 — Age×lag surface ({tag})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig18_surface_{tag}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"fig18_surface_{tag}.svg", bbox_inches="tight")
    plt.close(fig)


def fig19_additive_vs_softmax(add_m: dict, soft_m: dict, fig_dir: Path) -> None:
    labels = ["additive", "softmax"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3)
    width = 0.35
    keys = ["delta_auroc", "delta_bce_shuffle", "delta_bce_beta0"]
    titles = ["ΔAUROC", "ΔBCE shuffle", "ΔBCE β=0"]
    # Recompute from paired temporal_only if available in gate blobs
    add_vals = [
        add_m.get("delta_auroc", np.nan),
        add_m.get("delta_bce_shuffle", np.nan),
        add_m.get("delta_bce_beta0", np.nan),
    ]
    soft_vals = [
        soft_m.get("delta_auroc", np.nan),
        soft_m.get("delta_bce_shuffle", np.nan),
        soft_m.get("delta_bce_beta0", np.nan),
    ]
    ax.bar(x - width / 2, add_vals, width, label="additive", color="#2b6cb0")
    ax.bar(x + width / 2, soft_vals, width, label="softmax", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_title("Figure 19 — Additive vs softmax temporal aggregation (M1)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "fig19_additive_vs_softmax.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "fig19_additive_vs_softmax.svg", bbox_inches="tight")
    plt.close(fig)


def append_report(text: str) -> None:
    report = PKG_DIR / "report.md"
    body = report.read_text() if report.exists() else ""
    marker = "## Mechanism-identifiability and factorized architecture investigation"
    if marker in body:
        body = body.split(marker)[0].rstrip()
    report.write_text(body + "\n\n" + text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-seed", type=int, default=DATA_SEED)
    ap.add_argument("--skip-phase-a", action="store_true")
    ap.add_argument("--skip-ablation-train", action="store_true")
    ap.add_argument("--skip-m1", action="store_true", help="Reuse existing M1 metrics under factorized/")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    data_root = DEFAULT_OUTPUT_DIR / "data" / f"seed{args.data_seed}" / "controlled"
    s2 = data_root / "S2"
    s3 = data_root / "S3"
    runs = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    follow = DEFAULT_RESULTS_DIR / "followup"
    fig_dir = DEFAULT_RESULTS_DIR / "figures"
    follow.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Phase A ----------
    if not args.skip_phase_a:
        run([PY, "phase_a_diagnostics.py", "--device", args.device])

    # A3 variants
    run([PY, "background_ablation.py", "--scenario", "S2"])
    run([PY, "background_ablation.py", "--scenario", "S3"])

    ablation_results: dict[str, Any] = {}
    if not args.skip_ablation_train:
        for scen in ("S2", "S3"):
            beta = -2.5 if scen == "S2" else 2.5
            abl_root = data_root / f"{scen}_ablation"
            for variant in ("full", "signal_only", "bg_randomized"):
                vdir = abl_root / variant
                if not (vdir / "READY").exists() and not (vdir / "examples.parquet").exists():
                    continue
                # Ensure READY exists for train.py gate
                if not (vdir / "READY").exists():
                    (vdir / "READY").write_text("copied from parent\n")
                key = f"{scen}/{variant}"
                ablation_results[key] = {}
                for age_temporal, tag in ((False, "temporal_only"), (True, "age_temporal")):
                    rdir = runs / f"ablation_{scen}_{variant}_{tag}_m0"
                    # Use Transformer train.py for ablation (bypass test on current arch)
                    cmd = [
                        PY,
                        "train.py",
                        "--scenario",
                        scen,
                        "--arm",
                        tag,
                        "--cohort",
                        "controlled",
                        "--epochs",
                        str(min(args.epochs, 25)),
                        "--patience",
                        "8",
                        "--device",
                        args.device,
                        "--interaction-only",
                        "--scenario-dir",
                        str(vdir),
                        "--run-tag",
                        f"ablation_{variant}",
                    ]
                    run(cmd)
                    # Find metrics — prefer explicit run_tag naming
                    cands = sorted(
                        runs.glob(f"ablation_{variant}_{scen}_{tag}_*_interonly/metrics.json")
                    )
                    if not cands:
                        cands = sorted(runs.glob(f"*{variant}*{scen}*{tag}*/metrics.json"))
                    if cands:
                        ablation_results[key][tag] = _j(cands[-1])
                        print(f"Ablation metrics {key}/{tag}: {cands[-1]}")
                    else:
                        print(f"WARNING: no metrics for ablation {key}/{tag}")

        # Also store S2-focused dict for fig15
        s2_abl = {
            v: ablation_results.get(f"S2/{v}", {})
            for v in ("full", "signal_only", "bg_randomized")
        }
        (follow / "background_ablation_results.json").write_text(
            json.dumps(ablation_results, indent=2)
        )
        fig15_ablation(s2_abl, fig_dir)

        # Decision
        full_at = (ablation_results.get("S2/full") or {}).get("age_temporal") or {}
        sig_at = (ablation_results.get("S2/signal_only") or {}).get("age_temporal") or {}
        full_shuf = (full_at.get("ablations") or {}).get("delta_bce_shuffle_age", 0) or 0
        sig_shuf = (sig_at.get("ablations") or {}).get("delta_bce_shuffle_age", 0) or 0
        if sig_shuf > max(0.02, 2 * full_shuf):
            bypass = "BYPASS CONFIRMED: CONTENT PATH ENCODES AGE/DEVELOPMENTAL STATE"
        else:
            bypass = "BYPASS NOT EXPLAINED BY BACKGROUND CONTENT"
        (follow / "bypass_decision.json").write_text(
            json.dumps(
                {"decision": bypass, "full_shuffle": full_shuf, "signal_only_shuffle": sig_shuf},
                indent=2,
            )
        )
        print("BYPASS DECISION:", bypass)

    # ---------- Phase B: M1 ----------
    m1_dir = runs / "factorized"
    m1_dir.mkdir(parents=True, exist_ok=True)
    m1_results = {}
    for agg in ("additive", "softmax", "additive_mass"):
        pair = {}
        for age_temporal, tag in ((False, "temporal_only"), (True, "age_temporal")):
            rdir = m1_dir / f"M1_S2_{tag}_{agg}_m0"
            metrics_path = rdir / "metrics.json"
            if args.skip_m1 and metrics_path.exists():
                pair[tag] = _j(metrics_path)
                print(f"Reuse M1 {tag}/{agg} from {metrics_path}")
            else:
                pair[tag] = train_factorized(
                    family="M1",
                    age_temporal=age_temporal,
                    scenario_dir=s2,
                    run_dir=rdir,
                    beta_true=-2.5,
                    aggregation=agg,
                    interaction_only=True,
                    max_epochs=args.epochs,
                    device=args.device,
                )
        gate = m1_gate(pair["age_temporal"], pair["temporal_only"], -2.5)
        m1_results[agg] = {"pair": pair, "gate": gate}
        print(f"M1/{agg} GATE:", json.dumps(gate, indent=2))

    (follow / "M1_results.json").write_text(
        json.dumps(
            {k: {"gate": v["gate"], "age_temporal": v["pair"]["age_temporal"]} for k, v in m1_results.items()},
            indent=2,
        )
    )
    fig19_additive_vs_softmax(
        m1_results["additive"]["gate"], m1_results["softmax"]["gate"], fig_dir
    )
    # Prefer additive_mass if additive fails magnitude
    primary = "additive"
    if not m1_results["additive"]["gate"]["passed"] and m1_results["additive_mass"]["gate"]["passed"]:
        primary = "additive_mass"
        print("M1.1: additive_mass passes where additive failed — magnitude preservation needed.")
    elif m1_results["additive"]["gate"]["passed"] and not m1_results["softmax"]["gate"]["passed"]:
        print("SOFTMAX NORMALIZATION DESTROYS ABSOLUTE TEMPORAL EVIDENCE MAGNITUDE")

    m1_pass = m1_results[primary]["gate"]["passed"]
    selected = None
    ladder = {
        "M1_" + primary: {
            "delta_auroc": m1_results[primary]["gate"]["delta_auroc"],
            "delta_bce_shuffle": m1_results[primary]["gate"]["delta_bce_shuffle"],
            "delta_bce_beta0": m1_results[primary]["gate"]["delta_bce_beta0"],
        }
    }
    fig17_18_lambda_surface(m1_results[primary]["pair"]["age_temporal"], fig_dir, f"M1_{primary}")

    m2_pass = m3_pass = False
    m2_results = m3_results = None

    if not m1_pass:
        verdict = "MECHANISM NOT RECOVERED — BLOCKER IDENTIFIED: M1 kernel-only failed (isolated mechanism not optimized / representation mismatch)"
        print(verdict)
    else:
        # ---------- Phase C: M2 ----------
        m2_pair = {}
        for age_temporal, tag in ((False, "temporal_only"), (True, "age_temporal")):
            rdir = m1_dir / f"M2_S2_{tag}_{primary}_m0"
            m2_pair[tag] = train_factorized(
                family="M2",
                age_temporal=age_temporal,
                scenario_dir=s2,
                run_dir=rdir,
                beta_true=-2.5,
                aggregation=primary if primary != "additive_mass" else "additive",
                interaction_only=True,
                max_epochs=args.epochs,
                device=args.device,
            )
        # For M2 use same aggregation name in gate compare
        agg_m2 = primary if primary != "additive_mass" else "additive"
        m2_gate_res = m1_gate(m2_pair["age_temporal"], m2_pair["temporal_only"], -2.5)
        m2_results = {"pair": m2_pair, "gate": m2_gate_res}
        m2_pass = m2_gate_res["passed"]
        ladder["M2"] = {
            "delta_auroc": m2_gate_res["delta_auroc"],
            "delta_bce_shuffle": m2_gate_res["delta_bce_shuffle"],
            "delta_bce_beta0": m2_gate_res["delta_bce_beta0"],
        }
        print("M2 GATE:", json.dumps(m2_gate_res, indent=2))
        (follow / "M2_results.json").write_text(json.dumps({"gate": m2_gate_res, "age_temporal": m2_pair["age_temporal"]}, indent=2))

        if not m2_pass:
            verdict = "MECHANISM NOT RECOVERED — BLOCKER IDENTIFIED: M1 passed but M2 content×gate reintroduced bypass/scale competition"
            selected = f"M1_{primary}"
        else:
            # ---------- Phase D: M3 ----------
            m3_pair = {}
            for age_temporal, tag in ((False, "temporal_only"), (True, "age_temporal")):
                rdir = m1_dir / f"M3_S2_{tag}_{agg_m2}_m0"
                m3_pair[tag] = train_factorized(
                    family="M3",
                    age_temporal=age_temporal,
                    scenario_dir=s2,
                    run_dir=rdir,
                    beta_true=-2.5,
                    aggregation=agg_m2,
                    interaction_only=True,
                    max_epochs=args.epochs,
                    device=args.device,
                )
            m3_gate_res = m1_gate(m3_pair["age_temporal"], m3_pair["temporal_only"], -2.5)
            m3_results = {"pair": m3_pair, "gate": m3_gate_res}
            m3_pass = m3_gate_res["passed"]
            ladder["M3"] = {
                "delta_auroc": m3_gate_res["delta_auroc"],
                "delta_bce_shuffle": m3_gate_res["delta_bce_shuffle"],
                "delta_bce_beta0": m3_gate_res["delta_bce_beta0"],
            }
            print("M3 GATE:", json.dumps(m3_gate_res, indent=2))
            (follow / "M3_results.json").write_text(
                json.dumps({"gate": m3_gate_res, "age_temporal": m3_pair["age_temporal"]}, indent=2)
            )
            fig17_18_lambda_surface(m3_pair["age_temporal"], fig_dir, "M3")

            if m3_pass:
                selected = "M3"
                # Phase E: S0–S3
                s0s3 = {}
                for scen, beta in (("S0", 0.0), ("S1", 0.0), ("S2", -2.5), ("S3", 2.5)):
                    sdir = data_root / scen
                    pair = {}
                    for age_temporal, tag in ((False, "temporal_only"), (True, "age_temporal")):
                        rdir = m1_dir / f"M3_{scen}_{tag}_m0"
                        pair[tag] = train_factorized(
                            family="M3",
                            age_temporal=age_temporal,
                            scenario_dir=sdir,
                            run_dir=rdir,
                            beta_true=beta,
                            aggregation=agg_m2,
                            interaction_only=(scen in ("S2", "S3")),
                            max_epochs=args.epochs,
                            device=args.device,
                        )
                    s0s3[scen] = {
                        "gate": m1_gate(pair["age_temporal"], pair["temporal_only"], beta)
                        if abs(beta) > 0
                        else {
                            "beta_hat": pair["age_temporal"]["beta_hat"],
                            "delta_bce_shuffle": pair["age_temporal"]["ablations"][
                                "delta_bce_shuffle_age"
                            ],
                        },
                        "age_temporal": pair["age_temporal"],
                    }
                (follow / "M3_S0S3.json").write_text(json.dumps(s0s3, indent=2))
                # Check S0/S1 inert + S2/S3 pass
                s0_ok = abs(s0s3["S0"]["age_temporal"]["beta_hat"]) < 0.3 and abs(
                    s0s3["S0"]["age_temporal"]["ablations"]["delta_bce_shuffle_age"]
                ) < 0.02
                s2_ok = s0s3["S2"]["gate"].get("passed", False)
                s3_ok = s0s3["S3"]["gate"].get("passed", False) and s0s3["S3"]["age_temporal"][
                    "beta_hat"
                ] > 0
                if s0_ok and s2_ok and s3_ok:
                    verdict = "MECHANISM FUNCTIONALLY RECOVERED — READY FOR MULTI-SEED"
                elif s2_ok and not s3_ok:
                    verdict = "MECHANISM NOT RECOVERED — BLOCKER IDENTIFIED: S3 falsification failed"
                else:
                    verdict = "MECHANISM RECOVERED BUT M3 FAILS — REAL-EHR ARCHITECTURE NOT READY"
            else:
                selected = "M2"
                verdict = "MECHANISM RECOVERED BUT M3 FAILS — REAL-EHR ARCHITECTURE NOT READY"

    fig16_ladder(ladder, fig_dir)

    # Transformer reference on ladder if available
    tr_at = _j(runs / "followup_S2_age_temporal_d20260922_m0_interonly" / "metrics.json")
    tr_to = _j(runs / "followup_S2_temporal_only_d20260922_m0_interonly" / "metrics.json")
    if tr_at and tr_to:
        ladder["Transformer"] = {
            "delta_auroc": tr_at["test"]["micro_auroc"] - tr_to["test"]["micro_auroc"],
            "delta_bce_shuffle": tr_at["ablations"]["delta_bce_shuffle_age"],
            "delta_bce_beta0": tr_at["ablations"]["delta_bce_beta0"],
        }
        fig16_ladder(ladder, fig_dir)

    age_probe = _j(follow / "followup_age_probe.json")
    cf = _j(follow / "counterfactual_age_sensitivity.json")
    bypass = _j(follow / "bypass_decision.json")

    report = f"""## Mechanism-identifiability and factorized architecture investigation

Goal: identify and remove the architectural bypass that allows prediction without
functionally using age×time. Generator / splits / S0–S3 ground truth unchanged.

### 1. Age-decoding probe (A1)

See `results/followup/followup_age_probe.json` and fig13.

Summary (pooled representation, TEST):

"""
    if age_probe:
        for arm, sites in age_probe.items():
            p = sites.get("pooled", {})
            report += (
                f"- **{arm}**: MAE={p.get('mae', float('nan')):.3f}y, "
                f"R²={p.get('r2', float('nan')):.3f}, "
                f"band_acc={p.get('band_acc', float('nan')):.3f}\n"
            )
    report += f"""
### 2. Counterfactual age sensitivity (A2)

See `results/followup/counterfactual_age_sensitivity.json` and fig14.

Mean interaction-label P(y) Δ(a=17−a=2):
"""
    if cf:
        for k, v in cf.get("mean_p", {}).items():
            report += f"- {k}: Δ={v.get('delta_17_minus_2', float('nan')):.4f}\n"

    report += f"""
### 3. Background-content ablation (A3)

Decision: **{(bypass or {}).get('decision', 'n/a')}**

full shuffle ΔBCE={(bypass or {}).get('full_shuffle')} · signal-only shuffle ΔBCE={(bypass or {}).get('signal_only_shuffle')}

Figure: `fig15_background_ablation`.

### 4. M1 kernel-only

Primary aggregation: **{primary}**

| Aggregation | passed | ΔAUROC | ΔBCE shuffle | ΔBCE β=0 | corr λ |
|---|---|---|---|---|---|
"""
    for agg, blob in m1_results.items():
        g = blob["gate"]
        report += (
            f"| {agg} | {g['passed']} | {g['delta_auroc']:.4f} | "
            f"{g['delta_bce_shuffle']:.4f} | {g['delta_bce_beta0']:.4f} | "
            f"{g.get('corr_lambda')} |\n"
        )

    if m1_results["additive"]["gate"]["passed"] and not m1_results["softmax"]["gate"]["passed"]:
        report += (
            "\n**SOFTMAX NORMALIZATION DESTROYS ABSOLUTE TEMPORAL EVIDENCE MAGNITUDE**\n"
        )

    report += f"""
### 5–7. M2 / M3

- M2 reached: {m2_results is not None}; passed: {m2_pass}
- M3 reached: {m3_results is not None}; passed: {m3_pass}
- Selected architecture: `{selected}`

### 8–10. Final verdict

**{verdict}**

Selected={selected}; M1_pass={m1_pass}; M2_pass={m2_pass}; M3_pass={m3_pass}.

Figures: fig13–fig19 under `results/figures/`.
Artifacts under `results/followup/` and `outputs/runs/controlled/factorized/`.
"""
    append_report(report)
    (follow / "final_verdict.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "selected": selected,
                "m1_pass": m1_pass,
                "m2_pass": m2_pass,
                "m3_pass": m3_pass,
                "primary_aggregation": primary,
            },
            indent=2,
        )
    )
    print("FINAL VERDICT:", verdict)


if __name__ == "__main__":
    main()
