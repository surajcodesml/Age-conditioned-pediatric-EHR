#!/usr/bin/env python3
"""Follow-up mechanism investigation orchestration.

Runs visibility audit → visible oracle → interaction-only / long convergence →
per-head comparison → optional seq-len sweep → figures → report appendix.
Does **not** retune the generator.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import DEFAULT_OUTPUT_DIR, DEFAULT_RESULTS_DIR, DATA_SEED, MAX_SEQ_LEN

PKG = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(PKG))


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def find_run(cohort: str, scenario: str, arm: str, **flags) -> Path | None:
    root = DEFAULT_OUTPUT_DIR / "runs" / cohort
    if not root.exists():
        return None
    cands = []
    for p in root.glob(f"{scenario}_{arm}_*"):
        mpath = p / "metrics.json"
        if not mpath.exists():
            continue
        m = load_metrics(mpath)
        ok = True
        for k, v in flags.items():
            if m.get(k) != v:
                ok = False
                break
        if ok:
            cands.append(p)
    return sorted(cands)[-1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cohort", default="controlled")
    ap.add_argument("--data-seed", type=int, default=DATA_SEED)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-seqlen", action="store_true")
    ap.add_argument("--skip-multiseed", action="store_true")
    ap.add_argument("--epochs-inter", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--epochs-arch", type=int, default=40)
    ap.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    data_root = DEFAULT_OUTPUT_DIR / "data" / f"seed{args.data_seed}"
    follow_dir = DEFAULT_RESULTS_DIR / "followup"
    fig_dir = DEFAULT_RESULTS_DIR / "figures"
    follow_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1–2. Visibility + visible oracle ---------------------------------
    if not args.skip_audit:
        run(
            [
                PY,
                "audit_visibility.py",
                "--data-root",
                str(data_root),
                "--cohort",
                args.cohort,
                "--max-seq-len",
                str(MAX_SEQ_LEN),
                "--out-dir",
                str(follow_dir),
            ]
        )

    if args.skip_train:
        print("Skipping training; generating figures/report from existing runs.")
    else:
        # ---- 3–4. Interaction-only + long convergence (S2/S3) --------------
        for scen in ("S2", "S3"):
            for arm in ("temporal_only", "age_temporal"):
                run(
                    [
                        PY,
                        "train.py",
                        "--scenario",
                        scen,
                        "--arm",
                        arm,
                        "--cohort",
                        args.cohort,
                        "--data-seed",
                        str(args.data_seed),
                        "--model-seed",
                        "0",
                        "--epochs",
                        str(args.epochs_inter),
                        "--patience",
                        str(args.patience),
                        "--device",
                        args.device,
                        "--interaction-only",
                        "--track-mechanism",
                        "--scenario-dir",
                        str(data_root / args.cohort / scen),
                        "--run-tag",
                        "followup",
                    ]
                )

        # ---- 5–8. Global vs per-head on S0–S3 (interaction-only for S2/S3) -
        # Architecture comparison on interaction labels for S2/S3; all labels for S0/S1.
        arch_arms = (
            "temporal_only",
            "age_temporal",
            "temporal_only_per_head",
            "age_temporal_per_head",
        )
        for scen in ("S0", "S1", "S2", "S3"):
            inter = scen in ("S2", "S3")
            for arm in arch_arms:
                cmd = [
                    PY,
                    "train.py",
                    "--scenario",
                    scen,
                    "--arm",
                    arm,
                    "--cohort",
                    args.cohort,
                    "--data-seed",
                    str(args.data_seed),
                    "--model-seed",
                    "0",
                    "--epochs",
                    str(args.epochs_arch),
                    "--patience",
                    str(args.patience),
                    "--device",
                    args.device,
                    "--scenario-dir",
                    str(data_root / args.cohort / scen),
                    "--run-tag",
                    "arch",
                ]
                if inter:
                    cmd.append("--interaction-only")
                run(cmd)

        # ---- 11. Sequence-length follow-up (global only) if audit suggests --
        vis_path = follow_dir / f"visibility_summary_L{MAX_SEQ_LEN}.json"
        need_seqlen = False
        if vis_path.exists():
            vis = json.loads(vis_path.read_text())
            p730 = vis.get("S2", {}).get("P_visible_given_lag", {}).get("730", 1.0)
            all_vis = vis.get("S2", {}).get("frac_examples_all_visible", 1.0)
            if (isinstance(p730, float) and p730 < 0.95) or (
                isinstance(all_vis, float) and all_vis < 0.95
            ):
                need_seqlen = True
        if need_seqlen and not args.skip_seqlen:
            for L in (96, 192, 384):
                run(
                    [
                        PY,
                        "audit_visibility.py",
                        "--data-root",
                        str(data_root),
                        "--cohort",
                        args.cohort,
                        "--scenarios",
                        "S2",
                        "S3",
                        "--max-seq-len",
                        str(L),
                        "--out-dir",
                        str(follow_dir),
                    ]
                )
                for arm in ("temporal_only", "age_temporal"):
                    run(
                        [
                            PY,
                            "train.py",
                            "--scenario",
                            "S2",
                            "--arm",
                            arm,
                            "--cohort",
                            args.cohort,
                            "--data-seed",
                            str(args.data_seed),
                            "--model-seed",
                            "0",
                            "--epochs",
                            str(args.epochs_arch),
                            "--patience",
                            str(args.patience),
                            "--device",
                            args.device,
                            "--interaction-only",
                            "--max-seq-len",
                            str(L),
                            "--scenario-dir",
                            str(data_root / args.cohort / "S2"),
                            "--run-tag",
                            "seqlen",
                        ]
                    )

        # ---- 12. Multi-seed for selected candidate ------------------------
        # Decide after reading arch results; default to per-head if it beats global.
        decision = "age_temporal_per_head"
        try:
            g = load_metrics(
                next(
                    (DEFAULT_OUTPUT_DIR / "runs" / args.cohort).glob(
                        "arch_S2_age_temporal_*_interonly/metrics.json"
                    )
                )
            )
            p = load_metrics(
                next(
                    (DEFAULT_OUTPUT_DIR / "runs" / args.cohort).glob(
                        "arch_S2_age_temporal_per_head_*_interonly/metrics.json"
                    )
                )
            )
            # Prefer larger shuffle delta + better AUROC.
            score_g = g["ablations"]["delta_bce_shuffle_age"] + 0.5 * (
                g["test"]["micro_auroc"] - 0.5
            )
            score_p = p["ablations"]["delta_bce_shuffle_age"] + 0.5 * (
                p["test"]["micro_auroc"] - 0.5
            )
            decision = (
                "age_temporal_per_head" if score_p >= score_g else "age_temporal"
            )
        except StopIteration:
            pass
        (follow_dir / "selected_architecture.json").write_text(
            json.dumps({"selected": decision}, indent=2)
        )

        if not args.skip_multiseed:
            control = (
                "temporal_only_per_head"
                if decision == "age_temporal_per_head"
                else "temporal_only"
            )
            for seed in args.model_seeds:
                for scen in ("S0", "S1", "S2", "S3"):
                    inter = scen in ("S2", "S3")
                    for arm in (decision, control):
                        cmd = [
                            PY,
                            "train.py",
                            "--scenario",
                            scen,
                            "--arm",
                            arm,
                            "--cohort",
                            args.cohort,
                            "--data-seed",
                            str(args.data_seed),
                            "--model-seed",
                            str(seed),
                            "--epochs",
                            str(args.epochs_arch),
                            "--patience",
                            str(args.patience),
                            "--device",
                            args.device,
                            "--scenario-dir",
                            str(data_root / args.cohort / scen),
                            "--run-tag",
                            "multiseed",
                        ]
                        if inter:
                            cmd.append("--interaction-only")
                        run(cmd)

    # ---- Figures + report -------------------------------------------------
    run([PY, "plots_followup.py", "--follow-dir", str(follow_dir), "--fig-dir", str(fig_dir)])
    run([PY, "update_followup_report.py", "--follow-dir", str(follow_dir)])


if __name__ == "__main__":
    main()
