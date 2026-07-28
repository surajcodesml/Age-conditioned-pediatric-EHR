#!/usr/bin/env python3
"""Run the pretrained age-route audit (T0–T7).

    python -m model_new.audit.run_audit \\
        --run_root model_new/run \\
        --out_dir model_new/audit \\
        --max_val_batches 80 \\
        --n_perm 100

Nothing is trained. All printing / JSON goes through ``model_new.diagnostics``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from model_new import diagnostics as D
from model_new.audit import AUDIT_SEED
from model_new.audit.common import (
    DEFAULT_OUT_DIR,
    DEFAULT_RUN_ROOT,
    REPO_ROOT,
    build_shared_context,
    discover_runs,
)
from model_new.audit.report import write_report
from model_new.audit.t0_provenance import run_t0
from model_new.audit.t1_parity import run_t1
from model_new.audit.t2_generator import run_t2
from model_new.audit.t3_delta_alpha import run_t3
from model_new.audit.t4_softmax import run_t4
from model_new.audit.t5_intervention import run_t5
from model_new.audit.t6_permutation import run_t6
from model_new.audit.t7_localization import run_t7


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=AUDIT_SEED)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max_val_batches", type=int, default=None,
                   help="Cap validation batches for T1/T5/T6/T7. None = full held-out split. "
                        "Recorded in age_audit.json; never silently substituted.")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--n_perm", type=int, default=100)
    p.add_argument("--skip_t1", action="store_true")
    p.add_argument("--skip_t5", action="store_true")
    p.add_argument("--skip_t6", action="store_true")
    p.add_argument("--force_t7", action="store_true",
                   help="Run T7 even if T3–T6 are not all null.")
    p.add_argument("--allow_config_diff", nargs="*", default=["optim.epochs"],
                   help="Named config keys accepted as benign drift (recorded in JSON). "
                        "Default allows optim.epochs (vanilla/rc scheduled 20; kernel/additive 11).")
    return p.parse_args(argv)


def _t3_t6_null(t3, t4, t5, t6) -> bool:
    """T7 when the primary claim (T5∧T6) fails — especially T6 null."""
    return not bool(t6.get("kernel_age_dependent"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    run_root = args.run_root if args.run_root.is_absolute() else REPO_ROOT / args.run_root
    device = torch.device(args.device)
    t_wall0 = time.time()

    D.print_block("age audit start", [
        f"run_root={run_root}",
        f"out_dir={out_dir}",
        f"seed={args.seed}  device={device}  batch_size={args.batch_size}",
        f"max_val_batches={args.max_val_batches}  n_boot={args.n_boot}  n_perm={args.n_perm}",
    ])

    run_dirs = discover_runs(run_root)
    ctx = build_shared_context(
        run_dirs, seed=args.seed, device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, max_val_batches=args.max_val_batches,
        allow_config_diff=set(args.allow_config_diff or []),
    )

    audit: dict = {
        "meta": {
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_val_batches": args.max_val_batches,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "n_examples": ctx["n_examples"],
            "n_batches": ctx["n_batches"],
            "batch_order_hash": ctx["batch_order_hash"],
            "run_dirs": ctx["run_dirs"],
            "started_unix": t_wall0,
        }
    }

    # ---- T0 ---- #
    t0 = run_t0(ctx)
    audit["t0"] = t0
    D.write_json(out_dir / "age_audit.json", audit)
    if not t0["ok"]:
        D.print_block("ABORT", ["T0 provenance failed; refusing later comparisons."])
        write_report(audit, out_dir, {a: Path(p) for a, p in ctx["run_dirs"].items()})
        return 2

    # ---- T2 (cheap; before heavy forwards) ---- #
    audit["t2"] = run_t2(ctx)
    D.write_json(out_dir / "age_audit.json", audit)

    # ---- T3 ---- #
    audit["t3"] = run_t3(ctx)
    D.write_json(out_dir / "age_audit.json", audit)

    # ---- T4 ---- #
    audit["t4"] = run_t4(ctx)
    D.write_json(out_dir / "age_audit.json", audit)

    # ---- T1 ---- #
    if not args.skip_t1:
        audit["t1"] = run_t1(ctx, n_boot=args.n_boot)
        D.write_json(out_dir / "age_audit.json", audit)
    else:
        audit["t1"] = {"skipped": True}

    # ---- T5 ---- #
    if not args.skip_t5:
        audit["t5"] = run_t5(ctx, n_boot=args.n_boot)
        D.write_json(out_dir / "age_audit.json", audit)
    else:
        audit["t5"] = {"skipped": True, "kernel_effect_nonzero": False}

    # ---- T6 ---- #
    if not args.skip_t6:
        audit["t6"] = run_t6(ctx, n_perm=args.n_perm)
        D.write_json(out_dir / "age_audit.json", audit)
    else:
        audit["t6"] = {"skipped": True, "kernel_age_dependent": False}

    # ---- T7 if null (or forced) ---- #
    null = _t3_t6_null(audit["t3"], audit["t4"], audit["t5"], audit["t6"])
    if null or args.force_t7:
        audit["t7"] = run_t7(ctx, audit["t2"], audit["t3"], audit["t4"])
        audit["t7_reason"] = "forced" if args.force_t7 and not null else "T3–T6 null"
    else:
        audit["t7"] = None
        audit["t7_reason"] = "skipped: T5/T6 indicate an active age route"

    audit["meta"]["wall_clock_s"] = time.time() - t_wall0
    D.write_json(out_dir / "age_audit.json", audit)

    write_report(audit, out_dir, {a: Path(p) for a, p in ctx["run_dirs"].items()})
    D.print_block("age audit done", [
        f"wall_clock_s={audit['meta']['wall_clock_s']:.1f}",
        f"json={out_dir / 'age_audit.json'}",
        f"report={out_dir / 'AGE_AUDIT_REPORT.md'}",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
