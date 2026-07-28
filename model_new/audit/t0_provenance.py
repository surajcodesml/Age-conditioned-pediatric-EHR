"""T0 — Provenance: shared seed / buffers / init across arms; selected epochs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model_new import diagnostics as D
from model_new.audit.common import (
    ARMS,
    backbone_init_hash,
    build_model,
    fourier_buffer_digest,
    load_checkpoint,
    read_json,
)
from model_new.eval_pretrain import EXPECTED_TAU_MAX


def run_t0(ctx: dict) -> dict:
    shared = ctx["shared"]
    configs = ctx["configs"]
    run_dirs = {a: Path(p) for a, p in ctx["run_dirs"].items()}
    device = ctx["device"]
    selected = ctx["selected"]

    seeds = {a: int(configs[a]["seed"]) for a in ARMS}
    tau_maxes = {a: float(configs[a]["model"]["tau_max"]) for a in ARMS}
    age_means = {a: float(configs[a]["model"]["age_standardization"]["mean"]) for a in ARMS}
    age_sds = {a: float(configs[a]["model"]["age_standardization"]["sd"]) for a in ARMS}
    cheb_s = {a: int(configs[a]["model"]["s"]) for a in ARMS}
    race = {a: str(configs[a]["model"]["race_encoding"]) for a in ARMS}
    fourier = {a: dict(configs[a]["model"]["fourier"]) for a in ARMS}

    mismatches: list[str] = []

    def _all_same(d: dict, label: str, tol: float | None = None) -> Any:
        vals = list(d.values())
        ref = vals[0]
        for a, v in d.items():
            ok = (abs(float(v) - float(ref)) <= tol) if tol is not None else (v == ref)
            if not ok:
                mismatches.append(f"{label}: {a}={v!r} vs ref={ref!r}")
        return ref

    seed = _all_same(seeds, "seed")
    tau = _all_same(tau_maxes, "tau_max", tol=1e-12)
    age_mean = _all_same(age_means, "age_mean", tol=1e-6)
    age_sd = _all_same(age_sds, "age_sd", tol=1e-6)
    s = _all_same(cheb_s, "chebyshev_degree")
    race_enc = _all_same(race, "race_encoding")
    # Fourier dicts must match exactly.
    f0 = fourier[ARMS[0]]
    for a in ARMS[1:]:
        if fourier[a] != f0:
            mismatches.append(f"fourier: {a}={fourier[a]!r} vs {f0!r}")

    if abs(float(tau) - EXPECTED_TAU_MAX) > 1e-6:
        mismatches.append(f"tau_max={tau!r} != EXPECTED_TAU_MAX={EXPECTED_TAU_MAX!r}")

    init_hashes = {a: backbone_init_hash(shared, a) for a in ARMS}
    if len(set(init_hashes.values())) != 1:
        mismatches.append(f"backbone_init_hash differs: {init_hashes}")

    # Load selected checkpoints; assert Fourier buffers identical across arms that have them.
    fourier_digests: dict[str, dict[str, str]] = {}
    ckpt_meta: dict[str, dict] = {}
    for arm in ARMS:
        m = build_model(shared, arm)
        meta = load_checkpoint(
            m, Path(selected[arm]["checkpoint"]), arm=arm,
            epoch=selected[arm]["epoch"], device=device)
        ckpt_meta[arm] = meta
        if arm in ("kernel", "random_constant"):
            fourier_digests[arm] = fourier_buffer_digest(m)
        del m

    if "kernel" in fourier_digests and "random_constant" in fourier_digests:
        if fourier_digests["kernel"] != fourier_digests["random_constant"]:
            mismatches.append(
                f"Fourier buffers differ kernel vs random_constant: "
                f"{fourier_digests}")

    # Config hashes for provenance.
    config_hashes = {a: D.config_hash(configs[a]) for a in ARMS}

    paper = {}
    for arm in ARMS:
        pn = run_dirs[arm] / "paper_numbers.json"
        if pn.exists():
            paper[arm] = read_json(pn)

    ok = len(mismatches) == 0
    out = {
        "ok": ok,
        "mismatches": mismatches,
        "seed": seed,
        "tau_max": tau,
        "age_mean": age_mean,
        "age_sd": age_sd,
        "chebyshev_degree_s": s,
        "race_encoding": race_enc,
        "fourier": f0,
        "backbone_init_hash": init_hashes,
        "fourier_buffer_digests": fourier_digests,
        "config_hashes": config_hashes,
        "selected_epoch": {a: selected[a]["epoch"] for a in ARMS},
        "val_loss_train_json": {a: selected[a]["val_loss_train_json"] for a in ARMS},
        "checkpoint_paths": {a: selected[a]["checkpoint"] for a in ARMS},
        "checkpoint_meta": ckpt_meta,
        "config_check": ctx["config_check"],
        "batch_order_hash": ctx["batch_order_hash"],
        "n_examples": ctx["n_examples"],
        "n_batches": ctx["n_batches"],
        "available_seeds": [seed],
        "note_seeds": "Only seed 0 pretraining runs are present under model_new/run/.",
    }
    if not ok:
        D.print_block("T0 provenance FAILED — all later comparisons invalid", mismatches)
    else:
        D.print_block("T0 provenance", [
            f"seed={seed}  tau_max={tau}  age_mean={age_mean:.6f}  age_sd={age_sd:.6f}",
            f"s={s}  race={race_enc}  fourier={f0}",
            f"backbone_init_hash (all arms): {next(iter(init_hashes.values()))}",
            "selected epochs: " + ", ".join(
                f"{a}=ep{selected[a]['epoch']} (val_loss={selected[a]['val_loss_train_json']:.6f})"
                for a in ARMS),
        ])
    return out
