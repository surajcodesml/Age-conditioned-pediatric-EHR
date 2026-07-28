"""T5 — Counterfactual age intervention (primary). Vary only age_years fed to ψ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_new import diagnostics as D
from model_new.audit import INTERVENTION_AGES
from model_new.audit.common import (
    ARMS,
    age_last_of,
    build_model,
    iter_batches,
    load_checkpoint,
    to_device,
)
from model_new.data import tau_from_timestamps
from model_new.eval_pretrain import make_val_loader
from model_new.encoder import build_pair_mask


def _tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Total variation distance, last dim is the simplex. -> [...]"""
    return 0.5 * (p - q).abs().sum(dim=-1)


def _js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a * (a.log() - b.log())).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def _forward_bundle(model, batch: dict) -> dict:
    """Encoder attn (layer 0), pool attn, h, logits — demographics untouched."""
    out = model(batch, need_diagnostics=True)
    # Encoder attention with need_weights.
    tau, _ = tau_from_timestamps(batch["timestamps_days"], batch["attention_mask"],
                                 batch.get("lengths"))
    x = model.embedding_table[batch["code_indices"]]
    blk = model.encoder.blocks[0]
    h_in = blk.ln_attn(x) if blk.ln_attn is not None else x
    _, enc_attn, _ = blk.attn(h_in, tau, batch["attention_mask"], batch["age_years"],
                              need_weights=True)
    # Mean over heads -> [B, L, L]
    enc_attn = enc_attn.mean(dim=1)
    return {
        "enc_attn": enc_attn,
        "pool_attn": out["pool_attn"],
        "h": out["h"],
        "logits": out["code_logits"],
        "mask": batch["attention_mask"],
    }


@torch.no_grad()
def _compare(ref: dict, alt: dict, targets: torch.Tensor) -> dict:
    mask = ref["mask"]
    pair = build_pair_mask(mask)
    # Encoder: valid query rows only (rows with ≥1 valid key beyond the forced diagonal).
    tv_enc = _tv(ref["enc_attn"], alt["enc_attn"])
    js_enc = _js(ref["enc_attn"], alt["enc_attn"])
    tv_enc_valid = tv_enc[mask]
    js_enc_valid = js_enc[mask]

    tv_pool = _tv(ref["pool_attn"], alt["pool_attn"])  # [B]

    dh = alt["h"] - ref["h"]
    rel = dh.norm(dim=-1) / ref["h"].norm(dim=-1).clamp_min(1e-12)
    cos = F.cosine_similarity(ref["h"], alt["h"], dim=-1)

    dlogit = alt["logits"] - ref["logits"]
    # Val loss delta (per-example mean BCE).
    bce_ref = F.binary_cross_entropy_with_logits(ref["logits"], targets, reduction="none").mean(-1)
    bce_alt = F.binary_cross_entropy_with_logits(alt["logits"], targets, reduction="none").mean(-1)

    return {
        "tv_enc_mean": float(tv_enc_valid.mean()) if tv_enc_valid.numel() else float("nan"),
        "js_enc_mean": float(js_enc_valid.mean()) if js_enc_valid.numel() else float("nan"),
        "tv_pool_mean": float(tv_pool.mean()),
        "rel_dh_mean": float(rel.mean()),
        "cos_h_mean": float(cos.mean()),
        "max_abs_dlogit": float(dlogit.abs().max()),
        "sd_dlogit": float(dlogit.std()),
        "delta_bce_mean": float((bce_alt - bce_ref).mean()),
        # Per-example for bootstrap / headline contrasts.
        "_tv_pool": tv_pool.cpu().numpy(),
        "_rel_dh": rel.cpu().numpy(),
    }


def _broadcast_age(batch: dict, age: float) -> dict:
    """Replace age_years only; demographics (incl. standardized age channel) stay true."""
    b = dict(batch)
    mask = batch["attention_mask"].float()
    b["age_years"] = torch.full_like(batch["age_years"], float(age)) * mask
    return b


@torch.no_grad()
def run_t5(ctx: dict, *, n_boot: int = 1000) -> dict:
    shared = ctx["shared"]
    device = ctx["device"]
    selected = ctx["selected"]
    ds = ctx["dataset"]
    max_batches = ctx["max_val_batches"]
    patient_ids = ctx["patient_ids"]
    seed = ctx["seed"]

    ages = list(INTERVENTION_AGES)
    per_arm: dict[str, dict] = {}

    # Controls: verify age_years inertness on a short probe, then record exact zeros.
    # Full intervention metrics are only needed for the kernel arm.
    control_probe_batches = 5 if (max_batches is None or max_batches > 5) else max_batches
    for arm in ("random_constant", "vanilla"):
        m = build_model(shared, arm)
        load_checkpoint(m, Path(selected[arm]["checkpoint"]), arm=arm,
                        epoch=selected[arm]["epoch"], device=device)
        loader = make_val_loader(ds, ctx["batch_size"], 0, shared["race_encoding"])
        control_bugs = []
        n_rows = 0
        for batch in iter_batches(loader, control_probe_batches):
            b = to_device(batch, device)
            targets = b["target_codes"].float()
            ref = _forward_bundle(m, b)
            for age in ages:
                alt = _forward_bundle(m, _broadcast_age(b, age))
                cmp = _compare(ref, alt, targets)
                for key in ("tv_enc_mean", "js_enc_mean", "tv_pool_mean", "rel_dh_mean",
                            "max_abs_dlogit", "sd_dlogit", "delta_bce_mean"):
                    if abs(float(cmp[key])) > 1e-6:
                        control_bugs.append(f"{arm} age={age} {key}={cmp[key]:.3e}")
                if abs(1.0 - float(cmp["cos_h_mean"])) > 1e-6:
                    control_bugs.append(f"{arm} age={age} cos_h={cmp['cos_h_mean']:.8f}")
            n_rows += int(b["lengths"].shape[0])
        if control_bugs:
            D.print_block("T5 BUG — arm gating broken; stop", control_bugs[:40])
            raise AssertionError(
                f"[HARD] T5 controls non-zero ({len(control_bugs)} violations); "
                f"arm gating bug. Examples: {control_bugs[:5]}")
        zero_age = {
            k: 0.0 for k in (
                "tv_enc_mean", "js_enc_mean", "tv_pool_mean", "rel_dh_mean",
                "max_abs_dlogit", "sd_dlogit", "delta_bce_mean")
        }
        zero_age["cos_h_mean"] = 1.0
        per_arm[arm] = {
            "n_examples": n_rows,
            "control_probe_batches": control_probe_batches,
            "by_age": {str(a): dict(zero_age) for a in ages},
        }
        del m

    # Kernel: full intervention pass.
    m = build_model(shared, "kernel")
    load_checkpoint(m, Path(selected["kernel"]["checkpoint"]), arm="kernel",
                    epoch=selected["kernel"]["epoch"], device=device)
    loader = make_val_loader(ds, ctx["batch_size"], ctx["num_workers"],
                             shared["race_encoding"])
    agg = {a: {k: [] for k in (
        "tv_enc_mean", "js_enc_mean", "tv_pool_mean", "rel_dh_mean", "cos_h_mean",
        "max_abs_dlogit", "sd_dlogit", "delta_bce_mean")} for a in ages}
    per_ex_pool = {a: [] for a in ages}
    per_ex_rel = {a: [] for a in ages}
    n_rows = 0
    for batch in iter_batches(loader, max_batches):
        b = to_device(batch, device)
        targets = b["target_codes"].float()
        ref = _forward_bundle(m, b)
        for age in ages:
            alt = _forward_bundle(m, _broadcast_age(b, age))
            cmp = _compare(ref, alt, targets)
            for k in agg[age]:
                agg[age][k].append(cmp[k])
            per_ex_pool[age].append(cmp["_tv_pool"])
            per_ex_rel[age].append(cmp["_rel_dh"])
        n_rows += int(b["lengths"].shape[0])
    arm_out = {"n_examples": n_rows, "by_age": {}}
    for age in ages:
        means = {k: float(np.mean(v)) if v else float("nan") for k, v in agg[age].items()}
        means["tv_pool_per_example"] = (
            np.concatenate(per_ex_pool[age]) if per_ex_pool[age] else np.zeros(0))
        means["rel_dh_per_example"] = (
            np.concatenate(per_ex_rel[age]) if per_ex_rel[age] else np.zeros(0))
        arm_out["by_age"][str(age)] = means
    per_arm["kernel"] = arm_out
    del m

    # Headline: 25 vs 75 on pooling TV and ‖Δh‖/‖h‖.
    # These are each measured vs true-age; the contrast is |m(25)-m(75)| style difference
    # of intervention effects, and also the direct 25-vs-75 by comparing the two alts.
    # Spec: "Headline number: the 25-vs-75 contrast on pooling TV and ‖Δh‖/‖h‖."
    # Interpret as metrics when broadcasting 25 vs when broadcasting 75 (each vs true),
    # plus a direct 25-vs-75 forward contrast on a second pass for kernel only.
    k25 = per_arm["kernel"]["by_age"]["25.0"]
    k75 = per_arm["kernel"]["by_age"]["75.0"]

    def _mean_arr(arr):
        def stat(rows):
            return float(arr[rows].mean()) if rows.size else float("nan")
        return stat

    # Truncate patient_ids to evaluated rows.
    n = per_arm["kernel"]["n_examples"]
    pids = patient_ids[:n]

    headline = {
        "pool_tv_at_25": float(k25["tv_pool_mean"]),
        "pool_tv_at_75": float(k75["tv_pool_mean"]),
        "rel_dh_at_25": float(k25["rel_dh_mean"]),
        "rel_dh_at_75": float(k75["rel_dh_mean"]),
        "pool_tv_25_minus_75": float(k25["tv_pool_mean"] - k75["tv_pool_mean"]),
        "rel_dh_25_minus_75": float(k25["rel_dh_mean"] - k75["rel_dh_mean"]),
        "pool_tv_25_ci": D.bootstrap_ci(_mean_arr(k25["tv_pool_per_example"]), pids,
                                        n_boot=n_boot, seed=seed),
        "pool_tv_75_ci": D.bootstrap_ci(_mean_arr(k75["tv_pool_per_example"]), pids,
                                        n_boot=n_boot, seed=seed),
        "rel_dh_25_ci": D.bootstrap_ci(_mean_arr(k25["rel_dh_per_example"]), pids,
                                       n_boot=n_boot, seed=seed),
        "rel_dh_75_ci": D.bootstrap_ci(_mean_arr(k75["rel_dh_per_example"]), pids,
                                       n_boot=n_boot, seed=seed),
    }

    # Direct 25-vs-75 contrast (one more pass, kernel only).
    m = build_model(shared, "kernel")
    load_checkpoint(m, Path(selected["kernel"]["checkpoint"]), arm="kernel",
                    epoch=selected["kernel"]["epoch"], device=device)
    loader = make_val_loader(ds, ctx["batch_size"], ctx["num_workers"],
                             shared["race_encoding"])
    direct_pool = []
    direct_rel = []
    for batch in iter_batches(loader, max_batches):
        b = to_device(batch, device)
        a25 = _forward_bundle(m, _broadcast_age(b, 25.0))
        a75 = _forward_bundle(m, _broadcast_age(b, 75.0))
        cmp = _compare(a25, a75, b["target_codes"].float())
        direct_pool.append(cmp["_tv_pool"])
        direct_rel.append(cmp["_rel_dh"])
    del m
    d_pool = np.concatenate(direct_pool) if direct_pool else np.zeros(0)
    d_rel = np.concatenate(direct_rel) if direct_rel else np.zeros(0)
    headline["direct_25_vs_75_pool_tv"] = float(d_pool.mean()) if d_pool.size else float("nan")
    headline["direct_25_vs_75_rel_dh"] = float(d_rel.mean()) if d_rel.size else float("nan")
    headline["direct_25_vs_75_pool_tv_ci"] = D.bootstrap_ci(
        _mean_arr(d_pool), pids, n_boot=n_boot, seed=seed)
    headline["direct_25_vs_75_rel_dh_ci"] = D.bootstrap_ci(
        _mean_arr(d_rel), pids, n_boot=n_boot, seed=seed)

    # Strip per-example arrays from JSON-facing copy.
    per_arm_json = {}
    for arm, blob in per_arm.items():
        per_arm_json[arm] = {"n_examples": blob["n_examples"], "by_age": {}}
        for age, metrics in blob["by_age"].items():
            per_arm_json[arm]["by_age"][age] = {
                k: v for k, v in metrics.items()
                if not k.endswith("_per_example")
            }

    nonzero = any(
        abs(per_arm["kernel"]["by_age"][str(a)]["tv_pool_mean"]) > 1e-8
        or abs(per_arm["kernel"]["by_age"][str(a)]["rel_dh_mean"]) > 1e-8
        for a in ages
    )
    out = {
        "intervention_ages": ages,
        "controls_exactly_zero": True,
        "kernel_effect_nonzero": nonzero,
        "per_arm": per_arm_json,
        "headline_25_vs_75": headline,
        "isolates": "R2 only: age_years → ψ; demographics (R1) held at true values",
    }
    D.print_block("T5 counterfactual age intervention", [
        f"controls (vanilla, random_constant) exactly 0: True",
        f"kernel effect nonzero: {nonzero}",
        f"headline direct 25-vs-75 pool TV = {headline['direct_25_vs_75_pool_tv']:.6e} "
        f"CI[{headline['direct_25_vs_75_pool_tv_ci']['lo']:.6e},"
        f"{headline['direct_25_vs_75_pool_tv_ci']['hi']:.6e}]",
        f"headline direct 25-vs-75 ‖Δh‖/‖h‖ = {headline['direct_25_vs_75_rel_dh']:.6e} "
        f"CI[{headline['direct_25_vs_75_rel_dh_ci']['lo']:.6e},"
        f"{headline['direct_25_vs_75_rel_dh_ci']['hi']:.6e}]",
        *[f"kernel@{a}: pool_TV={per_arm['kernel']['by_age'][str(a)]['tv_pool_mean']:.4e} "
          f"‖Δh‖/‖h‖={per_arm['kernel']['by_age'][str(a)]['rel_dh_mean']:.4e} "
          f"Δbce={per_arm['kernel']['by_age'][str(a)]['delta_bce_mean']:.4e}"
          for a in ages],
    ])
    return out
