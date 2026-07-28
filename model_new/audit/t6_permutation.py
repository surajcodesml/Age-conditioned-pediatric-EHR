"""T6 — Permutation test of age trajectories (primary) + graded age shifts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_new import diagnostics as D
from model_new.audit import AGE_SHIFTS
from model_new.audit.common import (
    age_last_of,
    build_model,
    iter_batches,
    load_checkpoint,
    to_device,
)
from model_new.eval_pretrain import make_val_loader


@torch.no_grad()
def _metrics_on_loader(model, loader, device, max_batches) -> dict:
    bce_all: list[float] = []
    r10_all: list[float] = []
    for batch in iter_batches(loader, max_batches):
        b = to_device(batch, device)
        logits = model(b)["code_logits"].float()
        targets = b["target_codes"].float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(-1)
        bce_all.extend(bce.cpu().tolist())
        r = D.topk_per_example(logits, targets, ks=(10,))["recall@10"]
        r10_all.extend(r.tolist())
    return {
        "bce": float(np.mean(bce_all)) if bce_all else float("nan"),
        "recall@10": float(np.mean(r10_all)) if r10_all else float("nan"),
        "n": len(bce_all),
    }


def _collect_age_trajectories(ds, max_batches, batch_size, num_workers, race_encoding):
    """List of per-row age_years tensors (CPU) and the batches' structural shells."""
    loader = make_val_loader(ds, batch_size, num_workers, race_encoding)
    ages: list[torch.Tensor] = []
    for batch in iter_batches(loader, max_batches):
        ages.append(batch["age_years"].clone())
    return ages


def _apply_ages_to_batches(ds, max_batches, batch_size, num_workers, race_encoding,
                           age_list: list[torch.Tensor]):
    """Yield batches with age_years replaced (demographics untouched)."""
    loader = make_val_loader(ds, batch_size, num_workers, race_encoding)
    for i, batch in enumerate(iter_batches(loader, max_batches)):
        b = dict(batch)
        b["age_years"] = age_list[i]
        yield b


def _permute_ages(age_list: list[torch.Tensor], rng: np.random.Generator) -> list[torch.Tensor]:
    """Shuffle whole per-patient age trajectories across patients within each batch.

    Within-batch permutation preserves lag structure (timestamps untouched) and content.
    Cross-batch patient identity is not required for a valid null: each batch's rows are
    exchangeable under the null that age_years is irrelevant to the kernel.
    """
    out = []
    for ages in age_list:
        bsz = ages.shape[0]
        perm = rng.permutation(bsz)
        out.append(ages[perm].clone())
    return out


def _shift_ages(age_list: list[torch.Tensor], delta: float) -> list[torch.Tensor]:
    out = []
    for ages in age_list:
        # Preserve zeros on padding: ages are already zeroed on pad by collate.
        # Shift only where originally nonzero (valid events). Mask ≈ ages != 0 is wrong for
        # true age 0; use a companion — we don't have mask here. Re-derive: padding is 0
        # and real MIMIC ages are >> 0, so ages > 0 is safe on this corpus.
        shifted = ages.clone()
        valid = ages > 0
        shifted[valid] = (ages[valid] + float(delta)).clamp(min=0.0)
        out.append(shifted)
    return out


@torch.no_grad()
def _arm_age_years_inert(model, batch: dict, device) -> bool:
    """True iff changing age_years leaves logits unchanged (vanilla / random_constant)."""
    b = to_device(batch, device)
    ref = model(b)["code_logits"]
    alt = dict(b)
    alt["age_years"] = torch.full_like(b["age_years"], 42.0) * b["attention_mask"].float()
    got = model(alt)["code_logits"]
    return bool(torch.allclose(ref, got, atol=0.0, rtol=0.0))


@torch.no_grad()
def run_t6(ctx: dict, *, n_perm: int = 100) -> dict:
    shared = ctx["shared"]
    device = ctx["device"]
    selected = ctx["selected"]
    ds = ctx["dataset"]
    max_batches = ctx["max_val_batches"]
    seed = ctx["seed"]
    rng = np.random.default_rng(seed)

    age_list = _collect_age_trajectories(
        ds, max_batches, ctx["batch_size"], ctx["num_workers"], shared["race_encoding"])

    results: dict[str, dict] = {}
    for arm in ("kernel", "random_constant", "vanilla"):
        m = build_model(shared, arm)
        load_checkpoint(m, Path(selected[arm]["checkpoint"]), arm=arm,
                        epoch=selected[arm]["epoch"], device=device)

        # Short-circuit controls: if age_years is inert, p = 1 by construction.
        probe_loader = make_val_loader(ds, ctx["batch_size"], 0, shared["race_encoding"])
        probe = next(iter(probe_loader))
        inert = _arm_age_years_inert(m, probe, device)

        true_metrics = _metrics_on_loader(
            m,
            _apply_ages_to_batches(ds, max_batches, ctx["batch_size"], ctx["num_workers"],
                                   shared["race_encoding"], age_list),
            device, None)  # age_list already truncated

        if inert and arm in ("vanilla", "random_constant"):
            results[arm] = {
                "age_years_inert": True,
                "true": true_metrics,
                "n_perm": n_perm,
                "p_bce": 1.0,
                "p_recall@10": 1.0,
                "null_bce": [true_metrics["bce"]] * n_perm,
                "null_recall@10": [true_metrics["recall@10"]] * n_perm,
                "note": "age_years does not enter the forward; permutation leaves metrics "
                        "unchanged → p = 1 by construction (verified on one batch).",
                "shifts": {str(d): true_metrics for d in AGE_SHIFTS},
            }
            del m
            continue

        null_bce = []
        null_r10 = []
        for _ in range(n_perm):
            perm_ages = _permute_ages(age_list, rng)
            met = _metrics_on_loader(
                m,
                _apply_ages_to_batches(ds, max_batches, ctx["batch_size"],
                                       ctx["num_workers"], shared["race_encoding"],
                                       perm_ages),
                device, None)
            null_bce.append(met["bce"])
            null_r10.append(met["recall@10"])

        # p = fraction of permutations at least as good as true ages.
        # Lower BCE is better; higher recall is better.
        p_bce = float(np.mean(np.asarray(null_bce) <= true_metrics["bce"]))
        p_r10 = float(np.mean(np.asarray(null_r10) >= true_metrics["recall@10"]))

        shifts = {}
        for delta in AGE_SHIFTS:
            shifted = _shift_ages(age_list, delta)
            shifts[str(delta)] = _metrics_on_loader(
                m,
                _apply_ages_to_batches(ds, max_batches, ctx["batch_size"],
                                       ctx["num_workers"], shared["race_encoding"],
                                       shifted),
                device, None)

        results[arm] = {
            "age_years_inert": False,
            "true": true_metrics,
            "n_perm": n_perm,
            "p_bce": p_bce,
            "p_recall@10": p_r10,
            "null_bce": null_bce,
            "null_recall@10": null_r10,
            "shifts": shifts,
        }
        del m

    k = results["kernel"]
    active = bool(k["p_bce"] < 0.05 or k["p_recall@10"] < 0.05)
    out = {
        "n_perm": n_perm,
        "n_examples": results["kernel"]["true"]["n"],
        "per_arm": results,
        "kernel_age_dependent": active,
        "headline": (
            f"kernel permutation p_bce={k['p_bce']:.3f} p_r@10={k['p_recall@10']:.3f} "
            f"— {'ACTIVE (p<0.05)' if active else 'NULL (p≥0.05)'}"
        ),
    }
    D.print_block("T6 permutation test", [
        out["headline"],
        *[f"{arm}: true bce={results[arm]['true']['bce']:.6f} "
          f"r@10={results[arm]['true']['recall@10']:.4f} "
          f"p_bce={results[arm]['p_bce']:.3f} p_r@10={results[arm]['p_recall@10']:.3f} "
          f"inert={results[arm]['age_years_inert']}"
          for arm in results],
        "graded shifts (kernel): " + ", ".join(
            f"Δ={d}: bce={k['shifts'][str(d)]['bce']:.6f} "
            f"r@10={k['shifts'][str(d)]['recall@10']:.4f}"
            for d in AGE_SHIFTS),
    ])
    return out
