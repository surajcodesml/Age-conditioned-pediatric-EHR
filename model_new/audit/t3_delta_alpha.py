"""T3 — Does Δα vary with age on the empirical event-age support?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model_new import diagnostics as D
from model_new.audit import SUPPORT_AGE_MIN
from model_new.audit.common import ARMS, build_model, load_checkpoint, iter_batches, to_device
from model_new.data import corpus_stats_cached
from model_new.eval_pretrain import make_val_loader


@torch.no_grad()
def _sample_empirical_event_ages(ds, max_batches: int | None, batch_size: int,
                                 num_workers: int, race_encoding: str, seed: int,
                                 n_max: int = 200_000) -> np.ndarray:
    """Event ages from real validation windows (masked positions), not a uniform grid."""
    rng = np.random.default_rng(seed)
    loader = make_val_loader(ds, batch_size, num_workers, race_encoding)
    ages: list[np.ndarray] = []
    n = 0
    for batch in iter_batches(loader, max_batches):
        mask = batch["attention_mask"].numpy()
        a = batch["age_years"].numpy()
        flat = a[mask]
        ages.append(flat.astype(np.float64))
        n += flat.size
        if n >= n_max:
            break
    arr = np.concatenate(ages) if ages else np.zeros(0, np.float64)
    if arr.size > n_max:
        arr = rng.choice(arr, size=n_max, replace=False)
    return arr


@torch.no_grad()
def _decompose(delta: torch.Tensor, alpha_base: torch.Tensor) -> dict:
    """Δα(a) = Δᾱ + Δα̃(a); varying_frac / relative_scale as specified."""
    # delta: [N, s]
    mean = delta.mean(dim=0, keepdim=True)
    resid = delta - mean
    norms = delta.norm(dim=-1)
    resid_norms = resid.norm(dim=-1)
    e_full = float(norms.mean())
    e_var = float(resid_norms.mean())
    ab = float(alpha_base.detach().norm())
    return {
        "n": int(delta.shape[0]),
        "E_norm_delta": e_full,
        "E_norm_varying": e_var,
        "norm_mean_delta": float(mean.squeeze(0).norm()),
        "alpha_base_l2": ab,
        "varying_frac": (e_var / e_full) if e_full > 0 else 0.0,
        "relative_scale": (e_var / ab) if ab > 0 else float("nan"),
    }


@torch.no_grad()
def run_t3(ctx: dict) -> dict:
    shared = ctx["shared"]
    device = ctx["device"]
    selected = ctx["selected"]
    ds = ctx["dataset"]
    seed = ctx["seed"]

    ages = _sample_empirical_event_ages(
        ds, ctx["max_val_batches"], ctx["batch_size"], ctx["num_workers"],
        shared["race_encoding"], seed)
    if ages.size == 0:
        raise AssertionError("[HARD] no empirical event ages sampled for T3")

    support = ages[ages >= SUPPORT_AGE_MIN]
    extrap = ages[ages < SUPPORT_AGE_MIN]
    # Always also evaluate a grid of extrapolation ages even if val has none.
    if extrap.size == 0:
        extrap = np.array([1.0, 5.0, 10.0, 15.0], dtype=np.float64)

    ages_t = torch.as_tensor(ages, dtype=torch.float32, device=device)
    support_t = torch.as_tensor(support, dtype=torch.float32, device=device)
    extrap_t = torch.as_tensor(extrap, dtype=torch.float32, device=device)

    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        m = build_model(shared, arm)
        load_checkpoint(m, Path(selected[arm]["checkpoint"]), arm=arm,
                        epoch=selected[arm]["epoch"], device=device)
        sites = {}
        for name, site in m.kernel_sites():
            entry = {
                "mode": site.age.mode,
                "all_empirical": _decompose(site.age(ages_t), site.alpha_base),
                "support": _decompose(site.age(support_t), site.alpha_base)
                if support_t.numel() else None,
                "extrapolation": _decompose(site.age(extrap_t), site.alpha_base),
            }
            sites[name] = entry
        per_arm[arm] = sites
        del m

    # Assert random_constant varying_frac == 0 (up to float noise).
    bugs = []
    for name, site in per_arm["random_constant"].items():
        vf = site["all_empirical"]["varying_frac"]
        if vf > 1e-6:
            bugs.append(
                f"[BUG] random_constant/{name} varying_frac={vf:.3e} != 0 "
                f"(generator must ignore age)")
    if bugs:
        D.print_block("T3 BUG — stop", bugs)
        raise AssertionError("; ".join(bugs))

    # Inert thresholds on support for kernel.
    inert_flags = {}
    for name, site in per_arm["kernel"].items():
        s = site["support"] or site["all_empirical"]
        inert = bool(s["varying_frac"] < 0.1 or (
            np.isfinite(s["relative_scale"]) and s["relative_scale"] < 0.05))
        inert_flags[name] = {
            "inert": inert,
            "varying_frac": s["varying_frac"],
            "relative_scale": s["relative_scale"],
            "rule": "inert if varying_frac < 0.1 or relative_scale < 0.05 on support",
        }

    out = {
        "support_age_min": SUPPORT_AGE_MIN,
        "n_empirical_ages": int(ages.size),
        "n_support": int(support.size),
        "n_extrapolation_eval": int(extrap.size),
        "age_min": float(ages.min()),
        "age_max": float(ages.max()),
        "age_median": float(np.median(ages)),
        "per_arm": per_arm,
        "kernel_inert_on_support": inert_flags,
        "random_constant_varying_frac_asserted_zero": True,
    }
    lines = [
        f"empirical ages n={ages.size} median={float(np.median(ages)):.2f} "
        f"support(n>={SUPPORT_AGE_MIN})={support.size}",
    ]
    for arm in ("kernel", "random_constant", "vanilla"):
        for name, site in per_arm[arm].items():
            s = site["support"] or site["all_empirical"]
            lines.append(
                f"{arm}/{name}: varying_frac={s['varying_frac']:.4f} "
                f"relative_scale={s['relative_scale']:.4f} "
                f"E‖Δα‖={s['E_norm_delta']:.4e}")
    for name, f in inert_flags.items():
        lines.append(f"kernel/{name} inert_on_support={f['inert']}")
    D.print_block("T3 Δα(a) age variation", lines)
    return out
