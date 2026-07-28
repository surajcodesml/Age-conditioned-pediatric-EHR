"""T2 — Is the coefficient generator alive? Norms and ‖W2‖ trajectory."""

from __future__ import annotations

from pathlib import Path

import torch

from model_new import diagnostics as D
from model_new.audit.common import (
    ARMS,
    build_model,
    generator_final_weight,
    generator_first_weight,
    list_epochs,
    load_checkpoint,
    checkpoint_path,
)


def _site_norms(site) -> dict:
    w1 = generator_first_weight(site)
    w2 = generator_final_weight(site)
    alpha = site.alpha_base.detach()
    # Δα at a probe age — for norms of the generator output path we report param norms.
    return {
        "W1_fro": float(w1.detach().norm()) if w1 is not None else 0.0,
        "W2_fro": float(w2.detach().norm()) if w2 is not None else 0.0,
        "W2_abs_max": float(w2.detach().abs().max()) if w2 is not None else 0.0,
        "alpha_base_l2": float(alpha.norm()),
        "alpha_base_l1": float(alpha.abs().sum()),
        "has_generator": w2 is not None,
    }


@torch.no_grad()
def run_t2(ctx: dict) -> dict:
    shared = ctx["shared"]
    device = ctx["device"]
    selected = ctx["selected"]
    run_dirs = {a: Path(p) for a, p in ctx["run_dirs"].items()}

    selected_norms: dict[str, dict] = {}
    for arm in ARMS:
        m = build_model(shared, arm)
        load_checkpoint(m, Path(selected[arm]["checkpoint"]), arm=arm,
                        epoch=selected[arm]["epoch"], device=device)
        sites = {}
        for name, site in m.kernel_sites():
            # Δα mean norm over a fixed age probe for reporting ‖Δα‖ at the selected ckpt.
            ages = torch.tensor([25.0, 50.0, 75.0], device=device)
            delta = site.age(ages)
            entry = _site_norms(site)
            entry["delta_alpha_l2_mean_probe"] = float(delta.norm(dim=-1).mean())
            entry["delta_alpha_l2_probe"] = delta.norm(dim=-1).cpu().tolist()
            sites[name] = entry
        selected_norms[arm] = sites
        del m

    # ‖W2‖ trajectory across all saved epochs for arms that have a generator.
    trajectories: dict[str, dict] = {}
    escape: dict[str, dict] = {}
    for arm in ("kernel", "random_constant"):
        epochs = list_epochs(run_dirs[arm])
        traj: dict[str, list] = {}
        m = build_model(shared, arm)
        for ep in epochs:
            load_checkpoint(m, checkpoint_path(run_dirs[arm], ep), arm=arm,
                            epoch=ep, device=device)
            for name, site in m.kernel_sites():
                w2 = generator_final_weight(site)
                traj.setdefault(name, []).append({
                    "epoch": ep,
                    "W2_fro": float(w2.detach().norm()) if w2 is not None else 0.0,
                    "W2_abs_max": float(w2.detach().abs().max()) if w2 is not None else 0.0,
                })
        trajectories[arm] = traj
        # Escape: first epoch where ‖W2‖ > 1e-5 (above float noise / zero-init).
        esc = {}
        for name, series in traj.items():
            stepped = next((s["epoch"] for s in series if s["W2_fro"] > 1e-5), None)
            final = series[-1]["W2_fro"] if series else 0.0
            esc[name] = {
                "escaped": stepped is not None,
                "first_escape_epoch": stepped,
                "final_W2_fro": final,
                "stuck_near_zero": bool(final < 1e-5),
                "finding_if_stuck": (
                    "‖W2‖ ends at ~1e-6 — the zero-init saddle never broke."
                    if final < 1e-5 else None
                ),
            }
        escape[arm] = esc
        del m

    out = {
        "selected_epoch_norms": selected_norms,
        "W2_trajectory": trajectories,
        "saddle_escape": escape,
        "note": (
            "∂L/∂W1 ∝ W2ᵀ at init: zero-init W2 is a genuine saddle. "
            "If final ‖W2‖ ~ 1e-6 the generator stayed dead."
        ),
    }
    lines = []
    for arm in ARMS:
        for site, v in selected_norms[arm].items():
            lines.append(
                f"{arm}/{site}: ‖W1‖={v['W1_fro']:.4e} ‖W2‖={v['W2_fro']:.4e} "
                f"‖α_base‖₂={v['alpha_base_l2']:.4e} "
                f"‖Δα‖_probe={v['delta_alpha_l2_mean_probe']:.4e}")
    for arm, esc in escape.items():
        for site, v in esc.items():
            lines.append(
                f"escape {arm}/{site}: escaped={v['escaped']} "
                f"at_epoch={v['first_escape_epoch']} final‖W2‖={v['final_W2_fro']:.4e}")
    D.print_block("T2 generator alive?", lines)
    return out
