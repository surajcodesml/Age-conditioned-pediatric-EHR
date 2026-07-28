"""T7 — Localization of a null result (run when T3–T6 are null)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_new import diagnostics as D
from model_new.age_encoding import LogAgeFourier
from model_new.audit import SUPPORT_AGE_MIN
from model_new.audit.common import (
    REPO_ROOT,
    build_model,
    iter_batches,
    load_checkpoint,
    read_json,
    to_device,
)
from model_new.basis import chebyshev_basis
from model_new.data import (
    TensorizedPretrainDataset,
    pairwise_tau,
    sample_empirical_taus,
    tau_from_timestamps,
)
from model_new.eval_pretrain import make_val_loader
from model_new.encoder import build_pair_mask


@torch.no_grad()
def _fourier_band(shared: dict, ages: np.ndarray) -> dict:
    """T7(a): Gram of ψ(a), effective rank, mean cosine by age gap, cycles/period."""
    f = LogAgeFourier(M=shared["age_M"], p_min=shared["age_p_min"], p_max=shared["age_p_max"])
    a = torch.as_tensor(ages, dtype=torch.float32)
    psi = f(a)  # [N, 2M]
    # Gram over empirical ages.
    g = (psi.T @ psi) / max(1, psi.shape[0])
    eig = torch.linalg.eigvalsh(g.double()).clamp_min(0).cpu().numpy()
    eig = eig[::-1]
    total = eig.sum()
    # Effective rank: exp(entropy of normalized eigenvalues).
    p = eig / total if total > 0 else eig
    p = p[p > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0

    # Mean cosine for |a-a'| in {1,5,10,20}.
    cos_by_gap = {}
    a_np = ages.astype(np.float64)
    psi_np = psi.numpy().astype(np.float64)
    norms = np.linalg.norm(psi_np, axis=1, keepdims=True).clip(min=1e-12)
    psi_n = psi_np / norms
    rng = np.random.default_rng(0)
    # Subsample pairs for speed.
    n = min(len(a_np), 4000)
    idx = rng.choice(len(a_np), size=n, replace=False) if len(a_np) > n else np.arange(len(a_np))
    aa = a_np[idx]
    pp = psi_n[idx]
    for gap in (1.0, 5.0, 10.0, 20.0):
        cosines = []
        for i in range(len(aa)):
            # nearest j with |a_j - a_i| ≈ gap ± 0.5
            d = np.abs(aa - aa[i])
            sel = np.where((d >= gap - 0.5) & (d <= gap + 0.5) & (d > 0))[0]
            if sel.size:
                j = sel[int(np.argmin(np.abs(d[sel] - gap)))]
                cosines.append(float(pp[i] @ pp[j]))
        cos_by_gap[str(gap)] = float(np.mean(cosines)) if cosines else float("nan")

    # Cycles over corpus u-range.
    u = np.log1p(np.clip(a_np, 0, None))
    u_span = float(u.max() - u.min()) if u.size else 0.0
    periods = f.periods.cpu().numpy()
    components = []
    usable = 0
    for m, p_m in enumerate(periods):
        cycles = u_span / float(p_m) if p_m > 0 else float("inf")
        if cycles < 0.25:
            band = "near_constant"
        elif cycles > 3.0:
            band = "hash_like"
        else:
            band = "usable"
            usable += 1
        components.append({
            "m": m, "period": float(p_m), "cycles_over_u_span": cycles, "band": band,
        })
    return {
        "n_ages": int(len(ages)),
        "u_span": u_span,
        "u_min": float(u.min()) if u.size else float("nan"),
        "u_max": float(u.max()) if u.size else float("nan"),
        "gram_eff_rank": eff_rank,
        "gram_cond": float(np.linalg.cond(g.numpy())) if g.numel() else float("nan"),
        "mean_cosine_by_gap": cos_by_gap,
        "components": components,
        "n_usable_of_M": usable,
        "M": int(shared["age_M"]),
        "note": (
            f"With a∈[{SUPPORT_AGE_MIN},90], u_span≈log1p(90)-log1p({SUPPORT_AGE_MIN})≈"
            f"{np.log1p(90)-np.log1p(SUPPORT_AGE_MIN):.3f}; "
            f"p_min=0.15 → ~{u_span/0.15:.1f} cycles, p_max=6 → ~{u_span/6:.2f} cycles."
        ),
    }


@torch.no_grad()
def _row_basis_conditioning(ds, shared, max_batches, batch_size, seed) -> dict:
    """T7(c): Gram cond of Chebyshev basis on τ present in each real attention row."""
    loader = make_val_loader(ds, batch_size, 0, shared["race_encoding"])
    tau_max = float(shared["tau_max"])
    s = int(shared["s"])
    conds = []
    rng = np.random.default_rng(seed)
    n_rows_seen = 0
    for batch in iter_batches(loader, max_batches):
        tau, _ = tau_from_timestamps(batch["timestamps_days"], batch["attention_mask"],
                                     batch["lengths"])
        mask = batch["attention_mask"]
        pair = build_pair_mask(mask)
        B, L, _ = tau.shape
        for b in range(B):
            for i in range(int(batch["lengths"][b])):
                # τ values this query row attends over (valid keys).
                row_mask = pair[b, i]
                t = tau[b, i, row_mask].numpy()
                if t.size < s + 1:
                    continue
                # Subsample long rows.
                if t.size > 512:
                    t = rng.choice(t, 512, replace=False)
                x = np.clip(2.0 * t / tau_max - 1.0, -1.0, 1.0)
                basis = chebyshev_basis(torch.from_numpy(x.astype(np.float64)), s).numpy()
                g = basis.T @ basis / max(1, basis.shape[0])
                conds.append(float(np.linalg.cond(g)))
                n_rows_seen += 1
        if n_rows_seen >= 5000:
            break
    arr = np.asarray(conds, dtype=np.float64)
    if arr.size == 0:
        return {"n_rows": 0}
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "n_rows": int(arr.size),
        "median": float(med),
        "iqr": [float(q25), float(q75)],
        "frac_cond_gt_1e3": float(np.mean(arr > 1e3)),
        "corpus_chebyshev_no_constant_ref": 15.1,
    }


@torch.no_grad()
def _r1_probe_and_ablation(ctx, shared, selected) -> dict:
    """T7(e): linear probe h→age R²; ablate demographic age channel Δ val loss."""
    device = ctx["device"]
    ds = ctx["dataset"]
    max_batches = ctx["max_val_batches"]

    def collect(arm: str, zero_demo_age: bool = False):
        m = build_model(shared, arm)
        load_checkpoint(m, Path(selected[arm]["checkpoint"]), arm=arm,
                        epoch=selected[arm]["epoch"], device=device)
        loader = make_val_loader(ds, ctx["batch_size"], ctx["num_workers"],
                                 shared["race_encoding"])
        hs, ages, bces = [], [], []
        for batch in iter_batches(loader, max_batches):
            b = to_device(batch, device)
            if zero_demo_age:
                demo = b["demographics"].clone()
                demo[..., 0] = 0.0
                b = dict(b)
                b["demographics"] = demo
            out = m(b)
            hs.append(out["h"].cpu())
            ages.append(out["age_last"].cpu())
            logits = out["code_logits"].float()
            targets = b["target_codes"].float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(-1)
            bces.append(bce.cpu())
        del m
        H = torch.cat(hs)
        A = torch.cat(ages)
        BCE = torch.cat(bces)
        return H, A, float(BCE.mean())

    H, A, bce_true = collect("kernel", False)
    # Ridge linear probe.
    X = torch.cat([H, torch.ones(H.shape[0], 1)], dim=1).double()
    y = A.double().unsqueeze(1)
    # (X'X + λI)^{-1} X'y
    lam = 1e-3
    xtx = X.T @ X + lam * torch.eye(X.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(xtx, X.T @ y)
    pred = (X @ w).squeeze(1)
    ss_res = float(((y.squeeze(1) - pred) ** 2).sum())
    ss_tot = float(((y.squeeze(1) - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    _, _, bce_ablate_k = collect("kernel", True)
    _, _, bce_true_v = collect("vanilla", False)
    _, _, bce_ablate_v = collect("vanilla", True)

    return {
        "linear_probe_R2_h_to_age": r2,
        "kernel_bce_true": bce_true,
        "kernel_bce_demo_age_zeroed": bce_ablate_k,
        "kernel_delta_bce": bce_ablate_k - bce_true,
        "vanilla_bce_true": bce_true_v,
        "vanilla_bce_demo_age_zeroed": bce_ablate_v,
        "vanilla_delta_bce": bce_ablate_v - bce_true_v,
        "r1_saturates_age": bool(r2 > 0.5),
    }


def run_t7(ctx: dict, t2: dict, t3: dict, t4: dict) -> dict:
    shared = ctx["shared"]
    selected = ctx["selected"]
    ds = ctx["dataset"]
    seed = ctx["seed"]
    run_dirs = {a: Path(p) for a, p in ctx["run_dirs"].items()}

    # Empirical ages from T3 sampling support.
    # Reuse val event ages quickly.
    from model_new.audit.t3_delta_alpha import _sample_empirical_event_ages
    ages = _sample_empirical_event_ages(
        ds, ctx["max_val_batches"], ctx["batch_size"], 0,
        shared["race_encoding"], seed, n_max=50_000)
    support_ages = ages[ages >= SUPPORT_AGE_MIN]
    if support_ages.size < 100:
        support_ages = ages

    a = _fourier_band(shared, support_ages)

    b = {
        "from_t2": t2.get("saddle_escape", {}),
        "summary": {
            arm: {
                site: {
                    "escaped": v["escaped"],
                    "first_escape_epoch": v["first_escape_epoch"],
                    "final_W2_fro": v["final_W2_fro"],
                }
                for site, v in sites.items()
            }
            for arm, sites in t2.get("saddle_escape", {}).items()
        },
    }

    c = _row_basis_conditioning(ds, shared, ctx["max_val_batches"], ctx["batch_size"], seed)

    # T7(d) optimizer drift from train.json (no extra forwards).
    d = {}
    for arm in ("kernel", "random_constant", "vanilla"):
        train = read_json(run_dirs[arm] / "train.json")
        ep = selected[arm]["epoch"]
        rec = next(x for x in train if int(x["epoch"]) == ep)
        d[arm] = rec.get("param_drift", {})

    e = _r1_probe_and_ablation(ctx, shared, selected)

    # T7(f) content dominance from T4.
    f = {
        "kernel_R": t4.get("kernel_max_R"),
        "sigma_content": {
            site: t4["per_arm"]["kernel"][site]["sigma_content"]
            for site in t4.get("per_arm", {}).get("kernel", {})
        },
        "alpha_l1": {
            site: t3["per_arm"]["kernel"][site]["all_empirical"]["alpha_base_l2"]
            for site in t3.get("per_arm", {}).get("kernel", {})
        },
    }

    # Rank causes by how strongly each explains a null.
    causes = []
    # (a) band
    n_usable = a["n_usable_of_M"]
    causes.append({
        "id": "a_fourier_band",
        "strength": 1.0 - n_usable / max(1, a["M"]),
        "number": f"{n_usable}/{a['M']} usable; u_span={a['u_span']:.3f}",
        "explains_null_if": "few usable frequencies over corpus u-range",
    })
    # (b) saddle
    esc = t2.get("saddle_escape", {}).get("kernel", {})
    stuck = all(v.get("stuck_near_zero") for v in esc.values()) if esc else False
    causes.append({
        "id": "b_saddle_escape",
        "strength": 1.0 if stuck else 0.0,
        "number": f"stuck={stuck}; " + ", ".join(
            f"{s}:W2={v['final_W2_fro']:.2e}" for s, v in esc.items()),
        "explains_null_if": "‖W2‖ never left zero",
    })
    # (c) row cond
    frac_bad = c.get("frac_cond_gt_1e3", 0.0)
    causes.append({
        "id": "c_row_basis_cond",
        "strength": float(frac_bad),
        "number": f"median_cond={c.get('median')} frac>1e3={frac_bad:.3f}",
        "explains_null_if": "within-row τ variation is ill-conditioned",
    })
    # (d) optimizer
    age_drift = float((d.get("kernel") or {}).get("age") or 0.0)
    causes.append({
        "id": "d_optimizer_drift",
        "strength": float(min(1.0, age_drift / 5.0)),
        "number": f"kernel age drift={age_drift:.4f}",
        "explains_null_if": "large age-group drift with zero T5 effect → null direction",
    })
    # (e) R1
    causes.append({
        "id": "e_r1_redundancy",
        "strength": float(e["linear_probe_R2_h_to_age"]) if np.isfinite(e["linear_probe_R2_h_to_age"]) else 0.0,
        "number": f"R²(h→age)={e['linear_probe_R2_h_to_age']:.4f}; "
                  f"Δbce kernel={e['kernel_delta_bce']:.4e} vanilla={e['vanilla_delta_bce']:.4e}",
        "explains_null_if": "R1 already saturates age signal",
    })
    # (f) content
    R = t4.get("kernel_max_R") or 0.0
    causes.append({
        "id": "f_content_dominance",
        "strength": float(1.0 / (1.0 + max(R, 0.0) * 100)) if np.isfinite(R) else 1.0,
        "number": f"R={R}; σ_content={f['sigma_content']}",
        "explains_null_if": "content logits swamp kernel bias (R≪1)",
    })
    causes_sorted = sorted(causes, key=lambda x: -x["strength"])

    out = {
        "a_fourier_band": a,
        "b_saddle_escape": b,
        "c_row_basis_conditioning": c,
        "d_optimizer_drift": d,
        "e_r1_redundancy": e,
        "f_content_dominance": f,
        "ranked_causes": causes_sorted,
    }
    D.print_block("T7 localization", [
        f"({i+1}) {c_['id']}: strength={c_['strength']:.3f}  {c_['number']}"
        for i, c_ in enumerate(causes_sorted)
    ])
    return out
