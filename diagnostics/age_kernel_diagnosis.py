#!/usr/bin/env python3
"""Quantitative diagnosis of the multiplicative age-conditioned temporal kernel.

Loads checkpoints/age_real_202605112156/epoch_012.pt (trained with
kernel_injection='multiplicative', pre-flag code) and real tensorized val
batches, then measures where the age signal is / isn't doing work.

Sections:
  A. checkpoint statics: coefficients, Delta-alpha(age) curves, Fourier aliasing
  B. real-batch attention forensics: QK score scale vs kernel, suppression failure
  C. age / kernel ablations at inference: recall@k deltas
  D. gradient flow into the age path (multiplicative vs additive counterfactual)
  E. polynomial basis conditioning on the empirical log-delta-t distribution
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.dataset import TensorizedEHRDataset, ehr_collate
from model.tale_ehr_age import TALEEHRAge
from model.train import bce_code_loss, compute_metrics, weibull_nll_loss

# Overridden by --ckpt_dir / --ckpt_epoch / --injection at runtime.
CKPT_DIR = REPO_ROOT / "checkpoints/age_real_202605112156"
CKPT_EPOCH = 12
INJECTION = "multiplicative"
DAYS_GRID = torch.tensor([0.0, 1.0, 7.0, 30.0, 90.0, 365.0, 730.0, 1095.0])
DAYS_LABELS = ["0d", "1d", "7d", "30d", "90d", "1y", "2y", "3y"]
AGE_GRID = [1.0, 5.0, 12.0, 18.0, 30.0, 50.0, 65.0, 80.0]


def load_model(kernel_injection: str | None = None) -> TALEEHRAge:
    kernel_injection = kernel_injection or INJECTION
    model = TALEEHRAge(
        embedding_path=REPO_ROOT / "data/processed/bge_embeddings.pt",
        num_codes=30635,
        d_model=256,
        poly_degree=5,
        age_conditioning_mode="real",
        kernel_injection=kernel_injection,
    )
    ckpt = torch.load(CKPT_DIR / f"epoch_{CKPT_EPOCH:03d}.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[load] {CKPT_DIR.name}/epoch_{CKPT_EPOCH:03d} inj={kernel_injection} "
          f"val_loss={ckpt['val_loss']:.4f} missing={missing} unexpected={unexpected}")
    model.eval()
    return model


def get_batches(n_batches: int, batch_size: int, seed: int = 0):
    ds = TensorizedEHRDataset(REPO_ROOT / "data/processed/tensorized/val",
                              REPO_ROOT / "data/processed/code_vocab.json")
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=n_batches * batch_size, replace=False)
    batches = []
    for b in range(n_batches):
        items = [ds[int(i)] for i in idxs[b * batch_size:(b + 1) * batch_size]]
        batches.append(ehr_collate(items))
    return batches


# ---------------------------------------------------------------- section A
def section_a(model: TALEEHRAge) -> None:
    print("\n=== A. CHECKPOINT STATICS ===")
    attn_tw = model.time_aware_attention.temporal_weight
    agg_tw = model.temporal_aggregation.temporal_weight
    print(f"attn base coeffs: {[round(c, 4) for c in attn_tw.coefficients.tolist()]}")
    print(f"agg  base coeffs: {[round(c, 4) for c in agg_tw.coefficients.tolist()]}")
    print(f"q_base norm: {float(model.temporal_aggregation.q_base.norm()):.3f}")
    for name, gen in [("attn", attn_tw.age_coeff_gen), ("agg", agg_tw.age_coeff_gen)]:
        w1, w2 = gen.mlp[0].weight, gen.mlp[2].weight
        b2 = gen.mlp[2].bias
        print(f"{name} age_coeff_gen: ||W1||_F={float(w1.norm()):.3f} "
              f"||W2||_F={float(w2.norm()):.3f} ||b2||={float(b2.norm()):.3f}")

    age_emb = model.time_aware_attention.age_emb
    gen = attn_tw.age_coeff_gen

    # Delta-alpha over a fine age grid: magnitude and smoothness
    ages = torch.arange(0.0, 100.0, 0.02)
    with torch.no_grad():
        da = gen(age_emb(ages))            # [N, 6]
    norms = da.norm(dim=-1)
    print(f"\n||Dalpha(a)|| over a in [0,100): mean={float(norms.mean()):.3f} "
          f"std={float(norms.std()):.3f} min={float(norms.min()):.3f} max={float(norms.max()):.3f}")
    base = attn_tw.coefficients
    print(f"||base coeffs|| = {float(base.norm()):.3f}  ->  ratio ||Dalpha||/||base|| = "
          f"{float(norms.mean() / base.norm()):.2f}")
    per_coef_std = da.std(dim=0)
    per_coef_mean = da.mean(dim=0)
    print("Dalpha per-coefficient mean (age-invariant offset): "
          f"{[round(float(v), 3) for v in per_coef_mean]}")
    print("Dalpha per-coefficient std over ages (age-varying part): "
          f"{[round(float(v), 3) for v in per_coef_std]}")
    # decompose: constant offset vs age-varying
    da_centered = da - per_coef_mean
    print(f"||age-invariant offset|| = {float(per_coef_mean.norm()):.3f} ; "
          f"mean ||age-varying part|| = {float(da_centered.norm(dim=-1).mean()):.3f}")

    # smoothness / aliasing: how much does Dalpha move for tiny vs large age steps
    with torch.no_grad():
        for step_lbl, step in [("+7d", 7 / 365.25), ("+1mo", 1 / 12), ("+1y", 1.0), ("+10y", 10.0)]:
            da2 = gen(age_emb(ages + step))
            d = (da2 - da).norm(dim=-1)
            print(f"mean ||Dalpha(a{step_lbl}) - Dalpha(a)|| = {float(d.mean()):.3f}")
    with torch.no_grad():
        phi = age_emb(ages)
        for step_lbl, step in [("+7d", 7 / 365.25), ("+10y", 10.0)]:
            d = (age_emb(ages + step) - phi).norm(dim=-1)
            print(f"Fourier features: mean ||phi(a{step_lbl}) - phi(a)|| = {float(d.mean()):.3f} "
                  f"(||phi||={float(phi.norm(dim=-1).mean()):.3f})")

    # attention-level w(dt, a) table
    log_dt = torch.log1p(DAYS_GRID / 7.0)
    print("\nattention-level w(dt, age) [sigmoid(poly)] :")
    print("age   " + "  ".join(f"{l:>5s}" for l in DAYS_LABELS))
    with torch.no_grad():
        for a in AGE_GRID:
            feats = age_emb(torch.tensor([a]))
            w = torch.sigmoid(attn_tw.poly_value(log_dt.unsqueeze(0), feats)).squeeze(0)
            print(f"{a:5.1f} " + "  ".join(f"{float(x):5.2f}" for x in w))
    print("\naggregation-level w(dt, age_current):")
    agg_emb = model.temporal_aggregation.age_emb
    with torch.no_grad():
        for a in AGE_GRID:
            feats = agg_emb(torch.tensor([a]))
            w = agg_tw(log_dt.unsqueeze(0), feats).squeeze(0)
            print(f"{a:5.1f} " + "  ".join(f"{float(x):5.2f}" for x in w))

    # jitter of w at fixed dt across a fine age grid (aliasing seen through the kernel)
    with torch.no_grad():
        fine = torch.arange(40.0, 41.0, 1 / 365.25)  # one year of days at age 40
        feats = age_emb(fine)
        w90 = torch.sigmoid(attn_tw.poly_value(
            torch.log1p(torch.full((len(fine), 1), 90.0) / 7.0), feats)).squeeze(-1)
        print(f"\nw(90d, a) for a in [40, 41) sampled daily: mean={float(w90.mean()):.3f} "
              f"std={float(w90.std()):.3f} min={float(w90.min()):.3f} max={float(w90.max()):.3f}")


# ---------------------------------------------------------------- section B
def attn_internals(model: TALEEHRAge, batch: dict) -> dict:
    """Recompute AgeConditionedTimeAwareAttention internals for one batch."""
    attn_mod = model.time_aware_attention
    code_embeddings = model.embedding_table[batch["code_indices"]]
    delta_t = batch["delta_t"]
    mask = batch["attention_mask"]
    age_years = batch["demographics"][..., 0]
    with torch.no_grad():
        q = attn_mod.mlp_q(code_embeddings)
        k = attn_mod.mlp_k(code_embeddings)
        feats = attn_mod.age_emb(age_years.clamp(min=0.0))
        scale = 1.0 / math.sqrt(attn_mod.d_model)
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale
        poly = attn_mod.temporal_weight.poly_value(delta_t, feats)
        w = torch.sigmoid(poly)
        b, l, _ = scores.shape
        causal = torch.tril(torch.ones((l, l), dtype=torch.bool))
        full_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)) & causal.unsqueeze(0)
    return {"scores": scores, "w": w, "poly": poly, "mask": full_mask,
            "row_mask": mask, "q": q, "k": k}


def section_b(model: TALEEHRAge, batches: list[dict]) -> None:
    print(f"\n=== B. REAL-BATCH ATTENTION FORENSICS (trained injection = {INJECTION}) ===")
    all_s, all_w = [], []
    ent_mult, ent_add, ent_nok = [], [], []
    mass_supp_mult, mass_supp_add, frac_supp = [], [], []
    row_range_raw, row_range_mult = [], []
    for batch in batches:
        d = attn_internals(model, batch)
        s, w, m = d["scores"], d["w"], d["mask"]
        sv, wv = s[m], w[m]
        all_s.append(sv)
        all_w.append(wv)

        neg_inf = torch.finfo(s.dtype).min
        logits_mult = torch.where(m, s * w, torch.tensor(neg_inf))
        logits_add = torch.where(m, s + F.logsigmoid(d["poly"]), torch.tensor(neg_inf))
        logits_nok = torch.where(m, s, torch.tensor(neg_inf))
        for logits, ents, masses in [
            (logits_mult, ent_mult, mass_supp_mult),
            (logits_add, ent_add, mass_supp_add),
            (logits_nok, ent_nok, None),
        ]:
            attn = torch.softmax(logits, dim=-1)
            attn = torch.where(m, attn, torch.zeros(()))
            rows = d["row_mask"]
            av = attn[rows]
            ent = -(av * torch.log(av.clamp_min(1e-12))).sum(-1)
            # only rows with >= 8 valid keys are informative for entropy
            nkeys = m.sum(-1)[rows]
            ents.append(ent[nkeys >= 8])
            if masses is not None:
                supp = (w < 0.05) & m
                masses.append((attn * supp).sum(-1)[rows][nkeys >= 8])
        nkeys = d["mask"].sum(-1)[d["row_mask"]]
        supp_frac_rows = ((d["w"] < 0.05) & m).sum(-1)[d["row_mask"]] / nkeys.clamp(min=1)
        frac_supp.append(supp_frac_rows[nkeys >= 8])

        # per-row logit dynamic range
        raw = torch.where(m, s, torch.tensor(float("nan")))
        mul = torch.where(m, s * w, torch.tensor(float("nan")))
        rr = (raw.nan_to_num(nan=-1e9).max(-1).values
              - torch.where(m, s, torch.tensor(1e9)).min(-1).values)
        rm = (mul.nan_to_num(nan=-1e9).max(-1).values
              - torch.where(m, s * w, torch.tensor(1e9)).min(-1).values)
        sel = d["row_mask"] & (d["mask"].sum(-1) >= 8)
        row_range_raw.append(rr[sel])
        row_range_mult.append(rm[sel])

    sv = torch.cat(all_s); wv = torch.cat(all_w)
    qs = torch.quantile(sv, torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99]))
    print(f"raw QK scores (q.k/sqrt(d)) over {sv.numel()} valid causal pairs: "
          f"mean={float(sv.mean()):.3f} std={float(sv.std()):.3f} "
          f"q01/25/50/75/99={[round(float(x), 3) for x in qs]} "
          f"frac<0 = {float((sv < 0).float().mean()):.3f}")
    print(f"kernel w over same pairs: mean={float(wv.mean()):.3f} std={float(wv.std()):.3f} "
          f"frac<0.05={float((wv < 0.05).float().mean()):.3f} "
          f"frac>0.95={float((wv > 0.95).float().mean()):.3f}")
    rr = torch.cat(row_range_raw); rm = torch.cat(row_range_mult)
    print(f"per-row logit range (rows with >=8 keys): raw={float(rr.mean()):.3f} "
          f"after *w = {float(rm.mean()):.3f}  (ratio {float((rm / rr.clamp(min=1e-9)).mean()):.2f})")
    em, ea, en = torch.cat(ent_mult), torch.cat(ent_add), torch.cat(ent_nok)
    print(f"attention entropy (nats, rows >=8 keys): multiplicative={float(em.mean()):.3f} "
          f"additive-counterfactual={float(ea.mean()):.3f} no-kernel={float(en.mean()):.3f}")
    mm, ma, fs = torch.cat(mass_supp_mult), torch.cat(mass_supp_add), torch.cat(frac_supp)
    print(f"'suppressed' pairs (w<0.05): {float(fs.mean()) * 100:.1f}% of keys per row; "
          f"attention mass they still receive: multiplicative-op={float(mm.mean()) * 100:.1f}% "
          f"additive-op={float(ma.mean()) * 100:.1f}%   (trained op = {INJECTION})")


# ---------------------------------------------------------------- section C
def run_eval(model: TALEEHRAge, batches: list[dict], transform=None,
             kernel_off: bool = False) -> dict[str, float]:
    metrics_sum: dict[str, float] = {}
    losses = []
    handles = []
    if kernel_off:
        def ones_fwd_poly(dt, feats):  # poly -> +inf => sigmoid -> 1
            return torch.full_like(dt, 30.0)
        attn_tw = model.time_aware_attention.temporal_weight
        agg_tw = model.temporal_aggregation.temporal_weight
        orig = (attn_tw.poly_value, agg_tw.forward)
        attn_tw.poly_value = ones_fwd_poly
        agg_tw.forward = lambda dt, feats: torch.ones_like(dt)
    try:
        with torch.no_grad():
            for batch in batches:
                bb = {k: v.clone() for k, v in batch.items()}
                if transform is not None:
                    bb = transform(bb)
                out = model(bb)
                losses.append(float(bce_code_loss(out["code_logits"], bb["target_codes"])))
                m = compute_metrics(out["code_logits"], bb["target_codes"])
                for k, v in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
    finally:
        if kernel_off:
            attn_tw.poly_value, agg_tw.forward = orig
    n = len(batches)
    res = {k: v / n for k, v in metrics_sum.items()}
    res["code_bce"] = float(np.mean(losses))
    return res


def section_c(model: TALEEHRAge, batches: list[dict]) -> None:
    print(f"\n=== C. INFERENCE ABLATIONS ({len(batches)} val batches) ===")

    def const_age(v):
        def t(b):
            d = b["demographics"].clone()
            d[..., 0] = torch.where(b["attention_mask"], torch.tensor(v), torch.tensor(0.0))
            b["demographics"] = d
            return b
        return t

    def shuffle_age(b):
        d = b["demographics"].clone()
        perm = torch.randperm(d.shape[0])
        d[..., 0] = d[perm, :, 0]
        b["demographics"] = d
        return b

    rows = [
        ("real ages", None, False),
        ("const age=63y (data mean)", const_age(63.0), False),
        ("const age=30y", const_age(30.0), False),
        ("const age=5y", const_age(5.0), False),
        ("ages shuffled across batch", shuffle_age, False),
        ("temporal kernel OFF (w=1)", None, True),
    ]
    torch.manual_seed(0)
    for name, tr, koff in rows:
        r = run_eval(model, batches, transform=tr, kernel_off=koff)
        print(f"{name:32s} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(r.items())))


# ---------------------------------------------------------------- section D
def _group_grad_norm(groups: dict, model) -> dict[str, float]:
    out = {}
    for name, g in groups.items():
        params = [g] if isinstance(g, torch.nn.Parameter) else list(g.parameters())
        gn = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in params if p.grad is not None))
        out[name] = gn
    return out


def section_d(batches: list[dict]) -> None:
    """Decompose gradient into the age path by (a) injection op and (b) loss term.

    Answers two questions apples-to-apples on this checkpoint's weights:
      1. multiplicative vs additive injection -> age-path gradient magnitude
      2. how much gradient the (broken) Weibull time loss pushes into the shared
         trunk / age path vs the code loss, at each run's gamma.
    """
    print("\n=== D. GRADIENT FLOW: INJECTION OP x LOSS TERM (one batch) ===")
    batch = batches[0]

    def groups_of(model):
        return {
            "attn.age_coeff_gen": model.time_aware_attention.temporal_weight.age_coeff_gen,
            "attn.base_coeffs": model.time_aware_attention.temporal_weight.coefficients,
            "agg.age_coeff_gen": model.temporal_aggregation.temporal_weight.age_coeff_gen,
            "code_predictor": model.code_predictor,
            "time_params_predictor": model.time_params_predictor,
        }

    # (1) injection-op comparison at gamma=500 (matches the multiplicative run's objective)
    print("-- injection op, loss = weibull + 500*code (multiplicative run's objective) --")
    for mode in ["multiplicative", "additive_logspace"]:
        model = load_model(kernel_injection=mode)
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(batch)
        loss = weibull_nll_loss(out["time_params"], batch["target_time_gap"]) \
            + 500.0 * bce_code_loss(out["code_logits"], batch["target_codes"])
        loss.backward()
        gn = _group_grad_norm(groups_of(model), model)
        print(f"[{mode}] attn.age_coeff_gen ||g||={gn['attn.age_coeff_gen']:.2e}  "
              f"attn.base_coeffs ||g||={gn['attn.base_coeffs']:.2e}  "
              f"agg.age_coeff_gen ||g||={gn['agg.age_coeff_gen']:.2e}")

    # (2) loss-term decomposition on the checkpoint's actual injection
    print(f"-- loss-term decomposition (trained injection = {INJECTION}) --")
    for tag, use_time, use_code, gamma in [
        ("code-only (gamma=1)", False, True, 1.0),
        ("weibull-only", True, False, 1.0),
        ("weibull + 500*code", True, True, 500.0),
        ("weibull + 1*code", True, True, 1.0),
    ]:
        model = load_model()
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(batch)
        loss = torch.zeros(())
        lt = weibull_nll_loss(out["time_params"], batch["target_time_gap"])
        lc = bce_code_loss(out["code_logits"], batch["target_codes"])
        if use_time:
            loss = loss + lt
        if use_code:
            loss = loss + gamma * lc
        loss.backward()
        gn = _group_grad_norm(groups_of(model), model)
        print(f"[{tag:20s}] loss={float(loss):8.3f} (time={float(lt):.2f} code={float(lc):.4f}) | "
              f"attn.age_gen={gn['attn.age_coeff_gen']:.2e} "
              f"agg.age_gen={gn['agg.age_coeff_gen']:.2e} "
              f"time_pred={gn['time_params_predictor']:.2e} "
              f"code_pred={gn['code_predictor']:.2e}")


# ---------------------------------------------------------------- section E
def section_e(batches: list[dict]) -> None:
    print("\n=== E. POLYNOMIAL BASIS CONDITIONING ===")
    dts = []
    for batch in batches:
        m = batch["attention_mask"]
        l = m.shape[1]
        causal = torch.tril(torch.ones((l, l), dtype=torch.bool))
        fm = (m.unsqueeze(1) & m.unsqueeze(2)) & causal.unsqueeze(0)
        dts.append(batch["delta_t"][fm])
    t = torch.cat(dts)
    qs = torch.quantile(t, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]))
    print(f"empirical log1p(dt/7) over {t.numel()} pairs: "
          f"min/25/50/75/95/99/max = {[round(float(x), 2) for x in qs]}")
    n = min(len(t), 200_000)
    ts = t[torch.randperm(len(t))[:n]].double()
    V = torch.stack([ts ** k for k in range(6)], dim=-1)
    G = V.T @ V / n
    cond_mono = float(torch.linalg.cond(G))
    tmax = float(ts.max())
    x = 2 * ts / tmax - 1
    T = [torch.ones_like(x), x]
    for _ in range(4):
        T.append(2 * x * T[-1] - T[-2])
    C = torch.stack(T, dim=-1)
    Gc = C.T @ C / n
    cond_cheb = float(torch.linalg.cond(Gc))
    print(f"Gram condition number, monomial basis on raw log-dt: {cond_mono:.2e}")
    print(f"Gram condition number, Chebyshev basis on rescaled [-1,1]: {cond_cheb:.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sections", type=str, default="ABDE")
    p.add_argument("--n_batches", type=int, default=6)
    p.add_argument("--n_eval_batches", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/age_real_202605112156")
    p.add_argument("--ckpt_epoch", type=int, default=12)
    p.add_argument("--injection", type=str, default="multiplicative",
                   choices=["multiplicative", "additive_logspace"])
    args = p.parse_args()

    CKPT_DIR = REPO_ROOT / args.ckpt_dir
    CKPT_EPOCH = args.ckpt_epoch
    INJECTION = args.injection

    torch.manual_seed(0)
    model = load_model()
    n_load = max(args.n_batches,
                 args.n_eval_batches if "C" in args.sections else 0)
    batches = get_batches(n_load, args.batch_size)
    if "A" in args.sections:
        section_a(model)
    if "B" in args.sections:
        section_b(model, batches[: args.n_batches])
    if "C" in args.sections:
        section_c(model, batches[: args.n_eval_batches])
    if "D" in args.sections:
        section_d(batches)
    if "E" in args.sections:
        section_e(batches[: args.n_batches])
    print("\nDone.")
