#!/usr/bin/env python3
"""The **only** module in this package that prints or writes JSON (D11).

The legacy tree emitted the same ``w(t)`` statistics from three call sites in three formats.
Here every other module returns diagnostic *tensors* and this module owns all formatting,
console output and serialisation. ``tests/test_no_stray_logging.py`` enforces it.

Everything computed here is **MEASURE**: it has no correct value known in advance, it is
printed alongside the legacy comparison where one exists, and it is never turned into an
assertion. The one exception is :func:`review_and_exit`, which stops a run cleanly so a
human can look -- it raises nothing.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

__all__ = [
    "AGE_BANDS", "band_index", "band_names",
    "print_block", "print_kv", "print_band_characterization", "print_config_summary",
    "print_epoch", "print_finetune_epoch", "print_invariant_table", "review_and_exit",
    "write_json", "append_train_json", "jsonify",
    "recall_metrics", "recall_per_example", "aggregate_recall",
    "micro_auroc_diagnostic_only", "gram_condition_numbers",
    "alpha_diagnostics", "delta_alpha_grid", "w_curves", "attention_stats",
    "snapshot_parameters", "parameter_drift", "clamp_rates", "alpha_l1",
    "TAU_GRID_DAYS", "W_CURVE_AGES", "DELTA_ALPHA_GRID_AGES",
    # -- offline pretraining evaluation (eval_pretrain.py) ------------------- #
    "MIN_BAND_N", "DENSE_AGE_GRID", "KERNEL_SEPARATION_AGES", "EVAL_KS", "NDCG_K",
    "average_precision_from_counts", "ScoreHistogram", "PerCodeHistogram",
    "band_masks", "band_entry", "reliability",
    "topk_per_example", "bce_totals",
    "age_conditioner_sites", "gradient_group_norms", "generator_gradient_fractions",
    "delta_alpha_norms", "kernel_separation",
    "print_eval_header", "print_config_check", "print_eval_epoch", "print_selection",
    "print_cross_arm_summary", "print_report_back",
]

# --------------------------------------------------------------------------- #
# Shared constants. Age bands are defined ONCE and shared by metrics and by the #
# Delta-alpha decomposition, so a pooled number and a banded number can never    #
# disagree about what "1-5" means.                                              #
# --------------------------------------------------------------------------- #
AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.0),
    ("18-40", 18.0, 40.0),
    ("40-65", 40.0, 65.0),
    ("65+", 65.0, float("inf")),
)

TAU_GRID_DAYS: tuple[float, ...] = (0.0, 1.0, 7.0, 30.0, 90.0, 365.0, 730.0, 1095.0)
W_CURVE_AGES: tuple[float, ...] = (0.25, 1.0, 3.0, 8.0, 15.0, 40.0, 70.0)
DELTA_ALPHA_GRID_AGES = np.round(np.arange(0.0, 90.25, 0.25), 4)

_RULE = "-" * 78


def band_names() -> list[str]:
    return [b[0] for b in AGE_BANDS]


def band_index(ages: torch.Tensor | np.ndarray) -> np.ndarray:
    """-> int array of band indices, one per age. Ages outside every band give -1."""
    a = ages.detach().cpu().numpy() if isinstance(ages, torch.Tensor) else np.asarray(ages)
    out = np.full(a.shape, -1, dtype=np.int64)
    for i, (_, lo, hi) in enumerate(AGE_BANDS):
        out[(a >= lo) & (a < hi)] = i
    return out


# --------------------------------------------------------------------------- #
# Output primitives                                                            #
# --------------------------------------------------------------------------- #
def print_block(title: str, lines: Iterable[str], *, stream=sys.stdout) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}", file=stream)
    for line in lines:
        print(f"  {line}", file=stream)
    stream.flush()


def print_kv(title: str, mapping: dict, *, stream=sys.stdout, width: int = 34) -> None:
    lines = []
    for k, v in mapping.items():
        if isinstance(v, float):
            v = f"{v:.6g}"
        lines.append(f"{str(k):<{width}}: {v}")
    print_block(title, lines, stream=stream)


def review_and_exit(title: str, lines: Iterable[str], *, code: int = 0) -> None:
    """Stop for human judgement. Prints and exits **cleanly** -- never raises."""
    print_block(title, lines)
    print("\n  This is a review checkpoint, not a failure. Nothing was trained.\n")
    sys.stdout.flush()
    raise SystemExit(code)


def jsonify(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return jsonify(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic: write ``.tmp`` then ``os.replace``, so a crash leaves a valid file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(jsonify(obj), f, indent=2)
    os.replace(tmp, p)


def append_train_json(path: str | Path, records: list[dict]) -> None:
    """``train.json`` is a JSON array, re-flushed atomically after every epoch."""
    write_json(path, records)


# --------------------------------------------------------------------------- #
# Formatted reports                                                            #
# --------------------------------------------------------------------------- #
LEGACY_BAND = {
    "saturation_gap_months_at_a5": 7.4,
    "min_pairwise_distance": 3.35,
    "max_possible_distance": 8.00,
    "grad_ratio_0.5_over_40": 1.0,
    "adult_min_pairwise_distance": 4.06,
}


def print_band_characterization(got: dict, legacy: dict | None = None) -> None:
    """Phase 2.3. MEASURE: printed beside the legacy linear band, never asserted."""
    leg = legacy or {}

    def cell(key: str, fmt: str = "{:.3f}") -> str:
        a = got.get(key, float("nan"))
        b = leg.get(key, LEGACY_BAND.get(key, float("nan")))
        return f"{fmt.format(a):>12}  |{fmt.format(b):>12}"

    lines = [
        f"{'quantity':<44}{'log-age (this)':>13}  |{'legacy linear':>13}",
        "  " + "-" * 74,
        f"{'saturation gap at a=5 (months)':<44}" + cell("saturation_gap_months_at_a5"),
        f"{'min pairwise ||psi|| on 0.25y grid [0,90]':<44}" + cell("min_pairwise_distance"),
        f"{'  ... against max 2*sqrt(M)':<44}" + cell("max_possible_distance"),
        f"{'||dpsi/da|| at a=0.5 / at a=40':<44}" + cell("grad_ratio_0.5_over_40"),
        f"{'min pairwise distance among adults':<44}" + cell("adult_min_pairwise_distance"),
    ]
    na_got = got.get("nearest_adult_distance", {})
    na_leg = leg.get("nearest_adult_distance", {})
    for a in ("0.5", "2.0", "5.0", "10.0"):
        g = na_got.get(a, float("nan"))
        b = na_leg.get(a, float("nan"))
        lines.append(f"{'nearest-adult distance, a=' + a:<44}{g:>12.3f}  |{b:>12.3f}")
    r_got = got.get("nearest_adult_over_adult_min", {})
    r_leg = leg.get("nearest_adult_over_adult_min", {})
    for a in ("0.5", "5.0"):
        lines.append(f"{'  ... as a multiple of adult-adult min, a=' + a:<44}"
                     f"{r_got.get(a, float('nan')):>12.2f}  |{r_leg.get(a, float('nan')):>12.2f}")
    lines += [
        "  " + "-" * 74,
        "HARD here is only injectivity (min pairwise > 0). Everything else is for review.",
        "How to read it: the last rows are the load-bearing ones. A ratio near 1 means a",
        "child is no more distinguishable from an adult than two adults are from each other",
        "-- a hash of age. A large ratio means pediatric ages sit off the adult manifold,",
        "which is what a developmental coordinate has to do. A grad ratio far from 1.0 means",
        "resolution is allocated by developmental rate rather than uniformly in calendar time.",
        "",
        "NOTE: the legacy column is re-measured here by the same procedure, not quoted. It",
        "reproduces the brief's reference values for min pairwise (3.35), adult min (4.06)",
        "and grad ratio (~1.0), but NOT the 7.4-month saturation gap; with the analytic",
        "asymptote sqrt(2M) the same band measures ~12.3 months. The brief's 7.4 figure used",
        "some other definition. The relative comparison between columns is unaffected.",
    ]
    print_block("Phase 2.3  age-embedding band characterization  [MEASURE]", lines)


def print_config_summary(config: dict) -> None:
    m, o, d = config.get("model", {}), config.get("optim", {}), config.get("data", {})
    p = config.get("params", {})
    lines = [
        f"run_id        : {config.get('run_id')}   arm={config.get('arm')} seed={config.get('seed')}",
        f"git           : {config.get('git_commit')} (dirty={config.get('git_dirty')})",
        f"encoder       : n_layers={m.get('n_layers')} d_model={m.get('d_model')} "
        f"n_heads={m.get('n_heads')} residual={m.get('use_residual')} "
        f"layernorm={m.get('use_layernorm')} ffn={m.get('use_ffn')}",
        f"kernel        : s={m.get('s')} tau_max={m.get('tau_max')} "
        f"injection={m.get('injection')} masking={m.get('masking')} pooling={m.get('pooling')}",
        f"age band      : {m.get('fourier')}",
        f"generator     : final_bias={m.get('gen_final_bias')} "
        f"center_delta_alpha={m.get('center_delta_alpha')}",
        f"demographics  : demo_dim={m.get('demo_dim')} race={m.get('race_encoding')}",
        f"params        : {p}",
        f"optim         : {o}",
        f"data          : vocab={d.get('vocab_size')} n_examples={d.get('n_examples')} "
        f"tau_max_source={d.get('tau_max_source')}",
    ]
    if config.get("deviations_from_draft"):
        lines.append("deviations    :")
        for item in config["deviations_from_draft"]:
            lines.append(f"    - {item['item']}: {item['reason']}")
    print_block("run configuration", lines)


def print_epoch(record: dict) -> None:
    e = record.get("epoch")
    lines = [
        f"train_loss={record.get('train_loss'):.6f}   val_loss={record.get('val_loss'):.6f}   "
        f"steps={record.get('step')}   wall={record.get('wall_clock_s'):.1f}s",
        f"recall@5/10/20 = {record.get('recall@5'):.4f} / {record.get('recall@10'):.4f} / "
        f"{record.get('recall@20'):.4f}"
        f"   micro-AUROC={record.get('micro_auroc')} [diagnostic_only]",
    ]
    banded = record.get("recall_by_band", {})
    if banded:
        lines.append("recall@10 by age band:")
        for name in band_names():
            b = banded.get(name)
            if b and b.get("n"):
                lines.append(f"    {name:<6} n={b['n']:>7}  r@10={b.get('recall@10'):.4f}")
            else:
                lines.append(f"    {name:<6} n=      0  --")
    for site, st in (record.get("alpha", {}) or {}).items():
        lines.append(
            f"[{site}] alpha_base={np.round(st['alpha_base'], 4).tolist()}  "
            f"||alpha||_1={st['alpha_l1']:.4f}  clamp={st['clamp_fraction']:.2e}")
        lines.append(
            f"    ||d_alpha|| mean={st['delta_alpha_norm_mean']:.4e} "
            f"std={st['delta_alpha_norm_std']:.4e}  "
            f"const={st['constant_component']:.4e}  varying={st['varying_component']:.4e}")
        if st["constant_component"] > 0 and st["varying_component"] < 0.1 * st["constant_component"]:
            lines.append("    ^ constant component dominates: the MLP learned an offset, "
                         "which alpha_base absorbs.")
    drift = record.get("param_drift", {})
    if drift:
        lines.append("param drift ||theta_t - theta_0|| / ||theta_0||: " +
                     "  ".join(f"{k}={v:.3e}" if v is not None else f"{k}=n/a"
                               for k, v in drift.items()))
    att = record.get("attention", {})
    if att:
        lines.append(
            "attention: " + "  ".join(f"{k}={v:.4f}" for k, v in att.items() if v is not None))
    print_block(f"epoch {e}", lines)


def print_finetune_epoch(record: dict) -> None:
    lines = [f"train_loss={record.get('train_loss'):.6f}   "
             f"wall={record.get('wall_clock_s'):.1f}s"]
    for split, ev in (record.get("eval") or {}).items():
        lines.append(f"[{split}] n={ev.get('n')} prev={ev.get('prevalence', float('nan')):.4f} "
                     f"loss={ev.get('loss', float('nan')):.6f} "
                     f"auroc={ev.get('auroc', float('nan')):.4f} "
                     f"auprc={ev.get('auprc', float('nan')):.4f}")
        for name in band_names():
            b = (ev.get("by_band") or {}).get(name, {})
            if b.get("n"):
                lines.append(f"    {name:<6} n={b['n']:>6} pos={b.get('pos', 0):>5} "
                             f"auprc={b.get('auprc', float('nan')):.4f}")
    for site, st in (record.get("alpha", {}) or {}).items():
        lines.append(f"[{site}] const={st['constant_component']:.4e}  "
                     f"varying={st['varying_component']:.4e}  "
                     f"||alpha||_1={st['alpha_l1']:.4f}  clamp={st['clamp_fraction']:.2e}")
    drift = record.get("param_drift", {})
    if drift:
        lines.append("param drift: " + "  ".join(
            f"{k}={v:.3e}" if v is not None else f"{k}=n/a" for k, v in drift.items()))
    print_block(f"fine-tune epoch {record.get('epoch')}", lines)


def print_invariant_table(rows: Sequence[tuple[str, str, str]]) -> None:
    lines = [f"{'invariant':<18}{'test':<38}{'result'}", "  " + "-" * 74]
    lines += [f"{a:<18}{b:<38}{c}" for a, b, c in rows]
    print_block("invariant -> test", lines)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def recall_per_example(logits: torch.Tensor, targets: torch.Tensor,
                       ks: Sequence[int] = (5, 10, 20)) -> dict[int, torch.Tensor]:
    """Per-example recall@k, so an epoch can be accumulated without holding every logit."""
    probs = torch.sigmoid(logits.float())
    n_pos = targets.sum(dim=-1).clamp(min=1)
    out = {}
    for k in ks:
        topk = probs.topk(min(k, probs.shape[-1]), dim=-1).indices
        out[k] = (targets.gather(1, topk).sum(dim=-1) / n_pos).cpu()
    return out


def aggregate_recall(per_example: dict[int, torch.Tensor], ages: torch.Tensor) -> dict:
    """Pool the accumulated per-example recalls, overall and by age band."""
    out: dict[str, Any] = {f"recall@{k}": float(v.mean()) for k, v in per_example.items()}
    out["n"] = int(next(iter(per_example.values())).numel())
    idx = band_index(ages)
    by_band: dict[str, dict] = {}
    for i, (name, _, _) in enumerate(AGE_BANDS):
        sel = torch.from_numpy(idx == i)
        n = int(sel.sum())
        entry: dict[str, Any] = {"n": n}
        for k, v in per_example.items():
            entry[f"recall@{k}"] = float(v[sel].mean()) if n else float("nan")
        by_band[name] = entry
    out["by_band"] = by_band
    return out


@torch.no_grad()
def gram_condition_numbers(tau_samples: np.ndarray, s: int, tau_max: float) -> dict:
    """Condition number of the Gram matrix on the **empirical** lag distribution, for the
    monomial basis the legacy kernel used and for the Chebyshev basis used here.

    Reported with the constant term included (which is how the legacy figures were
    computed) and without it (which is what this implementation actually parameterises).
    """
    from model_new.basis import chebyshev_basis

    t = np.asarray(tau_samples, dtype=np.float64).ravel()
    x = np.clip(2.0 * t / tau_max - 1.0, -1.0, 1.0)
    mono = np.stack([t ** k for k in range(s + 1)], axis=1)
    # T_1..T_s from the single implementation in basis.py; T_0 is prepended here only so the
    # "with constant" column is comparable to the monomial one.
    t1_ts = chebyshev_basis(torch.from_numpy(x), s).numpy()
    cheb = np.concatenate([np.ones((x.size, 1)), t1_ts], axis=1)

    def cond(design: np.ndarray) -> float:
        g = design.T @ design / max(1, design.shape[0])
        return float(np.linalg.cond(g))

    return {
        "n_samples": int(t.size),
        "s": int(s),
        "tau_max": float(tau_max),
        "monomial_with_constant": cond(mono),
        "chebyshev_with_constant": cond(cheb),
        "monomial_no_constant": cond(mono[:, 1:]),
        "chebyshev_no_constant": cond(cheb[:, 1:]),
    }


def recall_metrics(logits: torch.Tensor, targets: torch.Tensor,
                   ages: torch.Tensor | None = None,
                   ks: Sequence[int] = (5, 10, 20)) -> dict:
    """recall@k overall and stratified by patient age band.

    Pooled-only metrics average away an effect concentrated in one band, so the banded
    breakdown is computed from the same tensors in the same pass.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits.float())
        n_pos = targets.sum(dim=-1).clamp(min=1)
        per_example: dict[int, torch.Tensor] = {}
        for k in ks:
            topk = probs.topk(min(k, probs.shape[-1]), dim=-1).indices
            per_example[k] = targets.gather(1, topk).sum(dim=-1) / n_pos
        out: dict[str, Any] = {f"recall@{k}": float(v.mean()) for k, v in per_example.items()}
        out["n"] = int(logits.shape[0])
        if ages is not None:
            idx = band_index(ages)
            by_band: dict[str, dict] = {}
            for i, (name, _, _) in enumerate(AGE_BANDS):
                sel = torch.from_numpy(idx == i).to(logits.device)
                n = int(sel.sum())
                entry = {"n": n}
                for k, v in per_example.items():
                    entry[f"recall@{k}"] = float(v[sel].mean()) if n else float("nan")
                by_band[name] = entry
            out["by_band"] = by_band
        return out


def micro_auroc_diagnostic_only(logits: torch.Tensor, targets: torch.Tensor) -> float | None:
    """Raveled over ~12k codes this reads ~0.99 and means nothing. Tagged, never headline."""
    try:
        from sklearn.metrics import roc_auc_score
        y = targets.detach().cpu().numpy().ravel()
        p = torch.sigmoid(logits.float()).detach().cpu().numpy().ravel()
        if 0 < y.sum() < len(y):
            return float(roc_auc_score(y, p))
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Age-pathway diagnostics                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def alpha_diagnostics(model, ages_flat: torch.Tensor) -> dict:
    """Per kernel site: ``alpha_base``, ``||alpha||_1``, clamp rate, and the constant /
    age-varying decomposition of ``Delta-alpha`` -- overall and per age band.

    The decomposition is the load-bearing number. A growing constant component with a flat
    varying component means the MLP learned an offset, which ``alpha_base`` absorbs and
    which cannot make the kernel age-dependent.
    """
    out: dict[str, dict] = {}
    idx = band_index(ages_flat)
    for name, site in model.kernel_sites():
        delta = site.age(ages_flat)  # [N, s]
        norms = delta.norm(dim=-1)
        mean_delta = delta.mean(dim=0)
        resid = delta - mean_delta
        entry = {
            "alpha_base": site.alpha_base.detach().cpu().numpy().tolist(),
            "alpha_l1": float(site.alpha_base.detach().abs().sum()),
            "clamp_fraction": site.kernel.clamp_fraction,
            "delta_alpha_norm_mean": float(norms.mean()),
            "delta_alpha_norm_std": float(norms.std()) if norms.numel() > 1 else 0.0,
            "constant_component": float(mean_delta.norm()),
            "varying_component": float(resid.norm(dim=-1).pow(2).mean().sqrt()),
            "by_band": {},
        }
        for i, (bname, _, _) in enumerate(AGE_BANDS):
            sel = torch.from_numpy(idx == i).to(delta.device)
            n = int(sel.sum())
            if n == 0:
                entry["by_band"][bname] = {"n": 0}
                continue
            d = delta[sel]
            md = d.mean(dim=0)
            entry["by_band"][bname] = {
                "n": n,
                "delta_alpha_norm_mean": float(d.norm(dim=-1).mean()),
                "constant_component": float(md.norm()),
                "varying_component": float((d - md).norm(dim=-1).pow(2).mean().sqrt()),
            }
        out[name] = entry
    return out


@torch.no_grad()
def delta_alpha_grid(model, ages: Sequence[float] | np.ndarray = DELTA_ALPHA_GRID_AGES) -> dict:
    """``Delta-alpha(a)`` on a dense 0-90 grid, every epoch, every site.

    MIMIC-IV hosp contains no patients under 18, so the pediatric range is pure
    extrapolation from a 2-layer MLP after pretraining, and the whole transfer claim rests
    on fine-tuning reshaping it from PIC data. This curve is how that is watched, and how
    one finds out whether pretraining left the pediatric range arbitrary.
    """
    dev = next(model.parameters()).device
    a = torch.as_tensor(np.asarray(ages, dtype=np.float32), device=dev)
    return {
        "ages": np.asarray(ages).tolist(),
        "sites": {name: site.age(a).detach().cpu().numpy().tolist()
                  for name, site in model.kernel_sites()},
    }


@torch.no_grad()
def w_curves(model, tau_grid_days: Sequence[float] = TAU_GRID_DAYS,
             ages: Sequence[float] = W_CURVE_AGES) -> dict:
    """``w(tau | a) = exp(log_w)`` on a fixed tau grid, at fixed ages, for every site.

    Dumping the raw curves means Figure 3 is generated from ``train.json`` with no extra run.
    """
    from model_new.data import lag_to_tau

    dev = next(model.parameters()).device
    tau = lag_to_tau(torch.tensor(list(tau_grid_days), dtype=torch.float32, device=dev))
    a = torch.tensor(list(ages), dtype=torch.float32, device=dev)
    sites = {}
    for name, site in model.kernel_sites():
        alpha = site.alpha_base + site.age(a)               # [A, s]
        log_w = site.kernel(tau.unsqueeze(0), alpha, count=False)  # [A, T]
        sites[name] = {"log_w": log_w.cpu().numpy().tolist(),
                       "w": log_w.exp().cpu().numpy().tolist()}
    return {"tau_grid_days": list(tau_grid_days), "tau_grid": tau.cpu().numpy().tolist(),
            "ages": list(ages), "sites": sites}


@torch.no_grad()
def attention_stats(model, batch: dict) -> dict:
    """Entropy and peakedness of the attention distributions on one batch."""
    out = model(batch, need_diagnostics=True)
    stats: dict[str, float] = {}

    def summarize(attn: torch.Tensor, valid: torch.Tensor, prefix: str) -> None:
        p = attn.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1)
        stats[f"{prefix}_entropy"] = float(ent[valid].mean())
        stats[f"{prefix}_frac_max_gt_0.9"] = float((attn.amax(dim=-1)[valid] > 0.9).float().mean())

    mask = batch["attention_mask"]
    summarize(out["pool_attn"], torch.ones_like(mask[:, 0], dtype=torch.bool), "pool")

    from model_new.data import pairwise_tau

    x = model.embedding_table[batch["code_indices"]]
    blk = model.encoder.blocks[0]
    h = blk.ln_attn(x) if blk.ln_attn is not None else x
    tau = pairwise_tau(batch["timestamps_days"], mask)   # tau is no longer in the batch
    _, attn, _ = blk.attn(h, tau, mask, batch["age_years"], need_weights=True)
    summarize(attn.mean(dim=1), mask, "encoder0")
    return stats


# --------------------------------------------------------------------------- #
# Parameter drift                                                              #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def snapshot_parameters(groups: Sequence[dict]) -> dict[str, torch.Tensor | None]:
    snap: dict[str, torch.Tensor | None] = {}
    for g in groups:
        ps = [p.detach().float().reshape(-1).clone() for p in g["params"]]
        snap[g["name"]] = torch.cat(ps).cpu() if ps else None
    return snap


@torch.no_grad()
def parameter_drift(groups: Sequence[dict],
                    snapshot: dict[str, torch.Tensor | None]) -> dict[str, float | None]:
    """``||theta_t - theta_0|| / ||theta_0||`` per group.

    This is the acceptance signal for the age pathway, not gradient norm: the generator's
    zero-initialised final layer gives its first layer exactly zero gradient at step 0, and
    under Adam's second-moment normalisation a tiny gradient still produces a full-size
    step, so gradient norm is uninformative here.
    """
    out: dict[str, float | None] = {}
    for g in groups:
        base = snapshot.get(g["name"])
        if base is None or base.numel() == 0:
            out[g["name"]] = None
            continue
        now = torch.cat([p.detach().float().reshape(-1) for p in g["params"]]).cpu()
        out[g["name"]] = float((now - base).norm() / base.norm().clamp_min(1e-12))
    return out


def clamp_rates(model) -> dict[str, float]:
    return {name: site.kernel.clamp_fraction for name, site in model.kernel_sites()}


def alpha_l1(model) -> dict[str, float]:
    return {name: float(site.alpha_base.detach().abs().sum())
            for name, site in model.kernel_sites()}


# =========================================================================== #
# Offline pretraining evaluation.                                             #
#                                                                             #
# Everything below is used by ``eval_pretrain.py``, which does no printing and #
# no JSON writing of its own (D11). Metric primitives live here rather than    #
# there for the same reason ``recall_metrics`` does: they are shared, they are #
# unit-tested, and a second implementation of average precision in a second    #
# file is exactly the duplication this package exists to avoid.                #
# =========================================================================== #

# Reliability threshold for a band-stratified metric. Below this many sequences a
# band metric is reported as NaN with ``unreliable: true`` rather than as a number.
# MIMIC-IV hosp has no patients under 18, so every band below 12-17 is empty and
# 12-17 itself has a handful of sequences -- reporting an AUPRC from those without a
# flag would put a meaningless number next to a meaningful one.
MIN_BAND_N = 200

# Δα(a) is measured on two age supports and they are NOT interchangeable: the dense
# grid says what the pathway *can* do, the empirical distribution says what it does on
# data. Both are reported, always labelled.
DENSE_AGE_GRID = np.round(np.linspace(0.0, 90.0, 181), 4)   # 0.5 y spacing
KERNEL_SEPARATION_AGES: tuple[float, ...] = (1.0, 5.0, 10.0, 15.0, 25.0, 45.0, 65.0, 85.0)

EVAL_KS: tuple[int, ...] = (10, 20)
NDCG_K = 20


# --------------------------------------------------------------------------- #
# Average precision from fixed-edge score histograms                          #
# --------------------------------------------------------------------------- #
def average_precision_from_counts(pos_counts, tot_counts):
    """Step-wise average precision from per-bin (positive, total) counts.

    ``pos_counts`` / ``tot_counts`` are indexed by bin in **ascending** score order;
    the trailing axis is the bin axis, so a ``[n_codes, n_bins]`` input returns one AP
    per code. Scores inside a bin are tied, and the estimate treats them as tied --
    which is precisely what ``sklearn.average_precision_score`` does with equal scores,
    so the only error is the quantisation of the scores themselves.

        AP = sum_i P_i * (R_i - R_{i+1}) = sum_i P_i * pos_i / n_pos

    with the sum running from the highest bin down and ``P_i`` computed inclusive of
    bin ``i``. Empty bins contribute exactly zero, so no masking is needed.

    -> float for a 1-D input, ``np.ndarray`` for 2-D. NaN where a stratum has no
    positives; never imputed.
    """
    pos = torch.as_tensor(pos_counts).double()
    tot = torch.as_tensor(tot_counts).double()
    if pos.shape != tot.shape:
        raise ValueError(f"count shapes disagree: {tuple(pos.shape)} vs {tuple(tot.shape)}")
    pos_desc, tot_desc = pos.flip(-1), tot.flip(-1)
    prec = pos_desc.cumsum(-1) / tot_desc.cumsum(-1).clamp(min=1.0)
    n_pos = pos.sum(-1)
    ap = (prec * pos_desc).sum(-1) / n_pos.clamp(min=1.0)
    ap = torch.where(n_pos > 0, ap, torch.full_like(ap, float("nan")))
    return float(ap) if ap.ndim == 0 else ap.cpu().numpy()


class ScoreHistogram:
    """Streaming fixed-edge score histogram -> micro-average precision.

    The full score matrix for the pretraining task is ``n_val x |V|`` =
    52,227 x 30,635 = 1.6e9 floats (6.4 GB in float32, and both a sort and an argsort of
    it are needed for an exact AP). It does not fit, so the pooled AP is accumulated
    into per-bin positive/total counts instead and integrated at the end.

    Edges are fixed **once** and shared by every arm and every epoch: a histogram whose
    binning depends on the scores being histogrammed is not comparable across arms.
    Values outside the range are clamped into the end bins and **counted**; a nonzero
    out-of-range count is reported, never absorbed.
    """

    def __init__(self, lo: float, hi: float, n_bins: int = 100_000,
                 device: torch.device | str = "cpu") -> None:
        if not (hi > lo):
            raise ValueError(f"histogram range must have hi > lo, got [{lo}, {hi}]")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.lo, self.hi, self.n_bins = float(lo), float(hi), int(n_bins)
        self.width = (self.hi - self.lo) / self.n_bins
        self.pos = torch.zeros(self.n_bins, dtype=torch.int64, device=device)
        self.tot = torch.zeros(self.n_bins, dtype=torch.int64, device=device)
        self.n_below = 0
        self.n_above = 0

    def _bin(self, scores: torch.Tensor) -> torch.Tensor:
        idx = ((scores.float() - self.lo) / self.width).floor().long()
        self.n_below += int((idx < 0).sum())
        self.n_above += int((idx >= self.n_bins).sum())
        return idx.clamp_(0, self.n_bins - 1)

    @torch.no_grad()
    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        idx = self._bin(scores.reshape(-1))
        y = labels.reshape(-1)
        self.tot += torch.bincount(idx, minlength=self.n_bins)
        self.pos += torch.bincount(idx[y > 0], minlength=self.n_bins)

    @property
    def n(self) -> int:
        return int(self.tot.sum())

    @property
    def n_pos(self) -> int:
        return int(self.pos.sum())

    @property
    def n_neg(self) -> int:
        return self.n - self.n_pos

    def average_precision(self) -> float:
        return average_precision_from_counts(self.pos.cpu(), self.tot.cpu())

    def to_json(self) -> dict:
        return {
            "n_pairs": self.n, "n_pos": self.n_pos, "n_neg": self.n_neg,
            "n_bins": self.n_bins, "range": [self.lo, self.hi], "bin_width": self.width,
            "n_below_range": self.n_below, "n_above_range": self.n_above,
            "out_of_range_fraction": ((self.n_below + self.n_above) / self.n) if self.n else 0.0,
        }


class PerCodeHistogram:
    """One fixed-edge score histogram per code, for macro (per-code) average precision.

    Only the codes handed in are tracked. Eligibility (``>= min_pos`` positives in the
    stratum) is the caller's decision and is decided from the targets alone, before any
    model is run, so the included set is identical for every arm and every epoch. A code
    that is excluded is excluded -- it never receives an imputed value.
    """

    def __init__(self, code_index: torch.Tensor, lo: float, hi: float, n_bins: int = 2048,
                 device: torch.device | str = "cpu") -> None:
        if not (hi > lo):
            raise ValueError(f"histogram range must have hi > lo, got [{lo}, {hi}]")
        self.codes = torch.as_tensor(code_index, dtype=torch.long, device=device).reshape(-1)
        self.n_codes = int(self.codes.numel())
        self.lo, self.hi, self.n_bins = float(lo), float(hi), int(n_bins)
        self.width = (self.hi - self.lo) / self.n_bins
        self.pos = torch.zeros(self.n_codes, self.n_bins, dtype=torch.int64, device=device)
        self.tot = torch.zeros(self.n_codes, self.n_bins, dtype=torch.int64, device=device)
        self._offsets = (torch.arange(self.n_codes, device=device) * self.n_bins).unsqueeze(0)
        self.n_below = 0
        self.n_above = 0

    @torch.no_grad()
    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        """``scores``/``targets`` are ``[B, |V|]``; the tracked columns are gathered here."""
        if self.n_codes == 0 or scores.shape[0] == 0:
            return
        s = scores.index_select(1, self.codes).float()
        y = targets.index_select(1, self.codes)
        idx = ((s - self.lo) / self.width).floor().long()
        self.n_below += int((idx < 0).sum())
        self.n_above += int((idx >= self.n_bins).sum())
        flat = (idx.clamp_(0, self.n_bins - 1) + self._offsets).reshape(-1)
        # scatter_add_ on int64 is order-independent, so the accumulation is exact and
        # reproducible regardless of how the GPU schedules the atomics.
        self.tot.view(-1).scatter_add_(0, flat, torch.ones_like(flat))
        self.pos.view(-1).scatter_add_(0, flat, y.reshape(-1).long())

    def positives_per_code(self) -> np.ndarray:
        return self.pos.sum(dim=1).cpu().numpy()

    def average_precision_per_code(self, chunk: int = 512) -> np.ndarray:
        out = np.full(self.n_codes, np.nan, dtype=np.float64)
        for i in range(0, self.n_codes, chunk):
            j = min(i + chunk, self.n_codes)
            out[i:j] = average_precision_from_counts(self.pos[i:j].cpu(), self.tot[i:j].cpu())
        return out

    def to_json(self) -> dict:
        n = int(self.tot.sum())
        return {
            "n_codes": self.n_codes, "n_bins": self.n_bins, "range": [self.lo, self.hi],
            "bin_width": self.width,
            "n_below_range": self.n_below, "n_above_range": self.n_above,
            "out_of_range_fraction": ((self.n_below + self.n_above) / n) if n else 0.0,
        }


# --------------------------------------------------------------------------- #
# Band stratification                                                         #
# --------------------------------------------------------------------------- #
def band_masks(ages: torch.Tensor | np.ndarray) -> dict[str, np.ndarray]:
    """-> ``{band name: bool mask}`` for every band in :data:`AGE_BANDS`, empty ones
    included. Bands are never dropped: an empty band is a reported fact about the
    corpus, not an absence of data to be hidden."""
    idx = band_index(ages)
    return {name: (idx == i) for i, (name, _, _) in enumerate(AGE_BANDS)}


def reliability(n: int, n_pos: int, n_neg: int, min_n: int = MIN_BAND_N) -> tuple[bool, str]:
    """-> (unreliable, reason). ``reason`` is the empty string when reliable."""
    if n == 0:
        return True, "band is empty (n = 0)"
    if n < min_n:
        return True, f"n = {n} < min_band_n = {min_n}"
    if n_pos == 0:
        return True, "no positive (sequence, code) pairs in this band"
    if n_neg == 0:
        return True, "no negative (sequence, code) pairs in this band"
    return False, ""


def band_entry(*, n: int, n_pos: int, n_neg: int, metrics: dict,
               min_n: int = MIN_BAND_N) -> dict:
    """The one shape every band-stratified block in this package uses.

    ``n`` / ``n_pos`` / ``n_neg`` are always present. When the band is unreliable every
    metric is set to NaN (``null`` in JSON) and the flag carries the reason -- a number
    computed from a handful of sequences is not reported as if it were comparable to one
    computed from twenty thousand.
    """
    bad, reason = reliability(n, n_pos, n_neg, min_n)
    entry: dict[str, Any] = {"n": int(n), "n_pos": int(n_pos), "n_neg": int(n_neg),
                             "unreliable": bool(bad)}
    if bad:
        entry["unreliable_reason"] = reason
    for k, v in metrics.items():
        entry[k] = float("nan") if bad else v
    return entry


# --------------------------------------------------------------------------- #
# Next-visit ranking metrics                                                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def topk_per_example(logits: torch.Tensor, targets: torch.Tensor,
                     ks: Sequence[int] = EVAL_KS, ndcg_k: int = NDCG_K,
                     cap_denominator: bool = False) -> dict[str, torch.Tensor]:
    """Per-sequence recall@k and nDCG@k, returned per example so an epoch can be pooled
    without holding every logit.

    The recall denominator is the number of true codes in the target visit,
    **uncapped** by default: with a median of 64 true codes, capping at k turns
    recall@10 into precision@10 and silently changes what is being measured.
    ``cap_denominator=True`` uses ``min(|true|, k)`` instead and the choice is recorded
    in the output JSON. A sequence with no true codes yields NaN rather than 0 and is
    counted separately -- a zero would be a real recall of zero, which it is not.

    nDCG uses binary relevance, the standard ``1/log2(i+1)`` discount and an ideal DCG
    over ``min(|true|, ndcg_k)`` positions. Sigmoid is monotone, so ranking on logits and
    ranking on probabilities are the same ranking; the logits are used directly.
    """
    scores = logits.float()
    n_true = targets.sum(dim=-1)
    k_max = min(int(max(max(ks), ndcg_k)), scores.shape[-1])
    top = scores.topk(k_max, dim=-1).indices
    hits = targets.gather(1, top).float()                      # [B, k_max], binary
    nan = torch.tensor(float("nan"), device=scores.device)

    out: dict[str, torch.Tensor] = {"n_true": n_true.detach().cpu()}
    for k in ks:
        num = hits[:, :k].sum(dim=-1)
        den = torch.minimum(n_true, torch.full_like(n_true, float(k))) if cap_denominator \
            else n_true
        out[f"recall@{k}"] = torch.where(den > 0, num / den.clamp(min=1.0), nan).detach().cpu()

    disc = 1.0 / torch.log2(torch.arange(2, ndcg_k + 2, device=scores.device).float())
    m = min(int(ndcg_k), int(hits.shape[1]))          # a vocabulary shorter than k is legal
    dcg = (hits[:, :m] * disc[:m]).sum(dim=-1)
    ideal_n = torch.minimum(n_true, torch.full_like(n_true, float(ndcg_k))).long()
    cum = torch.cat([torch.zeros(1, device=scores.device), disc.cumsum(0)])
    idcg = cum[ideal_n]
    out[f"ndcg@{ndcg_k}"] = torch.where(idcg > 0, dcg / idcg.clamp(min=1e-12), nan).detach().cpu()
    return out


@torch.no_grad()
def bce_totals(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """-> (per-sequence summed BCE ``[B]``, number of codes per sequence).

    ``reduction='none'`` summed over codes, so the caller can pool it either globally or
    within an age band from the same numbers. No ``pos_weight`` and no masking: every
    code in the vocabulary is a valid target for every sequence, which is what
    pretraining optimised.
    """
    import torch.nn.functional as F
    per = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none").sum(dim=-1)
    return per.detach().cpu(), int(logits.shape[-1])


# --------------------------------------------------------------------------- #
# DKM diagnostics at a trained checkpoint                                     #
# --------------------------------------------------------------------------- #
def age_conditioner_sites(model) -> list[tuple[str, Any]]:
    """-> ``[(site name, AgeConditioner)]`` for every coefficient generator in the model.

    Membership comes from the module structure (``kernel_sites()`` and the declared
    ``additive_age`` attribute), never from matching parameter names. The ``additive``
    arm's generator is not a kernel site -- it feeds the head -- but it is a coefficient
    generator and its gradients are read the same way.
    """
    sites = [(name, site.age) for name, site in model.kernel_sites()]
    extra = getattr(model, "additive_age", None)
    if extra is not None:
        sites.append(("additive_head", extra))
    return sites


@torch.no_grad()
def gradient_group_norms(groups: Sequence[dict]) -> dict[str, dict]:
    """L2 norm of the gradient of every optimizer group, read without stepping.

    Groups come from ``optim.build_param_groups``, so ``age`` / ``head`` / ``backbone``
    membership is the same declared partition the optimizer used. An empty group (the
    ``vanilla`` age group) reports **exactly** 0.0, never NaN and never a missing key.
    """
    out: dict[str, dict] = {}
    for g in groups:
        params = list(g["params"])
        sq, n_with_grad = 0.0, 0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().double().pow(2).sum())
                n_with_grad += 1
        out[g["name"]] = {
            "grad_l2": float(np.sqrt(sq)) if params else 0.0,
            "n_tensors": len(params),
            "n_params": int(sum(p.numel() for p in params)),
            "n_tensors_with_grad": n_with_grad,
            "empty_group": not params,
        }
    return out


@torch.no_grad()
def generator_gradient_fractions(model, eps: float = 1e-12) -> dict[str, dict]:
    """Fraction of coefficient-generator parameters carrying a nonzero gradient, per site.

    Reported at ``|g| > 0`` and ``|g| > eps`` separately, because a denormal-scale
    gradient is arithmetically nonzero and practically dead.

    Section 8 of the README explains a zero age gradient **at step 0**: the generator's
    final layer is zero-initialised, so its first layer gets ``dL/dW1 = 0`` exactly. That
    argument expires after step 0. At a trained checkpoint a zero fraction means the
    pathway stopped receiving signal, and it is flagged here rather than explained away.
    """
    out: dict[str, dict] = {}
    for name, cond in age_conditioner_sites(model):
        params = list(cond.age_parameters())
        n_total = int(sum(p.numel() for p in params))
        n_gt0 = n_gt_eps = 0
        n_missing = 0
        sq = 0.0
        for p in params:
            if p.grad is None:
                n_missing += p.numel()
                continue
            g = p.grad.detach()
            n_gt0 += int((g != 0).sum())
            n_gt_eps += int((g.abs() > eps).sum())
            sq += float(g.double().pow(2).sum())
        frac0 = (n_gt0 / n_total) if n_total else 0.0
        out[name] = {
            "mode": cond.mode,
            "n_params": n_total,
            "n_params_without_grad": n_missing,
            "frac_nonzero_grad_gt_0": frac0,
            "frac_nonzero_grad_gt_1e-12": (n_gt_eps / n_total) if n_total else 0.0,
            "grad_l2": float(np.sqrt(sq)),
            "zero_gradient_at_trained_checkpoint": bool(n_total > 0 and n_gt0 == 0),
        }
    return out


@torch.no_grad()
def delta_alpha_norms(model, empirical_ages: torch.Tensor,
                      dense_grid: Sequence[float] | np.ndarray = DENSE_AGE_GRID) -> dict:
    """``||Δα(a)||_2`` mean and max, per site, on two age supports.

    The two differ by construction and the difference is the point: the dense uniform
    grid on [0, 90] weights a one-year-old exactly as much as a sixty-year-old, while the
    empirical validation distribution is concentrated near the median (MIMIC-IV hosp has
    no patients under 18). A large dense-grid norm with a small empirical one means the
    pathway's variation lives where there is no data.
    """
    dev = next(model.parameters()).device
    dense = torch.as_tensor(np.asarray(dense_grid, dtype=np.float32), device=dev)
    emp = empirical_ages.detach().to(dev).float().reshape(-1)
    out: dict[str, dict] = {}
    for name, cond in age_conditioner_sites(model):
        entry = {"mode": cond.mode}
        for label, ages, desc in (
            ("dense_uniform_grid", dense,
             f"uniform grid on [0, 90], {len(dense)} points -- what the pathway CAN do"),
            ("empirical_validation", emp,
             "age at the last valid event of every validation sequence -- what it does on data"),
        ):
            norms = cond(ages).norm(dim=-1)
            entry[label] = {
                "n": int(ages.numel()),
                "support": desc,
                "mean": float(norms.mean()) if norms.numel() else float("nan"),
                "max": float(norms.max()) if norms.numel() else float("nan"),
                "age_min": float(ages.min()) if ages.numel() else float("nan"),
                "age_max": float(ages.max()) if ages.numel() else float("nan"),
            }
        out[name] = entry
    return out


@torch.no_grad()
def kernel_separation(model, ages: Sequence[float] = KERNEL_SEPARATION_AGES,
                      n_tau: int = 257) -> dict:
    """Pairwise kernel separation at representative ages, on **centered** ``log w`` curves.

    Softmax is invariant to a per-row constant, and within one attention row the query
    age -- hence ``α(a)`` -- is fixed, so a constant offset between two ages' ``log w``
    curves changes no attention weight whatsoever. Subtracting each age's mean over the
    τ grid before comparing is therefore not cosmetic: an uncentered ``max|Δ log w|``
    counts exactly the component that cannot matter, and overstates separation by however
    large that offset happens to be. Both are reported so the difference is visible.

    τ runs over ``[0, τ_max]`` with the checkpoint's frozen ``τ_max``; the Chebyshev
    evaluation goes through ``ChebyshevKernel``, the single implementation.
    """
    dev = next(model.parameters()).device
    tau = torch.linspace(0.0, float(model.tau_max), int(n_tau), device=dev)
    a = torch.as_tensor(list(ages), dtype=torch.float32, device=dev)
    out: dict[str, Any] = {"ages": list(ages), "n_tau": int(n_tau),
                           "tau_range": [0.0, float(model.tau_max)], "sites": {}}
    for name, site in model.kernel_sites():
        alpha = site.alpha_base + site.age(a)                         # [A, s]
        log_w = site.kernel(tau.unsqueeze(0), alpha, count=False)     # [A, T]
        centered = log_w - log_w.mean(dim=1, keepdim=True)
        d_cen = (centered.unsqueeze(1) - centered.unsqueeze(0)).abs().amax(dim=-1)
        d_raw = (log_w.unsqueeze(1) - log_w.unsqueeze(0)).abs().amax(dim=-1)
        out["sites"][name] = {
            "alpha_base": site.alpha_base.detach().cpu().numpy().tolist(),
            "log_w_centered": centered.cpu().numpy().tolist(),
            "max_abs_delta_log_w_centered": d_cen.cpu().numpy().tolist(),
            "max_pairwise_centered": float(d_cen.max()),
            "max_pairwise_uncentered": float(d_raw.max()),
            "note": ("centered = log w minus its own mean over the tau grid; the "
                     "uncentered figure includes a per-age constant that softmax ignores"),
        }
    return out


# --------------------------------------------------------------------------- #
# Offline evaluation reports                                                  #
# --------------------------------------------------------------------------- #
def _fmt(v: Any, spec: str = ".6f") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "nan"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def print_eval_header(info: dict) -> None:
    lines = [
        f"primary_rule       : {info['primary_rule']}   (from --primary_rule; written to "
        f"{info['selection_path']} before any cross-arm comparison)",
        f"runs               : {', '.join(info['runs'])}",
        f"arms               : {', '.join(info['arms'])}",
        f"val split          : {info['n_examples']} sequences, |V| = {info['vocab_size']}, "
        f"batch_size = {info['batch_size']}",
        f"batches per pass   : {info['n_batches']}  (shuffle=False, drop_last=False, "
        f"seed={info['seed']})",
        f"batch order hash   : {info['batch_order_hash']}",
        f"tau_max            : {info['tau_max']!r}  (from the checkpoint buffer, "
        f"asserted == {info['expected_tau_max']!r} to 1e-6)",
        f"device             : {info['device']}",
    ]
    if info.get("max_val_batches"):
        lines.append(f"SUBSAMPLED         : --max_val_batches {info['max_val_batches']} "
                     f"-- this is recorded in every output JSON")
    print_block("offline pretraining evaluation", lines)


def print_config_check(report: dict) -> None:
    lines = [f"reference run      : {report['reference_run']}",
             f"constructor kwargs : identical across all arms "
             f"({report['n_shared_kwargs']} keys compared)"]
    if report.get("arm_derived_differences"):
        lines.append("arm-derived differences (rebuilt from the shared kwargs + the arm, "
                     "and verified to reproduce each arm's own config.json):")
        for k in report["arm_derived_differences"]:
            lines.append(f"    {k}")
    if report.get("run_identity_differences"):
        lines.append("per-run identity fields (expected to differ): "
                     + ", ".join(report["run_identity_differences"]))
    if report.get("accepted_differences"):
        lines.append("ACCEPTED VIA --allow_config_diff (a real difference, not explained "
                     "by the arm):")
        for k, v in report["accepted_differences"].items():
            lines.append(f"    {k}: {v}")
    print_block("config agreement across arms  [HARD]", lines)


def print_eval_epoch(arm: str, rec: dict) -> None:
    ov = rec["overall"]
    lines = [
        f"val BCE        : {_fmt(ov['val_bce'])}   (element mean; batch-mean-of-means "
        f"{_fmt(ov['val_bce_batch_mean_of_means'])})",
        f"micro-AUPRC    : {_fmt(ov['micro_auprc'], '.6f')}   "
        f"macro-AUPRC    : {_fmt(ov['macro_auprc'], '.6f')} "
        f"({ov['macro_n_codes_included']} codes included, "
        f"{ov['macro_n_codes_excluded']} excluded)",
        f"recall@10/@20  : {_fmt(ov['recall@10'], '.4f')} / {_fmt(ov['recall@20'], '.4f')}"
        f"   nDCG@20 = {_fmt(ov['ndcg@20'], '.4f')}",
    ]
    lines.append("by age band:")
    for name in band_names():
        b = rec["by_band"].get(name, {})
        flag = "  UNRELIABLE" if b.get("unreliable") else ""
        lines.append(
            f"    {name:<6} n={b.get('n', 0):>7} n_pos={b.get('n_pos', 0):>9}  "
            f"bce={_fmt(b.get('val_bce'))}  microAUPRC={_fmt(b.get('micro_auprc'), '.4f')}  "
            f"r@10={_fmt(b.get('recall@10'), '.4f')}{flag}")
    gp = rec["diagnostics"]["gradient_probe"]["groups"]
    lines.append(
        "gradient probe : " + "  ".join(f"{k}={_fmt(v['grad_l2'], '.4e')}"
                                        for k, v in gp.items())
        + f"   age:backbone = {_fmt(rec['diagnostics']['gradient_probe']['age_over_backbone'], '.4e')}")
    for name, g in rec["diagnostics"]["generator_gradients"].items():
        lines.append(f"    [{name}] mode={g['mode']} nonzero-grad frac "
                     f"{g['frac_nonzero_grad_gt_0']:.4f} (>1e-12: "
                     f"{g['frac_nonzero_grad_gt_1e-12']:.4f})"
                     + ("   <-- ZERO AT A TRAINED CHECKPOINT"
                        if g["zero_gradient_at_trained_checkpoint"] else ""))
    for name, d in rec["diagnostics"]["delta_alpha_norms"].items():
        lines.append(f"    [{name}] ||d_alpha|| dense mean/max = "
                     f"{d['dense_uniform_grid']['mean']:.4e}/{d['dense_uniform_grid']['max']:.4e}"
                     f"   empirical mean/max = "
                     f"{d['empirical_validation']['mean']:.4e}/"
                     f"{d['empirical_validation']['max']:.4e}")
    ks = rec["diagnostics"]["kernel_separation"]["sites"]
    lines.append("kernel separation (centered / uncentered max pairwise |d log w|): "
                 + "  ".join(f"{n}={v['max_pairwise_centered']:.4e}/"
                             f"{v['max_pairwise_uncentered']:.4e}" for n, v in ks.items()))
    hr = rec["diagnostics"]["equal_norm_probe"]
    lines.append(f"equal-norm probe: max|d logit| = {hr['max_abs_delta_logit']:.6f}   "
                 f"max/logit sd = {hr['max_delta_over_logit_sd']:.4f}   "
                 f"(README 5 measured the same probe AT INIT: 0.0990 / 1.44 -- comparable "
                 f"in kind, not magnitude)")
    print_block(f"{arm}  epoch {rec['epoch']}  (step {rec.get('step')})", lines)


def print_selection(sel: dict) -> None:
    lines = [f"primary_rule : {sel['primary_rule']}  (declared on the command line)",
             "",
             f"{'rule':<18}" + "".join(f"{a:>22}" for a in sel["arms"])]
    for rule, chosen in sel["rules"].items():
        marker = " *" if rule == sel["primary_rule"] else "  "
        cells = []
        for a in sel["arms"]:
            if a not in chosen:      # the rule's anchor arm was not among --runs
                cells.append("--")
                continue
            e = chosen[a]
            v = sel["val_bce"][a].get(str(e))
            cells.append(f"ep {e} ({_fmt(v, '.6f')})")
        lines.append(f"{rule + marker:<18}" + "".join(f"{c:>22}" for c in cells))
    lines += [
        "",
        "val BCE is the full deterministic validation pass computed here (element mean over",
        "all (sequence, code) pairs), NOT train.json's --val_max_batches subset.",
        "* marks the rule written as primary_rule before any cross-arm number was printed.",
    ]
    print_block("epoch selection", lines)


def print_cross_arm_summary(summary: dict) -> None:
    arms = summary["arms"]
    rows = [
        ("epoch", "epoch", None),
        ("val BCE", "val_bce", ".6f"),
        ("micro-AUPRC", "micro_auprc", ".6f"),
        ("macro-AUPRC", "macro_auprc", ".6f"),
        ("recall@10", "recall@10", ".4f"),
        ("recall@20", "recall@20", ".4f"),
        ("nDCG@20", "ndcg@20", ".4f"),
        ("age:backbone grad", "age_over_backbone", ".3e"),
        ("kernel sep (centered)", "kernel_separation_centered", ".3e"),
        ("equal-norm max|dlogit|", "equal_norm_max_abs_delta_logit", ".5f"),
    ]
    lines = [f"rule: {summary['primary_rule']}", "",
             f"{'metric':<24}" + "".join(f"{a:>18}" for a in arms)]
    for label, key, spec in rows:
        cells = [_fmt(summary["table"][a].get(key), spec or ".6f") for a in arms]
        lines.append(f"{label:<24}" + "".join(f"{c:>18}" for c in cells))
    lines += ["", "kernel vs random_constant is the identifying comparison; vanilla is the",
              "floor; additive is a different architecture and is not parameter-matched."]
    print_block("cross-arm summary at primary_rule  [MEASURE]", lines)


def print_report_back(flags: Sequence[dict]) -> None:
    if not flags:
        print_block("report back", ["nothing to report: no config drift, no zero age "
                                    "gradients, no tau_max mismatch, no partially populated "
                                    "bands below the reliability threshold."])
        return
    lines = []
    for f in flags:
        lines.append(f"[{f['kind']}] {f['detail']}")
    lines += ["", "These are reported, not repaired. Each one is a decision for a human."]
    print_block("report back  [HARD to ignore]", lines)
