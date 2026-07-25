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
