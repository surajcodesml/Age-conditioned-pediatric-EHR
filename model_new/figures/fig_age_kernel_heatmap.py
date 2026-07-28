#!/usr/bin/env python3
"""Age-conditioning Chebyshev-kernel heatmap (AAAI-27 results).

Panel A plots the **softmax-valid** kernel deviation

    Δ(τ, a) = centered(log w_cond(τ̃, a)) − centered(log w_pop(τ̃))

where centering subtracts the per-row (per-age) mean over the τ grid. Softmax is
invariant to a per-row additive constant, and within one attention row the query age
is fixed, so the uncentered quantity contains a component that cannot affect any
attention weight.

Encoder-site and pooling-site kernels own separate parameters; this script never
averages or conflates them. The encoder figure is the main output; pooling is a
sibling PDF.

    python -m model_new.figures.fig_age_kernel_heatmap \\
        --run_dir model_new/run/<name> --ckpt epoch_NNN.pt --eval_split pic_test
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model_new import diagnostics as D
from model_new.basis import chebyshev_basis
from model_new.data import (
    DAYS_PER_YEAR,
    WEEK_DAYS,
    TensorizedPretrainDataset,
    lag_to_tau,
    pairwise_tau,
    sample_empirical_taus,
    spans_to_tau,
    tau_to_now_from_timestamps,
)
from model_new.data_finetune import TensorizedFinetuneDataset
from model_new.eval_finetune import pic_lag_sample
from model_new.eval_pretrain import build_model, model_kwargs_from_config

__all__ = [
    "AGE_TICKS_YEARS",
    "LAG_TICKS_DAYS",
    "N_AGE",
    "N_LAG",
    "PANEL_C_AGES",
    "REF_AGE_CAP",
    "band_midpoint",
    "build_age_grid",
    "build_lag_grid_days",
    "compute_figure_data",
    "centroid_lookback_days",
    "days_from_tau",
    "panel_a_delta",
    "panel_b_realized",
    "tau_tilde_from_days",
    "youngest_oldest_midpoints",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "out"

N_AGE = 300
N_LAG = 400
LAG_MIN_DAYS = 1.0 / 24.0          # 1 hour
LAG_MAX_DAYS = 10.0 * DAYS_PER_YEAR  # 10 years
AGE_MAX_YEARS = 90.0
REF_AGE_CAP = 90.0                 # finite stand-in for AGE_BANDS "65+" upper edge
PANEL_C_AGES = (0.5, 2.0, 8.0, 15.0, 40.0)
AGE_TICKS_YEARS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0, 40.0, 90.0)
LAG_TICKS_DAYS = (
    ("1h", 1.0 / 24.0),
    ("1d", 1.0),
    ("1w", 7.0),
    ("1mo", 30.0),
    ("1y", DAYS_PER_YEAR),
    ("5y", 5.0 * DAYS_PER_YEAR),
)
TAU_MAX_TOL = 1e-6
FIG_W = 7.0
FIG_H = 2.55
N_CONTOUR = 5


# --------------------------------------------------------------------------- #
# Grids and τ̃                                                                #
# --------------------------------------------------------------------------- #
def build_age_grid(n: int = N_AGE, a_max: float = AGE_MAX_YEARS) -> np.ndarray:
    """``n`` ages evenly spaced in ``u = log(1+a)`` over ``a ∈ [0, a_max]``."""
    u = np.linspace(0.0, math.log1p(a_max), int(n), dtype=np.float64)
    return np.expm1(u)


def build_lag_grid_days(n: int = N_LAG,
                        lo: float = LAG_MIN_DAYS,
                        hi: float = LAG_MAX_DAYS) -> np.ndarray:
    """``n`` lags log-spaced from ``lo`` to ``hi`` days."""
    return np.geomspace(lo, hi, int(n), dtype=np.float64)


def days_from_tau(tau: np.ndarray | float) -> np.ndarray | float:
    """Inverse of ``log1p(days/7)``: ``days = 7 · expm1(τ)``."""
    return WEEK_DAYS * np.expm1(tau)


def tau_tilde_from_days(days: np.ndarray | torch.Tensor, tau_max: float) -> torch.Tensor:
    """Day-lags → τ via ``data.lag_to_tau``, then τ̃ with the checkpoint's ``τ_max`` clip.

    Matches ``ChebyshevKernel.rescale(..., count=False)`` bit-for-bit on the τ path
    (acceptance #2 asserts equality against ``lag_to_tau`` + the same clip).
    """
    if isinstance(days, np.ndarray):
        t = torch.as_tensor(days, dtype=torch.float64)
    else:
        t = days.to(dtype=torch.float64)
    tau = lag_to_tau(t)
    tau_tilde = 2.0 * tau / float(tau_max) - 1.0
    return tau_tilde.clamp(-1.0, 1.0)


def band_midpoint(lo: float, hi: float, *, cap: float = REF_AGE_CAP) -> float:
    hi_eff = float(hi) if math.isfinite(hi) else float(cap)
    hi_eff = min(hi_eff, float(cap))
    return 0.5 * (float(lo) + hi_eff)


def youngest_oldest_midpoints(
    bands: tuple[tuple[str, float, float], ...] | None = None,
    *,
    cap: float = REF_AGE_CAP,
) -> tuple[float, float, str, str]:
    table = D.resolve_bands(bands)
    y_name, y_lo, y_hi = table[0]
    o_name, o_lo, o_hi = table[-1]
    return (
        band_midpoint(y_lo, y_hi, cap=cap),
        band_midpoint(o_lo, o_hi, cap=cap),
        y_name,
        o_name,
    )


# --------------------------------------------------------------------------- #
# Panel A                                                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def panel_a_delta(
    alpha_base: torch.Tensor,
    delta_alpha: torch.Tensor,
    tau_tilde: torch.Tensor,
) -> np.ndarray:
    """Softmax-valid Δ(τ, a) on a dense (age × τ) grid.

    ``alpha_base``: ``[s]``. ``delta_alpha``: ``[A, s]``. ``tau_tilde``: ``[T]``.
    Returns ``Δ`` shaped ``[A, T]`` (row-centered log-w difference). Setting
    ``delta_alpha ≡ 0`` yields an identically-zero plane (INV-HEATMAP-ZERO).
    """
    s = int(alpha_base.shape[-1])
    basis = chebyshev_basis(tau_tilde.to(dtype=torch.float32), s)  # [T, s]
    ab = alpha_base.to(dtype=torch.float32).reshape(1, 1, s)
    da = delta_alpha.to(dtype=torch.float32).unsqueeze(1)          # [A, 1, s]
    # logw_cond [A, T], logw_pop [T]
    logw_cond = ((ab + da) * basis.unsqueeze(0)).sum(dim=-1)
    logw_pop = (alpha_base.to(dtype=torch.float32).reshape(1, s) * basis).sum(dim=-1)
    cen_cond = logw_cond - logw_cond.mean(dim=1, keepdim=True)
    cen_pop = logw_pop - logw_pop.mean()
    return (cen_cond - cen_pop.unsqueeze(0)).cpu().numpy()


@torch.no_grad()
def site_delta_alpha(site, ages: torch.Tensor) -> torch.Tensor:
    """``Δα(a)`` for one kernel site (encoder or pooling)."""
    return site.age(ages)


@torch.no_grad()
def site_panel_a(site, ages_years: np.ndarray, tau_tilde: torch.Tensor) -> np.ndarray:
    a = torch.as_tensor(ages_years, dtype=torch.float32)
    da = site_delta_alpha(site, a)
    return panel_a_delta(site.alpha_base.detach(), da, tau_tilde)


# --------------------------------------------------------------------------- #
# Panel B                                                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _row_max_abs_delta_logit(
    tau_row: torch.Tensor,
    valid: torch.Tensor,
    alpha_true: torch.Tensor,
    alpha_refs: list[torch.Tensor],
    kernel,
) -> float:
    """Within-row softmax-valid max|Δlogit| vs reference ages.

    Differences are centered over valid keys before taking the max absolute value,
    matching Panel A's softmax rationale.
    """
    if int(valid.sum()) < 2:
        return float("nan")
    t = tau_row
    v = valid.bool()
    log_true = kernel(t, alpha_true, count=False)
    best = 0.0
    for a_ref in alpha_refs:
        log_ref = kernel(t, a_ref, count=False)
        d = log_true - log_ref
        d = d - d[v].mean()
        best = max(best, float(d[v].abs().max()))
    return best


@torch.no_grad()
def panel_b_realized(
    site,
    *,
    site_kind: str,
    batches: list[dict],
    bands: tuple[tuple[str, float, float], ...] | None = None,
) -> dict[str, Any]:
    """Bin attention rows by (AGE_BANDS × τ-spread quartile); mean max|Δlogit| + n."""
    table = D.resolve_bands(bands)
    a_young, a_old, y_name, o_name = youngest_oldest_midpoints(table)
    ages_ref = torch.tensor([a_young, a_old], dtype=torch.float32)
    alpha_refs = [
        (site.alpha_base + site.age(ages_ref[i : i + 1])).squeeze(0)
        for i in range(2)
    ]

    spreads: list[float] = []
    max_d: list[float] = []
    band_idx: list[int] = []

    for batch in batches:
        mask = batch["attention_mask"].bool()
        ages = batch["age_years"].float()
        if site_kind == "encoder":
            tau = pairwise_tau(batch["timestamps_days"], mask)  # [B, L, L]
            bsz, length, _ = tau.shape
            for bi in range(bsz):
                row_valid = mask[bi]
                if int(row_valid.sum()) < 2:
                    continue
                for qi in range(length):
                    if not bool(row_valid[qi]):
                        continue
                    t_row = tau[bi, qi]
                    spread = float(t_row[row_valid].max() - t_row[row_valid].min())
                    a_q = ages[bi, qi : qi + 1]
                    alpha_true = (site.alpha_base + site.age(a_q)).squeeze(0)
                    md = _row_max_abs_delta_logit(
                        t_row, row_valid, alpha_true, alpha_refs, site.kernel,
                    )
                    if not math.isfinite(md):
                        continue
                    spreads.append(spread)
                    max_d.append(md)
                    band_idx.append(int(D.band_index(a_q, table)[0]))
        elif site_kind == "pooling":
            tau_now = tau_to_now_from_timestamps(
                batch["timestamps_days"], mask, batch.get("lengths"),
            )
            bsz, length = tau_now.shape
            last = mask.long().sum(dim=1) - 1
            for bi in range(bsz):
                row_valid = mask[bi]
                if int(row_valid.sum()) < 2:
                    continue
                t_row = tau_now[bi]
                spread = float(t_row[row_valid].max() - t_row[row_valid].min())
                li = int(last[bi])
                a_q = ages[bi, li : li + 1]
                alpha_true = (site.alpha_base + site.age(a_q)).squeeze(0)
                md = _row_max_abs_delta_logit(
                    t_row, row_valid, alpha_true, alpha_refs, site.kernel,
                )
                if not math.isfinite(md):
                    continue
                spreads.append(spread)
                max_d.append(md)
                band_idx.append(int(D.band_index(a_q, table)[0]))
        else:
            raise ValueError(f"site_kind must be 'encoder' or 'pooling', got {site_kind!r}")

    spreads_a = np.asarray(spreads, dtype=np.float64)
    max_d_a = np.asarray(max_d, dtype=np.float64)
    band_a = np.asarray(band_idx, dtype=np.int64)
    n_bands = len(table)
    n_q = 4
    means = np.full((n_bands, n_q), np.nan, dtype=np.float64)
    counts = np.zeros((n_bands, n_q), dtype=np.int64)
    edges: list[float] = []
    if spreads_a.size:
        edges = [float(x) for x in np.quantile(spreads_a, [0.0, 0.25, 0.5, 0.75, 1.0])]
        # np.quantile unique-edge guard for degenerate spreads
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        q_idx = np.digitize(spreads_a, edges[1:-1], right=False)
        q_idx = np.clip(q_idx, 0, n_q - 1)
        for b in range(n_bands):
            for q in range(n_q):
                sel = (band_a == b) & (q_idx == q)
                counts[b, q] = int(sel.sum())
                if counts[b, q] > 0:
                    means[b, q] = float(max_d_a[sel].mean())
    return {
        "band_names": [n for n, _, _ in table],
        "quartile_labels": ["Q1", "Q2", "Q3", "Q4"],
        "spread_quartile_edges": edges,
        "mean_max_abs_delta_logit": means.tolist(),
        "n": counts.tolist(),
        "reference_ages": {
            "youngest_band": y_name,
            "oldest_band": o_name,
            "youngest_midpoint": a_young,
            "oldest_midpoint": a_old,
        },
        "n_rows": int(spreads_a.size),
    }


# --------------------------------------------------------------------------- #
# Panel C                                                                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def centroid_lookback_days(
    site,
    ages: np.ndarray,
    tau_grid: np.ndarray,
    p_emp: np.ndarray,
) -> np.ndarray:
    """``centroid(a) = Σ p_emp(τ) w(τ|a) τ / Σ p_emp(τ) w(τ|a)``, returned in days."""
    p = np.asarray(p_emp, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    if p.sum() <= 0:
        return np.full(len(ages), np.nan, dtype=np.float64)
    p = p / p.sum()
    tau_t = torch.as_tensor(tau_grid, dtype=torch.float32)
    a_t = torch.as_tensor(ages, dtype=torch.float32)
    alpha = site.alpha_base + site.age(a_t)  # [A, s]
    log_w = site.kernel(tau_t.unsqueeze(0), alpha, count=False)  # [A, T]
    w = log_w.exp().cpu().numpy() * p.reshape(1, -1)
    num = (w * tau_grid.reshape(1, -1)).sum(axis=1)
    den = w.sum(axis=1)
    cen_tau = np.where(den > 0, num / den, np.nan)
    return np.asarray(days_from_tau(cen_tau), dtype=np.float64)


def empirical_tau_density(tau_samples: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """Histogram density of empirical τ on the Panel-A τ grid (no smoothing)."""
    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    if tau_grid.size < 2:
        return np.ones_like(tau_grid) / max(1, tau_grid.size)
    mids = 0.5 * (tau_grid[:-1] + tau_grid[1:])
    edges = np.concatenate([
        [tau_grid[0] - (mids[0] - tau_grid[0])],
        mids,
        [tau_grid[-1] + (tau_grid[-1] - mids[-1])],
    ])
    counts, _ = np.histogram(np.asarray(tau_samples, dtype=np.float64), bins=edges)
    total = counts.sum()
    if total == 0:
        return np.ones_like(tau_grid) / tau_grid.size
    return counts.astype(np.float64) / total


# --------------------------------------------------------------------------- #
# Corpus overlays                                                             #
# --------------------------------------------------------------------------- #
def _tau_pct_from_corpus_stats(cs: dict) -> dict[str, float]:
    tq = cs.get("tau_quantiles") or {}
    out = {}
    for key in ("0.05", "0.5", "0.95"):
        if key not in tq:
            raise KeyError(f"corpus_stats.tau_quantiles missing {key!r}")
        out[key] = float(tq[key])
    return out


def _age_hist_from_corpus_stats(cs: dict) -> dict[str, Any]:
    h = cs.get("event_age_histogram") or {}
    return {
        "edges": [float(x) for x in h.get("edges", [])],
        "counts": [int(x) for x in h.get("counts", [])],
        "fractions": [float(x) for x in h.get("fractions", [])],
    }


def mimic_overlay_from_config(cfg: dict) -> dict[str, Any]:
    cs = cfg["data"]["corpus_stats"]
    return {
        "name": "MIMIC-IV",
        "tau_percentiles": _tau_pct_from_corpus_stats(cs),
        "age_histogram": _age_hist_from_corpus_stats(cs),
        "source": "config.json:data.corpus_stats",
    }


def _pic_cache_path(out_dir: Path, task: str, split: str) -> Path:
    return out_dir / f"cache_pic_lag_stats_{task}_{split}.json"


def pic_overlay(
    ds: TensorizedFinetuneDataset,
    *,
    out_dir: Path,
    task: str,
    split: str,
    n_windows: int = 2000,
    max_pairs: int = 50_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Pairwise-lag percentiles + age marginal for PIC; cached to JSON for cheap reruns."""
    cache = _pic_cache_path(out_dir, task, split)
    if cache.is_file():
        with cache.open("r", encoding="utf-8") as f:
            return json.load(f)

    taus = pic_lag_sample(ds, n_windows=n_windows, max_pairs=max_pairs, seed=seed)
    # Age marginal: sample event ages from a bounded set of windows.
    rng = np.random.default_rng(seed)
    ages: list[np.ndarray] = []
    from model_new.data import _sample_indices
    for j in _sample_indices(len(ds), min(n_windows, len(ds)), seed):
        item = ds[int(j)]
        a = item["age_days"].astype(np.float64) / DAYS_PER_YEAR
        if a.size:
            ages.append(a)
    age_arr = np.concatenate(ages) if ages else np.zeros(0)
    edges = np.array([0, 1, 6, 12, 18, 40, 65, 200], dtype=np.float64)
    counts, _ = np.histogram(age_arr, bins=edges)
    total = int(counts.sum()) or 1
    pct = {
        "0.05": float(np.percentile(taus, 5)) if taus.size else float("nan"),
        "0.5": float(np.percentile(taus, 50)) if taus.size else float("nan"),
        "0.95": float(np.percentile(taus, 95)) if taus.size else float("nan"),
    }
    payload = {
        "name": "PIC",
        "task": task,
        "split": split,
        "tau_percentiles": pct,
        "age_histogram": {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "fractions": (counts / total).tolist(),
        },
        "n_tau_samples": int(taus.size),
        "n_age_samples": int(age_arr.size),
        "sample_windows": int(n_windows),
        "seed": int(seed),
        "source": "sampled+cached",
        "tau_samples_for_density": None,  # filled by caller when needed; not persisted large
    }
    # Persist without the raw samples (cheap). Density uses a fresh sample below.
    D.write_json(cache, payload)
    # Attach an in-memory tau sample for Panel C density (not written to cache).
    if taus.size > 200_000:
        taus = rng.choice(taus, 200_000, replace=False)
    payload = dict(payload)
    payload["tau_samples"] = taus
    return payload


# --------------------------------------------------------------------------- #
# Checkpoint / data loading                                                   #
# --------------------------------------------------------------------------- #
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_run(run_dir: Path, ckpt_name: str) -> tuple[Any, dict, dict, Path]:
    """Load model on CPU; assert checkpoint ``τ_max`` buffer agrees with ``config.json``."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    shared = model_kwargs_from_config(cfg)
    model = build_model(shared, arm=str(cfg["arm"]))
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model.cpu()

    tau_buf = float(model.tau_max)
    tau_cfg = float(cfg["model"]["tau_max"])
    if abs(tau_buf - tau_cfg) > TAU_MAX_TOL:
        raise AssertionError(
            f"[HARD] ChebyshevKernel.tau_max buffer={tau_buf!r} disagrees with "
            f"config.json model.tau_max={tau_cfg!r} (tol={TAU_MAX_TOL})"
        )
    tau_src = ckpt.get("tau_max_source") or cfg["data"].get("tau_max_source")
    meta = {
        "checkpoint_path": str(ckpt_path.resolve()),
        "tau_max": tau_buf,
        "tau_max_source": tau_src,
        "arm": str(cfg["arm"]),
        "epoch": int(ckpt.get("epoch", -1)),
        "git_sha": git_sha(),
    }
    return model, cfg, meta, ckpt_path


def resolve_eval_split(
    name: str,
    *,
    cfg: dict,
    pic_root: Path,
    pic_task: str,
    max_seq_len: int,
) -> tuple[str, Any, str]:
    """Return ``(kind, dataset, label)`` where kind ∈ {pic, mimic}."""
    parts = name.lower().split("_")
    if len(parts) != 2:
        raise ValueError(
            f"--eval_split must look like 'pic_test' or 'mimic_val', got {name!r}"
        )
    corpus, split = parts
    if corpus == "pic":
        split_dir = Path(pic_root) / pic_task / split
        ds = TensorizedFinetuneDataset(split_dir, max_seq_len=max_seq_len)
        return "pic", ds, f"pic/{pic_task}/{split}"
    if corpus == "mimic":
        root = Path(cfg["data"]["paths"]["tensorized_dir"])
        if not root.is_absolute():
            root = REPO_ROOT / root
        ds = TensorizedPretrainDataset(root / split, max_seq_len=max_seq_len)
        return "mimic", ds, f"mimic/{split}"
    raise ValueError(f"unknown corpus in --eval_split: {corpus!r}")


def iter_eval_batches(
    kind: str,
    ds,
    *,
    race_encoding: str,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> list[dict]:
    from torch.utils.data import DataLoader, Subset
    from model_new.data import make_collate, _sample_indices
    from model_new.data_finetune import make_finetune_collate

    n = min(len(ds), max_batches * batch_size)
    idx = _sample_indices(len(ds), n, seed)
    subset = Subset(ds, [int(i) for i in idx])
    if kind == "pic":
        collate = make_finetune_collate(race_encoding)
    else:
        collate = make_collate(race_encoding)
    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    )
    out: list[dict] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        # ensure CPU tensors
        out.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()})
    return out


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _set_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "axes.titlesize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _age_to_u(a: np.ndarray | float) -> np.ndarray | float:
    return np.log1p(a)


def _format_lag_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks([d for _, d in LAG_TICKS_DAYS])
    ax.set_xticklabels([lab for lab, _ in LAG_TICKS_DAYS])
    ax.set_xlabel("inter-event lag")


def _format_age_axis(ax, age_ticks: tuple[float, ...] = AGE_TICKS_YEARS) -> None:
    ax.set_yticks([_age_to_u(a) for a in age_ticks])
    ax.set_yticklabels([str(int(a)) if float(a).is_integer() else str(a)
                        for a in age_ticks])
    ax.set_ylabel("age (years)")


def render_figure(
    *,
    delta: np.ndarray,
    ages: np.ndarray,
    lag_days: np.ndarray,
    panel_b: dict,
    overlays: list[dict],
    out_pdf: Path,
    site_label: str,
    age_ticks: tuple[float, ...] | None = None,
    centroid_days: np.ndarray | None = None,  # unused; kept for call-site compat
    centroid_ages: np.ndarray | None = None,
) -> dict[str, float]:
    """Write one site PDF/PNG (Panels A + B only)."""
    del centroid_days, centroid_ages  # Panel C removed
    _set_rc()
    if age_ticks is None:
        a_max = float(ages.max()) if ages.size else AGE_MAX_YEARS
        age_ticks = tuple(a for a in AGE_TICKS_YEARS if a <= a_max + 1e-9)
        if not age_ticks:
            age_ticks = (0.0, a_max)
    u_ages = _age_to_u(ages)
    abs_d = np.abs(delta)
    q99 = float(np.quantile(abs_d, 0.99)) if abs_d.size else 1.0
    if q99 <= 0:
        q99 = 1e-12
    lim = q99

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    left, right, bottom, top = 0.08, 0.98, 0.20, 0.88
    w_gap = 0.04
    widths = np.array([0.58, 0.42])
    widths = widths / widths.sum() * (right - left - w_gap)
    ax_a = fig.add_axes([left, bottom, widths[0], top - bottom])
    ax_b = fig.add_axes([left + widths[0] + w_gap, bottom, widths[1], top - bottom])

    X, Y = np.meshgrid(lag_days, u_ages)
    pcm = ax_a.pcolormesh(
        X, Y, delta, cmap="RdBu_r", shading="auto",
        norm=Normalize(vmin=-lim, vmax=lim),
    )
    levels = np.linspace(-lim, lim, N_CONTOUR + 2)[1:-1]
    if np.any(np.abs(levels) > 0):
        ax_a.contour(
            X, Y, delta, levels=levels, colors="black", linewidths=0.35, alpha=0.7,
        )
    _format_lag_axis(ax_a)
    _format_age_axis(ax_a, age_ticks)
    ax_a.set_xlim(lag_days.min(), lag_days.max())
    ax_a.set_ylim(u_ages.min(), u_ages.max())
    ax_a.set_title(f"A  kernel deviation ({site_label})", loc="left", pad=2)

    styles = {
        "MIMIC-IV": dict(color="#1b4f72", ls="--"),
        "PIC": dict(color="#196f3d", ls=":"),
    }
    for oi, ov in enumerate(overlays):
        if ov["name"] != "PIC":
            continue  # PIC-focused figure
        st = styles.get(ov["name"], dict(color="0.3", ls="-."))
        for qk in ("0.05", "0.5", "0.95"):
            d_q = max(float(days_from_tau(float(ov["tau_percentiles"][qk]))),
                      float(lag_days.min()))
            ax_a.axvline(
                d_q, color=st["color"], linestyle=st["ls"], linewidth=0.7, alpha=0.85,
                zorder=5,
            )
        d50 = max(float(days_from_tau(float(ov["tau_percentiles"]["0.5"]))),
                  float(lag_days.min()))
        ax_a.text(
            d50, u_ages.max(), ov["name"], color=st["color"], fontsize=5.5,
            ha="center", va="bottom", clip_on=False,
        )
        _draw_age_strip_axes(
            ax_a, ov.get("age_histogram") or {}, color=st["color"], offset=0.0,
            age_max=float(ages.max()) if ages.size else AGE_MAX_YEARS,
        )

    cax = make_axes_locatable(ax_a).append_axes("top", size="6%", pad=0.08)
    cb = fig.colorbar(pcm, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.set_label("additive attention-logit bias (nats)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    means = np.asarray(panel_b["mean_max_abs_delta_logit"], dtype=np.float64)
    counts = np.asarray(panel_b["n"], dtype=np.int64)
    finite = means[np.isfinite(means)]
    vmax_b = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax_b <= 0:
        vmax_b = 1e-12
    im = ax_b.imshow(
        means, aspect="auto", origin="lower", cmap="Reds",
        vmin=0.0, vmax=vmax_b, interpolation="nearest",
    )
    ax_b.set_xticks(range(len(panel_b["quartile_labels"])))
    ax_b.set_xticklabels(panel_b["quartile_labels"])
    ax_b.set_yticks(range(len(panel_b["band_names"])))
    ax_b.set_yticklabels(panel_b["band_names"])
    ax_b.set_xlabel("τ-spread quartile")
    ax_b.set_title("B  realized max|Δlogit|", loc="left", pad=2)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            n = int(counts[i, j])
            val = means[i, j]
            txt = f"{val:.2f}\nn={n}" if n and math.isfinite(val) else f"n={n}"
            ax_b.text(j, i, txt, ha="center", va="center", fontsize=5.0, color="0.1")
    cax_b = make_axes_locatable(ax_b).append_axes("top", size="6%", pad=0.08)
    cb_b = fig.colorbar(im, cax=cax_b, orientation="horizontal")
    cax_b.xaxis.set_ticks_position("top")
    cb_b.set_label("mean max|Δlogit| (nats)", fontsize=7)
    cb_b.ax.tick_params(labelsize=6)

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"q99_abs_delta": lim, "vmax_panel_b": vmax_b}


def _draw_age_strip_axes(ax, hist: dict, *, color: str, offset: float,
                         age_max: float = AGE_MAX_YEARS) -> None:
    """Thin age-marginal strip in axes coordinates on the right edge of Panel A."""
    edges = np.asarray(hist.get("edges") or [], dtype=np.float64)
    fracs = np.asarray(hist.get("fractions") or [], dtype=np.float64)
    if edges.size < 2 or fracs.size != edges.size - 1:
        return
    mx = float(fracs.max()) if fracs.size else 1.0
    mx = mx if mx > 0 else 1.0
    base = 1.0 - 0.055 - offset
    width = 0.045
    y0, y1 = ax.get_ylim()
    for i in range(len(fracs)):
        lo, hi = float(edges[i]), float(min(edges[i + 1], age_max))
        if hi <= lo or lo >= age_max:
            continue
        u0, u1 = _age_to_u(lo), _age_to_u(hi)
        ya0 = (u0 - y0) / (y1 - y0)
        ya1 = (u1 - y0) / (y1 - y0)
        h = fracs[i] / mx * width
        ax.add_patch(Rectangle(
            (base, ya0), h, max(ya1 - ya0, 1e-4),
            transform=ax.transAxes, color=color, alpha=0.4, linewidth=0, clip_on=False,
            zorder=6,
        ))


# --------------------------------------------------------------------------- #
# Numbers JSON                                                                #
# --------------------------------------------------------------------------- #
def summarize_numbers(
    *,
    delta: np.ndarray,
    ages: np.ndarray,
    lag_days: np.ndarray,
    tau_grid: np.ndarray,
    pic_overlay: dict | None,
    panel_b: dict,
    centroid_at: dict[str, float],
    meta: dict,
    site_name: str,
) -> dict[str, Any]:
    abs_d = np.abs(delta)
    max_abs = float(abs_d.max()) if abs_d.size else float("nan")
    # PIC 5-95 lag band
    pic_band: dict[str, Any] = {}
    if pic_overlay is not None:
        t5 = float(pic_overlay["tau_percentiles"]["0.05"])
        t95 = float(pic_overlay["tau_percentiles"]["0.95"])
        d5, d95 = float(days_from_tau(t5)), float(days_from_tau(t95))
        lag_mask = (lag_days >= d5) & (lag_days <= d95)
        sub = abs_d[:, lag_mask] if lag_mask.any() else abs_d
        max_pic = float(sub.max()) if sub.size else float("nan")
        per_band = {}
        for name, lo, hi in D.AGE_BANDS:
            am = (ages >= lo) & (ages < (hi if math.isfinite(hi) else AGE_MAX_YEARS + 1))
            if not am.any() or not lag_mask.any():
                per_band[name] = float("nan")
                continue
            per_band[name] = float(abs_d[np.ix_(am, lag_mask)].max())
        pic_band = {
            "lag_days_5_95": [d5, d95],
            "tau_5_95": [t5, t95],
            "max_abs_delta": max_pic,
            "max_abs_delta_by_age_band": per_band,
        }
    return {
        "site": site_name,
        "max_abs_delta": max_abs,
        "pic_lag_band": pic_band,
        "panel_b": {
            "mean_max_abs_delta_logit": panel_b["mean_max_abs_delta_logit"],
            "n": panel_b["n"],
            "band_names": panel_b["band_names"],
            "quartile_labels": panel_b["quartile_labels"],
            "reference_ages": panel_b["reference_ages"],
        },
        "centroid_days_at_ages": centroid_at,
        "tau_max": meta["tau_max"],
        "tau_max_source": meta["tau_max_source"],
        "checkpoint_path": meta["checkpoint_path"],
        "git_sha": meta["git_sha"],
        "arm": meta["arm"],
        "epoch": meta["epoch"],
    }


# --------------------------------------------------------------------------- #
# Compute (no plotting) — notebooks own the matplotlib styling                #
# --------------------------------------------------------------------------- #
def compute_figure_data(
    run_dir: Path | str,
    ckpt: str,
    *,
    eval_split: str = "pic_test",
    pic_root: Path | str | None = None,
    pic_task: str = "pneumonia",
    out_dir: Path | str | None = None,
    batch_size: int = 16,
    max_batches: int = 40,
    pic_sample_windows: int = 2000,
    seed: int = 0,
    age_max: float | None = None,
    n_age: int = N_AGE,
    n_lag: int = N_LAG,
    band_table: str | None = None,
) -> dict[str, Any]:
    """Load checkpoint + eval split; return arrays/dicts for plotting.

    ``age_max`` defaults to 18 for PIC eval splits (pediatric support) and 90
    otherwise. ``band_table`` defaults to ``\"pediatric\"`` on PIC, ``\"adult\"``
    on MIMIC — used only for Panel B banding.
    """
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    pic_root = Path(pic_root) if pic_root is not None else (
        REPO_ROOT / "data" / "tensorized" / "pic"
    )

    model, cfg, meta, _ = load_run(Path(run_dir), ckpt)
    tau_max = float(meta["tau_max"])

    kind, eval_ds, eval_label = resolve_eval_split(
        eval_split, cfg=cfg, pic_root=pic_root,
        pic_task=pic_task, max_seq_len=int(cfg["data"]["max_seq_len"]),
    )
    if age_max is None:
        age_max = 18.0 if kind == "pic" else AGE_MAX_YEARS
    if band_table is None:
        band_table = "pediatric" if kind == "pic" else "adult"
    bands = D.resolve_bands(band_table)

    ages = build_age_grid(n_age, a_max=float(age_max))
    lag_days = build_lag_grid_days(n_lag)
    tau_tilde = tau_tilde_from_days(lag_days, tau_max)
    tau_grid = lag_to_tau(torch.as_tensor(lag_days, dtype=torch.float64)).numpy()

    batches = iter_eval_batches(
        kind, eval_ds,
        race_encoding=str(cfg["model"]["race_encoding"]),
        batch_size=int(batch_size),
        max_batches=int(max_batches),
        seed=int(seed),
    )

    mimic_ov = mimic_overlay_from_config(cfg)
    pic_ov = None
    if kind == "pic":
        pic_ov = pic_overlay(
            eval_ds, out_dir=out_dir, task=pic_task,
            split=eval_split.split("_", 1)[1],
            n_windows=int(pic_sample_windows), seed=int(seed),
        )
        if "tau_samples" in pic_ov and pic_ov["tau_samples"] is not None:
            tau_emp = np.asarray(pic_ov["tau_samples"], dtype=np.float64)
        else:
            tau_emp = pic_lag_sample(
                eval_ds, n_windows=int(pic_sample_windows),
                max_pairs=50_000, seed=int(seed),
            )
        overlays = [mimic_ov, {k: v for k, v in pic_ov.items() if k != "tau_samples"}]
    else:
        tau_emp = sample_empirical_taus(
            eval_ds, n_examples=int(pic_sample_windows), seed=int(seed),
        )
        overlays = [mimic_ov]

    p_emp = empirical_tau_density(tau_emp, tau_grid)

    sites = dict(model.kernel_sites())
    enc_name = sorted(n for n in sites if n.startswith("encoder_"))[0]
    pool_name = "pooling"

    # Panel-C report ages: keep those inside the plotted age range.
    panel_c_ages = tuple(a for a in PANEL_C_AGES if a <= float(age_max))
    if not panel_c_ages:
        panel_c_ages = (float(age_max) * 0.5,)

    per_site: dict[str, Any] = {}
    for site_name, site_kind in ((enc_name, "encoder"), (pool_name, "pooling")):
        site = sites[site_name]
        delta = site_panel_a(site, ages, tau_tilde)
        pb = panel_b_realized(
            site, site_kind=site_kind, batches=batches, bands=bands,
        )
        cen_curve = centroid_lookback_days(site, ages, tau_grid, p_emp)
        cen_pts = centroid_lookback_days(
            site, np.asarray(panel_c_ages, dtype=np.float64), tau_grid, p_emp,
        )
        centroid_at = {str(a): float(v) for a, v in zip(panel_c_ages, cen_pts)}
        per_site[site_name] = {
            "site_kind": site_kind,
            "delta": delta,
            "panel_b": pb,
            "centroid_days": cen_curve,
            "centroid_at": centroid_at,
            "numbers": summarize_numbers(
                delta=delta, ages=ages, lag_days=lag_days, tau_grid=tau_grid,
                pic_overlay=pic_ov, panel_b=pb, centroid_at=centroid_at,
                meta=meta, site_name=site_name,
            ),
        }

    return {
        "meta": meta,
        "eval_label": eval_label,
        "kind": kind,
        "age_max": float(age_max),
        "band_table": band_table,
        "ages": ages,
        "lag_days": lag_days,
        "tau_grid": tau_grid,
        "overlays": overlays,
        "encoder_site": enc_name,
        "pooling_site": pool_name,
        "sites": per_site,
        "n_batches": len(batches),
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    data = compute_figure_data(
        args.run_dir, args.ckpt,
        eval_split=args.eval_split,
        pic_root=args.pic_root,
        pic_task=args.pic_task,
        out_dir=out_dir,
        batch_size=int(args.batch_size),
        max_batches=int(args.max_batches),
        pic_sample_windows=int(args.pic_sample_windows),
        seed=int(args.seed),
    )
    ages = data["ages"]
    lag_days = data["lag_days"]
    overlays = data["overlays"]
    meta = data["meta"]
    enc_name = data["encoder_site"]
    pool_name = data["pooling_site"]

    site_specs = [
        (enc_name, out_dir / "fig_age_kernel_heatmap.pdf"),
        (pool_name, out_dir / "fig_age_kernel_heatmap_pooling.pdf"),
    ]
    numbers_by_site: dict[str, Any] = {}
    for site_name, pdf_path in site_specs:
        s = data["sites"][site_name]
        a_max = float(data["age_max"])
        if data["kind"] == "pic":
            age_ticks = tuple(a for a in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 18.0)
                              if a <= a_max + 1e-9)
        else:
            age_ticks = tuple(a for a in AGE_TICKS_YEARS if a <= a_max + 1e-9)
        if not age_ticks:
            age_ticks = (0.0, a_max)
        render_figure(
            delta=s["delta"], ages=ages, lag_days=lag_days, panel_b=s["panel_b"],
            overlays=overlays, out_pdf=pdf_path, site_label=site_name,
            age_ticks=age_ticks,
        )
        numbers_by_site[site_name] = s["numbers"]

    enc_nums = numbers_by_site[enc_name]
    payload = {
        **enc_nums,
        "eval_split": data["eval_label"],
        "pooling": numbers_by_site[pool_name],
        "encoder_site": enc_name,
        "age_max": data["age_max"],
        "band_table": data["band_table"],
        "note": (
            "Top-level max|Δ| / Panel B / centroids are the encoder-site figure. "
            "Pooling-site numbers live under 'pooling' and were never averaged with encoder."
        ),
    }
    D.write_json(out_dir / "age_kernel_heatmap_numbers.json", payload)

    D.print_block("fig_age_kernel_heatmap", [
        f"arm / epoch          : {meta['arm']} / {meta['epoch']}",
        f"tau_max              : {meta['tau_max']!r}",
        f"tau_max_source       : {meta['tau_max_source']!r}",
        f"checkpoint           : {meta['checkpoint_path']}",
        f"eval_split           : {data['eval_label']}  ({data['n_batches']} batches)",
        f"age_max / bands      : {data['age_max']} / {data['band_table']}",
        f"encoder max|Δ|       : {enc_nums['max_abs_delta']:.6g}",
        f"pooling max|Δ|       : {numbers_by_site[pool_name]['max_abs_delta']:.6g}",
        f"pdf (encoder)        : {out_dir / 'fig_age_kernel_heatmap.pdf'}",
        f"png (encoder)        : {out_dir / 'fig_age_kernel_heatmap.png'}",
        f"pdf (pooling)        : {out_dir / 'fig_age_kernel_heatmap_pooling.pdf'}",
        f"png (pooling)        : {out_dir / 'fig_age_kernel_heatmap_pooling.png'}",
        f"numbers              : {out_dir / 'age_kernel_heatmap_numbers.json'}",
        f"git_sha              : {meta['git_sha']}",
    ])
    return payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--ckpt", type=str, required=True, help="e.g. epoch_011.pt or best.pt")
    p.add_argument("--eval_split", type=str, default="pic_test",
                   help="corpus_split, e.g. pic_test or mimic_val")
    p.add_argument("--pic_root", type=Path,
                   default=REPO_ROOT / "data" / "tensorized" / "pic")
    p.add_argument("--pic_task", type=str, default="pneumonia")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=40,
                   help="bounded Panel-B sample; recorded indirectly via n_rows")
    p.add_argument("--pic_sample_windows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())