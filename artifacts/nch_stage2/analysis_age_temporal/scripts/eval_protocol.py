"""Empirical NCH age×lag support and performance / truncation / ablation analysis."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model_new.data import DAYS_PER_YEAR
from stage1_mimic_pretrain.metrics import multilabel_metrics, ranking_per_example
from stage1_mimic_pretrain.model import MinimalDKMModel
from stage2_nch.analysis import ANALYSIS, ADKM_DIR, NINT_DIR, PRIMARY_CKPT_NAME
from stage2_nch.analysis.kernel_figures import AGE_BANDS, LAG_ANCHORS_DAYS, lag_to_tau
from stage2_nch.config import (
    EVAL_KS,
    PEDIATRIC_AGE_CENTER_YEARS,
    PEDIATRIC_AGE_SCALE_YEARS,
    VOCAB_PATH,
    EMBEDDING_PATH,
    age_band_name,
)
from stage2_nch.dataset import NCHForecastDataset, make_nch_collate
from stage2_nch.metrics import pos_neg_bce

HORIZONS_DAYS = [30, 90, 180, 365, 1095, None]  # None = full
HORIZON_LABELS = ["30d", "90d", "180d", "1y", "3y", "full"]
HISTORY_BINS = [
    ("<30d", 0, 30),
    ("30-90d", 30, 90),
    ("90-180d", 90, 180),
    ("180d-1y", 180, 365),
    ("1-3y", 365, 1095),
    (">3y", 1095, 1e9),
]
BOOT_SEED = 0
N_BOOT = 500


def _savefig(fig, stem: str, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_model_from_ckpt(ckpt_path: Path, device: torch.device) -> MinimalDKMModel:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    emb = sd["embedding_table"]
    num_codes = int(emb.shape[0]) - 2
    age_mean = float(sd["temporal.age_mean"])
    age_sd = float(sd["temporal.age_sd"])
    arm = ck.get("arm") or ck.get("config", {}).get("arm") or "age_temporal"
    cfg = ck.get("config") or {}
    model_cfg = cfg.get("model") or {}
    model = MinimalDKMModel(
        num_codes=num_codes,
        embedding_table=emb.float(),
        arm=arm,
        seed=int(ck.get("seed", 0)),
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 1)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        use_residual=bool(model_cfg.get("use_residual", True)),
        use_layernorm=bool(model_cfg.get("use_layernorm", True)),
        use_ffn=bool(model_cfg.get("use_ffn", True)),
        ffn_mult=int(model_cfg.get("ffn_mult", 4)),
        demo_dim=int(model_cfg.get("demo_dim", 9)),
        demo_hidden=int(model_cfg.get("demo_hidden", 64)),
        age_mean=age_mean,
        age_sd=age_sd,
        pool_temporal_bias=bool(model_cfg.get("pool_temporal_bias", False)),
        task="pretrain",
    )
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def make_test_loader(tensorized_dir: Path, batch_size: int = 8, num_workers: int = 0):
    ds = NCHForecastDataset(tensorized_dir / "test", VOCAB_PATH, max_seq_len=1024)
    collate = make_nch_collate(race_encoding="one_hot", assert_horizon=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate)
    return ds, loader


def truncate_item(item: dict, horizon_days: float | None) -> dict:
    """Keep events with lag from last timestamp <= horizon; preserve contract otherwise."""
    if horizon_days is None:
        return item
    ts = np.asarray(item["timestamps_days"], dtype=np.float64)
    if ts.size == 0:
        return item
    t_last = ts[-1]
    keep = (t_last - ts) <= float(horizon_days)
    if keep.all():
        return item
    if not keep.any():
        keep = np.zeros_like(keep, dtype=bool)
        keep[-1] = True  # always keep at least the last event
    out = dict(item)
    for key in ("code_indices", "timestamps_days", "age_days"):
        out[key] = np.asarray(item[key])[keep]
    out["n_input_events"] = int(out["code_indices"].shape[0])
    return out


@torch.no_grad()
def collect_window_rows(
    model: MinimalDKMModel,
    ds: NCHForecastDataset,
    device: torch.device,
    *,
    horizon_days: float | None = None,
    age_mode: str = "natural",  # natural | permute | constant
    constant_age: float = 9.0,
    permute_seed: int = 0,
    max_examples: int = 0,
    batch_size: int = 8,
    store_logits: bool = False,
    logit_cap: int = 8000,
) -> tuple[pd.DataFrame, torch.Tensor | None, torch.Tensor | None]:
    """Per-window metrics on the held-out test set (optionally truncated / age-perturbed)."""
    n = len(ds) if not max_examples else min(len(ds), max_examples)
    indices = list(range(n))
    # Precompute permuted last ages if needed
    natural_ages = []
    for i in indices:
        it = ds[i]
        natural_ages.append(float(it["last_age_years"]))
    natural_ages = np.asarray(natural_ages, dtype=np.float64)
    if age_mode == "permute":
        rng = np.random.default_rng(permute_seed)
        perm_ages = natural_ages.copy()
        rng.shuffle(perm_ages)
    else:
        perm_ages = natural_ages

    rows = []
    logit_chunks, target_chunks = [], []
    n_stored = 0
    collate = make_nch_collate(race_encoding="one_hot", assert_horizon=False)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        items = []
        meta = []
        for j, i in enumerate(batch_idx):
            item = truncate_item(ds[i], horizon_days)
            # Age perturbation on conditioning ages only (leave codes/times/labels)
            if age_mode == "constant":
                item = dict(item)
                item["age_days"] = np.full_like(item["age_days"], constant_age * DAYS_PER_YEAR)
                item["last_age_years"] = float(constant_age)
            elif age_mode == "permute":
                item = dict(item)
                # broadcast permuted last-age onto all positions (same protocol as age shuffle tests)
                item["age_days"] = np.full_like(item["age_days"], perm_ages[start + j] * DAYS_PER_YEAR)
                item["last_age_years"] = float(perm_ages[start + j])
            items.append(item)
            ts = np.asarray(item["timestamps_days"], dtype=np.float64)
            hist_span = float(ts[-1] - ts[0]) if ts.size else 0.0
            meta.append({
                "example_idx": i,
                "patient_id": int(item["patient_id"]),
                "age_years_natural": float(natural_ages[start + j]),
                "age_years_used": float(item["last_age_years"]),
                "age_band": age_band_name(float(natural_ages[start + j])),
                "n_input_events": int(item["n_input_events"]),
                "history_span_days": hist_span,
                "n_prior_visits": int(item.get("n_prior_visits", 0)),
            })
        batch = collate(items)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(batch)
        logits = out["code_logits"].float()
        targets = batch["target_codes"].float()
        per_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
        rank = ranking_per_example(logits, targets, ks=EVAL_KS)
        probs = torch.sigmoid(logits)
        # Brier (multilabel mean)
        brier = ((probs - targets) ** 2).mean(dim=-1)
        for j, m in enumerate(meta):
            row = {
                **m,
                "bce": float(per_bce[j].cpu()),
                "brier": float(brier[j].cpu()),
                "n_true": float(rank["n_true"][j]),
            }
            for k in EVAL_KS:
                row[f"recall@{k}"] = float(rank[f"recall@{k}"][j])
                row[f"precision@{k}"] = float(rank[f"precision@{k}"][j])
            rows.append(row)
        if store_logits and n_stored < logit_cap:
            take = min(logits.shape[0], logit_cap - n_stored)
            logit_chunks.append(logits[:take].detach().cpu())
            target_chunks.append(targets[:take].detach().cpu())
            n_stored += take

    df = pd.DataFrame(rows)
    logits_cat = torch.cat(logit_chunks) if logit_chunks else None
    targets_cat = torch.cat(target_chunks) if target_chunks else None
    return df, logits_cat, targets_cat


def history_bin_span(span_days: float) -> str:
    for name, lo, hi in HISTORY_BINS:
        if lo <= span_days < hi:
            return name
    return HISTORY_BINS[-1][0]


def patient_bootstrap_ci(
    df: pd.DataFrame,
    metric: str,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Mean of window-level metric, CI by resampling patients."""
    patients = df["patient_id"].unique()
    rng = np.random.default_rng(seed)
    point = float(df[metric].mean())
    boots = []
    pid_to_idx = {p: np.where(df["patient_id"].to_numpy() == p)[0] for p in patients}
    for _ in range(n_boot):
        samp = rng.choice(patients, size=len(patients), replace=True)
        idxs = np.concatenate([pid_to_idx[p] for p in samp])
        boots.append(float(df.iloc[idxs][metric].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_patients": int(len(patients)),
            "n_windows": int(len(df))}


def patient_bootstrap_metric_from_logits(
    df: pd.DataFrame,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """micro AUPRC/AUROC with patient-level bootstrap (rows aligned to logit cap prefix)."""
    n = min(len(df), logits.shape[0])
    df = df.iloc[:n].reset_index(drop=True)
    logits = logits[:n]
    targets = targets[:n]
    patients = df["patient_id"].unique()
    pid_to_idx = {p: np.where(df["patient_id"].to_numpy() == p)[0] for p in patients}
    base = multilabel_metrics(logits, targets, ks=EVAL_KS)
    rng = np.random.default_rng(seed)
    auprc_b, auroc_b = [], []
    for _ in range(n_boot):
        samp = rng.choice(patients, size=len(patients), replace=True)
        idxs = np.concatenate([pid_to_idx[p] for p in samp])
        m = multilabel_metrics(logits[idxs], targets[idxs], ks=EVAL_KS)
        auprc_b.append(m["micro_auprc"])
        auroc_b.append(m["micro_auroc"])
    return {
        "micro_auprc": base["micro_auprc"],
        "micro_auprc_ci": [float(np.percentile(auprc_b, 2.5)), float(np.percentile(auprc_b, 97.5))],
        "micro_auroc": base["micro_auroc"],
        "micro_auroc_ci": [float(np.percentile(auroc_b, 2.5)), float(np.percentile(auroc_b, 97.5))],
        "macro_auprc": base.get("macro_auprc"),
        "macro_auroc": base.get("macro_auroc"),
        "n_patients": int(len(patients)),
        "n_windows_in_logit_cap": int(n),
    }


def compute_age_lag_support(ds: NCHForecastDataset, max_examples: int = 0) -> dict:
    """Event-level density over (query age≈event age, lag-to-last) for support overlay."""
    age_edges = np.arange(0.0, 18.25, 0.25)
    lag_edges = np.unique(np.concatenate([
        np.array([0.0]),
        np.geomspace(1.0, 3650, 40),
        LAG_ANCHORS_DAYS.astype(float),
    ]))
    lag_edges.sort()
    counts = np.zeros((len(age_edges) - 1, len(lag_edges) - 1), dtype=np.int64)
    n = len(ds) if not max_examples else min(len(ds), max_examples)
    n_patients = set()
    for i in range(n):
        item = ds[i]
        n_patients.add(int(item["patient_id"]))
        ages = np.asarray(item["age_days"], dtype=np.float64) / DAYS_PER_YEAR
        ts = np.asarray(item["timestamps_days"], dtype=np.float64)
        if ts.size == 0:
            continue
        lags = ts[-1] - ts
        ai = np.clip(np.searchsorted(age_edges, ages, side="right") - 1, 0, counts.shape[0] - 1)
        li = np.clip(np.searchsorted(lag_edges, lags, side="right") - 1, 0, counts.shape[1] - 1)
        for a, l in zip(ai, li):
            counts[a, l] += 1
    return {
        "age_edges": age_edges,
        "lag_edges": lag_edges,
        "counts": counts,
        "n_windows": n,
        "n_patients": len(n_patients),
    }


def plot_support(support: dict, fig_dir: Path, mask_on: np.ndarray | None = None,
                 age_grid=None, lag_grid=None, Kmat=None):
    counts = support["counts"].astype(np.float64)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(np.log1p(counts), aspect="auto", origin="lower", cmap="magma",
                   extent=[0, counts.shape[1] - 1, support["age_edges"][0], support["age_edges"][-1]])
    ax.set_ylabel("Event age (years)")
    ax.set_xlabel("Lag bin index (log-spaced days to last event)")
    ax.set_title("NCH empirical age×lag event density (log1p counts)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="log1p(event count)")
    fig.tight_layout()
    _savefig(fig, "fig_age_lag_support_heatmap", fig_dir)

    if Kmat is not None and age_grid is not None:
        # Mask low-support regions on K heatmap
        # Map support counts onto K grid roughly by nearest bin
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        vmax = np.nanmax(np.abs(Kmat))
        im = ax.imshow(Kmat, aspect="auto", origin="lower", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax,
                       extent=[0, Kmat.shape[1] - 1, age_grid[0], age_grid[-1]])
        # low support contour from coarsened counts
        low = counts < max(1, np.percentile(counts[counts > 0], 10) if (counts > 0).any() else 1)
        if low.any():
            ax.contour(
                np.linspace(0, Kmat.shape[1] - 1, low.shape[1]),
                np.linspace(age_grid[0], age_grid[-1], low.shape[0]),
                low.astype(float), levels=[0.5], colors="k", linewidths=0.6,
            )
        ax.set_title(r"$K(a,\Delta t)$ with low-support contour overlay")
        ax.set_ylabel("Age (years)")
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$K$")
        fig.tight_layout()
        _savefig(fig, "fig_age_lag_kernel_with_support", fig_dir)
