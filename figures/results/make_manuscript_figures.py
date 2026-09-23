#!/usr/bin/env python3
"""Manuscript figures for MIMIC / NCH age×temporal results.

Checkpoints (validation micro-AUPRC selection):
  MIMIC ADKM epoch_005.pt, MIMIC NINT epoch_005.pt
  NCH ADKM / NINT checkpoint_best_auprc.pt (epoch 4)

Temporal quantities are taken from AgeTemporalBias.lambda_of / pairwise_bias
and model_new.data.lag_to_tau — not re-derived by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_new.data import lag_to_tau
from stage1_mimic_pretrain.config import MIMIC_AGE_MEAN_YEARS, MIMIC_AGE_STD_YEARS, DAYS_PER_YEAR
from stage1_mimic_pretrain.metrics import _safe_auprc
from stage2_nch.analysis.eval_protocol import build_model_from_ckpt
from stage2_nch.config import VOCAB_PATH, age_band_name
from stage2_nch.dataset import NCHForecastDataset, make_nch_collate

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

# --- Selected checkpoints -------------------------------------------------
MIMIC_ADKM = REPO / "stage1_mimic_pretrain/run/adkm_s0/epoch_005.pt"
MIMIC_NINT = REPO / "stage1_mimic_pretrain/run/nint_s0/epoch_005.pt"
NCH_ADKM = REPO / "stage2_nch/run/adkm_nch_s0/checkpoint_best_auprc.pt"
NCH_NINT = REPO / "stage2_nch/run/nint_nch_s0/checkpoint_best_auprc.pt"

AGE_ORDER = ["<1", "1-5", "6-11", "12-17"]
AGE_TICK = ["<1", "1–5", "6–11", "12–17"]
# Available history duration (days)
HIST_BINS = [
    ("<3 months", 0.0, 90.0),
    ("3–12 months", 90.0, 365.0),
    ("1–3 years", 365.0, 1095.0),
    (">3 years", 1095.0, 1e12),
]
MIN_PATIENTS = 20
N_BOOT = 100
BOOT_SEED = 0
COLOR_ADKM = "#0B6E4F"
COLOR_NINT = "#C45C26"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.svg", flush=True)


def load_temporal(ckpt_path: Path):
    """Return (model.temporal module on CPU, meta dict)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Build via shared Stage-2 loader (works for Stage-1 blobs too).
    model = build_model_from_ckpt(ckpt_path, torch.device("cpu"))
    sd = ck["model_state_dict"]
    meta = {
        "path": str(ckpt_path),
        "epoch": ck.get("epoch"),
        "val_bce": ck.get("val_bce"),
        "val_micro_auprc": ck.get("val_micro_auprc"),
        "arm": ck.get("arm") or (ck.get("config") or {}).get("arm"),
        "lambda0": float(sd["temporal.lambda0"].reshape(-1)[0]),
        "beta": float(sd["temporal.beta"].reshape(-1)[0]),
        "age_mean": float(sd["temporal.age_mean"]),
        "age_sd": float(sd["temporal.age_sd"]),
    }
    return model.temporal, meta


# =============================================================================
# Figure 1 — mimic_lambda_age
# =============================================================================
def sample_mimic_event_ages(n_shards: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    root = REPO / "data/processed/tensorized_flat/train"
    shards = sorted(root.glob("shard_*.npz"))
    pick = rng.choice(len(shards), size=min(n_shards, len(shards)), replace=False)
    ages = []
    for i in pick:
        z = np.load(shards[int(i)])
        ages.append(z["age_days"].astype(np.float64) / DAYS_PER_YEAR)
    return np.concatenate(ages, axis=0)


def fig_mimic_lambda_age() -> dict:
    print("=== fig1 mimic_lambda_age ===", flush=True)
    t_adkm, m_adkm = load_temporal(MIMIC_ADKM)
    t_nint, m_nint = load_temporal(MIMIC_NINT)

    # Observed MIMIC event-age support from corpus stats + live sample
    cfg = json.loads((REPO / "stage1_mimic_pretrain/run/adkm_s0/config.json").read_text())
    cs = cfg["data"]["corpus_stats"]
    age_lo = float(cs["event_age_min"])
    age_hi = min(float(cs["event_age_max"]), 100.0)  # visual cap; ≥89 are censored-heavy
    ages = np.linspace(age_lo, age_hi, 400)

    with torch.no_grad():
        a_t = torch.tensor(ages, dtype=torch.float32)
        lam_adkm = t_adkm.lambda_of(a_t).cpu().numpy()
        lam_nint = t_nint.lambda_of(a_t).cpu().numpy()

    event_ages = sample_mimic_event_ages()
    event_ages = event_ages[(event_ages >= age_lo) & (event_ages <= age_hi)]

    pd.DataFrame({
        "age_years": ages,
        "lambda_adkm": lam_adkm,
        "lambda_nint": lam_nint,
        "z_adkm": ((ages - m_adkm["age_mean"]) / m_adkm["age_sd"]),
        "z_nint": ((ages - m_nint["age_mean"]) / m_nint["age_sd"]),
    }).to_csv(RAW / "mimic_lambda_age.csv", index=False)
    (RAW / "mimic_lambda_age_meta.json").write_text(json.dumps({
        "adkm": m_adkm, "nint": m_nint,
        "age_range_years": [age_lo, age_hi],
        "age_transform": "z(a)=(a-μ)/σ with frozen MIMIC train event-level μ,σ",
        "mu": MIMIC_AGE_MEAN_YEARS, "sigma": MIMIC_AGE_STD_YEARS,
        "n_event_ages_sampled": int(event_ages.size),
        "quantity": "λ(a)=λ0+β z(a) via AgeTemporalBias.lambda_of",
    }, indent=2) + "\n")

    _style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax2 = ax.twinx()
    ax2.hist(event_ages, bins=40, color="0.75", alpha=0.35, density=True)
    ax2.set_ylabel("Event-age density (train sample)", color="0.45")
    ax2.tick_params(axis="y", colors="0.45", labelsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("0.7")

    ax.plot(ages, lam_nint, color=COLOR_NINT, lw=1.8, label="NINT (constant)")
    ax.plot(ages, lam_adkm, color=COLOR_ADKM, lw=1.8, label="ADKM")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\lambda(a)$")
    ax.set_xlim(age_lo, age_hi)
    ax.legend(loc="best", frameon=False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    fig.tight_layout()
    save_fig(fig, "mimic_lambda_age")
    return {"adkm": m_adkm, "nint": m_nint, "age_lo": age_lo, "age_hi": age_hi,
            "n_hist": int(event_ages.size)}


# =============================================================================
# Figure 2 — nch_age_lag_heatmap
# =============================================================================
def kernel_bias_matrix(temporal, ages: np.ndarray, lags_days: np.ndarray) -> np.ndarray:
    """K(a, Δt) = −λ(a) τ(Δt) using model lambda_of and lag_to_tau."""
    with torch.no_grad():
        age_t = torch.tensor(ages, dtype=torch.float32)
        lag_t = torch.tensor(lags_days, dtype=torch.float32)
        lam = temporal.lambda_of(age_t)  # [A]
        tau = lag_to_tau(lag_t)          # [T]  — model_new.data definition
        # Equivalent to pairwise_bias for a single query age vs lag keys:
        #   pairwise_bias(tau_row, age_row)_j = −λ(a) τ_j
        K = (-lam.unsqueeze(1) * tau.unsqueeze(0)).cpu().numpy()
        # Sanity: match pairwise_bias on a probe row
        a0 = age_t[:1].unsqueeze(0).expand(1, lag_t.numel())  # [1, T]
        tau_mat = tau.view(1, 1, -1).expand(1, lag_t.numel(), lag_t.numel())
        # Use first query position only vs diagonal lags constructed as row 0 distances
        # Build tau such that tau[0, j] = tau(lag_j)
        tau_row = torch.zeros(1, 1, lag_t.numel())
        tau_row[0, 0, :] = tau
        age_row = age_t[:1].view(1, 1)
        probe = temporal.pairwise_bias(tau_row, age_row)[0, 0].cpu().numpy()
        assert np.allclose(probe, K[0], atol=1e-5), "pairwise_bias mismatch"
    return K


def fig_nch_age_lag_heatmap() -> dict:
    print("=== fig2 nch_age_lag_heatmap ===", flush=True)
    t_adkm, m_adkm = load_temporal(NCH_ADKM)
    t_nint, m_nint = load_temporal(NCH_NINT)

    ages = np.linspace(0.0, 18.0, 181)  # 0.1y
    # Log-spaced lag: 1 day → ~10y (covers NCH max history ~14y; 10y is clinically labeled)
    lags = np.geomspace(1.0, 3650.0, 120)

    K_adkm = kernel_bias_matrix(t_adkm, ages, lags)
    K_nint = kernel_bias_matrix(t_nint, ages, lags)
    K_diff = K_adkm - K_nint

    np.savez_compressed(
        RAW / "nch_age_lag_kernel.npz",
        age_years=ages, lag_days=lags,
        K_adkm=K_adkm, K_nint=K_nint, K_diff=K_diff,
    )
    (RAW / "nch_age_lag_meta.json").write_text(json.dumps({
        "adkm": m_adkm, "nint": m_nint,
        "quantity": "K(a,Δt)=−λ(a)·τ(Δt) with τ=log1p(|Δt|/7); via lambda_of + lag_to_tau",
        "age_unit": "years", "lag_unit": "days",
        "lag_grid": "geomspace(1, 3650, 120)",
        "age_grid": "linspace(0, 18, 181)",
    }, indent=2) + "\n")

    vmax = float(np.nanmax(np.abs(np.concatenate([K_adkm.ravel(), K_nint.ravel()]))))
    vmax = max(vmax, 1e-6)
    dmax = float(np.nanmax(np.abs(K_diff))) or 1e-6

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.12})
    lag_ticks = np.array([1, 7, 30, 90, 365, 365 * 3, 3650], dtype=float)
    lag_ticklabels = ["1d", "1w", "1m", "3m", "1y", "3y", "10y"]
    # Map lag values to indices for imshow extent
    extent = [np.log10(lags[0]), np.log10(lags[-1]), ages[0], ages[-1]]

    mats = [K_adkm, K_nint, K_diff]
    titles = ["ADKM", "NINT", "ADKM − NINT"]
    cmaps = ["magma", "magma", "coolwarm"]
    vmins = [-vmax, -vmax, -dmax]
    vmaxs = [vmax, vmax, dmax]

    ims = []
    for ax, mat, title, cmap, vmin, vmax_ in zip(axes, mats, titles, cmaps, vmins, vmaxs):
        # Display with log x via extent in log10 lag
        # resample columns onto uniform log10 grid already (lags are geomspaced)
        im = ax.imshow(
            mat, origin="lower", aspect="auto", cmap=cmap,
            vmin=vmin, vmax=vmax_,
            extent=extent, interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Event lag")
        xt = np.log10(lag_ticks)
        ax.set_xticks(xt)
        ax.set_xticklabels(lag_ticklabels, rotation=0)
        ax.set_xlim(extent[0], extent[1])
    axes[0].set_ylabel("Age (years)")

    # Shared colorbar for ADKM/NINT
    c0 = fig.colorbar(ims[0], ax=axes[:2], fraction=0.046, pad=0.04)
    c0.set_label(r"$K(a,\Delta t)$")
    c1 = fig.colorbar(ims[2], ax=axes[2], fraction=0.046, pad=0.08)
    c1.set_label(r"$\Delta K$")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.88, wspace=0.18)
    save_fig(fig, "nch_age_lag_heatmap")
    return {"adkm": m_adkm, "nint": m_nint, "vmax": vmax, "dmax": dmax}


# =============================================================================
# Figure 3 — nch_subgroup_performance
# =============================================================================
def hist_bin_name(span_days: float) -> str:
    for name, lo, hi in HIST_BINS:
        if lo <= float(span_days) < hi:
            return name
    return HIST_BINS[-1][0]


@torch.no_grad()
def collect_logits(model, loader, device) -> dict:
    logits_l, targets_l, meta = [], [], []
    for batch in loader:
        batch_d = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
        out = model(batch_d)
        logits = out["code_logits"].float().cpu()
        targets = batch_d["target_codes"].float().cpu()
        logits_l.append(logits)
        targets_l.append(targets)
        ages = batch_d["age_years"]
        mask = batch_d["attention_mask"].bool()
        ts = batch_d["timestamps_days"]
        B = logits.shape[0]
        for i in range(B):
            n = int(mask[i].sum().item())
            age = float(ages[i, n - 1].cpu()) if n > 0 else float("nan")
            if n >= 2:
                span = float((ts[i, n - 1] - ts[i, 0]).cpu())
            else:
                span = 0.0
            pid = batch.get("patient_id")
            meta.append({
                "patient_id": int(pid[i]) if pid is not None else -1,
                "age_years": age,
                "age_band": age_band_name(age),
                "history_span_days": span,
                "history_bin": hist_bin_name(span),
            })
    return {
        "logits": torch.cat(logits_l, dim=0).numpy().astype(np.float16),
        "targets": torch.cat(targets_l, dim=0).numpy().astype(np.uint8),
        "meta": pd.DataFrame(meta),
    }


def active_class_mask(targets: np.ndarray) -> np.ndarray:
    return targets.sum(axis=0) > 0



def boot_stratum(logits_a, logits_n, targets, patients, idx_mask,
                 n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired patient bootstrap for ADKM / NINT / Δ micro-AUPRC."""
    from collections import defaultdict
    idx = np.flatnonzero(idx_mask)
    empty = dict(
        adkm=np.nan, adkm_ci_lo=np.nan, adkm_ci_hi=np.nan,
        nint=np.nan, nint_ci_lo=np.nan, nint_ci_hi=np.nan,
        delta=np.nan, delta_ci_lo=np.nan, delta_ci_hi=np.nan,
        n_windows=0, n_patients=0, flagged_small=True,
    )
    if idx.size == 0:
        return empty
    pats = patients[idx]
    uniq = np.unique(pats)
    n_patients = int(uniq.size)
    pa = float(_safe_auprc(targets[idx], logits_a[idx]))
    pn = float(_safe_auprc(targets[idx], logits_n[idx]))
    out = dict(
        adkm=pa, nint=pn, delta=pa - pn,
        n_windows=int(idx.size), n_patients=n_patients,
        flagged_small=n_patients < MIN_PATIENTS,
    )
    if out["flagged_small"]:
        out.update(adkm_ci_lo=np.nan, adkm_ci_hi=np.nan,
                   nint_ci_lo=np.nan, nint_ci_hi=np.nan,
                   delta_ci_lo=np.nan, delta_ci_hi=np.nan)
        return out
    buckets = defaultdict(list)
    for j, p_ in enumerate(pats):
        buckets[int(p_)].append(j)
    rng = np.random.default_rng(seed)
    ba, bn, bd = [], [], []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([buckets[int(p_)] for p_ in samp])
        ii = idx[sel]
        a = float(_safe_auprc(targets[ii], logits_a[ii]))
        n = float(_safe_auprc(targets[ii], logits_n[ii]))
        ba.append(a); bn.append(n); bd.append(a - n)
    for key, arr in (("adkm", ba), ("nint", bn), ("delta", bd)):
        a = np.asarray(arr, dtype=np.float64)
        out[f"{key}_ci_lo"] = float(np.nanpercentile(a, 2.5))
        out[f"{key}_ci_hi"] = float(np.nanpercentile(a, 97.5))
    return out


def fig_nch_subgroup_performance(device: torch.device) -> dict:
    print("=== fig3 nch_subgroup_performance ===", flush=True)
    cfg = json.loads((REPO / "stage2_nch/run/adkm_nch_s0/config.json").read_text())
    tensorized = Path(cfg["data"]["paths"]["tensorized_dir"])
    cache_a = RAW / "adkm_test_logits.npz"
    cache_n = RAW / "nint_test_logits.npz"
    cache_meta = RAW / "test_window_meta.parquet"

    def _collect(tag, ckpt, cache, write_meta=False):
        if cache.exists() and cache_meta.exists():
            print(f"loading cached {tag} logits...", flush=True)
            z = np.load(cache)
            return z["logits"], z["targets"]
        model = build_model_from_ckpt(ckpt, device)
        ds = NCHForecastDataset(tensorized / "test", VOCAB_PATH, max_seq_len=1024)
        loader = DataLoader(
            ds, batch_size=8, shuffle=False, num_workers=2,
            collate_fn=make_nch_collate(assert_horizon=False), pin_memory=True,
        )
        print(f"collecting {tag} logits...", flush=True)
        pack = collect_logits(model, loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        np.savez_compressed(cache, logits=pack["logits"], targets=pack["targets"])
        if write_meta:
            pack["meta"].to_parquet(cache_meta, index=False)
        return pack["logits"], pack["targets"]

    logits_a_full, targets = _collect("ADKM", NCH_ADKM, cache_a, write_meta=True)
    logits_n_full, targets_n = _collect("NINT", NCH_NINT, cache_n, write_meta=False)
    assert np.array_equal(targets, targets_n)
    meta = pd.read_parquet(cache_meta)
    patients = meta["patient_id"].to_numpy()
    cols = np.flatnonzero(active_class_mask(targets))
    print(f"windows={len(meta)} active_classes={cols.size}/{targets.shape[1]}", flush=True)
    logits_a = np.asarray(logits_a_full[:, cols], dtype=np.float32)
    logits_n = np.asarray(logits_n_full[:, cols], dtype=np.float32)
    targets_a = targets[:, cols]
    del logits_a_full, logits_n_full, targets, targets_n

    rows = []
    for band in AGE_ORDER:
        print(f"bootstrap age {band}...", flush=True)
        d = boot_stratum(logits_a, logits_n, targets_a, patients,
                         meta["age_band"].to_numpy() == band)
        rows.append({
            "stratum_type": "age", "stratum": band,
            "adkm_micro_auprc": d["adkm"], "adkm_ci_lo": d["adkm_ci_lo"], "adkm_ci_hi": d["adkm_ci_hi"],
            "nint_micro_auprc": d["nint"], "nint_ci_lo": d["nint_ci_lo"], "nint_ci_hi": d["nint_ci_hi"],
            "delta": d["delta"], "delta_ci_lo": d["delta_ci_lo"], "delta_ci_hi": d["delta_ci_hi"],
            "n_windows": d["n_windows"], "n_patients": d["n_patients"],
            "flagged_small": bool(d["flagged_small"]),
        })
        print(f"  {band}: ADKM={d['adkm']:.4f} NINT={d['nint']:.4f} Δ={d['delta']:.4f} n={d['n_patients']}", flush=True)

    for name, _, _ in HIST_BINS:
        print(f"bootstrap history {name}...", flush=True)
        d = boot_stratum(logits_a, logits_n, targets_a, patients,
                         meta["history_bin"].to_numpy() == name)
        rows.append({
            "stratum_type": "history", "stratum": name,
            "adkm_micro_auprc": d["adkm"], "adkm_ci_lo": d["adkm_ci_lo"], "adkm_ci_hi": d["adkm_ci_hi"],
            "nint_micro_auprc": d["nint"], "nint_ci_lo": d["nint_ci_lo"], "nint_ci_hi": d["nint_ci_hi"],
            "delta": d["delta"], "delta_ci_lo": d["delta_ci_lo"], "delta_ci_hi": d["delta_ci_hi"],
            "n_windows": d["n_windows"], "n_patients": d["n_patients"],
            "flagged_small": bool(d["flagged_small"]),
        })
        print(f"  {name}: ADKM={d['adkm']:.4f} NINT={d['nint']:.4f} Δ={d['delta']:.4f} n={d['n_patients']}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "nch_subgroup_performance.csv", index=False)
    table.to_json(OUT / "nch_subgroup_performance.json", orient="records", indent=2)
    (RAW / "nch_subgroup_meta.json").write_text(json.dumps({
        "adkm_ckpt": str(NCH_ADKM), "nint_ckpt": str(NCH_NINT),
        "n_windows": int(len(meta)), "n_patients": int(meta.patient_id.nunique()),
        "n_active_classes": int(cols.size),
        "active_class_rule": "codes with ≥1 positive label in the held-out test set",
        "metric": "micro-AUPRC (_safe_auprc / project multilabel_metrics definition)",
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "unit": "patient"},
        "min_patients": MIN_PATIENTS,
        "history_bins_days": {n: [lo, hi] for n, lo, hi in HIST_BINS},
        "age_bands": AGE_ORDER,
    }, indent=2) + "\n")

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    width = 0.36

    def _panel(ax, stratum_type, order, labels, title):
        sub = table[table.stratum_type == stratum_type].set_index("stratum").reindex(order)
        x = np.arange(len(sub))
        for arm, color, off, key in (
            ("NINT", COLOR_NINT, -width / 2, "nint"),
            ("ADKM", COLOR_ADKM, width / 2, "adkm"),
        ):
            y = sub[f"{key}_micro_auprc"].to_numpy(float)
            lo = sub[f"{key}_ci_lo"].to_numpy(float)
            hi = sub[f"{key}_ci_hi"].to_numpy(float)
            small = sub["flagged_small"].fillna(False).to_numpy(bool)
            y_plot = np.where(small, np.nan, y)
            yerr = np.vstack([
                np.where(small | ~np.isfinite(lo), np.nan, y - lo),
                np.where(small | ~np.isfinite(hi), np.nan, hi - y),
            ])
            ax.bar(x + off, y_plot, width=width, color=color, label=arm, zorder=3)
            ax.errorbar(x + off, y_plot, yerr=yerr, fmt="none", ecolor="k",
                        elinewidth=0.8, capsize=2, zorder=4)
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.text(i, 0.02, f"n={int(r.n_patients)}", ha="center", va="bottom",
                    fontsize=6, color="0.4", transform=ax.get_xaxis_transform())
            if bool(r.flagged_small):
                ax.text(i, 0.5, "small n", ha="center", va="center", fontsize=7,
                        color="0.5", rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=15 if stratum_type == "history" else 0,
            ha="right" if stratum_type == "history" else "center",
        )
        ax.set_title(title, pad=4)
        ax.set_ylabel("micro-AUPRC")
        ymax = float(np.nanmax(table[["adkm_micro_auprc", "nint_micro_auprc"]].to_numpy()))
        ax.set_ylim(0, max(0.55, ymax * 1.15))
        ax.grid(axis="y", color="0.9", lw=0.7, zorder=0)

    _panel(axes[0], "age", AGE_ORDER, AGE_TICK, "Developmental age")
    _panel(axes[1], "history", [h[0] for h in HIST_BINS], [h[0] for h in HIST_BINS],
           "Available history")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_fig(fig, "nch_subgroup_performance")
    return {"table": table, "n_active": int(cols.size), "n_windows": int(len(meta))}


def write_summary(info: dict) -> None:
    p = OUT / "figure_results_summary.md"
    m1, m2, m3 = info["fig1"], info["fig2"], info["fig3"]
    lines = [
        "# Figure results summary",
        "",
        "## Checkpoints (validation micro-AUPRC selection)",
        "",
        f"| Arm | Path | Epoch | Val micro-AUPRC | λ₀ | β | age μ/σ |",
        f"|---|---|---:|---:|---:|---:|---|",
        f"| MIMIC ADKM | `{MIMIC_ADKM.relative_to(REPO)}` | {m1['adkm']['epoch']} | "
        f"(from history.json: 0.4665) | {m1['adkm']['lambda0']:.4f} | {m1['adkm']['beta']:.4f} | "
        f"{m1['adkm']['age_mean']:.3f}/{m1['adkm']['age_sd']:.3f} |",
        f"| MIMIC NINT | `{MIMIC_NINT.relative_to(REPO)}` | {m1['nint']['epoch']} | "
        f"(from history.json: 0.4708) | {m1['nint']['lambda0']:.4f} | {m1['nint']['beta']:.4f} | "
        f"{m1['nint']['age_mean']:.3f}/{m1['nint']['age_sd']:.3f} |",
        f"| NCH ADKM | `{NCH_ADKM.relative_to(REPO)}` | {m2['adkm']['epoch']} | "
        f"{m2['adkm'].get('val_micro_auprc')} | {m2['adkm']['lambda0']:.4f} | {m2['adkm']['beta']:.4f} | "
        f"{m2['adkm']['age_mean']:.0f}/{m2['adkm']['age_sd']:.0f} |",
        f"| NCH NINT | `{NCH_NINT.relative_to(REPO)}` | {m2['nint']['epoch']} | "
        f"{m2['nint'].get('val_micro_auprc')} | {m2['nint']['lambda0']:.4f} | {m2['nint']['beta']:.4f} | "
        f"{m2['nint']['age_mean']:.0f}/{m2['nint']['age_sd']:.0f} |",
        "",
        "## Quantities plotted",
        "",
        "### `mimic_lambda_age`",
        "- **y:** λ(a) = λ₀ + β z(a) from `AgeTemporalBias.lambda_of`.",
        "- **z(a):** (a − μ)/σ with frozen MIMIC train event-level μ, σ.",
        "- **x:** age in years over observed MIMIC event-age range.",
        "- **Background:** density histogram of event ages sampled from MIMIC train NPZ shards.",
        "",
        "### `nch_age_lag_heatmap`",
        "- **Cell:** K(a, Δt) = −λ(a)·τ(Δt), τ = log1p(|Δt|/7) via `lag_to_tau`, λ via `lambda_of`",
        "  (verified equal to `pairwise_bias` on a probe).",
        "- Ages 0–18 y; lags log-spaced 1 d → 10 y.",
        "- Panels: ADKM, NINT (shared color scale), ADKM−NINT.",
        "",
        "### `nch_subgroup_performance`",
        "- Held-out NCH test set; epoch-4 ADKM/NINT `checkpoint_best_auprc.pt`.",
        "- Metric: **micro-AUPRC** (project `_safe_auprc`), on codes with ≥1 positive in the test set.",
        "- Strata: developmental age `<1`, `1–5`, `6–11`, `12–17`; history "
        "`<3 months`, `3–12 months`, `1–3 years`, `>3 years` (by available lookback span).",
        f"- Patient-level bootstrap 95% CI (n={N_BOOT}, seed={BOOT_SEED}); paired Δ = ADKM − NINT.",
        f"- Subgroups with <{MIN_PATIENTS} patients are flagged and CI omitted.",
        "",
        "## Subgroup sample sizes",
        "",
        "See `nch_subgroup_performance.csv`.",
        "",
        "## Outputs",
        "",
        f"- `{OUT / 'mimic_lambda_age.png'}` / `.svg`",
        f"- `{OUT / 'nch_age_lag_heatmap.png'}` / `.svg`",
        f"- `{OUT / 'nch_subgroup_performance.png'}` / `.svg`",
        f"- `{OUT / 'nch_subgroup_performance.csv'}` / `.json`",
        f"- `{RAW}/` plotting data and metadata",
        "",
        "## Assumptions / exclusions",
        "",
        "- No retraining; preprocessing unchanged.",
        "- MIMIC age histogram is a random sample of train shards (not the full 405M events).",
        "- Heatmap lag axis is log10-scaled for display; values use the model τ on raw days.",
        "- Micro-AUPRC bootstrap uses the active test-set code subset for tractability; "
        "never-positive codes in the full test set are omitted (cannot contribute TPs).",
        "",
    ]
    # Add table dump
    if "table" in m3:
        lines.append("### Numerical results")
        lines.append("")
        lines.append("```")
        lines.append(m3["table"].to_string(index=False))
        lines.append("```")
        lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", p, flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all","1","2","3"], default="all")
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    info1 = info2 = info3 = {}
    if args.only in ("all", "1"):
        info1 = fig_mimic_lambda_age()
    if args.only in ("all", "2"):
        info2 = fig_nch_age_lag_heatmap()
    if args.only in ("all", "3"):
        info3 = fig_nch_subgroup_performance(device)
    if args.only == "all":
        write_summary({"fig1": info1, "fig2": info2, "fig3": info3})
    elif args.only == "3" and info3:
        # minimal summary refresh using existing metas if present
        write_summary({
            "fig1": json.loads((RAW/"mimic_lambda_age_meta.json").read_text()) if (RAW/"mimic_lambda_age_meta.json").exists() else {"adkm":{},"nint":{}},
            "fig2": json.loads((RAW/"nch_age_lag_meta.json").read_text()) if (RAW/"nch_age_lag_meta.json").exists() else {"adkm":{},"nint":{}},
            "fig3": info3,
        })
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
