"""Age-conditioning extrapolation from a completed Stage-1 checkpoint (CPU)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing.NCH.v2 import paths as P


def _load_temporal(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    # stage1_mimic_pretrain saves under model_state_dict with temporal.lambda0 / temporal.beta
    def get(name_opts):
        for n in name_opts:
            if n in state:
                return float(state[n].detach().cpu().reshape(-1)[0])
        return None

    lambda0 = get(["temporal.lambda0", "lambda0"])
    beta = get(["temporal.beta", "beta"])
    age_mean = get(["age_mean", "temporal.age_mean"])
    age_sd = get(["age_sd", "temporal.age_sd"])
    cfg = ckpt.get("config") or {}
    if age_mean is None:
        age_mean = float(
            (((cfg.get("model") or {}).get("age_transform") or {}).get("mean"))
            or 63.336
        )
    if age_sd is None:
        age_sd = float(
            (((cfg.get("model") or {}).get("age_transform") or {}).get("sd"))
            or 16.575
        )
    return {
        "lambda0": lambda0,
        "beta": beta,
        "age_mean": age_mean,
        "age_sd": age_sd,
        "ckpt": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
    }


def analyze_age_extrapolation(ckpt_path: Path | None = None) -> dict:
    ckpt_path = ckpt_path or P.STAGE1_BEST
    if not ckpt_path.exists():
        alt = P.STAGE1_FINAL
        if alt.exists():
            ckpt_path = alt
        else:
            out = {
                "status": "pending",
                "reason": "No completed Stage-1 checkpoint available",
                "looked_for": [str(P.STAGE1_BEST), str(P.STAGE1_FINAL)],
            }
            P.write_json(P.DIRS["age_extrapolation"] / "age_extrapolation_report.json", out)
            return out

    params = _load_temporal(ckpt_path)
    if params["lambda0"] is None or params["beta"] is None:
        out = {"status": "error", "reason": "lambda0/beta not found in checkpoint", **params}
        P.write_json(P.DIRS["age_extrapolation"] / "age_extrapolation_report.json", out)
        return out

    mean, sd = params["age_mean"], params["age_sd"]
    l0, beta = params["lambda0"], params["beta"]

    ages = [0, 0.5] + list(range(1, 19)) + list(range(20, 91, 10))
    z = [(a - mean) / sd for a in ages]
    lam = [l0 + beta * zi for zi in z]

    # Effective temporal contribution: -lambda(a) * tau(lag)
    # tau = log1p(|Δt|/7)
    lags_days = [0, 7, 30, 90, 365, 730, 1825, 3650]
    lag_labels = ["0d", "7d", "30d", "90d", "1y", "2y", "5y", "10y"]
    rep_ages = [0.5, 2, 5, 10, 15, 18, 40, 65, 85]
    kernel = {}
    for a in rep_ages:
        zi = (a - mean) / sd
        la = l0 + beta * zi
        row = {}
        for lag, lab in zip(lags_days, lag_labels):
            tau = float(np.log1p(lag / 7.0))
            row[lab] = {
                "tau": tau,
                "lambda": la,
                "bias_term_minus_lambda_tau": -la * tau,
            }
        kernel[str(a)] = row

    # Pathology flags
    ped_lams = [l0 + beta * ((a - mean) / sd) for a in [0, 0.5, 1, 5, 10, 17]]
    flags = {
        "sign_change_across_0_90": (min(lam) < 0 < max(lam)) or (min(lam) > 0 > max(lam) is False and np.sign(lam[0]) != np.sign(lam[-1])),
        "pediatric_lambda_magnitude_max": float(np.max(np.abs(ped_lams))),
        "adult_support_approx": {"mean": mean, "sd": sd, "approx_range_years": [mean - 2 * sd, mean + 2 * sd]},
        "extrapolation_z_at_age_0": float((0 - mean) / sd),
        "extrapolation_z_at_age_10": float((10 - mean) / sd),
        "beta_near_zero": abs(beta) < 1e-3,
        "interpretation_hint": (
            "weak/no age dependence" if abs(beta) < 1e-3
            else (
                "pathological pediatric extrapolation risk"
                if abs((0 - mean) / sd) > 3 and abs(beta) > 0.05
                else "smooth linear extrapolation of adult-trained lambda(a)"
            )
        ),
    }

    # Plots
    fig_dir = P.DIRS["age_extrapolation"] / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ages, z, marker="o", ms=3)
    ax.axvspan(mean - 2 * sd, mean + 2 * sd, color="0.85", label="≈ MIMIC ±2σ support")
    ax.axvline(18, color="C1", ls="--", label="age 18")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("z(age)")
    ax.set_title("Adult age standardization extrapolated to pediatrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "z_age.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ages, lam, marker="o", ms=3)
    ax.axvspan(mean - 2 * sd, mean + 2 * sd, color="0.85", label="≈ MIMIC ±2σ support")
    ax.axvline(18, color="C1", ls="--", label="age 18")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("λ(a) = λ0 + β·z(a)")
    ax.set_title(f"Learned age-conditioned λ (λ0={l0:.4f}, β={beta:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "lambda_age.png", dpi=120)
    plt.close(fig)

    # Heatmap of bias term
    mat = np.array([[kernel[str(a)][lab]["bias_term_minus_lambda_tau"] for lab in lag_labels] for a in rep_ages])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(lag_labels)))
    ax.set_xticklabels(lag_labels)
    ax.set_yticks(range(len(rep_ages)))
    ax.set_yticklabels([str(a) for a in rep_ages])
    ax.set_xlabel("lag")
    ax.set_ylabel("age (years)")
    ax.set_title("Temporal bias −λ(a)·τ(lag)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "temporal_bias_heatmap.png", dpi=120)
    plt.close(fig)

    report = {
        "status": "complete",
        "checkpoint": params,
        "equation": "s_ij = q·k/√d − [λ0 + β·z(a_i)] · τ_ij ;  z(a)=(a-μ)/σ ; τ=log1p(|Δt|/7)",
        "ages_evaluated": ages,
        "z_of_age": dict(zip([str(a) for a in ages], z)),
        "lambda_of_age": dict(zip([str(a) for a in ages], lam)),
        "kernel_contribution": kernel,
        "flags": flags,
        "head_specific": False,
        "note": "Parameters are single scalars shared across heads (not per-head).",
    }
    P.write_json(P.DIRS["age_extrapolation"] / "age_extrapolation_report.json", report)
    return report
