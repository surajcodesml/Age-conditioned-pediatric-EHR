"""Evaluation metrics, mechanism recovery, and functional ablations."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from config import PRECISION_K, PROBE_AGES, RECALL_K, SURFACE_AGES, SURFACE_LAGS_DAYS
from config import lambda_true, relevance, tau_from_days
from model import BenchmarkModel


def _precision_recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    n, t = y.shape
    k = min(k, t)
    precs, recs = [], []
    for i in range(n):
        top = np.argpartition(-scores[i], kth=k - 1)[:k]
        hit = y[i, top].sum()
        precs.append(hit / k)
        denom = y[i].sum()
        recs.append(hit / denom if denom > 0 else 0.0)
    return float(np.mean(precs)), float(np.mean(recs))


def classification_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    y = y.astype(np.float64)
    logits = logits.astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    bce = float(
        -(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))).mean()
    )
    out = {
        "bce": bce,
        "micro_auroc": float("nan"),
        "macro_auroc": float("nan"),
        "micro_auprc": float("nan"),
        "macro_auprc": float("nan"),
    }
    try:
        out["micro_auroc"] = float(roc_auc_score(y.ravel(), p.ravel()))
        out["micro_auprc"] = float(average_precision_score(y.ravel(), p.ravel()))
    except ValueError:
        pass
    aurocs, auprcs = [], []
    for k in range(y.shape[1]):
        if y[:, k].sum() == 0 or y[:, k].sum() == len(y):
            continue
        try:
            aurocs.append(roc_auc_score(y[:, k], p[:, k]))
            auprcs.append(average_precision_score(y[:, k], p[:, k]))
        except ValueError:
            continue
    if aurocs:
        out["macro_auroc"] = float(np.mean(aurocs))
        out["macro_auprc"] = float(np.mean(auprcs))
    pk, rk = _precision_recall_at_k(y, p, PRECISION_K)
    out[f"precision@{PRECISION_K}"] = pk
    out[f"recall@{RECALL_K}"] = rk
    return out


@torch.no_grad()
def predict(model: BenchmarkModel, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, logits = [], []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(
            code_ids=batch["code_ids"],
            type_ids=batch["type_ids"],
            tau=batch["tau"],
            padding_mask=batch["padding_mask"],
            is_query=batch["is_query"],
            age=batch["age"],
            lag_days=batch["lag_days"],
        )
        ys.append(batch["labels"].cpu().numpy())
        logits.append(out.cpu().numpy())
    return np.concatenate(ys), np.concatenate(logits)


def evaluate_model(
    model: BenchmarkModel,
    loader,
    device: torch.device,
    target_idx: list[int] | None = None,
) -> dict[str, float]:
    y, logits = predict(model, loader, device)
    if target_idx is not None:
        y = y[:, target_idx]
        logits = logits[:, target_idx]
    return classification_metrics(y, logits)


def _learned_lambda_curve(model: BenchmarkModel, ages: np.ndarray) -> np.ndarray:
    """Mean λ(a) over heads for per-head; scalar λ(a) for global."""
    device = model.temporal.theta0.device
    out = []
    for a in ages:
        lam = model.temporal.lambda_of(torch.tensor([float(a)], device=device))
        if lam.ndim == 2:
            out.append(float(lam[0].mean().detach().cpu()))
        else:
            out.append(float(lam[0].detach().cpu()))
    return np.asarray(out, dtype=np.float64)


@torch.no_grad()
def functional_ablations(
    model: BenchmarkModel, loader, device: torch.device
) -> dict[str, Any]:
    """Age shuffle / beta=0 / constant-age / per-head β ablations."""
    base_y, base_logits = predict(model, loader, device)
    base = classification_metrics(base_y, base_logits)

    ages = []
    for batch in loader:
        ages.append(batch["age"].numpy())
    ages_cat = np.concatenate(ages)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(ages_cat)

    def run_with_age_override(
        age_array: np.ndarray | None,
        *,
        beta_zero: bool = False,
        head_zero: int | None = None,
    ):
        model.eval()
        ys, logits = [], []
        offset = 0
        saved = None
        saved_h = None
        if beta_zero:
            saved = model.zero_all_betas_()
        if head_zero is not None and model.per_head_kernel:
            saved_h = model.zero_one_head_beta_(head_zero)
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            bsz = batch["age"].size(0)
            if age_array is not None:
                batch["age"] = torch.tensor(
                    age_array[offset : offset + bsz], dtype=torch.float32, device=device
                )
                offset += bsz
            out = model(
                code_ids=batch["code_ids"],
                type_ids=batch["type_ids"],
                tau=batch["tau"],
                padding_mask=batch["padding_mask"],
                is_query=batch["is_query"],
                age=batch["age"],
                lag_days=batch["lag_days"],
            )
            ys.append(batch["labels"].cpu().numpy())
            logits.append(out.cpu().numpy())
        if beta_zero and saved is not None:
            model.restore_betas_(saved)
        if head_zero is not None and saved_h is not None:
            with torch.no_grad():
                model.temporal.beta[head_zero] = saved_h
        return classification_metrics(np.concatenate(ys), np.concatenate(logits))

    sh = run_with_age_override(shuffled)
    b0 = run_with_age_override(None, beta_zero=True)
    const = run_with_age_override(np.full_like(ages_cat, 9.0))

    out: dict[str, Any] = {
        "normal": base,
        "shuffle_age": sh,
        "beta0": b0,
        "constant_age": const,
        "delta_bce_shuffle_age": sh["bce"] - base["bce"],
        "delta_bce_beta0": b0["bce"] - base["bce"],
        "delta_bce_constant_age": const["bce"] - base["bce"],
        "delta_auroc_shuffle_age": base["micro_auroc"] - sh["micro_auroc"],
    }
    if model.per_head_kernel:
        per_head = {}
        for h in range(model.n_heads):
            mh = run_with_age_override(None, head_zero=h)
            per_head[f"h{h}"] = {
                "metrics": mh,
                "delta_bce": mh["bce"] - base["bce"],
            }
        out["per_head_beta_ablation"] = per_head
    return out


def parameter_recovery(
    model: BenchmarkModel,
    *,
    beta_true: float,
    theta0_true: float,
    scenario: str,
) -> dict[str, Any]:
    ages = np.asarray(SURFACE_AGES, dtype=np.float64)
    lam_true = lambda_true(ages, theta0_true, beta_true)
    lam_learned = _learned_lambda_curve(model, ages)

    rmse = float(np.sqrt(np.mean((lam_learned - lam_true) ** 2)))
    mae = float(np.mean(np.abs(lam_learned - lam_true)))
    if np.std(lam_learned) > 1e-8 and np.std(lam_true) > 1e-8:
        corr = float(np.corrcoef(lam_learned, lam_true)[0, 1])
    else:
        corr = float("nan")

    errs = []
    for a, lam_l in zip(ages, lam_learned):
        for d in SURFACE_LAGS_DAYS:
            t = float(tau_from_days(d))
            r_t = float(relevance(a, t, theta0_true, beta_true))
            r_l = float(np.exp(-lam_l * t))
            errs.append((r_l - r_t) ** 2)
    surface_rmse = float(np.sqrt(np.mean(errs)))

    beta_vec = model.temporal.beta_vector().cpu().numpy().astype(np.float64)
    theta_vec = model.temporal.theta0_vector().cpu().numpy().astype(np.float64)
    beta_hat = float(beta_vec.mean())
    theta0_hat = float(theta_vec.mean())

    sign_ok = None
    if scenario in ("S2", "S3") and model.arm in (
        "age_temporal",
        "age_temporal_per_head",
    ):
        # Majority / mean sign for per-head.
        sign_ok = bool(np.sign(beta_hat) == np.sign(beta_true) and abs(beta_hat) > 1e-3)

    out: dict[str, Any] = {
        "beta_hat": beta_hat,
        "beta_true": beta_true,
        "theta0_hat": theta0_hat,
        "theta0_true": theta0_true,
        "beta_vec": beta_vec.tolist(),
        "theta0_vec": theta_vec.tolist(),
        "beta_std": float(beta_vec.std()) if beta_vec.size > 1 else 0.0,
        "n_heads_nonzero_beta": int(np.sum(np.abs(beta_vec) > 0.05)),
        "sign_match": sign_ok,
        "abs_beta_error": abs(beta_hat - beta_true),
        "RMSE_lambda": rmse,
        "MAE_lambda": mae,
        "corr_lambda": corr,
        "RMSE_surface": surface_rmse,
        "lambda_true_by_age": {str(a): float(v) for a, v in zip(ages, lam_true)},
        "lambda_learned_by_age": {str(a): float(v) for a, v in zip(ages, lam_learned)},
        "probe_lambda_learned": model.temporal.lambda_at_ages(PROBE_AGES),
        "probe_lambda_true": {
            str(a): float(lambda_true(a, theta0_true, beta_true)) for a in PROBE_AGES
        },
        "per_head": bool(getattr(model, "per_head_kernel", False)),
    }
    if model.per_head_kernel:
        # Full per-head λ curves for fig12.
        device = model.temporal.theta0.device
        grid = [0.0, 1.0, 5.0, 10.0, 15.0, 18.0]
        a = torch.tensor(grid, dtype=torch.float32, device=device)
        lam = model.temporal.lambda_of(a).detach().cpu().numpy()  # [n_ages, H]
        out["lambda_per_head"] = {
            f"h{h}": {str(grid[i]): float(lam[i, h]) for i in range(len(grid))}
            for h in range(model.n_heads)
        }
        out["grad_norms"] = {}  # filled by train when available
    return out


@torch.no_grad()
def quick_ablation_deltas(
    model: BenchmarkModel, loader, device: torch.device
) -> dict[str, float]:
    """Cheap per-epoch mechanism trackers (shuffle + beta0 BCE deltas)."""
    abl = functional_ablations(model, loader, device)
    return {
        "delta_bce_shuffle_age": abl["delta_bce_shuffle_age"],
        "delta_bce_beta0": abl["delta_bce_beta0"],
        "val_auprc": abl["normal"]["micro_auprc"],
        "val_bce": abl["normal"]["bce"],
        "val_auroc": abl["normal"]["micro_auroc"],
    }
