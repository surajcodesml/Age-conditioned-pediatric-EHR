"""Task metrics, mechanism recovery, and counterfactual age interventions."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.functional import binary_cross_entropy_with_logits

from config import AGE_GROUPS


def _safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _per_example_bce(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(_sigmoid(logits), 1e-7, 1.0 - 1e-7)
    y = y.astype(np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    z_age: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the model. Optional ``z_age`` replaces batch z in order of loader examples."""
    model.eval()
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    groups_all: list[str] = []
    pids: list[str] = []
    ages: list[np.ndarray] = []
    offset = 0
    for batch in loader:
        bs = int(batch["labels"].size(0))
        z = batch["z_age"].to(device)
        if z_age is not None:
            z = torch.tensor(z_age[offset : offset + bs], dtype=torch.float32, device=device)
            offset += bs
        logits = model(
            batch["code_ids"].to(device),
            batch["type_ids"].to(device),
            batch["time_norm"].to(device),
            batch["days_before"].to(device),
            batch["padding_mask"].to(device),
            z,
            batch["is_query"].to(device),
        )
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(batch["labels"].numpy())
        groups_all.extend(batch["age_group"])
        pids.extend(batch["patient_id"])
        ages.append(batch["age_years"].numpy())
    logits_np = np.concatenate(logits_all)
    y = np.concatenate(y_all).astype(np.float64)
    return {
        "logits": logits_np,
        "y": y,
        "probs": _sigmoid(logits_np),
        "age_group": np.asarray(groups_all),
        "patient_id": np.asarray(pids),
        "age_years": np.concatenate(ages),
    }


def summarize_predictions(pred: dict[str, Any]) -> dict[str, Any]:
    y = pred["y"]
    logits = pred["logits"]
    probs = pred["probs"]
    bce = float(
        binary_cross_entropy_with_logits(
            torch.from_numpy(logits), torch.from_numpy(y).float()
        ).item()
    )
    acc = float(((probs >= 0.5).astype(np.float64) == y).mean()) if y.size else float("nan")
    out: dict[str, Any] = {
        "n": int(y.size),
        "prevalence": float(y.mean()) if y.size else float("nan"),
        "bce": bce,
        "accuracy": acc,
        "auroc": _safe_auroc(y, probs),
        "auprc": _safe_auprc(y, probs),
        "by_age_group": {},
    }
    groups = pred["age_group"]
    for g in AGE_GROUPS:
        mask = groups == g
        yg, pg, lg = y[mask], probs[mask], logits[mask]
        if yg.size == 0:
            out["by_age_group"][g] = {
                "n": 0,
                "bce": float("nan"),
                "accuracy": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
            }
            continue
        bce_g = float(
            binary_cross_entropy_with_logits(
                torch.from_numpy(lg), torch.from_numpy(yg).float()
            ).item()
        )
        out["by_age_group"][g] = {
            "n": int(yg.size),
            "prevalence": float(yg.mean()),
            "bce": bce_g,
            "accuracy": float(((pg >= 0.5).astype(np.float64) == yg).mean()),
            "auroc": _safe_auroc(yg, pg),
            "auprc": _safe_auprc(yg, pg),
        }
    return out


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, Any]:
    return summarize_predictions(predict(model, loader, device))


def paired_delta_stats(delta: np.ndarray) -> dict[str, float]:
    d = np.asarray(delta, dtype=np.float64)
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "mean": float(d.mean()) if n else float("nan"),
        "std": float(d.std(ddof=1)) if n > 1 else float("nan"),
        "se": se,
        "median": float(np.median(d)) if n else float("nan"),
        "frac_positive": float((d > 0).mean()) if n else float("nan"),
        "n": int(n),
    }


@torch.no_grad()
def counterfactual_age(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    z_correct: np.ndarray,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z_constant = np.zeros_like(z_correct)
    z_shuffle = rng.permutation(z_correct)

    pred_c = predict(model, loader, device, z_age=z_correct)
    pred_k = predict(model, loader, device, z_age=z_constant)
    pred_s = predict(model, loader, device, z_age=z_shuffle)
    y = pred_c["y"]
    bce_c = _per_example_bce(pred_c["logits"], y)
    bce_k = _per_example_bce(pred_k["logits"], y)
    bce_s = _per_example_bce(pred_s["logits"], y)
    d_k = bce_k - bce_c
    d_s = bce_s - bce_c
    return {
        "correct": summarize_predictions(pred_c),
        "constant": summarize_predictions(pred_k),
        "shuffled": summarize_predictions(pred_s),
        "delta_constant": paired_delta_stats(d_k),
        "delta_shuffle": paired_delta_stats(d_s),
        "correct_bce": float(bce_c.mean()),
        "constant_bce": float(bce_k.mean()),
        "shuffled_bce": float(bce_s.mean()),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def signal_attention_recovery(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Compare QUERY→signal attention (renormalized) with planted softmax weights."""
    model.eval()
    pears: list[float] = []
    spears: list[float] = []
    maes: list[float] = []
    jss: list[float] = []
    n_ok = 0
    for batch in loader:
        logits = model(
            batch["code_ids"].to(device),
            batch["type_ids"].to(device),
            batch["time_norm"].to(device),
            batch["days_before"].to(device),
            batch["padding_mask"].to(device),
            batch["z_age"].to(device),
            batch["is_query"].to(device),
        )
        _ = logits
        cache = model._cache
        assert cache is not None
        attn = cache["attn"].mean(dim=1)  # average heads: (B, T, T)
        qmask = cache["query_mask"]
        sig = batch["is_signal"].to(device)
        true_w = batch["true_w"].to(device)
        bsz = attn.size(0)
        for i in range(bsz):
            qpos = torch.where(qmask[i])[0]
            if qpos.numel() == 0:
                continue
            q = int(qpos[-1].item())
            sp = torch.where(sig[i])[0]
            if sp.numel() < 2:
                continue
            learned = attn[i, q, sp].detach().cpu().numpy().astype(np.float64)
            truth = true_w[i, sp].detach().cpu().numpy().astype(np.float64)
            if learned.sum() <= 0 or truth.sum() <= 0:
                continue
            learned = learned / learned.sum()
            truth = truth / truth.sum()
            if learned.std() == 0 or truth.std() == 0:
                pear = float("nan")
            else:
                pear = float(np.corrcoef(learned, truth)[0, 1])
            pears.append(pear)
            spears.append(_spearman(learned, truth))
            maes.append(float(np.mean(np.abs(learned - truth))))
            jss.append(_js_divergence(learned, truth))
            n_ok += 1
    return {
        "n": n_ok,
        "pearson_mean": float(np.nanmean(pears)) if pears else float("nan"),
        "spearman_mean": float(np.nanmean(spears)) if spears else float("nan"),
        "mae_mean": float(np.mean(maes)) if maes else float("nan"),
        "js_mean": float(np.mean(jss)) if jss else float("nan"),
    }
