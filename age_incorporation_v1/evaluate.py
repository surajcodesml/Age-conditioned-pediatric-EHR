"""Generic binary-classification metrics and evaluation.

Age groups are analysis-only; they never enter the model.
"""
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


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    groups_all: list[str] = []
    for batch in loader:
        code_ids = batch["code_ids"].to(device)
        type_ids = batch["type_ids"].to(device)
        time_norm = batch["time_norm"].to(device)
        age_event = batch["age_event_norm"].to(device)
        pad = batch["padding_mask"].to(device)
        index_age = batch["index_age_norm"].to(device)
        labels = batch["labels"].to(device)
        logits = model(code_ids, type_ids, time_norm, age_event, pad, index_age)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(labels.detach().cpu().numpy())
        groups_all.extend(batch["age_group"])

    logits_np = np.concatenate(logits_all)
    y = np.concatenate(y_all).astype(np.float64)
    # probabilities for ranking metrics
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -30.0, 30.0)))
    bce = float(
        binary_cross_entropy_with_logits(
            torch.from_numpy(logits_np), torch.from_numpy(y).float()
        ).item()
    )
    out: dict[str, Any] = {
        "n": int(y.size),
        "prevalence": float(y.mean()) if y.size else float("nan"),
        "auprc": _safe_auprc(y, probs),
        "auroc": _safe_auroc(y, probs),
        "bce": bce,
        "by_age_group": {},
    }
    groups = np.asarray(groups_all)
    for g in AGE_GROUPS:
        mask = groups == g
        yg, pg, lg = y[mask], probs[mask], logits_np[mask]
        if yg.size == 0:
            out["by_age_group"][g] = {"n": 0, "auprc": float("nan"), "auroc": float("nan"), "bce": float("nan")}
            continue
        bce_g = float(
            binary_cross_entropy_with_logits(
                torch.from_numpy(lg), torch.from_numpy(yg).float()
            ).item()
        )
        out["by_age_group"][g] = {
            "n": int(yg.size),
            "prevalence": float(yg.mean()),
            "auprc": _safe_auprc(yg, pg),
            "auroc": _safe_auroc(yg, pg),
            "bce": bce_g,
        }
    return out
