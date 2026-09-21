"""Stage-2 metrics: pos/neg BCE, prevalence baseline, stratified tables."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from stage1_mimic_pretrain.metrics import class_imbalance_report, multilabel_metrics
from stage2_nch.config import EVAL_KS


def elementwise_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")


def pos_neg_bce(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    loss = elementwise_bce(logits, targets)
    pos = targets.float() > 0.5
    neg = ~pos
    out = {
        "bce": float(loss.mean().detach()),
        "positive_bce": float(loss[pos].mean().detach()) if bool(pos.any()) else float("nan"),
        "negative_bce": float(loss[neg].mean().detach()) if bool(neg.any()) else float("nan"),
        "n_positive_labels": int(pos.sum()),
        "n_negative_labels": int(neg.sum()),
    }
    return out


def prevalence_logits(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


def train_prevalence(targets: torch.Tensor) -> np.ndarray:
    y = targets.float().cpu().numpy()
    return y.mean(axis=0)


def evaluate_prevalence_baseline(p: np.ndarray, targets: torch.Tensor,
                                 ks=EVAL_KS) -> dict[str, Any]:
    logits = torch.from_numpy(prevalence_logits(p).astype(np.float32))
    logits = logits.unsqueeze(0).expand(targets.shape[0], -1)
    ml = multilabel_metrics(logits, targets.float(), ks=ks)
    pn = pos_neg_bce(logits, targets)
    ml.update(pn)
    ml["imbalance"] = class_imbalance_report(targets)
    return ml


def history_bin_edges(n_events: np.ndarray) -> list[float]:
    """Tertile edges on the training n_input_events distribution."""
    x = np.asarray(n_events, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0, 1.0, 2.0, float("inf")]
    q = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    e1, e2 = float(q[0]), float(q[1])
    if e1 == e2:
        e2 = e1 + 1.0
    return [0.0, e1, e2, float("inf")]


def history_bin_name(n: float, edges: list[float]) -> str:
    if n < edges[1]:
        return "short"
    if n < edges[2]:
        return "medium"
    return "long"


def param_l2_delta(model, init_state: dict[str, torch.Tensor],
                   prefixes: tuple[str, ...] = ("encoder", "head", "demo_proj",
                                                "pooling", "temporal")) -> dict[str, float]:
    cur = model.state_dict()
    out: dict[str, float] = {}
    for pref in prefixes:
        num = 0.0
        den = 0.0
        for k, v0 in init_state.items():
            if not k.startswith(pref):
                continue
            if k not in cur:
                continue
            d = (cur[k].detach().cpu().float() - v0.detach().cpu().float()).norm().item()
            n0 = v0.detach().cpu().float().norm().item()
            num += d * d
            den += n0 * n0
        out[f"delta_l2_{pref}"] = float(num ** 0.5)
        out[f"rel_l2_{pref}"] = float((num ** 0.5) / ((den ** 0.5) + 1e-12))
    if "temporal.lambda0" in cur and "temporal.lambda0" in init_state:
        out["delta_lambda0"] = float(
            (cur["temporal.lambda0"] - init_state["temporal.lambda0"]).abs().cpu())
    if "temporal.beta" in cur and "temporal.beta" in init_state:
        out["delta_beta"] = float(
            (cur["temporal.beta"] - init_state["temporal.beta"]).abs().cpu())
    return out
