"""Shared evaluation metrics for the baseline suite.

Wraps / delegates to the stage1_mimic_pretrain.metrics implementation
where possible, adding synthetic-benchmark-specific functionality.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Low-level helpers (self-contained — no external dependency)
# ---------------------------------------------------------------------------

def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based midranks for Mann–Whitney AUROC."""
    n = int(scores.size)
    order = np.argsort(scores, kind="mergesort")
    sorted_s = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    n_pos, n_neg = float(y.sum()), float(y.size) - float(y.sum())
    if y.size == 0 or n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(_average_ranks(s)[y.astype(bool)].sum())
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    if y.size == 0 or n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y, dtype=np.float64)
    prec = tp / np.arange(1, y.size + 1, dtype=np.float64)
    return float(prec[y.astype(bool)].mean())


# ---------------------------------------------------------------------------
# Multi-label predictive metrics
# ---------------------------------------------------------------------------

def multilabel_metrics(
    y: np.ndarray,
    logits: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
    *,
    mode: str = "full",
) -> dict[str, Any]:
    """BCE, micro/macro AUROC/AUPRC, Precision@k, Recall@k.

    ``mode``:
      - ``"bce"``: BCE only (fast; used for early-stopping)
      - ``"fast"``: BCE + Precision/Recall@k (no AUROC/AUPRC)
      - ``"full"``: all metrics; macro is skipped when ``C > 2000`` (MIMIC-scale
        next-visit heads) because a pure-Python per-class loop over ~30k labels
        takes tens of minutes and thrashing host RAM.

    Macro metrics skip classes lacking both labels.
    """
    # float32 keeps peak host RAM manageable at MIMIC |V|≈30k
    y = np.asarray(y, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    bce = float(
        -(y * np.log(np.clip(p, 1e-7, 1)) +
          (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))).mean()
    )
    out: dict[str, Any] = {
        "bce": bce,
        "micro_auroc": float("nan"),
        "macro_auroc": float("nan"),
        "micro_auprc": float("nan"),
        "macro_auprc": float("nan"),
    }
    if mode == "bce":
        return out

    # Precision@k / Recall@k (cheap relative to AUROC on |V|≈30k)
    for k_val in ks:
        pk, rk = _precision_recall_at_k(y, p, k_val)
        out[f"precision@{k_val}"] = pk
        out[f"recall@{k_val}"] = rk
    if mode == "fast":
        return out

    # Micro (ravel) — dominant cost/RAM. Subsample if enormous (MIMIC next-visit).
    y_flat = y.ravel()
    p_flat = p.ravel()
    max_micro = 5_000_000
    if y_flat.size > max_micro:
        rng = np.random.default_rng(0)
        idx = rng.choice(y_flat.size, size=max_micro, replace=False)
        y_flat = y_flat[idx]
        p_flat = p_flat[idx]
        out["micro_subsampled_to"] = max_micro
    out["micro_auroc"] = safe_auroc(y_flat, p_flat)
    out["micro_auprc"] = safe_auprc(y_flat, p_flat)

    # Macro: skip on MIMIC-scale heads (C≈30k). Callers that need macro should
    # pass a column-subset or mode override.
    n_classes = int(y.shape[1]) if y.ndim == 2 else 1
    if n_classes <= 2000:
        aurocs, auprcs = [], []
        for k in range(n_classes):
            if y[:, k].sum() == 0 or y[:, k].sum() == len(y):
                continue
            aurocs.append(safe_auroc(y[:, k], p[:, k]))
            auprcs.append(safe_auprc(y[:, k], p[:, k]))
        aurocs = [x for x in aurocs if not np.isnan(x)]
        auprcs = [x for x in auprcs if not np.isnan(x)]
        if aurocs:
            out["macro_auroc"] = float(np.mean(aurocs))
        if auprcs:
            out["macro_auprc"] = float(np.mean(auprcs))
    else:
        out["macro_skipped"] = True
        out["macro_skip_reason"] = f"n_classes={n_classes}>2000"
    return out


def _precision_recall_at_k(
    y: np.ndarray, scores: np.ndarray, k: int,
) -> tuple[float, float]:
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


# ---------------------------------------------------------------------------
# Binary classification metrics (for PIC / single-label tasks)
# ---------------------------------------------------------------------------

def binary_metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """AUROC, AUPRC, BCE for a single binary outcome."""
    y = y.astype(np.float64).ravel()
    logits = logits.astype(np.float64).ravel()
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    bce = float(
        -(y * np.log(np.clip(p, 1e-7, 1)) +
          (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))).mean()
    )
    return {
        "bce": bce,
        "auroc": safe_auroc(y, p),
        "auprc": safe_auprc(y, p),
    }
