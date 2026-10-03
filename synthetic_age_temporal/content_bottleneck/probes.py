"""Diagnostic linear probes on frozen C01 representations (Part A/B)."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _safe_auroc(y, score) -> float | None:
    y = np.asarray(y).astype(np.int32)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, score))


def _safe_auprc(y, score) -> float | None:
    y = np.asarray(y).astype(np.int32)
    if y.min() == y.max():
        return None
    return float(average_precision_score(y, score))


def probe_signal_identity(
    X: np.ndarray,
    membership: np.ndarray,
    *,
    n_codes: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """A1: multi-label linear probes, one per signal."""
    n_signals = membership.shape[1]
    per_signal = []
    for j in range(n_signals):
        y = membership[:, j].astype(np.int32)
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            per_signal.append({"signal": j, "auroc": None, "auprc": None, "n_pos": int(y.sum())})
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xtr, ytr)
        score = clf.decision_function(Xte)
        per_signal.append({
            "signal": j,
            "auroc": _safe_auroc(yte, score),
            "auprc": _safe_auprc(yte, score),
            "n_pos": int(y.sum()),
        })
    aurocs = [p["auroc"] for p in per_signal if p["auroc"] is not None]
    auprcs = [p["auprc"] for p in per_signal if p["auprc"] is not None]
    out: dict[str, Any] = {
        "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
        "mean_auprc": float(np.mean(auprcs)) if auprcs else None,
        "per_signal": per_signal,
    }
    if n_codes is not None:
        strata = {}
        for lo, hi, name in ((1, 2, "1"), (2, 5, "2-4"), (5, 10, "5-9"), (10, 10_000, "10+")):
            mask = (n_codes >= lo) & (n_codes < hi)
            if mask.sum() < 50:
                continue
            # any-signal presence
            y = (membership[mask].sum(axis=1) > 0).astype(np.int32)
            if y.min() == y.max():
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X[mask], y, test_size=0.3, random_state=seed, stratify=y,
            )
            scaler = StandardScaler()
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            clf.fit(scaler.fit_transform(Xtr), ytr)
            score = clf.decision_function(scaler.transform(Xte))
            strata[name] = {
                "auroc": _safe_auroc(yte, score),
                "auprc": _safe_auprc(yte, score),
                "n": int(mask.sum()),
            }
        out["by_n_codes_any_signal"] = strata
    return out


def probe_oracle_content_vector(
    X: np.ndarray,
    w_true: np.ndarray,
    membership: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """A2: predict w_true from representation on signal encounters."""
    mask = membership.sum(axis=1) > 0
    X = X[mask]
    Y = w_true[mask]
    if X.shape[0] < 20:
        return {"rmse": None, "corr": None, "r2": None, "per_target_rmse": None}
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=seed)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    preds = []
    for t in range(Y.shape[1]):
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr_s, Ytr[:, t])
        preds.append(reg.predict(Xte_s))
    pred = np.stack(preds, axis=1)
    err = pred - Yte
    rmse = float(np.sqrt(np.mean(err ** 2)))
    per_t = np.sqrt(np.mean(err ** 2, axis=0)).tolist()
    flat_p, flat_y = pred.ravel(), Yte.ravel()
    corr = float(np.corrcoef(flat_p, flat_y)[0, 1]) if flat_y.std() > 0 else None
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((Yte - Yte.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return {"rmse": rmse, "corr": corr, "r2": r2, "per_target_rmse": per_t, "n": int(mask.sum())}


def background_leakage(u: np.ndarray, membership: np.ndarray, oracle_pre: np.ndarray) -> dict[str, Any]:
    """A3: content-score / oracle magnitude on background vs signal encounters."""
    bg = membership.sum(axis=1) == 0
    sig = ~bg
    mag = np.linalg.norm(oracle_pre, axis=1)
    return {
        "u_mean_background": float(np.mean(u[bg])) if bg.any() else None,
        "u_mean_signal": float(np.mean(u[sig])) if sig.any() else None,
        "u_std_background": float(np.std(u[bg])) if bg.any() else None,
        "u_std_signal": float(np.std(u[sig])) if sig.any() else None,
        "oracle_mag_mean_background": float(np.mean(mag[bg])) if bg.any() else None,
        "oracle_mag_mean_signal": float(np.mean(mag[sig])) if sig.any() else None,
        "n_background": int(bg.sum()),
        "n_signal": int(sig.sum()),
    }


def retrieval_alignment(u: np.ndarray, oracle_pre: np.ndarray, membership: np.ndarray) -> dict[str, Any]:
    """Part B: correlate global u with oracle target relevance summaries."""
    sig = membership.sum(axis=1) > 0
    u_s = u[sig]
    pre = oracle_pre[sig]
    mean_abs = np.mean(np.abs(pre), axis=1)
    max_abs = np.max(np.abs(pre), axis=1)
    n_affected = np.sum(np.abs(pre) > 1e-8, axis=1).astype(np.float64)

    def _corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    # Same global u for different signals? Compare mean u per primary signal.
    primary = np.argmax(membership[sig], axis=1)
    per_signal_u = []
    for j in range(membership.shape[1]):
        m = primary == j
        if m.sum() == 0:
            per_signal_u.append(None)
        else:
            per_signal_u.append(float(np.mean(u_s[m])))

    # Pairwise |mean u_i - mean u_j| for signals with different target profiles
    return {
        "corr_u_mean_abs_oracle": _corr(u_s, mean_abs),
        "corr_u_max_abs_oracle": _corr(u_s, max_abs),
        "corr_u_n_targets_affected": _corr(u_s, n_affected),
        "mean_u_per_signal": per_signal_u,
        "n_signal_encounters": int(sig.sum()),
    }
