"""S5 heterogeneous-persistence counterfactual evaluation (evaluation-only).

Computes group-specific surface RMSE for acute / intermediate / chronic
signal histories and checks relative persistence ordering from black-box
prediction surfaces — without requiring exposed decay parameters.

Group probes use *controlled* mono-group histories (all codes in a group at
a shared lag) so acute / intermediate / chronic curves are comparable.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from baselines.common.counterfactual import (
    FUNCTIONAL_SURFACE_RMSE,
    PARTIAL_SURFACE_RMSE,
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    surface_rmse,
)
from baselines.synthetic.data_adapter import (
    S5_GROUP_THETA,
    S5_PERSISTENCE_GROUPS,
    clone_batch,
)

PredictSurfaceFn = Callable[[float, float], np.ndarray]


def find_multigroup_template(
    loader,
    itos: dict[int, str],
    *,
    min_groups: int = 3,
    max_scan: int = 500,
) -> dict[str, torch.Tensor] | None:
    """Pick a test batch whose history covers ≥ ``min_groups`` persistence groups."""
    for i, batch in enumerate(loader):
        if i >= max_scan:
            break
        if "is_signal" not in batch or not batch["is_signal"].any():
            continue
        groups = set()
        codes = batch["code_ids"][0]
        pad = batch["padding_mask"][0]
        is_sig = batch["is_signal"][0]
        for j in range(codes.size(0)):
            if pad[j] or not is_sig[j]:
                continue
            name = code_id_to_signal(int(codes[j]), itos)
            g = group_for_signal(name) if name else None
            if g:
                groups.add(g)
        if len(groups) >= min_groups:
            return {k: v for k, v in batch.items()}
    return None


def code_id_to_signal(code_id: int, itos: dict[int, str]) -> str | None:
    """Map vocab id → SYN_SIGNAL_* string, or None if not a signal code."""
    tok = itos.get(int(code_id))
    if tok is None:
        return None
    if str(tok).startswith("SYN_SIGNAL_"):
        return str(tok)
    return None


def group_for_signal(code: str) -> str | None:
    for group, codes in S5_PERSISTENCE_GROUPS.items():
        if code in codes:
            return group
    return None


def build_controlled_group_batch(
    group: str,
    itos: dict[int, str],
    *,
    age: float = 9.0,
    lag: float = 7.0,
    n_targets: int = 32,
    stoi: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Build a minimal batch with only ``group`` signal codes + query.

    Evaluation-only. No persistence labels are written into the batch.
    """
    if stoi is None:
        stoi = {tok: i for i, tok in itos.items()}
    codes = list(S5_PERSISTENCE_GROUPS[group])
    code_ids = [int(stoi[c]) for c in codes]
    query_id = int(stoi.get("PRED_QUERY", 2))
    code_ids = code_ids + [query_id]
    L = len(code_ids)
    code_t = torch.tensor([code_ids], dtype=torch.long)
    type_t = torch.tensor([[3] * (L - 1) + [2]], dtype=torch.long)  # signal…, query
    lag_t = torch.full((1, L), float(lag), dtype=torch.float32)
    lag_t[0, -1] = 0.0
    tau_t = torch.log1p(lag_t / 7.0)
    tau_t[0, -1] = 0.0
    is_query = torch.zeros(1, L, dtype=torch.bool)
    is_query[0, -1] = True
    is_signal = torch.zeros(1, L, dtype=torch.bool)
    is_signal[0, : L - 1] = True
    pad = torch.zeros(1, L, dtype=torch.bool)
    age_t = torch.tensor([float(age)], dtype=torch.float32)
    z_t = (age_t - 9.0) / 9.0
    labels = torch.zeros(1, n_targets, dtype=torch.float32)
    return {
        "code_ids": code_t,
        "type_ids": type_t,
        "lag_days": lag_t,
        "tau": tau_t,
        "is_query": is_query,
        "is_signal": is_signal,
        "padding_mask": pad,
        "age": age_t,
        "z_age": z_t,
        "labels": labels,
    }


def make_controlled_oracle_surface_fn(
    group: str,
    specs: list[dict[str, Any]],
    theta0: float,
    beta: float,
    scenario: str = "S5",
) -> PredictSurfaceFn:
    """Oracle surface for a controlled all-codes-in-group history."""
    from synthetic_age_temporal.ground_truth import ExampleSignals, compute_target_logits

    codes = list(S5_PERSISTENCE_GROUPS[group])

    def _fn(age: float, lag: float) -> np.ndarray:
        lags = np.full(len(codes), float(lag), dtype=np.float64)
        signals = ExampleSignals(
            codes=np.array(codes, dtype=object),
            lag_days=lags,
            tau=np.log1p(lags / 7.0),
            times=np.array([], dtype="datetime64[ns]"),
        )
        _, probs, _, _ = compute_target_logits(
            age=float(age),
            signals=signals,
            specs=specs,
            scenario=scenario,
            theta0=theta0,
            beta=beta,
            noise=np.zeros(len(specs)),
        )
        return probs

    return _fn


def make_controlled_predict_surface_fn(
    predict_fn: Callable[[dict[str, torch.Tensor]], np.ndarray],
    group: str,
    itos: dict[int, str],
    n_targets: int,
) -> PredictSurfaceFn:
    """Model surface on controlled mono-group history (no persistence labels)."""
    stoi = {tok: i for i, tok in itos.items()}
    base = build_controlled_group_batch(
        group, itos, n_targets=n_targets, stoi=stoi,
    )

    def _fn(age: float, lag: float) -> np.ndarray:
        b = clone_batch(base)
        b["age"] = torch.full_like(b["age"], float(age))
        b["z_age"] = (b["age"] - 9.0) / 9.0
        is_query = b["is_query"]
        for j in range(b["code_ids"].size(1)):
            if bool(is_query[0, j]):
                continue
            b["lag_days"][0, j] = float(lag)
            b["tau"][0, j] = float(np.log1p(float(lag) / 7.0))
        return np.asarray(predict_fn(b), dtype=np.float64)

    return _fn


def make_empty_predict_fn(
    predict_fn: Callable[[dict[str, torch.Tensor]], np.ndarray],
    itos: dict[int, str],
    n_targets: int,
) -> PredictSurfaceFn:
    """Predictions with no clinical-signal history (evaluation-only).

    Several Transformer baselines (Med-BERT, BEHRT, …) mask ``is_query`` as
    padding. A *query-only* sequence therefore has zero valid tokens and
    crashes nested-tensor encoders. We keep one non-query UNK placeholder
    (not a SYN_SIGNAL_*) plus the query token so the forward pass is valid
    while remaining empty of clinical signal content.
    """
    stoi = {tok: i for i, tok in itos.items()}
    query_id = int(stoi.get("PRED_QUERY", 2))
    unk_id = int(stoi.get("<UNK>", 1))
    # type: background=10, query=2 (see synthetic_age_temporal.dataset.TYPE_STOI)
    base = {
        "code_ids": torch.tensor([[unk_id, query_id]], dtype=torch.long),
        "type_ids": torch.tensor([[10, 2]], dtype=torch.long),
        "lag_days": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "tau": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        "is_query": torch.tensor([[False, True]]),
        "is_signal": torch.tensor([[False, False]]),
        "padding_mask": torch.tensor([[False, False]]),
        "age": torch.tensor([9.0]),
        "z_age": torch.tensor([0.0]),
        "labels": torch.zeros(1, n_targets),
    }

    def _fn(age: float, lag: float) -> np.ndarray:
        b = clone_batch(base)
        b["age"] = torch.full_like(b["age"], float(age))
        b["z_age"] = (b["age"] - 9.0) / 9.0
        return np.asarray(predict_fn(b), dtype=np.float64)

    return _fn


def lag_decay_proxy(
    surface_fn: PredictSurfaceFn,
    empty_fn: PredictSurfaceFn | None = None,
    *,
    ages: tuple[float, ...] = (5.0, 9.0, 13.0),
    lag_short: float = 7.0,
    lag_long: float = 730.0,
) -> float:
    """Black-box decay score from CF surfaces (higher ⇒ decays faster).

    Cosine persistence of the history-effect vector:
        e(lag) = p(history@lag) − p(empty)
        persistence = cos(e(short), e(long))
        decay_score = 1 − persistence

    Expect: acute > intermediate > chronic.
    """
    scores = []
    for a in ages:
        p_short = np.asarray(surface_fn(a, lag_short), dtype=np.float64)
        p_long = np.asarray(surface_fn(a, lag_long), dtype=np.float64)
        if empty_fn is not None:
            p0 = np.asarray(empty_fn(a, lag_short), dtype=np.float64)
            e_s = p_short - p0
            e_l = p_long - p0
            n_s = float(np.linalg.norm(e_s))
            n_l = float(np.linalg.norm(e_l))
            if n_s < 1e-8 or n_l < 1e-8:
                scores.append(1.0)
            else:
                cos = float(np.dot(e_s, e_l) / (n_s * n_l))
                scores.append(1.0 - cos)
        else:
            scores.append(float(np.mean(p_short - p_long)))
    return float(np.mean(scores))


def persistence_order_correct_from_surfaces(
    predict_fn: Callable[[dict[str, torch.Tensor]], np.ndarray],
    batch: dict[str, torch.Tensor],
    itos: dict[int, str],
) -> dict[str, Any]:
    """Check acute decays fastest > intermediate > chronic from CF surfaces.

    Uses controlled mono-group histories. Does **not** inspect model internals.
    ``batch`` supplies ``n_targets`` via ``labels`` shape.
    """
    n_targets = int(batch["labels"].shape[-1])
    empty_fn = make_empty_predict_fn(predict_fn, itos, n_targets)

    scores: dict[str, float] = {}
    for group in ("acute", "intermediate", "chronic"):
        surf = make_controlled_predict_surface_fn(
            predict_fn, group, itos, n_targets,
        )
        scores[group] = lag_decay_proxy(surf, empty_fn=empty_fn)

    correct = scores["acute"] > scores["intermediate"] > scores["chronic"]
    return {
        "persistence_order_correct": bool(correct),
        "decay_proxy": scores,
        "retention_proxy": {g: 1.0 - scores[g] for g in scores},
        "expected_order": ["acute", "intermediate", "chronic"],
    }


def group_surface_rmses(
    predict_fn: Callable[[dict[str, torch.Tensor]], np.ndarray],
    batch: dict[str, torch.Tensor],
    itos: dict[int, str],
    specs: list[dict[str, Any]],
    theta0: float,
    beta: float,
    *,
    ages: tuple[float, ...] = SURFACE_AGES,
    lags_days: tuple[float, ...] = SURFACE_LAGS_DAYS,
) -> dict[str, float]:
    """Surface RMSE per persistence group + mean (controlled histories)."""
    n_targets = int(batch["labels"].shape[-1])
    out: dict[str, float] = {}
    for group in ("acute", "intermediate", "chronic"):
        p_surf = make_controlled_predict_surface_fn(
            predict_fn, group, itos, n_targets,
        )
        o_surf = make_controlled_oracle_surface_fn(
            group, specs, theta0, beta, scenario="S5",
        )
        out[group] = surface_rmse(p_surf, o_surf, ages=ages, lags_days=lags_days)
    out["mean"] = float(np.mean([out["acute"], out["intermediate"], out["chronic"]]))
    return out


def oracle_group_surfaces_differ(
    batch: dict[str, torch.Tensor],
    itos: dict[int, str],
    specs: list[dict[str, Any]],
    theta0: float,
    beta: float,
    *,
    ages: tuple[float, ...] = (5.0, 9.0, 13.0),
    lags_days: tuple[float, ...] = (7.0, 90.0, 365.0, 730.0),
    min_rmse: float = 1e-4,
) -> dict[str, Any]:
    """Sanity check: oracle surfaces for the three groups are not identical."""
    del batch, itos  # controlled oracles do not need a patient template
    surfaces = {}
    for group in ("acute", "intermediate", "chronic"):
        fn = make_controlled_oracle_surface_fn(
            group, specs, theta0, beta, scenario="S5",
        )
        grid = [fn(a, lag) for a in ages for lag in lags_days]
        surfaces[group] = np.stack(grid, axis=0)

    pairwise = {}
    for g1, g2 in (("acute", "intermediate"), ("intermediate", "chronic"), ("acute", "chronic")):
        pairwise[f"{g1}_vs_{g2}"] = float(
            np.sqrt(np.mean((surfaces[g1] - surfaces[g2]) ** 2))
        )
    differ = all(v > min_rmse for v in pairwise.values())
    return {
        "differ": differ,
        "pairwise_rmse": pairwise,
        "group_theta": dict(S5_GROUP_THETA),
    }


def classify_heterogeneous_persistence(
    surface_rmse_mean: float,
    order_correct: bool,
    *,
    functional: float = FUNCTIONAL_SURFACE_RMSE,
    partial: float = PARTIAL_SURFACE_RMSE,
) -> str:
    """S5-only mechanism classification (S0–S3 thresholds unchanged elsewhere)."""
    if surface_rmse_mean < functional and order_correct:
        return "HETEROGENEOUS_PERSISTENCE_RECOVERED"
    if surface_rmse_mean < partial:
        return "PARTIAL_HETEROGENEOUS_PERSISTENCE_RECOVERY"
    return "NO_HETEROGENEOUS_PERSISTENCE_RECOVERY"


def full_s5_counterfactual_report(
    predict_fn: Callable[[dict[str, torch.Tensor]], np.ndarray],
    batch: dict[str, torch.Tensor],
    itos: dict[int, str],
    specs: list[dict[str, Any]],
    theta0: float,
    beta: float,
    *,
    cf_rmse_age: float | None = None,
    cf_rmse_lag: float | None = None,
    surface_rmse_full: float | None = None,
) -> dict[str, Any]:
    """Assemble S5-specific evaluation fields for a result record."""
    group_rmses = group_surface_rmses(
        predict_fn, batch, itos, specs, theta0, beta,
    )
    order = persistence_order_correct_from_surfaces(predict_fn, batch, itos)
    classification = classify_heterogeneous_persistence(
        group_rmses["mean"], order["persistence_order_correct"],
    )
    return {
        "S5_Surface_RMSE_acute": group_rmses["acute"],
        "S5_Surface_RMSE_intermediate": group_rmses["intermediate"],
        "S5_Surface_RMSE_chronic": group_rmses["chronic"],
        "S5_Surface_RMSE_mean": group_rmses["mean"],
        "persistence_order_correct": order["persistence_order_correct"],
        "decay_proxy": order["decay_proxy"],
        "mechanism_classification": classification,
        "cf_rmse_age": cf_rmse_age,
        "cf_rmse_lag": cf_rmse_lag,
        "surface_rmse": surface_rmse_full,
    }
