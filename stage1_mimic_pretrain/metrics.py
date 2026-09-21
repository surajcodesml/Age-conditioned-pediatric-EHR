"""Multi-label predictive metrics and age-conditioning validation tests.

Macro AUROC / AUPRC skip classes that lack both a positive and a negative
example. Invalid scores are never imputed into the macro average.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from stage1_mimic_pretrain.config import EVAL_KS, N_SHUFFLE, PROBE_AGES_YEARS, SHUFFLE_SEED

__all__ = [
    "valid_class_mask",
    "multilabel_metrics",
    "attention_magnitude_stats",
    "shuffle_age_years",
    "constant_age_years",
    "age_conditioning_tests",
    "interpret_age_test",
]


def valid_class_mask(targets: np.ndarray) -> np.ndarray:
    """True for classes with ≥1 positive and ≥1 negative over the rows."""
    pos = targets.sum(axis=0) > 0
    neg = (1.0 - targets).sum(axis=0) > 0
    return pos & neg


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based midranks, so tied scores share a rank (Mann–Whitney AUROC)."""
    n = int(scores.size)
    order = np.argsort(scores, kind="mergesort")
    sorted_s = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    n_pos = float(y.sum())
    n_neg = float(y.size) - n_pos
    if y.size == 0 or n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(_average_ranks(s)[y.astype(bool)].sum())
    return float((rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1).astype(np.int8)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())
    if y.size == 0 or n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y, dtype=np.float64)
    prec = tp / np.arange(1, y.size + 1, dtype=np.float64)
    return float(prec[y.astype(bool)].mean())


@torch.no_grad()
def ranking_per_example(logits: torch.Tensor, targets: torch.Tensor,
                        ks=EVAL_KS) -> dict[str, torch.Tensor]:
    scores = logits.float()
    n_true = targets.sum(dim=-1)
    k_max = min(int(max(ks)), scores.shape[-1])
    top = scores.topk(k_max, dim=-1).indices
    hits = targets.gather(1, top).float()
    nan = torch.tensor(float("nan"), device=scores.device)
    out: dict[str, torch.Tensor] = {"n_true": n_true.detach().cpu()}
    for k in ks:
        num = hits[:, :k].sum(dim=-1)
        out[f"recall@{k}"] = torch.where(
            n_true > 0, num / n_true.clamp(min=1.0), nan).detach().cpu()
        out[f"precision@{k}"] = (num / float(k)).detach().cpu()
    return out


def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor,
                       ks=EVAL_KS) -> dict[str, Any]:
    """BCE, micro/macro AUROC & AUPRC, precision@k / recall@k.

    Macro metrics average only over classes that have both labels present.
    """
    logits_f = logits.float()
    targets_f = targets.float()
    bce = float(F.binary_cross_entropy_with_logits(logits_f, targets_f))
    rank = ranking_per_example(logits_f, targets_f, ks=ks)
    y = targets_f.detach().cpu().numpy()
    s = logits_f.detach().cpu().numpy()
    y_flat, s_flat = y.reshape(-1), s.reshape(-1)
    valid = valid_class_mask(y)
    n_valid = int(valid.sum())
    micro_auroc = _safe_auroc(y_flat, s_flat)
    micro_auprc = _safe_auprc(y_flat, s_flat)
    if n_valid == 0:
        macro_auroc = macro_auprc = float("nan")
    else:
        aurocs, auprcs = [], []
        for c in np.flatnonzero(valid):
            aurocs.append(_safe_auroc(y[:, c], s[:, c]))
            auprcs.append(_safe_auprc(y[:, c], s[:, c]))
        macro_auroc = float(np.nanmean(aurocs))
        macro_auprc = float(np.nanmean(auprcs))
    out: dict[str, Any] = {
        "bce": bce,
        "log_loss": bce,
        "micro_auroc": micro_auroc,
        "macro_auroc": macro_auroc,
        "micro_auprc": micro_auprc,
        "macro_auprc": macro_auprc,
        "n_classes": int(y.shape[1]),
        "n_valid_classes_macro": n_valid,
        "n_classes_no_positive": int((y.sum(axis=0) == 0).sum()),
        "n_classes_no_negative": int(((1.0 - y).sum(axis=0) == 0).sum()),
        "n_examples": int(y.shape[0]),
        "prevalence": float(y.mean()),
        "mean_positives_per_example": float(y.sum(axis=1).mean()) if y.size else float("nan"),
    }
    for k in ks:
        rec = rank[f"recall@{k}"].numpy()
        prec = rank[f"precision@{k}"].numpy()
        out[f"recall@{k}"] = float(np.nanmean(rec))
        out[f"precision@{k}"] = float(np.nanmean(prec))
    return out


def class_imbalance_report(targets: torch.Tensor) -> dict[str, Any]:
    y = targets.float().cpu().numpy()
    prev = y.mean(axis=0)
    return {
        "n_examples": int(y.shape[0]),
        "n_classes": int(y.shape[1]),
        "mean_prevalence": float(prev.mean()),
        "median_prevalence": float(np.median(prev)),
        "max_prevalence": float(prev.max()) if prev.size else float("nan"),
        "frac_classes_never_positive": float((prev == 0).mean()),
        "mean_positives_per_example": float(y.sum(axis=1).mean()) if y.size else float("nan"),
        "weighting": "none (existing pipeline: unweighted BCEWithLogitsLoss, no pos_weight)",
    }


@torch.no_grad()
def attention_magnitude_stats(content_logits: torch.Tensor, temporal_bias: torch.Tensor,
                              pair_mask: torch.Tensor) -> dict[str, float]:
    """Compare |−λ(a)τ| against |q⊤k / √d| on unmasked pairs.

    ``content_logits`` is ``[B, H, L, L]``; ``temporal_bias`` is ``[B, L, L]``.
    The same bias is added to every head, so we compare against the mean over heads.
    """
    keep = pair_mask.bool()
    if keep.ndim == 4:
        keep = keep.squeeze(1)
    content = content_logits.float()
    bias = temporal_bias.float()
    if content.ndim == 4:
        keep_h = keep.unsqueeze(1).expand_as(content)
        content_kept = content[keep_h]
    else:
        content_kept = content[keep]
    if bias.ndim == 3:
        bias_kept = bias[keep]
    else:
        bias_kept = bias[keep]
    c_abs = content_kept.abs()
    b_abs = bias_kept.abs()
    c_mean = float(c_abs.mean()) if c_abs.numel() else float("nan")
    b_mean = float(b_abs.mean()) if b_abs.numel() else float("nan")
    return {
        "content_mean": c_mean,
        "content_std": float(content_kept.std()) if content_kept.numel() else float("nan"),
        "content_abs_mean": c_mean,
        "content_abs_std": float(c_abs.std()) if c_abs.numel() else float("nan"),
        "temporal_bias_mean": float(bias_kept.mean()) if bias_kept.numel() else float("nan"),
        "temporal_bias_std": float(bias_kept.std()) if bias_kept.numel() else float("nan"),
        "temporal_bias_abs_mean": b_mean,
        "temporal_bias_abs_std": float(b_abs.std()) if b_abs.numel() else float("nan"),
        "ratio_temporal_abs_to_content_abs": b_mean / (c_mean + 1e-12),
        "n_pairs": int(keep.sum()),
    }


def shuffle_age_years(age_years: torch.Tensor, attention_mask: torch.Tensor,
                      generator: torch.Generator) -> torch.Tensor:
    """Permute last-event ages across sequences; broadcast onto valid positions.

    Padding stays zero. Using a per-sequence scalar avoids mixing a short
    trajectory's pad zeros into a longer sequence's valid events.
    """
    lengths = attention_mask.long().sum(dim=1).clamp(min=1)
    rows = torch.arange(age_years.shape[0], device=age_years.device)
    age_last = age_years[rows, lengths - 1]
    perm = torch.randperm(age_last.shape[0], generator=generator, device=age_last.device)
    shuffled_last = age_last[perm]
    out = shuffled_last.unsqueeze(1).expand_as(age_years).clone()
    return out * attention_mask.float()


def constant_age_years(age_years: torch.Tensor, attention_mask: torch.Tensor,
                       value: float) -> torch.Tensor:
    out = torch.full_like(age_years, float(value))
    return out * attention_mask.float()


def interpret_age_test(*, beta: float, delta_shuffle_mean: float,
                       delta_constant_mean: float, eps_beta: float = 1e-4,
                       eps_delta: float = 1e-4) -> str:
    b_zero = abs(beta) < eps_beta
    d_zero = abs(delta_shuffle_mean) < eps_delta
    if b_zero and d_zero:
        return ("beta≈0 and ΔL_shuffle≈0: age interaction is not being used")
    if (not b_zero) and delta_shuffle_mean > eps_delta:
        return ("beta≠0 and positive ΔL_shuffle: age information contributes to prediction")
    if (not b_zero) and d_zero:
        return ("nonzero beta but negligible ΔL_shuffle: parameter changed but likely has "
                "little functional effect")
    if b_zero and abs(delta_shuffle_mean) > eps_delta:
        return ("beta≈0 but ΔL_shuffle≠0: unexpected; check that shuffle isolates the "
                "attention ages (demographics still contain age)")
    if delta_shuffle_mean < -eps_delta:
        return ("negative ΔL_shuffle: shuffled ages scored better than true ages "
                "(noise, or the learned λ(a) is not helpful on this split)")
    return ("inconclusive: inspect beta, ΔL_shuffle, and the constant-age delta")


@torch.no_grad()
def age_conditioning_tests(
    model,
    batches: list[dict],
    *,
    device: torch.device,
    age_mean: float,
    age_median: float,
    n_shuffle: int = N_SHUFFLE,
    seed: int = SHUFFLE_SEED,
) -> dict[str, Any]:
    """L_correct vs shuffled-age and constant-age BCE. Deterministic seeds."""
    was_training = model.training
    model.eval()

    def _bce(age_override=None) -> float:
        losses = []
        for batch in batches:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            override = None
            if age_override is not None:
                override = age_override(b).to(device)
            out = model(b, age_years_for_bias=override)
            losses.append(float(F.binary_cross_entropy_with_logits(
                out["code_logits"].float(), b["target_codes"].float())))
        return float(np.mean(losses)) if losses else float("nan")

    l_correct = _bce(None)
    shuffle_losses = []
    for i in range(int(n_shuffle)):
        g = torch.Generator(device="cpu").manual_seed(int(seed) + i)

        def _shuff(batch, _g=g):
            return shuffle_age_years(batch["age_years"].cpu(),
                                     batch["attention_mask"].cpu(), _g)
        shuffle_losses.append(_bce(_shuff))
    shuffle_arr = np.asarray(shuffle_losses, dtype=np.float64)
    l_shuffle_mean = float(shuffle_arr.mean()) if shuffle_arr.size else float("nan")
    l_shuffle_std = float(shuffle_arr.std(ddof=1)) if shuffle_arr.size > 1 else 0.0
    delta_shuffle = l_shuffle_mean - l_correct

    l_const_mean = _bce(lambda b: constant_age_years(
        b["age_years"], b["attention_mask"], age_mean))
    l_const_median = _bce(lambda b: constant_age_years(
        b["age_years"], b["attention_mask"], age_median))

    def _const_all(batch, value: float):
        age = constant_age_years(batch["age_years"], batch["attention_mask"], value)
        demo = batch["demographics"].clone()
        demo[..., 0] = age
        b = dict(batch)
        b["age_years"] = age
        b["demographics"] = demo
        return b

    def _bce_edited(edit_fn) -> float:
        losses = []
        for batch in batches:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            b = edit_fn(b)
            out = model(b)
            losses.append(float(F.binary_cross_entropy_with_logits(
                out["code_logits"].float(), b["target_codes"].float())))
        return float(np.mean(losses)) if losses else float("nan")

    l_const_mean_all = _bce_edited(lambda b: _const_all(b, age_mean))
    l_const_median_all = _bce_edited(lambda b: _const_all(b, age_median))

    beta = float(model.temporal.beta.detach().cpu())
    lambda0 = float(model.temporal.lambda0.detach().cpu())
    result = {
        "L_correct": l_correct,
        "L_shuffle_mean": l_shuffle_mean,
        "L_shuffle_std": l_shuffle_std,
        "L_shuffle_values": [float(x) for x in shuffle_arr],
        "delta_L_shuffle": delta_shuffle,
        "delta_L_shuffle_mean": delta_shuffle,
        "n_shuffle": int(n_shuffle),
        "shuffle_seed": int(seed),
        "shuffle_protocol": (
            "permute last-event age across sequences within each batch; "
            "broadcast onto valid positions; demographics unchanged"
        ),
        "L_constant_mean_age": l_const_mean,
        "L_constant_median_age": l_const_median,
        "delta_L_constant_mean": l_const_mean - l_correct,
        "delta_L_constant_median": l_const_median - l_correct,
        "L_constant_mean_age_including_demographics": l_const_mean_all,
        "L_constant_median_age_including_demographics": l_const_median_all,
        "delta_L_constant_mean_including_demographics": l_const_mean_all - l_correct,
        "delta_L_constant_median_including_demographics": l_const_median_all - l_correct,
        "constant_mean_age": float(age_mean),
        "constant_median_age": float(age_median),
        "lambda0": lambda0,
        "beta": beta,
        "lambda_at_ages": model.temporal.lambda_at_ages(PROBE_AGES_YEARS),
        "interpretation": interpret_age_test(
            beta=beta, delta_shuffle_mean=delta_shuffle,
            delta_constant_mean=l_const_mean - l_correct,
        ),
    }
    if was_training:
        model.train()
    return result
