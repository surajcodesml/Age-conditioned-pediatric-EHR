"""Stage-2 evaluation: full metrics, age/history strata, age tests at z_P(9)=0."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from stage1_mimic_pretrain.metrics import (
    age_conditioning_tests,
    attention_magnitude_stats,
    class_imbalance_report,
    multilabel_metrics,
    ranking_per_example,
)
from stage2_nch.config import (
    CONSTANT_ATTENTION_AGE_YEARS,
    EVAL_KS,
    N_SHUFFLE,
    PROBE_AGES_YEARS,
    SHUFFLE_SEED,
    age_band_name,
)
from stage2_nch.metrics import history_bin_name, pos_neg_bce


@torch.no_grad()
def evaluate_loader(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 0,
    max_metric_examples: int = 4096,
    ks=EVAL_KS,
    history_edges: list[float] | None = None,
    collect_strata: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    bce_sum = pos_sum = neg_sum = 0.0
    n_pos = n_neg = 0
    n_batches = 0
    n_examples = 0
    rank_acc: dict[str, list[torch.Tensor]] = {}
    logit_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    n_metric = 0
    strata_rows: list[dict[str, Any]] = []
    t0 = torch.cuda.Event(enable_timing=False) if False else None
    _ = t0
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        out = model(batch)
        logits = out["code_logits"].float()
        targets = batch["target_codes"].float()
        pn = pos_neg_bce(logits, targets)
        bce_sum += pn["bce"]
        if pn["n_positive_labels"]:
            pos_sum += pn["positive_bce"] * pn["n_positive_labels"]
            n_pos += pn["n_positive_labels"]
        if pn["n_negative_labels"]:
            neg_sum += pn["negative_bce"] * pn["n_negative_labels"]
            n_neg += pn["n_negative_labels"]
        n_batches += 1
        n_examples += int(logits.shape[0])
        rank = ranking_per_example(logits, targets, ks=ks)
        for key, val in rank.items():
            rank_acc.setdefault(key, []).append(val)
        take = min(int(logits.shape[0]), max(0, max_metric_examples - n_metric))
        if take > 0:
            logit_chunks.append(logits[:take].detach().cpu())
            target_chunks.append(targets[:take].detach().cpu())
            n_metric += take
        if collect_strata:
            last_age = batch.get("last_age_years")
            n_in = batch.get("n_input_events")
            n_vis = batch.get("n_prior_visits")
            pid = batch.get("patient_id")
            per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
            for j in range(logits.shape[0]):
                age = float(last_age[j].cpu()) if last_age is not None else float("nan")
                nev = int(n_in[j].cpu()) if n_in is not None else 0
                strata_rows.append({
                    "patient_id": int(pid[j].cpu()) if pid is not None else -1,
                    "bce": float(per[j].cpu()),
                    "age_years": age,
                    "age_band": age_band_name(age),
                    "n_input_events": nev,
                    "n_prior_visits": int(n_vis[j].cpu()) if n_vis is not None else 0,
                    "history_bin": (history_bin_name(nev, history_edges)
                                    if history_edges is not None else "all"),
                    "n_true": float(rank["n_true"][j]),
                    **{f"recall@{k}": float(rank[f"recall@{k}"][j]) for k in ks},
                    **{f"precision@{k}": float(rank[f"precision@{k}"][j]) for k in ks},
                })
                if take > 0 and j < take:
                    strata_rows[-1]["_logit_idx"] = n_metric - take + j
    if was_training:
        model.train()
    result: dict[str, Any] = {
        "bce": bce_sum / max(n_batches, 1),
        "positive_bce": pos_sum / max(n_pos, 1) if n_pos else float("nan"),
        "negative_bce": neg_sum / max(n_neg, 1) if n_neg else float("nan"),
        "n_batches": n_batches,
        "n_examples": n_examples,
    }
    if rank_acc:
        for key, parts in rank_acc.items():
            cat = torch.cat(parts)
            result[key] = float(cat.float().mean() if key == "n_true" else torch.nanmean(cat))
    if logit_chunks:
        ml = multilabel_metrics(torch.cat(logit_chunks), torch.cat(target_chunks), ks=ks)
        ml["bce_on_metric_cap"] = ml["bce"]
        ml["bce"] = result["bce"]
        ml["log_loss"] = result["bce"]
        result.update(ml)
        result["imbalance"] = class_imbalance_report(torch.cat(target_chunks))
    if collect_strata:
        result["strata_rows"] = strata_rows
        if logit_chunks:
            result["_cap_logits"] = torch.cat(logit_chunks)
            result["_cap_targets"] = torch.cat(target_chunks)
    return result


def summarize_strata(rows: list[dict[str, Any]], logits: torch.Tensor | None,
                     targets: torch.Tensor | None, *, key: str,
                     ks=EVAL_KS) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(str(row[key]), []).append(i)
    out = {}
    for name, idxs in sorted(groups.items(), key=lambda kv: kv[0]):
        sub = [rows[i] for i in idxs]
        pats = {r["patient_id"] for r in sub}
        rec: dict[str, Any] = {
            "n_examples": len(sub),
            "n_patients": len(pats),
            "bce": float(np.mean([r["bce"] for r in sub])),
            "precision@5": float(np.nanmean([r["precision@5"] for r in sub])),
            "recall@5": float(np.nanmean([r["recall@5"] for r in sub])),
        }
        cap_idx = [rows[i].get("_logit_idx") for i in idxs]
        cap_idx = [int(j) for j in cap_idx if j is not None]
        if logits is not None and targets is not None and cap_idx:
            ml = multilabel_metrics(logits[cap_idx], targets[cap_idx], ks=ks)
            rec["micro_auprc"] = ml["micro_auprc"]
            rec["micro_auroc"] = ml["micro_auroc"]
            rec["n_valid_classes_macro"] = ml["n_valid_classes_macro"]
        out[name] = rec
    return out


@torch.no_grad()
def epoch_diagnostics(model, batch: dict, device: torch.device) -> dict[str, Any]:
    b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    out = model(b, need_diagnostics=True)
    stats = {}
    if "content_logits" in out and "temporal_bias" in out and "pair_mask" in out:
        stats = attention_magnitude_stats(out["content_logits"], out["temporal_bias"],
                                          out["pair_mask"])
        bias = out["temporal_bias"].float()
        keep = out["pair_mask"].bool()
        kept = bias[keep]
        if kept.numel():
            q = torch.quantile(kept.float(), torch.tensor([0.1, 0.5, 0.9], device=kept.device))
            stats["temporal_bias_q10"] = float(q[0])
            stats["temporal_bias_q50"] = float(q[1])
            stats["temporal_bias_q90"] = float(q[2])
        content = out["content_logits"].float()
        if content.ndim == 4:
            keep_h = keep.unsqueeze(1).expand_as(content)
            ck = content[keep_h]
        else:
            ck = content[keep]
        if ck.numel():
            q = torch.quantile(ck, torch.tensor([0.1, 0.5, 0.9], device=ck.device))
            stats["content_q10"] = float(q[0])
            stats["content_q50"] = float(q[1])
            stats["content_q90"] = float(q[2])
    stats["lambda0"] = float(model.temporal.lambda0.detach().cpu())
    stats["beta"] = float(model.temporal.beta.detach().cpu())
    stats["lambda_at_ages"] = model.temporal.lambda_at_ages(PROBE_AGES_YEARS)
    stats["age_last_mean"] = float(out["age_last"].float().mean())
    return stats


def run_age_tests(model, val_batches: list[dict], device: torch.device,
                  n_shuffle: int = N_SHUFFLE, seed: int = SHUFFLE_SEED) -> dict[str, Any]:
    """Primary constant-age uses 9 years so z_P(9)=0. Median is reported too."""
    ages = []
    for b in val_batches:
        mask = b["attention_mask"].bool()
        ages.append(b["age_years"][mask].cpu().numpy())
    age_median = float(np.median(np.concatenate(ages))) if ages else CONSTANT_ATTENTION_AGE_YEARS
    result = age_conditioning_tests(
        model, val_batches, device=device,
        age_mean=CONSTANT_ATTENTION_AGE_YEARS,
        age_median=age_median,
        n_shuffle=n_shuffle, seed=seed,
    )
    result["constant_preferred_age"] = CONSTANT_ATTENTION_AGE_YEARS
    result["z_P_at_constant_preferred"] = 0.0
    result["lambda_at_ages"] = model.temporal.lambda_at_ages(PROBE_AGES_YEARS)
    return result
