"""Validation / test evaluation, patient-ID dump, and age-conditioning tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_new.diagnostics import write_json
from stage1_mimic_pretrain.config import EVAL_KS, N_SHUFFLE, PROBE_AGES_YEARS, SHUFFLE_SEED
from stage1_mimic_pretrain.metrics import (
    age_conditioning_tests,
    attention_magnitude_stats,
    class_imbalance_report,
    multilabel_metrics,
    ranking_per_example,
)


def collect_subject_ids(split_dir: Path) -> np.ndarray:
    """Unique patient IDs from shard ``subject_id`` arrays (patient-level split)."""
    ids: list[np.ndarray] = []
    for path in sorted(Path(split_dir).glob("shard_*.npz")):
        z = np.load(path, mmap_mode="r", allow_pickle=False)
        if "subject_id" not in z.files:
            z.close()
            raise AssertionError(f"{path} has no subject_id; cannot enforce patient splits")
        ids.append(np.asarray(z["subject_id"], dtype=np.int64))
        z.close()
    if not ids:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(ids))


def assert_disjoint_patient_splits(splits: dict[str, np.ndarray]) -> dict[str, int]:
    names = list(splits)
    overlap: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = int(np.intersect1d(splits[a], splits[b]).size)
            overlap[f"{a}&{b}"] = n
            if n:
                raise AssertionError(
                    f"patient leakage: {n} subject_ids in both {a} and {b}")
    return overlap


def save_split_ids(run_dir: Path, tensorized_dir: Path) -> dict[str, Any]:
    splits = {}
    for name in ("train", "val", "test"):
        d = Path(tensorized_dir) / name
        if d.exists() and any(d.glob("shard_*.npz")):
            splits[name] = collect_subject_ids(d)
            np.save(run_dir / f"{name}_subject_ids.npy", splits[name])
            (run_dir / f"{name}_subject_ids.txt").write_text(
                "\n".join(str(int(x)) for x in splits[name]) + "\n")
    overlap = assert_disjoint_patient_splits(splits) if len(splits) > 1 else {}
    summary = {k: int(v.size) for k, v in splits.items()}
    write_json(run_dir / "split_ids.json", {"n_patients": summary, "overlap": overlap})
    return {"n_patients": summary, "overlap": overlap}


@torch.no_grad()
def evaluate_loader(model, loader: DataLoader, device: torch.device, *,
                    max_batches: int = 0, max_metric_examples: int = 2048,
                    ks=EVAL_KS) -> dict[str, Any]:
    """BCE over the scanned batches; ranking + AUROC/AUPRC on a logit cap.

    Full-val 52k × 30k logits cannot be materialised (same constraint as
    ``model_new.eval_pretrain``). Ranking metrics accumulate per example without
    storing the score matrix; AUROC/AUPRC use the first ``max_metric_examples``.
    """
    was_training = model.training
    model.eval()
    bce_sum = 0.0
    n_batches = 0
    rank_acc: dict[str, list[torch.Tensor]] = {}
    logit_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    n_metric = 0
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        out = model(batch)
        logits = out["code_logits"].float()
        targets = batch["target_codes"].float()
        bce_sum += float(F.binary_cross_entropy_with_logits(logits, targets))
        n_batches += 1
        rank = ranking_per_example(logits, targets, ks=ks)
        for key, val in rank.items():
            rank_acc.setdefault(key, []).append(val)
        if n_metric < max_metric_examples:
            take = min(int(logits.shape[0]), max_metric_examples - n_metric)
            logit_chunks.append(logits[:take].detach().cpu())
            target_chunks.append(targets[:take].detach().cpu())
            n_metric += take
    if was_training:
        model.train()
    result: dict[str, Any] = {
        "bce": bce_sum / max(n_batches, 1),
        "n_batches": n_batches,
    }
    if rank_acc:
        for key, parts in rank_acc.items():
            cat = torch.cat(parts)
            if key == "n_true":
                result[key] = float(cat.float().mean())
            else:
                result[key] = float(torch.nanmean(cat))
    if logit_chunks:
        ml = multilabel_metrics(torch.cat(logit_chunks), torch.cat(target_chunks), ks=ks)
        # Keep the loader-wide BCE (more batches) and overlay ranking/AUROC from the cap.
        ml["bce_on_metric_cap"] = ml["bce"]
        ml["bce"] = result["bce"]
        ml["log_loss"] = result["bce"]
        result.update(ml)
        result["imbalance"] = class_imbalance_report(torch.cat(target_chunks))
    return result


@torch.no_grad()
def collect_batches(loader: DataLoader, device: torch.device, max_batches: int) -> list[dict]:
    out = []
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        out.append({k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()})
        _ = device
    return out


@torch.no_grad()
def epoch_diagnostics(model, batch: dict, device: torch.device) -> dict[str, Any]:
    b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    out = model(b, need_diagnostics=True)
    stats = {}
    if "content_logits" in out and "temporal_bias" in out and "pair_mask" in out:
        stats = attention_magnitude_stats(out["content_logits"], out["temporal_bias"],
                                          out["pair_mask"])
    stats["lambda0"] = float(model.temporal.lambda0.detach().cpu())
    stats["beta"] = float(model.temporal.beta.detach().cpu())
    stats["lambda_at_ages"] = model.temporal.lambda_at_ages(PROBE_AGES_YEARS)
    stats["age_last_mean"] = float(out["age_last"].float().mean())
    return stats


def run_age_tests(model, val_batches: list[dict], device: torch.device,
                  age_mean: float, age_median: float,
                  n_shuffle: int = N_SHUFFLE, seed: int = SHUFFLE_SEED) -> dict[str, Any]:
    return age_conditioning_tests(
        model, val_batches, device=device, age_mean=age_mean, age_median=age_median,
        n_shuffle=n_shuffle, seed=seed,
    )


def maybe_subset(ds: Dataset, max_examples: int, seed: int) -> Dataset:
    if max_examples <= 0 or max_examples >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=int(max_examples), replace=False))
    from torch.utils.data import Subset
    return Subset(ds, idx.tolist())
