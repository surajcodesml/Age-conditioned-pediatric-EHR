#!/usr/bin/env python3
"""Fine-tune TALE-EHR pretrained backbone for binary disease prediction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.dataset import (
    DiseaseClassificationDataset,
    TensorizedDiseaseClassificationDataset,
    _dataloader_worker_init,
    disease_collate,
)
from finetune.model import TALEEHRClassifier


class _TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TALE-EHR for disease classification.")
    parser.add_argument("--disease", type=str, required=True)
    parser.add_argument("--pretrained_ckpt", type=Path, required=True)
    parser.add_argument("--cohort_dir", type=Path, required=True)
    parser.add_argument(
        "--tensorized_dir",
        type=Path,
        default=None,
        help="If set, use tensorized shards under <tensorized_dir>/{train,val,test}/",
    )
    parser.add_argument("--events_parquet", type=Path, default=Path("data/processed/patient_events_rolled_full.parquet"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_workers", type=int, default=max(4, min(12, (os.cpu_count() or 16) - 2)))
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--dry_run_one_epoch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cohort_stats(cohort_path: Path) -> tuple[int, int, float]:
    con = duckdb.connect()
    try:
        n_pos, n_neg = con.execute(
            """
            SELECT
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS n_pos,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS n_neg
            FROM read_parquet(?)
            """,
            [str(cohort_path.resolve())],
        ).fetchone()
    finally:
        con.close()
    n_pos = int(n_pos or 0)
    n_neg = int(n_neg or 0)
    pos_weight = float(n_neg / max(n_pos, 1))
    return n_pos, n_neg, pos_weight


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int32)
    acc = float((y_pred == y_true).mean())
    out = {"accuracy": acc, "auroc": float("nan"), "auprc": float("nan")}
    if y_true.min() != y_true.max():
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    return out


# Developmental bands (years at the prediction index) for age-stratified breakdown.
DEV_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.0),
    ("18-25", 18.0, 26.0),
)


def age_stratified_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    age_years: np.ndarray,
    min_band_n: int = 20,
) -> dict[str, dict[str, float]]:
    """AUROC/AUPRC per developmental band, using each subject's age at the
    prediction index. Bands with too few samples or a single class are marked
    ``unreliable`` (metrics left as NaN). Additive, reused by both the MIMIC and
    Synthea fine-tune paths (extends evaluate(), not a fork)."""
    out: dict[str, dict[str, float]] = {}
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float64)
    age_years = np.asarray(age_years).astype(np.float64)
    for name, lo, hi in DEV_BANDS:
        mask = (age_years >= lo) & (age_years < hi)
        yt = y_true[mask]
        yp = y_prob[mask]
        n = int(yt.size)
        n_pos = int((yt == 1).sum())
        n_neg = int((yt == 0).sum())
        rec: dict[str, float] = {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": float(n_pos / n) if n else float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "unreliable": bool(n < min_band_n or n_pos == 0 or n_neg == 0),
        }
        if not rec["unreliable"]:
            rec["auroc"] = float(roc_auc_score(yt, yp))
            rec["auprc"] = float(average_precision_score(yt, yp))
        out[name] = rec
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])
        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        labels = batch["labels"].detach().cpu().numpy().astype(np.int32)
        probs_all.append(probs)
        labels_all.append(labels)

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_prob = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float64)
    y_true = np.concatenate(labels_all, axis=0) if labels_all else np.array([], dtype=np.int32)
    metrics = _compute_metrics(y_true, y_prob) if y_true.size > 0 else {"accuracy": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
    return mean_loss, metrics


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    subj_all: list[np.ndarray] = []
    hadm_all: list[np.ndarray] = []
    n_events_all: list[np.ndarray] = []
    age_landmark_all: list[np.ndarray] = []
    sex_all: list[np.ndarray] = []
    race_all: list[np.ndarray] = []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])
        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        labels = batch["labels"].detach().cpu().numpy().astype(np.int32)
        logits_np = logits.detach().cpu().numpy().astype(np.float64)
        lengths = batch["attention_mask"].sum(dim=1).long().clamp(min=1)
        ridx = torch.arange(lengths.shape[0], device=device)
        demo_last = batch["demographics"][ridx, lengths - 1, :].detach().cpu().numpy().astype(np.float64)
        probs_all.append(probs)
        labels_all.append(labels)
        logits_all.append(logits_np)
        subj_all.append(np.asarray(batch["subject_id"], dtype=np.int64))
        hadm_all.append(np.asarray(batch["hadm_id"], dtype=np.int64))
        n_events_all.append(np.asarray(batch["n_events_in_window"], dtype=np.int64))
        age_landmark_all.append(demo_last[:, 0])
        sex_all.append(demo_last[:, 1])
        race_all.append(demo_last[:, 2])

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_prob = np.concatenate(probs_all, axis=0) if probs_all else np.array([], dtype=np.float64)
    y_true = np.concatenate(labels_all, axis=0) if labels_all else np.array([], dtype=np.int32)
    y_logit = np.concatenate(logits_all, axis=0) if logits_all else np.array([], dtype=np.float64)
    subject_id = np.concatenate(subj_all, axis=0) if subj_all else np.array([], dtype=np.int64)
    hadm_id = np.concatenate(hadm_all, axis=0) if hadm_all else np.array([], dtype=np.int64)
    n_events_in_window = np.concatenate(n_events_all, axis=0) if n_events_all else np.array([], dtype=np.int64)
    age_at_landmark = np.concatenate(age_landmark_all, axis=0) if age_landmark_all else np.array([], dtype=np.float64)
    sex = np.concatenate(sex_all, axis=0) if sex_all else np.array([], dtype=np.float64)
    race = np.concatenate(race_all, axis=0) if race_all else np.array([], dtype=np.float64)
    metrics = _compute_metrics(y_true, y_prob) if y_true.size > 0 else {"accuracy": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
    return {
        "mean_loss": mean_loss,
        "metrics": metrics,
        "y_prob": y_prob,
        "y_true": y_true,
        "y_logit": y_logit,
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "n_events_in_window": n_events_in_window,
        "age_at_landmark": age_at_landmark,
        "sex": sex,
        "race": race,
    }


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = float(y_true.shape[0])
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        acc_bin = float(y_true[mask].mean())
        conf_bin = float(y_prob[mask].mean())
        ece += (n_bin / n_total) * abs(acc_bin - conf_bin)
    return float(ece)


def _fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray, device: torch.device) -> float:
    if val_logits.size == 0:
        return 1.0
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(val_labels, dtype=torch.float32, device=device)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-6)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().cpu().item())


def _bootstrap_metric_cis(y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, list[float]]:
    if y_true.size == 0:
        return {"auroc_ci": [float("nan"), float("nan")], "auprc_ci": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    auroc_vals: list[float] = []
    auprc_vals: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = y_prob[idx]
        if y_b.min() != y_b.max():
            auroc_vals.append(float(roc_auc_score(y_b, p_b)))
        auprc_vals.append(float(average_precision_score(y_b, p_b)))

    def _pct(vals: list[float]) -> list[float]:
        if not vals:
            return [float("nan"), float("nan")]
        arr = np.asarray(vals, dtype=np.float64)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {"auroc_ci": _pct(auroc_vals), "auprc_ci": _pct(auprc_vals)}


def _resolve_age_conditioned_backbone(model: TALEEHRClassifier) -> nn.Module | None:
    backbone = model.backbone
    taa = getattr(backbone, "time_aware_attention", None)
    tagg = getattr(backbone, "temporal_aggregation", None)
    if taa is None or tagg is None:
        return None
    if not hasattr(taa, "age_emb") or not hasattr(tagg, "age_emb"):
        return None
    if not hasattr(taa, "temporal_weight") or not hasattr(tagg, "temporal_weight"):
        return None
    if not hasattr(taa.temporal_weight, "age_coeff_gen") or not hasattr(tagg.temporal_weight, "age_coeff_gen"):
        return None
    return backbone


def _compute_alpha_band_spread(backbone: nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    age_years = batch["demographics"][..., 0]
    mask = batch["attention_mask"].bool()
    coeff_base = backbone.time_aware_attention.temporal_weight.coefficients
    with torch.no_grad():
        gamma = backbone.time_aware_attention.age_emb(age_years.clamp(min=0.0))
        delta_alpha = backbone.time_aware_attention.temporal_weight.age_coeff_gen(gamma)
        alpha = coeff_base + delta_alpha

    alpha_valid = alpha[mask]
    ages_valid = age_years[mask]
    if alpha_valid.numel() == 0:
        return {"alpha_band_mean_vectors": {}, "alpha_band_pairwise_l2": {}, "alpha_band_vector_variance_mean": float("nan")}

    buckets = [
        ("neonate", (ages_valid < (1.0 / 12.0))),
        ("infant", ((ages_valid >= (1.0 / 12.0)) & (ages_valid < 2.0))),
        ("child", ((ages_valid >= 2.0) & (ages_valid < 12.0))),
        ("adolescent", ((ages_valid >= 12.0) & (ages_valid < 18.0))),
        ("young_adult", ((ages_valid >= 18.0) & (ages_valid < 40.0))),
        ("middle_age", ((ages_valid >= 40.0) & (ages_valid < 65.0))),
        ("older_adult", (ages_valid >= 65.0)),
    ]
    band_means: dict[str, torch.Tensor] = {}
    band_vectors_json: dict[str, list[float]] = {}
    for name, bmask in buckets:
        if bool(bmask.any()):
            mean_vec = alpha_valid[bmask].mean(dim=0)
            band_means[name] = mean_vec
            band_vectors_json[name] = [float(v) for v in mean_vec.detach().cpu().tolist()]
    pairwise_l2: dict[str, float] = {}
    names = sorted(band_means.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}__vs__{names[j]}"
            pairwise_l2[key] = float(torch.norm(band_means[names[i]] - band_means[names[j]], p=2).detach().cpu().item())

    if len(names) >= 2:
        stacked = torch.stack([band_means[n] for n in names], dim=0)
        vector_var_mean = float(stacked.var(dim=0, unbiased=False).mean().detach().cpu().item())
    else:
        vector_var_mean = float("nan")
    return {
        "alpha_band_mean_vectors": band_vectors_json,
        "alpha_band_pairwise_l2": pairwise_l2,
        "alpha_band_vector_variance_mean": vector_var_mean,
    }


def _compute_decay_grid(backbone: nn.Module, device: torch.device) -> tuple[list[dict[str, float]], dict[str, Any]]:
    ages = [0.5, 2.0, 8.0, 15.0, 40.0, 65.0]
    days = np.linspace(0.0, 1825.0, num=366, dtype=np.float64)
    weeks = days / 7.0
    log_dt = torch.as_tensor(np.log1p(days / 7.0), dtype=torch.float32, device=device).unsqueeze(0)
    rows: list[dict[str, float]] = []
    effective_window_weeks: dict[str, float] = {}
    all_w_by_age: dict[float, np.ndarray] = {}

    for age in ages:
        age_t = torch.tensor([age], dtype=torch.float32, device=device)
        with torch.no_grad():
            age_feat = backbone.time_aware_attention.age_emb(age_t)
            w = backbone.time_aware_attention.temporal_weight(log_dt, age_feat).squeeze(0).detach().cpu().numpy().astype(np.float64)
        all_w_by_age[age] = w
        below = np.where(w <= 0.5)[0]
        effective_window_weeks[str(age)] = float(weeks[int(below[0])]) if below.size > 0 else float("nan")
        for d, wk, wi in zip(days, weeks, w):
            rows.append({"days": float(d), "weeks": float(wk), "age": float(age), "w": float(wi)})

    week_buckets = [
        ("0_1w", 0.0, 1.0),
        ("1_4w", 1.0, 4.0),
        ("4_12w", 4.0, 12.0),
        ("12_26w", 12.0, 26.0),
        ("26_52w", 26.0, 52.0),
        ("52_261w", 52.0, 261.0),
    ]
    w_matrix = np.stack([all_w_by_age[a] for a in ages], axis=0)  # [A, D]
    saturation: dict[str, dict[str, float]] = {}
    for name, lo, hi in week_buckets:
        if hi >= week_buckets[-1][2]:
            mask = (weeks >= lo) & (weeks <= hi)
        else:
            mask = (weeks >= lo) & (weeks < hi)
        vals = w_matrix[:, mask].reshape(-1)
        if vals.size == 0:
            saturation[name] = {"frac_w_gt_0_9": float("nan"), "frac_w_lt_0_1": float("nan")}
            continue
        saturation[name] = {
            "frac_w_gt_0_9": float((vals > 0.9).mean()),
            "frac_w_lt_0_1": float((vals < 0.1).mean()),
        }
    return rows, {"effective_window_weeks_by_age": effective_window_weeks, "saturation_by_dt_bucket": saturation}


def _write_predictions_parquet(path: Path, eval_out: dict[str, Any]) -> None:
    subj = eval_out["subject_id"].astype(np.int64).tolist()
    hadm = eval_out["hadm_id"].astype(np.int64).tolist()
    label = eval_out["y_true"].astype(np.int32).tolist()
    prob = eval_out["y_prob"].astype(np.float64).tolist()
    logit = eval_out["y_logit"].astype(np.float64).tolist()
    n_events = eval_out["n_events_in_window"].astype(np.int64).tolist()
    age = eval_out["age_at_landmark"].astype(np.float64).tolist()
    sex = eval_out["sex"].astype(np.float64).tolist()
    race = eval_out["race"].astype(np.float64).tolist()
    rows = list(zip(subj, hadm, label, prob, logit, n_events, age, sex, race))
    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE pred_df (
                subject_id BIGINT,
                hadm_id BIGINT,
                label INTEGER,
                prob DOUBLE,
                logit DOUBLE,
                n_events_in_window BIGINT,
                age_at_landmark DOUBLE,
                sex DOUBLE,
                race DOUBLE
            )
            """
        )
        if rows:
            con.executemany(
                "INSERT INTO pred_df VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        con.execute("COPY pred_df TO ? (FORMAT PARQUET)", [str(path.resolve())])
    finally:
        con.close()


def _write_decay_grid_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["days", "weeks", "age", "w"])
        writer.writeheader()
        writer.writerows(rows)


def _check_gradients(model: TALEEHRClassifier) -> tuple[bool, bool]:
    has_classifier_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0 for p in model.classifier.parameters())
    has_backbone_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0 for p in model.backbone.parameters() if p.requires_grad)
    return has_classifier_grad, has_backbone_grad


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    run_name = f"{args.disease}_run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.run_dir or (Path("checkpoints/finetune") / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    console_log_path = run_dir / "console.log"
    console_fh = open(console_log_path, "w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, console_fh)
    sys.stderr = _TeeStream(orig_stderr, console_fh)

    try:
        run_t0 = time.perf_counter()

        print(f"[run] checkpoints -> {run_dir}", flush=True)
        print(f"[run] console log -> {console_log_path}", flush=True)

        train_cohort = args.cohort_dir / "train_cohort.parquet"
        val_cohort = args.cohort_dir / "val_cohort.parquet"
        test_cohort = args.cohort_dir / "test_cohort.parquet"
        for p in (train_cohort, val_cohort, test_cohort, args.events_parquet, args.vocab_path, args.pretrained_ckpt):
            if not p.exists():
                raise FileNotFoundError(f"Missing required path: {p}")

        if args.tensorized_dir is not None:
            train_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "train", max_seq_len=args.max_seq_len, shard_cache_size=4)
            val_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "val", max_seq_len=args.max_seq_len, shard_cache_size=4)
            test_ds = TensorizedDiseaseClassificationDataset(args.tensorized_dir / "test", max_seq_len=args.max_seq_len, shard_cache_size=4)
            print(f"[data] tensorized_dir={args.tensorized_dir}", flush=True)
        else:
            train_ds = DiseaseClassificationDataset(train_cohort, args.events_parquet, args.vocab_path, args.max_seq_len)
            val_ds = DiseaseClassificationDataset(val_cohort, args.events_parquet, args.vocab_path, args.max_seq_len)
            test_ds = DiseaseClassificationDataset(test_cohort, args.events_parquet, args.vocab_path, args.max_seq_len)
            print("[data] mode=on_the_fly_duckdb", flush=True)

        loader_kw = dict(
            batch_size=args.batch_size,
            collate_fn=disease_collate,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
            worker_init_fn=_dataloader_worker_init,
        )
        if args.num_workers > 0:
            loader_kw["multiprocessing_context"] = "spawn"
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

        n_pos, n_neg, pos_weight = _cohort_stats(train_cohort)
        print(f"[train cohort] positives={n_pos:,} negatives={n_neg:,} pos_weight={pos_weight:.6f}", flush=True)

        device = torch.device(args.device)
        model = TALEEHRClassifier(args.pretrained_ckpt, freeze_backbone=False).to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": model.classifier.parameters(), "lr": args.lr_head},
            ]
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Required pre-train sanity: one forward/loss/backward check.
        first_batch = next(iter(train_loader))
        first_batch = _move_batch_to_device(first_batch, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
        with ctx:
            logits = model(first_batch)
            loss = criterion(logits, first_batch["labels"])
        if logits.ndim != 1 or logits.shape[0] != first_batch["labels"].shape[0]:
            raise RuntimeError(f"Expected logits [B], got {tuple(logits.shape)}")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in sanity check")
        scaler.scale(loss).backward()
        has_cls_grad, has_backbone_grad = _check_gradients(model)
        sanity_elapsed = time.perf_counter() - run_t0
        print(
            f"[sanity] t+{sanity_elapsed:.1f}s logits_shape={tuple(logits.shape)} "
            f"loss={float(loss.item()):.6f} classifier_grad={has_cls_grad} backbone_grad={has_backbone_grad}",
            flush=True,
        )
        if not has_cls_grad or not has_backbone_grad:
            raise RuntimeError("Gradient check failed for classifier/backbone")
        optimizer.zero_grad(set_to_none=True)

        best_val_auroc = -float("inf")
        best_epoch = 0
        best_ckpt_path = run_dir / "best.pt"
        history: list[dict[str, float | int]] = []

        total_epochs = 1 if args.dry_run_one_epoch else args.epochs
        for epoch in range(1, total_epochs + 1):
            ep_start = time.perf_counter()
            model.train()
            train_losses: list[float] = []

            for step, batch in enumerate(train_loader, start=1):
                batch = _move_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
                with ctx:
                    logits = model(batch)
                    loss = criterion(logits, batch["labels"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.item()))
                if step % 200 == 0 or step == 1:
                    step_elapsed = time.perf_counter() - run_t0
                    print(
                        f"  ep{epoch:03d} step {step:6d} t+{step_elapsed:.1f}s loss={float(loss.item()):.6f}",
                        flush=True,
                    )

            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            elapsed = time.perf_counter() - ep_start

            msg = (
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | val_AUROC={val_metrics['auroc']:.6f} | "
                f"val_AUPRC={val_metrics['auprc']:.6f} | val_acc={val_metrics['accuracy']:.6f} | "
                f"time={elapsed:.1f}s"
            )
            print(msg, flush=True)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auroc": float(val_metrics["auroc"]),
                    "val_auprc": float(val_metrics["auprc"]),
                    "val_acc": float(val_metrics["accuracy"]),
                }
            )

            epoch_ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "args": vars(args),
                },
                epoch_ckpt_path,
            )
            print(f"[ckpt] Saved epoch checkpoint: {epoch_ckpt_path}", flush=True)

            val_auroc = float(val_metrics["auroc"])
            rank_auroc = val_auroc if np.isfinite(val_auroc) else -float("inf")
            if rank_auroc > best_val_auroc:
                best_val_auroc = rank_auroc
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "val_loss": val_loss,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )
                print(f"[best] Saved checkpoint: {best_ckpt_path}", flush=True)

        if not best_ckpt_path.exists():
            raise RuntimeError("No best checkpoint was saved.")
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        val_full = evaluate_full(model, val_loader, criterion, device)
        test_full = evaluate_full(model, test_loader, criterion, device)
        test_loss = float(test_full["mean_loss"])
        test_metrics = test_full["metrics"]
        _write_predictions_parquet(run_dir / "test_predictions.parquet", test_full)

        y_test = test_full["y_true"].astype(np.int32)
        p_test = test_full["y_prob"].astype(np.float64)
        z_test = test_full["y_logit"].astype(np.float64)
        n_events_test = test_full["n_events_in_window"].astype(np.float64)
        length_only_auroc = float("nan")
        if y_test.size > 0 and y_test.min() != y_test.max():
            length_only_auroc = float(roc_auc_score(y_test, n_events_test))
        leakage_gap = float(test_metrics["auroc"]) - length_only_auroc if np.isfinite(float(test_metrics["auroc"])) and np.isfinite(length_only_auroc) else float("nan")

        temperature = _fit_temperature(
            val_full["y_logit"].astype(np.float64),
            val_full["y_true"].astype(np.float64),
            device=device,
        )
        p_test_ts = _sigmoid_np(z_test / max(temperature, 1e-6))
        brier_raw = float(brier_score_loss(y_test, p_test)) if y_test.size > 0 else float("nan")
        brier_ts = float(brier_score_loss(y_test, p_test_ts)) if y_test.size > 0 else float("nan")
        ece_raw = _compute_ece(y_test.astype(np.float64), p_test, n_bins=15)
        ece_ts = _compute_ece(y_test.astype(np.float64), p_test_ts, n_bins=15)
        bootstrap = _bootstrap_metric_cis(y_test, p_test, n_bootstrap=1000, seed=42)

        decay_json_path = run_dir / "decay_alpha.json"
        decay_grid_csv_path = run_dir / "decay_kernel_grid.csv"
        decay_json: dict[str, Any] = {}
        decay_rows: list[dict[str, float]] = []
        age_backbone = _resolve_age_conditioned_backbone(model)
        if age_backbone is None:
            print("[decay] vanilla backbone, skipping", flush=True)
            decay_json = {"status": "vanilla_backbone_skipped"}
        else:
            from model.age_diagnostics import compute_alpha_delta_stats

            one_test_batch = next(iter(test_loader))
            one_test_batch = _move_batch_to_device(one_test_batch, device)
            alpha_stats = compute_alpha_delta_stats(age_backbone, one_test_batch)
            alpha_spread = _compute_alpha_band_spread(age_backbone, one_test_batch)
            decay_rows, decay_grid_stats = _compute_decay_grid(age_backbone, device)
            decay_json = {
                "status": "ok",
                "alpha_delta_stats": alpha_stats,
                "alpha_spread": alpha_spread,
                **decay_grid_stats,
            }
        _write_decay_grid_csv(decay_grid_csv_path, decay_rows)
        with decay_json_path.open("w", encoding="utf-8") as f:
            json.dump(decay_json, f, indent=2)

        test_extended = {
            "length_only_auroc": float(length_only_auroc),
            "leakage_gap": float(leakage_gap),
            "auroc_ci": [float(bootstrap["auroc_ci"][0]), float(bootstrap["auroc_ci"][1])],
            "auprc_ci": [float(bootstrap["auprc_ci"][0]), float(bootstrap["auprc_ci"][1])],
            "brier_raw": float(brier_raw),
            "brier_ts": float(brier_ts),
            "ece_raw": float(ece_raw),
            "ece_ts": float(ece_ts),
            "temperature": float(temperature),
        }

        print(
            f"[test @ best epoch {best_epoch}] loss={test_loss:.6f} "
            f"AUROC={test_metrics['auroc']:.6f} AUPRC={test_metrics['auprc']:.6f} "
            f"acc={test_metrics['accuracy']:.6f}",
            flush=True,
        )
        print("[calibration] raw probabilities are uncalibrated by construction (training used pos_weight).", flush=True)
        print(
            "[test_extended] "
            f"length_only_AUROC={test_extended['length_only_auroc']:.6f} "
            f"leakage_gap={test_extended['leakage_gap']:.6f} "
            f"AUROC_CI95=({test_extended['auroc_ci'][0]:.6f},{test_extended['auroc_ci'][1]:.6f}) "
            f"AUPRC_CI95=({test_extended['auprc_ci'][0]:.6f},{test_extended['auprc_ci'][1]:.6f}) "
            f"brier_raw={test_extended['brier_raw']:.6f} brier_ts={test_extended['brier_ts']:.6f} "
            f"ece_raw={test_extended['ece_raw']:.6f} ece_ts={test_extended['ece_ts']:.6f} "
            f"T={test_extended['temperature']:.6f}",
            flush=True,
        )

        test_age_stratified = age_stratified_metrics(
            test_full["y_true"],
            test_full["y_prob"],
            np.clip(test_full["age_at_landmark"].astype(np.float64), 0.0, None),
        )
        print("[age_stratified test] AUROC by developmental band:", flush=True)
        for band, rec in test_age_stratified.items():
            flag = " (unreliable)" if rec["unreliable"] else ""
            print(
                f"  {band:>6}: n={int(rec['n']):5d} prev={rec['prevalence']:.3f} "
                f"AUROC={rec['auroc']:.4f} AUPRC={rec['auprc']:.4f}{flag}",
                flush=True,
            )

        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_val_auroc": best_val_auroc,
                    "history": history,
                    "test_loss": test_loss,
                    "test_metrics": test_metrics,
                    "test_extended": test_extended,
                    "test_age_stratified": test_age_stratified,
                },
                f,
                indent=2,
            )
        return 0
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        console_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
