#!/usr/bin/env python3
"""Frozen-representation linear probe.

Measures what a **pretrained** encoder contains, separately from fine-tuning. For each
arm the encoder is loaded (or randomly initialised as a floor), frozen, and two patient
vectors are extracted:

* ``h_pool`` -- ``AttentionPooling`` output, before demographic combination and before
  any arm-specific concatenation.
* ``h_head`` -- what the prediction head sees, minus the demographic sub-vector.

A sklearn L2 logistic regression is fit on each (arm, representation, target); the
encoder is never trained or updated. All printing and JSON go through ``diagnostics``.

    python -m model_new.probe --smoke
    python -m model_new.probe --run_name probe_s0 \\
        --ckpt_map vanilla=.../epoch_008.pt kernel=.../epoch_006.pt ...
"""

from __future__ import annotations

import argparse
import hashlib
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from model_new import diagnostics as D
from model_new.arms import ARMS
from model_new.data import (
    TensorizedPretrainDataset, dataloader_worker_init, make_collate,
)
from model_new.data_finetune import TensorizedFinetuneDataset, make_finetune_collate
from model_new.eval_pretrain import build_model, model_kwargs_from_config
from model_new.model import DKMModel
from model_new.train import set_seed
from model_new.train_finetune import load_backbone

REPO_ROOT = Path(__file__).resolve().parents[1]

REPR_NAMES = ("h_pool", "h_head")
C_GRID = np.logspace(-4, 4, 9)
N_BOOTSTRAP_PROBE = 1000
DEFAULT_TRAIN_SUBSAMPLE = 50_000
MAX_ITER = 10_000

PROBE_ASYMMETRY_NOTE = (
    "h_pool is the AttentionPooling output before demographic combination and before "
    "any arm-specific concatenation. kernel puts age into the attention kernel, so age "
    "is present in h_pool by construction. additive concatenates the generator to the "
    "patient vector after pooling, so that pathway is excluded from h_pool (additive "
    "h_pool ≈ vanilla). vanilla and random_constant have no real age signal in h_pool. "
    "h_head is what the prediction head sees minus the demographic sub-vector, so it "
    "includes additive's concat. Report both."
)
DOWNSTREAM_ENDPOINT_NOTE = (
    "downstream_endpoint_* targets are not the pretraining task and the encoder was "
    "never fine-tuned for them. Pretraining is multi-label BCE over the full vocabulary, "
    "so every code appeared in the objective; these endpoints are not 'held out from "
    "pretraining'."
)

TARGET_LAST_EVENT_RECENCY = "last_event_recency"
TARGET_RECORD_SPAN = "record_span"
TARGET_AGE_BAND = "developmental_age_band"
TARGET_DOWNSTREAM_ID = "downstream_endpoint_indomain"
TARGET_DOWNSTREAM_OOD = "downstream_endpoint_ood"


# --------------------------------------------------------------------------- #
# Checkpoint / model helpers                                                   #
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    import json
    return json.loads(Path(path).read_text())


def select_best_epoch(run_dir: Path) -> tuple[int, float]:
    """Best pretrain epoch by validation loss in ``train.json``."""
    path = Path(run_dir) / "train.json"
    hist = _read_json(path)
    if not isinstance(hist, list) or not hist:
        raise AssertionError(f"[HARD] {path} is empty or not a list")
    best = min(hist, key=lambda e: float(e["val_loss"]))
    epoch = int(best["epoch"])
    ckpt = Path(run_dir) / f"epoch_{epoch:03d}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"[HARD] missing checkpoint {ckpt}")
    return epoch, float(best["val_loss"])


def checkpoint_file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_model_(model: DKMModel) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.buffers():
        if hasattr(p, "requires_grad_") and p.is_floating_point():
            p.requires_grad_(False)


def assert_no_grad_params(model: DKMModel) -> None:
    bad = [n for n, p in model.named_parameters() if p.requires_grad]
    if bad:
        raise AssertionError(
            f"[INV-PROBE-FROZEN] parameters still require grad during extraction: {bad[:8]}")


def snapshot_state(model: DKMModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assert_state_unchanged(before: dict[str, torch.Tensor], model: DKMModel) -> None:
    after = model.state_dict()
    for k, v0 in before.items():
        v1 = after[k].detach().cpu()
        if not torch.equal(v0, v1):
            raise AssertionError(
                f"[INV-PROBE-FROZEN] parameter {k} changed during the probe run")


def resolve_checkpoint(arm: str, ckpt_map: dict[str, Path] | None,
                       run_dirs: dict[str, Path] | None) -> tuple[Path, int, float]:
    if ckpt_map and arm in ckpt_map:
        path = Path(ckpt_map[arm])
        if not path.is_file():
            raise FileNotFoundError(f"[HARD] missing checkpoint for {arm}: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", -1))
        val_loss = float("nan")
        if run_dirs and arm in run_dirs and (run_dirs[arm] / "train.json").is_file():
            ep, vl = select_best_epoch(run_dirs[arm])
            if epoch < 0:
                epoch = ep
            val_loss = vl
            # Prefer the train.json-selected file when the map points at a run dir's best.
            preferred = run_dirs[arm] / f"epoch_{ep:03d}.pt"
            if preferred.is_file() and path.resolve() != preferred.resolve():
                # Map may point at best.pt / a copy; keep the mapped path but record the
                # train.json epoch when the loaded checkpoint's epoch matches.
                if int(ckpt.get("epoch", -1)) == ep:
                    val_loss = vl
        return path, epoch, val_loss
    if not run_dirs or arm not in run_dirs:
        raise FileNotFoundError(f"[HARD] no checkpoint for arm={arm}")
    ep, vl = select_best_epoch(run_dirs[arm])
    return run_dirs[arm] / f"epoch_{ep:03d}.pt", ep, vl


def load_pretrained_model(shared: dict, arm: str, ckpt_path: Path,
                          *, embedding_path: Path | None = None,
                          device: torch.device) -> tuple[DKMModel, dict]:
    """Load a frozen pretrained encoder. Optional embedding substitution for PIC (D3)."""
    if embedding_path is not None:
        emb = DKMModel._load_embedding_table(embedding_path, None)
        num_codes = int(emb.shape[0]) - 2
        model = DKMModel(
            num_codes=num_codes, embedding_table=emb, arm=arm, seed=shared["seed"],
            d_model=shared["d_model"], n_layers=shared["n_layers"], n_heads=shared["n_heads"],
            use_residual=shared["use_residual"], use_layernorm=shared["use_layernorm"],
            use_ffn=shared["use_ffn"], ffn_mult=shared["ffn_mult"], s=shared["s"],
            tau_max=shared["tau_max"], age_M=shared["age_M"], age_p_min=shared["age_p_min"],
            age_p_max=shared["age_p_max"], age_hidden=shared["age_hidden"],
            gen_final_bias=shared["gen_final_bias"],
            center_delta_alpha=shared["center_delta_alpha"], demo_dim=shared["demo_dim"],
            demo_channels=shared["demo_channels"], race_encoding=shared["race_encoding"],
            demo_hidden=shared["demo_hidden"], age_mean=shared["age_mean"],
            age_sd=shared["age_sd"], task="pretrain",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = dict(ckpt["model_state_dict"])
        if ("embedding_table" in state
                and tuple(state["embedding_table"].shape) != tuple(model.embedding_table.shape)):
            state["embedding_table"] = model.embedding_table.detach().clone()
        load_backbone(model, state, arm)
    else:
        model = build_model(shared, arm)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("arm") not in (None, arm):
            raise AssertionError(
                f"[HARD] {ckpt_path} holds arm={ckpt.get('arm')!r}, expected {arm!r}")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    tau = float(model.tau_max)
    freeze_model_(model)
    model.to(device)
    meta = {
        "path": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "tau_max": tau,
        "seed": int(ckpt.get("seed", shared["seed"])),
        "checkpoint_hash": checkpoint_file_hash(ckpt_path),
    }
    return model, meta


def random_init_model(shared: dict, arm: str, *, device: torch.device,
                      embedding_path: Path | None = None) -> DKMModel:
    """Architecture-matched floor: same config, per-parameter (seed, name) init, no ckpt."""
    if embedding_path is not None:
        emb = DKMModel._load_embedding_table(embedding_path, None)
        num_codes = int(emb.shape[0]) - 2
        model = DKMModel(
            num_codes=num_codes, embedding_table=emb, arm=arm, seed=shared["seed"],
            d_model=shared["d_model"], n_layers=shared["n_layers"], n_heads=shared["n_heads"],
            use_residual=shared["use_residual"], use_layernorm=shared["use_layernorm"],
            use_ffn=shared["use_ffn"], ffn_mult=shared["ffn_mult"], s=shared["s"],
            tau_max=shared["tau_max"], age_M=shared["age_M"], age_p_min=shared["age_p_min"],
            age_p_max=shared["age_p_max"], age_hidden=shared["age_hidden"],
            gen_final_bias=shared["gen_final_bias"],
            center_delta_alpha=shared["center_delta_alpha"], demo_dim=shared["demo_dim"],
            demo_channels=shared["demo_channels"], race_encoding=shared["race_encoding"],
            demo_hidden=shared["demo_hidden"], age_mean=shared["age_mean"],
            age_sd=shared["age_sd"], task="pretrain",
        )
    else:
        model = build_model(shared, arm)
    freeze_model_(model)
    model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Patient-level indices                                                        #
# --------------------------------------------------------------------------- #
def pretrain_patient_rows(ds: TensorizedPretrainDataset
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per patient: the last visit-window index, plus subject_id.

    Returns ``(dataset_indices, subject_ids, (shard_id, pos) keys)``.
    """
    # Last window per (shard, pos).
    last: dict[tuple[int, int], int] = {}
    for i, (shard_id, pos, _v) in enumerate(ds._index):
        last[(int(shard_id), int(pos))] = i
    # subject_id lives on the shard but is not part of the training item contract.
    subject_of: dict[tuple[int, int], int] = {}
    for (shard_id, pos) in last:
        s = ds._load_shard(shard_id)
        if "subject_id" not in s:
            npz = s["_npz"]
            if "subject_id" not in npz.files:
                raise AssertionError(
                    f"[HARD] shard {shard_id} has no subject_id; cannot enforce patient splits")
            s["subject_id"] = npz["subject_id"]
        subject_of[(shard_id, pos)] = int(s["subject_id"][pos])

    keys = sorted(last.keys())
    idxs = np.asarray([last[k] for k in keys], dtype=np.int64)
    subjects = np.asarray([subject_of[k] for k in keys], dtype=np.int64)
    key_arr = np.asarray([[k[0], k[1]] for k in keys], dtype=np.int64)
    return idxs, subjects, key_arr


def finetune_subject_ids(ds: TensorizedFinetuneDataset) -> np.ndarray:
    """Subject ids for every row, read from shard arrays (no per-row __getitem__)."""
    out = np.empty(len(ds), dtype=np.int64)
    for i, (shard_id, pos) in enumerate(ds._index):
        s = ds._load_shard(shard_id)
        out[i] = int(s["subject_id"][pos])
    return out


def assert_disjoint_subjects(splits: dict[str, np.ndarray]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = np.intersect1d(splits[a], splits[b])
            if inter.size:
                raise AssertionError(
                    f"[HARD] patients appear in both probe splits {a} and {b}: "
                    f"n_overlap={inter.size} e.g. {inter[:5].tolist()}")


def subsample_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=int(k), replace=False).astype(np.int64))


# --------------------------------------------------------------------------- #
# Targets                                                                      #
# --------------------------------------------------------------------------- #
def _valid_timestamps(ts: torch.Tensor, mask: torch.Tensor) -> list[np.ndarray]:
    out = []
    for b in range(ts.shape[0]):
        out.append(ts[b, mask[b]].detach().cpu().numpy().astype(np.float64))
    return out


def last_event_recency_days(batch: dict) -> np.ndarray:
    """Gap between the final two valid timestamps; NaN if the sequence has <2 events."""
    rows = _valid_timestamps(batch["timestamps_days"], batch["attention_mask"])
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for i, t in enumerate(rows):
        if t.size >= 2:
            out[i] = float(t[-1] - t[-2])
    return out


def record_span_days(batch: dict) -> np.ndarray:
    rows = _valid_timestamps(batch["timestamps_days"], batch["attention_mask"])
    out = np.zeros(len(rows), dtype=np.float64)
    for i, t in enumerate(rows):
        if t.size:
            out[i] = float(t[-1] - t[0])
    return out


def age_last_years(batch: dict) -> np.ndarray:
    lengths = batch["lengths"]
    rows = torch.arange(lengths.shape[0])
    return batch["age_years"][rows, lengths - 1].detach().cpu().numpy().astype(np.float64)


def quartile_edges(values: np.ndarray) -> list[float]:
    """Inclusive right edges for 4 quartile buckets from finite values only."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        raise AssertionError("[HARD] cannot set bucket edges on an empty probe-train set")
    qs = np.quantile(v, [0.25, 0.5, 0.75]).astype(np.float64)
    # Make edges strictly increasing for digitize; collapse ties with a tiny nudge.
    edges = qs.tolist()
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def bucketize(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """-> {0,1,2,3}; NaN inputs stay as -1."""
    out = np.full(values.shape, -1, dtype=np.int64)
    ok = np.isfinite(values)
    out[ok] = np.digitize(values[ok], bins=np.asarray(edges, dtype=np.float64), right=True)
    return out


def age_band_labels(ages: np.ndarray) -> np.ndarray:
    """Developmental / adult age band via ``diagnostics.AGE_BANDS`` (do not redefine)."""
    return D.band_index(ages, bands=D.AGE_BANDS).astype(np.int64)


# --------------------------------------------------------------------------- #
# Extraction + cache                                                           #
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path, arm: str, ckpt_hash: str, corpus: str, split: str) -> Path:
    return Path(cache_dir) / f"{arm}_{ckpt_hash}_{corpus}_{split}.pt"


def extract_split(
    model: DKMModel,
    loader: DataLoader,
    device: torch.device,
    *,
    subject_ids: np.ndarray | None = None,
    collect_pretrain_targets: bool = False,
    collect_labels: bool = False,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Run ``model.extract_representations`` over a loader. Returns CPU tensors / arrays."""
    assert_no_grad_params(model)
    h_pool, h_head = [], []
    recency, span, age, labels = [], [], [], []
    n = 0
    t0 = time.time()
    amp = bool(use_amp and device.type == "cuda")
    # Keep the GPU fed: one freeze check at entry/exit, not once per batch.
    with torch.no_grad():
        for batch in loader:
            batch_t = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                       for k, v in batch.items()}
            if amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.extract_representations(batch_t)
            else:
                out = model.extract_representations(batch_t)
            h_pool.append(out["h_pool"].float().detach().cpu())
            h_head.append(out["h_head"].float().detach().cpu())
            if collect_pretrain_targets:
                recency.append(last_event_recency_days(batch))
                span.append(record_span_days(batch))
                age.append(age_last_years(batch))
            if collect_labels:
                labels.append(batch["labels"].detach().cpu().numpy().astype(np.float64))
            n += int(out["h_pool"].shape[0])
    assert_no_grad_params(model)
    wall = time.time() - t0
    result: dict[str, Any] = {
        "h_pool": torch.cat(h_pool, dim=0) if h_pool else torch.zeros(0, model.d_model),
        "h_head": torch.cat(h_head, dim=0) if h_head else torch.zeros(0, 1),
        "n": n,
        "wall_clock_s": wall,
    }
    if subject_ids is not None:
        if subject_ids.shape[0] != n:
            raise AssertionError(
                f"[HARD] subject_ids length {subject_ids.shape[0]} != extracted n={n}")
        result["subject_id"] = subject_ids.astype(np.int64)
    if collect_pretrain_targets:
        result["last_event_recency_days"] = (np.concatenate(recency) if recency
                                             else np.zeros(0))
        result["record_span_days"] = np.concatenate(span) if span else np.zeros(0)
        result["age_last_years"] = np.concatenate(age) if age else np.zeros(0)
    if collect_labels:
        result["labels"] = np.concatenate(labels) if labels else np.zeros(0)
    return result


# Back-compat alias used by tests.
extract_representations = extract_split


def load_or_extract(
    cache_path: Path,
    expected_hash: str,
    extract_fn,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    if cache_path.is_file():
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        if obj.get("checkpoint_hash") != expected_hash:
            cache_path.unlink()
        else:
            return obj
    if label:
        D.print_block("extracting representations", [label, f"cache -> {cache_path.name}"])
    obj = extract_fn()
    obj["checkpoint_hash"] = expected_hash
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, cache_path)
    if label:
        D.print_block("extraction done", [
            f"{label}: n={obj.get('n')} wall={obj.get('wall_clock_s'):.1f}s",
        ])
    return obj


# --------------------------------------------------------------------------- #
# Probe fitting                                                                #
# --------------------------------------------------------------------------- #
def _binary_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {"auprc": float("nan"), "auroc": float("nan"),
           "prevalence": float(y.mean()) if y.size else float("nan")}
    if y.size and 0 < y.sum() < y.size:
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc"] = float(roc_auc_score(y, p))
    return out


def _macro_ovr_auprc(y: np.ndarray, P: np.ndarray, classes: np.ndarray) -> float:
    aps = []
    for j, c in enumerate(classes):
        yb = (y == c).astype(np.float64)
        if yb.sum() == 0 or yb.sum() == yb.size:
            continue
        aps.append(average_precision_score(yb, P[:, j]))
    return float(np.mean(aps)) if aps else float("nan")


def _fit_one_C(C, X_train_s, y_train, X_val_s, y_val, binary, seed):
    """Fit one C; returns (C, score, clf, warnings). Picklable for joblib."""
    warns: list[str] = []
    clf = LogisticRegression(
        penalty="l2", C=float(C), class_weight=None, max_iter=MAX_ITER,
        solver="lbfgs", random_state=seed, multi_class="auto",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train_s, y_train)
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                warns.append(str(w.message))
    if binary:
        if len(np.unique(y_val)) < 2:
            score = float("nan")
        else:
            score = float(average_precision_score(y_val, clf.predict_proba(X_val_s)[:, 1]))
    else:
        score = _macro_ovr_auprc(y_val, clf.predict_proba(X_val_s), clf.classes_)
    return float(C), score, clf, warns


def _bootstrap_ci_parallel(stat, patient_ids: np.ndarray, *, n_boot: int, seed: int,
                           n_jobs: int = -1) -> dict:
    """Same contract as ``diagnostics.bootstrap_ci``, parallel over resamples."""
    from joblib import Parallel, delayed

    blocks = D._patient_blocks(np.asarray(patient_ids))
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, n_blocks, size=n_blocks) for _ in range(int(n_boot))]

    def _one(pick):
        rows = np.concatenate([blocks[i] for i in pick]) if n_blocks else np.zeros(0, int)
        v = stat(rows)
        return float(v) if v is not None and np.isfinite(v) else None

    raw = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(p) for p in picks)
    vals = [v for v in raw if v is not None]
    arr = np.asarray(vals, dtype=np.float64)
    alpha = 0.05
    return {
        "n_boot": int(n_boot),
        "n_patients": int(n_blocks),
        "n_resamples_undefined": int(n_boot) - len(vals),
        "ci_level": 1.0 - alpha,
        "lo": float(np.percentile(arr, 100 * alpha / 2)) if arr.size else float("nan"),
        "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))) if arr.size else float("nan"),
        "boot_mean": float(arr.mean()) if arr.size else float("nan"),
        "boot_sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "resampling_unit": "patient (subject_id), sampled with replacement",
    }


def fit_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *,
    subject_test: np.ndarray,
    seed: int,
    n_boot: int = N_BOOTSTRAP_PROBE,
    binary: bool,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """StandardScaler on train only; C selected on val by the headline metric."""
    from joblib import Parallel, delayed

    def _mask(y):
        return np.isfinite(y) & (y >= 0)

    m_tr, m_va, m_te = _mask(y_train), _mask(y_val), _mask(y_test)
    X_train, y_train = X_train[m_tr], y_train[m_tr].astype(np.int64 if not binary else np.float64)
    X_val, y_val = X_val[m_va], y_val[m_va].astype(np.int64 if not binary else np.float64)
    X_test, y_test = X_test[m_te], y_test[m_te].astype(np.int64 if not binary else np.float64)
    subject_test = subject_test[m_te]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    conv_warnings: list[str] = []
    best_C, best_score, best_clf = None, -np.inf, None

    if len(np.unique(y_train)) >= 2:
        fitted = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_one_C)(C, X_train_s, y_train, X_val_s, y_val, binary, seed)
            for C in C_GRID
        )
        for C, score, clf, warns in fitted:
            conv_warnings.extend(warns)
            if np.isfinite(score) and score > best_score:
                best_score, best_C, best_clf = score, float(C), clf

    if best_clf is None:
        if len(np.unique(y_train)) < 2:
            metrics = ({"auprc": float("nan"), "auroc": float("nan")} if binary else
                       {"macro_auprc": float("nan"), "balanced_accuracy": float("nan")})
            return {
                "selected_C": float(C_GRID[len(C_GRID) // 2]),
                "val_selection_score": None,
                "metrics": metrics,
                "prevalence": (float(y_test.mean()) if binary and y_test.size else {}),
                "n_train": int(y_train.size),
                "n_val": int(y_val.size),
                "n_test": int(y_test.size),
                "convergence_warnings": conv_warnings + ["skipped: <2 classes in train"],
                "binary": binary,
            }
        best_C = float(C_GRID[len(C_GRID) // 2])
        best_clf = LogisticRegression(
            penalty="l2", C=best_C, class_weight=None, max_iter=MAX_ITER,
            solver="lbfgs", random_state=seed, multi_class="auto",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            best_clf.fit(X_train_s, y_train)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    conv_warnings.append(str(w.message))
        best_score = float("nan")

    metrics: dict[str, Any]
    if binary:
        p = best_clf.predict_proba(X_test_s)[:, 1]
        metrics = _binary_scores(y_test, p)
        y_ref, p_ref = y_test, p

        def stat_auprc(rows):
            yy, pp = y_ref[rows], p_ref[rows]
            if yy.size == 0 or yy.sum() == 0 or yy.sum() == yy.size:
                return float("nan")
            return float(average_precision_score(yy, pp))

        def stat_auroc(rows):
            yy, pp = y_ref[rows], p_ref[rows]
            if yy.size == 0 or yy.sum() == 0 or yy.sum() == yy.size:
                return float("nan")
            return float(roc_auc_score(yy, pp))

        metrics["auprc_ci"] = _bootstrap_ci_parallel(stat_auprc, subject_test,
                                                     n_boot=n_boot, seed=seed, n_jobs=n_jobs)
        metrics["auroc_ci"] = _bootstrap_ci_parallel(stat_auroc, subject_test,
                                                     n_boot=n_boot, seed=seed + 1, n_jobs=n_jobs)
    else:
        classes = best_clf.classes_
        P = best_clf.predict_proba(X_test_s)
        pred = best_clf.predict(X_test_s)
        metrics = {
            "macro_auprc": _macro_ovr_auprc(y_test, P, classes),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred))
            if y_test.size else float("nan"),
            "class_counts": {int(c): int((y_test == c).sum()) for c in classes},
        }
        y_ref, P_ref, classes_ref = y_test, P, classes

        def stat_macro(rows):
            return _macro_ovr_auprc(y_ref[rows], P_ref[rows], classes_ref)

        metrics["macro_auprc_ci"] = _bootstrap_ci_parallel(
            stat_macro, subject_test, n_boot=n_boot, seed=seed, n_jobs=n_jobs)

    prevalence = (float(y_test.mean()) if binary and y_test.size
                  else {int(c): float((y_test == c).mean()) for c in np.unique(y_test)})
    return {
        "selected_C": best_C,
        "val_selection_score": float(best_score) if np.isfinite(best_score) else None,
        "metrics": metrics,
        "prevalence": prevalence,
        "n_train": int(y_train.size),
        "n_val": int(y_val.size),
        "n_test": int(y_test.size),
        "convergence_warnings": conv_warnings,
        "binary": binary,
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _make_pretrain_loader(ds: TensorizedPretrainDataset, indices: np.ndarray,
                          batch_size: int, num_workers: int, race_encoding: str,
                          *, pin_memory: bool = False) -> DataLoader:
    subset = Subset(ds, indices.tolist())
    kw: dict[str, Any] = dict(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, collate_fn=make_collate(race_encoding),
        pin_memory=pin_memory, worker_init_fn=dataloader_worker_init,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 4
    return DataLoader(subset, **kw)


def _make_finetune_loader(ds: TensorizedFinetuneDataset, indices: np.ndarray | None,
                          batch_size: int, num_workers: int, race_encoding: str,
                          *, pin_memory: bool = False) -> DataLoader:
    data = ds if indices is None else Subset(ds, indices.tolist())
    kw: dict[str, Any] = dict(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, collate_fn=make_finetune_collate(race_encoding),
        pin_memory=pin_memory, worker_init_fn=dataloader_worker_init,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 4
    return DataLoader(data, **kw)


def _X(store: dict, repr_name: str) -> np.ndarray:
    return store[repr_name].numpy().astype(np.float64)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "repr_cache"

    # Resolve per-arm run dirs / checkpoints.
    ckpt_map: dict[str, Path] = {}
    if args.ckpt_map:
        for entry in args.ckpt_map:
            arm, _, path = entry.partition("=")
            if arm not in ARMS or not path:
                raise ValueError(f"ckpt_map entry must be arm=path, got {entry!r}")
            ckpt_map[arm] = Path(path)
    run_dirs: dict[str, Path] = {}
    if args.runs:
        for entry in args.runs:
            arm, _, path = entry.partition("=")
            if arm not in ARMS or not path:
                raise ValueError(f"runs entry must be arm=path, got {entry!r}")
            run_dirs[arm] = Path(path)

    arms = list(args.arms)
    # Config / shared kwargs from the first arm that has a run dir or checkpoint config.
    shared = None
    configs = {}
    for arm in arms:
        cfg_path = None
        if arm in run_dirs:
            cfg_path = run_dirs[arm] / "config.json"
        elif arm in ckpt_map:
            ck = torch.load(ckpt_map[arm], map_location="cpu", weights_only=False)
            if "config" in ck:
                configs[arm] = ck["config"]
                continue
        if cfg_path and cfg_path.is_file():
            configs[arm] = _read_json(cfg_path)
    if not configs:
        raise AssertionError("[HARD] need at least one config.json or checkpoint config")
    ref_arm = next(a for a in arms if a in configs)
    shared = model_kwargs_from_config(configs[ref_arm])

    # Select checkpoints and enforce a single tau_max.
    selected: dict[str, dict] = {}
    tau_values = {}
    for arm in arms:
        path, epoch, val_loss = resolve_checkpoint(arm, ckpt_map or None, run_dirs or None)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        tau = float(ck.get("tau_max", ck.get("config", {}).get("model", {}).get("tau_max")))
        tau_values[arm] = tau
        selected[arm] = {
            "path": path, "epoch": epoch, "val_loss": val_loss,
            "checkpoint_hash": checkpoint_file_hash(path), "tau_max": tau,
        }
    uniq = {round(v, 12) for v in tau_values.values()}
    if len(uniq) != 1:
        raise AssertionError(f"[HARD] tau_max disagrees across checkpoints: {tau_values}")

    D.print_probe_header({
        "run_name": args.run_name,
        "seed": args.seed,
        "arms": arms,
        "representations": list(REPR_NAMES),
        "targets": [
            TARGET_LAST_EVENT_RECENCY, TARGET_RECORD_SPAN, TARGET_AGE_BAND,
            TARGET_DOWNSTREAM_ID, TARGET_DOWNSTREAM_OOD,
        ],
        "train_subsample_n": args.train_subsample,
        "train_subsample_seed": args.seed,
        "C_grid": C_GRID.tolist(),
        "n_bootstrap": args.n_bootstrap,
    })

    # ---- Pretrain corpus (sequence / age targets) -------------------------- #
    tensorized = REPO_ROOT / shared["tensorized_dir"]
    vocab = REPO_ROOT / shared["vocab_path"]
    race = shared["race_encoding"]
    max_len = shared["max_seq_len"]

    splits_ds = {
        s: TensorizedPretrainDataset(tensorized / s, vocab, max_seq_len=max_len)
        for s in ("train", "val", "test")
    }
    split_rows = {s: pretrain_patient_rows(ds) for s, ds in splits_ds.items()}
    subjects = {s: split_rows[s][1] for s in split_rows}
    assert_disjoint_subjects(subjects)

    train_idxs_all, train_subj_all, _ = split_rows["train"]
    pick = subsample_indices(len(train_idxs_all), args.train_subsample, args.seed)
    train_idxs = train_idxs_all[pick]
    train_subj = train_subj_all[pick]
    val_idxs, val_subj, _ = split_rows["val"]
    test_idxs, test_subj, _ = split_rows["test"]

    pin = device.type == "cuda"
    loaders = {
        "train": _make_pretrain_loader(splits_ds["train"], train_idxs, args.batch_size,
                                       args.num_workers, race, pin_memory=pin),
        "val": _make_pretrain_loader(splits_ds["val"], val_idxs, args.batch_size,
                                     args.num_workers, race, pin_memory=pin),
        "test": _make_pretrain_loader(splits_ds["test"], test_idxs, args.batch_size,
                                      args.num_workers, race, pin_memory=pin),
    }
    subj_by_split = {"train": train_subj, "val": val_subj, "test": test_subj}

    # ---- Downstream corpora ------------------------------------------------ #
    id_root = Path(args.indomain_tensorized)
    ood_root = Path(args.ood_tensorized)
    id_splits = {s: TensorizedFinetuneDataset(id_root / s, max_seq_len=max_len)
                 for s in ("train", "val", "test")}
    ood_splits = {s: TensorizedFinetuneDataset(ood_root / s, max_seq_len=max_len)
                  for s in ("train", "val", "test")}
    id_subj = {s: finetune_subject_ids(ds) for s, ds in id_splits.items()}
    ood_subj = {s: finetune_subject_ids(ds) for s, ds in ood_splits.items()}
    assert_disjoint_subjects(id_subj)
    assert_disjoint_subjects(ood_subj)

    id_train_pick = subsample_indices(len(id_splits["train"]),
                                      min(args.train_subsample, len(id_splits["train"])),
                                      args.seed)
    id_loaders = {
        "train": _make_finetune_loader(id_splits["train"], id_train_pick, args.batch_size,
                                       args.num_workers, race, pin_memory=pin),
        "val": _make_finetune_loader(id_splits["val"], None, args.batch_size,
                                     args.num_workers, race, pin_memory=pin),
        "test": _make_finetune_loader(id_splits["test"], None, args.batch_size,
                                      args.num_workers, race, pin_memory=pin),
    }
    id_subj_use = {
        "train": id_subj["train"][id_train_pick],
        "val": id_subj["val"],
        "test": id_subj["test"],
    }
    ood_train_pick = subsample_indices(len(ood_splits["train"]),
                                       min(args.train_subsample, len(ood_splits["train"])),
                                       args.seed)
    ood_loaders = {
        "train": _make_finetune_loader(ood_splits["train"], ood_train_pick, args.batch_size,
                                       args.num_workers, race, pin_memory=pin),
        "val": _make_finetune_loader(ood_splits["val"], None, args.batch_size,
                                     args.num_workers, race, pin_memory=pin),
        "test": _make_finetune_loader(ood_splits["test"], None, args.batch_size,
                                      args.num_workers, race, pin_memory=pin),
    }
    ood_subj_use = {
        "train": ood_subj["train"][ood_train_pick],
        "val": ood_subj["val"],
        "test": ood_subj["test"],
    }

    results: list[dict] = []
    extraction_costs: dict[str, Any] = {}
    floor_close: list[dict] = []

    for arm in arms:
        sel = selected[arm]
        model, meta = load_pretrained_model(
            shared, arm, sel["path"], device=device)
        # Prefer train.json epoch when available.
        pretrain_epoch = int(sel["epoch"]) if sel["epoch"] >= 0 else int(meta["epoch"])
        ckpt_hash = sel["checkpoint_hash"]
        before = snapshot_state(model)

        # Pretrain representations (trained).
        pre_stores = {}
        for split, loader in loaders.items():
            cpath = _cache_path(cache_dir, arm, ckpt_hash, "mimic_pretrain", split)

            def _ex(loader=loader, split=split, _extract=extract_split):
                return _extract(
                    model, loader, device, subject_ids=subj_by_split[split],
                    collect_pretrain_targets=True)

            store = load_or_extract(cpath, ckpt_hash, _ex,
                                       label=f"{arm}/mimic_pretrain/{split}")
            pre_stores[split] = store
            extraction_costs[f"{arm}/mimic_pretrain/{split}"] = {
                "wall_clock_s": store.get("wall_clock_s"),
                "n": store.get("n"),
                "cached": cpath.is_file(),
            }

        # Bucket edges from probe-train only.
        edges_recency = quartile_edges(pre_stores["train"]["last_event_recency_days"])
        edges_span = quartile_edges(pre_stores["train"]["record_span_days"])

        def pre_labels(store, target):
            if target == TARGET_LAST_EVENT_RECENCY:
                return bucketize(store["last_event_recency_days"], edges_recency)
            if target == TARGET_RECORD_SPAN:
                return bucketize(store["record_span_days"], edges_span)
            if target == TARGET_AGE_BAND:
                return age_band_labels(store["age_last_years"])
            raise KeyError(target)

        # Random-init floor for pretrain corpus.
        floor = random_init_model(shared, arm, device=device)
        before_floor = snapshot_state(floor)
        floor_stores = {}
        floor_hash = f"random_init_{arm}_s{shared['seed']}"
        for split, loader in loaders.items():
            cpath = _cache_path(cache_dir, arm, floor_hash, "mimic_pretrain", split)

            def _ex(loader=loader, split=split, floor=floor, _extract=extract_split):
                return _extract(
                    floor, loader, device, subject_ids=subj_by_split[split],
                    collect_pretrain_targets=True)

            floor_stores[split] = load_or_extract(
                cpath, floor_hash, _ex, label=f"{arm}/floor/mimic_pretrain/{split}")

        pretrain_by_repr: dict[str, list] = {}
        for repr_name in REPR_NAMES:
            # Non-additive arms: h_head == h_pool, so reuse the h_pool probe fits.
            if arm != "additive" and repr_name == "h_head" and "h_pool" in pretrain_by_repr:
                for rec0 in pretrain_by_repr["h_pool"]:
                    rec = dict(rec0)
                    rec["representation"] = "h_head"
                    results.append(rec)
                    D.print_probe_result(rec)
                continue
            pretrain_by_repr[repr_name] = []
            for target in (TARGET_LAST_EVENT_RECENCY, TARGET_RECORD_SPAN, TARGET_AGE_BAND):
                y_tr = pre_labels(pre_stores["train"], target)
                y_va = pre_labels(pre_stores["val"], target)
                y_te = pre_labels(pre_stores["test"], target)
                fitted = fit_linear_probe(
                    _X(pre_stores["train"], repr_name), y_tr,
                    _X(pre_stores["val"], repr_name), y_va,
                    _X(pre_stores["test"], repr_name), y_te,
                    subject_test=pre_stores["test"]["subject_id"],
                    seed=args.seed, n_boot=args.n_bootstrap, binary=False,
                )
                floor_fit = fit_linear_probe(
                    _X(floor_stores["train"], repr_name),
                    pre_labels(floor_stores["train"], target),
                    _X(floor_stores["val"], repr_name),
                    pre_labels(floor_stores["val"], target),
                    _X(floor_stores["test"], repr_name),
                    pre_labels(floor_stores["test"], target),
                    subject_test=floor_stores["test"]["subject_id"],
                    seed=args.seed, n_boot=args.n_bootstrap, binary=False,
                )
                rec = {
                    "arm": arm, "representation": repr_name, "target": target,
                    "source": "trained",
                    "pretrain_epoch": pretrain_epoch,
                    "checkpoint_hash": ckpt_hash,
                    "selected_C": fitted["selected_C"],
                    "metrics": fitted["metrics"],
                    "floor_metrics": floor_fit["metrics"],
                    "floor_selected_C": floor_fit["selected_C"],
                    "prevalence": fitted["prevalence"],
                    "n_train": fitted["n_train"], "n_val": fitted["n_val"],
                    "n_test": fitted["n_test"],
                    "bucket_edges": {
                        TARGET_LAST_EVENT_RECENCY: edges_recency,
                        TARGET_RECORD_SPAN: edges_span,
                        TARGET_AGE_BAND: {
                            "bands": [[n, lo, hi] for n, lo, hi in D.AGE_BANDS],
                            "note": "class count follows diagnostics.AGE_BANDS; not redefined",
                        },
                    }.get(target),
                    "convergence_warnings": fitted["convergence_warnings"],
                }
                results.append(rec)
                pretrain_by_repr[repr_name].append(rec)
                D.print_probe_result(rec)
                _maybe_floor_close(floor_close, rec)

        # Downstream in-domain (MIMIC labels).
        id_stores = {}
        for split, loader in id_loaders.items():
            cpath = _cache_path(cache_dir, arm, ckpt_hash,
                                f"mimic_ft_{args.indomain_task}", split)

            def _ex(loader=loader, split=split, _extract=extract_split):
                return _extract(
                    model, loader, device, subject_ids=id_subj_use[split],
                    collect_labels=True)

            id_stores[split] = load_or_extract(
                cpath, ckpt_hash, _ex, label=f"{arm}/mimic_ft/{split}")

        id_floor_stores = {}
        for split, loader in id_loaders.items():
            cpath = _cache_path(cache_dir, arm, floor_hash,
                                f"mimic_ft_{args.indomain_task}", split)

            def _ex(loader=loader, split=split, floor=floor, _extract=extract_split):
                return _extract(
                    floor, loader, device, subject_ids=id_subj_use[split],
                    collect_labels=True)

            id_floor_stores[split] = load_or_extract(
                cpath, floor_hash, _ex, label=f"{arm}/floor/mimic_ft/{split}")

        # Downstream OOD (PIC labels) — embedding substitution.
        ood_emb = Path(args.ood_embedding_path)
        model_ood, _ = load_pretrained_model(
            shared, arm, sel["path"], embedding_path=ood_emb, device=device)
        before_ood = snapshot_state(model_ood)
        floor_ood = random_init_model(shared, arm, device=device, embedding_path=ood_emb)

        ood_stores = {}
        for split, loader in ood_loaders.items():
            cpath = _cache_path(cache_dir, arm, ckpt_hash,
                                f"pic_ft_{args.ood_task}", split)

            def _ex(loader=loader, split=split, model_ood=model_ood, _extract=extract_split):
                return _extract(
                    model_ood, loader, device, subject_ids=ood_subj_use[split],
                    collect_labels=True)

            ood_stores[split] = load_or_extract(
                cpath, ckpt_hash, _ex, label=f"{arm}/pic_ft/{split}")

        ood_floor_hash = f"random_init_{arm}_s{shared['seed']}_pic"
        ood_floor_stores = {}
        for split, loader in ood_loaders.items():
            cpath = _cache_path(cache_dir, arm, ood_floor_hash,
                                f"pic_ft_{args.ood_task}", split)

            def _ex(loader=loader, split=split, floor_ood=floor_ood, _extract=extract_split):
                return _extract(
                    floor_ood, loader, device, subject_ids=ood_subj_use[split],
                    collect_labels=True)

            ood_floor_stores[split] = load_or_extract(
                cpath, ood_floor_hash, _ex, label=f"{arm}/floor/pic_ft/{split}")

        down_by_repr: dict[str, list] = {}
        for repr_name in REPR_NAMES:
            if arm != "additive" and repr_name == "h_head" and "h_pool" in down_by_repr:
                for rec0 in down_by_repr["h_pool"]:
                    rec = dict(rec0)
                    rec["representation"] = "h_head"
                    results.append(rec)
                    D.print_probe_result(rec)
                continue
            down_by_repr[repr_name] = []
            for target, stores, floor_s, task_name in (
                (TARGET_DOWNSTREAM_ID, id_stores, id_floor_stores, args.indomain_task),
                (TARGET_DOWNSTREAM_OOD, ood_stores, ood_floor_stores, args.ood_task),
            ):
                fitted = fit_linear_probe(
                    _X(stores["train"], repr_name), stores["train"]["labels"],
                    _X(stores["val"], repr_name), stores["val"]["labels"],
                    _X(stores["test"], repr_name), stores["test"]["labels"],
                    subject_test=stores["test"]["subject_id"],
                    seed=args.seed, n_boot=args.n_bootstrap, binary=True,
                )
                floor_fit = fit_linear_probe(
                    _X(floor_s["train"], repr_name), floor_s["train"]["labels"],
                    _X(floor_s["val"], repr_name), floor_s["val"]["labels"],
                    _X(floor_s["test"], repr_name), floor_s["test"]["labels"],
                    subject_test=floor_s["test"]["subject_id"],
                    seed=args.seed, n_boot=args.n_bootstrap, binary=True,
                )
                rec = {
                    "arm": arm, "representation": repr_name, "target": target,
                    "task": task_name, "source": "trained",
                    "pretrain_epoch": pretrain_epoch,
                    "checkpoint_hash": ckpt_hash,
                    "selected_C": fitted["selected_C"],
                    "metrics": fitted["metrics"],
                    "floor_metrics": floor_fit["metrics"],
                    "floor_selected_C": floor_fit["selected_C"],
                    "prevalence": fitted["prevalence"],
                    "n_train": fitted["n_train"], "n_val": fitted["n_val"],
                    "n_test": fitted["n_test"],
                    "convergence_warnings": fitted["convergence_warnings"],
                }
                results.append(rec)
                down_by_repr[repr_name].append(rec)
                D.print_probe_result(rec)
                _maybe_floor_close(floor_close, rec)

        assert_state_unchanged(before, model)
        assert_state_unchanged(before_floor, floor)
        assert_state_unchanged(before_ood, model_ood)
        del model, floor, model_ood, floor_ood

    payload = {
        "run_name": args.run_name,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
        "C_grid": C_GRID.tolist(),
        "train_subsample_n": args.train_subsample,
        "train_subsample_seed": args.seed,
        "selected_checkpoints": {a: {**selected[a], "path": str(selected[a]["path"])}
                                 for a in selected},
        "tau_max": next(iter(tau_values.values())),
        "split_n_patients": {
            "pretrain": {s: int(subjects[s].size) for s in subjects},
            "indomain": {s: int(id_subj[s].size) for s in id_subj},
            "ood": {s: int(ood_subj[s].size) for s in ood_subj},
            "pretrain_train_subsample": int(train_subj.size),
        },
        "notes": {
            "h_pool_h_head_asymmetry": PROBE_ASYMMETRY_NOTE,
            "downstream_endpoint": DOWNSTREAM_ENDPOINT_NOTE,
        },
        "extraction_costs": extraction_costs,
        "results": results,
        "floor_close": floor_close,
    }
    D.write_json(run_dir / "probe.json", payload)

    # Headline numbers into paper_numbers.json under a probe key.
    headline = []
    for r in results:
        m = r["metrics"]
        headline.append({
            "arm": r["arm"], "representation": r["representation"], "target": r["target"],
            "auprc": m.get("auprc", m.get("macro_auprc")),
            "floor_auprc": (r.get("floor_metrics") or {}).get(
                "auprc", (r.get("floor_metrics") or {}).get("macro_auprc")),
            "selected_C": r["selected_C"],
            "pretrain_epoch": r["pretrain_epoch"],
        })
    paper_path = run_dir / "paper_numbers.json"
    paper = _read_json(paper_path) if paper_path.is_file() else {}
    paper["probe"] = {
        "headline": headline,
        "notes": payload["notes"],
        "tau_max": payload["tau_max"],
        "seed": args.seed,
    }
    D.write_json(paper_path, paper)

    # Also attach a compact probe block onto each arm's own paper_numbers when available.
    for arm in arms:
        if arm not in run_dirs:
            continue
        pn = run_dirs[arm] / "paper_numbers.json"
        if not pn.is_file():
            continue
        obj = _read_json(pn)
        obj["probe"] = {
            "run_name": args.run_name,
            "headline": [h for h in headline if h["arm"] == arm],
            "pretrain_epoch": selected[arm]["epoch"],
            "checkpoint_hash": selected[arm]["checkpoint_hash"],
        }
        D.write_json(pn, obj)

    D.print_probe_summary(results, floor_close=floor_close)
    return payload


def _maybe_floor_close(acc: list, rec: dict, tol: float = 0.02) -> None:
    m = rec["metrics"]
    f = rec.get("floor_metrics") or {}
    t = m.get("auprc", m.get("macro_auprc"))
    fl = f.get("auprc", f.get("macro_auprc"))
    if isinstance(t, float) and isinstance(fl, float) and np.isfinite(t) and np.isfinite(fl):
        if abs(t - fl) <= tol:
            acc.append({"arm": rec["arm"], "representation": rec["representation"],
                        "target": rec["target"], "trained": t, "floor": fl,
                        "gap": float(t - fl)})


# --------------------------------------------------------------------------- #
# CLI / smoke                                                                  #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", type=str, default="probe_s0")
    p.add_argument("--run_root", type=Path, default=Path("model_new/run"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--runs", nargs="*", default=None,
                   help="arm=run_dir entries (config.json + train.json + epoch_NNN.pt)")
    p.add_argument("--ckpt_map", nargs="*", default=None,
                   help="arm=checkpoint.pt entries (same shape as finetune.sh CKPT_MAP)")
    p.add_argument("--train_subsample", type=int, default=DEFAULT_TRAIN_SUBSAMPLE)
    p.add_argument("--batch_size", type=int, default=128,
                   help="extraction batch size; raise until GPU memory is the limit")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP_PROBE)
    p.add_argument("--indomain_tensorized", type=Path,
                   default=REPO_ROOT / "data/finetune/t2d_tensorized")
    p.add_argument("--indomain_task", type=str, default="t2d")
    p.add_argument("--ood_tensorized", type=Path,
                   default=REPO_ROOT / "data/tensorized/pic/pneumonia")
    p.add_argument("--ood_task", type=str, default="pneumonia")
    p.add_argument("--ood_embedding_path", type=Path,
                   default=REPO_ROOT / "data/processed/pic/bge_embeddings_pic.pt")
    p.add_argument("--smoke", action="store_true",
                   help="CPU-only synthetic smoke; does not touch real corpora")
    return p


def _smoke() -> dict:
    """Tiny synthetic end-to-end check; routes output through diagnostics."""
    set_seed(0)
    from model_new.tests.conftest import make_items
    from model_new.data import pretrain_collate

    table = torch.randn(34, 24)
    batch = pretrain_collate(make_items(np.random.default_rng(0), lengths=(6, 4, 5, 3)))
    # Pad demographics width already handled by collate.
    rows = []
    for arm in ARMS:
        m = DKMModel(num_codes=32, embedding_table=table, arm=arm, seed=0, d_model=16,
                     age_hidden=8, demo_hidden=8, tau_max=6.5)
        freeze_model_(m)
        assert_no_grad_params(m)
        before = snapshot_state(m)
        with torch.no_grad():
            out = m.extract_representations(batch)
        assert out["h_pool"].shape == (batch["code_indices"].shape[0], 16)
        if arm == "additive":
            assert out["h_head"].shape[-1] == 16 + m.s
        else:
            assert torch.equal(out["h_pool"], out["h_head"])
        # demographics must not affect h_pool
        b2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        b2["demographics"] = b2["demographics"] + 3.0
        with torch.no_grad():
            out2 = m.extract_representations(b2)
        assert torch.equal(out["h_pool"], out2["h_pool"]), arm
        assert_state_unchanged(before, m)
        rows.append(f"{arm}: h_pool={tuple(out['h_pool'].shape)} "
                    f"h_head={tuple(out['h_head'].shape)} demo-invariant=True")

        # Fit a tiny probe on synthetic features with both classes present.
        X = out["h_pool"].numpy()
        y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        fit = fit_linear_probe(
            X[:2], y[:2], X[2:3], y[2:3], X[3:], y[3:],
            subject_test=np.arange(1), seed=0, n_boot=20, binary=True,
        )
        rows.append(f"  probe C={fit['selected_C']} n_test={fit['n_test']}")

    D.print_block("probe.py smoke", rows)
    return {"ok": True, "arms": list(ARMS)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        _smoke()
        return 0
    if not args.runs and not args.ckpt_map:
        raise SystemExit("provide --runs arm=dir ... and/or --ckpt_map arm=path ...")
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
