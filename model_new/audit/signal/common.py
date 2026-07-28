"""Shared fixed-batch store, metrics, bootstrap, DuckDB, and model helpers.

Reuses ``audit.common``, ``eval_pretrain``, ``data.corpus_stats_cached``,
``data.tau_from_timestamps``, and ``diagnostics`` writers exclusively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from model_new import diagnostics as D
from model_new.audit import AUDIT_SEED
from model_new.audit.common import (
    ARMS,
    DEFAULT_RUN_ROOT,
    REPO_ROOT,
    age_last_of,
    checkpoint_path,
    discover_runs,
    load_checkpoint,
    patient_ids_from_dataset,
    select_best_epoch,
    to_device,
)
from model_new.audit.signal import (
    CACHE_VERSION,
    DATALOADER_WORKERS,
    DUCKDB_MEMORY,
    DUCKDB_THREADS,
    KS,
    MIN_GPU_BATCH_SIZE,
    N_BOOT,
    N_VAL_BATCHES,
    SIGNAL_SEED,
    SMOKE_N_BATCHES,
    SMOKE_N_BOOT,
    SMOKE_N_PERM,
    SMOKE_PATIENT_FRAC,
    SMOKE_TOP_CODES,
    T4_SIGMA_CONTENT,
    TOP_CODES,
    N_PERM,
)
from model_new.data import (
    TensorizedPretrainDataset,
    corpus_stats_cached,
    tau_from_timestamps,
)
from model_new.eval_pretrain import (
    BatchOrderHash,
    build_model,
    check_configs,
    make_val_loader,
    model_kwargs_from_config,
)
from model_new.train import set_seed

DEFAULT_OUT = REPO_ROOT / "model_new" / "audit" / "signal" / "out"
BATCH_STORE_NAME = "fixed_batches.pt"
BATCH_META_NAME = "fixed_batches_meta.json"


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Atomic JSON write via diagnostics (``.tmp`` then ``os.replace``)."""
    D.write_json(path, obj)


def open_duckdb():
    """In-memory DuckDB with the resource contract (threads + memory_limit)."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads={int(DUCKDB_THREADS)}")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    return con


def smoke_flags(smoke: bool) -> dict[str, Any]:
    if smoke:
        return {
            "smoke": True,
            "n_batches": SMOKE_N_BATCHES,
            "n_boot": SMOKE_N_BOOT,
            "n_perm": SMOKE_N_PERM,
            "patient_frac": SMOKE_PATIENT_FRAC,
            "top_codes": SMOKE_TOP_CODES,
            "num_workers": 0,
        }
    return {
        "smoke": False,
        "n_batches": N_VAL_BATCHES,
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "patient_frac": 1.0,
        "top_codes": TOP_CODES,
        "num_workers": DATALOADER_WORKERS,
    }


def resolve_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def require_cuda(device: torch.device, *, batch_size: int, label: str) -> None:
    """HARD: GPU tests must run on CUDA with B ≥ 128 (age audit CPU/32 must not leak)."""
    if device.type != "cuda":
        raise AssertionError(
            f"[HARD] {label}: device={device} (expected cuda). "
            f"Refusing CPU — runtime estimates assume GPU.")
    if not torch.cuda.is_available():
        raise AssertionError(f"[HARD] {label}: torch.cuda.is_available() is False")
    if int(batch_size) < int(MIN_GPU_BATCH_SIZE):
        raise AssertionError(
            f"[HARD] {label}: batch_size={batch_size} < {MIN_GPU_BATCH_SIZE}")


def load_shared(run_root: Path | None = None) -> dict:
    """Configs + shared kwargs; never recomputes τ_max or corpus stats."""
    run_root = Path(run_root or DEFAULT_RUN_ROOT)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    run_dirs = discover_runs(run_root)
    configs = {arm: read_json(run_dirs[arm] / "config.json") for arm in ARMS}
    check_configs(configs, list(ARMS), allow={"optim.epochs"})
    shared = model_kwargs_from_config(configs["vanilla"])
    # INV-STATS-SINGLE: cached read only.
    tensorized = REPO_ROOT / shared["tensorized_dir"]
    vocab = REPO_ROOT / shared["vocab_path"]
    train_ds = TensorizedPretrainDataset(tensorized / "train", vocab,
                                         max_seq_len=shared["max_seq_len"])
    stats = corpus_stats_cached(train_ds, tensorized / "train", split="train",
                                sample_windows=4000, seed=SIGNAL_SEED)
    selected = {}
    for arm in ARMS:
        ep, vl = select_best_epoch(run_dirs[arm])
        selected[arm] = {
            "epoch": ep,
            "val_loss_train_json": vl,
            "checkpoint": str(checkpoint_path(run_dirs[arm], ep)),
            "run_dir": str(run_dirs[arm]),
        }
    return {
        "run_root": str(run_root),
        "run_dirs": {a: str(run_dirs[a]) for a in ARMS},
        "configs": configs,
        "shared": shared,
        "corpus_stats": stats.to_json(),
        "selected": selected,
        "tensorized_dir": str(tensorized),
        "vocab_path": str(vocab),
        "tau_max": float(shared["tau_max"]),
    }


def target_gap_days(ds: TensorizedPretrainDataset, idx: int) -> float:
    """Signed days between last window event and first target-visit event."""
    shard_id, pos, visit_k = ds._index[idx]
    s = ds._load_shard(shard_id)
    ev_start = int(s["event_offsets"][pos])
    vis_start = int(s["visit_offsets"][pos])
    end_curr = int(s["visit_ends"][vis_start + visit_k])
    start_next = int(s["visit_starts"][vis_start + visit_k + 1])
    ts = s["timestamps_days"]
    t_end = float(ts[ev_start + end_curr - 1])
    t_tgt = float(ts[ev_start + start_next])
    return t_tgt - t_end


def materialize_fixed_batches(
    out_dir: Path,
    *,
    shared: dict,
    n_batches: int,
    batch_size: int,
    seed: int = SIGNAL_SEED,
    num_workers: int = 0,
    force: bool = False,
) -> dict:
    """Materialize packed index tensors once; hash; reuse by every condition/arm.

    Cache layout (v2): code_indices, timestamps_days, age_years, demographics,
    attention_mask, lengths, sparse target_code_idx, gaps — pinned-friendly CPU
    tensors. Embeddings are gathered on device per batch from the model table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / BATCH_STORE_NAME
    meta_path = out_dir / BATCH_META_NAME

    if store_path.is_file() and meta_path.is_file() and not force:
        meta = read_json(meta_path)
        if (int(meta.get("cache_version", -1)) == int(CACHE_VERSION)
                and int(meta.get("n_batches", -1)) == int(n_batches)
                and int(meta.get("batch_size", -1)) == int(batch_size)
                and int(meta.get("seed", -1)) == int(seed)):
            return meta

    set_seed(seed)
    ds = TensorizedPretrainDataset(
        REPO_ROOT / shared["tensorized_dir"] / "val",
        REPO_ROOT / shared["vocab_path"],
        max_seq_len=shared["max_seq_len"],
    )
    loader = make_val_loader(ds, batch_size, num_workers, shared["race_encoding"])
    hasher = BatchOrderHash()

    code_parts: list[torch.Tensor] = []
    ts_parts: list[torch.Tensor] = []
    age_parts: list[torch.Tensor] = []
    demo_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    len_parts: list[torch.Tensor] = []
    gap_parts: list[np.ndarray] = []
    age_last_parts: list[torch.Tensor] = []
    tgt_offsets = [0]
    tgt_values: list[np.ndarray] = []
    example_offset = 0
    max_len = 0

    for i, batch in enumerate(loader, 1):
        if i > n_batches:
            break
        hasher.update(batch)
        bsz = int(batch["lengths"].shape[0])
        L = int(batch["code_indices"].shape[1])
        max_len = max(max_len, L)
        gap = np.array(
            [target_gap_days(ds, example_offset + j) for j in range(bsz)],
            dtype=np.float64,
        )
        example_offset += bsz
        code_parts.append(batch["code_indices"].detach().cpu().contiguous())
        ts_parts.append(batch["timestamps_days"].detach().cpu().contiguous())
        age_parts.append(batch["age_years"].detach().cpu().contiguous())
        demo_parts.append(batch["demographics"].detach().cpu().contiguous())
        mask_parts.append(batch["attention_mask"].detach().cpu().contiguous())
        len_parts.append(batch["lengths"].detach().cpu().contiguous())
        gap_parts.append(gap)
        age_last_parts.append(age_last_of(batch).detach().cpu().contiguous())
        tgt = batch["target_codes"].detach().cpu().numpy()
        for r in range(bsz):
            pos = np.flatnonzero(tgt[r] > 0).astype(np.int32)
            tgt_values.append(pos)
            tgt_offsets.append(tgt_offsets[-1] + int(pos.size))

    if not code_parts:
        raise RuntimeError("[HARD] no validation batches materialized")

    # Pad variable-L batches to a common max_len for packing.
    def _pad_2d(parts, fill=0):
        out = []
        for t in parts:
            if t.shape[1] == max_len:
                out.append(t)
            else:
                pad = torch.full((t.shape[0], max_len - t.shape[1]), fill,
                                dtype=t.dtype)
                if t.dtype == torch.bool:
                    pad = torch.zeros((t.shape[0], max_len - t.shape[1]), dtype=torch.bool)
                out.append(torch.cat([t, pad], dim=1))
        return torch.cat(out, dim=0)

    def _pad_3d(parts):
        out = []
        for t in parts:
            if t.shape[1] == max_len:
                out.append(t)
            else:
                pad = torch.zeros(t.shape[0], max_len - t.shape[1], t.shape[2],
                                 dtype=t.dtype)
                out.append(torch.cat([t, pad], dim=1))
        return torch.cat(out, dim=0)

    patient_ids = patient_ids_from_dataset(ds, hasher.n_rows)
    payload = {
        "cache_version": int(CACHE_VERSION),
        "code_indices": _pad_2d(code_parts, fill=0).to(torch.int64),
        "timestamps_days": _pad_2d(
            [t.to(torch.float64) for t in ts_parts], fill=0.0).to(torch.float64),
        "age_years": _pad_2d(age_parts, fill=0.0).to(torch.float32),
        "demographics": _pad_3d(demo_parts).to(torch.float32),
        "attention_mask": _pad_2d(mask_parts).to(torch.bool),
        "lengths": torch.cat(len_parts, dim=0).to(torch.int64),
        "target_offsets": torch.tensor(tgt_offsets, dtype=torch.int64),
        "target_values": (torch.from_numpy(np.concatenate(tgt_values))
                          if tgt_values else torch.zeros(0, dtype=torch.int32)),
        "target_gap_days": torch.from_numpy(np.concatenate(gap_parts)).to(torch.float64),
        "age_last": torch.cat(age_last_parts, dim=0).to(torch.float32),
        "patient_ids": torch.from_numpy(np.asarray(patient_ids, dtype=np.int64)),
        "num_codes": int(shared["num_codes"]),
        "batch_list_hash": hasher.hexdigest,
        "n_batches": int(hasher.n_batches),
        "n_examples": int(hasher.n_rows),
        "batch_size": int(batch_size),
        "max_len": int(max_len),
        "seed": int(seed),
    }
    for k, v in list(payload.items()):
        if isinstance(v, torch.Tensor):
            try:
                payload[k] = v.pin_memory()
            except RuntimeError:
                pass

    tmp = store_path.with_suffix(store_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(store_path)

    gap_all = np.concatenate(gap_parts)
    age_all = payload["age_last"].numpy()
    meta = {
        "cache_version": int(CACHE_VERSION),
        "batch_list_hash": hasher.hexdigest,
        "n_batches": int(hasher.n_batches),
        "n_examples": int(hasher.n_rows),
        "batch_size": int(batch_size),
        "max_len": int(max_len),
        "seed": int(seed),
        "store_path": str(store_path),
        "gap_days_median": float(np.median(gap_all)) if gap_all.size else float("nan"),
        "age_median": float(np.median(age_all)) if age_all.size else float("nan"),
        "tau_max": float(shared["tau_max"]),
        "t4_sigma_content": float(T4_SIGMA_CONTENT),
    }
    write_json_atomic(meta_path, meta)
    return meta


def load_fixed_batches(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    store = torch.load(out_dir / BATCH_STORE_NAME, map_location="cpu", weights_only=False)
    meta = read_json(out_dir / BATCH_META_NAME)
    if int(store.get("cache_version", -1)) != int(CACHE_VERSION):
        raise AssertionError(
            f"[HARD] cache_version {store.get('cache_version')} != {CACHE_VERSION}; "
            f"re-materialize with --force")
    if store["batch_list_hash"] != meta["batch_list_hash"]:
        raise AssertionError(
            f"[HARD] store hash {store['batch_list_hash']} != meta {meta['batch_list_hash']}")
    if isinstance(store.get("patient_ids"), torch.Tensor):
        store["patient_ids"] = store["patient_ids"].numpy()
    return store


def densify_targets(store: dict, start: int, end: int) -> torch.Tensor:
    """Build dense [B, V] target matrix from sparse target_code_idx."""
    V = int(store["num_codes"])
    off = store["target_offsets"]
    vals = store["target_values"]
    B = end - start
    out = torch.zeros(B, V, dtype=torch.float32)
    for i, ei in enumerate(range(start, end)):
        a, b = int(off[ei]), int(off[ei + 1])
        if b > a:
            out[i, vals[a:b].long()] = 1.0
    return out


def iter_packed_batches(store: dict) -> Iterator[dict]:
    """Yield model-ready batch dicts by slicing the packed cache."""
    N = int(store["n_examples"])
    B = int(store["batch_size"])
    for start in range(0, N, B):
        end = min(start + B, N)
        yield slice_batch(store, start, end)


def slice_batch(store: dict, start: int, end: int) -> dict:
    return {
        "code_indices": store["code_indices"][start:end],
        "timestamps_days": store["timestamps_days"][start:end],
        "age_years": store["age_years"][start:end],
        "demographics": store["demographics"][start:end],
        "attention_mask": store["attention_mask"][start:end],
        "lengths": store["lengths"][start:end],
        "target_codes": densify_targets(store, start, end),
        "target_gap_days": store["target_gap_days"][start:end],
        "age_last": store["age_last"][start:end],
        "_row_start": start,
        "_row_end": end,
    }


def assert_batch_hash(store: dict, expected: str | None = None) -> str:
    hasher = BatchOrderHash()
    for batch in iter_packed_batches(store):
        hasher.update({k: v for k, v in batch.items() if not k.startswith("_")
                       and k not in ("target_gap_days", "age_last")})
    got = hasher.hexdigest
    if got != store["batch_list_hash"]:
        raise AssertionError(
            f"[HARD] recomputed batch hash {got} != stored {store['batch_list_hash']}")
    if expected is not None and got != expected:
        raise AssertionError(
            f"[HARD] batch hash {got} != expected {expected}")
    return got


def iter_store_batches(store: dict) -> Iterator[dict]:
    if "batches" in store and isinstance(store["batches"], list) and store["batches"]:
        yield from store["batches"]
    else:
        yield from iter_packed_batches(store)


def recall_from_scores(scores: torch.Tensor, targets: torch.Tensor,
                       ks: tuple[int, ...] = KS) -> dict[int, np.ndarray]:
    """Per-example recall@k from arbitrary score matrix (same def as diagnostics)."""
    out = {}
    metrics = D.topk_per_example(scores, targets, ks=ks)
    for k in ks:
        out[k] = metrics[f"recall@{k}"].numpy().astype(np.float64)
    return out


def per_code_hit_miss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    ks: tuple[int, ...] = KS,
    n_pos: np.ndarray | None = None,
    n_hit: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Accumulate per-code positive / top-k hit counts.

    For every example where code ``c`` is a true label:
      ``n_pos[c] += 1``
      ``n_hit[k][c] += 1`` if ``c`` is among the top-``k`` scored codes.

    Miss@k is ``n_pos - n_hit[k]`` (computed at serialize time).
    """
    scores = scores.float()
    targets = targets.float()
    V = int(targets.shape[-1])
    if n_pos is None:
        n_pos = np.zeros(V, dtype=np.int64)
    if n_hit is None:
        n_hit = {k: np.zeros(V, dtype=np.int64) for k in ks}

    pos = targets > 0.5
    n_pos += pos.sum(dim=0).detach().cpu().numpy().astype(np.int64)

    k_max = int(max(ks))
    top = scores.topk(min(k_max, scores.shape[-1]), dim=-1).indices  # [B, k_max]
    for k in ks:
        # Hit mask: code is positive AND appears in top-k.
        hit = torch.zeros_like(targets)
        top_k = top[:, :k]
        # Scatter 1s at top-k positions, then AND with positives.
        hit.scatter_(1, top_k, torch.ones_like(top_k, dtype=hit.dtype))
        hit = (hit > 0) & pos
        n_hit[k] += hit.sum(dim=0).detach().cpu().numpy().astype(np.int64)
    return n_pos, n_hit


def serialize_per_code_hit_miss(
    n_pos: np.ndarray,
    n_hit: dict[int, np.ndarray],
    *,
    ks: tuple[int, ...] = KS,
) -> dict:
    """Sparse JSON form: only codes with at least one positive label."""
    codes = np.flatnonzero(n_pos > 0).astype(np.int32)
    out: dict[str, Any] = {
        "n_codes_with_positives": int(codes.size),
        "code_id": codes.tolist(),
        "n_pos": n_pos[codes].astype(np.int64).tolist(),
    }
    for k in ks:
        hits = n_hit[k][codes].astype(np.int64)
        misses = n_pos[codes].astype(np.int64) - hits
        out[f"hit@{k}"] = hits.tolist()
        out[f"miss@{k}"] = misses.tolist()
    return out


def mean_bce(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """Per-example mean BCE over codes (matches T1 / train val_loss scale)."""
    per = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none").mean(dim=-1)
    return per.detach().cpu().numpy().astype(np.float64)


def paired_delta_ci(a: np.ndarray, b: np.ndarray, patient_ids: np.ndarray, *,
                    n_boot: int, seed: int) -> dict:
    """Paired bootstrap CI for mean(a) - mean(b). Degenerate when a==b exactly."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    point = float(np.nanmean(a) - np.nanmean(b))
    if a.size and np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True):
        return {
            "point": 0.0,
            "ci": {"lo": 0.0, "hi": 0.0, "n_boot": int(n_boot), "degenerate": True,
                   "excludes_zero": False},
            "covers_zero": True,
        }

    def _mean(arr):
        def stat(rows):
            if rows.size == 0:
                return float("nan")
            v = arr[rows]
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size else float("nan")
        return stat

    ci = D.paired_bootstrap_ci(_mean(a), _mean(b), patient_ids, n_boot=n_boot, seed=seed)
    return {
        "point": point,
        "ci": ci,
        "covers_zero": bool(ci["lo"] <= 0.0 <= ci["hi"]),
    }


def nanmean_safe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


@torch.inference_mode()
def eval_model_on_store(
    model,
    store: dict,
    device: torch.device,
    *,
    mutate_batch: Callable[[dict], dict] | None = None,
    dtype: torch.dtype = torch.float32,
    ks: tuple[int, ...] = KS,
    collect_per_code: bool = False,
) -> dict:
    """Forward over the fixed store; optional timestamp mutation before forward."""
    model.eval()
    hasher = BatchOrderHash()
    bce_parts: list[np.ndarray] = []
    rec: dict[int, list[np.ndarray]] = {k: [] for k in ks}
    ages: list[np.ndarray] = []
    gaps: list[np.ndarray] = []
    n_pos: np.ndarray | None = None
    n_hit: dict[int, np.ndarray] | None = None

    for raw in iter_store_batches(store):
        batch = {k: v for k, v in raw.items()
                 if k not in ("target_gap_days", "age_last") and not str(k).startswith("_")}
        hasher.update(batch)
        if mutate_batch is not None:
            batch = mutate_batch({k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                  for k, v in batch.items()})
        b = to_device(batch, device)
        if dtype != torch.float32 and device.type == "cuda":
            for key in ("timestamps_days", "age_years", "demographics"):
                if key in b and b[key].is_floating_point():
                    b[key] = b[key].to(dtype)
        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=(dtype != torch.float32 and device.type == "cuda")):
            logits = model(b)["code_logits"]
        logits_f = logits.float()
        targets = b["target_codes"].float()
        bce_parts.append(mean_bce(logits_f, targets))
        for k, arr in recall_from_scores(logits_f, targets, ks=ks).items():
            rec[k].append(arr)
        if collect_per_code:
            n_pos, n_hit = per_code_hit_miss(
                logits_f, targets, ks=ks, n_pos=n_pos, n_hit=n_hit,
            )
        ages.append(raw["age_last"].numpy())
        gaps.append(raw["target_gap_days"].numpy())

    bce = np.concatenate(bce_parts) if bce_parts else np.zeros(0)
    recall = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rec.items()}
    out = {
        "batch_list_hash": hasher.hexdigest,
        "bce_per_example": bce,
        "bce_mean": nanmean_safe(bce),
        "recall_per_example": {f"recall@{k}": recall[k] for k in ks},
        "recall": {f"recall@{k}": nanmean_safe(recall[k]) for k in ks},
        "age_last": np.concatenate(ages) if ages else np.zeros(0),
        "target_gap_days": np.concatenate(gaps) if gaps else np.zeros(0),
    }
    if collect_per_code and n_pos is not None and n_hit is not None:
        out["per_code_hit_miss"] = serialize_per_code_hit_miss(n_pos, n_hit, ks=ks)
    return out


def probe_precision(model, store: dict, device: torch.device, *,
                    tol: float = 1e-6) -> dict:
    """Confirm baseline val BCE agrees fp32 vs bf16 to ``tol``; else stay fp32."""
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        return {"dtype": "fp32", "reason": "no_bf16_device", "agree": False,
                "delta_bce": float("nan")}
    # Tiny packed store: first 2 batches worth of rows.
    B = int(store["batch_size"])
    n = min(int(store["n_examples"]), 2 * B)
    tiny = {k: (v[:n] if isinstance(v, torch.Tensor) and v.shape[0] == store["n_examples"]
                else v)
            for k, v in store.items() if k not in ("batches",)}
    # Fix sparse targets for the prefix.
    tiny["n_examples"] = n
    tiny["n_batches"] = int(np.ceil(n / B))
    tiny["target_offsets"] = store["target_offsets"][: n + 1].clone()
    last = int(store["target_offsets"][n])
    tiny["target_values"] = store["target_values"][:last]
    tiny["patient_ids"] = np.asarray(store["patient_ids"][:n])
    r32 = eval_model_on_store(model, tiny, device, dtype=torch.float32)
    r16 = eval_model_on_store(model, tiny, device, dtype=torch.bfloat16)
    delta = abs(float(r32["bce_mean"]) - float(r16["bce_mean"]))
    agree = bool(delta < tol)
    return {
        "dtype": "bf16" if agree else "fp32",
        "reason": "agree" if agree else f"delta_bce={delta:.3e} >= {tol}",
        "agree": agree,
        "delta_bce": delta,
        "bce_fp32": float(r32["bce_mean"]),
        "bce_bf16": float(r16["bce_mean"]),
    }


def autotune_batch_size(model, store: dict, device: torch.device, *,
                        start: int = 128, max_try: int = 512) -> int:
    """Validate materialized batch size fits; record it (hash keeps batch boundaries)."""
    if device.type != "cuda":
        return int(store.get("batch_size", start))
    model.eval()
    probe = next(iter_packed_batches(store))
    b0 = to_device({k: v for k, v in probe.items()
                    if k not in ("target_gap_days", "age_last") and not str(k).startswith("_")},
                   device)
    bsz = int(b0["lengths"].shape[0])
    try:
        with torch.inference_mode():
            _ = model(b0)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        raise RuntimeError(
            f"[HARD] materialized batch_size={bsz} OOMs on device; "
            f"re-materialize with a smaller --batch_size") from e
    if bsz < int(MIN_GPU_BATCH_SIZE):
        raise AssertionError(
            f"[HARD] autotune: effective batch_size={bsz} < {MIN_GPU_BATCH_SIZE}")
    return int(bsz)


def load_arm_model(ctx: dict, arm: str, device: torch.device, *,
                   epoch: int | None = None):
    shared = ctx["shared"]
    sel = ctx["selected"][arm]
    ep = int(epoch if epoch is not None else sel["epoch"])
    path = checkpoint_path(Path(sel["run_dir"]), ep)
    model = build_model(shared, arm)
    meta = load_checkpoint(model, path, arm=arm, epoch=ep, device=device)
    model.eval()
    return model, meta


def mutate_constant_timestamps(batch: dict) -> dict:
    """All valid timestamps equal → τ = 0 → τ̃ = −1."""
    out = dict(batch)
    ts = batch["timestamps_days"].clone()
    mask = batch["attention_mask"]
    # Set every valid position to the same constant (0.0).
    ts = torch.where(mask, torch.zeros((), dtype=ts.dtype, device=ts.device), ts)
    out["timestamps_days"] = ts
    return out


def mutate_shuffle_within(batch: dict, rng: np.random.Generator) -> dict:
    """Permute timestamps across valid positions; codes fixed. Multiset preserved."""
    out = dict(batch)
    ts = batch["timestamps_days"].clone()
    mask = batch["attention_mask"].numpy()
    ts_np = ts.numpy().copy()
    for i in range(ts_np.shape[0]):
        idx = np.flatnonzero(mask[i])
        if idx.size <= 1:
            continue
        vals = ts_np[i, idx].copy()
        rng.shuffle(vals)
        ts_np[i, idx] = vals
    out["timestamps_days"] = torch.from_numpy(ts_np)
    return out


def mutate_jitter(batch: dict, days: float, rng: np.random.Generator) -> dict:
    """±k days uniform noise per timestamp, then re-sort valid positions by time.

    Codes move with their timestamps (pair integrity); the sequence order becomes
    chronological under the jittered clock. Attention mask / lengths unchanged.
    """
    out = dict(batch)
    ts = batch["timestamps_days"].numpy().copy()
    codes = batch["code_indices"].numpy().copy()
    ages = batch["age_years"].numpy().copy()
    demo = batch["demographics"].numpy().copy()
    mask = batch["attention_mask"].numpy()
    for i in range(ts.shape[0]):
        idx = np.flatnonzero(mask[i])
        if idx.size == 0:
            continue
        noise = rng.uniform(-float(days), float(days), size=idx.size)
        new_ts = ts[i, idx] + noise
        order = np.argsort(new_ts, kind="mergesort")
        ts[i, idx] = new_ts[order]
        codes[i, idx] = codes[i, idx][order]
        ages[i, idx] = ages[i, idx][order]
        demo[i, idx] = demo[i, idx][order]
    out["timestamps_days"] = torch.from_numpy(ts)
    out["code_indices"] = torch.from_numpy(codes)
    out["age_years"] = torch.from_numpy(ages)
    out["demographics"] = torch.from_numpy(demo)
    return out


def assert_constant_tau_zero(batch: dict) -> None:
    tau, _ = tau_from_timestamps(batch["timestamps_days"], batch["attention_mask"],
                                 batch["lengths"])
    # On valid pairs τ must be 0; padded entries are zeroed by pairwise_tau.
    if float(tau.max()) != 0.0:
        raise AssertionError(
            f"[HARD] constant condition: τ.max()={float(tau.max())} != 0")


def assert_shuffle_preserves_multiset(orig: dict, shuffled: dict) -> None:
    mask = orig["attention_mask"].numpy()
    a = orig["timestamps_days"].numpy()
    b = shuffled["timestamps_days"].numpy()
    for i in range(a.shape[0]):
        idx = np.flatnonzero(mask[i])
        if not np.allclose(np.sort(a[i, idx]), np.sort(b[i, idx])):
            raise AssertionError(
                f"[HARD] shuffle_within altered timestamp multiset at row {i}")
        if idx.size > 1 and np.allclose(a[i, idx], b[i, idx]):
            # Extremely unlikely under RNG; warn via assert only if identical and size large
            pass


def ensure_batches(out_dir: Path, *, smoke: bool, batch_size: int = 128,
                   force: bool = False, run_root: Path | None = None) -> tuple[dict, dict]:
    """Load shared ctx + materialize / load fixed batches."""
    flags = smoke_flags(smoke)
    ctx = load_shared(run_root)
    meta = materialize_fixed_batches(
        out_dir, shared=ctx["shared"], n_batches=flags["n_batches"],
        batch_size=batch_size, seed=SIGNAL_SEED,
        num_workers=flags["num_workers"], force=force,
    )
    store = load_fixed_batches(out_dir)
    assert_batch_hash(store, meta["batch_list_hash"])
    ctx["batch_meta"] = meta
    ctx["flags"] = flags
    return ctx, store


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Re-materialize fixed batches even if they exist.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=SIGNAL_SEED)
    return p


def base_result_meta(ctx: dict, store: dict) -> dict:
    return {
        "seed": int(SIGNAL_SEED),
        "batch_list_hash": store["batch_list_hash"],
        "n_batches": int(store["n_batches"]),
        "n_examples": int(store["n_examples"]),
        "batch_size": int(store["batch_size"]),
        "smoke": bool(ctx["flags"]["smoke"]),
        "n_boot": int(ctx["flags"]["n_boot"]),
        "tau_max": float(ctx["tau_max"]),
        "audit_seed": AUDIT_SEED,
    }
