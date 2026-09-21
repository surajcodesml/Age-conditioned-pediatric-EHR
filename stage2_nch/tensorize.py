"""Window already-mapped NCH events into Stage-1-style next-encounter shards.

This is NOT a re-run of NCH code mapping. It reads v2 cleaned events (already
rolled onto the frozen MIMIC vocabulary) and writes forecast shards whose
schema matches ``model_new.tensorize_pretrain._write_flat_shard``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_new.data import DAYS_PER_YEAR, load_vocab
from model_new.tensorize_pretrain import _write_flat_shard
from stage2_nch.config import (
    NCH_CLEAN_EVENTS,
    NCH_INDEX_FIRST,
    NCH_SPLIT_DIR,
    NCH_TENSORIZED_DIR,
    PEDIATRIC_AGE_MAX_YEARS,
    PEDIATRIC_AGE_MIN_YEARS,
    PRIMARY_MODALITIES,
    PRIMARY_REPRESENTATION,
    VOCAB_PATH,
)


def _load_split_ids(split_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for name in ("train", "val", "test"):
        path = Path(split_dir) / f"{name}_patient_ids.json"
        ids = json.loads(path.read_text(encoding="utf-8"))
        out[name] = np.asarray(ids, dtype=np.int64)
    return out


def _payload_for_patient(
    g: pd.DataFrame,
    *,
    unk: int,
    max_age_days: float,
    min_age_days: float,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    stats = {
        "n_events": int(len(g)),
        "n_oov": 0,
        "n_mapped": 0,
        "n_age_lt0": 0,
        "n_age_gt18": 0,
        "n_age_clipped": 0,
        "n_encounters": 0,
        "kept": 0,
    }
    if g.empty:
        return None, stats
    g = g.sort_values(["event_time", "encounter_id", "mimic_token_id"], kind="mergesort")
    ages = g["age_at_event_days"].to_numpy(dtype=np.float64)
    stats["n_age_lt0"] = int((ages < min_age_days).sum())
    stats["n_age_gt18"] = int((ages > max_age_days).sum())
    if stats["n_age_lt0"] or stats["n_age_gt18"]:
        ages = np.clip(ages, min_age_days, max_age_days)
        stats["n_age_clipped"] = int(stats["n_age_lt0"] + stats["n_age_gt18"])

    tid = g["mimic_token_id"].to_numpy(dtype=np.float64)
    mapped = np.isfinite(tid) & (tid >= 0) & (tid < unk)
    codes = np.full(len(g), unk, dtype=np.int64)
    codes[mapped] = tid[mapped].astype(np.int64)
    stats["n_mapped"] = int(mapped.sum())
    stats["n_oov"] = int((~mapped).sum())

    times = pd.to_datetime(g["event_time"])
    t0 = times.iloc[0]
    ts_days = ((times - t0).dt.total_seconds().to_numpy(dtype=np.float64)) / 86400.0
    enc = g["encounter_id"].to_numpy(dtype=np.int64)
    order_enc = []
    seen: set[int] = set()
    for e in enc.tolist():
        if e not in seen:
            seen.add(e)
            order_enc.append(int(e))
    if len(order_enc) < 2:
        stats["n_encounters"] = len(order_enc)
        return None, stats

    code_blocks, ts_blocks, age_blocks, spans = [], [], [], []
    cursor = 0
    for e in order_enc:
        ix = np.flatnonzero(enc == e)
        n = int(ix.size)
        if n == 0:
            continue
        code_blocks.append(codes[ix])
        ts_blocks.append(ts_days[ix].astype(np.float32))
        age_blocks.append(ages[ix].astype(np.float32))
        spans.append((cursor, cursor + n))
        cursor += n
    if len(spans) < 2 or cursor == 0:
        stats["n_encounters"] = len(spans)
        return None, stats

    stats["n_encounters"] = len(spans)
    stats["kept"] = 1
    sex = int(g["sex"].iloc[0]) if pd.notna(g["sex"].iloc[0]) else 0
    race = int(g["race"].iloc[0]) if pd.notna(g["race"].iloc[0]) else 6
    payload = {
        "subject_id": int(g["patient_id"].iloc[0]),
        "sex": sex,
        "race": race,
        "code_indices": np.concatenate(code_blocks),
        "timestamps_days": np.concatenate(ts_blocks),
        "age_days": np.concatenate(age_blocks),
        "visit_spans": np.asarray(spans, dtype=np.int32),
    }
    return payload, stats


def build_forecast_shards(
    *,
    events_path: Path = NCH_CLEAN_EVENTS,
    index_path: Path = NCH_INDEX_FIRST,
    split_dir: Path = NCH_SPLIT_DIR,
    out_dir: Path = NCH_TENSORIZED_DIR / PRIMARY_REPRESENTATION,
    vocab_path: Path = VOCAB_PATH,
    modalities: tuple[str, ...] = PRIMARY_MODALITIES,
    force: bool = False,
) -> dict[str, Any]:
    """Write train/val/test forecast shards if missing (or if ``force``)."""
    out_dir = Path(out_dir)
    marker = out_dir / "tensorize_report.json"
    if marker.exists() and not force and all(
        (out_dir / split / "shard_000.npz").exists() for split in ("train", "val", "test")
    ):
        return json.loads(marker.read_text(encoding="utf-8"))

    vocab = load_vocab(vocab_path)
    unk = len(vocab)
    splits = _load_split_ids(split_dir)
    overlap = {
        "train&val": int(len(set(splits["train"]) & set(splits["val"]))),
        "train&test": int(len(set(splits["train"]) & set(splits["test"]))),
        "val&test": int(len(set(splits["val"]) & set(splits["test"]))),
    }
    if any(overlap.values()):
        raise AssertionError(f"patient leakage in NCH splits: {overlap}")

    events = pd.read_parquet(events_path)
    events = events[events["event_type"].isin(modalities)].copy()
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")
    events = events.dropna(subset=["event_time", "patient_id", "encounter_id"])

    index = pd.read_parquet(index_path)
    index["index_time"] = pd.to_datetime(index["index_time"], errors="coerce")
    first = (
        index.sort_values(["patient_id", "index_time", "sleep_study_id"])
        .groupby("patient_id", as_index=False)
        .first()[["patient_id", "index_time", "index_age_days", "sex", "race"]]
    )
    merged = events.merge(first, on="patient_id", how="inner", suffixes=("", "_idx"))
    pre = merged[merged["event_time"] < merged["index_time"]].copy()

    min_age_days = PEDIATRIC_AGE_MIN_YEARS * DAYS_PER_YEAR
    max_age_days = PEDIATRIC_AGE_MAX_YEARS * DAYS_PER_YEAR
    totals = {
        "n_age_lt0": 0, "n_age_gt18": 0, "n_age_clipped": 0,
        "n_oov": 0, "n_mapped": 0, "n_events": 0,
        "n_patients_lt2_encounters": 0,
    }
    split_payloads: dict[str, list[dict]] = {k: [] for k in splits}
    id_to_split = {}
    for name, arr in splits.items():
        for pid in arr.tolist():
            id_to_split[int(pid)] = name

    for pid, g in pre.groupby("patient_id", sort=False):
        payload, st = _payload_for_patient(
            g, unk=unk, max_age_days=max_age_days, min_age_days=min_age_days)
        for k in ("n_age_lt0", "n_age_gt18", "n_age_clipped", "n_oov", "n_mapped", "n_events"):
            totals[k] += st[k]
        split_name = id_to_split.get(int(pid))
        if payload is None:
            totals["n_patients_lt2_encounters"] += 1
            continue
        if split_name is None:
            continue
        split_payloads[split_name].append(payload)

    clip_needed = totals["n_age_clipped"] > 0
    report: dict[str, Any] = {
        "representation": PRIMARY_REPRESENTATION,
        "modalities": list(modalities),
        "events_path": str(events_path),
        "index_path": str(index_path),
        "vocab_size": unk,
        "unk_vocab_index": unk,
        "horizon": "next NCH encounter_id after a strict time cut; pre first sleep-study index",
        "age_out_of_range": {
            "n_age_lt0": totals["n_age_lt0"],
            "n_age_gt18": totals["n_age_gt18"],
            "n_clipped_to_0_18": totals["n_age_clipped"],
            "clip_applied": clip_needed,
            "note": (
                "Primary z_P(a)=(a-9)/9 is unclipped. Ages outside [0,18] are clipped "
                "in the shards only when they occur."
                if clip_needed else
                "No pre-index pediatric diagnosis ages outside [0, 18]; z_P is unclipped."
            ),
        },
        "oov": {
            "n_mapped": totals["n_mapped"],
            "n_oov_as_unk": totals["n_oov"],
            "event_weighted_mapped_pct": (
                100.0 * totals["n_mapped"] / max(1, totals["n_mapped"] + totals["n_oov"])
            ),
            "target_drops_unk": True,
        },
        "patient_splits": {k: int(v.size) for k, v in splits.items()},
        "overlap": overlap,
        "n_patients_lt2_encounters": totals["n_patients_lt2_encounters"],
        "splits": {},
        "clip_applied": clip_needed,
    }
    for name, payloads in split_payloads.items():
        shard = out_dir / name / "shard_000.npz"
        n_pat, n_ev = _write_flat_shard(shard, payloads, unk)
        n_windows = 0
        n_ge2 = 0
        for p in payloads:
            nv = int(p["visit_spans"].shape[0])
            if nv >= 2:
                n_ge2 += 1
                n_windows += nv - 1
        report["splits"][name] = {
            "n_patients_written": n_pat,
            "n_events": n_ev,
            "n_patients_with_ge2_encounters": n_ge2,
            "n_forecast_windows_upper_bound": n_windows,
            "shard": str(shard),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def ensure_forecast_shards(**kwargs) -> dict[str, Any]:
    return build_forecast_shards(**kwargs)
