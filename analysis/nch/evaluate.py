"""Micro-AUPRC of NCH Content-Persistence DTR by age band and history horizon.

Loads ``results/baselines/nch/dtr_age_temporal_new`` and
``dtr_temporal_only_new``. For each horizon the input is restricted to events
within that many days of the last input timestamp (the full history is the
untruncated test window). Labels are never truncated.

Micro-AUPRC uses the same 5e6 label–score reservoir as
``baselines.common.metrics.multilabel_metrics`` when a cell exceeds that cap.
The full held-out test split is used (not the 100-batch cap from training).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baselines.common.metrics import safe_auprc  # noqa: E402
from baselines.mimic.runner import MIMICCPDTRAdapter, make_cp_collate  # noqa: E402
from stage2_nch.config import AGE_BANDS, VOCAB_PATH  # noqa: E402
from stage2_nch.dataset import NCHForecastDataset, make_nch_collate  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "auprc_by_age_horizon.json"
DTR_DIR = REPO / "results" / "baselines" / "nch" / "dtr_age_temporal_new"
TEMPORAL_DIR = REPO / "results" / "baselines" / "nch" / "dtr_temporal_only_new"
TENSORIZED_TEST = (
    REPO / "artifacts" / "nch_stage2" / "v2" / "tensorized_forecast" / "diagnoses_only" / "test"
)

# Display order. Day cutoffs are calendar days before the last input event.
HORIZONS: tuple[tuple[str, float | None], ...] = (
    ("30d", 30.0),
    ("90d", 90.0),
    ("180d", 180.0),
    ("1y", 365.25),
    ("3y", 3 * 365.25),
    ("5y", 5 * 365.25),
    ("Full", None),
)
HEATMAP_HORIZONS: tuple[str, ...] = ("30d", "90d", "180d", "1y", "3y", "Full")
AGE_BAND_NAMES: tuple[str, ...] = tuple(name for name, _, _ in AGE_BANDS)
MICRO_CAP = 5_000_000
MAX_SEQ_LEN = 256


class ScoreReservoir:
    """Uniform reservoir of (label, score) pairs, Vitter's Algorithm R."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.y = np.empty(self.capacity, dtype=np.int8)
        self.s = np.empty(self.capacity, dtype=np.float32)
        self.n_seen = 0
        self.n_filled = 0

    def add(self, y: np.ndarray, scores: np.ndarray) -> None:
        y = np.asarray(y, dtype=np.int8).ravel()
        scores = np.asarray(scores, dtype=np.float32).ravel()
        n = int(y.size)
        if n == 0:
            return
        if y.shape != scores.shape:
            raise ValueError(f"label/score length mismatch: {y.shape} vs {scores.shape}")
        if self.n_filled < self.capacity:
            take = min(n, self.capacity - self.n_filled)
            self.y[self.n_filled : self.n_filled + take] = y[:take]
            self.s[self.n_filled : self.n_filled + take] = scores[:take]
            self.n_filled += take
            self.n_seen += take
            y = y[take:]
            scores = scores[take:]
            n = int(y.size)
            if n == 0:
                return
        # 1-based indices of the incoming items among everything seen so far.
        idx = np.arange(self.n_seen + 1, self.n_seen + n + 1, dtype=np.float64)
        accept = self.rng.random(n) < (self.capacity / idx)
        chosen = np.flatnonzero(accept)
        if chosen.size:
            slots = self.rng.integers(0, self.capacity, size=int(chosen.size))
            self.y[slots] = y[chosen]
            self.s[slots] = scores[chosen]
        self.n_seen += n

    def auprc(self) -> float:
        if self.n_filled == 0:
            return float("nan")
        return float(safe_auprc(self.y[: self.n_filled], self.s[: self.n_filled]))


def truncate_item(item: dict[str, Any], horizon_days: float | None) -> dict[str, Any]:
    """Keep input events within ``horizon_days`` of the last timestamp.

    The prediction-time age and the labels are left unchanged. At least the
    last input event is kept so the collate never sees an empty sequence.
    """
    if horizon_days is None:
        return item
    ts = np.asarray(item["timestamps_days"], dtype=np.float64)
    if ts.size == 0:
        return item
    t_last = float(ts[-1])
    keep = (t_last - ts) <= float(horizon_days)
    if bool(keep.all()):
        return item
    if not bool(keep.any()):
        keep = np.zeros(ts.shape, dtype=bool)
        keep[-1] = True
    out = dict(item)
    for key in ("code_indices", "timestamps_days", "age_days"):
        out[key] = np.asarray(item[key])[keep].copy()
    out["n_input_events"] = int(out["code_indices"].shape[0])
    return out


class TruncatedHistory(Dataset):
    def __init__(self, base: Dataset, horizon_days: float | None) -> None:
        self.base = base
        self.horizon_days = horizon_days

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return truncate_item(self.base[idx], self.horizon_days)


def _prefer_best_checkpoint(model: MIMICCPDTRAdapter, run_dir: Path) -> Path:
    best = run_dir / "best_checkpoint.pt"
    if best.exists():
        state = torch.load(best, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return best
    model.load_checkpoint(run_dir)
    return run_dir / "checkpoint.pt"


def _collate_with_meta(base_collate):
    def _collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        bands = [str(ex["age_band"]) for ex in examples]
        pids = [int(ex["patient_id"]) for ex in examples]
        batch = base_collate(examples)
        batch["age_band"] = bands
        batch["patient_id"] = pids
        return batch

    return _collate


def _new_reservoirs(seed: int) -> dict[str, ScoreReservoir]:
    out = {"_overall": ScoreReservoir(MICRO_CAP, seed)}
    for i, name in enumerate(AGE_BAND_NAMES):
        out[name] = ScoreReservoir(MICRO_CAP, seed + 17 * (i + 1))
    return out


def _empty_counts() -> dict[str, dict[str, Any]]:
    keys = ("_overall",) + AGE_BAND_NAMES
    return {k: {"n_windows": 0, "patients": set()} for k in keys}


def evaluate(
    *,
    max_examples: int = 0,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    """Run both NCH arms over every history horizon and return AUPRC tables."""
    if not TENSORIZED_TEST.exists():
        raise FileNotFoundError(f"NCH test split missing: {TENSORIZED_TEST}")
    for run_dir in (DTR_DIR, TEMPORAL_DIR):
        if not (run_dir / "best_checkpoint.pt").exists() and not (run_dir / "checkpoint.pt").exists():
            raise FileNotFoundError(f"checkpoint missing under {run_dir}")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    dev = torch.device(device)
    base = NCHForecastDataset(TENSORIZED_TEST, VOCAB_PATH, max_seq_len=MAX_SEQ_LEN)
    n_codes = int(base.num_codes)
    if max_examples and max_examples < len(base):
        base = torch.utils.data.Subset(base, range(int(max_examples)))

    models: dict[str, MIMICCPDTRAdapter] = {}
    ckpts: dict[str, str] = {}
    for arm, run_dir in (("dtr", DTR_DIR), ("temporal", TEMPORAL_DIR)):
        model = MIMICCPDTRAdapter(
            n_codes=n_codes,
            arm="age_temporal" if arm == "dtr" else "temporal_only",
            d_model=64,
        )
        ckpts[arm] = str(_prefer_best_checkpoint(model, run_dir))
        model.to(dev)
        model.eval()
        models[arm] = model
    print(f"loaded DTR {ckpts['dtr']}", flush=True)
    print(f"loaded temporal-only {ckpts['temporal']}", flush=True)

    base_collate = make_cp_collate(make_nch_collate("one_hot", assert_horizon=False))
    collate = _collate_with_meta(base_collate)

    horizons_out: dict[str, Any] = {}
    t_all = time.time()
    for h_i, (label, days) in enumerate(HORIZONS):
        ds = TruncatedHistory(base, days)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        reservoirs = {arm: _new_reservoirs(seed + 1000 * h_i) for arm in models}
        counts = _empty_counts()
        t0 = time.time()
        n_seen = 0
        for batch in loader:
            bands = batch.pop("age_band")
            pids = batch.pop("patient_id")
            labels = batch["labels"]
            model_batch = {
                k: v.to(dev) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k != "labels"
            }
            y = labels.detach().cpu().numpy().astype(np.int8, copy=False)
            with torch.no_grad():
                scores = {
                    arm: models[arm].predict(model_batch)["logits"].detach().cpu().numpy()
                    for arm in models
                }
            band_arr = np.asarray(bands, dtype=object)
            for band_name in AGE_BAND_NAMES:
                sel = band_arr == band_name
                n_sel = int(sel.sum())
                if n_sel == 0:
                    continue
                counts[band_name]["n_windows"] += n_sel
                counts[band_name]["patients"].update(int(pid) for pid, keep in zip(pids, sel) if keep)
                for arm in models:
                    reservoirs[arm][band_name].add(y[sel], scores[arm][sel])
            counts["_overall"]["n_windows"] += len(bands)
            counts["_overall"]["patients"].update(int(pid) for pid in pids)
            for arm in models:
                reservoirs[arm]["_overall"].add(y, scores[arm])
            n_seen += len(bands)
            if n_seen % (batch_size * 40) < batch_size:
                print(f"  {label}: {n_seen} windows", flush=True)

        cell: dict[str, Any] = {}
        for key, meta in counts.items():
            a_dtr = reservoirs["dtr"][key].auprc() if key in reservoirs["dtr"] else float("nan")
            a_to = reservoirs["temporal"][key].auprc() if key in reservoirs["temporal"] else float("nan")
            cell[key] = {
                "auprc_dtr": a_dtr,
                "auprc_temporal": a_to,
                "delta_auprc": (
                    float(a_dtr - a_to) if np.isfinite(a_dtr) and np.isfinite(a_to) else float("nan")
                ),
                "n_windows": int(meta["n_windows"]),
                "n_patients": int(len(meta["patients"])),
                "n_pairs_seen_dtr": int(reservoirs["dtr"][key].n_seen) if key in reservoirs["dtr"] else 0,
                "micro_subsampled_to": MICRO_CAP,
            }
        horizons_out[label] = {
            "horizon_days": days,
            "cells": cell,
        }
        overall = cell["_overall"]["delta_auprc"]
        print(
            f"{label}: ΔAUPRC={overall:+.6f}  "
            f"DTR={cell['_overall']['auprc_dtr']:.4f}  "
            f"temporal={cell['_overall']['auprc_temporal']:.4f}  "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    payload = {
        "task": "nch_next_encounter",
        "models": {
            "dtr": "dtr_age_temporal_new",
            "temporal_only": "dtr_temporal_only_new",
            "checkpoints": ckpts,
        },
        "definition": "delta_auprc = micro_AUPRC(DTR) - micro_AUPRC(temporal-only)",
        "history_rule": (
            "Keep input events with (t_last - t) <= horizon_days; "
            "Full leaves the test window unchanged. Labels are not truncated."
        ),
        "micro_cap": MICRO_CAP,
        "micro_seed": seed,
        "max_examples": int(max_examples),
        "n_examples": int(len(base)),
        "age_bands": list(AGE_BAND_NAMES),
        "horizons": [label for label, _ in HORIZONS],
        "heatmap_horizons": list(HEATMAP_HORIZONS),
        "by_horizon": horizons_out,
        "elapsed_s": time.time() - t_all,
    }
    return payload


def save_metrics(payload: dict[str, Any], path: Path | None = None) -> Path:
    path = CACHE_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return path


def load_metrics(path: Path | None = None) -> dict[str, Any]:
    path = CACHE_PATH if path is None else Path(path)
    return json.loads(path.read_text())


def metrics_match_request(payload: dict[str, Any], max_examples: int) -> bool:
    return int(payload.get("max_examples", -1)) == int(max_examples) and int(
        payload.get("n_examples", 0)
    ) > 0


def get_metrics(*, recompute: bool = False, max_examples: int = 0, batch_size: int = 128, device: str = "cpu") -> dict[str, Any]:
    if CACHE_PATH.exists() and not recompute:
        cached = load_metrics()
        if metrics_match_request(cached, max_examples):
            print(f"using cached NCH AUPRC table {CACHE_PATH}")
            return cached
        print("cache does not match this evaluation setting; recomputing")
    payload = evaluate(max_examples=max_examples, batch_size=batch_size, device=device)
    save_metrics(payload)
    return payload


def delta_matrix(payload: dict[str, Any], horizons: tuple[str, ...] | list[str]) -> np.ndarray:
    """Rows = age bands, columns = horizons. Values are ΔAUPRC."""
    mat = np.full((len(AGE_BAND_NAMES), len(horizons)), np.nan, dtype=np.float64)
    for j, horizon in enumerate(horizons):
        cells = payload["by_horizon"][horizon]["cells"]
        for i, band in enumerate(AGE_BAND_NAMES):
            mat[i, j] = float(cells[band]["delta_auprc"])
    return mat


def overall_curve(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    labels = list(payload["horizons"])
    dtr, temporal, delta, n_win = [], [], [], []
    for label in labels:
        cell = payload["by_horizon"][label]["cells"]["_overall"]
        dtr.append(cell["auprc_dtr"])
        temporal.append(cell["auprc_temporal"])
        delta.append(cell["delta_auprc"])
        n_win.append(cell["n_windows"])
    return {
        "horizon": np.asarray(labels),
        "auprc_dtr": np.asarray(dtr, dtype=np.float64),
        "auprc_temporal": np.asarray(temporal, dtype=np.float64),
        "delta_auprc": np.asarray(delta, dtype=np.float64),
        "n_windows": np.asarray(n_win, dtype=np.int64),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0, help="0 = full test split")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    return get_metrics(
        recompute=args.recompute,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
