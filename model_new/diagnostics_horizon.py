#!/usr/bin/env python3
"""Horizon / future-visit ordering diagnostic for materialized shards (MEASURE).

Contract under audit (future-visit forecasting)::

    max(input_timestamps) < target_visit_start_time   # strict

A violation is ``target_time <= input_end_time``. Ties
(``target_time == input_end_time``) are reported separately from the strict-
negative tail.

All console / JSON output goes through :mod:`model_new.diagnostics` (D11).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from model_new import diagnostics as D
from model_new.data import TensorizedPretrainDataset, _sample_indices

__all__ = [
    "horizon_stats",
    "horizon_stats_from_gaps",
    "detect_shard_schema",
    "QUANTILES",
]

QUANTILES: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_shard_schema(split_dir: str | Path) -> str:
    """``'pretrain_flat' | 'finetune_flat' | 'legacy_object' | 'empty'``."""
    split_dir = Path(split_dir)
    paths = sorted(split_dir.glob("shard_*.npz"))
    if not paths:
        return "empty"
    z = np.load(paths[0], mmap_mode="r", allow_pickle=False)
    try:
        files = set(z.files)
    finally:
        getattr(z, "close", lambda: None)()
    if "visit_offsets" in files and "visit_starts" in files:
        return "pretrain_flat"
    if "offsets" in files and "label" in files:
        return "finetune_flat"
    return "legacy_object"


def horizon_stats_from_gaps(gaps_days: np.ndarray, *, n_examples: int | None = None,
                            split: str | None = None, path: str | None = None,
                            schema: str = "pretrain_flat") -> dict[str, Any]:
    """Aggregate horizon statistics from signed gaps ``target_time - input_end_time``."""
    g = np.asarray(gaps_days, dtype=np.float64)
    g = g[np.isfinite(g)]
    n = int(g.size)
    if n_examples is None:
        n_examples = n

    def _pct(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, 100.0 * q)) if a.size else float("nan")

    viol = g <= 0.0
    tie = g == 0.0
    neg = g < 0.0
    qdict: dict[str, float] = {}
    for q in QUANTILES:
        if q == 0.0:
            key = "min"
        elif q == 1.0:
            key = "max"
        else:
            key = f"p{int(round(100 * q))}"
        qdict[key] = _pct(g, q)

    return {
        "schema": schema,
        "path": path,
        "split": split,
        "n_examples": int(n_examples),
        "n_measured": n,
        "frac_violation_target_le_input_end": float(viol.mean()) if n else float("nan"),
        "frac_tie_target_eq_input_end": float(tie.mean()) if n else float("nan"),
        "frac_strict_negative": float(neg.mean()) if n else float("nan"),
        "n_violation": int(viol.sum()) if n else 0,
        "n_tie": int(tie.sum()) if n else 0,
        "n_strict_negative": int(neg.sum()) if n else 0,
        "horizon_days_quantiles": qdict,
        "negative_tail_days_quantiles": {
            k: _pct(g[neg], q) for k, q in (
                ("min", 0.0), ("p01", 0.01), ("p05", 0.05), ("p10", 0.10),
                ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("max", 1.0),
            )
        } if int(neg.sum()) else {k: float("nan") for k in (
            "min", "p01", "p05", "p10", "p25", "p50", "p75", "max")},
    }


def _gaps_pretrain_shard_walk(split_dir: Path, *, max_seq_len: int = 1024,
                              indices: np.ndarray | None = None) -> np.ndarray:
    """Signed gaps matching :meth:`TensorizedPretrainDataset.__getitem__`.

    ``input_end_time = max(timestamps in the time-cut, tail-truncated input window)``
    ``target_time    = min(timestamps of visit k+1)``
    """
    from model_new.data import select_forecast_input_indices, target_visit_start_time

    shard_paths = sorted(split_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.npz in {split_dir}")

    index: list[tuple[int, int, int]] = []
    for shard_id, path in enumerate(shard_paths):
        z = np.load(path, mmap_mode="r", allow_pickle=False)
        vo = np.asarray(z["visit_offsets"])
        eo = np.asarray(z["event_offsets"])
        vs = np.asarray(z["visit_starts"])
        ve = np.asarray(z["visit_ends"])
        ts_all = np.asarray(z["timestamps_days"])
        n = int(vo.shape[0]) - 1
        for pos in range(n):
            ev0, ev1 = int(eo[pos]), int(eo[pos + 1])
            v0, v1 = int(vo[pos]), int(vo[pos + 1])
            n_visits = v1 - v0
            if n_visits < 2 or ev1 <= ev0:
                continue
            ts = ts_all[ev0:ev1]
            for v in range(n_visits - 1):
                s_next = int(vs[v0 + v + 1])
                e_next = int(ve[v0 + v + 1])
                if e_next <= s_next:
                    continue
                t_tgt = target_visit_start_time(ts, s_next, e_next)
                if bool(np.any(ts < t_tgt)):
                    index.append((shard_id, pos, v))
        getattr(z, "close", lambda: None)()

    if indices is None:
        chosen = range(len(index))
    else:
        chosen = (int(i) for i in indices)

    by_shard: dict[int, list[tuple[int, int, int]]] = {}
    for i in chosen:
        shard_id, pos, visit_k = index[i]
        by_shard.setdefault(shard_id, []).append((i, pos, visit_k))

    gaps = np.empty(len(index) if indices is None else len(indices), dtype=np.float64)
    if indices is None:
        slot = {i: i for i in range(len(index))}
    else:
        slot = {int(i): j for j, i in enumerate(indices)}

    for shard_id, items in by_shard.items():
        z = np.load(shard_paths[shard_id], mmap_mode="r", allow_pickle=False)
        eo = np.asarray(z["event_offsets"])
        vo = np.asarray(z["visit_offsets"])
        vs = np.asarray(z["visit_starts"])
        ve = np.asarray(z["visit_ends"])
        ts_all = np.asarray(z["timestamps_days"])
        for orig_i, pos, visit_k in items:
            ev_start = int(eo[pos])
            ev_end = int(eo[pos + 1])
            vis_start = int(vo[pos])
            start_next = int(vs[vis_start + visit_k + 1])
            end_next = int(ve[vis_start + visit_k + 1])
            ts = ts_all[ev_start:ev_end]
            target_time = target_visit_start_time(ts, start_next, end_next)
            sel = select_forecast_input_indices(ts, target_time, max_seq_len)
            if sel.size == 0:
                gaps[slot[orig_i]] = float("nan")
                continue
            input_end = float(np.max(ts[sel].astype(np.float64)))
            gaps[slot[orig_i]] = float(target_time) - input_end
        getattr(z, "close", lambda: None)()
    return gaps


def horizon_stats(example_source: Any, *, split: str | None = None,
                  max_seq_len: int = 1024, n_samples: int | None = None,
                  seed: int = 0, path: str | None = None) -> dict[str, Any]:
    """Horizon violation stats for a built example set.

    ``example_source`` may be:

    * a :class:`~model_new.data.TensorizedPretrainDataset`
    * a split directory containing ``shard_*.npz`` (schema auto-detected)
    * an ndarray of precomputed signed gaps (days)

    ``n_samples=None`` measures the full split. A finite ``n_samples`` draws a
    seeded subset (minutes on a sample; re-runnable on the full split).
    """
    if isinstance(example_source, np.ndarray):
        return horizon_stats_from_gaps(
            example_source, split=split, path=path, schema="precomputed_gaps")

    if isinstance(example_source, TensorizedPretrainDataset):
        split_dir = example_source.tensorized_dir
        max_seq_len = int(example_source.max_seq_len)
        n_examples = len(example_source)
        schema = "pretrain_flat"
        path = path or str(split_dir)
        if n_samples is None or n_samples >= n_examples:
            gaps = _gaps_pretrain_shard_walk(split_dir, max_seq_len=max_seq_len)
        else:
            idxs = _sample_indices(n_examples, int(n_samples), seed)
            gaps = _gaps_pretrain_shard_walk(split_dir, max_seq_len=max_seq_len, indices=idxs)
        out = horizon_stats_from_gaps(gaps, n_examples=n_examples, split=split,
                                      path=path, schema=schema)
        out["n_patients"] = int(example_source.n_patients)
        out["max_seq_len"] = int(max_seq_len)
        out["sampled"] = bool(n_samples is not None and n_samples < n_examples)
        out["seed"] = int(seed) if out["sampled"] else None
        return out

    split_dir = Path(example_source)
    schema = detect_shard_schema(split_dir)
    path = path or str(split_dir)
    if schema == "empty":
        return {
            "schema": schema, "path": path, "split": split,
            "n_examples": 0, "n_measured": 0,
            "frac_violation_target_le_input_end": float("nan"),
            "frac_tie_target_eq_input_end": float("nan"),
            "applicable": False,
            "note": "no shard_*.npz",
        }
    if schema == "finetune_flat":
        # Classification shards have no next-visit target boundary in-schema.
        # Count rows; horizon contract is N/A.
        n = 0
        for p in sorted(split_dir.glob("shard_*.npz")):
            z = np.load(p, mmap_mode="r", allow_pickle=False)
            n += int(len(z["subject_id"]))
            getattr(z, "close", lambda: None)()
        return {
            "schema": schema, "path": path, "split": split,
            "n_examples": n, "n_measured": 0,
            "frac_violation_target_le_input_end": float("nan"),
            "frac_tie_target_eq_input_end": float("nan"),
            "applicable": False,
            "note": ("finetune classification schema has no visit-level target "
                     "timestamp; next-visit horizon check does not apply"),
        }
    if schema == "legacy_object":
        return {
            "schema": schema, "path": path, "split": split,
            "n_examples": 0, "n_measured": 0,
            "frac_violation_target_le_input_end": float("nan"),
            "frac_tie_target_eq_input_end": float("nan"),
            "applicable": False,
            "note": "legacy object-array shards; model_new.data requires flat visit schema",
        }

    # pretrain_flat via directory
    # Need a vocab only to construct the Dataset index; use a dummy if absent —
    # we walk shards directly, but Dataset.__init__ loads vocab. Prefer walking.
    if n_samples is None:
        gaps = _gaps_pretrain_shard_walk(split_dir, max_seq_len=max_seq_len)
        # count examples = len(gaps)
        out = horizon_stats_from_gaps(gaps, split=split, path=path, schema=schema)
        out["max_seq_len"] = int(max_seq_len)
        out["sampled"] = False
        out["seed"] = None
        out["applicable"] = True
        return out

    # Sampled: count valid examples the same way the Dataset index does.
    from model_new.data import target_visit_start_time as _tvst
    n_examples = 0
    for p in sorted(split_dir.glob("shard_*.npz")):
        z = np.load(p, mmap_mode="r", allow_pickle=False)
        vo = np.asarray(z["visit_offsets"])
        eo = np.asarray(z["event_offsets"])
        vs = np.asarray(z["visit_starts"])
        ve = np.asarray(z["visit_ends"])
        ts_all = np.asarray(z["timestamps_days"])
        for pos in range(int(vo.shape[0]) - 1):
            ev0, ev1 = int(eo[pos]), int(eo[pos + 1])
            v0, v1 = int(vo[pos]), int(vo[pos + 1])
            if v1 - v0 < 2 or ev1 <= ev0:
                continue
            ts = ts_all[ev0:ev1]
            for v in range(v1 - v0 - 1):
                s_next = int(vs[v0 + v + 1])
                e_next = int(ve[v0 + v + 1])
                if e_next <= s_next:
                    continue
                t_tgt = _tvst(ts, s_next, e_next)
                if bool(np.any(ts < t_tgt)):
                    n_examples += 1
        getattr(z, "close", lambda: None)()
    idxs = _sample_indices(n_examples, int(n_samples), seed)
    gaps = _gaps_pretrain_shard_walk(split_dir, max_seq_len=max_seq_len, indices=idxs)
    out = horizon_stats_from_gaps(gaps, n_examples=n_examples, split=split,
                                  path=path, schema=schema)
    out["max_seq_len"] = int(max_seq_len)
    out["sampled"] = True
    out["seed"] = int(seed)
    out["applicable"] = True
    return out


def _print_result(title: str, r: dict[str, Any]) -> None:
    lines = [
        f"path={r.get('path')}",
        f"schema={r.get('schema')}  split={r.get('split')}  "
        f"n_examples={r.get('n_examples')}  n_measured={r.get('n_measured')}",
    ]
    if r.get("applicable") is False:
        lines.append(f"applicable=False  note={r.get('note')}")
        D.print_block(title, lines)
        return
    lines.extend([
        f"frac_violation (target<=input_end)={r['frac_violation_target_le_input_end']:.6f}  "
        f"n={r['n_violation']}",
        f"frac_tie       (target==input_end)={r['frac_tie_target_eq_input_end']:.6f}  "
        f"n={r['n_tie']}",
        f"frac_strict_neg (target<input_end)={r['frac_strict_negative']:.6f}  "
        f"n={r['n_strict_negative']}",
        f"sampled={r.get('sampled')}  seed={r.get('seed')}  max_seq_len={r.get('max_seq_len')}",
        "horizon_days quantiles:",
    ])
    hq = r.get("horizon_days_quantiles") or {}
    lines.append("  " + "  ".join(f"{k}={v:.4g}" for k, v in hq.items()))
    nq = r.get("negative_tail_days_quantiles") or {}
    lines.append("negative_tail_days quantiles:")
    lines.append("  " + "  ".join(f"{k}={v:.4g}" for k, v in nq.items()))
    D.print_block(title, lines)


def _default_targets() -> list[tuple[str, Path, str]]:
    """(label, split_dir, split_name) for Phase-0 sweep."""
    root = REPO_ROOT
    out: list[tuple[str, Path, str]] = []
    mimic = root / "data/processed/tensorized_flat"
    for split in ("train", "val", "test"):
        out.append((f"MIMIC-pretrain/{split}", mimic / split, split))
    pic_root = root / "data/tensorized/pic"
    if pic_root.is_dir():
        for task_dir in sorted(pic_root.iterdir()):
            if not task_dir.is_dir():
                continue
            for split in ("train", "val", "test"):
                d = task_dir / split
                if d.is_dir():
                    out.append((f"PIC/{task_dir.name}/{split}", d, split))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Horizon leakage diagnostic (MEASURE).")
    p.add_argument("--split_dir", type=Path, default=None,
                   help="Single split directory. Default: Phase-0 full sweep.")
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--n_samples", type=int, default=None,
                   help="If set, sample this many examples (seeded). Default: full split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=Path, default=None)
    args = p.parse_args(argv)

    results: list[dict[str, Any]] = []
    if args.split_dir is not None:
        targets = [("custom", args.split_dir, args.split)]
    else:
        targets = _default_targets()

    for label, split_dir, split in targets:
        r = horizon_stats(split_dir, split=split, max_seq_len=args.max_seq_len,
                          n_samples=args.n_samples, seed=args.seed)
        r["label"] = label
        results.append(r)
        _print_result(f"horizon  {label}", r)

    # Summary table
    hdr = (f"{'label':<42} {'n':>8} {'viol':>8} {'tie':>8} "
           f"{'p50_h':>10} {'p10_h':>10} {'neg_p50':>10}")
    rows = [hdr, "-" * len(hdr)]
    for r in results:
        if r.get("applicable") is False:
            rows.append(f"{r['label']:<42} {r.get('n_examples', 0):>8} {'n/a':>8} "
                        f"{'n/a':>8} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
            continue
        hq = r.get("horizon_days_quantiles") or {}
        nq = r.get("negative_tail_days_quantiles") or {}
        rows.append(
            f"{r['label']:<42} {r.get('n_measured', 0):>8} "
            f"{r['frac_violation_target_le_input_end']:>8.4f} "
            f"{r['frac_tie_target_eq_input_end']:>8.4f} "
            f"{hq.get('p50', float('nan')):>10.3f} "
            f"{hq.get('p10', float('nan')):>10.3f} "
            f"{nq.get('p50', float('nan')):>10.3f}"
        )
    D.print_block("horizon summary", rows)

    if args.out_json is not None:
        D.write_json(args.out_json, {"results": results})
        D.print_kv("wrote", {"out_json": str(args.out_json)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
