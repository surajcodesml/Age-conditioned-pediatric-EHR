#!/usr/bin/env python3
"""Pretraining dataset + collate.

Batch contract:

    code_indices     [B, L]      int64     PAD=0, UNK=1, real=v+2
    timestamps_days  [B, L]      float64   days from first event (padded with 0.0)
    lengths          [B]         int64     number of valid events per row
    attention_mask   [B, L]      bool
    age_years        [B, L]      float32   age_at_event_days / 365.25
    demographics     [B, L, D]   float32   (age_years, sex, race...)
    target_codes     [B, |V|]    float32

``tau [B, L, L]`` and ``tau_to_now [B, L]`` are **not** in the batch. At L=1024 ``tau`` is
4.19 MB/sample, so shipping it from the worker processes across the PCIe bus (doubled by
``pin_memory``) made host RAM -- not the 32 GB of VRAM -- the binding constraint on batch
size, and put the O(L^2) arithmetic in the workers. The model computes both on the GPU from
``timestamps_days`` via :func:`tau_from_timestamps`; the ``/7`` and ``log1p`` convention
still lives in exactly one function (:func:`lag_to_tau`).

D3 -- ``age_years`` is its own tensor, so the kernel input and the demographic feature can
be varied independently. Age **stays** in the demographic vector: that is the route age
already has and the one DKM has to improve on. Every arm receives the identical
demographic tensor.

**Race.** The legacy pipeline stores race as a scalar float, imposing an arbitrary ordinal
on a categorical (WHITE=0 < BLACK=1 < ASIAN=2 ...). Cardinality here is 7, so the default
is one-hot and ``demo_dim = 2 + n_race = 9``. This is identical across arms and recorded in
``config.json``; ``--race_encoding scalar`` reproduces the legacy layout with
``demo_dim = 3``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "lag_to_tau", "spans_to_tau", "pairwise_tau", "tau_to_now_from_timestamps",
    "tau_from_timestamps",
    "encode_race", "N_RACE", "RACE_LABELS", "demo_layout",
    "target_visit_start_time", "select_forecast_input_indices",
    "TensorizedPretrainDataset", "pretrain_collate", "make_collate",
    "corpus_stats", "corpus_stats_cached", "CorpusStats", "sample_empirical_taus",
    "load_vocab",
    "dataloader_worker_init",
    "DAYS_PER_YEAR", "WEEK_DAYS",
]

WEEK_DAYS = 7.0
DAYS_PER_YEAR = 365.25

RACE_LABELS = ("WHITE", "BLACK", "ASIAN", "HISPANIC", "AMERICAN_INDIAN", "OTHER", "UNKNOWN")
N_RACE = len(RACE_LABELS)


# --------------------------------------------------------------------------- #
# The one and only lag convention.                                            #
# --------------------------------------------------------------------------- #
def lag_to_tau(delta_t_days: torch.Tensor) -> torch.Tensor:
    """``tau = log1p(|dt_days| / 7)``. The single definition; nothing else recomputes it.

    Dtype-preserving, so the callers below can drive it in float64 (D-precision, see
    :func:`pairwise_tau`).
    """
    return torch.log1p(delta_t_days.abs() / WEEK_DAYS)


def spans_to_tau(span_days: np.ndarray) -> np.ndarray:
    """The numpy-side twin of :func:`lag_to_tau`, for corpus statistics over window spans.
    Shares the ``/7`` and ``log1p`` constants so the two can never drift apart."""
    return np.log1p(np.abs(np.asarray(span_days, dtype=np.float64)) / WEEK_DAYS)


def pairwise_tau(timestamps_days: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, L] -> [B, L, L]``, zeroed at padded pairs.

    **Differencing happens in float64.** At the observed ``span_days_max`` of ~5800 the
    float32 ulp is ~40-60 s, and ``t_i - t_j`` for two large nearby timestamps loses most of
    its significant digits to cancellation -- exactly where ``tau`` is most sensitive, since
    ``d tau / d(dt)`` is largest at small ``dt``. The result is cast back to float32 only
    after ``log1p``, so the tensor the encoder consumes stays cheap.

    This removes *arithmetic* error. It cannot remove *storage* error: the shards hold
    ``timestamps_days`` as float32, so at large ``t`` the recorded times are already
    quantised regardless of what is done downstream. See ``timestamp_resolution`` in
    :class:`CorpusStats`, which reports the limit by timestamp magnitude.

    Called from :meth:`model.DKMModel.forward` on the GPU batch, not from the collate.
    """
    t = timestamps_days.double()
    tau = lag_to_tau(t.unsqueeze(2) - t.unsqueeze(1))
    pair = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    return (tau * pair.to(tau.dtype)).to(torch.float32)


def tau_to_now_from_timestamps(timestamps_days: torch.Tensor, attention_mask: torch.Tensor,
                               lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Lag from every event to the last valid event. ``[B, L] -> [B, L]``. float64 internally.

    ``lengths`` (from the collate) is used when supplied; otherwise it is derived from the
    mask. The two agree by construction -- the collate asserts it -- so this is only a
    convenience for callers that already hold one or the other.
    """
    if lengths is None:
        lengths = attention_mask.sum(dim=1).long()
    if bool((lengths == 0).any()):
        raise ValueError("zero-length sequence in batch: tau_to_now is undefined")
    rows = torch.arange(timestamps_days.shape[0], device=timestamps_days.device)
    t = timestamps_days.double()
    t_last = t[rows, lengths - 1].unsqueeze(1)
    return (lag_to_tau(t_last - t) * attention_mask.to(t.dtype)).to(torch.float32)


def tau_from_timestamps(timestamps_days: torch.Tensor, attention_mask: torch.Tensor,
                        lengths: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(tau [B,L,L], tau_to_now [B,L])`` from timestamps. The model-facing entry point.

    Both go through :func:`lag_to_tau`, so the ``/7`` and ``log1p`` convention lives in one
    torch function regardless of where it runs. The transient ``[B, L, L]`` float64
    intermediate is freed after the float32 cast (~537 MB at B=64, L=1024).
    """
    return (pairwise_tau(timestamps_days, attention_mask),
            tau_to_now_from_timestamps(timestamps_days, attention_mask, lengths))


# --------------------------------------------------------------------------- #
# Demographics                                                                #
# --------------------------------------------------------------------------- #
def encode_race(race_val: Any) -> int:
    """-> index into RACE_LABELS. Self-contained; matches the tensorizer's encoding."""
    if race_val is None:
        return 6
    if isinstance(race_val, float) and np.isnan(race_val):
        return 6
    s = str(race_val).strip().upper()
    if not s or s == "NAN":
        return 6
    if s in {"UNKNOWN", "UNABLE TO OBTAIN", "PREFER NOT TO SAY", "N/A", "DECLINED"}:
        return 6
    if s.startswith("WHITE"):
        return 0
    if s.startswith("BLACK"):
        return 1
    if s.startswith("ASIAN"):
        return 2
    if s.startswith("HISPANIC"):
        return 3
    if s.startswith("AMERICAN INDIAN") or s.startswith("ALASKA NATIVE"):
        return 4
    return 5


def demo_layout(race_encoding: str = "one_hot") -> tuple[int, tuple[str, ...]]:
    """-> (demo_dim, channel names). Identical across arms; recorded in config.json."""
    if race_encoding == "one_hot":
        return 2 + N_RACE, ("age_years", "sex") + tuple(f"race_{r}" for r in RACE_LABELS)
    if race_encoding == "scalar":
        return 3, ("age_years", "sex", "race")
    raise ValueError(f"race_encoding must be 'one_hot' or 'scalar', got {race_encoding!r}")


# --------------------------------------------------------------------------- #
# Future-visit forecasting window (INV-HORIZON)                               #
# --------------------------------------------------------------------------- #
def target_visit_start_time(timestamps_days: np.ndarray, start: int, end: int) -> float:
    """``start_time(V) = min(timestamps in V)``. Empty visits are a hard error."""
    block = np.asarray(timestamps_days[start:end], dtype=np.float64)
    if block.size == 0:
        raise ValueError(f"empty target visit [{start}:{end})")
    return float(block.min())


def select_forecast_input_indices(timestamps_days: np.ndarray, target_time: float,
                                  max_seq_len: int) -> np.ndarray:
    """Indices of events with ``timestamp < target_time``, time-ordered, tail-truncated.

    Contract (future-visit forecasting, INV-HORIZON):

    * input  = every event with ``t < start_time(V_{m+1})`` (strict; ties go to the target)
    * truncation drops the **oldest** events (by time), keeping the window adjacent to the
      boundary, and never pulls events from across it
    """
    ts = np.asarray(timestamps_days, dtype=np.float64)
    pre = np.flatnonzero(ts < float(target_time))
    if pre.size == 0:
        return pre
    # Chronological order: primary key timestamp, secondary original index (stable).
    order = np.lexsort((pre.astype(np.int64, copy=False), ts[pre]))
    pre = pre[order]
    m = int(max_seq_len)
    if m > 0 and pre.size > m:
        pre = pre[-m:]
    return pre


def _horizon_assert_enabled(flag: bool | None) -> bool:
    """Constructor flag wins when explicit; otherwise env ``MODEL_NEW_CHECK_HORIZON`` (default on)."""
    if flag is not None:
        return bool(flag)
    v = os.environ.get("MODEL_NEW_CHECK_HORIZON", "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
def load_vocab(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def dataloader_worker_init(_worker_id: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


class TensorizedPretrainDataset(Dataset):
    """Future-visit forecasting samples from flat mmap-able shards.

    One sample per (patient, target visit ``V_{m+1}``) with at least one prior event:

    * ``input``  = every event with ``timestamp < start_time(V_{m+1})`` (strict)
    * ``target`` = the code set of ``V_{m+1}``

    Visits remain the hadm-derived blocks stored in the shard; the **window** is a time cut
    at the target visit's start, not the index end of the previous visit. That is why
    padding-only masking is not leakage (D4 / INV-HORIZON): no input event is at or after
    the target boundary. Count truncation keeps the newest ``max_seq_len`` pre-boundary
    events.
    """

    def __init__(self, tensorized_dir: str | Path, code_vocab_path: str | Path,
                 max_seq_len: int = 1024, shard_cache_size: int = 4,
                 assert_horizon: bool | None = None) -> None:
        self.tensorized_dir = Path(tensorized_dir)
        self.max_seq_len = int(max_seq_len)
        self.shard_cache_size = int(shard_cache_size)
        self.assert_horizon = _horizon_assert_enabled(assert_horizon)
        self.code_vocab = load_vocab(code_vocab_path)
        self.num_codes = len(self.code_vocab)
        self.unk_vocab_index = self.num_codes

        self._shard_paths = sorted(self.tensorized_dir.glob("shard_*.npz"))
        if not self._shard_paths:
            raise FileNotFoundError(f"no shard_*.npz in {self.tensorized_dir}")
        self._index: list[tuple[int, int, int]] = []
        self.n_patients = 0
        for shard_id, shard_path in enumerate(self._shard_paths):
            npz = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            if "visit_offsets" not in npz.files:
                raise RuntimeError(f"{shard_path} is not the flat schema; re-tensorize.")
            visit_offsets = np.asarray(npz["visit_offsets"])
            event_offsets = np.asarray(npz["event_offsets"])
            visit_starts = np.asarray(npz["visit_starts"])
            visit_ends = np.asarray(npz["visit_ends"])
            timestamps = np.asarray(npz["timestamps_days"])
            n = int(visit_offsets.shape[0]) - 1
            self.n_patients += n
            for pos in range(n):
                ev0, ev1 = int(event_offsets[pos]), int(event_offsets[pos + 1])
                v0, v1 = int(visit_offsets[pos]), int(visit_offsets[pos + 1])
                n_visits = v1 - v0
                if n_visits < 2 or ev1 <= ev0:
                    continue
                ts = timestamps[ev0:ev1]
                for v in range(n_visits - 1):
                    s_next = int(visit_starts[v0 + v + 1])
                    e_next = int(visit_ends[v0 + v + 1])
                    if e_next <= s_next:
                        continue
                    t_tgt = target_visit_start_time(ts, s_next, e_next)
                    # Valid forecasting example iff there is strict-past context.
                    if bool(np.any(ts < t_tgt)):
                        self._index.append((shard_id, pos, v))
            npz.close()
        self._shard_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_id: int) -> dict[str, Any]:
        if shard_id in self._shard_cache:
            d = self._shard_cache.pop(shard_id)
            self._shard_cache[shard_id] = d
            return d
        if len(self._shard_cache) >= self.shard_cache_size:
            _, old = self._shard_cache.popitem(last=False)
            try:
                old["_npz"].close()
            except Exception:
                pass
        npz = np.load(self._shard_paths[shard_id], mmap_mode="r", allow_pickle=False)
        d = {"_npz": npz, "unk_vocab_index": int(np.asarray(npz["unk_vocab_index"]).reshape(-1)[0])}
        for key in ("event_offsets", "code_indices", "timestamps_days", "age_days",
                    "visit_offsets", "visit_starts", "visit_ends", "sex", "race"):
            d[key] = npz[key]
        self._shard_cache[shard_id] = d
        return d

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_id, pos, visit_k = self._index[idx]
        s = self._load_shard(shard_id)
        ev_start = int(s["event_offsets"][pos])
        ev_end = int(s["event_offsets"][pos + 1])
        vis_start = int(s["visit_offsets"][pos])

        start_next = int(s["visit_starts"][vis_start + visit_k + 1])
        end_next = int(s["visit_ends"][vis_start + visit_k + 1])

        codes_all = np.asarray(s["code_indices"][ev_start:ev_end], dtype=np.int64)
        ts_all = np.asarray(s["timestamps_days"][ev_start:ev_end], dtype=np.float32)
        ages_all = np.asarray(s["age_days"][ev_start:ev_end], dtype=np.float32)

        target_time = target_visit_start_time(ts_all, start_next, end_next)
        sel = select_forecast_input_indices(ts_all, target_time, self.max_seq_len)
        if sel.size == 0:
            raise RuntimeError(
                f"empty forecast input at idx={idx} shard={shard_id} pos={pos} "
                f"visit_k={visit_k}: index construction should have excluded this row")
        codes = codes_all[sel]
        ts = ts_all[sel]
        ages = ages_all[sel]

        if self.assert_horizon:
            if not (float(np.max(ts.astype(np.float64))) < float(target_time)):
                raise AssertionError(
                    f"INV-HORIZON violated: max(input_ts)={float(np.max(ts))} "
                    f">= target_time={float(target_time)} "
                    f"(idx={idx} shard={shard_id} pos={pos} visit_k={visit_k})")

        unk = int(s["unk_vocab_index"])
        nxt = codes_all[start_next:end_next]
        target = np.zeros(self.num_codes, dtype=np.float32)
        valid = nxt[nxt != unk]
        if valid.size:
            target[np.unique(valid)] = 1.0

        return {
            "code_indices": codes,
            "timestamps_days": ts,
            "age_days": ages,
            "sex": int(s["sex"][pos]),
            "race": int(s["race"][pos]),
            "unk_vocab_index": unk,
            "target_codes": target,
            "target_time": float(target_time),
        }

    def __del__(self) -> None:
        for shard in getattr(self, "_shard_cache", {}).values():
            try:
                shard["_npz"].close()
            except Exception:
                pass
        if hasattr(self, "_shard_cache"):
            self._shard_cache.clear()


# --------------------------------------------------------------------------- #
# Collate                                                                     #
# --------------------------------------------------------------------------- #
def _pad_common(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Padding + the tensors every task shares. Used by data.py and data_finetune.py."""
    if not batch:
        raise ValueError("empty batch")
    bsz = len(batch)
    unk_id = int(batch[0]["unk_vocab_index"])
    max_len = max(int(item["code_indices"].shape[0]) for item in batch)
    if max_len == 0:
        raise ValueError("every sequence in the batch is empty")

    code_np = np.zeros((bsz, max_len), dtype=np.int64)
    ts_np = np.zeros((bsz, max_len), dtype=np.float64)   # f64: lag arithmetic runs in f64
    age_np = np.zeros((bsz, max_len), dtype=np.float32)
    mask_np = np.zeros((bsz, max_len), dtype=bool)
    lengths_np = np.zeros((bsz,), dtype=np.int64)
    for b, item in enumerate(batch):
        n = int(item["code_indices"].shape[0])
        if n == 0:
            raise ValueError(f"zero-length sequence at batch row {b}")
        code_np[b, :n] = item["code_indices"]
        ts_np[b, :n] = item["timestamps_days"]
        age_np[b, :n] = item["age_days"]
        mask_np[b, :n] = True
        lengths_np[b] = n

    code_indices = torch.from_numpy(code_np)
    attention_mask = torch.from_numpy(mask_np)
    timestamps_days = torch.from_numpy(ts_np)           # [B, L] float64
    # PAD=0, UNK=1, real=v+2.
    bge_codes = torch.where(
        attention_mask,
        torch.where(code_indices == unk_id, torch.ones((), dtype=torch.long), code_indices + 2),
        torch.zeros((), dtype=torch.long),
    )
    age_years = torch.from_numpy(age_np / DAYS_PER_YEAR) * attention_mask.float()

    sex = np.array([item["sex"] for item in batch], dtype=np.float32)
    race = np.array([item["race"] for item in batch], dtype=np.int64)
    # tau [B, L, L] and tau_to_now [B, L] are NO LONGER emitted here. At L=1024, tau is
    # 4.19 MB/sample and crossing the worker boundary + PCIe + pin_memory made host RAM the
    # binding constraint on batch size, and put the O(L^2) arithmetic in the workers. The
    # model computes both on the GPU from timestamps_days (8 KB/sample), a factor-L reduction
    # in host traffic. See model.DKMModel.forward and tests/test_tau_equivalence.py.
    return {
        "code_indices": bge_codes,
        "timestamps_days": timestamps_days,
        "lengths": torch.from_numpy(lengths_np),
        "attention_mask": attention_mask,
        "age_years": age_years,
        "_sex": sex,
        "_race": race,
    }


def _build_demographics(common: dict[str, Any], race_encoding: str) -> torch.Tensor:
    age_years = common["age_years"]
    mask = common["attention_mask"]
    bsz, max_len = age_years.shape
    demo_dim, _ = demo_layout(race_encoding)
    demo = np.zeros((bsz, max_len, demo_dim), dtype=np.float32)
    demo[:, :, 0] = age_years.numpy()
    demo[:, :, 1] = common["_sex"][:, None]
    if race_encoding == "one_hot":
        for b, r in enumerate(common["_race"]):
            demo[b, :, 2 + int(r)] = 1.0
    else:
        demo[:, :, 2] = common["_race"][:, None].astype(np.float32)
    return torch.from_numpy(demo) * mask.unsqueeze(-1).float()


def pretrain_collate(batch: list[dict[str, Any]], *, race_encoding: str = "one_hot",
                     assert_horizon: bool | None = None) -> dict:
    if _horizon_assert_enabled(assert_horizon):
        for b, item in enumerate(batch):
            t_tgt = item.get("target_time", None)
            if t_tgt is None:
                continue
            ts = np.asarray(item["timestamps_days"], dtype=np.float64)
            if ts.size and not (float(ts.max()) < float(t_tgt)):
                raise AssertionError(
                    f"INV-HORIZON violated in collate row {b}: "
                    f"max(input_ts)={float(ts.max())} >= target_time={float(t_tgt)}")
    common = _pad_common(batch)
    out = {k: v for k, v in common.items() if not k.startswith("_")}
    out["demographics"] = _build_demographics(common, race_encoding)
    out["target_codes"] = torch.from_numpy(
        np.stack([item["target_codes"] for item in batch], axis=0))
    return out


def make_collate(race_encoding: str = "one_hot", assert_horizon: bool | None = None):
    """Picklable collate for spawned DataLoader workers."""
    from functools import partial
    return partial(pretrain_collate, race_encoding=race_encoding,
                   assert_horizon=assert_horizon)


# --------------------------------------------------------------------------- #
# tau_max and corpus MEASURE quantities                                        #
# --------------------------------------------------------------------------- #
def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def _f32_ceil(x: float) -> float:
    """Smallest float32 that is >= x. ``tau_max`` is stored as a float32 buffer while the
    corpus maximum is computed in float64; rounding the buffer *down* would make the very
    window that defined the maximum clamp. One ulp of headroom absorbs both that and the
    float64->float32 cast of ``tau`` itself."""
    v = np.float32(x)
    while float(v) < x:
        v = np.nextafter(v, np.float32(np.inf))
    return float(np.nextafter(v, np.float32(np.inf)))


@dataclass(frozen=True)
class CorpusStats:
    """The single source of truth for corpus statistics. Nothing else computes these.

    Provenance is explicit per field group, because the two groups cannot be gathered the
    same way:

    ``exact_*``   -- one pass over the **full** split. Cheap because every quantity is
                     per-event or per-window: reading each patient's timestamp and age
                     arrays once is O(events), and window spans are O(visits).
    ``sampled_*`` -- pairwise quantities, which are O(L^2) per window. On the train split
                     that is ~6.5e10 pairs, so these come from a seeded sample whose size is
                     recorded on the object. There is no way to make them exact at this
                     scale, and pretending otherwise would be worse than saying so.

    ``tau_max`` is in the exact group. It is frozen into the checkpoint and every learned
    coefficient is defined relative to it, so a sampled maximum that the full split exceeds
    would mean silent clamping for the whole run with no way to correct it afterwards.
    """

    split: str
    n_examples: int
    n_patients: int
    n_events: int

    # --- exact, full split -------------------------------------------------- #
    tau_max: float
    tau_max_source: str
    span_days_max_tau: float   # the float64 tau of the widest window; tau_max is its f32 ceil
    span_days_max: float
    span_tau_quantiles: dict
    event_age_min: float
    event_age_max: float
    event_age_median: float
    event_age_mean: float
    event_age_sd: float
    event_age_band_counts: dict
    event_age_histogram: dict
    event_integer_age_fraction: float
    event_age_ge_89_fraction: float
    event_age_under_18_count: int
    event_age_under_18_fraction: float
    example_age_band_counts: dict
    example_age_min: float
    example_age_median: float
    seq_len_quantiles: dict
    seq_len_max: int

    # --- seeded sample ------------------------------------------------------- #
    sample_seed: int
    sample_n_windows: int
    sample_n_pairs: int
    dt_zero_fraction: float
    integer_dt_fraction: float
    timestamp_resolution: dict
    tau_quantiles: dict
    spread_padding_only: dict
    spread_causal: dict
    frac_rows_spread_below_0p1: float
    causal_frac_rows_spread_below_0p1: float

    def to_json(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "CorpusStats":
        from dataclasses import fields
        return cls(**{f.name: d[f.name] for f in fields(cls)})


def _quantile_dict(x: np.ndarray, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    if x.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.percentile(x, 100.0 * q)) for q in qs}


def _band_counts(ages: np.ndarray) -> dict:
    """Band edges are the diagnostics bands, imported so metrics and corpus stats can never
    disagree about what "1-5" means."""
    from model_new.diagnostics import AGE_BANDS
    return {name: int(((ages >= lo) & (ages < hi)).sum()) for name, lo, hi in AGE_BANDS}


def corpus_stats(dataset: TensorizedPretrainDataset, *, split: str = "train",
                 sample_windows: int = 4000, max_pairs_per_window: int = 200_000,
                 seed: int = 0) -> CorpusStats:
    """Compute every corpus statistic in one place. See :class:`CorpusStats` for provenance.

    The exact pass walks the shards directly rather than going through ``__getitem__``, so
    each patient's arrays are touched once instead of once per window.
    """
    # ---- exact pass over the full split ------------------------------------ #
    span_days: list[np.ndarray] = []
    example_last_age: list[np.ndarray] = []
    seq_lens: list[np.ndarray] = []
    age_hist_edges = np.array([0, 1, 6, 12, 18, 40, 65, 200], dtype=np.float64)
    age_hist = np.zeros(len(age_hist_edges) - 1, dtype=np.int64)
    age_band_counts = {name: 0 for name in _band_counts(np.zeros(0))}
    n_events = n_int_age = n_ge89 = n_under18 = 0
    age_min, age_max = np.inf, -np.inf
    age_sum = age_sumsq = 0.0
    age_sample_for_median: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for path in dataset._shard_paths:
        z = np.load(path, mmap_mode="r", allow_pickle=False)
        eo = np.asarray(z["event_offsets"])
        vo = np.asarray(z["visit_offsets"])
        vs = np.asarray(z["visit_starts"])
        ve = np.asarray(z["visit_ends"])
        ts_all = np.asarray(z["timestamps_days"])
        age_all = np.asarray(z["age_days"])
        n_pat = vo.shape[0] - 1

        for pos in range(n_pat):
            ev0, ev1 = int(eo[pos]), int(eo[pos + 1])
            if ev1 <= ev0:
                continue
            ts = ts_all[ev0:ev1].astype(np.float64)
            ages = age_all[ev0:ev1].astype(np.float64) / DAYS_PER_YEAR

            # per-event age statistics, each event counted exactly once
            n_events += ages.size
            age_min = min(age_min, float(ages.min()))
            age_max = max(age_max, float(ages.max()))
            age_sum += float(ages.sum())
            age_sumsq += float((ages ** 2).sum())
            n_int_age += int(np.isclose(ages, np.round(ages), atol=1e-6).sum())
            n_ge89 += int((ages >= 89.0).sum())
            n_under18 += int((ages < 18.0).sum())
            age_hist += np.histogram(ages, bins=age_hist_edges)[0]
            for name, c in _band_counts(ages).items():
                age_band_counts[name] += c
            if ages.size:
                k = min(ages.size, 8)
                age_sample_for_median.append(rng.choice(ages, k, replace=False))

            # per-window span and last-event age, exactly as __getitem__ builds them
            # (time cut at target visit start, then drop oldest to max_seq_len).
            v0, v1 = int(vo[pos]), int(vo[pos + 1])
            n_visits = v1 - v0
            if n_visits < 2:
                continue
            win_spans: list[float] = []
            win_last_age: list[float] = []
            win_lens: list[int] = []
            for k in range(n_visits - 1):
                s_next = int(vs[v0 + k + 1])
                e_next = int(ve[v0 + k + 1])
                if e_next <= s_next:
                    continue
                t_tgt = target_visit_start_time(ts, s_next, e_next)
                sel = select_forecast_input_indices(ts, t_tgt, dataset.max_seq_len)
                if sel.size == 0:
                    continue
                win_ts = ts[sel]
                win_spans.append(float(win_ts[-1] - win_ts[0]))
                win_last_age.append(float(ages[sel[-1]]))
                win_lens.append(int(sel.size))
            if not win_spans:
                continue
            span_days.append(np.asarray(win_spans, dtype=np.float64))
            example_last_age.append(np.asarray(win_last_age, dtype=np.float64))
            seq_lens.append(np.asarray(win_lens, dtype=np.int64))
        getattr(z, "close", lambda: None)()   # real shards are NpzFile; test mocks are dicts

    spans = np.concatenate(span_days) if span_days else np.zeros(1)
    last_ages = np.concatenate(example_last_age) if example_last_age else np.zeros(1)
    lens = np.concatenate(seq_lens) if seq_lens else np.zeros(1, dtype=np.int64)
    span_taus = spans_to_tau(spans)
    tau_max_exact = float(span_taus.max())
    med_pool = np.concatenate(age_sample_for_median) if age_sample_for_median else np.zeros(1)

    # ---- seeded sample for the O(L^2) quantities ---------------------------- #
    idxs = _sample_indices(len(dataset), sample_windows, seed)
    n_pairs = n_zero = n_dt = n_int_dt = 0
    taus_s: list[np.ndarray] = []
    spread_pad: list[np.ndarray] = []
    spread_cau: list[np.ndarray] = []
    res_by_mag: dict[str, float] = {}
    for j in idxs:
        ts = dataset[int(j)]["timestamps_days"].astype(np.float64)
        if ts.size < 2:
            continue
        d = np.abs(ts[:, None] - ts[None, :])
        iu = np.triu_indices(ts.size, k=1)
        dv = d[iu]
        n_pairs += dv.size
        n_zero += int((dv == 0).sum())
        pos = dv[dv > 0]
        if pos.size:
            n_dt += pos.size
            n_int_dt += int(np.isclose(pos, np.round(pos), atol=1e-6).sum())
            # resolution limit depends on timestamp MAGNITUDE: float32 storage quantises
            # large t far more coarsely than small t, so one global minimum is misleading.
            mag = max(float(ts.max()), 1e-9)
            bucket = "t<10d" if mag < 10 else ("t<100d" if mag < 100 else
                                               ("t<1000d" if mag < 1000 else "t>=1000d"))
            res_by_mag[bucket] = min(res_by_mag.get(bucket, np.inf), float(pos.min()))
        tau_full = spans_to_tau(dv)
        if tau_full.size > max_pairs_per_window:
            tau_full = rng.choice(tau_full, max_pairs_per_window, replace=False)
        taus_s.append(tau_full)
        tau_mat = spans_to_tau(d)
        spread_pad.append(tau_mat.max(axis=1) - tau_mat.min(axis=1))
        tril = np.tril(np.ones((ts.size, ts.size), dtype=bool))
        big = np.where(tril, tau_mat, -np.inf).max(axis=1)
        small = np.where(tril, tau_mat, np.inf).min(axis=1)
        spread_cau.append(big - small)

    tau_s = np.concatenate(taus_s) if taus_s else np.zeros(0)
    sp_pad = np.concatenate(spread_pad) if spread_pad else np.zeros(0)
    sp_cau = np.concatenate(spread_cau) if spread_cau else np.zeros(0)

    return CorpusStats(
        split=split,
        n_examples=len(dataset),
        n_patients=dataset.n_patients,
        n_events=int(n_events),
        tau_max=_f32_ceil(tau_max_exact),
        tau_max_source=(f"exact max over full {split} split, N={spans.size} windows, "
                        f"of log1p(span_days/7); stored with 1 ulp of float32 headroom "
                        f"(float64 value {tau_max_exact!r})"),
        span_days_max_tau=float(tau_max_exact),
        span_days_max=float(spans.max()),
        span_tau_quantiles=_quantile_dict(span_taus),
        event_age_min=float(age_min), event_age_max=float(age_max),
        event_age_median=float(np.median(med_pool)),
        event_age_mean=float(age_sum / max(1, n_events)),
        event_age_sd=float(np.sqrt(max(0.0, age_sumsq / max(1, n_events)
                                       - (age_sum / max(1, n_events)) ** 2))),
        event_age_band_counts=age_band_counts,
        event_age_histogram={"edges": age_hist_edges.tolist(),
                             "counts": age_hist.tolist(),
                             "fractions": (age_hist / max(1, n_events)).tolist()},
        event_integer_age_fraction=n_int_age / max(1, n_events),
        event_age_ge_89_fraction=n_ge89 / max(1, n_events),
        event_age_under_18_count=int(n_under18),
        event_age_under_18_fraction=n_under18 / max(1, n_events),
        example_age_band_counts=_band_counts(last_ages),
        example_age_min=float(last_ages.min()),
        example_age_median=float(np.median(last_ages)),
        seq_len_quantiles=_quantile_dict(lens.astype(np.float64)),
        seq_len_max=int(lens.max()),
        sample_seed=int(seed),
        sample_n_windows=int(len(idxs)),
        sample_n_pairs=int(n_pairs),
        dt_zero_fraction=(n_zero / n_pairs) if n_pairs else float("nan"),
        integer_dt_fraction=(n_int_dt / n_dt) if n_dt else float("nan"),
        timestamp_resolution={k: float(v) for k, v in sorted(res_by_mag.items())},
        tau_quantiles=_quantile_dict(tau_s),
        spread_padding_only=_quantile_dict(sp_pad),
        spread_causal=_quantile_dict(sp_cau),
        frac_rows_spread_below_0p1=float((sp_pad < 0.1).mean()) if sp_pad.size else float("nan"),
        causal_frac_rows_spread_below_0p1=(float((sp_cau < 0.1).mean()) if sp_cau.size
                                           else float("nan")),
    )


def corpus_stats_cached(dataset: TensorizedPretrainDataset, split_dir: Path, *,
                        split: str = "train", sample_windows: int = 4000, seed: int = 0,
                        max_seq_len: int = 1024) -> CorpusStats:
    """:func:`corpus_stats`, cached to ``<split_dir>/corpus_stats.json``.

    The statistics depend only on the corpus and the sampling parameters, never on the arm,
    so the four arms (and any extra seeds) share one full-split pass. The cache key includes
    everything that changes the result; a mismatch recomputes. The cache is JSON, so it is
    inspectable and survives being copied with the data.
    """
    import json

    key = {"n_examples": len(dataset), "n_patients": dataset.n_patients,
           "sample_windows": int(sample_windows), "seed": int(seed),
           "max_seq_len": int(max_seq_len), "split": split,
           "schema": 2, "horizon": "strict_time_cut"}
    cache = Path(split_dir) / "corpus_stats.json"
    if cache.exists():
        try:
            blob = json.loads(cache.read_text())
            if blob.get("_key") == key:
                return CorpusStats.from_json(blob["stats"])
        except Exception:
            pass  # unreadable or stale -> recompute
    stats = corpus_stats(dataset, split=split, sample_windows=sample_windows, seed=seed)
    try:
        from model_new import diagnostics  # diagnostics owns all JSON writing (D11)
        diagnostics.write_json(cache, {"_key": key, "stats": stats.to_json()})
    except Exception:
        pass  # a read-only data dir is not fatal; we just recompute next time
    return stats


def sample_empirical_taus(dataset: TensorizedPretrainDataset, n_examples: int = 400,
                          max_pairs_per_example: int = 3000, seed: int = 0) -> np.ndarray:
    """Real within-window pairwise lags, for the basis condition number.

    The condition number depends entirely on where the mass sits: a uniform sample over
    ``[0, tau_max]`` flatters both bases and understates how badly the monomial one is
    conditioned on data whose lags pile up at particular scales.
    """
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for j in _sample_indices(len(dataset), n_examples, seed):
        ts = dataset[int(j)]["timestamps_days"].astype(np.float64)
        if ts.size < 2:
            continue
        iu = np.triu_indices(ts.size, k=1)
        d = np.abs(ts[:, None] - ts[None, :])[iu]
        if d.size > max_pairs_per_example:
            d = rng.choice(d, max_pairs_per_example, replace=False)
        out.append(spans_to_tau(d))
    return np.concatenate(out) if out else np.zeros(0)


def _smoke() -> None:
    from model_new import diagnostics

    rng = np.random.default_rng(0)
    items = []
    for n in (5, 1, 9):
        items.append({
            "code_indices": rng.integers(0, 100, size=n),
            "timestamps_days": np.sort(rng.random(n) * 400).astype(np.float32),
            "age_days": (np.sort(rng.random(n) * 400) + 2000).astype(np.float32),
            "sex": 1, "race": 3, "unk_vocab_index": 100,
            "target_codes": np.zeros(100, dtype=np.float32),
        })
    b = pretrain_collate(items)
    dim, names = demo_layout("one_hot")
    tau, tau_to_now = tau_from_timestamps(b["timestamps_days"], b["attention_mask"], b["lengths"])
    pair = b["attention_mask"].unsqueeze(2) & b["attention_mask"].unsqueeze(1)
    diagnostics.print_block("data.py smoke", [
        f"batch keys     : {sorted(k for k in b)}",
        f"timestamps_days: {tuple(b['timestamps_days'].shape)} {b['timestamps_days'].dtype}  "
        f"(8 KB/sample vs tau's 4.19 MB/sample at L=1024)",
        f"lengths        : {b['lengths'].tolist()}",
        f"tau (in model) : {tuple(tau.shape)} {tau.dtype}  finite={bool(torch.isfinite(tau).all())}",
        f"tau_to_now     : {tuple(tau_to_now.shape)}  max={float(tau_to_now.max()):.4f}",
        f"demographics   : {tuple(b['demographics'].shape)}  demo_dim={dim}",
        f"age in demo[0] : {bool(torch.allclose(b['demographics'][..., 0].double(), b['age_years'].double()))}  "
        f"(raw years; standardization happens inside the model, not the collate)",
        f"padded tau zero: {float(tau[~pair].abs().max()):.1e}",
    ])


if __name__ == "__main__":
    _smoke()
