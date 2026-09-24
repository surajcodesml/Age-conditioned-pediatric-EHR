"""Encounter construction from event-level histories.

Grouping rule (deterministic, documented):
1. Apply the same event truncation as the event-level benchmark.
2. Group remaining events by rounded lag-from-cutoff (same timestamp ≈ one encounter).
3. Signal events share a group with any co-timed codes; otherwise each signal lag
   forms its own synthetic encounter at the injected timestamp.
4. Encounter τ / lag = mean over events in the group.
5. Encounter order: oldest → newest (decreasing lag).

The encounter encoder never sees age, τ, or absolute time — only code identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import tau_from_days
from dataset import apply_model_truncation


LAG_ROUND_DECIMALS = 3  # group events whose lags agree to ~0.001 day


@dataclass
class Encounter:
    codes: list[str]
    types: list[str]
    lag_days: float
    tau: float
    n_signal: int
    is_synthetic_signal: bool  # True if group contains only signal(s) (no Synthea ENC_*)


def group_events_to_encounters(
    codes: list[str],
    types: list[str],
    lags: list[float],
    taus: list[float] | None = None,
    *,
    lag_round: int = LAG_ROUND_DECIMALS,
) -> list[Encounter]:
    """Group co-timed events into encounters. Does not use age."""
    if not codes:
        return []
    if taus is None or len(taus) != len(lags):
        taus = list(tau_from_days(np.asarray(lags, dtype=np.float64)))
    buckets: dict[float, list[int]] = {}
    for i, lag in enumerate(lags):
        key = round(float(lag), lag_round)
        buckets.setdefault(key, []).append(i)
    encounters: list[Encounter] = []
    for key in sorted(buckets.keys(), reverse=True):  # oldest first
        idxs = buckets[key]
        c = [str(codes[i]) for i in idxs]
        t = [str(types[i]) for i in idxs]
        lag_vals = [float(lags[i]) for i in idxs]
        tau_vals = [float(taus[i]) for i in idxs]
        n_sig = sum(1 for x in t if x == "signal")
        has_enc = any(x.startswith("ENC_") for x in c) or any(x == "encounter" for x in t)
        encounters.append(
            Encounter(
                codes=c,
                types=t,
                lag_days=float(np.mean(lag_vals)),
                tau=float(np.mean(tau_vals)),
                n_signal=n_sig,
                is_synthetic_signal=(n_sig > 0 and not has_enc and n_sig == len(c)),
            )
        )
    return encounters


def truncate_and_group(
    codes: list[str],
    types: list[str],
    lags: list[float],
    taus: list[float] | None,
    *,
    max_seq_len: int,
    max_background: int,
    max_encounters: int | None = None,
) -> list[Encounter]:
    trunc = apply_model_truncation(
        codes, types, lags, taus, max_seq_len=max_seq_len, max_background=max_background
    )
    # Drop prediction/query token if present — DTR does not use a query event.
    keep = [i for i, q in enumerate(trunc["is_query"]) if not q]
    encs = group_events_to_encounters(
        [trunc["codes"][i] for i in keep],
        [trunc["types"][i] for i in keep],
        [trunc["lags"][i] for i in keep],
        [trunc["taus"][i] for i in keep],
    )
    if max_encounters is not None and len(encs) > max_encounters:
        # Keep newest encounters (end of oldest→newest list).
        encs = encs[-max_encounters:]
    return encs


def encounter_stats(examples, *, max_seq_len: int = 96, max_background: int = 64) -> dict[str, Any]:
    """Summarize encounter construction over a cohort."""
    n_enc, codes_per, n_sig_enc, hist_len = [], [], [], []
    for row in examples.itertuples(index=False):
        encs = truncate_and_group(
            list(row.history_codes),
            list(row.history_types),
            list(row.history_lag_days),
            list(row.history_tau) if isinstance(row.history_tau, list) else None,
            max_seq_len=max_seq_len,
            max_background=max_background,
        )
        n_enc.append(len(encs))
        hist_len.append(sum(len(e.codes) for e in encs))
        codes_per.extend(len(e.codes) for e in encs)
        n_sig_enc.append(sum(1 for e in encs if e.n_signal > 0))
    def _summ(arr):
        a = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(a.mean()) if len(a) else float("nan"),
            "median": float(np.median(a)) if len(a) else float("nan"),
            "p90": float(np.percentile(a, 90)) if len(a) else float("nan"),
            "max": float(a.max()) if len(a) else float("nan"),
            "min": float(a.min()) if len(a) else float("nan"),
        }
    return {
        "n_examples": len(examples),
        "encounters_per_patient": _summ(n_enc),
        "codes_per_encounter": _summ(codes_per),
        "event_history_length": _summ(hist_len),
        "signal_encounters_per_patient": _summ(n_sig_enc),
        "grouping_rule": (
            f"round(lag_days, {LAG_ROUND_DECIMALS}); oldest→newest; "
            "signals share co-timed groups or form synthetic encounters"
        ),
    }
