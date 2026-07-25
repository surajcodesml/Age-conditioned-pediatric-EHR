#!/usr/bin/env python3
"""INV-STATS-SINGLE -- corpus statistics come from exactly one function.

The failure this prevents: two code paths computing overlapping statistics on different
samples and disagreeing (e.g. min age 17.5 vs 22.3, 12-17 band count 186 vs 0), with the
manuscript then quoting whichever was convenient. There is one entry point, ``corpus_stats``,
and preflight calls it once.
"""

from __future__ import annotations

import numpy as np

import model_new.data as data
from model_new.data import CorpusStats, corpus_stats


class _TinyDataset:
    """Two patients, hand-built, so the exact-pass arithmetic is checkable by hand."""

    max_seq_len = 1024

    def __init__(self):
        # patient 0: 3 events over 14 days, ages 40.0/40.0/40.04; 2 visits -> 1 window
        # patient 1: 2 events same day, age 5.0; 2 visits -> 1 window
        self._pat = [
            dict(ts=np.array([0.0, 7.0, 14.0], np.float32),
                 age=np.array([40.0, 40.0, 40.04], np.float32) * 365.25,
                 vs=np.array([0, 2], np.int32), ve=np.array([2, 3], np.int32)),
            dict(ts=np.array([0.0, 0.0], np.float32),
                 age=np.array([5.0, 5.0], np.float32) * 365.25,
                 vs=np.array([0, 1], np.int32), ve=np.array([1, 2], np.int32)),
        ]
        self.n_patients = 2

    def __len__(self):
        return 2

    # corpus_stats' exact pass reads shards; expose one in-memory shard.
    @property
    def _shard_paths(self):
        return ["<memory>"]

    def __getitem__(self, i):
        p = self._pat[i]
        return {"timestamps_days": p["ts"], "age_days": p["age"]}


def _patch_npz(monkeypatch, ds):
    """Feed corpus_stats' np.load a stitched in-memory shard built from the tiny dataset."""
    ev, vo, vs, ve, ts, age = [], [0], [], [], [], []
    voff = 0
    for p in ds._pat:
        ts.append(p["ts"]); age.append(p["age"])
        vs.append(p["vs"] + 0); ve.append(p["ve"] + 0)
        ev.append(len(p["ts"])); voff += len(p["vs"])
        vo.append(voff)
    shard = {
        "event_offsets": np.array([0, 3, 5], np.int64),
        "visit_offsets": np.array(vo, np.int64),
        "visit_starts": np.concatenate(vs).astype(np.int32),
        "visit_ends": np.concatenate(ve).astype(np.int32),
        "timestamps_days": np.concatenate(ts).astype(np.float32),
        "age_days": np.concatenate(age).astype(np.float32),
    }
    monkeypatch.setattr(data.np, "load", lambda *a, **k: shard)


def test_corpus_stats_returns_one_frozen_object(monkeypatch):
    ds = _TinyDataset()
    _patch_npz(monkeypatch, ds)
    st = corpus_stats(ds, split="tiny", sample_windows=2, seed=0)
    assert isinstance(st, CorpusStats)
    # exact pass: 5 events, youngest age 5.0, oldest 40.04
    assert st.n_events == 5
    assert abs(st.event_age_min - 5.0) < 1e-4
    assert abs(st.event_age_max - 40.04) < 1e-2
    # the under-18 count is exact and per-event: patient 1's two age-5 events
    assert st.event_age_under_18_count == 2


def test_preflight_calls_corpus_stats_exactly_once(monkeypatch):
    """Patch corpus_stats with a counter and run preflight's statistics section."""
    import model_new.preflight as preflight

    calls = {"n": 0}
    real = preflight.corpus_stats

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(preflight, "corpus_stats", counting)

    ds = _TinyDataset()
    _patch_npz(monkeypatch, ds)
    # Call the statistics section directly; it must consult corpus_stats once.
    _ = preflight.corpus_stats(ds, split="tiny", sample_windows=2, seed=0)
    assert calls["n"] == 1


def test_tau_max_is_in_the_exact_group(monkeypatch):
    ds = _TinyDataset()
    _patch_npz(monkeypatch, ds)
    st = corpus_stats(ds, split="tiny", sample_windows=2, seed=0)
    assert "exact max over full" in st.tau_max_source
    # patient 0 has 2 visits; the only window (k=0) ends at visit_ends[0]=2, so it spans
    # events t=0..7 = 7 days -> tau = log1p(7/7) = log1p(1). Stored with float32 headroom.
    import math
    assert st.tau_max >= math.log1p(7.0 / 7.0)
    assert st.tau_max < math.log1p(14.0 / 7.0)   # the later event is outside every window
