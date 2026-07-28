#!/usr/bin/env python3
"""INV-HORIZON -- max(input_timestamps) < target_time for every constructed example.

Future-visit forecasting (not masked-visit prediction): input is every event with
``timestamp < start_time(V_{m+1})``; the target is the code set of ``V_{m+1}``. Ties at
the target start belong to the target and are excluded from input. Count truncation
drops the oldest pre-boundary events only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from model_new.data import (
    select_forecast_input_indices,
    target_visit_start_time,
    TensorizedPretrainDataset,
    pretrain_collate,
)


def _write_flat_shard(path: Path, *, subject_id: int, codes: np.ndarray,
                      ts: np.ndarray, ages: np.ndarray,
                      visit_starts: np.ndarray, visit_ends: np.ndarray,
                      unk: int = 100) -> None:
    n_ev = int(codes.shape[0])
    np.savez(
        path,
        subject_id=np.asarray([subject_id], dtype=np.int64),
        sex=np.asarray([1], dtype=np.int8),
        race=np.asarray([0], dtype=np.int16),
        event_offsets=np.asarray([0, n_ev], dtype=np.int64),
        code_indices=np.asarray(codes, dtype=np.int64),
        timestamps_days=np.asarray(ts, dtype=np.float32),
        age_days=np.asarray(ages, dtype=np.float32),
        visit_offsets=np.asarray([0, len(visit_starts)], dtype=np.int64),
        visit_starts=np.asarray(visit_starts, dtype=np.int32),
        visit_ends=np.asarray(visit_ends, dtype=np.int32),
        unk_vocab_index=np.asarray([unk], dtype=np.int64),
    )


def _vocab(path: Path, n: int = 100) -> Path:
    import json
    path.write_text(json.dumps({str(i): i for i in range(n)}))
    return path


def test_select_excludes_ties_and_drops_oldest():
    ts = np.asarray([0.0, 5.0, 10.0, 10.0, 12.0], dtype=np.float64)
    # target at 10: keep only t<10, then truncate to 1 -> keep the newest (=5).
    sel = select_forecast_input_indices(ts, target_time=10.0, max_seq_len=1)
    assert sel.tolist() == [1]
    assert float(ts[sel].max()) < 10.0


def test_overlapping_hadm_style_window(tmp_path: Path):
    """Null/earlier visit spans past the next admission; time cut must drop the leak."""
    # Visit 0: t=0, 5, 20 (extends past target). Visit 1: t=10, 10.5.
    codes = np.arange(5, dtype=np.int64)
    ts = np.asarray([0.0, 5.0, 20.0, 10.0, 10.5], dtype=np.float32)
    ages = np.asarray([365.0, 370.0, 385.0, 375.0, 375.5], dtype=np.float32)
    # Stored in visit-block order (as tensorize writes): visit0 then visit1.
    # Re-order to match _build_subject_payload: [0,5,20 | 10,10.5]
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _write_flat_shard(
        shard_dir / "shard_0000.npz",
        subject_id=42,
        codes=codes,
        ts=ts,
        ages=ages,
        visit_starts=np.asarray([0, 3], dtype=np.int32),
        visit_ends=np.asarray([3, 5], dtype=np.int32),
    )
    vocab = _vocab(tmp_path / "vocab.json", n=100)
    ds = TensorizedPretrainDataset(shard_dir, vocab, max_seq_len=1024, assert_horizon=True)
    assert len(ds) == 1
    item = ds[0]
    assert float(item["timestamps_days"].max()) < float(item["target_time"])
    assert float(item["target_time"]) == pytest.approx(10.0)
    # Events at/after 10 excluded: only t=0 and t=5 remain (t=20 dropped).
    assert sorted(float(x) for x in item["timestamps_days"]) == [0.0, 5.0]
    assert int(item["target_codes"].sum()) == 2  # codes 3 and 4


def test_same_timestamp_prior_visit_yields_no_example(tmp_path: Path):
    """If every prior event ties the target start, there is no strict-past context."""
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    _write_flat_shard(
        shard_dir / "shard_0000.npz",
        subject_id=7,
        codes=np.asarray([1, 2], dtype=np.int64),
        ts=np.asarray([3.0, 3.0], dtype=np.float32),
        ages=np.asarray([100.0, 100.0], dtype=np.float32),
        visit_starts=np.asarray([0, 1], dtype=np.int32),
        visit_ends=np.asarray([1, 2], dtype=np.int32),
    )
    vocab = _vocab(tmp_path / "vocab.json")
    ds = TensorizedPretrainDataset(shard_dir, vocab, assert_horizon=True)
    assert len(ds) == 0


def test_collate_asserts_horizon():
    items = [{
        "code_indices": np.asarray([1, 2], dtype=np.int64),
        "timestamps_days": np.asarray([1.0, 5.0], dtype=np.float32),
        "age_days": np.asarray([10.0, 20.0], dtype=np.float32),
        "sex": 1, "race": 0, "unk_vocab_index": 10,
        "target_codes": np.zeros(8, dtype=np.float32),
        "target_time": 4.0,  # 5.0 >= 4.0 -> violation
    }]
    with pytest.raises(AssertionError, match="INV-HORIZON"):
        pretrain_collate(items, assert_horizon=True)


def test_target_visit_start_time_is_min():
    ts = np.asarray([9.0, 7.0, 8.0], dtype=np.float64)
    assert target_visit_start_time(ts, 0, 3) == 7.0
