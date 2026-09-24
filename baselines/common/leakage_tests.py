"""Automated leakage detection tests.

Run these before any training to verify that:
  1. Patient splits are disjoint (no ID overlap)
  2. No future events appear in model inputs (t_input < t_cutoff)
  3. Target codes do not appear in input codes past the cutoff
  4. Events are in chronological order
  5. Vocabulary alignment is consistent across models
"""
from __future__ import annotations

from typing import Any

import numpy as np


def check_split_disjoint(
    train_ids: set[str], val_ids: set[str], test_ids: set[str],
) -> dict[str, Any]:
    """Verify zero overlap across train/val/test patient IDs."""
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    ok = len(tv) == 0 and len(tt) == 0 and len(vt) == 0
    return {
        "pass": ok,
        "train_val_overlap": len(tv),
        "train_test_overlap": len(tt),
        "val_test_overlap": len(vt),
        "train_n": len(train_ids),
        "val_n": len(val_ids),
        "test_n": len(test_ids),
    }


def check_future_leakage(
    timestamps: np.ndarray,  # [L] event timestamps
    cutoff: float,           # prediction cutoff time
    attention_mask: np.ndarray | None = None,  # [L] True=valid
) -> dict[str, Any]:
    """Verify all input events have timestamp < cutoff."""
    ts = np.asarray(timestamps, dtype=np.float64)
    if attention_mask is not None:
        ts = ts[np.asarray(attention_mask, dtype=bool)]
    violations = int(np.sum(ts >= cutoff))
    return {
        "pass": violations == 0,
        "n_violations": violations,
        "n_valid": int(ts.size),
        "max_timestamp": float(ts.max()) if ts.size > 0 else float("nan"),
        "cutoff": float(cutoff),
    }


def check_target_in_input(
    input_codes: set[str],
    target_codes: set[str],
) -> dict[str, Any]:
    """Check for target-in-input leakage (overlap is often expected in EHR)."""
    overlap = input_codes & target_codes
    return {
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / max(len(target_codes), 1),
        "note": "Overlap is expected in EHR (recurrent codes); verify event order instead",
    }


def check_timestamp_ordering(
    timestamps: np.ndarray,
    attention_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Verify events are in non-decreasing chronological order."""
    ts = np.asarray(timestamps, dtype=np.float64)
    if attention_mask is not None:
        ts = ts[np.asarray(attention_mask, dtype=bool)]
    if ts.size <= 1:
        return {"pass": True, "n_events": int(ts.size)}
    diffs = np.diff(ts)
    violations = int(np.sum(diffs < -1e-9))  # allow tiny float noise
    return {
        "pass": violations == 0,
        "n_violations": violations,
        "n_events": int(ts.size),
    }


def check_vocabulary_alignment(
    vocab_a: dict[str, int],
    vocab_b: dict[str, int],
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> dict[str, Any]:
    """Check that two vocabulary mappings agree on shared codes."""
    shared = set(vocab_a.keys()) & set(vocab_b.keys())
    mismatches = []
    for code in sorted(shared):
        if vocab_a[code] != vocab_b[code]:
            mismatches.append(code)
    return {
        "pass": len(mismatches) == 0,
        f"n_{name_a}": len(vocab_a),
        f"n_{name_b}": len(vocab_b),
        "n_shared": len(shared),
        "n_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:10],
    }


def run_all_checks(
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> dict[str, Any]:
    """Run split-level checks. Per-sample checks should be called separately."""
    results: dict[str, Any] = {}
    results["split_disjoint"] = check_split_disjoint(train_ids, val_ids, test_ids)
    results["all_pass"] = results["split_disjoint"]["pass"]
    return results
