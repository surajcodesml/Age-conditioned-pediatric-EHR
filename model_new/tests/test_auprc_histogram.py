#!/usr/bin/env python3
"""The streaming histogram AUPRC must agree with sklearn's exact average precision.

``eval_pretrain`` cannot materialise the 52,227 x 30,635 score matrix the pretraining
task produces, so micro- and macro-AUPRC are integrated from fixed-edge score histograms
instead. That is an approximation, and this is the test that says how good it is: on a
small synthetic case where the exact answer is computable, the histogram estimate must
match ``sklearn.average_precision_score`` to 1e-3.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model_new import diagnostics as D

TOL = 1e-3


def _synthetic(n: int = 4000, seed: int = 0, prevalence: float = 0.05):
    """Overlapping positive/negative score distributions, so AP is nontrivial."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.float32)
    scores = rng.normal(-7.0, 2.0, size=n) + 3.0 * y
    return scores.astype(np.float32), y


@pytest.mark.parametrize("prevalence", [0.05, 0.4])
def test_micro_histogram_matches_sklearn(prevalence):
    from sklearn.metrics import average_precision_score

    scores, y = _synthetic(prevalence=prevalence)
    exact = float(average_precision_score(y, scores))

    hist = D.ScoreHistogram(float(scores.min()) - 0.5, float(scores.max()) + 0.5,
                            n_bins=100_000)
    # fed in several chunks: the estimate must not depend on the batching
    for lo in range(0, len(scores), 137):
        hist.update(torch.from_numpy(scores[lo:lo + 137]), torch.from_numpy(y[lo:lo + 137]))
    got = hist.average_precision()

    assert hist.n == len(scores)
    assert hist.n_pos == int(y.sum())
    assert hist.n_below == 0 and hist.n_above == 0
    assert abs(got - exact) < TOL, f"histogram AP {got} vs sklearn {exact}"


def test_out_of_range_scores_are_counted_not_hidden():
    scores, y = _synthetic(n=500)
    hist = D.ScoreHistogram(-8.0, -6.0, n_bins=1000)
    hist.update(torch.from_numpy(scores), torch.from_numpy(y))
    assert hist.n_below + hist.n_above > 0
    assert hist.to_json()["out_of_range_fraction"] > 0


def test_per_code_histogram_matches_sklearn_per_code():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(3)
    n, v = 800, 6
    y = (rng.random((n, v)) < np.linspace(0.03, 0.3, v)).astype(np.float32)
    scores = (rng.normal(-6.0, 1.5, size=(n, v)) + 2.5 * y).astype(np.float32)

    codes = torch.arange(v)
    hist = D.PerCodeHistogram(codes, float(scores.min()) - 0.5, float(scores.max()) + 0.5,
                              n_bins=100_000)
    for lo in range(0, n, 97):
        hist.update(torch.from_numpy(scores[lo:lo + 97]), torch.from_numpy(y[lo:lo + 97]))

    got = hist.average_precision_per_code()
    want = np.array([average_precision_score(y[:, c], scores[:, c]) for c in range(v)])
    assert np.allclose(hist.positives_per_code(), y.sum(axis=0))
    assert np.max(np.abs(got - want)) < TOL, f"{got} vs {want}"


def test_average_precision_is_nan_without_positives():
    hist = D.ScoreHistogram(-1.0, 1.0, n_bins=100)
    hist.update(torch.zeros(10), torch.zeros(10))
    assert np.isnan(hist.average_precision())


def test_ndcg_and_recall_denominators():
    """Recall is over |true| by default, and nDCG's ideal DCG is over min(|true|, k)."""
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]])          # 3 true codes
    out = D.topk_per_example(logits, targets, ks=(2,), ndcg_k=3)
    assert out["recall@2"].item() == pytest.approx(1.0 / 3.0)          # 1 hit of 3 true
    capped = D.topk_per_example(logits, targets, ks=(2,), ndcg_k=3, cap_denominator=True)
    assert capped["recall@2"].item() == pytest.approx(0.5)             # 1 hit of min(3, 2)

    disc = 1.0 / np.log2(np.arange(2, 5))
    dcg = disc[0] + disc[2]                                            # hits at ranks 1, 3
    idcg = disc[:3].sum()
    assert out["ndcg@3"].item() == pytest.approx(dcg / idcg, rel=1e-6)


def test_empty_target_row_is_nan_not_zero():
    logits = torch.zeros(1, 5)
    targets = torch.zeros(1, 5)
    out = D.topk_per_example(logits, targets, ks=(2,), ndcg_k=2)
    assert torch.isnan(out["recall@2"]).all()
    assert torch.isnan(out["ndcg@2"]).all()


def test_band_entry_flags_and_nans_a_thin_band():
    entry = D.band_entry(n=44, n_pos=1000, n_neg=10_000, metrics={"micro_auprc": 0.42},
                         min_n=200)
    assert entry["unreliable"] is True and "44" in entry["unreliable_reason"]
    assert np.isnan(entry["micro_auprc"])
    assert entry["n"] == 44 and entry["n_pos"] == 1000 and entry["n_neg"] == 10_000

    ok = D.band_entry(n=20_000, n_pos=1000, n_neg=10_000, metrics={"micro_auprc": 0.42},
                      min_n=200)
    assert ok["unreliable"] is False and ok["micro_auprc"] == 0.42

    empty = D.band_entry(n=0, n_pos=0, n_neg=0, metrics={"micro_auprc": 0.1}, min_n=200)
    assert empty["unreliable"] is True and np.isnan(empty["micro_auprc"])
