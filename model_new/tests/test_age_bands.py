#!/usr/bin/env python3
"""D-1 -- one band definition, parameterised, consumed everywhere.

The failure this guards against is a forked band table: a fine-tune metric stratifying by
one partition while the ``Delta-alpha`` decomposition next to it in the same JSON uses
another, so a "band" in one block and a "band" in the other are different sets of patients.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model_new import diagnostics as D

from .conftest import wake_generators_


@pytest.mark.parametrize("name", sorted(D.BAND_TABLES))
def test_every_table_partitions_its_range(name):
    table = D.resolve_bands(name)
    assert table[0][1] == 0.0, "a band table must start at age 0"
    assert table[-1][2] == float("inf"), "the last band must be a catch-all"
    for (_, _, hi), (_, lo, _) in zip(table, table[1:]):
        assert hi == lo, "bands must abut exactly: no gaps, no overlaps"
    for n, lo, hi in table:
        assert hi > lo, f"band {n} is empty or inverted"
    assert len({n for n, _, _ in table}) == len(table), "band names must be unique"


def test_resolve_bands_accepts_a_name_a_table_or_none():
    assert D.resolve_bands(None) is D.AGE_BANDS
    assert D.resolve_bands("adult") is D.AGE_BANDS
    assert D.resolve_bands("pediatric") is D.PEDIATRIC_AGE_BANDS
    assert D.resolve_bands(D.PEDIATRIC_AGE_BANDS) == D.PEDIATRIC_AGE_BANDS
    with pytest.raises(ValueError):
        D.resolve_bands("neonatal-ish")


def test_pediatric_table_resolves_what_the_adult_one_cannot():
    """Every PIC age in the audit falls in 0-18; the adult table puts almost all of it in
    one band, which is the whole reason PEDIATRIC_AGE_BANDS exists."""
    ages = np.array([0.003, 0.05, 0.3, 0.8, 2.0, 4.5, 9.0, 15.0])
    adult = D.band_index(ages, "adult")
    ped = D.band_index(ages, "pediatric")
    assert len(set(adult.tolist())) == 4
    assert len(set(ped.tolist())) == 6
    assert (adult >= 0).all() and (ped >= 0).all(), "no age may fall outside every band"


def test_band_index_and_names_and_masks_agree_on_the_same_table():
    ages = np.array([0.01, 0.5, 2.0, 4.0, 8.0, 15.0, 40.0])
    for table in ("adult", "pediatric"):
        names = D.band_names(table)
        idx = D.band_index(ages, table)
        masks = D.band_masks(ages, table)
        assert list(masks) == names, "band_masks must emit every band, empty ones included"
        for i, n in enumerate(names):
            assert np.array_equal(masks[n], idx == i)
        assert sum(m.sum() for m in masks.values()) == ages.size


def test_defaults_are_unchanged_for_the_pretraining_path():
    """Every existing caller passes no table and must keep the adult partition."""
    ages = np.array([0.5, 3.0, 8.0, 15.0, 30.0, 50.0, 80.0])
    assert D.band_names() == [n for n, _, _ in D.AGE_BANDS]
    assert np.array_equal(D.band_index(ages), D.band_index(ages, "adult"))
    assert list(D.band_masks(ages)) == D.band_names()


def test_aggregate_recall_honours_the_table():
    per_example = {10: torch.tensor([1.0, 0.5, 0.0, 0.25])}
    ages = torch.tensor([0.01, 0.5, 2.0, 15.0])
    ped = D.aggregate_recall(per_example, ages, bands="pediatric")
    assert list(ped["by_band"]) == D.band_names("pediatric")
    assert ped["by_band"]["neonate"]["n"] == 1
    adult = D.aggregate_recall(per_example, ages)
    assert list(adult["by_band"]) == D.band_names()
    assert adult["by_band"]["<1"]["n"] == 2, "0.01 and 0.5 both land in the adult '<1'"


def test_alpha_diagnostics_honours_the_table(model_factory):
    m = model_factory("kernel")
    wake_generators_(m)
    ages = torch.tensor([0.01, 0.5, 2.0, 4.0, 8.0, 15.0])
    ped = D.alpha_diagnostics(m, ages, bands="pediatric")
    site = next(iter(ped))
    assert list(ped[site]["by_band"]) == D.band_names("pediatric")
    assert ped[site]["by_band"]["toddler"]["n"] == 1
    adult = D.alpha_diagnostics(m, ages)
    assert list(adult[site]["by_band"]) == D.band_names()


def test_pediatric_supports_cover_only_the_pediatric_range():
    assert D.PEDIATRIC_DENSE_AGE_GRID.min() == 0.0
    assert D.PEDIATRIC_DENSE_AGE_GRID.max() == 18.0
    assert D.DENSE_AGE_GRID.max() == 90.0
    assert max(D.PEDIATRIC_W_CURVE_AGES) <= 18.0
    assert D.PEDIATRIC_MIN_BAND_N < D.MIN_BAND_N, (
        "PIC cohorts are ~9k sequences against MIMIC's 52k, so the adult threshold would "
        "mark every pediatric band unreliable on arithmetic alone")
