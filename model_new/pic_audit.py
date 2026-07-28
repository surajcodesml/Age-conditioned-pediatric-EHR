#!/usr/bin/env python3
"""PIC fine-tune readiness audit. Measures, prints, writes JSON, trains nothing, exits 0.

    python -m model_new.pic_audit
    python -m model_new.pic_audit --tasks heart_malformations --skip_unclipped

Two phases, both read-only:

**Phase A -- inventory.** Shard schema against what ``model_new``'s fine-tune collate
actually consumes, cohort sizes and patient-level split disjointness, sequence length and
observation-window span, vocabulary transfer from PIC's own vocab into the MIMIC vocab the
backbone was pretrained on, the demographic layout under PIC's all-``UNKNOWN`` race, and
pediatric age-band coverage.

**Phase B -- the M1 measurement.** README section 5c reports that under the frozen MIMIC
``tau_max = 6.7238`` the PIC ``tau_tilde`` distribution occupies 2.0% of ``[-1, 1]`` with a
Chebyshev Gram condition number of 5.7e16. This reproduces that per task and extends it in
the one direction that decides what to do about it: the **unclipped** PIC event stream. If
the 24 h cohort window is what puts every lag in a sliver near -1, M1 is a data artifact and
the fix is on the data side; if the full pre-index history occupies no more of the domain,
it is a design fault in the shared frozen ``tau_max``.

Nothing here chooses. Decisions D1 (``tau_max``), D2 (which backbone each arm fine-tunes
from) and D3 (vocabulary / embedding table) are printed with numbers attached and left open.

Every quantity that already exists somewhere is imported, never re-derived:
``data.spans_to_tau`` for the lag convention, ``basis.chebyshev_basis`` through
``diagnostics.gram_condition_numbers`` for conditioning, ``encoder.build_pair_mask`` for
validity, ``preflight.headroom`` for the equal-norm probe, ``diagnostics`` for all output.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from model_new import diagnostics as D
from model_new.arms import ARMS
from model_new.basis import chebyshev_basis
from model_new.data import (
    DAYS_PER_YEAR, N_RACE, RACE_LABELS, _sample_indices, demo_layout, load_vocab,
    spans_to_tau,
)
from model_new.data_finetune import TensorizedFinetuneDataset, finetune_collate
from model_new.encoder import build_pair_mask
from model_new.model import DKMModel
from model_new.preflight import headroom
from model_new.train_finetune import load_backbone

REPO_ROOT = Path(__file__).resolve().parents[1]

PIC_TASKS = ("mortality", "los_gt7", "pneumonia", "heart_malformations")
SPLITS = ("train", "val", "test")

# What model_new's fine-tune path reads out of a shard, and what it derives rather than
# reads. Sourced from data_finetune.TensorizedFinetuneDataset._load_shard / __getitem__ and
# data._pad_common; if either changes, this table is wrong and the audit says so.
CONSUMED_FIELDS = ("offsets", "code_indices", "timestamps_days", "age_days", "subject_id",
                   "sex", "race", "label", "unk_vocab_index")
OPTIONAL_FIELDS = ("hadm_id",)
DERIVED_BATCH_KEYS = {
    "age_years": "age_days / 365.25, its OWN [B, L] tensor (D3); never read out of "
                 "demographics",
    "lengths": "from offsets, asserted equal to attention_mask.sum(1)",
    "attention_mask": "from offsets",
    "demographics": "[B, L, demo_dim] built from age_years + sex + one-hot race",
    "split": "the containing directory (train/ val/ test/); NOT a shard field",
    "tau": "computed in DKMModel.forward from timestamps_days via data.tau_from_timestamps",
}
# Fields the legacy finetune/dataset.py supplied that model_new does not consume.
LEGACY_ONLY_FIELDS = ("n_events_in_window",)

# The pretraining constants, quoted here only so a disagreement is loud. Both are read
# from the checkpoint at run time; nothing below uses these literals for arithmetic.
EXPECTED_TAU_MAX = 6.72380256652832
EXPECTED_AGE_MEAN = 63.33600997924805
EXPECTED_AGE_SD = 16.574804306030273

TAU_MAX_SWEEP = (1.0, 2.0, 3.0, 6.72380256652832)
WINDOW_SWEEP_DAYS = (1.0, 3.0, 7.0, 30.0, float("inf"))


# --------------------------------------------------------------------------- #
# small shared helpers                                                        #
# --------------------------------------------------------------------------- #
def _q(x: np.ndarray, qs=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)) -> dict:
    if x.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.percentile(x, 100.0 * q)) for q in qs}


def _tau_tilde(tau: np.ndarray, tau_max: float) -> np.ndarray:
    """The single rescaling, matching ChebyshevKernel.rescale, in float64 and unclamped so
    the clamp rate can be counted rather than assumed."""
    return 2.0 * np.asarray(tau, dtype=np.float64) / float(tau_max) - 1.0


def _occupancy(tt: np.ndarray) -> float:
    """Fraction of ``[-1, 1]`` the (clamped) tau_tilde distribution spans."""
    if tt.size == 0:
        return float("nan")
    c = np.clip(tt, -1.0, 1.0)
    return float((c.max() - c.min()) / 2.0)


def _shifted_chebyshev_cond(tt: np.ndarray, s: int) -> dict:
    """Gram condition of ``T_1..T_s`` re-mapped onto the sub-interval the data occupies.

    This is README 5c option 2, evaluated. The basis evaluation is ``basis.chebyshev_basis``
    -- the single implementation -- applied to an affine remap of tau_tilde onto [-1, 1],
    which is exactly what a shifted Chebyshev basis on ``[tt_min, tt_max]`` is.
    """
    c = np.clip(np.asarray(tt, dtype=np.float64), -1.0, 1.0)
    lo, hi = float(c.min()), float(c.max())
    if not (hi > lo):
        return {"sub_interval": [lo, hi], "chebyshev_no_constant": float("inf"),
                "degenerate": True}
    x = 2.0 * (c - lo) / (hi - lo) - 1.0
    basis = chebyshev_basis(torch.from_numpy(x), s).numpy()
    g = basis.T @ basis / max(1, basis.shape[0])
    return {"sub_interval": [lo, hi], "chebyshev_no_constant": float(np.linalg.cond(g)),
            "degenerate": False}


def _split_dir(root: Path, task: str, split: str) -> Path:
    return root / task / split


def _open_shards(root: Path, task: str, split: str) -> list[np.lib.npyio.NpzFile]:
    d = _split_dir(root, task, split)
    return [np.load(p, allow_pickle=False) for p in sorted(d.glob("shard_*.npz"))]


# --------------------------------------------------------------------------- #
# Phase A                                                                     #
# --------------------------------------------------------------------------- #
def a1_schema(root: Path, task: str) -> dict:
    """A1. Field names / dtypes / shapes, against what the collate consumes."""
    per_split: dict[str, dict] = {}
    for split in SPLITS:
        d = _split_dir(root, task, split)
        paths = sorted(d.glob("shard_*.npz"))
        if not paths:
            per_split[split] = {"present": False, "dir": str(d)}
            continue
        fields: dict[str, dict] = {}
        for p in paths:
            z = np.load(p, allow_pickle=False)
            for k in z.files:
                a = z[k]
                prev = fields.get(k)
                entry = {"dtype": str(a.dtype), "shape": list(a.shape)}
                if prev is not None and prev["dtype"] != entry["dtype"]:
                    entry["dtype_disagrees_across_shards"] = True
                fields[k] = entry
            z.close()
        present = set(fields)
        per_split[split] = {
            "present": True,
            "dir": str(d),
            "n_shards": len(paths),
            "fields": fields,
            "missing_required": sorted(set(CONSUMED_FIELDS) - present),
            "missing_optional": sorted(set(OPTIONAL_FIELDS) - present),
            "present_but_unconsumed": sorted(present - set(CONSUMED_FIELDS)
                                             - set(OPTIONAL_FIELDS)),
        }
    return {"per_split": per_split,
            "consumed_by_model_new": list(CONSUMED_FIELDS),
            "optional": list(OPTIONAL_FIELDS),
            "derived_not_stored": DERIVED_BATCH_KEYS,
            "supplied_by_legacy_finetune_dataset_only": list(LEGACY_ONLY_FIELDS)}


def a2_cohort(root: Path, task: str) -> dict:
    """A2. Sizes, positive rates, and patient-level disjointness (a hard error if violated)."""
    per_split: dict[str, dict] = {}
    subjects: dict[str, set] = {}
    for split in SPLITS:
        subj: list[np.ndarray] = []
        lab: list[np.ndarray] = []
        for z in _open_shards(root, task, split):
            subj.append(np.asarray(z["subject_id"]))
            lab.append(np.asarray(z["label"], dtype=np.float64))
            z.close()
        if not subj:
            per_split[split] = {"present": False}
            continue
        s = np.concatenate(subj)
        y = np.concatenate(lab)
        subjects[split] = set(int(v) for v in np.unique(s))
        per_split[split] = {
            "present": True,
            "n_sequences": int(y.size),
            "n_unique_subjects": int(len(subjects[split])),
            "sequences_per_subject": float(y.size / max(1, len(subjects[split]))),
            "n_positive": int(y.sum()),
            "positive_rate": float(y.mean()),
        }
    overlaps: dict[str, Any] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in subjects and b in subjects:
            inter = subjects[a] & subjects[b]
            overlaps[f"{a}&{b}"] = {"n": len(inter), "examples": sorted(inter)[:20]}
    leaked = sum(v["n"] for v in overlaps.values())
    sizes = [(k, v["n_sequences"]) for k, v in per_split.items() if v.get("present")]
    smallest = min(sizes, key=lambda kv: kv[1])[0] if sizes else None
    return {
        "per_split": per_split,
        "subject_overlaps": overlaps,
        "patient_level_disjoint": leaked == 0,
        "n_subjects_in_more_than_one_split": leaked,
        "smallest_split": smallest,
        "positives_in_smallest_split": (per_split[smallest]["n_positive"]
                                        if smallest else None),
    }


def a3_length_and_window(root: Path, task: str) -> dict:
    """A3. Events per sequence and window span in days. The single most consequential number."""
    per_split: dict[str, dict] = {}
    for split in SPLITS:
        lens: list[np.ndarray] = []
        spans: list[np.ndarray] = []
        for z in _open_shards(root, task, split):
            off = np.asarray(z["offsets"])
            ts = np.asarray(z["timestamps_days"], dtype=np.float64)
            n = off.size - 1
            L = np.diff(off)
            lens.append(L)
            sp = np.zeros(n, dtype=np.float64)
            for i in range(n):
                a, b = int(off[i]), int(off[i + 1])
                if b > a:
                    sp[i] = ts[b - 1] - ts[a] if b - a > 1 else 0.0
            spans.append(sp)
            z.close()
        if not lens:
            per_split[split] = {"present": False}
            continue
        L = np.concatenate(lens).astype(np.float64)
        S = np.concatenate(spans)
        per_split[split] = {
            "present": True,
            "n_sequences": int(L.size),
            "events_per_sequence": {"min": float(L.min()), "median": float(np.median(L)),
                                    "p90": float(np.percentile(L, 90)),
                                    "max": float(L.max()),
                                    "mean": float(L.mean()), "total": int(L.sum())},
            "span_days": {"min": float(S.min()), "median": float(np.median(S)),
                          "p90": float(np.percentile(S, 90)), "p99": float(np.percentile(S, 99)),
                          "max": float(S.max())},
            "n_sequences_length_1": int((L == 1).sum()),
            "frac_span_at_or_above_1d": float((S >= 0.999).mean()),
        }
    maxes = [v["span_days"]["max"] for v in per_split.values() if v.get("present")]
    overall_max = max(maxes) if maxes else float("nan")
    if overall_max <= 1.0 + 1e-6:
        verdict = ("24 h observation window: the widest span in any split is "
                   f"{overall_max:.6f} d, so OBS_WINDOW_DAYS = 1.0 was applied at cohort "
                   "build time and no sequence carries pre-index history")
    elif overall_max <= 2.0:
        verdict = f"window of about {overall_max:.2f} d; not full pre-index history"
    else:
        verdict = (f"full or long pre-index history: widest span {overall_max:.2f} d "
                   f"({overall_max / 365.25:.2f} y)")
    return {"per_split": per_split, "max_span_days_any_split": overall_max,
            "window_verdict": verdict,
            "legacy_obs_window_days": 1.0}


def _code_family(code: str) -> str:
    return code.split("_", 1)[0] + "_" if "_" in code else code


def a4_vocab(root: Path, task: str, pic_vocab: dict, mimic_vocab: dict) -> dict:
    """A4. Vocabulary transfer, by code family and at the token level."""
    inv = np.empty(len(pic_vocab), dtype=object)
    for code, idx in pic_vocab.items():
        inv[int(idx)] = code
    in_mimic = np.array([str(c) in mimic_vocab for c in inv], dtype=bool)

    by_family: dict[str, dict] = {}
    fam_of = np.array([_code_family(str(c)) for c in inv])
    for fam in sorted(set(fam_of.tolist())):
        sel = fam_of == fam
        by_family[fam] = {"n_pic_codes": int(sel.sum()),
                          "n_in_mimic": int(in_mimic[sel].sum()),
                          "fraction_in_mimic": float(in_mimic[sel].mean())}

    # token-level: every event in every split, counted exactly.
    unk_pic = len(pic_vocab)
    tok: dict[str, dict] = {}
    for split in SPLITS:
        n_tok = n_miss = n_already_unk = 0
        miss_counter: collections.Counter = collections.Counter()
        for z in _open_shards(root, task, split):
            ci = np.asarray(z["code_indices"], dtype=np.int64)
            z.close()
            n_tok += ci.size
            already = ci == unk_pic
            n_already_unk += int(already.sum())
            real = ci[~already]
            miss = ~in_mimic[real]
            n_miss += int(miss.sum())
            miss_counter.update(fam_of[real[miss]].tolist())
        if n_tok == 0:
            tok[split] = {"present": False}
            continue
        tok[split] = {
            "present": True,
            "n_event_tokens": int(n_tok),
            "n_already_unk_in_pic_vocab": int(n_already_unk),
            "n_tokens_absent_from_mimic_vocab": int(n_miss),
            "token_unk_rate_under_reindexing": float((n_miss + n_already_unk) / n_tok),
            "missing_tokens_by_family": dict(miss_counter),
        }
    return {
        "n_pic_codes": len(pic_vocab),
        "n_mimic_codes": len(mimic_vocab),
        "n_pic_codes_in_mimic": int(in_mimic.sum()),
        "fraction_pic_types_in_mimic": float(in_mimic.mean()),
        "by_family": by_family,
        "token_level": tok,
        "option_a_reindex": "PIC code -> MIMIC vocab index, UNK for misses; keeps the "
                            "checkpoint's frozen embedding_table (INV-FROZEN holds)",
        "option_b_pic_table": "swap in bge_embeddings_pic.pt [2200, 1024] and rely on the "
                              "learned 1024 -> d_model input projection; the checkpoint's "
                              "embedding_table buffer CANNOT be restored (shape mismatch), "
                              "so INV-FROZEN's 'restored, never rebuilt' clause does not hold "
                              "for the table",
        "supported_by_current_code": ("neither is implemented. data_finetune passes "
                                      "code_indices through unchanged and train_finetune "
                                      "builds DKMModel with --embedding_path, so today the "
                                      "only working configuration is option (b) with "
                                      "--embedding_path bge_embeddings_pic.pt, and it would "
                                      "fail load_backbone on the embedding_table shape."),
    }


def a5_demographics(root: Path, task: str, demo_channels: tuple[str, ...],
                    age_mean: float, age_sd: float) -> dict:
    """A5. Race distribution, the UNKNOWN column index, and standardized PIC age."""
    counts: collections.Counter = collections.Counter()
    sex_counts: collections.Counter = collections.Counter()
    ages: list[np.ndarray] = []
    for split in SPLITS:
        for z in _open_shards(root, task, split):
            counts.update(np.asarray(z["race"]).tolist())
            sex_counts.update(np.asarray(z["sex"]).tolist())
            off = np.asarray(z["offsets"])
            ad = np.asarray(z["age_days"], dtype=np.float64)
            last = off[1:] - 1
            ages.append(ad[last] / DAYS_PER_YEAR)
            z.close()
    a = np.concatenate(ages) if ages else np.zeros(0)
    z_a = (a - age_mean) / age_sd

    # The ordering is READ from the checkpoint's demo_channels, never rebuilt here. The
    # local RACE_LABELS is compared against it and a disagreement is reported, not patched.
    channel_names = list(demo_channels)
    unknown_channel = None
    for i, name in enumerate(channel_names):
        if name == "race_UNKNOWN":
            unknown_channel = i
    local_idx = RACE_LABELS.index("UNKNOWN")
    return {
        "race_value_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "race_labels_from_checkpoint": channel_names,
        "race_unknown_demo_channel_index": unknown_channel,
        "race_unknown_index_within_one_hot": local_idx,
        "race_ordering_source": "checkpoint config.model.demo_channels",
        "race_ordering_agrees_with_data.RACE_LABELS": (
            channel_names[2:] == [f"race_{r}" for r in RACE_LABELS]),
        "n_race": N_RACE,
        "all_pic_race_is_unknown": set(counts) == {local_idx},
        "sex_value_counts": {str(k): int(v) for k, v in sorted(sex_counts.items())},
        "age_years_at_last_event": {"min": float(a.min()) if a.size else float("nan"),
                                    "median": float(np.median(a)) if a.size else float("nan"),
                                    "max": float(a.max()) if a.size else float("nan")},
        "standardized_age": {
            "constants": {"mean": age_mean, "sd": age_sd,
                          "source": "checkpoint config.model.age_standardization (frozen)"},
            "min": float(z_a.min()) if z_a.size else float("nan"),
            "median": float(np.median(z_a)) if z_a.size else float("nan"),
            "max": float(z_a.max()) if z_a.size else float("nan"),
            "expected_range_from_readme": [-3.8, -2.7],
            "note": "reported, not fixed. Re-standardizing on PIC would put a child at ~0 "
                    "-- PIC's own mean -- instead of ~-3.5 relative to the adult corpus, "
                    "changing the meaning of every learned demo_proj weight (INV-AGESTD).",
        },
    }


def a6_age_bands(root: Path, task: str, bands, min_band_n: int) -> dict:
    """A6. n per pediatric band, per split, with thin bands marked."""
    names = D.band_names(bands)
    out: dict[str, dict] = {}
    for split in SPLITS:
        ages: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for z in _open_shards(root, task, split):
            off = np.asarray(z["offsets"])
            ad = np.asarray(z["age_days"], dtype=np.float64)
            ages.append(ad[off[1:] - 1] / DAYS_PER_YEAR)
            labels.append(np.asarray(z["label"], dtype=np.float64))
            z.close()
        if not ages:
            out[split] = {"present": False}
            continue
        a = np.concatenate(ages)
        y = np.concatenate(labels)
        idx = D.band_index(a, bands)
        entry: dict[str, Any] = {}
        for i, name in enumerate(names):
            sel = idx == i
            n = int(sel.sum())
            n_pos = int(y[sel].sum()) if n else 0
            unreliable, reason = D.reliability(n, n_pos, n - n_pos, min_n=min_band_n)
            entry[name] = {"n": n, "n_pos": n_pos, "n_neg": n - n_pos,
                           "positive_rate": float(y[sel].mean()) if n else float("nan"),
                           "unreliable": bool(unreliable),
                           "unreliable_reason": reason}
        entry["_unbanded"] = int((idx < 0).sum())
        out[split] = entry
    return {"min_band_n": int(min_band_n),
            "bands": {n: [lo, hi] for n, lo, hi in D.resolve_bands(bands)},
            "per_split": out}


# --------------------------------------------------------------------------- #
# Phase B -- lag geometry                                                     #
# --------------------------------------------------------------------------- #
def _window_taus(ds: TensorizedFinetuneDataset, n_windows: int, max_pairs: int,
                 seed: int) -> dict:
    """Pairwise lags and within-row spread from a seeded sample of windows.

    ``tau`` is float64 throughout and goes through ``data.spans_to_tau``, the numpy twin of
    ``data.lag_to_tau``; the two share the ``/7`` and ``log1p`` constants, so this cannot
    drift from what the model computes on the GPU. Validity comes from
    ``encoder.build_pair_mask``, the same helper the encoder and pooling use.
    """
    rng = np.random.default_rng(seed)
    idxs = _sample_indices(len(ds), n_windows, seed)
    taus: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    n_pairs_total = 0
    n_rows = 0
    for j in idxs:
        ts = ds[int(j)]["timestamps_days"].astype(np.float64)
        if ts.size < 2:
            n_rows += int(ts.size)
            spreads.append(np.zeros(ts.size))
            continue
        mask = torch.ones(1, ts.size, dtype=torch.bool)
        pair = build_pair_mask(mask)[0].numpy()          # all-True here: no padding in a
        d = np.abs(ts[:, None] - ts[None, :])            # single un-padded window
        tau_mat = spans_to_tau(d)
        iu = np.triu_indices(ts.size, k=1)
        valid = pair[iu]
        dv = tau_mat[iu][valid]
        n_pairs_total += int(dv.size)
        if dv.size > max_pairs:
            dv = rng.choice(dv, max_pairs, replace=False)
        taus.append(dv)
        big = np.where(pair, tau_mat, -np.inf).max(axis=1)
        small = np.where(pair, tau_mat, np.inf).min(axis=1)
        spreads.append(big - small)
        n_rows += int(ts.size)
    return {
        "tau": np.concatenate(taus) if taus else np.zeros(0),
        "spread": np.concatenate(spreads) if spreads else np.zeros(0),
        "n_windows": int(len(idxs)),
        "n_pairs_total_in_sampled_windows": int(n_pairs_total),
        "n_rows": int(n_rows),
    }


def b1_domain(tau: np.ndarray, tau_max: float, s: int) -> dict:
    """B1. tau_tilde range, occupancy, clamp rate, Gram condition, per task."""
    tt = _tau_tilde(tau, tau_max)
    cond = D.gram_condition_numbers(tau, s, tau_max)
    return {
        "n_pairs_sampled": int(tau.size),
        "tau_max": float(tau_max),
        "tau_max_source": "checkpoint buffer (frozen at pretraining, D8 / INV-TMAX)",
        "tau_min": float(tau.min()) if tau.size else float("nan"),
        "tau_max_observed": float(tau.max()) if tau.size else float("nan"),
        "tau_tilde_min": float(np.clip(tt, -1, 1).min()) if tt.size else float("nan"),
        "tau_tilde_max": float(np.clip(tt, -1, 1).max()) if tt.size else float("nan"),
        "tau_tilde_quantiles": _q(np.clip(tt, -1, 1)),
        "occupancy_fraction_of_domain": _occupancy(tt),
        "clamp_rate": float(((tt < -1.0) | (tt > 1.0)).mean()) if tt.size else float("nan"),
        "chebyshev_cond_no_constant": cond["chebyshev_no_constant"],
        "chebyshev_cond_with_constant": cond["chebyshev_with_constant"],
        "monomial_cond_no_constant": cond["monomial_no_constant"],
        "mimic_reference_chebyshev_cond_no_constant": 15.1,
        "readme_5c_reference_occupancy": 0.020,
    }


def b2_spread(spread: np.ndarray) -> dict:
    """B2. Within-row tau spread -- what actually decides whether kernel SHAPE can matter."""
    if spread.size == 0:
        return {"n_rows": 0}
    return {
        "n_rows": int(spread.size),
        "median": float(np.median(spread)),
        "p10": float(np.percentile(spread, 10)),
        "p90": float(np.percentile(spread, 90)),
        "max": float(spread.max()),
        "frac_rows_below_0.1": float((spread < 0.1).mean()),
        "mimic_pretrain_reference": {"median": 4.46, "frac_rows_below_0.1": 0.015,
                                     "masking": "padding_only"},
        "note": "softmax ignores a per-row constant, so a row whose keys all sit at the "
                "same lag gives the kernel nothing to discriminate regardless of alpha",
    }


def b4_unclipped(events_parquet: Path, ds: TensorizedFinetuneDataset, n_windows: int,
                 max_pairs: int, seed: int, tau_max: float, s: int,
                 obs_window_days: float) -> dict | None:
    """B4. The counterfactual: is M1 the 24 h cohort window, or the frozen ``tau_max``?

    The shipped shards keep ``[t0, t0 + OBS_WINDOW_DAYS]`` of each admission, where ``t0``
    is the admission's first event. "Unclipped" can mean two different things and they give
    different answers, so both are measured:

    * **backward** -- the *pre-index history*: everything the same subject has strictly
      before ``t0``. Widening here is free of outcome leakage, which is why it is the fix
      that would actually be available. Its size is reported first, because if PIC subjects
      have no pre-index history the option does not exist regardless of what it would buy.
    * **forward** -- ``[t0, t0 + W]`` for a sweep of ``W``, ending at the whole stream.
      This is the observation-window trade-off as a curve, and it is a **geometry
      measurement only**: extending forward past the index time leaks the outcome for
      ``mortality`` and ``los_gt7`` and would not be a legitimate cohort definition.
    """
    if not Path(events_parquet).exists():
        return None
    idxs = _sample_indices(len(ds), n_windows, seed)
    want: dict[int, list[int]] = collections.defaultdict(list)
    for j in idxs:
        item = ds[int(j)]
        want[int(item["subject_id"])].append(int(item["hadm_id"]))
    subjects = np.array(sorted(want), dtype=np.int64)

    import pyarrow.parquet as pq
    tbl = pq.read_table(events_parquet,
                        columns=["subject_id", "hadm_id", "timestamp_days"])
    sid = tbl.column("subject_id").to_numpy()
    hid = tbl.column("hadm_id").to_numpy()
    tds = tbl.column("timestamp_days").to_numpy().astype(np.float64)
    del tbl
    keep = np.isin(sid, subjects)
    sid, hid, tds = sid[keep], hid[keep], tds[keep]
    order = np.lexsort((tds, sid))
    sid, hid, tds = sid[order], hid[order], tds[order]
    starts = np.searchsorted(sid, subjects, side="left")
    ends = np.searchsorted(sid, subjects, side="right")
    by_subject = {int(v): (int(a), int(b)) for v, a, b in zip(subjects, starts, ends)}

    rng = np.random.default_rng(seed)

    def geometry(windows: list[np.ndarray]) -> dict:
        taus: list[np.ndarray] = []
        spreads: list[np.ndarray] = []
        counts = [int(w.size) for w in windows]
        for w in windows:
            if w.size < 2:
                spreads.append(np.zeros(max(1, w.size)))
                continue
            d = np.abs(w[:, None] - w[None, :])
            tau_mat = spans_to_tau(d)
            dv = tau_mat[np.triu_indices(w.size, k=1)]
            if dv.size > max_pairs:
                dv = rng.choice(dv, max_pairs, replace=False)
            taus.append(dv)
            spreads.append(tau_mat.max(axis=1) - tau_mat.min(axis=1))
        tau = np.concatenate(taus) if taus else np.zeros(0)
        spr = np.concatenate(spreads) if spreads else np.zeros(0)
        entry = b1_domain(tau, tau_max, s)
        entry["within_row_spread"] = b2_spread(spr)
        entry["events_per_sequence_median"] = (float(np.median(counts)) if counts
                                               else float("nan"))
        entry["events_per_sequence_max"] = (float(np.max(counts)) if counts
                                            else float("nan"))
        entry["span_days_median"] = (float(np.median([w.max() - w.min() for w in windows
                                                      if w.size > 1]))
                                     if any(w.size > 1 for w in windows) else float("nan"))
        return entry

    # ---- the admissions, resolved once ------------------------------------- #
    admissions: list[tuple[np.ndarray, float]] = []   # (subject event times, t0)
    pre_counts: list[int] = []
    pre_spans: list[float] = []
    for subject, hadms in want.items():
        a, b = by_subject.get(subject, (0, 0))
        if b <= a:
            continue
        st, sh = tds[a:b], hid[a:b]
        for hadm in hadms:
            sel = sh == hadm
            if not sel.any():
                continue
            t0 = float(st[sel].min())
            admissions.append((st, t0))
            pre = st[st < t0]
            pre_counts.append(int(pre.size))
            pre_spans.append(float(t0 - pre.min()) if pre.size else 0.0)

    forward: dict[str, dict] = {}
    for W in WINDOW_SWEEP_DAYS:
        key = "inf" if np.isinf(W) else f"{W:g}"
        wins = [st[(st >= t0) & (st <= (np.inf if np.isinf(W) else t0 + W))]
                for st, t0 in admissions]
        entry = geometry(wins)
        entry["window_days"] = None if np.isinf(W) else float(W)
        forward[key] = entry

    backward = geometry([st[st <= t0 + obs_window_days] for st, t0 in admissions])
    backward["definition"] = ("pre-index history plus the shipped 24 h window: every event "
                              "of the same subject at or before t0 + OBS_WINDOW_DAYS")

    clipped = forward[f"{WINDOW_SWEEP_DAYS[0]:g}"]
    unclipped = forward["inf"]
    ratio = (unclipped["occupancy_fraction_of_domain"]
             / max(1e-12, clipped["occupancy_fraction_of_domain"]))
    pre = np.asarray(pre_counts, dtype=np.float64)
    frac_with_pre = float((pre > 0).mean()) if pre.size else float("nan")
    if ratio >= 2.0:
        verdict = (f"M1 is dominated by the 24 h cohort window: the full admission stream "
                   f"occupies {ratio:.1f}x more of [-1, 1] than the shipped window")
    else:
        verdict = (f"M1 is NOT explained by the 24 h window: the full admission stream "
                   f"occupies {ratio:.2f}x the shipped occupancy, so the frozen tau_max is "
                   f"the binding constraint")
    return {
        "events_parquet": str(events_parquet),
        "n_cohort_examples_sampled": int(len(idxs)),
        "n_admissions_resolved": int(len(admissions)),
        "n_subjects": int(len(want)),
        "obs_window_days_shipped": float(obs_window_days),
        "pre_index_history": {
            "fraction_of_admissions_with_any": frac_with_pre,
            "events_median": float(np.median(pre)) if pre.size else float("nan"),
            "events_max": float(pre.max()) if pre.size else float("nan"),
            "lookback_span_days_max": float(np.max(pre_spans)) if pre_spans else float("nan"),
            "note": "widening BACKWARD is the leakage-free option; if this is ~0 the option "
                    "does not exist for this corpus",
        },
        "backward_pre_index_plus_window": backward,
        "forward_sweep": forward,
        "forward_sweep_caveat": ("geometry only. Extending past t0 + OBS_WINDOW_DAYS leaks "
                                 "the outcome for mortality and los_gt7 and is not a "
                                 "legitimate cohort definition; it is measured to show what "
                                 "the lag geometry would look like, not to propose it."),
        "occupancy_unclipped_over_clipped": float(ratio),
        "verdict": verdict,
    }


def b5_options(pic_tau: dict[str, np.ndarray], mimic_tau: np.ndarray, s: int,
               tau_max_frozen: float) -> dict:
    """B5. The numbers each README 5c option needs, with nothing chosen."""
    option1: dict[str, dict] = {}
    for tm in TAU_MAX_SWEEP:
        row: dict[str, Any] = {"tau_max": float(tm)}
        if mimic_tau.size:
            tt_m = _tau_tilde(mimic_tau, tm)
            cond_m = D.gram_condition_numbers(mimic_tau, s, tm)
            row["mimic"] = {
                "clamp_rate": float(((tt_m < -1) | (tt_m > 1)).mean()),
                "occupancy": _occupancy(tt_m),
                "chebyshev_cond_no_constant": cond_m["chebyshev_no_constant"],
            }
        row["pic"] = {}
        for task, tau in pic_tau.items():
            if tau.size == 0:
                continue
            tt_p = _tau_tilde(tau, tm)
            cond_p = D.gram_condition_numbers(tau, s, tm)
            row["pic"][task] = {
                "clamp_rate": float(((tt_p < -1) | (tt_p > 1)).mean()),
                "occupancy": _occupancy(tt_p),
                "chebyshev_cond_no_constant": cond_p["chebyshev_no_constant"],
            }
        option1[f"{tm:g}"] = row

    option2 = {}
    for task, tau in pic_tau.items():
        if tau.size == 0:
            continue
        tt = _tau_tilde(tau, tau_max_frozen)
        option2[task] = _shifted_chebyshev_cond(tt, s)
    return {
        "option_1_rescale_shared_domain": {
            "sweep": option1,
            "cost": ("tau_max is a persistent buffer in all four checkpoints. Changing it "
                     "invalidates every one of them and requires re-pretraining: alpha is "
                     "defined against the domain, so the learned coefficients do not "
                     "transfer to a new tau_max even numerically."),
        },
        "option_2_shifted_chebyshev_on_pic_subinterval": {
            "per_task": option2,
            "cost": ("coefficient comparability is lost. INV-TMAX exists so that alpha_k "
                     "means the same thing at pretrain and fine-tune; a basis shifted onto "
                     "the PIC sub-interval makes alpha_k a coefficient of a DIFFERENT "
                     "polynomial, so the pretrained alpha_base and Delta-alpha are no longer "
                     "the initialization of the thing being fine-tuned. Every pretrain-to-"
                     "finetune comparison of alpha, and the whole 'the age pathway is "
                     "trained at pretraining' claim, would have to be dropped."),
        },
        "option_3_accept_and_report": {
            "minimum_numbers_for_a_negative_transfer_result": [
                "per task: tau_tilde occupancy and Chebyshev Gram condition under the frozen "
                "tau_max (B1)",
                "per task: within-row tau spread median and the fraction of rows below 0.1 "
                "(B2), beside the MIMIC pretraining values",
                "per arm: the equal-norm headroom probe on real PIC batches (B3), beside the "
                "same probe on MIMIC",
                "the unclipped-vs-clipped occupancy ratio (B4), to establish whether the "
                "result is about PIC or about the cohort window",
                "per arm: ||Delta-alpha(a)||_2 over 0-18 y before and after fine-tuning, to "
                "show whether the pathway can move at all in the pediatric range",
                "AUPRC with patient-level bootstrap CIs per arm and the paired kernel-minus-"
                "vanilla delta, so 'no transfer' is a CI that contains zero rather than an "
                "absence of significance",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# B3 / D2 -- the loaded backbone on real PIC batches                          #
# --------------------------------------------------------------------------- #
def _pic_batches(ds: TensorizedFinetuneDataset, n_batches: int, batch_size: int,
                 race_encoding: str, remap: np.ndarray | None, unk_out: int | None,
                 seed: int) -> list[dict]:
    """Deterministic PIC batches, optionally reindexed into the MIMIC vocabulary."""
    idxs = _sample_indices(len(ds), n_batches * batch_size, seed)
    batches = []
    for b in range(n_batches):
        chunk = idxs[b * batch_size:(b + 1) * batch_size]
        if chunk.size == 0:
            break
        items = []
        for j in chunk:
            item = dict(ds[int(j)])
            if remap is not None:
                item["code_indices"] = remap[item["code_indices"]]
                item["unk_vocab_index"] = int(unk_out)
            items.append(item)
        batches.append(finetune_collate(items, race_encoding=race_encoding))
    return batches


def _build_from_ckpt(ckpt: dict, arm: str, num_codes: int, table: torch.Tensor,
                     tau_max: float) -> tuple[DKMModel, dict]:
    """A classification-head model at the checkpoint's hyperparameters, backbone loaded.

    Constructor arguments come from the checkpoint's own ``config.model`` block, exactly as
    ``train_finetune.main`` builds them, so this probe measures the shipped backbone rather
    than a re-specified one.
    """
    m = ckpt.get("config", {}).get("model", {})
    demo_dim, demo_channels = demo_layout(m.get("race_encoding", "one_hot"))
    std = m["age_standardization"]
    model = DKMModel(
        num_codes=num_codes, embedding_table=table, arm=arm,
        seed=int(ckpt.get("config", {}).get("seed", 0)),
        d_model=m["d_model"], n_layers=m["n_layers"], n_heads=m["n_heads"],
        use_residual=m["use_residual"], use_layernorm=m["use_layernorm"],
        use_ffn=m["use_ffn"], ffn_mult=m.get("ffn_mult", 4), s=m["s"], tau_max=tau_max,
        age_M=m["fourier"]["M"], age_p_min=m["fourier"]["p_min"],
        age_p_max=m["fourier"]["p_max"], age_hidden=m["age_hidden"],
        gen_final_bias=m["gen_final_bias"], center_delta_alpha=m["center_delta_alpha"],
        demo_dim=demo_dim, demo_channels=demo_channels,
        race_encoding=m.get("race_encoding", "one_hot"), demo_hidden=m["demo_hidden"],
        age_mean=float(std["mean"]), age_sd=float(std["sd"]), task="classification",
    )
    state = dict(ckpt["model_state_dict"])
    substituted = False
    if ("embedding_table" in state
            and tuple(state["embedding_table"].shape) != tuple(table.shape)):
        # Option (b): the checkpoint's frozen MIMIC table cannot be restored into a model
        # whose table is [2200, 1024]. Rather than relax load_backbone -- whose whole job is
        # to refuse a partial backbone transfer -- the substitution is made explicit here and
        # recorded. This is a real consequence of DECISION D3, not a detail of the probe.
        state["embedding_table"] = model.embedding_table.detach().clone()
        substituted = True
    info = load_backbone(model, state, arm)
    info["embedding_table_substituted_not_restored"] = substituted
    if substituted:
        info["inv_frozen_note"] = (
            "INV-FROZEN's 'restored from the checkpoint rather than rebuilt' clause does "
            "not hold for embedding_table under option (b); the table is a different "
            "matrix over a different vocabulary.")
    return model.eval(), info


def b3_headroom(ckpt_map: dict[str, Path], ds: TensorizedFinetuneDataset, *,
                mimic_table: torch.Tensor | None, pic_table: torch.Tensor | None,
                remap: np.ndarray | None, mimic_num_codes: int, pic_num_codes: int,
                n_batches: int, batch_size: int, seed: int, bands,
                pediatric_grid) -> dict:
    """B3 + the D2 per-checkpoint block: headroom and ||Delta-alpha|| on real PIC data.

    Two vocabulary configurations are run because they are not the same measurement. Under
    reindexing the frozen MIMIC embedding table is preserved and the probe reads the shipped
    backbone; under the PIC table the input distribution to QK changes, and since LayerNorm
    on the attention input is what gives the kernel its authority (README section 5), so does
    the headroom.
    """
    out: dict[str, dict] = {}
    for arm, path in sorted(ckpt_map.items()):
        if not Path(path).exists():
            out[arm] = {"checkpoint": str(path), "present": False}
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        tau_max = float(ckpt["tau_max"])
        s = int(ckpt["config"]["model"]["s"])
        race_encoding = ckpt["config"]["model"].get("race_encoding", "one_hot")
        entry: dict[str, Any] = {
            "checkpoint": str(Path(path).resolve()),
            "present": True,
            "arm_in_checkpoint": ckpt.get("arm"),
            "epoch": ckpt.get("epoch"),
            "tau_max": tau_max,
            "age_standardization": ckpt["config"]["model"]["age_standardization"],
        }
        for label, table, num_codes, rm, unk in (
            ("reindexed_into_mimic_vocab", mimic_table, mimic_num_codes, remap,
             mimic_num_codes),
            ("pic_bge_table", pic_table, pic_num_codes, None, None),
        ):
            if table is None:
                continue
            model, info = _build_from_ckpt(ckpt, arm, num_codes, table, tau_max)
            batches = _pic_batches(ds, n_batches, batch_size, race_encoding, rm, unk, seed)
            with torch.no_grad():
                hr = dict(headroom(model, batches, s))
            ages = torch.cat([b["age_years"][b["attention_mask"]] for b in batches])
            hr["state_dict_load"] = info
            entry[label] = {
                "headroom": hr,
                "delta_alpha_norms_pediatric": D.delta_alpha_norms(
                    model, ages, dense_grid=pediatric_grid),
                "alpha": D.alpha_diagnostics(model, ages, bands=bands),
                "clamp_rate": D.clamp_rates(model),
            }
            del model, batches
        del ckpt
        out[arm] = entry
    return out


# --------------------------------------------------------------------------- #
# printing (diagnostics owns the primitives; the line lists are built here,    #
# exactly as preflight.py does)                                                #
# --------------------------------------------------------------------------- #
def _print_phase_a(report: dict, tasks: list[str], bands) -> None:
    lines = [f"{'task':<22}{'split':<7}{'n_seq':>8}{'n_subj':>8}{'pos':>7}{'pos_rate':>10}"
             f"{'ev/seq p50':>12}{'span_max_d':>12}", "  " + "-" * 88]
    for task in tasks:
        a2 = report[task]["A2_cohort"]["per_split"]
        a3 = report[task]["A3_length_window"]["per_split"]
        for split in SPLITS:
            c, l = a2.get(split, {}), a3.get(split, {})
            if not c.get("present"):
                lines.append(f"{task:<22}{split:<7}   (absent)")
                continue
            lines.append(
                f"{task:<22}{split:<7}{c['n_sequences']:>8}{c['n_unique_subjects']:>8}"
                f"{c['n_positive']:>7}{c['positive_rate']:>10.4f}"
                f"{l['events_per_sequence']['median']:>12.0f}"
                f"{l['span_days']['max']:>12.4f}")
    lines.append("  " + "-" * 88)
    for task in tasks:
        a2 = report[task]["A2_cohort"]
        ok = "disjoint" if a2["patient_level_disjoint"] else \
            f"*** {a2['n_subjects_in_more_than_one_split']} SUBJECTS IN >1 SPLIT ***"
        lines.append(f"{task:<22} patient-level split: {ok}   smallest split "
                     f"{a2['smallest_split']} holds {a2['positives_in_smallest_split']} "
                     f"positives")
    D.print_block("A2/A3  cohort sizes, positives, sequence length and window span", lines)

    lines = []
    for task in tasks:
        a3 = report[task]["A3_length_window"]
        lines.append(f"{task:<22} {a3['window_verdict']}")
    lines += ["", "This is the single most consequential number in the audit: it is what puts "
                  "every", "PIC lag below log1p(1/7) = 0.133 and therefore every tau_tilde "
                  "within 2% of -1."]
    D.print_block("A3  observation window  [MEASURE]", lines)

    a1 = report[tasks[0]]["A1_schema"]["per_split"]["train"]
    lines = [f"fields present ({len(a1['fields'])}):"]
    for k, v in sorted(a1["fields"].items()):
        lines.append(f"    {k:<22} {v['dtype']:<10} {tuple(v['shape'])}")
    lines += [
        "",
        f"required by model_new, missing : {a1['missing_required'] or 'none'}",
        f"optional, missing              : {a1['missing_optional'] or 'none'}",
        f"present but never consumed     : {a1['present_but_unconsumed'] or 'none'}  "
        f"(legacy finetune/dataset.py supplied {list(LEGACY_ONLY_FIELDS)})",
        "",
        "derived by model_new rather than stored:",
    ]
    for k, v in DERIVED_BATCH_KEYS.items():
        lines.append(f"    {k:<16} {v}")
    D.print_block(f"A1  shard schema ({tasks[0]} / train; identical across tasks)", lines)

    lines = [f"{'family':<10}{'PIC codes':>11}{'in MIMIC':>11}{'fraction':>10}",
             "  " + "-" * 42]
    a4 = report[tasks[0]]["A4_vocab"]
    for fam, v in sorted(a4["by_family"].items()):
        lines.append(f"{fam:<10}{v['n_pic_codes']:>11}{v['n_in_mimic']:>11}"
                     f"{v['fraction_in_mimic']:>10.4f}")
    lines += ["  " + "-" * 42,
              f"{'TOTAL':<10}{a4['n_pic_codes']:>11}{a4['n_pic_codes_in_mimic']:>11}"
              f"{a4['fraction_pic_types_in_mimic']:>10.4f}",
              "",
              "token-level UNK rate under option (a), reindex into the MIMIC vocab:"]
    for task in tasks:
        t = report[task]["A4_vocab"]["token_level"]
        cells = "  ".join(f"{s}={t[s]['token_unk_rate_under_reindexing']:.4f}"
                          for s in SPLITS if t.get(s, {}).get("present"))
        lines.append(f"    {task:<22} {cells}")
    lines += ["", a4["supported_by_current_code"]]
    D.print_block("A4  vocabulary transfer  [MEASURE -> DECISION D3]", lines)

    a5 = report[tasks[0]]["A5_demographics"]
    lines = [
        f"race value counts (all tasks, all splits) : {a5['race_value_counts']}",
        f"all PIC race == UNKNOWN                    : {a5['all_pic_race_is_unknown']}",
        f"one-hot ordering source                    : {a5['race_ordering_source']}",
        f"race_UNKNOWN demographic channel index     : "
        f"{a5['race_unknown_demo_channel_index']}  (of demo_dim = 2 + {a5['n_race']})",
        f"ordering agrees with data.RACE_LABELS      : "
        f"{a5['race_ordering_agrees_with_data.RACE_LABELS']}",
        "",
        f"age constants (frozen, from the checkpoint): "
        f"mean={a5['standardized_age']['constants']['mean']:.4f} "
        f"sd={a5['standardized_age']['constants']['sd']:.4f}",
    ]
    for task in tasks:
        z = report[task]["A5_demographics"]["standardized_age"]
        a = report[task]["A5_demographics"]["age_years_at_last_event"]
        lines.append(f"    {task:<22} age y  min={a['min']:.3f} med={a['median']:.3f} "
                     f"max={a['max']:.2f}    standardized  min={z['min']:.3f} "
                     f"med={z['median']:.3f} max={z['max']:.3f}")
    lines += ["", "README expects roughly -3.8 to -2.7. Reported, never re-standardized: "
                  "changing", "the constants changes the meaning of every learned demo_proj "
                  "weight (INV-AGESTD)."]
    D.print_block("A5  demographics under the frozen MIMIC layout  [MEASURE]", lines)

    names = D.band_names(bands)
    lines = [f"{'task':<22}{'split':<7}" + "".join(f"{n:>12}" for n in names),
             "  " + "-" * (29 + 12 * len(names))]
    for task in tasks:
        for split in SPLITS:
            e = report[task]["A6_age_bands"]["per_split"].get(split, {})
            if not e or e.get("present") is False:
                continue
            cells = []
            for n in names:
                b = e[n]
                cells.append(f"{b['n']}{'*' if b['unreliable'] and b['n'] else ''}")
            lines.append(f"{task:<22}{split:<7}" + "".join(f"{c:>12}" for c in cells))
    mn = report[tasks[0]]["A6_age_bands"]["min_band_n"]
    lines += ["  " + "-" * (29 + 12 * len(names)),
              f"* below --min_band_n = {mn}; every metric for such a band is reported null "
              f"with unreliable=true"]
    D.print_block("A6  pediatric age-band coverage (age at last valid event)", lines)


def _print_phase_b(report: dict, tasks: list[str], b5: dict, mimic_ref: dict | None) -> None:
    lines = [f"{'task':<22}{'n_pairs':>11}{'tt_min':>9}{'tt_max':>9}{'occupancy':>11}"
             f"{'clamp':>9}{'cheb cond':>13}", "  " + "-" * 84]
    for task in tasks:
        b = report[task]["B1_domain"]
        lines.append(f"{task:<22}{b['n_pairs_sampled']:>11,}{b['tau_tilde_min']:>9.4f}"
                     f"{b['tau_tilde_max']:>9.4f}"
                     f"{b['occupancy_fraction_of_domain'] * 100:>10.2f}%"
                     f"{b['clamp_rate']:>9.2e}{b['chebyshev_cond_no_constant']:>13.3e}")
    if mimic_ref:
        lines.append(f"{'MIMIC (val, measured)':<22}{mimic_ref['n_pairs_sampled']:>11,}"
                     f"{mimic_ref['tau_tilde_min']:>9.4f}{mimic_ref['tau_tilde_max']:>9.4f}"
                     f"{mimic_ref['occupancy_fraction_of_domain'] * 100:>10.2f}%"
                     f"{mimic_ref['clamp_rate']:>9.2e}"
                     f"{mimic_ref['chebyshev_cond_no_constant']:>13.3e}")
    lines += ["  " + "-" * 84,
              "README 5c reports 2.0% occupancy and cond 5.7e16 for heart_malformations "
              "against 15.1 on MIMIC.",
              "Chebyshev polynomials are near-orthogonal on the WHOLE [-1, 1]; on a sliver "
              "near -1 they",
              "collapse toward a common affine shape and the Gram matrix goes singular to "
              "float precision."]
    D.print_block("B1  M1: tau_tilde domain under the frozen tau_max  [MEASURE]", lines)

    lines = [f"{'task':<22}{'n_rows':>10}{'median':>10}{'p10':>10}{'p90':>10}"
             f"{'frac < 0.1':>12}", "  " + "-" * 74]
    for task in tasks:
        b = report[task]["B2_spread"]
        lines.append(f"{task:<22}{b['n_rows']:>10,}{b['median']:>10.4f}{b['p10']:>10.4f}"
                     f"{b['p90']:>10.4f}{b['frac_rows_below_0.1']:>12.4f}")
    lines += [f"{'MIMIC pretrain (README)':<22}{'--':>10}{4.46:>10.4f}{'--':>10}{'--':>10}"
              f"{0.015:>12.4f}",
              "  " + "-" * 74,
              "Within-row spread, not the marginal tau histogram, is what decides whether "
              "kernel SHAPE",
              "can matter: softmax is invariant to a per-row constant and the query age is "
              "fixed within a row."]
    D.print_block("B2  within-row tau spread  [MEASURE]", lines)

    hr = {t: report[t].get("B3_headroom") for t in tasks}
    any_hr = next((v for v in hr.values() if v), None)
    if any_hr:
        lines = [f"{'arm':<18}{'vocab mode':<28}{'max|dlogit|':>13}{'max/logit sd':>14}",
                 "  " + "-" * 74]
        for task in tasks:
            if not hr[task]:
                continue
            lines.append(f"  task: {task}")
            for arm, e in sorted(hr[task].items()):
                if not e.get("present"):
                    lines.append(f"{arm:<18}{'(checkpoint absent)':<28}")
                    continue
                for mode in ("reindexed_into_mimic_vocab", "pic_bge_table"):
                    if mode not in e:
                        continue
                    h = e[mode]["headroom"]
                    lines.append(f"{arm:<18}{mode:<28}{h['max_abs_delta_logit']:>13.6f}"
                                 f"{h['max_delta_over_logit_sd']:>14.4f}")
        lines += ["  " + "-" * 74,
                  "Equal-norm probe: a decaying kernel [-2,0,-1,0,0] against a growing one "
                  "[+2,0,+1,0,0].",
                  "README section 5 measured 0.0990 / 1.44 on MIMIC batches at init with "
                  "the standard block;",
                  "preflight's stored PIC reference is 0.0059. This is the direct answer to "
                  "'can the kernel do",
                  "anything on this data'."]
        D.print_block("B3  equal-norm headroom on real PIC batches, per arm  [MEASURE]",
                      lines)

    for task in tasks:
        b4 = report[task].get("B4_unclipped")
        if not b4:
            continue
        pre = b4["pre_index_history"]
        lines = [
            "BACKWARD -- pre-index history, the leakage-free direction:",
            f"    admissions with any event before t0 : {pre['fraction_of_admissions_with_any']:.4f}",
            f"    pre-index events, median / max      : {pre['events_median']:.0f} / "
            f"{pre['events_max']:.0f}",
            f"    lookback span, max                  : {pre['lookback_span_days_max']:.3f} d",
            f"    occupancy with pre-index + window   : "
            f"{b4['backward_pre_index_plus_window']['occupancy_fraction_of_domain'] * 100:.2f}%"
            f"   cheb cond = "
            f"{b4['backward_pre_index_plus_window']['chebyshev_cond_no_constant']:.3e}",
            "",
            "FORWARD -- [t0, t0 + W]. GEOMETRY ONLY: past t0 + 1 d this leaks the outcome",
            "for mortality and los_gt7 and is not a legitimate cohort definition.",
            f"{'window':<10}{'ev/seq p50':>12}{'span p50 d':>12}{'occupancy':>11}"
            f"{'cheb cond':>13}{'spread p50':>12}{'frac<0.1':>10}",
            "  " + "-" * 81]
        for key in [f"{w:g}" if not np.isinf(w) else "inf" for w in WINDOW_SWEEP_DAYS]:
            e = b4["forward_sweep"][key]
            lines.append(
                f"{key + ' d':<10}{e['events_per_sequence_median']:>12.0f}"
                f"{e['span_days_median']:>12.3f}"
                f"{e['occupancy_fraction_of_domain'] * 100:>10.2f}%"
                f"{e['chebyshev_cond_no_constant']:>13.3e}"
                f"{e['within_row_spread'].get('median', float('nan')):>12.4f}"
                f"{e['within_row_spread'].get('frac_rows_below_0.1', float('nan')):>10.4f}")
        lines += ["  " + "-" * 81,
                  f"unclipped / clipped occupancy ratio : "
                  f"{b4['occupancy_unclipped_over_clipped']:.2f}x",
                  f"verdict: {b4['verdict']}",
                  "",
                  "'1 d' reproduces the shipped shards."]
        D.print_block(f"B4  unclipped counterfactual and window sweep -- {task}  [MEASURE]",
                      lines)

    o1 = b5["option_1_rescale_shared_domain"]["sweep"]
    lines = [f"{'tau_max':<10}{'MIMIC clamp':>13}{'MIMIC cond':>13}{'MIMIC occ':>11}"
             f"{'PIC occ':>10}{'PIC cond':>13}", "  " + "-" * 72]
    for tm_key, row in o1.items():
        m = row.get("mimic", {})
        p = row["pic"].get(tasks[0], {})
        lines.append(
            f"{tm_key:<10}{m.get('clamp_rate', float('nan')):>13.4f}"
            f"{m.get('chebyshev_cond_no_constant', float('nan')):>13.3e}"
            f"{m.get('occupancy', float('nan')) * 100:>10.1f}%"
            f"{p.get('occupancy', float('nan')) * 100:>9.1f}%"
            f"{p.get('chebyshev_cond_no_constant', float('nan')):>13.3e}")
    lines += ["  " + "-" * 72,
              f"(PIC column shown for {tasks[0]}; every task is in audit.json)",
              b5["option_1_rescale_shared_domain"]["cost"],
              "",
              "OPTION 2  shifted Chebyshev on the PIC sub-interval:"]
    for task, v in b5["option_2_shifted_chebyshev_on_pic_subinterval"]["per_task"].items():
        lines.append(f"    {task:<22} sub-interval [{v['sub_interval'][0]:.4f}, "
                     f"{v['sub_interval'][1]:.4f}]  cond = "
                     f"{v['chebyshev_no_constant']:.3e}")
    lines += ["    " + b5["option_2_shifted_chebyshev_on_pic_subinterval"]["cost"],
              "", "OPTION 3  accept and report. The smallest sufficient set of numbers:"]
    for item in b5["option_3_accept_and_report"][
            "minimum_numbers_for_a_negative_transfer_result"]:
        lines.append(f"    - {item}")
    D.print_block("B5  the three README 5c options, with numbers  [DECISION D1]", lines)


def _print_decisions(report: dict, tasks: list[str], d2: dict, b5: dict) -> None:
    lines = [
        "D1  tau_max / M1.  BLOCKING.  Phase B supplies the numbers; B4 says whether this is",
        "    a data artifact (fix the cohort window) or a design fault (fix tau_max or the",
        "    basis). Nothing here has been changed.",
        "",
        "D2  which backbone each arm fine-tunes from.",
        f"{'arm':<18}{'pretrain run':<38}{'best val BCE':>14}{'ep':>5}"
        f"{'PIC max|dlogit|':>17}{'||da|| 0-18y max':>18}",
        "  " + "-" * 108,
    ]
    for arm in ARMS:
        e = d2.get(arm, {})
        hr = (e.get("pic_headroom") or {}).get("pic_bge_table") \
            or (e.get("pic_headroom") or {}).get("reindexed_into_mimic_vocab") or {}
        da = (e.get("delta_alpha_0_18y") or {}).get("pic_bge_table") \
            or (e.get("delta_alpha_0_18y") or {}).get("reindexed_into_mimic_vocab") or {}
        da_max = max((v["dense_uniform_grid"]["max"] for v in da.values()
                      if isinstance(v, dict) and "dense_uniform_grid" in v),
                     default=float("nan"))
        lines.append(f"{arm:<18}{str(e.get('run_id', '--'))[:36]:<38}"
                     f"{e.get('best_val_bce', float('nan')):>14.6f}"
                     f"{str(e.get('best_epoch', '--')):>5}"
                     f"{hr.get('max_abs_delta_logit', float('nan')):>17.6f}"
                     f"{da_max:>18.4e}")
    lines += [
        "  " + "-" * 108,
        "    best val BCE is train.json's --val_max_batches training monitor, not "
        "eval_pretrain's full pass.",
        "    max|dlogit| and ||Delta-alpha|| are measured on PIC batches through each arm's "
        "own loaded backbone.",
        "  " + "-" * 86,
        "    arm-matched   : each arm fine-tunes its OWN pretrained backbone. Measures "
        "end-to-end",
        "                    OOD transfer of the mechanism -- the paper's claim -- but "
        "confounds",
        "                    pretraining differences with fine-tune-time behaviour.",
        "    shared-vanilla: every arm fine-tunes the vanilla backbone. Isolates the "
        "fine-tune-time",
        "                    mechanism on an identical backbone, but supports no claim "
        "about what",
        "                    pretraining with the kernel bought.",
        "    Both designs need the same per-arm checkpoint map, which run/finetune.sh now "
        "takes",
        "    (MODE=matched|shared, CKPT_MAP=...). The plumbing is in place either way; the "
        "choice is not made.",
        "",
        "D3  vocabulary and embedding table.",
    ]
    for task in tasks:
        t = report[task]["A4_vocab"]["token_level"]
        r = t.get("train", {}).get("token_unk_rate_under_reindexing", float("nan"))
        verdict = "NOT VIABLE (> 20%)" if r > 0.20 else "viable"
        lines.append(f"    {task:<22} train token UNK rate under option (a) = {r:.4f}  "
                     f"-> {verdict}")
    a4 = report[tasks[0]]["A4_vocab"]
    lines += [
        f"    PIC code types present in the MIMIC vocab: {a4['n_pic_codes_in_mimic']} of "
        f"{a4['n_pic_codes']} ({a4['fraction_pic_types_in_mimic'] * 100:.1f}%)",
        f"    (a) {a4['option_a_reindex']}",
        f"    (b) {a4['option_b_pic_table']}",
    ]
    D.print_block("decisions to make, with numbers attached  [STOP HERE]", lines)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pic_root", type=Path, default=REPO_ROOT / "data/tensorized/pic")
    p.add_argument("--pic_vocab", type=Path,
                   default=REPO_ROOT / "data/processed/pic/code_vocab_pic.json")
    p.add_argument("--pic_embeddings", type=Path,
                   default=REPO_ROOT / "data/processed/pic/bge_embeddings_pic.pt")
    p.add_argument("--pic_events_dir", type=Path, default=REPO_ROOT / "data/processed/pic")
    p.add_argument("--mimic_vocab", type=Path,
                   default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--mimic_embeddings", type=Path,
                   default=REPO_ROOT / "data/processed/bge_embeddings.pt")
    p.add_argument("--mimic_tensorized_dir", type=Path,
                   default=REPO_ROOT / "data/processed/tensorized_flat")
    p.add_argument("--mimic_split", type=str, default="val")
    p.add_argument("--ckpt_root", type=Path, default=REPO_ROOT / "model_new/run_selected",
                   help="<ckpt_root>/<arm>_s<seed>/best.pt, the per-arm checkpoint map")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", nargs="*", default=list(PIC_TASKS))
    p.add_argument("--split", type=str, default="train",
                   help="split used for the Phase B lag geometry")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--sample_windows", type=int, default=2000)
    p.add_argument("--max_pairs_per_window", type=int, default=3000)
    p.add_argument("--unclipped_windows", type=int, default=600)
    p.add_argument("--probe_batches", type=int, default=6)
    p.add_argument("--probe_batch_size", type=int, default=8)
    p.add_argument("--band_table", choices=sorted(D.BAND_TABLES), default="pediatric")
    p.add_argument("--min_band_n", type=int, default=D.PEDIATRIC_MIN_BAND_N)
    p.add_argument("--mimic_tau_windows", type=int, default=300)
    p.add_argument("--skip_unclipped", action="store_true")
    p.add_argument("--skip_headroom", action="store_true")
    p.add_argument("--skip_mimic_reference", action="store_true")
    p.add_argument("--out", type=Path, default=Path("model_new/run/pic_audit/audit.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t_start = time.time()
    bands = D.resolve_bands(args.band_table)
    tasks = [t for t in args.tasks if (args.pic_root / t).exists()]
    missing_tasks = [t for t in args.tasks if t not in tasks]
    if not tasks:
        D.review_and_exit("pic_audit", [f"no PIC task directories under {args.pic_root}"])

    pic_vocab = load_vocab(args.pic_vocab)
    mimic_vocab = load_vocab(args.mimic_vocab)

    # The per-arm checkpoint map. Everything frozen is read from these, never re-derived.
    ckpt_map = {arm: args.ckpt_root / f"{arm}_s{args.seed}" / "best.pt" for arm in ARMS}
    reference = next((p for p in ckpt_map.values() if p.exists()), None)
    if reference is None:
        D.review_and_exit("pic_audit", [
            f"no checkpoint found under {args.ckpt_root} (looked for "
            f"<arm>_s{args.seed}/best.pt for {list(ARMS)}). tau_max and the age constants "
            f"must come from a checkpoint; this audit refuses to invent them."])
    ref_ckpt = torch.load(reference, map_location="cpu", weights_only=False)
    tau_max = float(ref_ckpt["tau_max"])
    s = int(ref_ckpt["config"]["model"]["s"])
    std = ref_ckpt["config"]["model"]["age_standardization"]
    age_mean, age_sd = float(std["mean"]), float(std["sd"])
    demo_channels = tuple(ref_ckpt["config"]["model"]["demo_channels"])
    race_encoding = ref_ckpt["config"]["model"].get("race_encoding", "one_hot")
    frozen = {
        "reference_checkpoint": str(reference.resolve()),
        "tau_max": tau_max,
        "tau_max_matches_expected": abs(tau_max - EXPECTED_TAU_MAX) < 1e-6,
        "s": s,
        "age_mean": age_mean, "age_sd": age_sd,
        "age_constants_match_expected": (abs(age_mean - EXPECTED_AGE_MEAN) < 1e-4
                                         and abs(age_sd - EXPECTED_AGE_SD) < 1e-4),
        "demo_channels": list(demo_channels),
        "race_encoding": race_encoding,
        "fourier": ref_ckpt["config"]["model"]["fourier"],
    }
    del ref_ckpt

    # PIC -> MIMIC index remap, built once (used by A4's token counts and B3's probe).
    remap = np.full(len(pic_vocab) + 1, len(mimic_vocab), dtype=np.int64)  # default = UNK
    for code, idx in pic_vocab.items():
        j = mimic_vocab.get(str(code))
        if j is not None:
            remap[int(idx)] = int(j)

    report: dict[str, Any] = {}
    pic_tau: dict[str, np.ndarray] = {}
    for task in tasks:
        entry: dict[str, Any] = {
            "A1_schema": a1_schema(args.pic_root, task),
            "A2_cohort": a2_cohort(args.pic_root, task),
            "A3_length_window": a3_length_and_window(args.pic_root, task),
            "A4_vocab": a4_vocab(args.pic_root, task, pic_vocab, mimic_vocab),
            "A5_demographics": a5_demographics(args.pic_root, task, demo_channels,
                                               age_mean, age_sd),
            "A6_age_bands": a6_age_bands(args.pic_root, task, bands, args.min_band_n),
        }
        ds = TensorizedFinetuneDataset(_split_dir(args.pic_root, task, args.split),
                                       max_seq_len=args.max_seq_len)
        sample = _window_taus(ds, args.sample_windows, args.max_pairs_per_window, args.seed)
        pic_tau[task] = sample["tau"]
        entry["B1_domain"] = b1_domain(sample["tau"], tau_max, s)
        entry["B1_domain"]["split"] = args.split
        entry["B1_domain"]["n_windows_sampled"] = sample["n_windows"]
        entry["B1_domain"]["n_pairs_in_sampled_windows"] = \
            sample["n_pairs_total_in_sampled_windows"]
        entry["B2_spread"] = b2_spread(sample["spread"])
        if not args.skip_unclipped:
            entry["B4_unclipped"] = b4_unclipped(
                args.pic_events_dir / f"{args.split}_events.parquet", ds,
                args.unclipped_windows, args.max_pairs_per_window, args.seed, tau_max, s,
                obs_window_days=entry["A3_length_window"]["legacy_obs_window_days"])
        report[task] = entry
        del ds

    # ---- B3 + D2, once, on the smallest task's dataset to keep the pass short ---- #
    d2: dict[str, dict] = {}
    for arm in ARMS:
        path = ckpt_map[arm]
        e: dict[str, Any] = {"checkpoint": str(path), "present": path.exists()}
        if path.exists():
            real = path.resolve()
            cfg_path = real.parent / "config.json"
            train_path = real.parent / "train.json"
            if train_path.exists():
                hist = json.loads(train_path.read_text())
                best = min(hist, key=lambda r: r["val_loss"])
                e.update({"best_val_bce": float(best["val_loss"]),
                          "best_epoch": int(best["epoch"]),
                          "final_val_bce": float(hist[-1]["val_loss"]),
                          "n_epochs": len(hist),
                          "note": "val_loss from train.json is the --val_max_batches "
                                  "training monitor, not eval_pretrain's full-split pass"})
            if cfg_path.exists():
                e["run_id"] = json.loads(cfg_path.read_text()).get("run_id")
        d2[arm] = e

    if not args.skip_headroom:
        probe_task = tasks[0]
        ds = TensorizedFinetuneDataset(_split_dir(args.pic_root, probe_task, args.split),
                                       max_seq_len=args.max_seq_len)
        mimic_table = None
        pic_table = None
        if args.mimic_embeddings.exists():
            obj = torch.load(args.mimic_embeddings, map_location="cpu")
            mimic_table = obj["embeddings"] if isinstance(obj, dict) else obj
        if args.pic_embeddings.exists():
            obj = torch.load(args.pic_embeddings, map_location="cpu")
            pic_table = obj["embeddings"] if isinstance(obj, dict) else obj
        hr = b3_headroom(ckpt_map, ds, mimic_table=mimic_table, pic_table=pic_table,
                         remap=remap, mimic_num_codes=len(mimic_vocab),
                         pic_num_codes=len(pic_vocab), n_batches=args.probe_batches,
                         batch_size=args.probe_batch_size, seed=args.seed, bands=bands,
                         pediatric_grid=D.PEDIATRIC_DENSE_AGE_GRID)
        report[probe_task]["B3_headroom"] = hr
        for arm, v in hr.items():
            if v.get("present"):
                d2[arm]["pic_headroom"] = {
                    m: v[m]["headroom"] for m in
                    ("reindexed_into_mimic_vocab", "pic_bge_table") if m in v}
                d2[arm]["delta_alpha_0_18y"] = {
                    m: v[m]["delta_alpha_norms_pediatric"] for m in
                    ("reindexed_into_mimic_vocab", "pic_bge_table") if m in v}
        del ds, mimic_table, pic_table

    # ---- the MIMIC reference lag distribution, for B5 option 1 ------------------- #
    mimic_tau = np.zeros(0)
    mimic_ref = None
    if not args.skip_mimic_reference:
        split_dir = args.mimic_tensorized_dir / args.mimic_split
        if list(split_dir.glob("shard_*.npz")):
            from model_new.data import TensorizedPretrainDataset, sample_empirical_taus
            mds = TensorizedPretrainDataset(split_dir, args.mimic_vocab,
                                            max_seq_len=args.max_seq_len)
            mimic_tau = sample_empirical_taus(
                mds, n_examples=args.mimic_tau_windows,
                max_pairs_per_example=args.max_pairs_per_window, seed=args.seed)
            mimic_ref = b1_domain(mimic_tau, tau_max, s)
            mimic_ref["split"] = args.mimic_split
            del mds

    b5 = b5_options(pic_tau, mimic_tau, s, tau_max)

    audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_clock_s": time.time() - t_start,
        "trains_nothing": True,
        "frozen_constants_from_checkpoint": frozen,
        "band_table": {"name": args.band_table,
                       "bands": {n: [lo, hi] for n, lo, hi in bands},
                       "min_band_n": args.min_band_n},
        "sampling": {"seed": args.seed, "split": args.split,
                     "sample_windows": args.sample_windows,
                     "max_pairs_per_window": args.max_pairs_per_window,
                     "unclipped_windows": args.unclipped_windows,
                     "probe_batches": args.probe_batches,
                     "probe_batch_size": args.probe_batch_size},
        "tasks": tasks,
        "tasks_absent": missing_tasks,
        "per_task": report,
        "B5_options": b5,
        "mimic_reference_domain": mimic_ref,
        "D2_checkpoints": d2,
        "checkpoint_map": {a: str(p) for a, p in ckpt_map.items()},
    }
    D.write_json(args.out, audit)

    D.print_block("PIC fine-tune audit  [MEASURE unless marked HARD]", [
        f"pic_root       : {args.pic_root}",
        f"tasks          : {', '.join(tasks)}" +
        (f"   (absent: {missing_tasks})" if missing_tasks else ""),
        f"frozen tau_max : {tau_max!r}  from {frozen['reference_checkpoint']}",
        f"frozen age std : mean={age_mean:.6f} sd={age_sd:.6f}   s={s}",
        f"band table     : {args.band_table}  ({', '.join(D.band_names(bands))})",
        f"report written : {args.out}",
    ])
    _print_phase_a(report, tasks, bands)
    _print_phase_b(report, tasks, b5, mimic_ref)
    _print_decisions(report, tasks, d2, b5)
    D.print_block("audit complete", [
        f"wall {time.time() - t_start:.1f}s. Nothing was trained and no checkpoint was "
        f"modified.",
        "D1, D2 and D3 are open. See model_new/run/pic_audit/readiness.md for the "
        "per-file code assessment.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
