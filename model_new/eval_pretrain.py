#!/usr/bin/env python3
"""Offline evaluation and epoch selection for finished pretraining runs.

    python -m model_new.eval_pretrain --runs model_new/run/vanilla_s0 \
        model_new/run/kernel_s0_072420260946 ... --primary_rule per_arm_best

Nothing is trained, nothing is fine-tuned and no checkpoint is written to. Every arm is
read from ``model_new/run/<run_name>/{config.json,train.json,epoch_NNN.pt}`` and every
number below comes from **one** deterministic validation pass shared by every metric and
every arm: same seed, same collate, ``shuffle=False``, no dropout anywhere in the model,
``torch.no_grad`` except inside the gradient probe. The batch sequence is hashed and the
hash is asserted identical across arms, so a cross-arm difference cannot come from a
different batch order.

Four things here are deliberately awkward, and each is awkward on purpose:

* **The score matrix is never materialised.** ``52,227 x 30,635`` is 1.6e9 scores, 6.4 GB
  in float32 before the sort an exact AP needs, on a machine with 32 GB of system RAM.
  Micro- and macro-AUPRC are accumulated into fixed-edge histograms whose edges are
  chosen once, from a first pass over every checkpoint, and shared by all arms. The
  estimator is unit-tested against ``sklearn.average_precision_score``
  (``tests/test_auprc_histogram.py``).
* **``tau_max`` is read, never re-derived.** It comes off the checkpoint buffer through
  ``DKMModel.tau_max`` and is asserted equal to the frozen value to 1e-6 for every arm.
  A re-derived ``tau_max`` would silently redefine every learned coefficient.
* **Configs are compared before anything is measured.** Differences that are a
  deterministic consequence of the arm are verified by rebuilding the model from the
  *shared* constructor kwargs; any other difference is a hard error, because a silent
  config drift between arms invalidates the whole comparison. ``--allow_config_diff``
  accepts a named difference explicitly and records it in every output file.
* **``primary_rule`` is a required flag and is written to disk before any cross-arm
  number is printed.** Choosing the selection rule after seeing the comparison is the
  cheapest way to manufacture a result, so the code cannot choose it and there is no
  default.

Outputs (all written through ``diagnostics``, all atomic ``.tmp`` -> ``os.replace``):

    model_new/run/eval_pretrain/<arm>/epochs.json   per-epoch metrics + DKM diagnostics
    model_new/run/eval_pretrain/selection.json      the three selection rules
    model_new/run/eval_pretrain/summary.json        cross-arm table at primary_rule
    model_new/run/selection.json                    the same selection block, at the path
                                                    the brief names
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_new import diagnostics as D
from model_new.data import (
    TensorizedPretrainDataset, corpus_stats_cached, dataloader_worker_init, make_collate,
    sample_empirical_taus,
)
from model_new.model import DKMModel
from model_new.optim import build_param_groups
from model_new.preflight import headroom
from model_new.train import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]

# D8 / INV-TMAX. Frozen at pretraining from the exact full-split maximum and carried in
# every checkpoint buffer. It is asserted, never recomputed: every learned Chebyshev
# coefficient is defined against this domain, so a different value would silently change
# the meaning of alpha rather than fail.
EXPECTED_TAU_MAX = 6.72380256652832
TAU_MAX_TOL = 1e-6

SELECTION_RULES = ("per_arm_best", "vanilla_matched", "kernel_matched")

# Config keys that identify a *run* rather than a *configuration*. Everything else must
# either be identical across arms, be reproduced by rebuilding the model from the shared
# kwargs with that arm, or be named in --allow_config_diff.
RUN_IDENTITY_KEYS = ("run_id", "arm", "timestamp")
# Prefixes of the config blocks that DKMModel regenerates from (shared kwargs + arm).
# Membership here grants nothing on its own: a key under one of these prefixes is only
# accepted after the rebuilt model reproduces that arm's stored value exactly.
ARM_DERIVED_PREFIXES = ("model.", "params.", "optim.n_params.", "optim.n_tensors.")

_CORPUS_STATS_CALLS = 0


# --------------------------------------------------------------------------- #
# Config loading, comparison and model reconstruction                         #
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    out[prefix[:-1]] = obj
    return out


def _same(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if np.isnan(fa) and np.isnan(fb):
            return True
        return abs(fa - fb) <= tol * max(1.0, abs(fa), abs(fb))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k], tol) for k in a)
    return a == b


def model_kwargs_from_config(cfg: dict) -> dict:
    """The constructor arguments a run used, recovered from its ``config.json``.

    These are the *inputs*. They must be bit-identical across arms -- everything the four
    runs are supposed to share lives here. ``arm`` is deliberately absent.
    """
    m, d = cfg["model"], cfg["data"]
    f = m["fourier"]
    std = m["age_standardization"]
    return {
        "num_codes": int(d["vocab_size"]),
        "embedding_dim": int(m["embedding_dim"]),
        "seed": int(cfg["seed"]),
        "d_model": int(m["d_model"]),
        "n_layers": int(m["n_layers"]),
        "n_heads": int(m["n_heads"]),
        "use_residual": bool(m["use_residual"]),
        "use_layernorm": bool(m["use_layernorm"]),
        "use_ffn": bool(m["use_ffn"]),
        "ffn_mult": int(m["ffn_mult"]),
        "s": int(m["s"]),
        "tau_max": float(m["tau_max"]),
        "age_M": int(f["M"]),
        "age_p_min": float(f["p_min"]),
        "age_p_max": float(f["p_max"]),
        "age_hidden": int(m["age_hidden"]),
        "gen_final_bias": bool(m["gen_final_bias"]),
        "center_delta_alpha": bool(m["center_delta_alpha"]),
        "demo_dim": int(m["demo_dim"]),
        "demo_channels": tuple(m["demo_channels"]),
        "race_encoding": str(m["race_encoding"]),
        "demo_hidden": int(m["demo_hidden"]),
        "age_mean": float(std["mean"]),
        "age_sd": float(std["sd"]),
        "task": str(m.get("task", "pretrain")),
        "max_seq_len": int(d["max_seq_len"]),
        "tensorized_dir": str(d["paths"]["tensorized_dir"]),
        "vocab_path": str(d["paths"]["vocab_path"]),
    }


def build_model(shared: dict, arm: str) -> DKMModel:
    """A model at the SHARED hyperparameters with one arm swapped in.

    The embedding table is a zeros placeholder of the right shape: it is a persistent
    buffer, so ``load_state_dict`` restores the real one from the checkpoint rather than
    rebuilding it from ``bge_embeddings.pt`` (INV-FROZEN). Loading with ``strict=True``
    is itself a check that ``config.json`` describes the architecture the checkpoint
    actually holds.
    """
    if shared["task"] != "pretrain":
        raise AssertionError(
            f"[HARD] eval_pretrain evaluates pretraining checkpoints; config says "
            f"task={shared['task']!r}")
    table = torch.zeros(shared["num_codes"] + 2, shared["embedding_dim"], dtype=torch.float32)
    return DKMModel(
        num_codes=shared["num_codes"], embedding_table=table, arm=arm, seed=shared["seed"],
        d_model=shared["d_model"], n_layers=shared["n_layers"], n_heads=shared["n_heads"],
        use_residual=shared["use_residual"], use_layernorm=shared["use_layernorm"],
        use_ffn=shared["use_ffn"], ffn_mult=shared["ffn_mult"], s=shared["s"],
        tau_max=shared["tau_max"], age_M=shared["age_M"], age_p_min=shared["age_p_min"],
        age_p_max=shared["age_p_max"], age_hidden=shared["age_hidden"],
        gen_final_bias=shared["gen_final_bias"],
        center_delta_alpha=shared["center_delta_alpha"], demo_dim=shared["demo_dim"],
        demo_channels=shared["demo_channels"], race_encoding=shared["race_encoding"],
        demo_hidden=shared["demo_hidden"], age_mean=shared["age_mean"],
        age_sd=shared["age_sd"], task="pretrain",
    )


def check_configs(configs: dict[str, dict], order: list[str], allow: set[str]) -> dict:
    """HARD. All four configs identical apart from the arm, its consequences, and run identity.

    Three separate statements, checked in order:

      1. the constructor kwargs are identical across arms -- this is the one that matters,
         since it is what "identical flags apart from --arm" means operationally;
      2. rebuilding the model from those shared kwargs with each arm reproduces that arm's
         own ``model`` / ``params`` / group-count blocks exactly, so every difference in
         them is *demonstrably* a consequence of the arm rather than an assumed one;
      3. nothing else differs.

    A violation of (3) that the operator has decided is benign must be named on the
    command line; it is then recorded in every output file. There is no silent tolerance.
    """
    ref_arm = order[0]
    shared = model_kwargs_from_config(configs[ref_arm])
    for arm in order[1:]:
        other = model_kwargs_from_config(configs[arm])
        bad = sorted(k for k in shared if not _same(shared[k], other.get(k, "<MISSING>")))
        if bad:
            detail = "; ".join(f"{k}: {ref_arm}={shared[k]!r} vs {arm}={other.get(k)!r}"
                               for k in bad)
            raise AssertionError(
                f"[HARD] arms {ref_arm} and {arm} were trained with different model/data "
                f"configuration, so no cross-arm comparison is valid: {detail}")

    # (2) rebuild and verify the arm-derived blocks, per arm.
    derived_ok: dict[str, dict] = {}
    for arm in order:
        m = build_model(shared, arm)
        groups, group_report = build_param_groups(
            m, configs[arm]["optim"]["lr_backbone"], configs[arm]["optim"]["lr_age"],
            configs[arm]["optim"]["lr_head"])
        rebuilt = {
            "model": m.config_dict(),
            "params": m.parameter_report(),
            "optim.n_params": group_report["n_params"],
            "optim.n_tensors": group_report["n_tensors"],
        }
        stored = {
            "model": configs[arm]["model"],
            "params": configs[arm]["params"],
            "optim.n_params": configs[arm]["optim"]["n_params"],
            "optim.n_tensors": configs[arm]["optim"]["n_tensors"],
        }
        mismatched = []
        for block, got in rebuilt.items():
            want = stored[block]
            for k in sorted(set(got) | set(want)):
                if not _same(got.get(k, "<MISSING>"), want.get(k, "<MISSING>")):
                    mismatched.append(f"{block}.{k}: rebuilt={got.get(k)!r} "
                                      f"stored={want.get(k)!r}")
        if mismatched:
            raise AssertionError(
                f"[HARD] rebuilding arm={arm} from the shared constructor kwargs does not "
                f"reproduce its own config.json, so its differences from the other arms "
                f"are NOT explained by the arm: " + "; ".join(mismatched))
        derived_ok[arm] = rebuilt
        del m

    # (3) everything else.
    flat = {arm: _flatten(cfg) for arm, cfg in configs.items()}
    keys = sorted(set().union(*[set(f) for f in flat.values()]))
    arm_derived, identity, unexplained, accepted = [], [], [], {}
    for key in keys:
        vals = [flat[a].get(key, "<MISSING>") for a in order]
        if all(_same(vals[0], v) for v in vals[1:]):
            continue
        if key in RUN_IDENTITY_KEYS:
            identity.append(key)
        elif key.startswith(ARM_DERIVED_PREFIXES):
            arm_derived.append(key)
        elif key in allow:
            accepted[key] = {a: flat[a].get(key, "<MISSING>") for a in order}
        else:
            unexplained.append({"key": key,
                                "values": {a: flat[a].get(key, "<MISSING>") for a in order}})
    if unexplained:
        detail = "; ".join(f"{u['key']} = " + ", ".join(f"{a}:{v!r}" for a, v in
                                                        u["values"].items())
                           for u in unexplained)
        raise AssertionError(
            f"[HARD] config.json differs between arms beyond arm/run_name and beyond what "
            f"the arm explains, on {len(unexplained)} key(s): {detail}. A silent config "
            f"drift between arms invalidates the comparison. Re-run with "
            f"--allow_config_diff " + " ".join(u["key"] for u in unexplained) +
            " to accept it explicitly; the acceptance is recorded in every output file.")
    return {
        "reference_run": configs[ref_arm]["run_id"],
        "reference_arm": ref_arm,
        "n_shared_kwargs": len(shared),
        "shared_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in shared.items()},
        "arm_derived_differences": arm_derived,
        "arm_derived_verified_by_rebuild": True,
        "run_identity_differences": identity,
        "accepted_differences": accepted,
    }


# --------------------------------------------------------------------------- #
# The shared deterministic validation pass                                    #
# --------------------------------------------------------------------------- #
class BatchOrderHash:
    """A digest of the sequence of batches a pass actually consumed.

    Recorded in every output file and asserted equal across arms. ``shuffle=False`` makes
    this true by construction; the hash is what turns "by construction" into "checked",
    and it costs one blake2b pass over the code indices.
    """

    def __init__(self) -> None:
        self._h = hashlib.blake2b(digest_size=16)
        self.n_batches = 0
        self.n_rows = 0

    def update(self, batch: dict) -> None:
        self.n_batches += 1
        self.n_rows += int(batch["lengths"].shape[0])
        self._h.update(np.ascontiguousarray(batch["lengths"].numpy()).tobytes())
        self._h.update(np.ascontiguousarray(batch["code_indices"].numpy()).tobytes())
        rows, cols = np.nonzero(batch["target_codes"].numpy())
        self._h.update(np.ascontiguousarray(rows.astype(np.int32)).tobytes())
        self._h.update(np.ascontiguousarray(cols.astype(np.int32)).tobytes())

    @property
    def hexdigest(self) -> str:
        return self._h.hexdigest()


def make_val_loader(ds: TensorizedPretrainDataset, batch_size: int, num_workers: int,
                    race_encoding: str) -> DataLoader:
    kw: dict[str, Any] = dict(num_workers=num_workers, collate_fn=make_collate(race_encoding),
                              pin_memory=False, worker_init_fn=dataloader_worker_init,
                              persistent_workers=num_workers > 0)
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    # shuffle=False and drop_last=False: every arm sees every sequence, in one order.
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, **kw)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _band_of_batch(batch: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> (age at the last valid event per row, band index per row)."""
    lengths = batch["lengths"]
    rows = torch.arange(lengths.shape[0])
    age_last = batch["age_years"][rows, lengths - 1].float().numpy()
    return age_last, D.band_index(age_last)


def targets_pass(loader: DataLoader, vocab_size: int, max_batches: int) -> dict:
    """One pass over the targets alone -- no model, no scores.

    Everything that depends on the data and not on the arm is decided here, once: the
    reference batch-order hash, the per-sequence age band, and the per-code positive
    counts that decide macro-AUPRC eligibility. Deciding eligibility from the targets
    means the included code set is identical for every arm and every epoch, which is the
    only way a macro average is comparable between them.
    """
    hasher = BatchOrderHash()
    pos_pooled = torch.zeros(vocab_size, dtype=torch.int64)
    band_pos = torch.zeros(len(D.AGE_BANDS), vocab_size, dtype=torch.int64)
    n_true: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    bands: list[np.ndarray] = []
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        hasher.update(batch)
        t = (batch["target_codes"] > 0)
        pos_pooled += t.sum(dim=0).long()
        age_last, band = _band_of_batch(batch)
        for b in range(len(D.AGE_BANDS)):
            sel = torch.from_numpy(band == b)
            if bool(sel.any()):
                band_pos[b] += t[sel].sum(dim=0).long()
        n_true.append(t.sum(dim=1).numpy().astype(np.int64))
        ages.append(age_last)
        bands.append(band)
    return {
        "hash": hasher.hexdigest,
        "n_batches": hasher.n_batches,
        "n_examples": hasher.n_rows,
        "n_true": np.concatenate(n_true) if n_true else np.zeros(0, dtype=np.int64),
        "age_last": np.concatenate(ages) if ages else np.zeros(0, dtype=np.float32),
        "band": np.concatenate(bands) if bands else np.zeros(0, dtype=np.int64),
        "pos_pooled": pos_pooled.numpy(),
        "pos_by_band": band_pos.numpy(),
    }


@torch.no_grad()
def logit_range_pass(model: DKMModel, batches: list[dict], device: torch.device
                     ) -> tuple[float, float]:
    lo, hi = float("inf"), float("-inf")
    for b in batches:
        logits = model(_to_device(b, device))["code_logits"].float()
        lo = min(lo, float(logits.min()))
        hi = max(hi, float(logits.max()))
    return lo, hi


@torch.no_grad()
def evaluate_pass(model: DKMModel, loader: DataLoader, device: torch.device, *,
                  edges: tuple[float, float], micro_bins: int, macro_bins: int,
                  eligible: dict[str, torch.Tensor], ks: tuple[int, ...], ndcg_k: int,
                  cap_denominator: bool, max_batches: int) -> dict:
    """The single validation pass. Every Part-2 metric is accumulated here, streaming.

    Nothing per-(sequence, code) is retained: BCE arrives as per-sequence sums, AUPRC as
    histogram counts, and the ranking metrics as one float per sequence. Peak host memory
    is therefore independent of ``|V|`` beyond one batch of logits.
    """
    model.eval()
    model.reset_clamp_stats()
    lo, hi = edges
    band_names = D.band_names()
    micro = D.ScoreHistogram(lo, hi, micro_bins, device=device)
    micro_band = {name: D.ScoreHistogram(lo, hi, micro_bins, device=device)
                  for name in band_names}
    macro = D.PerCodeHistogram(eligible["pooled"], lo, hi, macro_bins, device=device)
    macro_band = {name: D.PerCodeHistogram(eligible[name], lo, hi, macro_bins, device=device)
                  for name in band_names}

    hasher = BatchOrderHash()
    bce_sum = 0.0
    n_elements = 0
    batch_means: list[float] = []
    per_ex: dict[str, list[np.ndarray]] = {}
    ages: list[np.ndarray] = []
    bands: list[np.ndarray] = []

    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        hasher.update(batch)
        age_last, band = _band_of_batch(batch)
        ages.append(age_last)
        bands.append(band)
        dev_batch = _to_device(batch, device)
        logits = model(dev_batch)["code_logits"].float()
        targets = dev_batch["target_codes"].float()

        per_seq_bce, n_codes = D.bce_totals(logits, targets)
        bce_sum += float(per_seq_bce.double().sum())
        n_elements += int(per_seq_bce.numel()) * n_codes
        batch_means.append(float(per_seq_bce.double().mean() / n_codes))

        for k, v in D.topk_per_example(logits, targets, ks=ks, ndcg_k=ndcg_k,
                                       cap_denominator=cap_denominator).items():
            per_ex.setdefault(k, []).append(v.numpy())
        per_ex.setdefault("bce_per_sequence", []).append(per_seq_bce.numpy())

        micro.update(logits, targets)
        macro.update(logits, targets)
        for b, name in enumerate(band_names):
            sel = torch.from_numpy(band == b).to(device)
            if not bool(sel.any()):
                continue
            micro_band[name].update(logits[sel], targets[sel])
            macro_band[name].update(logits[sel], targets[sel])

    return {
        "hash": hasher.hexdigest,
        "n_batches": hasher.n_batches,
        "n_examples": hasher.n_rows,
        "bce_sum": bce_sum,
        "n_elements": n_elements,
        "batch_means": batch_means,
        "per_example": {k: np.concatenate(v) for k, v in per_ex.items()},
        "age_last": np.concatenate(ages) if ages else np.zeros(0, dtype=np.float32),
        "band": np.concatenate(bands) if bands else np.zeros(0, dtype=np.int64),
        "micro": micro,
        "micro_band": micro_band,
        "macro": macro,
        "macro_band": macro_band,
        "clamp_rate": D.clamp_rates(model),
    }


def _macro_block(hist: D.PerCodeHistogram, n_codes_total: int, min_pos: int) -> dict:
    ap = hist.average_precision_per_code()
    finite = ap[~np.isnan(ap)]
    return {
        "macro_auprc": float(finite.mean()) if finite.size else float("nan"),
        "macro_n_codes_included": int(hist.n_codes),
        "macro_n_codes_excluded": int(n_codes_total - hist.n_codes),
        "macro_n_codes_undefined": int(hist.n_codes - finite.size),
        "min_pos": int(min_pos),
        "histogram": hist.to_json(),
    }


def assemble_metrics(res: dict, *, eligible_counts: dict, vocab_size: int, min_pos: int,
                     min_band_n: int, ks: tuple[int, ...], ndcg_k: int,
                     cap_denominator: bool) -> dict:
    """Pooled and band-stratified Part-2 metrics from one pass's accumulators."""
    pe = res["per_example"]
    n = int(res["n_examples"])
    n_codes = vocab_size
    band = res["band"]

    def _mean(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        return float(x.mean()) if x.size else float("nan")

    overall = {
        "n_sequences": n,
        "val_bce": res["bce_sum"] / res["n_elements"] if res["n_elements"] else float("nan"),
        "val_bce_batch_mean_of_means": float(np.mean(res["batch_means"]))
        if res["batch_means"] else float("nan"),
        "micro_auprc": res["micro"].average_precision(),
        "micro_histogram": res["micro"].to_json(),
        "n_sequences_without_targets": int((pe["n_true"] == 0).sum()),
        "mean_n_true": float(pe["n_true"].mean()) if pe["n_true"].size else float("nan"),
    }
    overall.update(_macro_block(res["macro"], n_codes, min_pos))
    for k in ks:
        overall[f"recall@{k}"] = _mean(pe[f"recall@{k}"])
    overall[f"ndcg@{ndcg_k}"] = _mean(pe[f"ndcg@{ndcg_k}"])

    by_band: dict[str, dict] = {}
    for b, name in enumerate(D.band_names()):
        sel = band == b
        n_b = int(sel.sum())
        hist = res["micro_band"][name]
        n_pos = hist.n_pos
        n_neg = hist.n_neg
        metrics: dict[str, Any] = {
            # float64 accumulation, as the pooled figure uses, so a band total and the
            # pooled total are summed the same way.
            "val_bce": (float(pe["bce_per_sequence"][sel].astype(np.float64).sum())
                        / (n_b * n_codes)) if n_b else float("nan"),
            "micro_auprc": hist.average_precision(),
        }
        for k in ks:
            metrics[f"recall@{k}"] = _mean(pe[f"recall@{k}"][sel]) if n_b else float("nan")
        metrics[f"ndcg@{ndcg_k}"] = _mean(pe[f"ndcg@{ndcg_k}"][sel]) if n_b else float("nan")
        mb = _macro_block(res["macro_band"][name], n_codes, min_pos)
        metrics["macro_auprc"] = mb["macro_auprc"]
        entry = D.band_entry(n=n_b, n_pos=n_pos, n_neg=n_neg, metrics=metrics,
                             min_n=min_band_n)
        entry["macro_n_codes_included"] = mb["macro_n_codes_included"]
        entry["macro_n_codes_excluded"] = mb["macro_n_codes_excluded"]
        entry["n_codes_with_at_least_min_pos"] = int(eligible_counts[name])
        by_band[name] = entry

    return {"overall": overall, "by_band": by_band}


# --------------------------------------------------------------------------- #
# Part 3 -- DKM diagnostics at a trained checkpoint                           #
# --------------------------------------------------------------------------- #
def gradient_probe(model: DKMModel, batches: list[dict], device: torch.device,
                   lrs: dict) -> dict:
    """Forward + backward on a fixed set of validation batches, reading gradients only.

    ``optimizer.step`` is never called, no optimizer is constructed, and the gradients are
    cleared afterwards, so the loaded checkpoint is untouched (and is reloaded for the
    next epoch regardless). Groups come from ``optim.build_param_groups``: the age / head
    / backbone partition is the declared one the optimizer used, not a name match.

    The reported gradient is that of the **mean** loss over the K batches.
    """
    groups, _ = build_param_groups(model, lrs["lr_backbone"], lrs["lr_age"], lrs["lr_head"])
    model.eval()
    model.zero_grad(set_to_none=True)
    for b in batches:
        dev = _to_device(b, device)
        out = model(dev)
        loss = F.binary_cross_entropy_with_logits(out["code_logits"].float(),
                                                  dev["target_codes"].float())
        (loss / len(batches)).backward()

    # Both readings happen BEFORE the gradients are cleared. Reading the generator
    # fractions after zero_grad would report a zero fraction for every arm -- the exact
    # bug signal this probe exists to detect.
    group_norms = D.gradient_group_norms(groups)
    generators = D.generator_gradient_fractions(model)
    model.zero_grad(set_to_none=True)

    arm = model.cfg.arm
    age, backbone = group_norms["age"], group_norms["backbone"]
    if arm == "vanilla":
        if not age["empty_group"]:
            raise AssertionError(
                f"[INV-GROUPS] arm=vanilla must have an empty age group, got "
                f"{age['n_tensors']} tensors")
        if age["grad_l2"] != 0.0:
            raise AssertionError(
                f"[HARD] arm=vanilla reported a nonzero age-group gradient norm "
                f"{age['grad_l2']!r}; an empty group must report exactly 0")
    return {
        "probe": {
            "n_batches": len(batches),
            "loss_reduction": "mean over the K probe batches of the per-batch mean BCE",
            "groups": group_norms,
            "age_over_backbone": (age["grad_l2"] / backbone["grad_l2"])
            if backbone["grad_l2"] > 0 else float("nan"),
        },
        "generators": generators,
    }


def dkm_diagnostics(model: DKMModel, probe_batches: list[dict], device: torch.device,
                    empirical_ages: torch.Tensor, lrs: dict, *, s: int,
                    tau_grid_points: int) -> dict:
    grad = gradient_probe(model, probe_batches, device, lrs)
    # After the probe, and only after it: the equal-norm probe overwrites alpha_base at
    # every site and restores it, so it must not run while gradients are being read.
    dev_batches = [_to_device(b, device) for b in probe_batches]
    with torch.no_grad():
        probe = dict(headroom(model, dev_batches, s))
    del dev_batches
    probe["note"] = (
        "same probe as README section 5, run on validation batches at a TRAINED "
        "checkpoint rather than at initialization. The section 5 table (standard_block: "
        "max|dlogit| 0.0990, max/sd 1.44) is an init-time measurement of headroom; these "
        "numbers are larger because the trained head amplifies the pooled representation, "
        "and the two are comparable in kind, not in magnitude.")
    probe["reference_readme_section5_at_init"] = {"max_abs_delta_logit": 0.0990,
                                                  "max_delta_over_logit_sd": 1.44}
    return {
        "gradient_probe": grad["probe"],
        "generator_gradients": grad["generators"],
        "delta_alpha_norms": D.delta_alpha_norms(model, empirical_ages),
        "kernel_separation": D.kernel_separation(model, n_tau=tau_grid_points),
        "equal_norm_probe": probe,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True,
                   help="pretraining run directories (config.json + train.json + epoch_NNN.pt)")
    p.add_argument("--primary_rule", choices=SELECTION_RULES, required=True,
                   help="REQUIRED and never defaulted: it is written to selection.json "
                        "before any cross-arm number is printed.")
    p.add_argument("--out_root", type=Path, default=Path("model_new/run/eval_pretrain"))
    p.add_argument("--run_root", type=Path, default=Path("model_new/run"))
    p.add_argument("--allow_config_diff", nargs="*", default=[],
                   help="dotted config keys allowed to differ between arms; each one is "
                        "recorded in every output file")

    p.add_argument("--split", type=str, default="val")
    p.add_argument("--tensorized_dir", type=Path, default=None,
                   help="override; by default taken from config.json (and required to "
                        "agree across arms)")
    p.add_argument("--vocab_path", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=0, help="0 = the training batch size")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, nargs="*", default=None,
                   help="epochs to evaluate; default every epoch_NNN.pt in the run dir")
    p.add_argument("--max_val_batches", type=int, default=0,
                   help="0 = the full validation split. Any other value is an explicit "
                        "subsample and is recorded in every output file.")

    p.add_argument("--min_pos", type=int, default=10,
                   help="minimum validation positives for a code to enter macro-AUPRC")
    p.add_argument("--min_band_n", type=int, default=D.MIN_BAND_N,
                   help="below this many sequences a band metric is NaN + unreliable")
    p.add_argument("--micro_bins", type=int, default=100_000)
    p.add_argument("--macro_bins", type=int, default=8192,
                   help="per-code bins. Measured against sklearn at validation scale "
                        "(n = 52,227): 2048 bins give a per-code AP error up to 6e-4, "
                        "8192 up to 1.7e-4. Memory is n_eligible_codes x bins x 16 B, "
                        "summed over the pooled set and every band (~3.4 GB at 8192).")
    p.add_argument("--logit_range", type=float, nargs=2, default=None,
                   help="fixed histogram edges; by default from a first pass over every "
                        "checkpoint, shared by all arms")
    p.add_argument("--edge_batches", type=int, default=8)
    p.add_argument("--edge_margin", type=float, default=2.0,
                   help="the edge pass sees a few batches, the metric pass sees all of "
                        "them, so the range is widened by this many logits on each side. "
                        "Scores still outside it are clamped into the end bins and the "
                        "fraction is recorded in every output file.")
    p.add_argument("--oor_flag_threshold", type=float, default=1e-5,
                   help="report an out-of-range fraction above this in the final report; "
                        "the fraction itself is always recorded")
    p.add_argument("--cap_recall_denominator", action="store_true",
                   help="use min(|true|, k) as the recall denominator; OFF by default")
    p.add_argument("--grad_batches", type=int, default=8)
    p.add_argument("--tau_grid_points", type=int, default=257)
    p.add_argument("--tau_sample_examples", type=int, default=400,
                   help="validation windows sampled for the Gram condition number")
    p.add_argument("--stats_sample_windows", type=int, default=4000)
    p.add_argument("--skip_corpus_stats", action="store_true")
    return p


def corpus_stats_once(ds: TensorizedPretrainDataset, split_dir: Path, split: str,
                      sample_windows: int, seed: int, max_seq_len: int):
    """INV-STATS-SINGLE: ``data.corpus_stats`` runs at most once per process."""
    global _CORPUS_STATS_CALLS
    if _CORPUS_STATS_CALLS:
        raise AssertionError("[INV-STATS-SINGLE] corpus statistics were requested twice in "
                             "one process; compute them once and reuse the result")
    _CORPUS_STATS_CALLS += 1
    return corpus_stats_cached(ds, split_dir, split=split, sample_windows=sample_windows,
                               seed=seed, max_seq_len=max_seq_len)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    t_start = time.time()
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    flags: list[dict] = []

    # ---- runs, arms, configs ------------------------------------------------ #
    configs, run_dirs, train_json = {}, {}, {}
    order: list[str] = []
    for rd in args.runs:
        rd = Path(rd)
        cfg = _read_json(rd / "config.json")
        arm = cfg["arm"]
        if arm in configs:
            raise AssertionError(
                f"[HARD] two runs with arm={arm!r} were passed ({run_dirs[arm]} and {rd}); "
                f"outputs are keyed by arm, so one arm means one run")
        configs[arm], run_dirs[arm], order = cfg, rd, order + [arm]
        train_json[arm] = _read_json(rd / "train.json")

    # ---- primary_rule on disk BEFORE any cross-arm comparison --------------- #
    out_root = Path(args.out_root)
    selection_paths = [out_root / "selection.json", Path(args.run_root) / "selection.json"]
    declared = {
        "primary_rule": args.primary_rule,
        "declared_before_any_cross_arm_comparison": True,
        "declared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "declared; evaluation in progress",
        "arms": order,
        "runs": {a: str(run_dirs[a]) for a in order},
        "rules_available": list(SELECTION_RULES),
    }
    for p in selection_paths:
        D.write_json(p, declared)

    config_report = check_configs(configs, order, set(args.allow_config_diff))
    if config_report["accepted_differences"]:
        for k, v in config_report["accepted_differences"].items():
            flags.append({"kind": "config-difference-accepted",
                          "detail": f"{k} differs between arms and was accepted via "
                                    f"--allow_config_diff: {v}"})
    D.print_config_check(config_report)

    shared = config_report["shared_kwargs"]
    ref_cfg = configs[order[0]]
    batch_size = args.batch_size or int(ref_cfg["optim"]["batch_size"])
    race_encoding = shared["race_encoding"]
    vocab_size = int(shared["num_codes"])

    tensorized_dir = Path(args.tensorized_dir or shared["tensorized_dir"])
    vocab_path = Path(args.vocab_path or shared["vocab_path"])
    split_dir = tensorized_dir / args.split
    ds = TensorizedPretrainDataset(split_dir, vocab_path, max_seq_len=shared["max_seq_len"])
    if ds.num_codes != vocab_size:
        raise AssertionError(f"[HARD] vocabulary is {ds.num_codes} codes but the runs were "
                             f"trained on {vocab_size}")

    # ---- pass 0: targets only. Data facts, shared by every arm. ------------- #
    loader = make_val_loader(ds, batch_size, args.num_workers, race_encoding)
    tgt = targets_pass(loader, vocab_size, args.max_val_batches)
    eligible = {"pooled": torch.from_numpy(
        np.nonzero(tgt["pos_pooled"] >= args.min_pos)[0]).long()}
    eligible_counts = {}
    for b, name in enumerate(D.band_names()):
        idx = np.nonzero(tgt["pos_by_band"][b] >= args.min_pos)[0]
        eligible[name] = torch.from_numpy(idx).long()
        eligible_counts[name] = int(idx.size)

    empirical_ages = torch.from_numpy(tgt["age_last"].astype(np.float32))
    band_counts = {name: int((tgt["band"] == b).sum())
                   for b, name in enumerate(D.band_names())}
    for name, n_b in band_counts.items():
        if 0 < n_b < args.min_band_n:
            flags.append({"kind": "band-below-reliability-threshold",
                          "detail": f"age band {name} has n = {n_b} validation sequences, "
                                    f"below --min_band_n {args.min_band_n}; every metric "
                                    f"for it is reported as NaN with unreliable=true"})

    # ---- corpus-level quantities: computed once, shared by every arm -------- #
    tau_sample = sample_empirical_taus(ds, n_examples=args.tau_sample_examples, seed=args.seed)
    gram = D.gram_condition_numbers(tau_sample, shared["s"], shared["tau_max"])
    gram["distribution"] = (f"empirical within-window pairwise lags on the {args.split} "
                            f"split, {args.tau_sample_examples} sampled windows")
    corpus = None
    if not args.skip_corpus_stats:
        corpus = corpus_stats_once(ds, split_dir, args.split, args.stats_sample_windows,
                                   args.seed, shared["max_seq_len"]).to_json()

    # ---- checkpoints -------------------------------------------------------- #
    def epochs_of(arm: str) -> list[int]:
        found = sorted(int(p.stem.split("_")[1]) for p in run_dirs[arm].glob("epoch_*.pt"))
        return [e for e in found if (args.epochs is None or e in set(args.epochs))]

    epochs = {a: epochs_of(a) for a in order}
    common = sorted(set.intersection(*[set(v) for v in epochs.values()])) if epochs else []
    if not common:
        raise AssertionError("[HARD] the runs share no saved epoch to compare at")
    for a in order:
        extra = sorted(set(epochs[a]) - set(common))
        if extra:
            flags.append({"kind": "epoch-not-shared",
                          "detail": f"arm {a} has epochs {extra} that another arm does not; "
                                    f"they are not evaluated, because a cross-arm rule can "
                                    f"only be applied at a shared epoch"})

    def load_into(model: DKMModel, arm: str, epoch: int) -> dict:
        path = run_dirs[arm] / f"epoch_{epoch:03d}.pt"
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if ckpt.get("arm") != arm or ckpt.get("config", {}).get("arm") != arm:
            raise AssertionError(
                f"[HARD] {path} holds arm={ckpt.get('arm')!r} "
                f"(config {ckpt.get('config', {}).get('arm')!r}) but its run directory's "
                f"config.json says arm={arm!r}")
        if int(ckpt.get("epoch", -1)) != epoch:
            raise AssertionError(f"[HARD] {path} holds epoch {ckpt.get('epoch')}, not {epoch}")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        got = float(model.tau_max)
        if abs(got - EXPECTED_TAU_MAX) > TAU_MAX_TOL:
            raise AssertionError(
                f"[INV-TMAX] {path}: tau_max from the checkpoint buffer is {got!r}, which "
                f"differs from the frozen {EXPECTED_TAU_MAX!r} by more than {TAU_MAX_TOL}. "
                f"Every learned coefficient is defined against tau_max; the arms are not "
                f"comparable.")
        if abs(float(ckpt["tau_max"]) - got) > TAU_MAX_TOL:
            raise AssertionError(
                f"[INV-TMAX] {path}: the checkpoint's tau_max field {ckpt['tau_max']!r} "
                f"disagrees with the kernel buffers {got!r}")
        return {"epoch": epoch, "tau_max": got, "tau_max_source": ckpt.get("tau_max_source"),
                "age_standardization": ckpt.get("age_standardization")}

    # ---- pass 1: fixed histogram edges, from every checkpoint --------------- #
    probe_batches: list[dict] = []
    for i, b in enumerate(make_val_loader(ds, batch_size, 0, race_encoding)):
        if i >= max(args.grad_batches, args.edge_batches):
            break
        probe_batches.append(b)
    edge_batches = probe_batches[: args.edge_batches]
    grad_batches = probe_batches[: args.grad_batches]

    if args.logit_range is not None:
        lo, hi = float(args.logit_range[0]), float(args.logit_range[1])
        edge_source = "--logit_range (explicit)"
    else:
        lo, hi = float("inf"), float("-inf")
        for arm in order:
            m = build_model(shared, arm)
            for epoch in common:
                load_into(m, arm, epoch)
                a, b_ = logit_range_pass(m, edge_batches, device)
                lo, hi = min(lo, a), max(hi, b_)
            del m
        lo, hi = lo - args.edge_margin, hi + args.edge_margin
        edge_source = (f"min/max logit over {args.edge_batches} validation batches at every "
                       f"one of the {len(order) * len(common)} checkpoints, widened by "
                       f"{args.edge_margin}")
    edges = (lo, hi)

    header = {
        "primary_rule": args.primary_rule, "selection_path": str(selection_paths[0]),
        "runs": [str(run_dirs[a]) for a in order], "arms": order,
        "n_examples": tgt["n_examples"], "vocab_size": vocab_size, "batch_size": batch_size,
        "n_batches": tgt["n_batches"], "seed": args.seed,
        "batch_order_hash": tgt["hash"], "tau_max": shared["tau_max"],
        "expected_tau_max": EXPECTED_TAU_MAX, "device": str(device),
        "max_val_batches": args.max_val_batches,
    }
    D.print_eval_header(header)

    shared_block = {
        "harness": {
            "split": args.split, "n_sequences": tgt["n_examples"],
            "n_batches": tgt["n_batches"], "batch_size": batch_size,
            "shuffle": False, "drop_last": False, "seed": args.seed,
            "num_workers": args.num_workers, "amp": False, "device": str(device),
            "batch_order_hash": tgt["hash"],
            "batch_order_hash_definition": "blake2b-128 over (lengths, code_indices, "
                                           "target nonzero indices) of every batch in order",
            "max_val_batches": args.max_val_batches,
            "subsampled": bool(args.max_val_batches),
        },
        "tau_max": {"value": shared["tau_max"], "expected": EXPECTED_TAU_MAX,
                    "tolerance": TAU_MAX_TOL,
                    "source": "checkpoint buffer via DKMModel.tau_max; never re-derived"},
        "config_check": config_report,
        "histogram_edges": {"lo": lo, "hi": hi, "source": edge_source,
                            "micro_bins": args.micro_bins, "macro_bins": args.macro_bins,
                            "shared_by_all_arms": True},
        "targets": {
            "mean_n_true": float(tgt["n_true"].mean()) if tgt["n_true"].size else float("nan"),
            "median_n_true": float(np.median(tgt["n_true"])) if tgt["n_true"].size else
            float("nan"),
            "n_sequences_without_targets": int((tgt["n_true"] == 0).sum()),
            "prevalence": float(tgt["n_true"].sum() / max(1, tgt["n_examples"] * vocab_size)),
            "n_codes_with_at_least_one_positive": int((tgt["pos_pooled"] > 0).sum()),
            "n_codes_eligible_for_macro": int(eligible["pooled"].numel()),
            "n_codes_eligible_by_band": eligible_counts,
            "min_pos": args.min_pos,
            "band_counts": band_counts,
            "band_definition": "age at the last valid event of each validation sequence",
        },
        "gram_condition_numbers": gram,
        "corpus_stats": corpus,
        "age_bands": {name: [lo_, hi_] for name, lo_, hi_ in D.AGE_BANDS},
        "min_band_n": args.min_band_n,
    }

    schema = {
        "nan_encoding": "NaN and infinities serialise as null; an unreliable band carries "
                        "unreliable=true and unreliable_reason alongside the nulls",
        "val_bce": "binary_cross_entropy_with_logits, reduction = sum over all (sequence, "
                   "code) pairs divided by their count (element mean), pos_weight = null, "
                   "no masking: every code is a valid target for every sequence. Identical "
                   "for every arm. val_bce_batch_mean_of_means is the mean of per-batch "
                   "means, reported only for continuity with train.json's series.",
        "micro_auprc": "streaming fixed-edge score histogram over all (sequence, code) "
                       "pairs; edges shared by every arm; ties within a bin are treated as "
                       "tied, as sklearn does",
        "macro_auprc": "per-code average precision averaged over codes with at least "
                       "min_pos positives in this stratum. Excluded codes are excluded, "
                       "never imputed.",
        "recall": f"per sequence, denominator = number of true codes in the target visit, "
                  f"capped_at_k = {bool(args.cap_recall_denominator)}; averaged over "
                  f"sequences. Sequences with no true codes are NaN and counted separately.",
        "ndcg@20": "binary relevance, 1/log2(i+1) discount, ideal DCG over min(|true|, 20)",
        "train_json_val_bce": "the run's own per-epoch validation loss, computed during "
                              "training on --val_max_batches batches only; it is NOT the "
                              "val_bce used for selection here",
        "gradient_probe": "forward+backward on fixed validation batches; optimizer.step is "
                          "never called and the checkpoint is not mutated",
        "kernel_separation": "log w curves centered per age over the tau grid before "
                             "comparison, because softmax ignores a per-row constant",
    }

    # ---- pass 2: metrics + diagnostics, per arm, per epoch ------------------ #
    records: dict[str, list[dict]] = {a: [] for a in order}
    val_bce: dict[str, dict[int, float]] = {a: {} for a in order}
    for arm in order:
        model = build_model(shared, arm)
        lrs = {k: float(configs[arm]["optim"][k]) for k in
               ("lr_backbone", "lr_age", "lr_head")}
        steps = {int(r["epoch"]): int(r["step"]) for r in train_json[arm]}
        for epoch in common:
            t0 = time.time()
            meta = load_into(model, arm, epoch)
            res = evaluate_pass(
                model, loader, device, edges=edges, micro_bins=args.micro_bins,
                macro_bins=args.macro_bins, eligible=eligible, ks=D.EVAL_KS,
                ndcg_k=D.NDCG_K, cap_denominator=args.cap_recall_denominator,
                max_batches=args.max_val_batches)
            if res["hash"] != tgt["hash"]:
                raise AssertionError(
                    f"[HARD] arm={arm} epoch={epoch} consumed a different batch sequence "
                    f"({res['hash']} vs the reference {tgt['hash']}); the arms are not "
                    f"being compared on the same data")
            metrics = assemble_metrics(
                res, eligible_counts=eligible_counts, vocab_size=vocab_size,
                min_pos=args.min_pos, min_band_n=args.min_band_n, ks=D.EVAL_KS,
                ndcg_k=D.NDCG_K, cap_denominator=args.cap_recall_denominator)
            diag = dkm_diagnostics(model, grad_batches, device,
                                   empirical_ages=empirical_ages, lrs=lrs, s=shared["s"],
                                   tau_grid_points=args.tau_grid_points)

            rec = {
                "arm": arm, "epoch": epoch, "step": steps.get(epoch),
                "checkpoint": str(run_dirs[arm] / f"epoch_{epoch:03d}.pt"),
                "tau_max": meta["tau_max"],
                "wall_clock_s": time.time() - t0,
                **metrics,
                "diagnostics": diag,
                "clamp_rate": res["clamp_rate"],
                "batch_order_hash": res["hash"],
            }
            del res            # the per-code histograms are ~1 GB of VRAM; free before the next pass
            records[arm].append(rec)
            val_bce[arm][epoch] = rec["overall"]["val_bce"]

            # Written after every epoch, atomically, so an interrupted run still leaves a
            # valid file with everything finished so far.
            D.write_json(out_root / arm / "epochs.json", {
                "_schema": schema, "arm": arm, "run_id": configs[arm]["run_id"],
                "run_dir": str(run_dirs[arm]), "shared": shared_block,
                "train_json_val_bce": {str(r["epoch"]): r["val_loss"]
                                       for r in train_json[arm]},
                "epochs": records[arm],
            })
            D.print_eval_epoch(arm, rec)

            for name, g in diag["generator_gradients"].items():
                if g["zero_gradient_at_trained_checkpoint"]:
                    flags.append({
                        "kind": "zero-generator-gradient",
                        "detail": f"{arm} epoch {epoch}: generator at site {name} has a "
                                  f"zero-gradient fraction of 1.0 over {g['n_params']} "
                                  f"parameters at a TRAINED checkpoint. The step-0 "
                                  f"zero-init argument does not apply here."})
            gp = diag["gradient_probe"]["groups"]
            if arm != "vanilla" and gp["age"]["grad_l2"] == 0.0:
                flags.append({"kind": "zero-age-group-gradient",
                              "detail": f"{arm} epoch {epoch}: the age optimizer group has "
                                        f"gradient norm exactly 0 with "
                                        f"{gp['age']['n_params']} parameters"})
            oor = rec["overall"]["micro_histogram"]["out_of_range_fraction"]
            if oor > args.oor_flag_threshold:
                flags.append({"kind": "histogram-out-of-range",
                              "detail": f"{arm} epoch {epoch}: {oor:.3e} of scores fell "
                                        f"outside the shared histogram range [{lo}, {hi}] "
                                        f"and were clamped into the end bins"})
            for site, cr in rec["clamp_rate"].items():
                if cr > 0:
                    flags.append({"kind": "tau-clamped",
                                  "detail": f"{arm} epoch {epoch}: kernel site {site} "
                                            f"clamped tau_tilde on {cr:.3e} of pairs"})
        del model

    # ---- Part 4: selection -------------------------------------------------- #
    def best_epoch(arm: str) -> int:
        return min(common, key=lambda e: val_bce[arm][e])

    rules: dict[str, dict[str, int]] = {"per_arm_best": {a: best_epoch(a) for a in order}}
    for rule, anchor in (("vanilla_matched", "vanilla"), ("kernel_matched", "kernel")):
        if anchor in val_bce:
            rules[rule] = {a: best_epoch(anchor) for a in order}
        elif rule == args.primary_rule:
            raise AssertionError(
                f"[HARD] --primary_rule {rule} needs the {anchor!r} arm, which was not "
                f"among --runs")
        else:
            rules[rule] = {}
            flags.append({"kind": "selection-rule-unavailable",
                          "detail": f"{rule} needs the {anchor!r} arm, which was not among "
                                    f"--runs; it is written as an empty rule"})

    selection = {
        **declared,
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epochs_evaluated": common,
        "val_bce_definition": schema["val_bce"],
        "val_bce": {a: {str(e): val_bce[a][e] for e in common} for a in order},
        "train_json_val_bce": {a: {str(r["epoch"]): r["val_loss"] for r in train_json[a]}
                               for a in order},
        "rules": rules,
        "candidate_epochs": {r: sorted(set(v.values())) for r, v in rules.items()},
        "val_bce_at_candidate_epochs": {
            r: {a: {"epoch": e, "val_bce": val_bce[a][e]} for a, e in v.items()}
            for r, v in rules.items()},
        "convergence": {
            a: {
                "best_epoch": best_epoch(a),
                "best_step": {int(r["epoch"]): int(r["step"])
                              for r in train_json[a]}.get(best_epoch(a)),
                "best_val_bce": val_bce[a][best_epoch(a)],
                "val_bce_series": {str(e): val_bce[a][e] for e in common},
                "train_json_val_bce_series": {str(r["epoch"]): r["val_loss"]
                                              for r in train_json[a]},
                "train_json_note": schema["train_json_val_bce"],
            } for a in order},
    }
    for p in selection_paths:
        D.write_json(p, selection)
    D.print_selection({"primary_rule": args.primary_rule, "arms": order, "rules": rules,
                       "val_bce": selection["val_bce"]})

    # ---- summary at primary_rule -------------------------------------------- #
    chosen = rules[args.primary_rule]
    table: dict[str, dict] = {}
    for arm in order:
        e = chosen[arm]
        rec = next(r for r in records[arm] if r["epoch"] == e)
        ov, dg = rec["overall"], rec["diagnostics"]
        table[arm] = {
            "epoch": e, "step": rec["step"], "checkpoint": rec["checkpoint"],
            "val_bce": ov["val_bce"], "micro_auprc": ov["micro_auprc"],
            "macro_auprc": ov["macro_auprc"],
            "macro_n_codes_included": ov["macro_n_codes_included"],
            "recall@10": ov["recall@10"], "recall@20": ov["recall@20"],
            "ndcg@20": ov["ndcg@20"],
            "age_over_backbone": dg["gradient_probe"]["age_over_backbone"],
            "age_grad_l2": dg["gradient_probe"]["groups"]["age"]["grad_l2"],
            "kernel_separation_centered": max(
                (v["max_pairwise_centered"] for v in dg["kernel_separation"]["sites"].values()),
                default=float("nan")),
            "equal_norm_max_abs_delta_logit": dg["equal_norm_probe"]["max_abs_delta_logit"],
            "equal_norm_max_over_logit_sd": dg["equal_norm_probe"]["max_delta_over_logit_sd"],
            "by_band": rec["by_band"],
        }
    summary = {
        "_schema": schema, "primary_rule": args.primary_rule, "arms": order,
        "selected_epochs": chosen, "shared": shared_block, "table": table,
        "flags": flags, "wall_clock_s": time.time() - t_start,
    }
    D.write_json(out_root / "summary.json", summary)
    D.print_cross_arm_summary(summary)
    D.print_report_back(flags)
    D.print_block("done", [
        f"per-arm metrics : {out_root}/<arm>/epochs.json",
        f"selection       : {selection_paths[0]}  and  {selection_paths[1]}",
        f"summary         : {out_root}/summary.json",
        f"wall            : {time.time() - t_start:.1f}s",
        "Nothing was trained and no checkpoint was modified.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
