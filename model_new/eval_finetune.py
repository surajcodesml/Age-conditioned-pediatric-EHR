#!/usr/bin/env python3
"""Offline evaluation and checkpoint selection for finished PIC fine-tuning runs.

    python -m model_new.eval_finetune --task heart_malformations \\
        --runs model_new/run_finetune_matched_heart_malformations/{vanilla,kernel,random_constant,additive}_s0 \\
        --primary_rule per_arm_best

Nothing is trained, nothing is fine-tuned and no checkpoint is written to. This mirrors
``eval_pretrain.py`` in the four properties that are structural rather than incidental:

1. **One deterministic pass, shared by every metric and every arm.** ``shuffle=False``,
   ``drop_last=False``, no dropout, ``torch.no_grad`` everywhere except inside the
   gradient probe. The batch sequence is hashed and the hash is asserted identical for
   every arm, so a cross-arm difference cannot come from a different batch order -- and
   the paired per-patient bootstrap is only valid *because* of it.
2. **Configs are compared before anything is measured.** Differences that are a
   consequence of the arm are verified by rebuilding each model from the *shared*
   constructor kwargs and checking that the rebuild reproduces that arm's own
   ``config.json``. Anything else is a hard error unless named in ``--allow_config_diff``,
   which is then recorded in every output file.
3. **``--primary_rule`` is required, has no default, and is written to disk before any
   cross-arm number is printed.** All rules are computed and written regardless.
4. **A band with ``n < --min_band_n`` reports ``n``, ``n_pos``, ``n_neg``,
   ``unreliable: true`` and a reason, with every metric ``null``.**

Two things are specific to fine-tuning and not inherited from ``eval_pretrain``:

* **Every headline number carries a patient-level bootstrap CI, and every cross-arm
  comparison is a paired per-patient delta.** A PIC validation split is ~1,280 sequences
  with a few hundred positives; a point-estimate AUPRC difference at that size is not
  reportable, and an unpaired comparison throws away the shared-cohort variance that
  dominates it.
* **The acceptance signal for the age pathway is parameter drift from the pretrained
  backbone, not gradient norm** (README section 8). Gradients are read only for the
  per-site nonzero-gradient *fraction* of the coefficient generators, which answers a
  different question: whether the pathway still receives signal at all.

Outputs, mirroring the pretrain layout:

    model_new/run/eval_finetune/<task>/<arm>/epochs.json
    model_new/run/eval_finetune/<task>/selection.json
    model_new/run/eval_finetune/<task>/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_new import diagnostics as D
from model_new.data import (
    dataloader_worker_init, demo_layout, spans_to_tau, _sample_indices,
)
from model_new.data_finetune import TensorizedFinetuneDataset, make_finetune_collate
from model_new.encoder import build_pair_mask
from model_new.eval_pretrain import _flatten, _read_json, _same
from model_new.model import DKMModel
from model_new.optim import build_param_groups
from model_new.preflight import headroom
from model_new.train import set_seed
from model_new.train_finetune import (
    FinetuneBatchOrderHash, assert_frozen_constants, checkpoint_arm, load_backbone,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SELECTION_RULES = ("per_arm_best", "vanilla_matched", "kernel_matched")

# Config keys that identify a *run* rather than a *configuration*.
RUN_IDENTITY_KEYS = ("run_id", "arm", "timestamp", "git_dirty", "git_commit",
                     "arm_source", "pretrain_arm")
# Blocks DKMModel regenerates from (shared kwargs + the arm). Membership grants nothing on
# its own: a key under one of these prefixes is accepted only after the rebuilt model
# reproduces that arm's stored value exactly.
ARM_DERIVED_PREFIXES = ("model.", "params.", "optim.n_params.", "optim.n_tensors.",
                        "frozen_constants.", "state_dict_load.")
# Under the arm-matched design each arm legitimately loads a different checkpoint. Under
# the shared-backbone design it does not, and a difference here is unexplained.
MATCHED_ONLY_KEYS = ("pretrained_ckpt", "tau_max_source")


# --------------------------------------------------------------------------- #
# Configs                                                                     #
# --------------------------------------------------------------------------- #
def model_kwargs_from_config(cfg: dict) -> dict:
    """The constructor arguments a fine-tune run used, recovered from its ``config.json``.

    These are the *inputs*: everything the four arms are supposed to share lives here and
    must be bit-identical between them. ``arm`` is deliberately absent.
    """
    m = cfg["model"]
    f = m["fourier"]
    std = m["age_standardization"]
    return {
        "num_codes": int(m["head_out"]) if int(m["head_out"]) > 1 else None,
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
        "task": str(m["task"]),
        "tensorized_dir": str(cfg["data"]["tensorized_dir"]),
        "band_table": str(cfg.get("band_table", "adult")),
    }


def build_model(shared: dict, arm: str, num_codes: int, embedding_dim: int) -> DKMModel:
    """A classification model at the SHARED hyperparameters with one arm swapped in.

    The embedding table is a zeros placeholder of the right shape: it is a persistent
    buffer, so ``load_state_dict`` restores the real one from the fine-tune checkpoint
    rather than rebuilding it from disk (INV-FROZEN). Loading with ``strict=True`` is
    itself a check that ``config.json`` describes the architecture the checkpoint holds.
    """
    if shared["task"] != "classification":
        raise AssertionError(
            f"[HARD] eval_finetune evaluates classification checkpoints; config says "
            f"task={shared['task']!r}")
    table = torch.zeros(int(num_codes) + 2, int(embedding_dim), dtype=torch.float32)
    return DKMModel(
        num_codes=int(num_codes), embedding_table=table, arm=arm, seed=shared["seed"],
        d_model=shared["d_model"], n_layers=shared["n_layers"], n_heads=shared["n_heads"],
        use_residual=shared["use_residual"], use_layernorm=shared["use_layernorm"],
        use_ffn=shared["use_ffn"], ffn_mult=shared["ffn_mult"], s=shared["s"],
        tau_max=shared["tau_max"], age_M=shared["age_M"], age_p_min=shared["age_p_min"],
        age_p_max=shared["age_p_max"], age_hidden=shared["age_hidden"],
        gen_final_bias=shared["gen_final_bias"],
        center_delta_alpha=shared["center_delta_alpha"], demo_dim=shared["demo_dim"],
        demo_channels=shared["demo_channels"], race_encoding=shared["race_encoding"],
        demo_hidden=shared["demo_hidden"], age_mean=shared["age_mean"],
        age_sd=shared["age_sd"], task="classification",
    )


def check_configs(configs: dict[str, dict], pic_configs: dict[str, dict], order: list[str],
                  allow: set[str], num_codes: int, embedding_dim: int) -> dict:
    """HARD. All configs identical apart from the arm, its consequences and run identity.

    Same three statements as ``eval_pretrain.check_configs``, plus one that is specific to
    fine-tuning: the four arms must agree on the **backbone design** (DECISION D2). A
    comparison in which two arms are arm-matched and two are shared-vanilla answers no
    question at all, so it is refused here rather than discovered in the write-up.
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
                f"[HARD] arms {ref_arm} and {arm} were fine-tuned with different "
                f"model/data configuration, so no cross-arm comparison is valid: {detail}")

    designs = {a: (pic_configs.get(a, {}).get("backbone_design")
                   or ("arm_matched" if pic_configs.get(a, {}).get("pretrain_arm", a) == a
                       else "shared_backbone"))
               for a in order}
    if len(set(designs.values())) > 1:
        raise AssertionError(
            f"[HARD] the arms disagree about DECISION D2: {designs}. Some were fine-tuned "
            f"from their own pretrained backbone and some from a shared one; the two "
            f"designs answer different questions and cannot appear in one table.")
    design = designs[ref_arm]

    derived_ok: dict[str, dict] = {}
    for arm in order:
        m = build_model(shared, arm, num_codes, embedding_dim)
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
        elif key in MATCHED_ONLY_KEYS and design == "arm_matched":
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
        "backbone_design": design,
        "backbone_design_per_arm": designs,
        "n_shared_kwargs": len(shared),
        "shared_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in shared.items()},
        "arm_derived_differences": arm_derived,
        "arm_derived_verified_by_rebuild": True,
        "run_identity_differences": identity,
        "accepted_differences": accepted,
    }


# --------------------------------------------------------------------------- #
# The shared deterministic pass                                               #
# --------------------------------------------------------------------------- #
def make_loader(ds: TensorizedFinetuneDataset, batch_size: int, num_workers: int,
                race_encoding: str) -> DataLoader:
    kw: dict[str, Any] = dict(num_workers=num_workers,
                              collate_fn=make_finetune_collate(race_encoding),
                              pin_memory=False, worker_init_fn=dataloader_worker_init,
                              persistent_workers=num_workers > 0)
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, **kw)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def cohort_pass(loader: DataLoader) -> dict:
    """One pass over the labels alone -- no model, no scores.

    Everything that depends on the data and not on the arm is decided here, once: the
    reference batch-order hash, the per-sequence label, patient id and age band. Deciding
    them before any model runs is what makes them identical for every arm.
    """
    hasher = FinetuneBatchOrderHash()
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    ages: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for batch in loader:
        hasher.update(batch)
        labels.append(batch["labels"].numpy().astype(np.float64))
        subjects.append(np.asarray(batch["subject_id"], dtype=np.int64))
        n = batch["lengths"]
        rows = torch.arange(n.shape[0])
        ages.append(batch["age_years"][rows, n - 1].float().numpy())
        lengths.append(n.numpy())
    return {
        "hash": hasher.hexdigest,
        "order": hasher.to_json(),
        "y": np.concatenate(labels) if labels else np.zeros(0),
        "subject_id": np.concatenate(subjects) if subjects else np.zeros(0, dtype=np.int64),
        "age_last": np.concatenate(ages) if ages else np.zeros(0, dtype=np.float32),
        "lengths": np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64),
    }


@torch.no_grad()
def score_pass(model: DKMModel, loader: DataLoader, device: torch.device) -> dict:
    """The single scoring pass. One probability per sequence; nothing else is retained."""
    model.eval()
    model.reset_clamp_stats()
    hasher = FinetuneBatchOrderHash()
    logits: list[np.ndarray] = []
    for batch in loader:
        hasher.update(batch)
        out = model(_to_device(batch, device))
        logits.append(out["logits"].float().cpu().numpy())
    return {"hash": hasher.hexdigest,
            "logit": np.concatenate(logits) if logits else np.zeros(0),
            "clamp_rate": D.clamp_rates(model)}


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _metric_fn(name: str, y: np.ndarray, p: np.ndarray) -> Callable[[np.ndarray], float]:
    """A statistic of a set of row indices, for the bootstrap. Undefined resamples (no
    positives or no negatives drawn) return NaN and are counted, never imputed."""
    def f(rows: np.ndarray) -> float:
        yy, pp = y[rows], p[rows]
        if yy.size == 0 or not (0 < yy.sum() < yy.size):
            return float("nan")
        from sklearn.metrics import average_precision_score, roc_auc_score
        if name == "auroc":
            return float(roc_auc_score(yy, pp))
        if name == "auprc":
            return float(average_precision_score(yy, pp))
        if name == "ece":
            return float(D.reliability_curve(yy, pp)["ece"])
        raise ValueError(name)
    return f


def _stratum(y: np.ndarray, p: np.ndarray, subjects: np.ndarray, *, n_boot: int,
             alpha: float, seed: int, n_bins: int) -> dict:
    entry = dict(D.binary_metrics(y, p))
    entry["bce"] = (float(F.binary_cross_entropy(
        torch.from_numpy(np.clip(p, 1e-7, 1 - 1e-7)), torch.from_numpy(y)))
        if y.size else float("nan"))
    entry["calibration"] = D.reliability_curve(y, p, n_bins)
    entry["n_patients"] = int(np.unique(subjects).size)
    for metric in ("auroc", "auprc"):
        entry[f"{metric}_ci"] = D.bootstrap_ci(
            _metric_fn(metric, y, p), subjects, n_boot=n_boot, alpha=alpha, seed=seed)
    entry["ece_ci"] = D.bootstrap_ci(_metric_fn("ece", y, p), subjects, n_boot=n_boot,
                                     alpha=alpha, seed=seed)
    return entry


def assemble_metrics(cohort: dict, p: np.ndarray, *, bands, min_band_n: int, n_boot: int,
                     alpha: float, seed: int, n_bins: int) -> dict:
    y, subj = cohort["y"], cohort["subject_id"]
    overall = _stratum(y, p, subj, n_boot=n_boot, alpha=alpha, seed=seed, n_bins=n_bins)

    idx = D.band_index(cohort["age_last"], bands)
    by_band: dict[str, dict] = {}
    for i, name in enumerate(D.band_names(bands)):
        sel = idx == i
        n = int(sel.sum())
        n_pos = int(y[sel].sum()) if n else 0
        unreliable, _ = D.reliability(n, n_pos, n - n_pos, min_n=min_band_n)
        if unreliable:
            # Every metric null, with n / n_pos / n_neg and the reason. No bootstrap is
            # run: a CI from six patients is not a weaker number, it is a wrong one.
            by_band[name] = D.band_entry(
                n=n, n_pos=n_pos, n_neg=n - n_pos, min_n=min_band_n,
                metrics={"auroc": None, "auprc": None, "bce": None, "prevalence": None,
                         "calibration": None, "auroc_ci": None, "auprc_ci": None,
                         "ece_ci": None, "n_patients": None})
            continue
        metrics = _stratum(y[sel], p[sel], subj[sel], n_boot=n_boot, alpha=alpha,
                           seed=seed, n_bins=n_bins)
        by_band[name] = D.band_entry(n=n, n_pos=n_pos, n_neg=n - n_pos, min_n=min_band_n,
                                     metrics=metrics)
    return {"overall": overall, "by_band": by_band}


def paired_deltas(cohort: dict, p_arm: np.ndarray, p_ref: np.ndarray, reference_arm: str,
                  *, bands, min_band_n: int, n_boot: int, alpha: float, seed: int) -> dict:
    """Paired per-patient deltas against the reference arm, pooled and by band."""
    y, subj = cohort["y"], cohort["subject_id"]

    def block(sel: np.ndarray | None) -> dict:
        yy = y if sel is None else y[sel]
        pa = p_arm if sel is None else p_arm[sel]
        pr = p_ref if sel is None else p_ref[sel]
        ss = subj if sel is None else subj[sel]
        out: dict[str, Any] = {}
        for metric in ("auroc", "auprc"):
            fa, fr = _metric_fn(metric, yy, pa), _metric_fn(metric, yy, pr)
            rows = np.arange(yy.size)
            point = fa(rows) - fr(rows)
            out[metric] = {
                "delta": float(point),
                "arm": float(fa(rows)), "reference": float(fr(rows)),
                "ci": D.paired_bootstrap_ci(fa, fr, ss, n_boot=n_boot, alpha=alpha,
                                            seed=seed),
            }
        return out

    idx = D.band_index(cohort["age_last"], bands)
    by_band: dict[str, dict] = {}
    for i, name in enumerate(D.band_names(bands)):
        sel = idx == i
        n = int(sel.sum())
        n_pos = int(y[sel].sum()) if n else 0
        unreliable, _ = D.reliability(n, n_pos, n - n_pos, min_n=min_band_n)
        by_band[name] = D.band_entry(
            n=n, n_pos=n_pos, n_neg=n - n_pos, min_n=min_band_n,
            metrics=({"auroc": None, "auprc": None} if unreliable else block(sel)))
    return {"reference_arm": reference_arm, "overall": block(None), "by_band": by_band}


# --------------------------------------------------------------------------- #
# Mechanism diagnostics                                                       #
# --------------------------------------------------------------------------- #
def pic_lag_sample(ds: TensorizedFinetuneDataset, n_windows: int, max_pairs: int,
                   seed: int) -> np.ndarray:
    """Empirical within-window pairwise lags on the evaluated split.

    Goes through ``data.spans_to_tau`` (the numpy twin of ``data.lag_to_tau``) and
    ``encoder.build_pair_mask``, so the conditioning number below is computed on exactly
    the lags the model's kernel sees -- not on a uniform grid, which flatters every basis.
    """
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for j in _sample_indices(len(ds), n_windows, seed):
        ts = ds[int(j)]["timestamps_days"].astype(np.float64)
        if ts.size < 2:
            continue
        pair = build_pair_mask(torch.ones(1, ts.size, dtype=torch.bool))[0].numpy()
        d = spans_to_tau(np.abs(ts[:, None] - ts[None, :]))
        iu = np.triu_indices(ts.size, k=1)
        dv = d[iu][pair[iu]]
        if dv.size > max_pairs:
            dv = rng.choice(dv, max_pairs, replace=False)
        out.append(dv)
    return np.concatenate(out) if out else np.zeros(0)


def generator_gradient_probe(model: DKMModel, batches: list[dict],
                             device: torch.device) -> dict:
    """Forward + backward on fixed batches, reading the generator gradient FRACTIONS only.

    ``optimizer.step`` is never called, no optimizer is constructed and the gradients are
    cleared afterwards, so the loaded checkpoint is untouched. Group gradient *norms* are
    deliberately not reported: under Adam's second-moment normalisation a tiny gradient
    still produces a full-size step, so the acceptance signal is parameter drift (README
    section 8). The fraction is a different question -- whether the pathway receives any
    signal at all -- and at a trained checkpoint a zero fraction is a flag, not an
    explanation.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    for b in batches:
        dev = _to_device(b, device)
        out = model(dev)
        loss = F.binary_cross_entropy_with_logits(out["logits"].float(), dev["labels"])
        (loss / max(1, len(batches))).backward()
    fractions = D.generator_gradient_fractions(model)
    model.zero_grad(set_to_none=True)
    return fractions


def delta_alpha_change(finetuned: DKMModel, pretrained: DKMModel | None,
                       grid: np.ndarray) -> dict:
    """The pretrain-to-fine-tune change in ``Delta-alpha(a)`` over the pediatric grid.

    This is the actual transfer claim. Pretraining saw a minimum event age of 16.6 y, so
    everything below 18 is extrapolation from a 2-layer MLP at the end of pretraining;
    whether fine-tuning reshapes it *there* is what "the mechanism transfers" would mean.
    A large norm that has not moved is a pathway that was never trained on pediatric data.
    """
    if pretrained is None:
        return {}
    dev = next(finetuned.parameters()).device
    a = torch.as_tensor(np.asarray(grid, dtype=np.float32), device=dev)
    out: dict[str, dict] = {}
    pre_sites = dict(D.age_conditioner_sites(pretrained))
    with torch.no_grad():
        for name, cond in D.age_conditioner_sites(finetuned):
            if name not in pre_sites:
                continue
            post = cond(a)
            prev = pre_sites[name](a.to(next(pretrained.parameters()).device)).to(dev)
            change = (post - prev)
            base = float(prev.norm())
            out[name] = {
                "grid": [float(np.min(grid)), float(np.max(grid)), int(len(grid))],
                "pretrain_l2": base,
                "finetune_l2": float(post.norm()),
                "l2_change": float(change.norm()),
                "relative_change": float(change.norm() / base) if base > 0 else
                float("inf") if float(change.norm()) > 0 else 0.0,
                "max_abs_change_per_age": float(change.abs().amax(dim=-1).max()),
                "pretrain_delta_alpha": prev.cpu().numpy().tolist(),
                "finetune_delta_alpha": post.cpu().numpy().tolist(),
            }
    return out


def mechanism_diagnostics(model: DKMModel, pretrained: DKMModel | None,
                          probe_batches: list[dict], device: torch.device, *,
                          cohort: dict, tau_sample: np.ndarray, s: int, tau_max: float,
                          bands, groups, theta_pretrain, w_curve_ages, dense_grid,
                          kernel_sep_ages, tau_grid_points: int) -> dict:
    fractions = generator_gradient_probe(model, probe_batches, device)
    # After the gradient probe, and only after it: the equal-norm probe overwrites
    # alpha_base at every site and restores it, so it must not run while gradients are
    # being read.
    dev_batches = [_to_device(b, device) for b in probe_batches]
    with torch.no_grad():
        probe = dict(headroom(model, dev_batches, s))
    del dev_batches
    probe["note"] = ("the probe from README section 5 / preflight.headroom, run on PIC "
                     "batches at a fine-tuned checkpoint. preflight's stored PIC reference "
                     "is max|d logit| = 0.0059.")

    gram = D.gram_condition_numbers(tau_sample, s, tau_max)
    tt = np.clip(2.0 * tau_sample / tau_max - 1.0, -1.0, 1.0) if tau_sample.size else \
        np.zeros(0)
    gram["occupancy_fraction_of_domain"] = (float((tt.max() - tt.min()) / 2.0)
                                            if tt.size else float("nan"))
    gram["distribution"] = "empirical within-window pairwise lags on the evaluated split"

    empirical_ages = torch.from_numpy(cohort["age_last"].astype(np.float32))
    return {
        "generator_gradients": fractions,
        "equal_norm_probe": probe,
        "gram_condition_numbers": gram,
        # Both age supports, always labelled: the dense pediatric grid says what the
        # pathway CAN do, the empirical PIC distribution says what it does on data.
        "delta_alpha_norms": D.delta_alpha_norms(model, empirical_ages,
                                                 dense_grid=dense_grid),
        "alpha": {name: {k: v for k, v in entry.items() if k != "by_band"}
                  for name, entry in
                  D.alpha_diagnostics(model, empirical_ages, bands=bands).items()},
        "alpha_by_band": {name: entry["by_band"] for name, entry in
                          D.alpha_diagnostics(model, empirical_ages, bands=bands).items()},
        "delta_alpha_grid": D.delta_alpha_grid(model, ages=dense_grid),
        "w_curves": D.w_curves(model, ages=w_curve_ages),
        "kernel_separation": D.kernel_separation(model, ages=kernel_sep_ages,
                                                 n_tau=tau_grid_points),
        "param_drift_from_pretrain": (D.parameter_drift(groups, theta_pretrain)
                                      if theta_pretrain is not None else None),
        "param_drift_note": ("||theta_finetune - theta_pretrain|| / ||theta_pretrain|| per "
                             "optimizer group. This is the acceptance signal, NOT gradient "
                             "norm (README section 8)."),
        "delta_alpha_change_from_pretrain": delta_alpha_change(model, pretrained,
                                                               dense_grid),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True,
                   help="fine-tune run directories (config.json + pic_config.json + "
                        "best.pt / epoch_NNN.pt)")
    p.add_argument("--primary_rule", choices=SELECTION_RULES, required=True,
                   help="REQUIRED and never defaulted: it is written to selection.json "
                        "before any cross-arm number is printed.")
    p.add_argument("--task", type=str, default=None,
                   help="task name; by default read from pic_config.json and required to "
                        "agree across arms")
    p.add_argument("--out_root", type=Path, default=Path("model_new/run/eval_finetune"))
    p.add_argument("--allow_config_diff", nargs="*", default=[],
                   help="dotted config keys allowed to differ between arms; each one is "
                        "recorded in every output file")
    p.add_argument("--reference_arm", type=str, default="vanilla",
                   help="the arm every paired delta is taken against")

    p.add_argument("--split", type=str, default="val")
    p.add_argument("--tensorized_dir", type=Path, default=None,
                   help="override; by default from config.json and required to agree")
    p.add_argument("--batch_size", type=int, default=0, help="0 = the training batch size")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--band_table", type=str, default=None,
                   help="default: the table the runs recorded in config.json, which must "
                        "agree across arms")
    p.add_argument("--min_band_n", type=int, default=D.PEDIATRIC_MIN_BAND_N,
                   help="below this many sequences a band metric is null + unreliable")
    p.add_argument("--n_boot", type=int, default=D.N_BOOTSTRAP)
    p.add_argument("--ci_alpha", type=float, default=0.05)
    p.add_argument("--calibration_bins", type=int, default=D.N_CALIBRATION_BINS)

    p.add_argument("--grad_batches", type=int, default=8)
    p.add_argument("--tau_grid_points", type=int, default=257)
    p.add_argument("--tau_sample_windows", type=int, default=400)
    p.add_argument("--max_pairs_per_window", type=int, default=3000)
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="checkpoint file names to evaluate; default every best.pt and "
                        "epoch_NNN.pt in each run directory")
    return p


def _checkpoints_of(run_dir: Path, only: list[str] | None) -> list[Path]:
    found = sorted(run_dir.glob("epoch_*.pt")) + sorted(run_dir.glob("best.pt"))
    if only:
        found = [p for p in found if p.name in set(only)]
    return found


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    t_start = time.time()
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu")
                          else "cpu")
    flags: list[dict] = []

    # ---- runs, arms, configs ------------------------------------------------ #
    configs, pic_configs, run_dirs = {}, {}, {}
    order: list[str] = []
    for rd in args.runs:
        rd = Path(rd)
        cfg = _read_json(rd / "config.json")
        arm = cfg["arm"]
        if arm in configs:
            raise AssertionError(
                f"[HARD] two runs with arm={arm!r} were passed ({run_dirs[arm]} and {rd}); "
                f"outputs are keyed by arm, so one arm means one run")
        configs[arm], run_dirs[arm] = cfg, rd
        pic_path = rd / "pic_config.json"
        pic_configs[arm] = _read_json(pic_path) if pic_path.exists() else {}
        if not pic_configs[arm]:
            flags.append({"kind": "missing-pic-config",
                          "detail": f"{rd} has no pic_config.json, so the declarations "
                                    f"made before that run (primary endpoint, backbone "
                                    f"design, data-order hash) cannot be checked"})
        order.append(arm)

    tasks = {pic_configs[a].get("task") for a in order if pic_configs[a].get("task")}
    if len(tasks) > 1:
        raise AssertionError(f"[HARD] the runs are not all on one task: {sorted(tasks)}")
    task = args.task or (tasks.pop() if tasks else "unknown_task")
    if args.reference_arm not in order:
        raise AssertionError(
            f"[HARD] --reference_arm {args.reference_arm!r} is not among the runs "
            f"({order}); every paired delta is taken against it")

    # ---- primary_rule on disk BEFORE any cross-arm comparison --------------- #
    out_root = Path(args.out_root) / task
    selection_path = out_root / "selection.json"
    declared = {
        "task": task,
        "primary_rule": args.primary_rule,
        "declared_before_any_cross_arm_comparison": True,
        "declared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "declared; evaluation in progress",
        "arms": order,
        "reference_arm": args.reference_arm,
        "runs": {a: str(run_dirs[a]) for a in order},
        "rules_available": list(SELECTION_RULES),
        "primary_endpoint_declared_at_train_time": {
            a: {"primary_task": pic_configs[a].get("primary_task"),
                "primary_endpoint": pic_configs[a].get("primary_endpoint")} for a in order},
    }
    D.write_json(selection_path, declared)

    ref_cfg = configs[order[0]]
    batch_size = args.batch_size or int(ref_cfg["optim"]["batch_size"])
    race_encoding = ref_cfg["model"]["race_encoding"]
    num_codes = int(ref_cfg["model"].get("num_codes") or 0)
    embedding_dim = int(ref_cfg["model"]["embedding_dim"])
    if num_codes <= 0:
        # config_dict records head_out (= 1 for classification), not the vocabulary size;
        # the embedding table in the checkpoint is the authority on it.
        probe_ckpt = torch.load(_checkpoints_of(run_dirs[order[0]], args.checkpoints)[0],
                                map_location="cpu", weights_only=False)
        num_codes = int(probe_ckpt["model_state_dict"]["embedding_table"].shape[0]) - 2
        del probe_ckpt

    config_report = check_configs(configs, pic_configs, order,
                                 set(args.allow_config_diff), num_codes, embedding_dim)
    for k, v in config_report["accepted_differences"].items():
        flags.append({"kind": "config-difference-accepted",
                      "detail": f"{k} differs between arms and was accepted via "
                                f"--allow_config_diff: {v}"})
    D.print_config_check(config_report)
    shared = config_report["shared_kwargs"]

    band_table = args.band_table or shared["band_table"]
    bands = D.resolve_bands(band_table)
    tau_max = float(shared["tau_max"])
    s = int(shared["s"])

    tensorized_dir = Path(args.tensorized_dir or shared["tensorized_dir"])
    split_dir = tensorized_dir / args.split
    ds = TensorizedFinetuneDataset(split_dir, max_seq_len=int(
        ref_cfg.get("data", {}).get("max_seq_len", 1024)))
    loader = make_loader(ds, batch_size, args.num_workers, race_encoding)

    # ---- pass 0: labels only. Data facts, shared by every arm. -------------- #
    cohort = cohort_pass(loader)
    n = int(cohort["y"].size)
    idx = D.band_index(cohort["age_last"], bands)
    band_counts = {name: int((idx == i).sum())
                   for i, name in enumerate(D.band_names(bands))}
    for name, n_b in band_counts.items():
        if 0 < n_b < args.min_band_n:
            flags.append({"kind": "band-below-reliability-threshold",
                          "detail": f"age band {name} has n = {n_b} sequences on the "
                                    f"{args.split} split, below --min_band_n "
                                    f"{args.min_band_n}; every metric for it is null with "
                                    f"unreliable=true"})
    for arm in order:
        declared_order = pic_configs[arm].get("data_order", {}).get("hashes", {})
        got = declared_order.get(args.split, {}).get("hash")
        if got and got != cohort["hash"]:
            flags.append({"kind": "train-time-order-hash-differs",
                          "detail": f"{arm}: the {args.split} order hash recorded at train "
                                    f"time ({got}) differs from this pass ({cohort['hash']}"
                                    f"); --batch_size or --split differs from the run"})

    tau_sample = pic_lag_sample(ds, args.tau_sample_windows, args.max_pairs_per_window,
                                args.seed)
    probe_batches = []
    for i, b in enumerate(make_loader(ds, batch_size, 0, race_encoding)):
        if i >= args.grad_batches:
            break
        probe_batches.append(b)

    header = {
        "task": task, "primary_rule": args.primary_rule,
        "selection_path": str(selection_path),
        "runs": [str(run_dirs[a]) for a in order], "arms": order,
        "reference_arm": args.reference_arm, "split": args.split,
        "n_examples": n, "n_patients": int(np.unique(cohort["subject_id"]).size),
        "n_pos": int(cohort["y"].sum()),
        "prevalence": float(cohort["y"].mean()) if n else float("nan"),
        "batch_size": batch_size, "n_batches": cohort["order"]["n_batches"],
        "batch_order_hash": cohort["hash"], "tau_max": tau_max,
        "band_table": band_table, "min_band_n": args.min_band_n,
        "n_boot": args.n_boot, "ci_level": 1.0 - args.ci_alpha, "device": str(device),
    }
    D.print_finetune_eval_header(header)

    shared_block = {
        "harness": {
            "split": args.split, "n_sequences": n,
            "n_patients": header["n_patients"], "n_batches": header["n_batches"],
            "batch_size": batch_size, "shuffle": False, "drop_last": False,
            "seed": args.seed, "num_workers": args.num_workers, "amp": False,
            "device": str(device), "batch_order_hash": cohort["hash"],
            "batch_order_hash_definition": FinetuneBatchOrderHash.DEFINITION,
        },
        "tau_max": {"value": tau_max,
                    "source": "checkpoint buffer via DKMModel.tau_max; never re-derived"},
        "config_check": config_report,
        "cohort": {"n_sequences": n, "n_positive": int(cohort["y"].sum()),
                   "prevalence": header["prevalence"],
                   "n_patients": header["n_patients"],
                   "band_counts": band_counts,
                   "band_definition": "age at the last valid event of each sequence"},
        "age_bands": {"table": band_table,
                      "bands": {name: [lo, hi] for name, lo, hi in bands}},
        "min_band_n": args.min_band_n,
        "bootstrap": {"n_boot": args.n_boot, "alpha": args.ci_alpha, "seed": args.seed,
                      "unit": "patient (subject_id)"},
        "calibration": {"n_bins": args.calibration_bins,
                        "binning": "equal width on [0, 1], fixed across arms"},
    }
    schema = {
        "nan_encoding": "NaN and infinities serialise as null; an unreliable band carries "
                        "unreliable=true and unreliable_reason alongside the nulls",
        "auroc": "sklearn.roc_auc_score on the sequence-level probability; NaN where the "
                 "stratum has no positives or no negatives, never imputed to 0.5",
        "auprc": "sklearn.average_precision_score; the prevalence is the chance floor and "
                 "differs per band, so a banded AUPRC is not comparable to the pooled one "
                 "without it",
        "ci": "percentile bootstrap over PATIENTS resampled with replacement; row-level "
              "resampling would treat two sequences from one patient as independent",
        "paired_vs_reference": "delta computed within the same patient resample for both "
                               "arms, which is valid only because both arms are evaluated "
                               "on one hash-asserted batch sequence",
        "calibration": "equal-width reliability curve on [0, 1] with ECE, MCE and Brier",
        "param_drift_from_pretrain": "the acceptance signal for the age pathway, NOT "
                                     "gradient norm (README section 8)",
        "generator_gradients": "forward+backward on fixed batches; optimizer.step is never "
                               "called and no checkpoint is mutated. Reported as a nonzero "
                               "FRACTION, which answers whether the pathway still receives "
                               "signal -- a different question from drift.",
    }

    # ---- pass 1: per arm, per checkpoint ------------------------------------ #
    records: dict[str, list[dict]] = {a: [] for a in order}
    probs: dict[str, dict[str, np.ndarray]] = {a: {} for a in order}
    scores: dict[str, dict[str, float]] = {a: {} for a in order}

    def evaluate_arm(arm: str, with_paired: bool) -> None:
        model = build_model(shared, arm, num_codes, embedding_dim)
        theta_pretrain = None
        pretrained = None
        pre_path = configs[arm].get("pretrained_ckpt")
        if pre_path and Path(pre_path).exists():
            pre_ckpt = torch.load(pre_path, map_location="cpu", weights_only=False)
            pretrained = build_model(shared, arm, num_codes, embedding_dim)
            # The pretrain checkpoint's embedding table may be a different vocabulary
            # (DECISION D3 option b); substitute rather than relax load_backbone.
            pre_state = dict(pre_ckpt["model_state_dict"])
            if ("embedding_table" in pre_state
                    and tuple(pre_state["embedding_table"].shape)
                    != tuple(pretrained.embedding_table.shape)):
                pre_state["embedding_table"] = pretrained.embedding_table.detach().clone()
            load_backbone(pretrained, pre_state, checkpoint_arm(pre_ckpt))
            pretrained.eval().to(device)
            g_pre, _ = build_param_groups(
                pretrained, configs[arm]["optim"]["lr_backbone"],
                configs[arm]["optim"]["lr_age"], configs[arm]["optim"]["lr_head"])
            theta_pretrain = D.snapshot_parameters(g_pre)
            del pre_ckpt

        for path in _checkpoints_of(run_dirs[arm], args.checkpoints):
            t0 = time.time()
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if ckpt.get("arm") != arm:
                raise AssertionError(
                    f"[HARD] {path} holds arm={ckpt.get('arm')!r} but its run directory's "
                    f"config.json says arm={arm!r}")
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.to(device)
            frozen = assert_frozen_constants(model, ckpt)

            res = score_pass(model, loader, device)
            if res["hash"] != cohort["hash"]:
                raise AssertionError(
                    f"[HARD] arm={arm} {path.name} consumed a different batch sequence "
                    f"({res['hash']} vs the reference {cohort['hash']}); the arms are not "
                    f"being compared on the same data, and the paired bootstrap would be "
                    f"invalid.")
            p = 1.0 / (1.0 + np.exp(-res["logit"].astype(np.float64)))
            label = path.stem

            metrics = assemble_metrics(cohort, p, bands=bands, min_band_n=args.min_band_n,
                                       n_boot=args.n_boot, alpha=args.ci_alpha,
                                       seed=args.seed, n_bins=args.calibration_bins)
            groups, _ = build_param_groups(
                model, configs[arm]["optim"]["lr_backbone"],
                configs[arm]["optim"]["lr_age"], configs[arm]["optim"]["lr_head"])
            diag = mechanism_diagnostics(
                model, pretrained, probe_batches, device, cohort=cohort,
                tau_sample=tau_sample, s=s, tau_max=tau_max, bands=bands, groups=groups,
                theta_pretrain=theta_pretrain,
                w_curve_ages=D.PEDIATRIC_W_CURVE_AGES,
                dense_grid=D.PEDIATRIC_DENSE_AGE_GRID,
                kernel_sep_ages=D.PEDIATRIC_KERNEL_SEPARATION_AGES,
                tau_grid_points=args.tau_grid_points)

            rec: dict[str, Any] = {
                "arm": arm, "checkpoint": str(path), "checkpoint_label": label,
                "epoch": ckpt.get("epoch"), "tau_max": float(model.tau_max),
                "frozen_constants": frozen, "wall_clock_s": time.time() - t0,
                **metrics, "diagnostics": diag, "clamp_rate": res["clamp_rate"],
                "batch_order_hash": res["hash"],
            }
            if with_paired and arm != args.reference_arm:
                ref_p = probs[args.reference_arm].get(label)
                if ref_p is None:
                    flags.append({"kind": "no-paired-reference",
                                  "detail": f"{arm} {label}: the reference arm "
                                            f"{args.reference_arm} has no checkpoint "
                                            f"{label}, so no paired delta is reported"})
                else:
                    rec["paired_vs_reference"] = paired_deltas(
                        cohort, p, ref_p, args.reference_arm, bands=bands,
                        min_band_n=args.min_band_n, n_boot=args.n_boot,
                        alpha=args.ci_alpha, seed=args.seed)

            probs[arm][label] = p
            scores[arm][label] = rec["overall"]["auprc"]
            records[arm].append(rec)
            D.write_json(out_root / arm / "epochs.json", {
                "_schema": schema, "task": task, "arm": arm,
                "run_id": configs[arm]["run_id"], "run_dir": str(run_dirs[arm]),
                "pic_config": pic_configs[arm], "shared": shared_block,
                "checkpoints": records[arm],
            })
            D.print_finetune_eval_epoch(arm, rec, bands=bands)

            for name, g in diag["generator_gradients"].items():
                if g["zero_gradient_at_trained_checkpoint"]:
                    flags.append({
                        "kind": "zero-generator-gradient",
                        "detail": f"{arm} {label}: the generator at site {name} has a "
                                  f"zero-gradient fraction of 1.0 over {g['n_params']} "
                                  f"parameters at a trained checkpoint"})
            drift = diag.get("param_drift_from_pretrain") or {}
            if arm != "vanilla" and drift.get("age") == 0.0:
                flags.append({"kind": "age-group-did-not-move",
                              "detail": f"{arm} {label}: the age optimizer group has "
                                        f"exactly zero drift from the pretrained backbone"})
            for site, cr in res["clamp_rate"].items():
                if cr > 0:
                    flags.append({"kind": "tau-clamped",
                                  "detail": f"{arm} {label}: kernel site {site} clamped "
                                            f"tau_tilde on {cr:.3e} of pairs"})
            del ckpt
        del model, pretrained

    # The reference arm runs first so every other arm can take its paired delta in the
    # same loop, against probabilities from the identical batch sequence.
    evaluate_arm(args.reference_arm, with_paired=False)
    for arm in order:
        if arm != args.reference_arm:
            evaluate_arm(arm, with_paired=True)

    # ---- selection ---------------------------------------------------------- #
    labels = sorted(set.intersection(*[set(scores[a]) for a in order])) if order else []
    if not labels:
        raise AssertionError("[HARD] the runs share no checkpoint label to compare at")

    def best_label(arm: str) -> str:
        finite = [l for l in labels if np.isfinite(scores[arm][l])]
        pool = finite or labels
        return max(pool, key=lambda l: scores[arm][l])

    rules: dict[str, dict[str, str]] = {"per_arm_best": {a: best_label(a) for a in order}}
    for rule, anchor in (("vanilla_matched", "vanilla"), ("kernel_matched", "kernel")):
        if anchor in scores:
            rules[rule] = {a: best_label(anchor) for a in order}
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
        "checkpoints_evaluated": labels,
        "selection_metric": "val AUPRC from this full deterministic pass",
        "auprc": {a: {l: scores[a][l] for l in labels} for a in order},
        "rules": rules,
    }
    D.write_json(selection_path, selection)

    # ---- summary at primary_rule -------------------------------------------- #
    chosen = rules[args.primary_rule]
    table: dict[str, dict] = {}
    for arm in order:
        label = chosen[arm]
        rec = next(r for r in records[arm] if r["checkpoint_label"] == label)
        ov, dg = rec["overall"], rec["diagnostics"]
        pv = (rec.get("paired_vs_reference") or {}).get("overall", {}).get("auprc")
        drift = dg.get("param_drift_from_pretrain") or {}
        table[arm] = {
            "checkpoint_label": label, "checkpoint": rec["checkpoint"],
            "auroc": ov["auroc"], "auroc_ci": ov["auroc_ci"],
            "auprc": ov["auprc"], "auprc_ci": ov["auprc_ci"],
            "ece": ov["calibration"]["ece"], "brier": ov["calibration"]["brier"],
            "bce": ov["bce"], "prevalence": ov["prevalence"],
            "delta_auprc": (pv or {}).get("delta"),
            "delta_auprc_lo": ((pv or {}).get("ci") or {}).get("lo"),
            "delta_auprc_hi": ((pv or {}).get("ci") or {}).get("hi"),
            "equal_norm_max_abs_delta_logit":
                dg["equal_norm_probe"]["max_abs_delta_logit"],
            "delta_alpha_pediatric_max": max(
                (v["dense_uniform_grid"]["max"] for v in dg["delta_alpha_norms"].values()),
                default=float("nan")),
            "param_drift_age": drift.get("age"),
            "param_drift_backbone": drift.get("backbone"),
            "gram_cond_no_constant":
                dg["gram_condition_numbers"]["chebyshev_no_constant"],
            "by_band": rec["by_band"],
            "paired_vs_reference": rec.get("paired_vs_reference"),
        }
    summary = {
        "_schema": schema, "task": task, "primary_rule": args.primary_rule,
        "arms": order, "reference_arm": args.reference_arm,
        "selected_checkpoints": chosen, "shared": shared_block, "table": table,
        "flags": flags, "wall_clock_s": time.time() - t_start,
    }
    D.write_json(out_root / "summary.json", summary)
    D.print_finetune_cross_arm_summary(summary)
    D.print_report_back(flags)
    D.print_block("done", [
        f"per-arm metrics : {out_root}/<arm>/epochs.json",
        f"selection       : {selection_path}",
        f"summary         : {out_root}/summary.json",
        f"wall            : {time.time() - t_start:.1f}s",
        "Nothing was trained and no checkpoint was modified.",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
