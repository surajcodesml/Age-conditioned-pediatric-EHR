"""T1 — Pretraining-objective parity on the held-out split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_new import diagnostics as D
from model_new.audit.common import (
    ARMS,
    build_model,
    iter_batches,
    load_checkpoint,
    to_device,
    age_last_of,
)
from model_new.eval_pretrain import BatchOrderHash


@torch.no_grad()
def _eval_arm(model, loader, device, max_batches, ks=(5, 10, 20)) -> dict:
    model.eval()
    hasher = BatchOrderHash()
    bce_per: list[np.ndarray] = []
    recall: dict[int, list[np.ndarray]] = {k: [] for k in ks}
    ages: list[np.ndarray] = []

    for batch in iter_batches(loader, max_batches):
        hasher.update(batch)
        b = to_device(batch, device)
        logits = model(b)["code_logits"].float()
        targets = b["target_codes"].float()
        # Per-example mean BCE over codes (matches training val_loss scale).
        per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=-1)
        bce_per.append(per.cpu().numpy())
        ages.append(age_last_of(batch).cpu().numpy())
        for k, v in D.topk_per_example(logits, targets, ks=ks).items():
            if k.startswith("recall@"):
                kk = int(k.split("@")[1])
                if kk in recall:
                    recall[kk].append(v.numpy())

    age_arr = np.concatenate(ages) if ages else np.zeros(0, np.float32)
    bce_arr = np.concatenate(bce_per) if bce_per else np.zeros(0, np.float64)
    rec = {k: np.concatenate(v) if v else np.zeros(0, np.float64) for k, v in recall.items()}
    bands = D.band_index(age_arr)
    out = {
        "hash": hasher.hexdigest,
        "n_examples": int(hasher.n_rows),
        "bce_mean": float(bce_arr.mean()) if bce_arr.size else float("nan"),
        "bce_per_example": bce_arr,
        "age_last": age_arr,
        "band": bands,
        "recall": {f"recall@{k}": float(rec[k].mean()) if rec[k].size else float("nan")
                   for k in ks},
        "recall_per_example": {f"recall@{k}": rec[k] for k in ks},
    }
    # Stratified.
    by_band = {}
    for i, (name, _, _) in enumerate(D.AGE_BANDS):
        sel = bands == i
        n = int(sel.sum())
        entry: dict = {"n": n}
        if n:
            entry["bce"] = float(bce_arr[sel].mean())
            for k in ks:
                entry[f"recall@{k}"] = float(rec[k][sel].mean())
        by_band[name] = entry
    out["by_band"] = by_band
    return out


def run_t1(ctx: dict, *, n_boot: int = 1000) -> dict:
    shared = ctx["shared"]
    device = ctx["device"]
    selected = ctx["selected"]
    max_batches = ctx["max_val_batches"]
    patient_ids = ctx["patient_ids"]
    seed = ctx["seed"]
    order = ctx["order"]

    from model_new.eval_pretrain import make_val_loader

    ds = ctx["dataset"]
    results: dict[str, dict] = {}
    hashes = {}
    for arm in order:
        m = build_model(shared, arm)
        load_checkpoint(m, Path(selected[arm]["checkpoint"]), arm=arm,
                        epoch=selected[arm]["epoch"], device=device)
        # Rebuild loader each arm; shuffle=False keeps the hash-asserted order.
        loader = make_val_loader(ds, ctx["batch_size"], ctx["num_workers"],
                                 shared["race_encoding"])
        res = _eval_arm(m, loader, device, max_batches)
        hashes[arm] = res["hash"]
        results[arm] = res
        del m

    if len(set(hashes.values())) != 1:
        raise AssertionError(
            f"[HARD] batch-order hash differs across arms: {hashes}; "
            f"paired comparison is invalid")

    if hashes[order[0]] != ctx["batch_order_hash"]:
        raise AssertionError(
            f"[HARD] eval hash {hashes[order[0]]} != context hash {ctx['batch_order_hash']}")

    # Point estimates + paired bootstrap kernel - random_constant.
    ks = (5, 10, 20)
    summary = {arm: {
        "bce": results[arm]["bce_mean"],
        **results[arm]["recall"],
        "by_band": results[arm]["by_band"],
        "epoch": selected[arm]["epoch"],
    } for arm in order}

    def _mean(arr):
        def stat(rows):
            if rows.size == 0:
                return float("nan")
            return float(arr[rows].mean())
        return stat

    deltas = {}
    for name, key in [("bce", "bce_per_example")] + [
            (f"recall@{k}", None) for k in ks]:
        if name.startswith("recall@"):
            a = results["kernel"]["recall_per_example"][name]
            b = results["random_constant"]["recall_per_example"][name]
        else:
            a = results["kernel"][key]
            b = results["random_constant"][key]
        point = float(a.mean() - b.mean())
        ci = D.paired_bootstrap_ci(_mean(a), _mean(b), patient_ids,
                                   n_boot=n_boot, seed=seed)
        # For BCE, "good" is lower; for recall, higher. Record raw kernel - rc.
        covers_zero = bool(ci["lo"] <= 0.0 <= ci["hi"])
        deltas[name] = {
            "kernel_minus_random_constant": point,
            "ci": ci,
            "covers_zero": covers_zero,
        }

    # Prominent flag: if all recall CIs and BCE CI cover 0, age route contributed nothing measurable.
    measurable = any(not d["covers_zero"] for d in deltas.values())
    report = {
        "batch_order_hash": hashes[order[0]],
        "n_examples": results[order[0]]["n_examples"],
        "n_boot": n_boot,
        "per_arm": summary,
        "kernel_minus_random_constant": deltas,
        "age_route_measurable_on_pretrain_objective": measurable,
        "headline": (
            "Age route contributed a measurable pretraining-objective difference "
            "(kernel − random_constant CI excludes 0)."
            if measurable else
            "PROMINENT: every kernel − random_constant CI covers 0 — the age route "
            "contributed nothing measurable during pretraining; no fine-tune result "
            "can be attributed to it on this evidence."
        ),
    }
    # Drop bulky arrays from return (kept only for CI computation above).
    D.print_block("T1 pretraining-objective parity", [
        report["headline"],
        *[f"{arm}: bce={summary[arm]['bce']:.6f}  "
          f"r@5/10/20="
          f"{summary[arm]['recall@5']:.4f}/"
          f"{summary[arm]['recall@10']:.4f}/"
          f"{summary[arm]['recall@20']:.4f}"
          for arm in order],
        *[f"Δ({name})={d['kernel_minus_random_constant']:+.6e}  "
          f"CI[{d['ci']['lo']:+.6e},{d['ci']['hi']:+.6e}]  "
          f"covers0={d['covers_zero']}"
          for name, d in deltas.items()],
    ])
    return report
