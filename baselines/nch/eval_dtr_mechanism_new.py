#!/usr/bin/env python3
"""Post-train mechanism metrics for NCH Content-Persistence DTR ``*_new``.

Reports for each arm under results/baselines/nch/dtr_*_new/:
  theta0, beta, lambda(age) at pediatric ages 1/5/10/15/18,
  beta=0 inference ΔBCE (age_temporal), age-shuffle ΔBCE,
  BCE by developmental age band when ages are available.

Writes ``results/baselines/nch/dtr_new_mechanism.json``.
Does not touch legacy unsuffixed DTR dirs / all_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stage2_nch.dataset import (  # noqa: E402
    NCHForecastDataset as TensorizedPretrainDataset,
    make_nch_collate as make_collate,
)
from stage2_nch.config import AGE_BANDS, age_band_name  # noqa: E402
from baselines.mimic.runner import CP_D_MODEL, MIMICCPDTRAdapter, make_cp_collate  # noqa: E402
from baselines.nch.runner import MAX_SEQ_LEN, MODEL_SEED, TEST_MAX_BATCHES  # noqa: E402
from baselines.common.training import evaluate_loader, get_device, set_seed  # noqa: E402

PROBE_AGES = [1.0, 5.0, 10.0, 15.0, 18.0]
ARMS = ("temporal_only", "age_temporal")


def _load_arm(arm: str, n_codes: int, run_dir: Path, device: torch.device) -> MIMICCPDTRAdapter:
    model = MIMICCPDTRAdapter(n_codes=n_codes, arm=arm, d_model=CP_D_MODEL)
    model.load_checkpoint(run_dir)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def lambda_curve(model: MIMICCPDTRAdapter, ages: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for a in ages:
        age_t = torch.tensor([a], device=next(model.parameters()).device, dtype=torch.float32)
        lam = model.model.lambda_of(
            age_t, persistence_offset=torch.zeros(1, device=age_t.device)
        )
        out[str(a)] = float(lam.squeeze().cpu())
    out["_theta0"] = float(model.model.theta0.detach().cpu())
    out["_beta"] = float(model.model.beta.detach().cpu())
    out["_age_temporal"] = bool(model.model.age_temporal)
    return out


def _bce_on_loader(model, loader, device, max_batches: int) -> float:
    metrics = evaluate_loader(
        model, model.predict, loader, device,
        max_batches=max_batches, metrics_mode="bce",
    )
    return float(metrics["bce"])


def beta0_delta_bce(model, loader, device, max_batches: int) -> dict[str, float]:
    base = _bce_on_loader(model, loader, device, max_batches)
    saved = float(model.model.beta.detach().cpu())
    with torch.no_grad():
        model.model.beta.fill_(0.0)
    b0 = _bce_on_loader(model, loader, device, max_batches)
    with torch.no_grad():
        model.model.beta.fill_(saved)
    return {
        "bce_full": base,
        "bce_beta0": b0,
        "delta_bce": b0 - base,
        "beta_restored": float(model.model.beta.detach().cpu()),
    }


def age_shuffle_delta_bce(model, loader, device, max_batches: int, seed: int = 0) -> dict[str, float]:
    base = _bce_on_loader(model, loader, device, max_batches)
    rng = np.random.default_rng(seed)

    class AgeShuffleLoader:
        def __init__(self, inner):
            self.inner = inner

        def __iter__(self):
            for batch in self.inner:
                b = dict(batch)
                age = b["age"].clone()
                perm = rng.permutation(age.shape[0])
                b["age"] = age[perm]
                yield b

        def __len__(self):
            return len(self.inner)

    sbce = _bce_on_loader(model, AgeShuffleLoader(loader), device, max_batches)
    return {
        "bce_full": base,
        "bce_age_shuffle": sbce,
        "delta_bce": sbce - base,
    }


@torch.no_grad()
def bce_by_age_band(model, loader, device, max_batches: int) -> dict[str, Any]:
    """Per developmental age-band mean BCE (example-level)."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    n_batches = 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        n_batches += 1
        age = batch["age"].to(device)
        labels = batch["labels"].to(device)
        pred = model.predict(
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        )
        logits = pred["logits"]
        # per-example mean BCE across codes
        per_ex = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        ).mean(dim=1)
        for i in range(age.shape[0]):
            band = age_band_name(float(age[i].cpu()))
            sums[band] += float(per_ex[i].cpu())
            counts[band] += 1
    out = {}
    for name, _, _ in AGE_BANDS:
        if counts[name]:
            out[name] = {
                "bce": sums[name] / counts[name],
                "n": counts[name],
            }
        else:
            out[name] = {"bce": None, "n": 0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name-suffix", default="_new")
    ap.add_argument("--test_max_batches", type=int, default=TEST_MAX_BATCHES)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--tensorized_dir",
        default="artifacts/nch_stage2/v2/tensorized_forecast/diagnoses_only",
    )
    ap.add_argument("--vocab_path", default="data/processed/code_vocab.json")
    args = ap.parse_args()

    set_seed(MODEL_SEED)
    device = get_device(args.device)
    out_root = REPO_ROOT / "results" / "baselines" / "nch"
    suffix = args.name_suffix

    test_ds = TensorizedPretrainDataset(
        REPO_ROOT / args.tensorized_dir / "test",
        REPO_ROOT / args.vocab_path,
        max_seq_len=MAX_SEQ_LEN,
    )
    n_codes = test_ds.num_codes
    collate = make_cp_collate(make_collate("one_hot"))
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )

    init_verify_path = out_root / f"dtr{suffix}_init_verify.json"
    init_verify = None
    if init_verify_path.exists():
        with init_verify_path.open() as f:
            init_verify = json.load(f)

    report: dict[str, Any] = {
        "name_suffix": suffix,
        "architecture": "Content-Persistence DTR",
        "seed": MODEL_SEED,
        "test_max_batches": args.test_max_batches,
        "init_verify": init_verify,
        "arms": {},
    }

    for arm in ARMS:
        run_dir = out_root / f"dtr_{arm}{suffix}"
        result_path = run_dir / "result.json"
        if not result_path.exists():
            report["arms"][arm] = {"error": f"missing {result_path}"}
            continue
        with result_path.open() as f:
            result = json.load(f)
        model = _load_arm(arm, n_codes, run_dir, device)
        lam = lambda_curve(model, PROBE_AGES)
        entry: dict[str, Any] = {
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "best_epoch": result.get("train", {}).get("best_epoch"),
            "best_val_bce": result.get("best_val_bce") or result.get("train", {}).get("best_val_bce"),
            "pretrained_from": result.get("pretrained_from"),
            "pretrained_fingerprint_ex_beta": result.get("pretrained_fingerprint_ex_beta"),
            "beta_init": result.get("beta_init"),
            "test_metrics": result.get("test_metrics"),
            "AUROC": result.get("AUROC"),
            "AUPRC": result.get("AUPRC"),
            "BCE": result.get("BCE"),
            "theta0": lam["_theta0"],
            "beta": lam["_beta"],
            "lambda_of_age": {k: v for k, v in lam.items() if not k.startswith("_")},
            "checkpoint": str(run_dir / "best_checkpoint.pt"),
        }
        entry["age_shuffle"] = age_shuffle_delta_bce(
            model, loader, device, args.test_max_batches, seed=MODEL_SEED,
        )
        entry["bce_by_age_band"] = bce_by_age_band(
            model, loader, device, args.test_max_batches,
        )
        if arm == "age_temporal":
            entry["beta0"] = beta0_delta_bce(model, loader, device, args.test_max_batches)
        report["arms"][arm] = entry
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    at = report["arms"].get("age_temporal", {})
    to = report["arms"].get("temporal_only", {})
    if "test_metrics" in at and "test_metrics" in to:
        tm_at, tm_to = at["test_metrics"], to["test_metrics"]
        report["delta_age_temporal_minus_temporal_only"] = {
            k: (tm_at.get(k) - tm_to.get(k))
            if isinstance(tm_at.get(k), (int, float)) and isinstance(tm_to.get(k), (int, float))
            else None
            for k in ("bce", "micro_auroc", "micro_auprc", "precision@5", "recall@5")
        }
        report["delta_age_temporal_minus_temporal_only"]["beta_at"] = at.get("beta")
        report["delta_age_temporal_minus_temporal_only"]["beta_to"] = to.get("beta")
        report["delta_age_temporal_minus_temporal_only"]["theta0_at"] = at.get("theta0")
        report["delta_age_temporal_minus_temporal_only"]["theta0_to"] = to.get("theta0")
        fp_at = at.get("pretrained_fingerprint_ex_beta")
        fp_to = to.get("pretrained_fingerprint_ex_beta")
        report["delta_age_temporal_minus_temporal_only"]["init_fingerprints_match"] = (
            fp_at is not None and fp_at == fp_to
        )

    out_path = out_root / f"dtr{suffix}_mechanism.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
