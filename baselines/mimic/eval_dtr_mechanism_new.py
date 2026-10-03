#!/usr/bin/env python3
"""Post-train mechanism metrics for MIMIC Content-Persistence DTR ``*_new``.

Reports for each arm under results/baselines/mimic/dtr_*_new/:
  theta0, beta, lambda(age) at representative ages,
  beta=0 inference ΔBCE, age-shuffle ΔBCE.

Writes ``results/baselines/mimic/dtr_new_mechanism.json`` and prints a comparison.
Does not touch legacy unsuffixed DTR dirs / all_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model_new.data import (  # noqa: E402
    TensorizedPretrainDataset,
    make_collate,
)
from baselines.mimic.runner import (  # noqa: E402
    CP_D_MODEL,
    EncounterLoader,
    MIMICCPDTRAdapter,
    RenameLoader,
    MAX_SEQ_LEN,
    MODEL_SEED,
    TEST_MAX_BATCHES,
)
from baselines.common.training import evaluate_loader, get_device, set_seed  # noqa: E402

# Representative MIMIC ages (years) for λ(a) curve.
PROBE_AGES = [1.0, 5.0, 10.0, 18.0, 40.0, 65.0, 80.0]
ARMS = ("temporal_only", "age_temporal")


def _load_arm(arm: str, n_codes: int, run_dir: Path, device: torch.device) -> MIMICCPDTRAdapter:
    model = MIMICCPDTRAdapter(n_codes=n_codes, arm=arm, d_model=CP_D_MODEL)
    model.load_checkpoint(run_dir)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def lambda_curve(model: MIMICCPDTRAdapter, ages: list[float]) -> dict[str, float]:
    gate = model.model.gate
    out = {}
    for a in ages:
        age_t = torch.tensor([a], device=next(model.parameters()).device, dtype=torch.float32)
        # Content-independent λ probe: persistence_offset=0 → softplus(θ₀ + β z(a))
        lam = model.model.lambda_of(age_t, persistence_offset=torch.zeros(1, device=age_t.device))
        out[str(a)] = float(lam.squeeze().cpu())
    # Also record theta0/beta
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
    """ΔBCE = BCE(β←0) − BCE(full) on age_temporal."""
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
    """Permute ages across the batch; ΔBCE = BCE(shuffled) − BCE(full)."""
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

    shuf = AgeShuffleLoader(loader)
    sbce = _bce_on_loader(model, shuf, device, max_batches)
    return {
        "bce_full": base,
        "bce_age_shuffle": sbce,
        "delta_bce": sbce - base,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name-suffix", default="_new")
    ap.add_argument("--test_max_batches", type=int, default=TEST_MAX_BATCHES)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tensorized_dir", default="data/processed/tensorized_flat")
    ap.add_argument("--vocab_path", default="data/processed/code_vocab.json")
    args = ap.parse_args()

    set_seed(MODEL_SEED)
    device = get_device(args.device)
    out_root = REPO_ROOT / "results" / "baselines" / "mimic"
    suffix = args.name_suffix

    test_ds = TensorizedPretrainDataset(
        REPO_ROOT / args.tensorized_dir / "test",
        REPO_ROOT / args.vocab_path,
        max_seq_len=MAX_SEQ_LEN,
    )
    n_codes = test_ds.num_codes
    collate = make_collate("one_hot")
    base = RenameLoader(
        DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=0, pin_memory=False,
        )
    )
    loader = EncounterLoader(base)

    report: dict[str, Any] = {
        "name_suffix": suffix,
        "architecture": "Content-Persistence DTR",
        "seed": MODEL_SEED,
        "test_max_batches": args.test_max_batches,
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
            "val_history_tail": (result.get("train", {}) or {}).get("history", [])[-1:],
            "test_metrics": result.get("test_metrics"),
            "theta0": lam["_theta0"],
            "beta": lam["_beta"],
            "lambda_of_age": {k: v for k, v in lam.items() if not k.startswith("_")},
            "checkpoint": str(run_dir / "best_checkpoint.pt"),
        }
        # Mechanism ablations only meaningful when β can move (AT); still report TO.
        entry["age_shuffle"] = age_shuffle_delta_bce(
            model, loader, device, args.test_max_batches, seed=MODEL_SEED,
        )
        if arm == "age_temporal":
            entry["beta0"] = beta0_delta_bce(model, loader, device, args.test_max_batches)
        report["arms"][arm] = entry
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Direct AT − TO comparison from result.json metrics
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

    out_path = out_root / f"dtr{suffix}_mechanism.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
