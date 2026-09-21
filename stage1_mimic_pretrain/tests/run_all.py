#!/usr/bin/env python3
"""Run Stage-1 sanity tests, then a tiny MIMIC overfit if shards are present."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage1_mimic_pretrain.config import (
    EMBEDDING_PATH,
    MIMIC_AGE_MEAN_YEARS,
    MIMIC_AGE_STD_YEARS,
    TENSORIZED_DIR,
    VOCAB_PATH,
)
from stage1_mimic_pretrain.tests.test_sanity import run_all


def _run_tiny_mimic() -> dict:
    from stage1_mimic_pretrain.train import main as train_main

    if not TENSORIZED_DIR.exists():
        return {"skipped": True, "reason": f"missing {TENSORIZED_DIR}"}
    if not EMBEDDING_PATH.exists() or not VOCAB_PATH.exists():
        return {"skipped": True, "reason": "missing embeddings or vocab"}
    stamp = time.strftime("%Y%m%d%H%M%S")
    results = {}
    for arm in ("no_interaction", "age_temporal"):
        name = f"smoke_{arm}_{stamp}"
        argv = [
            "--arm", arm,
            "--run_name", name,
            "--seed", "0",
            "--max_shards", "1",
            "--max_examples", "8",
            "--max_seq_len", "32",
            "--batch_size", "4",
            "--epochs", "2",
            "--max_steps", "6",
            "--d_model", "64",
            "--demo_hidden", "16",
            "--val_max_batches", "2",
            "--age_test_batches", "2",
            "--n_shuffle", "3",
            "--skip_corpus_stats",
            "--skip_split_ids",
            "--num_workers", "0",
            "--device", "cpu",
            "--age_mean", str(MIMIC_AGE_MEAN_YEARS),
            "--age_sd", str(MIMIC_AGE_STD_YEARS),
        ]
        t0 = time.time()
        rc = train_main(argv)
        results[arm] = {"returncode": rc, "run_name": name, "wall_s": time.time() - t0}
        if rc != 0:
            break
    return {"skipped": False, "arms": results}


def main() -> int:
    t0 = time.time()
    unit = run_all()
    mimic = _run_tiny_mimic()
    summary = {
        "unit": unit,
        "tiny_mimic": mimic,
        "wall_s": time.time() - t0,
        "all_unit_passed": unit["n_ok"] == unit["n"],
    }
    out = REPO_ROOT / "stage1_mimic_pretrain" / "run" / "smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"wrote": str(out), "unit_ok": summary["all_unit_passed"],
                      "tiny_mimic": mimic.get("skipped", False) or mimic}, indent=2))
    if not summary["all_unit_passed"]:
        return 1
    if not mimic.get("skipped"):
        arms = mimic.get("arms") or {}
        if any(v.get("returncode", 1) != 0 for v in arms.values()):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
