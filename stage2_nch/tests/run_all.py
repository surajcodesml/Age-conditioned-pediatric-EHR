#!/usr/bin/env python3
"""Run Stage-2 unit tests, then a tiny-NCH overfit if processed data is present."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2_nch.config import EMBEDDING_PATH, NCH_CLEAN_EVENTS, STAGE1_BEST_CKPT, VOCAB_PATH
from stage2_nch.tests.test_sanity import run_all


def _run_tiny_nch() -> dict:
    from stage2_nch.train import main as train_main

    if not NCH_CLEAN_EVENTS.exists():
        return {"skipped": True, "reason": f"missing {NCH_CLEAN_EVENTS}"}
    if not STAGE1_BEST_CKPT.exists() or not EMBEDDING_PATH.exists() or not VOCAB_PATH.exists():
        return {"skipped": True, "reason": "missing Stage-1 checkpoint, embeddings, or vocab"}
    stamp = time.strftime("%Y%m%d%H%M%S")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    results = {}
    for arm in ("no_interaction", "age_temporal"):
        name = f"smoke_{arm}_{stamp}"
        argv = [
            "--arm", arm,
            "--run_name", name,
            "--seed", "0",
            "--max_examples", "8",
            "--max_seq_len", "32",
            "--batch_size", "4",
            "--epochs", "2",
            "--max_steps", "6",
            "--patience", "0",
            "--d_model", "256",
            "--demo_hidden", "64",
            "--val_max_batches", "2",
            "--max_metric_examples", "8",
            "--age_test_batches", "2",
            "--n_shuffle", "3",
            "--num_workers", "0",
            "--device", device,
        ]
        t0 = time.time()
        rc = train_main(argv)
        results[arm] = {"returncode": rc, "run_name": name, "wall_s": time.time() - t0}
        if rc != 0:
            break
    return {"skipped": False, "device": device, "arms": results}


def main() -> int:
    t0 = time.time()
    unit = run_all()
    nch = _run_tiny_nch()
    summary = {
        "unit": unit,
        "tiny_nch": nch,
        "wall_s": time.time() - t0,
        "all_unit_passed": unit["n_ok"] == unit["n"],
    }
    out = REPO_ROOT / "stage2_nch" / "run" / "smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"wrote": str(out), "unit_ok": summary["all_unit_passed"],
                      "tiny_nch": nch.get("skipped", False) or nch}, indent=2))
    if not summary["all_unit_passed"]:
        return 1
    if not nch.get("skipped"):
        arms = nch.get("arms") or {}
        if any(v.get("returncode", 1) != 0 for v in arms.values()):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
