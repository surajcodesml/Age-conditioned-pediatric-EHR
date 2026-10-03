#!/usr/bin/env python3
"""Run all publication figure scripts."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPTS = [
    "synthetic_mechanism_recovery.py",
    "synthetic_lambda_curves.py",
    "nch_developmental_behavior.py",
    "synthetic_architecture_ladder.py",
    "additive_vs_softmax.py",
    "age_decoding_probe.py",
    "background_content_ablation.py",
    "synthetic_multiseed_summary.py",
    "mimic_learning_curves.py",
    "pic_results_by_age.py",
    "persistence_distribution.py",
    "umap_embeddings.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    failed = []
    for name in SCRIPTS:
        print(f"\n=== {name} ===", flush=True)
        try:
            runpy.run_path(str(here / name), run_name="__main__")
        except Exception as e:
            print(f"FAILED {name}: {e}", flush=True)
            failed.append((name, str(e)))
    if failed:
        print("\nFailures:")
        for n, e in failed:
            print(f"  {n}: {e}")
        sys.exit(1)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
