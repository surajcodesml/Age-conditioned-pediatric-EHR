"""Thin import shim that re-exports reusable helpers from the parent MIMIC pipeline.

The parent ``preprocessing/`` scripts are standalone modules (not an installed
package), so we put the parent directory on ``sys.path`` and import the functions
we need. Nothing in the MIMIC scripts is modified — this is read-only reuse.

Re-exported helpers:
- ``create_phecode_maps`` (PheWAS ICD-9/ICD-10 -> PheCode map loading/building)
- ``save_json`` / ``load_json`` (consistent JSON IO with the MIMIC pipeline)
- ``assign_splits`` / ``write_split`` (patient-level stratified split logic)
- ``run_bge_embeddings`` (BGE-m3 embedding routine, invoked via the MIMIC CLI main)
"""

from __future__ import annotations

import sys
from pathlib import Path

# parents[1] == preprocessing/  (the parent package holding the MIMIC scripts).
_PREPROC_DIR = Path(__file__).resolve().parents[1]
if str(_PREPROC_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROC_DIR))

from rollup_and_describe import create_phecode_maps, load_json, save_json  # noqa: E402
from build_splits import assign_splits, write_split  # noqa: E402
import compute_bge_embeddings as _bge  # noqa: E402


def run_bge_embeddings(
    input_json: Path,
    embeddings_out: Path,
    vocab_out: Path,
    batch_size: int = 64,
    force: bool = True,
) -> int:
    """Invoke the MIMIC BGE-m3 routine unchanged via its CLI main().

    We drive ``compute_bge_embeddings.main()`` by temporarily setting ``sys.argv`` so
    the exact same model, config, PAD/UNK layout and validation run for PIC.
    """
    argv_backup = sys.argv[:]
    sys.argv = [
        "compute_bge_embeddings.py",
        "--input", str(input_json),
        "--embeddings_out", str(embeddings_out),
        "--vocab_out", str(vocab_out),
        "--batch_size", str(int(batch_size)),
    ]
    if force:
        sys.argv.append("--force")
    try:
        return int(_bge.main())
    finally:
        sys.argv = argv_backup


__all__ = [
    "create_phecode_maps",
    "load_json",
    "save_json",
    "assign_splits",
    "write_split",
    "run_bge_embeddings",
]
