#!/usr/bin/env python3
"""Compute BGE-M3 embeddings for PIC code descriptions (reuses the MIMIC routine).

This is a thin driver that invokes the parent ``compute_bge_embeddings`` routine
unchanged (same BAAI/bge-m3 model, batch config, PAD/UNK layout and validation) on the
PIC ``code_descriptions_pic.json``, producing:
- ``code_vocab_pic.json``     ({code_id: int_index} for N real codes, no PAD/UNK)
- ``bge_embeddings_pic.pt``   ({code_ids: [N+2], embeddings: FloatTensor[N+2,1024], ...},
                                index 0 = PAD, 1 = UNK) — identical format to MIMIC's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _shared import run_bge_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BGE-M3 embeddings for PIC descriptions.")
    parser.add_argument(
        "--pic_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "pic",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pic_dir: Path = args.pic_dir
    return run_bge_embeddings(
        input_json=pic_dir / "code_descriptions_pic.json",
        embeddings_out=pic_dir / "bge_embeddings_pic.pt",
        vocab_out=pic_dir / "code_vocab_pic.json",
        batch_size=args.batch_size,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
