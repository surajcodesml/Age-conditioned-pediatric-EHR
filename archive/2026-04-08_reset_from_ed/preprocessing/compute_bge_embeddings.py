#!/usr/bin/env python3
"""Compute BGE embeddings for ED code descriptions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute BGE embeddings for description strings.")
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "ed_code_descriptions.csv",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace/SentenceTransformers model id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "ed_code_embeddings.npz",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.descriptions)
    texts = df["description"].fillna("").astype(str).tolist()
    model = SentenceTransformer(args.model_name)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_source=df["event_source"].astype(str).to_numpy(),
        event_code=df["event_code"].astype(str).to_numpy(),
        description=df["description"].astype(str).to_numpy(),
        embeddings=np.asarray(vectors, dtype=np.float32),
    )
    print(f"Wrote embeddings for {len(df):,} codes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
