#!/usr/bin/env python3
"""Compute BGE-M3 embeddings for rolled clinical code descriptions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

LOGGER = logging.getLogger("compute_bge_embeddings")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BGE-M3 embeddings for code descriptions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "code_descriptions.json",
        help="Path to code_descriptions.json",
    )
    parser.add_argument(
        "--embeddings_out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "bge_embeddings.pt",
        help="Path to output embedding lookup table (.pt)",
    )
    parser.add_argument(
        "--vocab_out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "code_vocab.json",
        help="Path to output vocab json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for BGE encoding.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output files already exist.",
    )
    return parser.parse_args()


def cosine_for_codes(embeddings: torch.Tensor, bge_index: dict[str, int], code_a: str, code_b: str) -> float | None:
    if code_a not in bge_index or code_b not in bge_index:
        return None
    vec_a = embeddings[bge_index[code_a]].unsqueeze(0)
    vec_b = embeddings[bge_index[code_b]].unsqueeze(0)
    return float(F.cosine_similarity(vec_a, vec_b).item())


def main() -> int:
    setup_logging()
    args = parse_args()

    if args.embeddings_out.exists() and args.vocab_out.exists() and not args.force:
        print("files already exist, use --force to recompute")
        return 0

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input file: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        code_descriptions = json.load(f)
    if not isinstance(code_descriptions, dict):
        raise ValueError("code_descriptions.json must be an object mapping {code_id: description}")

    real_codes = sorted(str(c) for c in code_descriptions.keys())
    descriptions = [str(code_descriptions[c]) for c in real_codes]
    n_codes = len(real_codes)
    if n_codes == 0:
        raise ValueError("No codes found in code_descriptions.json")

    LOGGER.info("Code descriptions loaded: %d", n_codes)
    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        LOGGER.info("CUDA device count: %d", torch.cuda.device_count())
        LOGGER.info("Primary CUDA device: %s", torch.cuda.get_device_name(0))

    from FlagEmbedding import BGEM3FlagModel

    LOGGER.info("Loading BGEM3 model...")
    # M3Embedder (BGEM3FlagModel) is not a torch Module; use `devices=` instead of `.to(...)`.
    model_kwargs: dict = {"use_fp16": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        # Single GPU avoids a known multi-process tokenizer crash on some setups.
        model_kwargs["devices"] = "cuda:0"
    model = BGEM3FlagModel("BAAI/bge-m3", **model_kwargs)

    LOGGER.info("Encoding %d descriptions in batches of %d", n_codes, args.batch_size)
    dense_batches = []
    for start in tqdm(range(0, n_codes, args.batch_size), desc="BGE-M3 encoding"):
        end = min(start + args.batch_size, n_codes)
        batch_text = descriptions[start:end]
        result = model.encode(batch_text, return_dense=True)
        dense = result["dense_vecs"]
        dense_batches.append(torch.tensor(dense, dtype=torch.float32))

    real_embeddings = torch.cat(dense_batches, dim=0)  # [N, 1024]
    if real_embeddings.ndim != 2 or real_embeddings.shape[1] != 1024:
        raise ValueError(f"Expected real embedding shape [N, 1024], got {tuple(real_embeddings.shape)}")

    zeros = torch.zeros((2, 1024), dtype=torch.float32)
    all_embeddings = torch.cat([zeros, real_embeddings], dim=0)  # [N+2, 1024]
    code_ids = ["[PAD]", "[UNK]"] + real_codes

    assert len(code_ids) == all_embeddings.shape[0] == n_codes + 2

    bge_index = {c: i for i, c in enumerate(code_ids)}
    code_vocab = {c: i for i, c in enumerate(real_codes)}  # [0, N)

    # Strict offset relationship assertion.
    for c in real_codes:
        assert code_vocab[c] == bge_index[c] - 2

    args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "code_ids": code_ids,
            "embeddings": all_embeddings,
            "model": "BAAI/bge-m3",
            "embedding_dim": 1024,
        },
        args.embeddings_out,
    )
    with args.vocab_out.open("w", encoding="utf-8") as f:
        json.dump(code_vocab, f, ensure_ascii=True, indent=2)

    # Validation: reload files and verify invariants.
    emb_obj = torch.load(args.embeddings_out, map_location="cpu")
    reload_code_ids = [str(x) for x in emb_obj["code_ids"]]
    reload_embeddings = emb_obj["embeddings"].float()
    with args.vocab_out.open("r", encoding="utf-8") as f:
        reload_vocab = {str(k): int(v) for k, v in json.load(f).items()}

    # 1) Shape
    assert tuple(reload_embeddings.shape) == (n_codes + 2, 1024)
    # 2) PAD/UNK zero
    assert float(reload_embeddings[0].norm().item()) == 0.0
    assert float(reload_embeddings[1].norm().item()) == 0.0
    # 3) No NaN/Inf for real embeddings
    assert bool(torch.isfinite(reload_embeddings[2:]).all().item())
    # 4) No zero vectors in real embeddings
    assert bool((reload_embeddings[2:].norm(dim=1) > 0).all().item())

    real_norms = reload_embeddings[2:].norm(dim=1)
    print(
        "Real embedding L2 norm stats:",
        f"min={float(real_norms.min().item()):.6f}",
        f"max={float(real_norms.max().item()):.6f}",
        f"mean={float(real_norms.mean().item()):.6f}",
    )

    # 6) Offset assertion for first 10 real codes
    reload_bge_index = {c: i for i, c in enumerate(reload_code_ids)}
    first10 = real_codes[:10]
    for c in first10:
        assert reload_vocab[c] == reload_bge_index[c] - 2

    # 7) Semantic similarity spot-check
    sim_a = cosine_for_codes(reload_embeddings, reload_bge_index, "PHE_250.2", "PHE_250.1")
    sim_b = cosine_for_codes(reload_embeddings, reload_bge_index, "PHE_250.2", "CCS_231")
    if sim_a is None:
        print("Semantic check skipped: PHE_250.2 or PHE_250.1 not present")
    else:
        print(f"Cosine(PHE_250.2, PHE_250.1) = {sim_a:.4f} (expected > 0.75)")
    if sim_b is None:
        print("Semantic check skipped: PHE_250.2 or CCS_231 not present")
    else:
        print(f"Cosine(PHE_250.2, CCS_231) = {sim_b:.4f} (expected < 0.50)")

    # 8) code_vocab size == N
    assert len(reload_vocab) == n_codes

    # 9) Summary
    file_size_mb = args.embeddings_out.stat().st_size / (1024 * 1024)
    print(f"Total real codes N: {n_codes:,}")
    print(f"Embedding file size MB: {file_size_mb:.2f}")
    first5 = list(reload_vocab.items())[:5]
    print("First 5 code_vocab entries:", first5)

    LOGGER.info("Saved embeddings: %s", args.embeddings_out)
    LOGGER.info("Saved vocab: %s", args.vocab_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
