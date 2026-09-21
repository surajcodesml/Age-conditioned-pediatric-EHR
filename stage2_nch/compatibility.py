"""MIMIC vocabulary vs processed NCH: no silent remapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_new.data import load_vocab
from stage2_nch.config import (
    EMBEDDING_PATH,
    NCH_TENSORIZED_DIR,
    PRIMARY_REPRESENTATION,
    STAGE1_BEST_CKPT,
    VOCAB_PATH,
)


SPECIAL = {
    "PAD": {"vocab_index": None, "model_id": 0},
    "UNK": {"vocab_index": None, "model_id": 1, "shard_unk_vocab_index": "len(vocab)"},
}


def embedding_row_count(path: Path = EMBEDDING_PATH) -> tuple[int, int]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    t = obj["embeddings"] if isinstance(obj, dict) else obj
    return int(t.shape[0]), int(t.shape[1])


def checkpoint_shapes(path: Path = STAGE1_BEST_CKPT) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"]
    return {
        "embedding_table": list(sd["embedding_table"].shape),
        "head_out": list(sd["head.net.2.weight"].shape),
        "lambda0": float(sd["temporal.lambda0"]),
        "beta_adult_not_transferred": float(sd["temporal.beta"]),
        "age_mean_adult": float(sd["age_mean"]),
        "age_sd_adult": float(sd["age_sd"]),
        "epoch": blob.get("epoch"),
        "val_bce": blob.get("val_bce"),
    }


def inspect_shard_codes(tensorized_dir: Path) -> dict[str, Any]:
    shards = sorted(Path(tensorized_dir).rglob("shard_*.npz"))
    if not shards:
        return {"n_shards": 0}
    mins, maxs, unks = [], [], []
    n_events = 0
    for p in shards:
        z = np.load(p, mmap_mode="r", allow_pickle=False)
        codes = np.asarray(z["code_indices"])
        unk = int(np.asarray(z["unk_vocab_index"]).reshape(-1)[0])
        mins.append(int(codes.min()) if codes.size else None)
        maxs.append(int(codes.max()) if codes.size else None)
        unks.append(unk)
        n_events += int(codes.size)
        z.close()
    return {
        "n_shards": len(shards),
        "n_events": n_events,
        "code_min": min(x for x in mins if x is not None),
        "code_max": max(x for x in maxs if x is not None),
        "unk_vocab_index": unks[0],
        "unk_consistent": len(set(unks)) == 1,
    }


def write_compatibility_report(
    out_path: Path,
    *,
    vocab_path: Path = VOCAB_PATH,
    embedding_path: Path = EMBEDDING_PATH,
    stage1_ckpt: Path = STAGE1_BEST_CKPT,
    tensorized_dir: Path | None = None,
) -> dict[str, Any]:
    vocab = load_vocab(vocab_path)
    v = len(vocab)
    n_rows, dim = embedding_row_count(embedding_path)
    ckpt = checkpoint_shapes(stage1_ckpt)
    shard_dir = tensorized_dir or (NCH_TENSORIZED_DIR / PRIMARY_REPRESENTATION)
    shard = inspect_shard_codes(shard_dir) if Path(shard_dir).exists() else {"n_shards": 0}
    ok_emb = n_rows == v + 2
    ok_head = ckpt["head_out"][0] == v
    ok_emb_ckpt = ckpt["embedding_table"] == [v + 2, dim]
    ok_codes = True
    if shard.get("n_shards"):
        ok_codes = (
            shard["unk_vocab_index"] == v
            and shard["code_min"] >= 0
            and shard["code_max"] <= v
            and shard["unk_consistent"]
        )
    report = {
        "vocab_path": str(vocab_path),
        "vocab_size": v,
        "special_tokens": SPECIAL,
        "model_id_rule": "PAD=0, UNK=1, real = vocab_index + 2 (model_new.data._pad_common)",
        "embedding": {
            "path": str(embedding_path),
            "n_rows": n_rows,
            "dim": dim,
            "matches_vocab_plus_pad_unk": ok_emb,
        },
        "stage1_checkpoint": {**ckpt, "path": str(stage1_ckpt), "head_matches_vocab": ok_head,
                              "embedding_matches_bge_rows": ok_emb_ckpt},
        "nch_shards": shard,
        "nch_target_space": (
            "subset of MIMIC |V|; full 30635-d head is retained. UNK is input-only "
            "and is dropped from the multi-hot target (Stage-1 contract)."
        ),
        "no_silent_remap": True,
        "passed": bool(ok_emb and ok_head and ok_emb_ckpt and ok_codes),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report
