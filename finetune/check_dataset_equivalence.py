#!/usr/bin/env python3
"""Check sample-wise equivalence between on-the-fly and tensorized disease datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.dataset import DiseaseClassificationDataset, TensorizedDiseaseClassificationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check equivalence of on-the-fly and tensorized disease datasets.")
    parser.add_argument("--cohort_parquet", type=Path, required=True)
    parser.add_argument("--events_parquet", type=Path, default=Path("data/processed/patient_events_rolled_full.parquet"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/processed/code_vocab.json"))
    parser.add_argument("--tensorized_split_dir", type=Path, required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--report_path", type=Path, default=None)
    return parser.parse_args()


def _cmp_scalar(a, b) -> bool:
    return a == b


def main() -> int:
    args = parse_args()
    onfly = DiseaseClassificationDataset(
        cohort_parquet=args.cohort_parquet,
        events_parquet=args.events_parquet,
        code_vocab_path=args.vocab_path,
        max_seq_len=args.max_seq_len,
    )
    tensor = TensorizedDiseaseClassificationDataset(
        tensorized_split_dir=args.tensorized_split_dir,
        max_seq_len=args.max_seq_len,
        shard_cache_size=4,
    )

    if len(onfly) != len(tensor):
        raise RuntimeError(f"Length mismatch: onfly={len(onfly)} tensor={len(tensor)}")

    fields = ["subject_id", "label", "sex", "race", "code_indices", "timestamps_days", "age_days"]
    mismatch_counts = {k: 0 for k in fields}
    mismatch_examples: list[dict[str, object]] = []

    n = len(onfly)
    for i in range(n):
        a = onfly[i]
        b = tensor[i]
        sid = int(a["subject_id"])

        if not _cmp_scalar(int(a["subject_id"]), int(b["subject_id"])):
            mismatch_counts["subject_id"] += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append(
                    {
                        "index": i,
                        "subject_id_onfly": int(a["subject_id"]),
                        "subject_id_tensor": int(b["subject_id"]),
                        "field": "subject_id",
                    }
                )

        if not _cmp_scalar(float(a["label"]), float(b["label"])):
            mismatch_counts["label"] += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append({"index": i, "subject_id": sid, "field": "label", "onfly": float(a["label"]), "tensor": float(b["label"])})

        if not _cmp_scalar(int(a["sex"]), int(b["sex"])):
            mismatch_counts["sex"] += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append({"index": i, "subject_id": sid, "field": "sex", "onfly": int(a["sex"]), "tensor": int(b["sex"])})

        if not _cmp_scalar(int(a["race"]), int(b["race"])):
            mismatch_counts["race"] += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append({"index": i, "subject_id": sid, "field": "race", "onfly": int(a["race"]), "tensor": int(b["race"])})

        a_codes = np.asarray(a["code_indices"], dtype=np.int64)
        b_codes = np.asarray(b["code_indices"], dtype=np.int64)
        if a_codes.shape != b_codes.shape or not np.array_equal(a_codes, b_codes):
            mismatch_counts["code_indices"] += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append(
                    {
                        "index": i,
                        "subject_id": sid,
                        "field": "code_indices",
                        "onfly_len": int(a_codes.shape[0]),
                        "tensor_len": int(b_codes.shape[0]),
                        "onfly_head": a_codes[:10].tolist(),
                        "tensor_head": b_codes[:10].tolist(),
                    }
                )

        a_ts = np.asarray(a["timestamps_days"], dtype=np.float32)
        b_ts = np.asarray(b["timestamps_days"], dtype=np.float32)
        if a_ts.shape != b_ts.shape or not np.allclose(a_ts, b_ts):
            mismatch_counts["timestamps_days"] += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append(
                    {
                        "index": i,
                        "subject_id": sid,
                        "field": "timestamps_days",
                        "onfly_len": int(a_ts.shape[0]),
                        "tensor_len": int(b_ts.shape[0]),
                        "onfly_head": a_ts[:10].tolist(),
                        "tensor_head": b_ts[:10].tolist(),
                    }
                )

        a_age = np.asarray(a["age_days"], dtype=np.float32)
        b_age = np.asarray(b["age_days"], dtype=np.float32)
        if a_age.shape != b_age.shape or not np.allclose(a_age, b_age):
            mismatch_counts["age_days"] += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append(
                    {
                        "index": i,
                        "subject_id": sid,
                        "field": "age_days",
                        "onfly_len": int(a_age.shape[0]),
                        "tensor_len": int(b_age.shape[0]),
                        "onfly_head": a_age[:10].tolist(),
                        "tensor_head": b_age[:10].tolist(),
                    }
                )

    total_mismatches = int(sum(mismatch_counts.values()))
    if total_mismatches == 0:
        print(f"OK {n}/{n} samples agree.")
    else:
        print("Mismatch counts by field:")
        for k in fields:
            print(f"  - {k}: {mismatch_counts[k]}")
        print("First 3 mismatch examples:")
        for ex in mismatch_examples[:3]:
            print(json.dumps(ex, ensure_ascii=True))

    if args.report_path is not None:
        payload = {"n": n, "mismatch_counts": mismatch_counts, "examples": mismatch_examples[:3]}
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[report] wrote {args.report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
