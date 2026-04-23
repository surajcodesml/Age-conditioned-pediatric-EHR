#!/usr/bin/env python3
"""PyTorch Dataset and collate for TALE-EHR pretraining on MIMIC-IV event sequences."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("ehr_dataset")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def encode_race(race_val: Any) -> int:
    """Map MIMIC-IV race string to 7 buckets (prefix / literal rules)."""
    if race_val is None:
        return 6
    if isinstance(race_val, float) and np.isnan(race_val):
        return 6
    s = str(race_val).strip().upper()
    if not s or s == "NAN":
        return 6
    unknown_literals = {"UNKNOWN", "UNABLE TO OBTAIN", "PREFER NOT TO SAY", "N/A", "DECLINED"}
    if s in unknown_literals:
        return 6
    if s.startswith("WHITE"):
        return 0
    if s.startswith("BLACK"):
        return 1
    if s.startswith("ASIAN"):
        return 2
    if s.startswith("HISPANIC"):
        return 3
    if s.startswith("AMERICAN INDIAN") or s.startswith("ALASKA NATIVE"):
        return 4
    if s == "OTHER" or s.startswith("OTHER "):
        return 5
    return 5


def encode_sex(val: Any) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        v = int(val)
    except (TypeError, ValueError):
        return 0
    return 1 if v == 1 else 0


def _read_parquet_head(path: Path, max_rows: int | None) -> pd.DataFrame:
    """Load full parquet or only the first `max_rows` rows (for smoke tests)."""
    if max_rows is None:
        return pq.read_table(path).to_pandas()
    pf = pq.ParquetFile(path)
    chunks: list = []
    n = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        chunks.append(t)
        n += t.num_rows
        if n >= max_rows:
            break
    if not chunks:
        return pd.DataFrame()
    table = pa.concat_tables(chunks)
    if table.num_rows > max_rows:
        table = table.slice(0, max_rows)
    return table.to_pandas()


class EHRDataset(Dataset):
    """Event-sequence samples for next-visit code and timing prediction."""

    def __init__(
        self,
        parquet_path: str | Path,
        code_vocab_path: str | Path,
        max_seq_len: int = 1024,
        max_rows: int | None = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.code_vocab_path = Path(code_vocab_path)
        self.max_seq_len = int(max_seq_len)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet: {self.parquet_path}")
        if not self.code_vocab_path.exists():
            raise FileNotFoundError(f"Missing code vocab: {self.code_vocab_path}")

        with self.code_vocab_path.open("r", encoding="utf-8") as f:
            self.code_vocab: dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}
        self.num_codes = len(self.code_vocab)
        # Input sequences use UNK_VOCAB = N so v+2 never collides with BGE rows for real codes.
        self.unk_vocab_index = int(self.num_codes)

        df = _read_parquet_head(self.parquet_path, max_rows)

        required = (
            "subject_id",
            "hadm_id",
            "event_time",
            "code_id",
            "timestamp_days",
            "age_at_event_days",
            "sex",
            "race",
        )
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Parquet missing required column {c!r}")

        self._patients: list[dict[str, Any]] = []
        self._samples: list[tuple[int, int]] = []

        for _subject_id, g in df.groupby("subject_id", sort=False):
            g = g.sort_values(["event_time", "code_id"], kind="mergesort").reset_index(drop=True)
            if len(g) < 2:
                continue

            h = g["hadm_id"].fillna(-1)
            try:
                h_int = h.astype("int64")
            except (TypeError, ValueError):
                h_int = h.astype("float64").fillna(-1).astype("int64")

            n_events = len(g)
            visit_starts: list[int] = [0]
            for i in range(1, n_events):
                if int(h_int.iloc[i]) != int(h_int.iloc[i - 1]):
                    visit_starts.append(i)
            visit_starts.append(n_events)
            visit_spans = [(visit_starts[i], visit_starts[i + 1]) for i in range(len(visit_starts) - 1)]

            if len(visit_spans) < 2:
                continue

            code_indices = np.empty(n_events, dtype=np.int32)
            timestamps_days = np.empty(n_events, dtype=np.float32)
            age_days = np.empty(n_events, dtype=np.float32)

            code_id_col = g["code_id"].astype(str)
            ts_col = g["timestamp_days"].astype("float64")
            age_col = g["age_at_event_days"].astype("float64")

            for i in range(n_events):
                cid = str(code_id_col.iloc[i])
                if cid in self.code_vocab:
                    code_indices[i] = np.int32(self.code_vocab[cid])
                else:
                    code_indices[i] = np.int32(self.unk_vocab_index)

                td = ts_col.iloc[i]
                timestamps_days[i] = np.float32(td) if pd.notna(td) else np.float32(0.0)

                ad = age_col.iloc[i]
                age_days[i] = np.float32(ad) if pd.notna(ad) else np.float32(0.0)

            sex = encode_sex(g["sex"].iloc[0])
            race = encode_race(g["race"].iloc[0])

            pat_idx = len(self._patients)
            self._patients.append(
                {
                    "code_indices": code_indices,
                    "timestamps_days": timestamps_days,
                    "age_days": age_days,
                    "sex": sex,
                    "race": race,
                    "visit_spans": visit_spans,
                }
            )

            n_visits = len(visit_spans)
            for v in range(n_visits - 1):
                self._samples.append((pat_idx, v))

        LOGGER.info(
            "EHRDataset: %d patients, %d (patient, visit) samples from %s",
            len(self._patients),
            len(self._samples),
            self.parquet_path,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def _target_codes_for_visit(self, pat: dict[str, Any], visit_idx: int) -> np.ndarray:
        start, end = pat["visit_spans"][visit_idx]
        target = np.zeros(self.num_codes, dtype=np.float32)
        seen: set[int] = set()
        codes = pat["code_indices"][start:end]
        for v in codes:
            vi = int(v)
            if vi == self.unk_vocab_index or vi in seen:
                continue
            seen.add(vi)
            target[vi] = 1.0
        return target

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pat_idx, visit_k = self._samples[idx]
        pat = self._patients[pat_idx]
        visit_spans: list[tuple[int, int]] = pat["visit_spans"]

        end_curr = visit_spans[visit_k][1]
        start_next = visit_spans[visit_k + 1][0]

        code_indices = pat["code_indices"][:end_curr].copy()
        timestamps_days = pat["timestamps_days"][:end_curr].copy()
        age_days = pat["age_days"][:end_curr].copy()

        if code_indices.shape[0] > self.max_seq_len:
            sl = slice(-self.max_seq_len, None)
            code_indices = code_indices[sl]
            timestamps_days = timestamps_days[sl]
            age_days = age_days[sl]

        target_codes = self._target_codes_for_visit(pat, visit_k + 1)

        last_curr = visit_spans[visit_k][1] - 1
        ts_curr_last = float(pat["timestamps_days"][last_curr])
        ts_next_first = float(pat["timestamps_days"][start_next])
        target_time_gap = float(ts_next_first - ts_curr_last)

        return {
            "code_indices": code_indices,
            "timestamps_days": timestamps_days,
            "age_days": age_days,
            "sex": int(pat["sex"]),
            "race": int(pat["race"]),
            "target_codes": target_codes,
            "target_time_gap": target_time_gap,
        }


def ehr_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences and build pairwise Δt and BGE-aligned code indices."""
    if not batch:
        raise ValueError("empty batch")

    bsz = len(batch)
    num_codes = batch[0]["target_codes"].shape[0]
    unk_id = num_codes

    max_len = max(int(item["code_indices"].shape[0]) for item in batch)

    code_np = np.full((bsz, max_len), -1, dtype=np.int64)
    ts_np = np.zeros((bsz, max_len), dtype=np.float32)
    age_np = np.zeros((bsz, max_len), dtype=np.float32)
    mask_np = np.zeros((bsz, max_len), dtype=bool)

    target_codes_np = np.stack([item["target_codes"] for item in batch], axis=0)
    target_gap_np = np.array([item["target_time_gap"] for item in batch], dtype=np.float32)

    sex_arr = np.array([item["sex"] for item in batch], dtype=np.float32)
    race_arr = np.array([item["race"] for item in batch], dtype=np.float32)

    for b, item in enumerate(batch):
        seq = item["code_indices"]
        L = int(seq.shape[0])
        if L == 0:
            continue
        code_np[b, :L] = seq.astype(np.int64)
        ts_np[b, :L] = item["timestamps_days"].astype(np.float32)
        age_np[b, :L] = item["age_days"].astype(np.float32)
        mask_np[b, :L] = True

    code_indices = torch.from_numpy(code_np)
    timestamps_days = torch.from_numpy(ts_np)
    attention_mask = torch.from_numpy(mask_np)

    # BGE row indices: PAD=0, UNK=1, real vocab v -> v+2
    bge_codes = torch.where(
        attention_mask,
        torch.where(
            code_indices == unk_id,
            torch.ones((), dtype=torch.long),
            code_indices + 2,
        ),
        torch.zeros((), dtype=torch.long),
    )

    t = timestamps_days
    delta_t = torch.log1p(torch.abs(t.unsqueeze(2) - t.unsqueeze(1)))
    pair_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    delta_t = delta_t * pair_mask.float()

    age_years = age_np / 365.25
    demo = np.zeros((bsz, max_len, 3), dtype=np.float32)
    demo[:, :, 0] = age_years
    demo[:, :, 1] = sex_arr[:, np.newaxis]
    demo[:, :, 2] = race_arr[:, np.newaxis]
    demo = demo * mask_np[:, :, np.newaxis].astype(np.float32)

    return {
        "code_indices": bge_codes,
        "timestamps_days": timestamps_days,
        "delta_t": delta_t,
        "attention_mask": attention_mask,
        "demographics": torch.from_numpy(demo),
        "target_codes": torch.from_numpy(target_codes_np),
        "target_time_gap": torch.from_numpy(target_gap_np),
    }


def _invert_vocab(code_vocab: dict[str, int]) -> list[str]:
    n = len(code_vocab)
    out = [""] * n
    for cid, i in code_vocab.items():
        if 0 <= int(i) < n:
            out[int(i)] = str(cid)
    return out


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Smoke test EHRDataset + ehr_collate")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Use test_events.parquet instead of train_events.parquet",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=200_000,
        help="Read only the first N rows of the parquet (smoke test; use 0 for full file).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    processed = repo_root / "data" / "processed"
    split_name = "test_events.parquet" if args.test_mode else "train_events.parquet"
    parquet_path = processed / split_name
    vocab_path = processed / "code_vocab.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}")

    with vocab_path.open("r", encoding="utf-8") as f:
        code_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    index_to_code = _invert_vocab(code_vocab)

    row_limit = None if args.max_rows == 0 else args.max_rows
    dataset = EHRDataset(parquet_path=parquet_path, code_vocab_path=vocab_path, max_rows=row_limit)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=ehr_collate)
    batch = next(iter(loader))

    print("Tensor shapes and dtypes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    bsz, L = batch["code_indices"].shape
    assert batch["delta_t"].shape == (bsz, L, L), "delta_t must be [B, L, L]"
    for k, v in batch.items():
        assert torch.isfinite(v).all(), f"non-finite values in {k}"

    print("\nSample code_indices (BGE rows) first two positions:", batch["code_indices"][:2, : min(8, L)])

    print("\nExample vocab strings for first row (map BGE index j -> vocab j-2 for j>=2):")
    row0 = batch["code_indices"][0, : min(10, L)]
    mask0 = batch["attention_mask"][0, : min(10, L)]
    for i in range(min(10, L)):
        if not mask0[i]:
            break
        j = int(row0[i].item())
        if j <= 1:
            print(f"  pos {i}: BGE={j} (PAD/UNK)")
        else:
            vi = j - 2
            cid = index_to_code[vi] if 0 <= vi < len(index_to_code) else "?"
            print(f"  pos {i}: BGE={j} vocab={vi} code_id={cid!r}")

    pos = batch["target_codes"].sum(dim=1)
    print("\nPositive labels per sample in batch:", pos.tolist())

    print("\nSmoke test passed.")
