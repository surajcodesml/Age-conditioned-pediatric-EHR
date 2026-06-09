#!/usr/bin/env python3
"""STEP 3a: patient-level train/val/test splits for the Synthea cohort.

``preprocessing/build_splits.py`` hardcodes ``data/processed`` (the MIMIC dir),
so -- exactly as the PIC path does (``preprocessing/PIC/build_splits_pic.py``) --
this wrapper REUSES its ``assign_splits`` (70/10/20 event-count-quintile
stratified) and ``write_split`` functions UNCHANGED, pointed at the isolated
Synthea directory. Nothing in build_splits.py is modified.

Writes ``train_events.parquet`` / ``val_events.parquet`` / ``test_events.parquet``
into ``--data_dir`` (default ``data/synthea/processed``).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
_PREPROC_DIR = REPO_ROOT / "preprocessing"
if str(_PREPROC_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROC_DIR))

from build_splits import assign_splits, write_split  # noqa: E402  (reused, read-only)

LOGGER = logging.getLogger("build_splits_synthea")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split rolled Synthea events into train/val/test.")
    p.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data" / "synthea" / "processed")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    data_dir: Path = args.data_dir
    input_parquet = data_dir / "patient_events_rolled_full.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_parquet}")

    train_out = data_dir / "train_events.parquet"
    val_out = data_dir / "val_events.parquet"
    test_out = data_dir / "test_events.parquet"

    con = duckdb.connect()
    try:
        escaped_in = str(input_parquet).replace("'", "''")
        counts_df = con.execute(
            f"SELECT subject_id, COUNT(*) AS event_count FROM read_parquet('{escaped_in}') GROUP BY subject_id"
        ).df()
        splits = assign_splits(counts_df, seed=args.seed)

        for name, subject_ids in splits.items():
            output = {"train": train_out, "val": val_out, "test": test_out}[name]
            write_split(con, input_parquet, subject_ids, output)
            con.execute("DROP TABLE IF EXISTS split_subjects")

        sets = {name: set(int(s) for s in ids.tolist()) for name, ids in splits.items()}
        assert not (sets["train"] & sets["val"]), "train/val overlap"
        assert not (sets["train"] & sets["test"]), "train/test overlap"
        assert not (sets["val"] & sets["test"]), "val/test overlap"

        print("Patient counts per split:")
        print(f"  - train: {len(splits['train']):,}")
        print(f"  - val:   {len(splits['val']):,}")
        print(f"  - test:  {len(splits['test']):,}")
        print("Row counts per split:")
        for name, path in [("train", train_out), ("val", val_out), ("test", test_out)]:
            esc = str(path).replace("'", "''")
            rc = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{esc}')").fetchone()[0])
            print(f"  - {name}: {rc:,}")
        print("Subject-disjointness assertion: PASSED")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
