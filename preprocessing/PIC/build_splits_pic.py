#!/usr/bin/env python3
"""Build patient-level train/val/test splits for PIC from the rolled event parquet.

Mirrors ``preprocessing/build_splits.py`` (MIMIC) and reuses its split logic
(``assign_splits`` for the 70/10/20 event-count-quintile-stratified assignment and
``write_split`` for materializing each split) from the parent package unchanged.

Writes ``train_events.parquet`` / ``val_events.parquet`` / ``test_events.parquet`` to
``data/processed/pic/`` with the MIMIC 10-column schema, and asserts no subject_id
appears in more than one split.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb

from _shared import assign_splits, write_split

LOGGER = logging.getLogger("build_splits_pic")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split rolled PIC events into train/val/test.")
    parser.add_argument(
        "--pic_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "pic",
        help="Directory holding patient_events_rolled_pic.parquet and split outputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    pic_dir: Path = args.pic_dir
    input_parquet = pic_dir / "patient_events_rolled_pic.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_parquet}")

    train_out = pic_dir / "train_events.parquet"
    val_out = pic_dir / "val_events.parquet"
    test_out = pic_dir / "test_events.parquet"

    con = duckdb.connect()
    try:
        escaped_in = str(input_parquet).replace("'", "''")
        counts_df = con.execute(
            f"""
            SELECT subject_id, COUNT(*) AS event_count
            FROM read_parquet('{escaped_in}')
            GROUP BY subject_id
            """
        ).df()

        splits = assign_splits(counts_df, seed=args.seed)

        for name, subject_ids in splits.items():
            output = {"train": train_out, "val": val_out, "test": test_out}[name]
            write_split(con, input_parquet, subject_ids, output)
            con.execute("DROP TABLE IF EXISTS split_subjects")

        # Subject-disjointness assertion.
        sets = {name: set(int(s) for s in ids.tolist()) for name, ids in splits.items()}
        assert not (sets["train"] & sets["val"]), "train/val overlap"
        assert not (sets["train"] & sets["test"]), "train/test overlap"
        assert not (sets["val"] & sets["test"]), "val/test overlap"

        total_input = int(con.execute(
            f"SELECT COUNT(DISTINCT subject_id) FROM read_parquet('{escaped_in}')").fetchone()[0])
        total_split = len(sets["train"]) + len(sets["val"]) + len(sets["test"])
        assert total_split == total_input, f"split subject total {total_split} != input {total_input}"

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
        LOGGER.info("Wrote splits: %s, %s, %s", train_out, val_out, test_out)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
