#!/usr/bin/env python3
"""Build patient-level train/val/test splits from rolled event parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("build_splits")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split rolled patient events into train/val/test.")
    parser.add_argument("--test_mode", action="store_true", help="Use patient_events_rolled_test.parquet input.")
    return parser.parse_args()


def assign_splits(counts_df: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    # Quintiles by event count with stable tie-breaking.
    ranked = counts_df.copy()
    ranked["rank"] = ranked["event_count"].rank(method="first")
    ranked["quintile"] = pd.qcut(ranked["rank"], q=5, labels=False)

    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []

    for q in sorted(ranked["quintile"].dropna().unique()):
        group = ranked.loc[ranked["quintile"] == q, "subject_id"].astype("int64").to_numpy()
        rng.shuffle(group)
        n = len(group)
        n_train = int(np.floor(0.7 * n))
        n_val = int(np.floor(0.1 * n))
        n_test = n - n_train - n_val

        train_ids.extend(group[:n_train].tolist())
        val_ids.extend(group[n_train : n_train + n_val].tolist())
        test_ids.extend(group[n_train + n_val : n_train + n_val + n_test].tolist())

    return {
        "train": np.array(train_ids, dtype=np.int64),
        "val": np.array(val_ids, dtype=np.int64),
        "test": np.array(test_ids, dtype=np.int64),
    }


def write_split(
    con: duckdb.DuckDBPyConnection,
    input_parquet: Path,
    subject_ids: np.ndarray,
    output_parquet: Path,
) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE split_subjects(subject_id BIGINT)")
    if len(subject_ids) > 0:
        con.executemany("INSERT INTO split_subjects VALUES (?)", [(int(s),) for s in subject_ids.tolist()])
    escaped_in = str(input_parquet).replace("'", "''")
    escaped_out = str(output_parquet).replace("'", "''")
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet('{escaped_in}') e
            JOIN split_subjects s USING (subject_id)
            ORDER BY subject_id, event_time, code_id
        ) TO '{escaped_out}' (FORMAT PARQUET)
        """
    )


def main() -> int:
    setup_logging()
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    processed = repo_root / "data" / "processed"
    suffix = "test" if args.test_mode else "full"
    input_parquet = processed / f"patient_events_rolled_{suffix}.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_parquet}")

    train_out = processed / "train_events.parquet"
    val_out = processed / "val_events.parquet"
    test_out = processed / "test_events.parquet"

    con = duckdb.connect()
    try:
        escaped_in = str(input_parquet).replace("'", "''")
        counts_df = con.execute(
            f"""
            SELECT
                subject_id,
                COUNT(*) AS event_count
            FROM read_parquet('{escaped_in}')
            GROUP BY subject_id
            """
        ).df()
        splits = assign_splits(counts_df, seed=42)

        for name, subject_ids in splits.items():
            output = {"train": train_out, "val": val_out, "test": test_out}[name]
            write_split(con, input_parquet, subject_ids, output)
            con.execute("DROP TABLE IF EXISTS split_subjects")

        # Reporting
        print("Patient counts per split:")
        print(f"  - train: {len(splits['train']):,}")
        print(f"  - val:   {len(splits['val']):,}")
        print(f"  - test:  {len(splits['test']):,}")

        row_counts = {}
        for name, path in [("train", train_out), ("val", val_out), ("test", test_out)]:
            escaped = str(path).replace("'", "''")
            row_counts[name] = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{escaped}')").fetchone()[0])
        print("Row counts per split:")
        print(f"  - train: {row_counts['train']:,}")
        print(f"  - val:   {row_counts['val']:,}")
        print(f"  - test:  {row_counts['test']:,}")

        LOGGER.info("Wrote split files: %s, %s, %s", train_out, val_out, test_out)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
