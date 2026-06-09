#!/usr/bin/env python3
"""Step 0: clip negative ages to 0 in the PIC event parquets, in place.

PIC date de-identification leaves a small fraction of events timestamped before the
recorded DOB (negative ``age_at_event_days``). For fine-tuning we clip these to 0 so the
age channel is non-negative. The clip is applied to the three split parquets the dataset
reads, plus ``patient_events_rolled_pic.parquet`` for provenance.

For each file: load, set ``age_at_event_days = GREATEST(age_at_event_days, 0.0)``, rewrite
the same file (via temp + atomic replace, since DuckDB cannot COPY onto its own input).
Afterwards asserts ``min(age_at_event_days) == 0.0`` across the three split files and logs
the number of rows clipped.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import duckdb

LOGGER = logging.getLogger("prep_clip_age")


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clip negative ages to 0 in PIC parquets.")
    parser.add_argument(
        "--pic_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "pic",
    )
    return parser.parse_args()


def clip_file(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    """Clip age in `path` in place; return number of rows that were negative."""
    n_clip = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{_esc(path)}') WHERE age_at_event_days < 0"
        ).fetchone()[0]
    )
    tmp = path.with_suffix(".clip_tmp.parquet")
    con.execute(
        f"""
        COPY (
            SELECT * REPLACE (GREATEST(age_at_event_days, 0.0) AS age_at_event_days)
            FROM read_parquet('{_esc(path)}')
        ) TO '{_esc(tmp)}' (FORMAT PARQUET)
        """
    )
    os.replace(tmp, path)
    return n_clip


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    pic_dir: Path = args.pic_dir

    split_files = [pic_dir / f"{s}_events.parquet" for s in ("train", "val", "test")]
    rolled_file = pic_dir / "patient_events_rolled_pic.parquet"
    all_files = split_files + [rolled_file]

    missing = [str(p) for p in all_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing parquet(s):\n" + "\n".join(missing))

    con = duckdb.connect()
    try:
        con.execute("PRAGMA memory_limit='24GB'")
        total_clipped = 0
        for f in all_files:
            n = clip_file(con, f)
            total_clipped += n
            LOGGER.info("Clipped %d negative-age rows in %s", n, f.name)

        # Assert non-negative across the three split files (which the dataset reads).
        for f in split_files:
            mn = float(con.execute(f"SELECT MIN(age_at_event_days) FROM read_parquet('{_esc(f)}')").fetchone()[0])
            assert mn == 0.0, f"min(age_at_event_days)={mn} != 0.0 in {f.name}"

        print(f"Total rows clipped across all files: {total_clipped:,}")
        print("Per-split min(age_at_event_days) == 0.0: ASSERTED for train/val/test")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
