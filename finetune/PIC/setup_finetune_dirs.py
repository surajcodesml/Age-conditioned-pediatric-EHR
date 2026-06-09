#!/usr/bin/env python3
"""Lay out PIC fine-tune inputs in the directory shape expected by the trainer.

``finetune/train.py`` / ``finetune/build_disease_tensors.py`` expect a cohort
directory holding ``{train,val,test}_cohort.parquet`` plus a single events
parquet. The PIC cohort builder emits flat ``{task}_{split}_cohort.parquet`` and,
for diagnosis tasks, per-split *filtered* event streams
``{task}_{split}_events.parquet`` (target codes removed).

For each task this script produces::

    data/processed/pic/finetune/<task>/cohort/{train,val,test}_cohort.parquet
    data/processed/pic/finetune/<task>/events.parquet

where ``events.parquet`` is:
  - the shared rolled event table for mortality / los_gt7 (no leakage filtering); or
  - the union of the three per-split filtered streams for the diagnosis tasks.

Train/val/test subject sets are disjoint, so a single unioned events parquet
joins cleanly against each split's cohort during tensorization.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
PIC_DIR = REPO_ROOT / "data" / "processed" / "pic"
COHORTS_DIR = PIC_DIR / "cohorts"
ROLLED_EVENTS = PIC_DIR / "patient_events_rolled_pic.parquet"

# task -> uses per-split filtered events (diagnosis tasks) or the shared rolled stream
TASKS = {
    "mortality": {"filtered": False},
    "los_gt7": {"filtered": False},
    "pneumonia": {"filtered": True},
    "heart_malformations": {"filtered": True},
}

SPLITS = ("train", "val", "test")


def _esc(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")


def setup_task(task: str, filtered: bool, out_root: Path) -> None:
    base = out_root / task
    cohort_dir = base / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        src = COHORTS_DIR / f"{task}_{split}_cohort.parquet"
        if not src.exists():
            raise FileNotFoundError(f"Missing cohort parquet: {src}")
        shutil.copyfile(src, cohort_dir / f"{split}_cohort.parquet")

    events_out = base / "events.parquet"
    if filtered:
        srcs = [COHORTS_DIR / f"{task}_{split}_events.parquet" for split in SPLITS]
        for s in srcs:
            if not s.exists():
                raise FileNotFoundError(f"Missing filtered events parquet: {s}")
        union_sql = " UNION ALL ".join(f"SELECT * FROM read_parquet('{_esc(s)}')" for s in srcs)
        con = duckdb.connect()
        try:
            con.execute("PRAGMA memory_limit='6GB'")
            con.execute("PRAGMA threads=8")
            con.execute(f"COPY ({union_sql}) TO '{_esc(events_out)}' (FORMAT PARQUET)")
        finally:
            con.close()
        print(f"[setup] {task}: filtered events union -> {events_out}", flush=True)
    else:
        if not ROLLED_EVENTS.exists():
            raise FileNotFoundError(f"Missing rolled events parquet: {ROLLED_EVENTS}")
        if events_out.exists() or events_out.is_symlink():
            events_out.unlink()
        events_out.symlink_to(ROLLED_EVENTS.resolve())
        print(f"[setup] {task}: events.parquet -> {ROLLED_EVENTS} (symlink)", flush=True)

    # quick sanity: every cohort subject must exist in the events stream
    con = duckdb.connect()
    try:
        for split in SPLITS:
            cp = cohort_dir / f"{split}_cohort.parquet"
            missing = con.execute(
                f"""
                SELECT COUNT(*) FROM read_parquet('{_esc(cp)}') c
                WHERE c.subject_id NOT IN (
                    SELECT DISTINCT subject_id FROM read_parquet('{_esc(events_out)}')
                )
                """
            ).fetchone()[0]
            if int(missing) != 0:
                raise RuntimeError(f"{task}/{split}: {missing} cohort subjects absent from events.parquet")
    finally:
        con.close()
    print(f"[setup] {task}: cohort/events consistency OK", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up PIC fine-tune directories.")
    parser.add_argument("--out_root", type=Path, default=PIC_DIR / "finetune")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS.keys()))
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task} (known: {list(TASKS)})")
        setup_task(task, TASKS[task]["filtered"], args.out_root)
    print(f"[setup] done -> {args.out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
