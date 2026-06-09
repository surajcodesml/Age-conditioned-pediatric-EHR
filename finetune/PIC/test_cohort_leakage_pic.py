#!/usr/bin/env python3
"""Step 2: verify PIC fine-tune cohorts (leakage, length-confound, dataset smoke test).

For each task x split it checks:
 1. Resolved target code set (diagnosis tasks) — printed for human review.
 2. Cohort size + label prevalence.
 3. TARGET-ABSENCE assertion (diagnosis tasks): no target-family code appears in
    events[0 .. last_event_idx] of the FILTERED events parquet. Hard-fail otherwise.
 4. LENGTH-ONLY AUROC: logistic regression on a single feature (# events in the
    observation window) vs label; fit on train, evaluate each split. Report only; flagged
    if >0.85 for any task except LOS (where ~0.7 is a legitimate severity signal).
 5. SMOKE TEST: DiseaseClassificationDataset + disease_collate; pull 8 samples; check
    tensor shapes, last_event_idx truncation, and labels in {0,1}.
 6. min(age_at_event_days) >= 0 in every events parquet read.

Prints a one-line PASS/FAIL per task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "finetune"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import DiseaseClassificationDataset, disease_collate  # noqa: E402
from build_cohorts_pic import derive_target_sets, SPLITS  # noqa: E402

DIAGNOSIS_TASKS = ("pneumonia", "heart_malformations")
ALL_TASKS = ("pneumonia", "heart_malformations", "los_gt7", "mortality")
LENGTH_AUROC_FLAG = 0.85


def _esc(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PIC cohorts.")
    parser.add_argument("--obs_window_days", type=float, default=1.0)
    parser.add_argument("--pic_dir", type=Path, default=_REPO_ROOT / "data" / "processed" / "pic")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    return parser.parse_args()


def events_for(task: str, split: str, pic_dir: Path) -> Path:
    """FILTERED parquet for diagnosis tasks, FULL split parquet otherwise."""
    if task in DIAGNOSIS_TASKS:
        return pic_dir / "cohorts" / f"{task}_{split}_events.parquet"
    return pic_dir / f"{split}_events.parquet"


def cohort_for(task: str, split: str, pic_dir: Path) -> Path:
    return pic_dir / "cohorts" / f"{task}_{split}_cohort.parquet"


def auroc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U AUROC (handles ties)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    sum_pos = ranks[labels == 1].sum()
    auc = (sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def length_feature(con, events: Path, cohort: Path, obs: float) -> tuple[np.ndarray, np.ndarray]:
    df = con.execute(
        f"""
        WITH cohort AS (SELECT subject_id, label FROM read_parquet('{_esc(cohort)}')),
        nwin AS (
            SELECT CAST(subject_id AS BIGINT) AS subject_id,
                   SUM(CASE WHEN timestamp_days <= {obs} THEN 1 ELSE 0 END) AS n_in_window
            FROM read_parquet('{_esc(events)}')
            GROUP BY 1
        )
        SELECT c.label, COALESCE(n.n_in_window, 0) AS n_in_window
        FROM cohort c LEFT JOIN nwin n USING (subject_id)
        ORDER BY c.subject_id
        """
    ).df()
    return df["n_in_window"].to_numpy(dtype=np.float64), df["label"].to_numpy(dtype=np.int64)


def fit_length_auroc(con, task: str, pic_dir: Path, obs: float) -> dict[str, float]:
    """Fit LR on train length feature, report AUROC per split (sklearn, rank fallback)."""
    feats = {s: length_feature(con, events_for(task, s, pic_dir), cohort_for(task, s, pic_dir), obs)
             for s in SPLITS}
    Xtr, ytr = feats["train"]
    out: dict[str, float] = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        if len(np.unique(ytr)) < 2:
            raise ValueError("single-class train")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr.reshape(-1, 1), ytr)
        for s in SPLITS:
            Xs, ys = feats[s]
            if len(np.unique(ys)) < 2:
                out[s] = float("nan")
            else:
                out[s] = float(roc_auc_score(ys, clf.predict_proba(Xs.reshape(-1, 1))[:, 1]))
    except Exception:
        # Direction from train rank-AUC, then report rank-AUC per split (monotone LR equiv).
        tr_auc = auroc_rank(Xtr, ytr)
        flip = tr_auc < 0.5
        for s in SPLITS:
            Xs, ys = feats[s]
            a = auroc_rank(-Xs if flip else Xs, ys)
            out[s] = a
    return out


def target_absence_ok(con, task: str, split: str, pic_dir: Path, targets: set[str], obs: float) -> int:
    """Return count of target codes within events[0..last_event_idx] in FILTERED stream (want 0)."""
    events = events_for(task, split, pic_dir)
    cohort = cohort_for(task, split, pic_dir)
    con.execute("CREATE OR REPLACE TEMP TABLE _tgt(code_id VARCHAR)")
    if targets:
        con.executemany("INSERT INTO _tgt VALUES (?)", [(c,) for c in sorted(targets)])
    n = int(con.execute(
        f"""
        WITH ev AS (
            SELECT CAST(subject_id AS BIGINT) AS subject_id, code_id,
                   ROW_NUMBER() OVER (PARTITION BY CAST(subject_id AS BIGINT)
                       ORDER BY timestamp_days, event_time, code_id) - 1 AS event_idx
            FROM read_parquet('{_esc(events)}')
        ),
        cohort AS (SELECT subject_id, last_event_idx FROM read_parquet('{_esc(cohort)}'))
        SELECT COUNT(*)
        FROM ev JOIN cohort c USING (subject_id)
        WHERE ev.event_idx <= c.last_event_idx
          AND ev.code_id IN (SELECT code_id FROM _tgt)
        """
    ).fetchone()[0])
    return n


def smoke_test(task: str, split: str, pic_dir: Path, max_seq_len: int) -> str:
    cohort = cohort_for(task, split, pic_dir)
    events = events_for(task, split, pic_dir)
    vocab = pic_dir / "code_vocab_pic.json"
    ds = DiseaseClassificationDataset(cohort, events, vocab, max_seq_len=max_seq_len)
    if len(ds) == 0:
        return "EMPTY"
    k = min(8, len(ds))
    loader = DataLoader(ds, batch_size=k, shuffle=False, collate_fn=disease_collate, num_workers=0)
    batch = next(iter(loader))
    B, L = batch["code_indices"].shape
    assert batch["delta_t"].shape == (B, L, L), "delta_t shape"
    assert batch["demographics"].shape == (B, L, 3), "demographics shape"
    assert batch["labels"].shape == (B,), "labels shape"
    assert set(int(x) for x in batch["labels"].tolist()) <= {0, 1}, "labels not in {0,1}"
    # last_event_idx truncation: returned length == min(last_event_idx+1, max_seq_len)
    for i in range(k):
        sample = ds[i]
        expected = int(ds._rows[i]["last_event_idx"])
        got = int(sample["code_indices"].shape[0])
        assert got == min(expected + 1, max_seq_len), (
            f"{task}:{split} idx {i} trunc mismatch got={got} expected={min(expected+1, max_seq_len)}"
        )
    return f"B={B} L={L}"


def main() -> int:
    args = parse_args()
    pic_dir: Path = args.pic_dir
    obs = float(args.obs_window_days)
    vocab_codes = set(json.load((pic_dir / "code_vocab_pic.json").open()).keys())
    descriptions = json.load((pic_dir / "code_descriptions_pic.json").open())

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='24GB'")

    # (1) resolve + print target sets (also used for absence assertion).
    targets = derive_target_sets(
        con, _REPO_ROOT / "data" / "processed" / "mappings", vocab_codes, descriptions
    )

    task_pass: dict[str, bool] = {t: True for t in ALL_TASKS}
    task_notes: dict[str, list[str]] = {t: [] for t in ALL_TASKS}

    # (6) min age >= 0 in every events parquet read.
    print("\n=== min(age_at_event_days) >= 0 check ===")
    checked_events = set()
    for task in ALL_TASKS:
        for split in SPLITS:
            ev = events_for(task, split, pic_dir)
            if ev in checked_events:
                continue
            checked_events.add(ev)
            mn = float(con.execute(
                f"SELECT MIN(age_at_event_days) FROM read_parquet('{_esc(ev)}')").fetchone()[0])
            ok = mn >= 0.0
            print(f"  {ev.name}: min_age={mn:.4f} {'OK' if ok else 'FAIL'}")
            if not ok:
                task_pass[task] = False

    # length AUROC per task (fit once, all splits).
    print("\n=== LENGTH-ONLY AUROC (report only; flag>0.85 except LOS) ===")
    for task in ALL_TASKS:
        aucs = fit_length_auroc(con, task, pic_dir, obs)
        flagged = any((not np.isnan(v)) and v > LENGTH_AUROC_FLAG for v in aucs.values())
        line = " ".join(f"{s}={aucs[s]:.4f}" for s in SPLITS)
        flag_str = ""
        if flagged and task != "los_gt7":
            flag_str = "  <-- FLAG (>0.85 length confound)"
            task_notes[task].append("length-AUROC>0.85")
        elif flagged and task == "los_gt7":
            flag_str = "  (LOS: expected legitimate severity signal)"
        print(f"  [{task}] AUROC {line}{flag_str}")

    # per task x split: prevalence, absence assertion, smoke test.
    print("\n=== per-task / per-split checks ===")
    for task in ALL_TASKS:
        for split in SPLITS:
            cohort = cohort_for(task, split, pic_dir)
            cdf = con.execute(
                f"SELECT COUNT(*) n, AVG(CAST(label AS DOUBLE)) p, MIN(label) mn, MAX(label) mx "
                f"FROM read_parquet('{_esc(cohort)}')"
            ).fetchone()
            n, prev = int(cdf[0]), (float(cdf[1]) if cdf[1] is not None else float("nan"))
            assert int(cdf[2]) in (0, 1) and int(cdf[3]) in (0, 1), f"{task}:{split} labels not 0/1"

            absence = ""
            if task in DIAGNOSIS_TASKS:
                n_leak = target_absence_ok(con, task, split, pic_dir, targets[task], obs)
                absence = f" target_in_context={n_leak}"
                if n_leak != 0:
                    task_pass[task] = False
                    task_notes[task].append(f"{split}:LEAK={n_leak}")

            try:
                smoke = smoke_test(task, split, pic_dir, args.max_seq_len)
            except AssertionError as e:
                smoke = f"SMOKE-FAIL: {e}"
                task_pass[task] = False
                task_notes[task].append(f"{split}:smoke")
            print(f"  [{task}:{split}] n={n:,} prevalence={prev:.4%}{absence} | smoke[{smoke}]")

    con.close()

    print("\n=== FINAL SUMMARY ===")
    overall = True
    for task in ALL_TASKS:
        status = "PASS" if task_pass[task] else "FAIL"
        notes = (" | " + ",".join(task_notes[task])) if task_notes[task] else ""
        print(f"  [{task}] {status}{notes}")
        overall &= task_pass[task]
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
