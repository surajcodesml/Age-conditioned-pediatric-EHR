"""D9 — Trivial baselines (CPU, DuckDB).

Persistence / co-occurrence / global-prior recall@{5,10,20} on the fixed val batch
list. Co-occurrence is computed in SQL over a flat (window_id, code, role) table;
lift is restricted to the top-2000 codes by frequency (validated vs full-vocab on smoke).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from model_new import diagnostics as D
from model_new.audit.common import REPO_ROOT
from model_new.audit.signal import D9_TOP_CODES_LIFT, KS
from model_new.audit.signal.common import (
    add_common_args,
    base_result_meta,
    ensure_batches,
    iter_store_batches,
    nanmean_safe,
    open_duckdb,
    paired_delta_ci,
    per_code_hit_miss,
    recall_from_scores,
    serialize_per_code_hit_miss,
    write_json_atomic,
)
from model_new.data import TensorizedPretrainDataset


def _val_flat_from_store(store: dict):
    """Build (example_id, code, count, role) from the packed cache."""
    ex_ids: list[int] = []
    codes: list[int] = []
    counts: list[int] = []
    roles: list[int] = []
    eid = 0
    for batch in iter_store_batches(store):
        bsz = int(batch["lengths"].shape[0])
        mask = batch["attention_mask"].numpy()
        raw = batch["code_indices"].numpy()
        targets = batch["target_codes"].numpy()
        for i in range(bsz):
            idx = np.flatnonzero(mask[i])
            seq = raw[i, idx]
            real = seq[seq >= 2] - 2
            if real.size:
                uniq, cnt = np.unique(real, return_counts=True)
                for c, n in zip(uniq.tolist(), cnt.tolist()):
                    ex_ids.append(eid); codes.append(int(c))
                    counts.append(int(n)); roles.append(0)
            for c in np.flatnonzero(targets[i] > 0).tolist():
                ex_ids.append(eid); codes.append(int(c))
                counts.append(1); roles.append(1)
            eid += 1
    return (
        np.asarray(ex_ids, np.int32),
        np.asarray(codes, np.int32),
        np.asarray(counts, np.int32),
        np.asarray(roles, np.int8),
    )


def _train_flat_table(ctx: dict, patient_frac: float, seed: int,
                      max_windows: int | None) -> tuple[np.ndarray, int]:
    """Flat (window_id, code, role) for train subsample. role: 0=input, 1=target."""
    shared = ctx["shared"]
    ds = TensorizedPretrainDataset(
        REPO_ROOT / shared["tensorized_dir"] / "train",
        REPO_ROOT / shared["vocab_path"],
        max_seq_len=shared["max_seq_len"],
    )
    V = int(shared["num_codes"])
    rng = np.random.default_rng(seed)
    n = len(ds)
    n_keep = max(1, int(n * patient_frac)) if patient_frac < 1.0 else n
    indices = rng.choice(n, size=n_keep, replace=False) if n_keep < n else np.arange(n)
    if max_windows is not None:
        indices = indices[: int(max_windows)]
    FULL_CAP = 50_000
    if max_windows is None and indices.size > FULL_CAP:
        indices = rng.choice(indices, size=FULL_CAP, replace=False)
    N = int(indices.size)

    by_shard: dict[int, list[tuple[int, int, int]]] = {}
    for wi, idx in enumerate(indices):
        shard_id, pos, visit_k = ds._index[int(idx)]
        by_shard.setdefault(int(shard_id), []).append((wi, int(pos), int(visit_k)))

    rows: list[tuple[int, int, int]] = []
    for shard_id, items in by_shard.items():
        s = ds._load_shard(shard_id)
        unk = int(s["unk_vocab_index"])
        for wi, pos, visit_k in items:
            ev_start = int(s["event_offsets"][pos])
            vis_start = int(s["visit_offsets"][pos])
            end_curr = int(s["visit_ends"][vis_start + visit_k])
            start_next = int(s["visit_starts"][vis_start + visit_k + 1])
            end_next = int(s["visit_ends"][vis_start + visit_k + 1])
            codes = np.asarray(s["code_indices"][ev_start:ev_start + end_curr], dtype=np.int64)
            if codes.shape[0] > ds.max_seq_len:
                codes = codes[-ds.max_seq_len:]
            real = np.unique(codes[(codes != unk) & (codes < V)])
            nxt = np.asarray(
                s["code_indices"][ev_start + start_next:ev_start + end_next], dtype=np.int64
            )
            tgt = np.unique(nxt[(nxt != unk) & (nxt < V)])
            for c in real.tolist():
                rows.append((wi, int(c), 0))
            for c in tgt.tolist():
                rows.append((wi, int(c), 1))
    if not rows:
        return np.zeros((0, 3), dtype=np.int32), N
    return np.asarray(rows, dtype=np.int32), N


def _scores_persistence_duckdb(con, n_examples: int, V: int) -> np.ndarray:
    scores = np.zeros((n_examples, V), dtype=np.float32)
    rows = con.execute(
        "SELECT example_id, code, count FROM val_events WHERE role = 0"
    ).fetchall()
    for eid, c, n in rows:
        if 0 <= int(c) < V:
            scores[int(eid), int(c)] = float(n)
    return scores


def _cooc_scores_sql(con, n_examples: int, V: int, n_windows: int,
                     top_k_codes: int | None) -> tuple[np.ndarray, dict]:
    """Score via SQL MAX(lift); return sparse-filled dense scores + diagnostics.

    If ``top_k_codes`` is set, restrict both sides of the lift table to the top-K
    codes by input-window frequency.
    """
    top_filter = ""
    if top_k_codes is not None:
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE top_codes AS
            SELECT code FROM train_flat WHERE role = 0
            GROUP BY code ORDER BY COUNT(*) DESC LIMIT {int(top_k_codes)}
        """)
        top_filter = """
            AND i.code IN (SELECT code FROM top_codes)
            AND t.code IN (SELECT code FROM top_codes)
        """

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pair_counts AS
        SELECT i.code AS c_in, t.code AS c_tgt, COUNT(*)::BIGINT AS n
        FROM train_flat i
        INNER JOIN train_flat t
          ON i.window_id = t.window_id AND i.role = 0 AND t.role = 1
        WHERE 1=1 {top_filter}
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE code_marg AS
        SELECT code, COUNT(DISTINCT window_id)::BIGINT AS n_win
        FROM train_flat WHERE role = 0
        GROUP BY 1
    """)
    # lift(c→c') = P(c'|c) / P(c') = (n(c,c')/n(c)) / (n(c')/N)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE lift AS
        SELECT p.c_in, p.c_tgt,
               (p.n::DOUBLE * {float(n_windows)})
                 / (NULLIF(m_in.n_win, 0)::DOUBLE * NULLIF(m_tgt.n_win, 0)::DOUBLE)
               AS lift
        FROM pair_counts p
        INNER JOIN code_marg m_in  ON m_in.code  = p.c_in
        INNER JOIN code_marg m_tgt ON m_tgt.code = p.c_tgt
        WHERE p.n > 0
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE val_scores AS
        SELECT v.example_id, l.c_tgt AS code, MAX(l.lift) AS score
        FROM val_events v
        INNER JOIN lift l ON v.code = l.c_in AND v.role = 0
        GROUP BY 1, 2
    """)
    scores = np.zeros((n_examples, V), dtype=np.float32)
    rows = con.execute("SELECT example_id, code, score FROM val_scores").fetchall()
    for eid, c, s in rows:
        if 0 <= int(c) < V:
            scores[int(eid), int(c)] = float(s)
    n_pairs = con.execute("SELECT COUNT(*) FROM pair_counts").fetchone()[0]
    n_lift = con.execute("SELECT COUNT(*) FROM lift").fetchone()[0]
    return scores, {
        "top_k_codes": top_k_codes,
        "n_pair_rows": int(n_pairs),
        "n_lift_rows": int(n_lift),
        "n_scored_cells": int(len(rows)),
    }


def _targets_matrix(store: dict) -> np.ndarray:
    parts = [b["target_codes"].numpy() for b in iter_store_batches(store)]
    return np.concatenate(parts, axis=0).astype(np.float32)


def _recall_dict(scores: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    rec = recall_from_scores(torch.from_numpy(scores), torch.from_numpy(targets), ks=KS)
    return {f"recall@{k}": rec[k] for k in KS}


def persistence_handcheck(store: dict, n_patients: int = 10) -> dict:
    V = int(store["num_codes"])
    scores = np.zeros((n_patients, V), dtype=np.float32)
    targets = np.zeros((n_patients, V), dtype=np.float32)
    eid = 0
    for batch in iter_store_batches(store):
        bsz = int(batch["lengths"].shape[0])
        mask = batch["attention_mask"].numpy()
        raw = batch["code_indices"].numpy()
        tgt = batch["target_codes"].numpy()
        for i in range(bsz):
            if eid >= n_patients:
                break
            idx = np.flatnonzero(mask[i])
            real = raw[i, idx]
            real = real[real >= 2] - 2
            if real.size:
                uniq, cnt = np.unique(real, return_counts=True)
                scores[eid, uniq] = cnt.astype(np.float32)
            targets[eid] = tgt[i]
            eid += 1
        if eid >= n_patients:
            break
    rec = recall_from_scores(torch.from_numpy(scores[:eid]),
                             torch.from_numpy(targets[:eid]), ks=(10,))
    return {
        "n": int(eid),
        "recall@10_per_example": rec[10].tolist(),
        "recall@10": float(np.nanmean(rec[10])),
    }


def run_d9(ctx: dict, store: dict) -> dict:
    flags = ctx["flags"]
    V = int(ctx["shared"]["num_codes"])
    n_examples = int(store["n_examples"])
    patient_ids = np.asarray(store["patient_ids"])
    seed = int(ctx.get("seed", 0))

    ex_ids, codes, counts, roles = _val_flat_from_store(store)
    con = open_duckdb()
    con.execute(
        "CREATE TABLE val_events (example_id INTEGER, code INTEGER, count INTEGER, role INTEGER)"
    )
    import pandas as pd
    ev_df = pd.DataFrame({
        "example_id": ex_ids, "code": codes, "count": counts, "role": roles.astype(np.int32),
    })
    con.register("ev_df", ev_df)
    con.execute("INSERT INTO val_events SELECT * FROM ev_df")
    con.unregister("ev_df")

    persist_scores = _scores_persistence_duckdb(con, n_examples, V)
    targets = _targets_matrix(store)

    hand = persistence_handcheck(store, n_patients=10)
    duck_r10 = _recall_dict(persist_scores[: hand["n"]], targets[: hand["n"]])["recall@10"]
    if not np.allclose(duck_r10, np.asarray(hand["recall@10_per_example"]),
                       rtol=0, atol=1e-7, equal_nan=True):
        raise AssertionError("[HARD] D9 hand-check failed: DuckDB persistence != plain Python")

    max_win = 2000 if flags["smoke"] else None
    train_flat, n_windows = _train_flat_table(
        ctx, patient_frac=float(flags["patient_frac"]), seed=seed, max_windows=max_win,
    )
    con.execute("CREATE TABLE train_flat (window_id INTEGER, code INTEGER, role INTEGER)")
    if train_flat.size:
        con.register("tf", train_flat)
        con.execute("INSERT INTO train_flat SELECT column0, column1, column2 FROM tf")
        con.unregister("tf")

    # Global prior from train input frequencies.
    marg = con.execute("""
        SELECT code, COUNT(DISTINCT window_id) AS n
        FROM train_flat WHERE role = 0 GROUP BY 1
    """).fetchall()
    code_count = np.zeros(V, dtype=np.float32)
    for c, n in marg:
        if 0 <= int(c) < V:
            code_count[int(c)] = float(n)
    prior_scores = np.broadcast_to(code_count, (n_examples, V)).copy()

    # Top-2000 restricted co-occurrence (production path).
    top_n = min(int(D9_TOP_CODES_LIFT), V)
    if flags["smoke"]:
        top_n = min(max(50, int(flags["top_codes"]) * 5), V)
    cooc_scores, cooc_meta = _cooc_scores_sql(
        con, n_examples, V, n_windows, top_k_codes=top_n,
    )

    # Smoke validation: full-vocab vs restricted recall@10 must agree to < 0.002.
    full_vs_restricted = None
    if flags["smoke"]:
        full_scores, full_meta = _cooc_scores_sql(
            con, n_examples, V, n_windows, top_k_codes=None,
        )
        r_full = nanmean_safe(_recall_dict(full_scores, targets)["recall@10"])
        r_rest = nanmean_safe(_recall_dict(cooc_scores, targets)["recall@10"])
        delta = abs(r_full - r_rest)
        full_vs_restricted = {
            "recall@10_full_vocab": r_full,
            "recall@10_top_restricted": r_rest,
            "abs_delta": delta,
            "threshold": 0.002,
            "agree": bool(delta < 0.002),
            "full_meta": full_meta,
        }
        if not full_vs_restricted["agree"]:
            D.print_block("D9 WARNING", [
                f"full vs top-{top_n} recall@10 delta={delta:.4f} >= 0.002",
            ])

    results = {}
    for name, scores in (
        ("persistence", persist_scores),
        ("cooccurrence", cooc_scores),
        ("global_prior", prior_scores),
    ):
        rec = _recall_dict(scores, targets)
        n_pos, n_hit = per_code_hit_miss(
            torch.from_numpy(scores), torch.from_numpy(targets), ks=KS,
        )
        results[name] = {
            **{f"recall@{k}": nanmean_safe(rec[f"recall@{k}"]) for k in KS},
            "recall_per_example": rec,
            "per_code_hit_miss": serialize_per_code_hit_miss(n_pos, n_hit, ks=KS),
        }

    deltas = {}
    for name in ("cooccurrence", "global_prior"):
        deltas[f"{name}_minus_persistence_recall@10"] = paired_delta_ci(
            results[name]["recall_per_example"]["recall@10"],
            results["persistence"]["recall_per_example"]["recall@10"],
            patient_ids, n_boot=int(flags["n_boot"]), seed=seed,
        )

    summary = {
        name: {
            **{f"recall@{k}": results[name][f"recall@{k}"] for k in KS},
            "per_code_hit_miss": results[name]["per_code_hit_miss"],
        }
        for name in results
    }
    near_arm = bool(max(summary["persistence"]["recall@10"],
                        summary["cooccurrence"]["recall@10"]) >= 0.12)

    out = {
        **base_result_meta(ctx, store),
        "baselines": summary,
        "cooccurrence_sql": cooc_meta,
        "full_vs_restricted": full_vs_restricted,
        "deltas_vs_persistence": {
            k: {
                "point": v["point"],
                "ci": {
                    "lo": v["ci"]["lo"], "hi": v["ci"]["hi"],
                    "excludes_zero": v["ci"].get("excludes_zero"),
                    "degenerate": v["ci"].get("degenerate", False),
                },
                "covers_zero": v["covers_zero"],
            }
            for k, v in deltas.items()
        },
        "handcheck": {
            "n": hand["n"],
            "recall@10_python": hand["recall@10"],
            "recall@10_duckdb": float(np.nanmean(duck_r10)),
            "match": True,
        },
        "arm_reference": {
            "vanilla_recall@10": 0.1381,
            "kernel_recall@10": 0.1339,
            "band": "0.134–0.138",
            "standing_note": (
                "persistence 0.083 and global prior 0.105 against arms at ~0.137 "
                "means neither crossed the escalation threshold; co-occurrence is "
                "the last baseline that could."
            ),
        },
        "verdict": {
            "persistence_or_cooc_near_arm_recall@10": near_arm,
            "route": (
                "backbone near-baseline → stop, escalate"
                if near_arm
                else "baselines below arms → backbone has headroom over recurrence"
            ),
        },
        "n_train_windows_for_stats": n_windows,
    }
    con.close()
    return out


def main(argv: list[str] | None = None) -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = p.parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    D.print_block("D9 trivial baselines", [f"out={out_dir}  smoke={args.smoke}"])
    ctx, store = ensure_batches(
        out_dir, smoke=args.smoke, batch_size=args.batch_size,
        force=args.force, run_root=args.run_root,
    )
    ctx["seed"] = args.seed
    result = run_d9(ctx, store)
    write_json_atomic(out_dir / "d9_baselines.json", result)
    b = result["baselines"]
    lines = [
        f"batch_list_hash={result['batch_list_hash']}",
        f"persistence  r@5/10/20="
        f"{b['persistence']['recall@5']:.4f}/"
        f"{b['persistence']['recall@10']:.4f}/"
        f"{b['persistence']['recall@20']:.4f}",
        f"cooccurrence r@5/10/20="
        f"{b['cooccurrence']['recall@5']:.4f}/"
        f"{b['cooccurrence']['recall@10']:.4f}/"
        f"{b['cooccurrence']['recall@20']:.4f}",
        f"global_prior r@5/10/20="
        f"{b['global_prior']['recall@5']:.4f}/"
        f"{b['global_prior']['recall@10']:.4f}/"
        f"{b['global_prior']['recall@20']:.4f}",
        f"handcheck match={result['handcheck']['match']}",
        f"verdict: {result['verdict']['route']}",
    ]
    if result.get("full_vs_restricted"):
        fvr = result["full_vs_restricted"]
        lines.append(
            f"full vs restricted r@10: {fvr['recall@10_full_vocab']:.4f} vs "
            f"{fvr['recall@10_top_restricted']:.4f}  Δ={fvr['abs_delta']:.4f} "
            f"agree={fvr['agree']}"
        )
    D.print_block("D9 results", lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
