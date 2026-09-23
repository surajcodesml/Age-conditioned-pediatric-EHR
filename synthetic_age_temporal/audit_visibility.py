#!/usr/bin/env python3
"""Signal visibility audit + visible-oracle recomputation after model truncation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    MAX_BACKGROUND_EVENTS,
    MAX_SEQ_LEN,
    SIGNAL_LAGS_DAYS,
    DATA_SEED,
)
from dataset import apply_model_truncation, load_scenario_dir
from ground_truth import ExampleSignals, oracle_predict


LAG_BINS = list(SIGNAL_LAGS_DAYS)
AGE_GROUPS = [
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.01),
]


def _age_group(age: float) -> str:
    for name, lo, hi in AGE_GROUPS:
        if lo <= age < hi:
            return name
    return "other"


def _nearest_lag_bin(lag: float) -> float:
    return float(min(LAG_BINS, key=lambda b: abs(b - lag)))


def _bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _auroc_pack(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {
        "bce": _bce(y, p),
        "micro_auroc": float("nan"),
        "micro_auprc": float("nan"),
    }
    try:
        out["micro_auroc"] = float(roc_auc_score(y.ravel(), p.ravel()))
        out["micro_auprc"] = float(average_precision_score(y.ravel(), p.ravel()))
    except ValueError:
        pass
    return out


def audit_visibility(
    scenario_dir: Path,
    *,
    max_seq_len: int = MAX_SEQ_LEN,
    max_background: int = MAX_BACKGROUND_EVENTS,
) -> dict[str, Any]:
    examples, labels, meta, specs = load_scenario_dir(scenario_dir)
    gt = pd.read_parquet(scenario_dir / "ground_truth.parquet")
    sig_gt = gt[gt["signal_event_code"].notna()].copy()

    # Per injected signal: was it retained after truncation?
    rows = []
    ex_stats = []
    for row in examples.itertuples(index=False):
        types = list(row.history_types)
        lags = [float(x) for x in row.history_lag_days]
        codes = list(row.history_codes)
        sig_before = [
            (str(codes[i]), float(lags[i]))
            for i, t in enumerate(types)
            if t == "signal"
        ]
        trunc = apply_model_truncation(
            codes,
            types,
            lags,
            list(row.history_tau) if isinstance(row.history_tau, list) else None,
            max_seq_len=max_seq_len,
            max_background=max_background,
        )
        retained = set(
            zip(
                [str(c) for c in trunc["retained_signal_codes"]],
                [round(float(l), 6) for l in trunc["retained_signal_lags"]],
            )
        )
        # Match by code+approx lag (jitter makes exact float match fragile).
        retained_lags = trunc["retained_signal_lags"]
        visible_flags = []
        for code, lag in sig_before:
            # Visible if some retained signal has same code and lag within 1%.
            ok = False
            for rc, rl in zip(trunc["retained_signal_codes"], retained_lags):
                if str(rc) == code and abs(float(rl) - lag) / max(lag, 1e-6) < 0.02:
                    ok = True
                    break
            visible_flags.append(ok)
            rows.append(
                {
                    "example_id": int(row.example_id),
                    "age": float(row.age_at_cutoff),
                    "age_group": _age_group(float(row.age_at_cutoff)),
                    "lag": lag,
                    "lag_bin": _nearest_lag_bin(lag),
                    "code": code,
                    "visible": bool(ok),
                    "n_events_before": trunc["n_events_before"],
                    "seq_len_history": trunc["seq_len_history"],
                    "truncated": trunc["truncated"],
                }
            )
        ex_stats.append(
            {
                "example_id": int(row.example_id),
                "age": float(row.age_at_cutoff),
                "n_signal_before": len(sig_before),
                "n_signal_after": int(sum(visible_flags)),
                "any_visible": bool(any(visible_flags)) if sig_before else False,
                "all_visible": bool(all(visible_flags)) if sig_before else True,
                "n_events_before": trunc["n_events_before"],
                "seq_len_history": trunc["seq_len_history"],
                "truncated": trunc["truncated"],
            }
        )

    sig_df = pd.DataFrame(rows)
    ex_df = pd.DataFrame(ex_stats)

    p_vis_lag = {}
    for b in LAG_BINS:
        sub = sig_df[np.isclose(sig_df["lag_bin"], b)]
        p_vis_lag[str(int(b))] = float(sub["visible"].mean()) if len(sub) else float("nan")

    p_vis_lag_age = {}
    for gname, _, _ in AGE_GROUPS:
        p_vis_lag_age[gname] = {}
        for b in LAG_BINS:
            sub = sig_df[
                (sig_df["age_group"] == gname) & np.isclose(sig_df["lag_bin"], b)
            ]
            p_vis_lag_age[gname][str(int(b))] = (
                float(sub["visible"].mean()) if len(sub) else float("nan")
            )

    # Visibility vs raw sequence length (before truncation).
    length_bins = [0, 50, 100, 200, 400, 800, 10_000]
    vis_by_len = {}
    for lo, hi in zip(length_bins[:-1], length_bins[1:]):
        sub = ex_df[(ex_df["n_events_before"] >= lo) & (ex_df["n_events_before"] < hi)]
        if len(sub) == 0:
            continue
        vis_by_len[f"[{lo},{hi})"] = {
            "n_examples": int(len(sub)),
            "frac_any_visible": float(sub["any_visible"].mean()),
            "frac_all_visible": float(sub["all_visible"].mean()),
            "mean_n_signal_after": float(sub["n_signal_after"].mean()),
            "mean_n_signal_before": float(sub["n_signal_before"].mean()),
        }

    summary = {
        "scenario": meta["scenario"],
        "max_seq_len": max_seq_len,
        "max_background": max_background,
        "n_examples": int(len(ex_df)),
        "n_injected_signals": int(len(sig_df)),
        "mean_n_signal_before": float(ex_df["n_signal_before"].mean()),
        "mean_n_signal_after": float(ex_df["n_signal_after"].mean()),
        "frac_examples_any_visible": float(ex_df["any_visible"].mean()),
        "frac_examples_all_visible": float(ex_df["all_visible"].mean()),
        "frac_examples_truncated": float(ex_df["truncated"].mean()),
        "P_visible_given_lag": p_vis_lag,
        "P_visible_given_lag_by_age_group": p_vis_lag_age,
        "visibility_by_raw_sequence_length": vis_by_len,
        "overall_signal_visibility": float(sig_df["visible"].mean()) if len(sig_df) else float("nan"),
    }
    return summary, sig_df, ex_df, examples, labels, meta, specs


def visible_signal_list(
    examples: pd.DataFrame,
    *,
    max_seq_len: int,
    max_background: int,
) -> list[ExampleSignals]:
    out: list[ExampleSignals] = []
    for row in examples.itertuples(index=False):
        trunc = apply_model_truncation(
            list(row.history_codes),
            list(row.history_types),
            list(row.history_lag_days),
            list(row.history_tau) if isinstance(row.history_tau, list) else None,
            max_seq_len=max_seq_len,
            max_background=max_background,
        )
        codes = np.array(trunc["retained_signal_codes"], dtype=object)
        lags = np.asarray(trunc["retained_signal_lags"], dtype=np.float64)
        from config import tau_from_days

        taus = tau_from_days(lags) if lags.size else np.zeros(0)
        out.append(
            ExampleSignals(
                codes=codes,
                lag_days=lags,
                tau=np.asarray(taus, dtype=np.float64),
                times=np.array([], dtype="datetime64[ns]"),
            )
        )
    return out


def full_signal_list(examples: pd.DataFrame) -> list[ExampleSignals]:
    from config import tau_from_days

    out: list[ExampleSignals] = []
    for row in examples.itertuples(index=False):
        types = list(row.history_types)
        codes = list(row.history_codes)
        lags = [float(x) for x in row.history_lag_days]
        sig_codes = [codes[i] for i, t in enumerate(types) if t == "signal"]
        sig_lags = np.asarray(
            [lags[i] for i, t in enumerate(types) if t == "signal"], dtype=np.float64
        )
        taus = tau_from_days(sig_lags) if sig_lags.size else np.zeros(0)
        out.append(
            ExampleSignals(
                codes=np.array(sig_codes, dtype=object),
                lag_days=sig_lags,
                tau=np.asarray(taus, dtype=np.float64),
                times=np.array([], dtype="datetime64[ns]"),
            )
        )
    return out


def compute_visible_oracle(
    scenario_dir: Path,
    *,
    max_seq_len: int = MAX_SEQ_LEN,
    max_background: int = MAX_BACKGROUND_EVENTS,
) -> dict[str, Any]:
    examples, labels, meta, specs = load_scenario_dir(scenario_dir)
    ages = examples["age_at_cutoff"].to_numpy(dtype=np.float64)
    theta0 = float(meta["theta0"])
    beta = float(meta["beta_true"])
    scenario = meta["scenario"]
    inter_idx = [i for i, sp in enumerate(specs) if sp["mechanism"] == "interaction"]

    full_sigs = full_signal_list(examples)
    vis_sigs = visible_signal_list(
        examples, max_seq_len=max_seq_len, max_background=max_background
    )

    def pack(sigs: list[ExampleSignals], mode: str, seed: int = 0) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        p = oracle_predict(
            ages=ages,
            signal_list=sigs,
            specs=specs,
            scenario=scenario,
            theta0=theta0,
            beta=beta,
            mode=mode,
            shuffle_rng=rng,
        )
        y = labels
        if inter_idx:
            m_all = _auroc_pack(y, p)
            m_int = _auroc_pack(y[:, inter_idx], p[:, inter_idx])
            m_all["interaction"] = m_int
            return m_all
        return _auroc_pack(y, p)

    full_correct = pack(full_sigs, "correct", 1)
    full_shuf = pack(full_sigs, "shuffle_age", 2)
    full_lag = pack(full_sigs, "shuffle_lag", 3)
    full_noint = pack(full_sigs, "no_interaction", 4)

    vis_correct = pack(vis_sigs, "correct", 1)
    vis_shuf = pack(vis_sigs, "shuffle_age", 2)
    vis_lag = pack(vis_sigs, "shuffle_lag", 3)
    vis_noint = pack(vis_sigs, "no_interaction", 4)

    def deltas(correct, shuf, lag, noint):
        return {
            "delta_bce_shuffle_age": shuf["bce"] - correct["bce"],
            "delta_bce_shuffle_lag": lag["bce"] - correct["bce"],
            "delta_bce_no_interaction": noint["bce"] - correct["bce"],
            "delta_bce_shuffle_age_interaction": (
                shuf.get("interaction", shuf)["bce"]
                - correct.get("interaction", correct)["bce"]
            ),
            "delta_bce_no_interaction_interaction": (
                noint.get("interaction", noint)["bce"]
                - correct.get("interaction", correct)["bce"]
            ),
        }

    return {
        "scenario": scenario,
        "max_seq_len": max_seq_len,
        "full_oracle": {
            "correct": full_correct,
            "shuffle_age": full_shuf,
            "shuffle_lag": full_lag,
            "no_interaction": full_noint,
            **deltas(full_correct, full_shuf, full_lag, full_noint),
        },
        "visible_oracle": {
            "correct": vis_correct,
            "shuffle_age": vis_shuf,
            "shuffle_lag": vis_lag,
            "no_interaction": vis_noint,
            **deltas(vis_correct, vis_shuf, vis_lag, vis_noint),
        },
    }


def plot_visibility(summary: dict, out_path: Path) -> None:
    lags = [int(k) for k in summary["P_visible_given_lag"]]
    p = [summary["P_visible_given_lag"][str(l)] for l in lags]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lags, p, "o-", color="#2b6cb0", lw=2)
    ax.set_xlabel("Injected signal lag (days)")
    ax.set_ylabel(r"$P(\mathrm{signal\ visible}\mid \mathrm{lag})$")
    ax.set_title(f"Figure 7 — Signal visibility after truncation (L={summary['max_seq_len']})")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Age-group curves
    for gname, style in zip(
        [g[0] for g in AGE_GROUPS], ["--", "-.", ":", (0, (3, 1, 1, 1))]
    ):
        pg = summary["P_visible_given_lag_by_age_group"].get(gname, {})
        if not pg:
            continue
        ys = [pg.get(str(l), np.nan) for l in lags]
        ys_arr = np.asarray(ys, dtype=np.float64)
        if np.all(np.isnan(ys_arr)):
            continue
        ax.plot(lags, ys_arr, linestyle=style, marker=".", label=gname, alpha=0.85)
    ax.legend(frameon=False, title="age group")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_OUTPUT_DIR / "data" / f"seed{DATA_SEED}")
    ap.add_argument("--cohort", default="controlled")
    ap.add_argument("--scenarios", nargs="+", default=["S0", "S1", "S2", "S3"])
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR / "followup")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_vis = {}
    all_oracle = {}
    for scen in args.scenarios:
        sdir = args.data_root / args.cohort / scen
        if not sdir.exists():
            print("skip missing", sdir)
            continue
        summary, sig_df, ex_df, *_ = audit_visibility(
            sdir, max_seq_len=args.max_seq_len
        )
        oracle = compute_visible_oracle(sdir, max_seq_len=args.max_seq_len)
        all_vis[scen] = summary
        all_oracle[scen] = oracle
        (args.out_dir / f"visibility_{scen}_L{args.max_seq_len}.json").write_text(
            json.dumps(summary, indent=2)
        )
        (args.out_dir / f"visible_oracle_{scen}_L{args.max_seq_len}.json").write_text(
            json.dumps(oracle, indent=2)
        )
        if scen == "S2":
            plot_visibility(
                summary, args.out_dir / ".." / "figures" / f"fig7_signal_visibility_L{args.max_seq_len}"
            )
        print(
            f"[{scen}] vis@730d={summary['P_visible_given_lag'].get('730', float('nan')):.3f} "
            f"all_vis={summary['frac_examples_all_visible']:.3f} "
            f"full_dAge={oracle['full_oracle']['delta_bce_shuffle_age_interaction']:.4f} "
            f"vis_dAge={oracle['visible_oracle']['delta_bce_shuffle_age_interaction']:.4f}"
        )

    with (args.out_dir / f"visibility_summary_L{args.max_seq_len}.json").open("w") as f:
        json.dump(all_vis, f, indent=2)
    with (args.out_dir / f"visible_oracle_summary_L{args.max_seq_len}.json").open("w") as f:
        json.dump(all_oracle, f, indent=2)


if __name__ == "__main__":
    main()
