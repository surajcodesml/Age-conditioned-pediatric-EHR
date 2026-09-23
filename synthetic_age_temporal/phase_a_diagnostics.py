#!/usr/bin/env python3
"""Phase A diagnostics: age probe, counterfactual age sensitivity, helpers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    AGE_CENTER,
    AGE_SCALE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    DATA_SEED,
    z_age,
)
from dataset import make_loaders
from ground_truth import ExampleSignals, compute_target_logits, sigmoid
from model import BenchmarkModel

AGE_BANDS = [("<1", 0.0, 1.0), ("1-5", 1.0, 6.0), ("6-11", 6.0, 12.0), ("12-17", 12.0, 18.01)]
CF_AGES = [2.0, 5.0, 9.0, 13.0, 17.0]


def _age_band(a: float) -> int:
    for i, (_, lo, hi) in enumerate(AGE_BANDS):
        if lo <= a < hi:
            return i
    return len(AGE_BANDS) - 1


def _load_model(run_dir: Path, device: torch.device, n_targets: int | None = None) -> BenchmarkModel:
    cfg = json.loads((run_dir / "config.json").read_text())
    # Infer vocab size from checkpoint embedding.
    state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    n_codes = int(state["code_emb.weight"].shape[0])
    n_types = int(state["type_emb.weight"].shape[0])
    if n_targets is None:
        n_targets = int(state["head.weight"].shape[0])
    model = BenchmarkModel(
        arm=cfg["arm"],
        n_codes=n_codes,
        n_types=n_types,
        n_targets=n_targets,
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 1),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=0.0,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def collect_reprs(
    model: BenchmarkModel, loader, device: torch.device
) -> dict[str, np.ndarray]:
    pre, pooled, ages, bands = [], [], [], []
    heads = []
    for batch in loader:
        batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        r = model.extract_repr(
            batch_t["code_ids"],
            batch_t["type_ids"],
            batch_t["tau"],
            batch_t["padding_mask"],
            batch_t["is_query"],
            batch_t["age"],
            lag_days=batch_t["lag_days"],
        )
        pre.append(r["pre_pool"].cpu().numpy())
        pooled.append(r["pooled"].cpu().numpy())
        heads.append(r["head_ctx"].mean(dim=1).cpu().numpy())  # mean over heads
        ages.append(batch["age"].numpy())
        bands.append(np.array([_age_band(float(a)) for a in batch["age"].numpy()]))
    return {
        "pre_pool": np.concatenate(pre),
        "pooled": np.concatenate(pooled),
        "head_mean": np.concatenate(heads),
        "age": np.concatenate(ages),
        "band": np.concatenate(bands),
    }


def fit_age_probe(X_tr, y_tr, X_te, y_te, band_te) -> dict[str, float]:
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))
    # Band accuracy from continuous prediction
    pred_band = np.array([_age_band(float(a)) for a in pred])
    band_acc = float((pred_band == band_te).mean())
    # Also multinomial on true bands from features
    clf = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, multi_class="multinomial")),
        ]
    )
    # Need train bands
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "band_acc_from_reg": band_acc,
    }


def run_age_probe(
    *,
    scenario_dir: Path,
    run_dirs: dict[str, Path],
    device: torch.device,
    out_json: Path,
    fig_path: Path,
) -> dict[str, Any]:
    results = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(3)
    width = 0.25
    sites = ["pre_pool", "pooled", "head_mean"]
    for i, (arm, run_dir) in enumerate(run_dirs.items()):
        inter = "interonly" in str(run_dir)
        target_idx = None
        if inter:
            specs = json.loads((scenario_dir / "target_specs.json").read_text())
            target_idx = [j for j, s in enumerate(specs) if s["mechanism"] == "interaction"]
        train_loader, _, test_loader, _, info = make_loaders(
            scenario_dir, batch_size=64, target_idx=target_idx
        )
        model = _load_model(run_dir, device, n_targets=info["n_targets"])
        tr = collect_reprs(model, train_loader, device)
        te = collect_reprs(model, test_loader, device)
        arm_res = {}
        maes = []
        for site in sites:
            m = fit_age_probe(
                tr[site], tr["age"], te[site], te["age"], te["band"]
            )
            # band clf
            clf = Pipeline(
                [
                    ("sc", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(max_iter=3000, multi_class="multinomial"),
                    ),
                ]
            )
            clf.fit(tr[site], tr["band"])
            m["band_acc"] = float(clf.score(te[site], te["band"]))
            arm_res[site] = m
            maes.append(m["mae"])
        results[arm] = arm_res
        ax.bar(x + (i - 1) * width, maes, width, label=arm)
        print(
            f"[age-probe] {arm}: pooled MAE={arm_res['pooled']['mae']:.3f} "
            f"R2={arm_res['pooled']['r2']:.3f} band_acc={arm_res['pooled']['band_acc']:.3f}"
        )
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Age MAE (years)")
    ax.set_title("Figure 13 — Age decoding from frozen representations (S2)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    return results


def _signals_from_example(row) -> ExampleSignals:
    types = list(row.history_types)
    codes = list(row.history_codes)
    lags = [float(x) for x in row.history_lag_days]
    sig_c = [codes[i] for i, t in enumerate(types) if t == "signal"]
    sig_l = np.asarray([lags[i] for i, t in enumerate(types) if t == "signal"], dtype=np.float64)
    from config import tau_from_days

    tau = tau_from_days(sig_l) if sig_l.size else np.zeros(0)
    return ExampleSignals(
        codes=np.array(sig_c, dtype=object),
        lag_days=sig_l,
        tau=np.asarray(tau, dtype=np.float64),
        times=np.array([], dtype="datetime64[ns]"),
    )


@torch.no_grad()
def _model_probs_at_age(
    model: BenchmarkModel,
    batch: dict[str, torch.Tensor],
    age: float,
    device: torch.device,
    inter_idx: list[int] | None,
) -> np.ndarray:
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    bsz = b["age"].size(0)
    b["age"] = torch.full((bsz,), float(age), device=device)
    logits = model(
        code_ids=b["code_ids"],
        type_ids=b["type_ids"],
        tau=b["tau"],
        padding_mask=b["padding_mask"],
        is_query=b["is_query"],
        age=b["age"],
        lag_days=b["lag_days"],
    )
    p = torch.sigmoid(logits).cpu().numpy()
    if inter_idx is not None and p.shape[1] > len(inter_idx):
        # full-label model: slice interaction cols
        return p[:, inter_idx]
    return p


def run_counterfactual(
    *,
    scenario_dir: Path,
    run_dirs: dict[str, Path],
    device: torch.device,
    scenario: str,
    out_json: Path,
    fig_path: Path,
    n_examples: int = 200,
) -> dict[str, Any]:
    import pandas as pd

    examples = pd.read_parquet(scenario_dir / "examples.parquet")
    specs = json.loads((scenario_dir / "target_specs.json").read_text())
    meta = json.loads((scenario_dir / "meta.json").read_text())
    inter_idx = [i for i, s in enumerate(specs) if s["mechanism"] == "interaction"]
    test = examples[examples["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(0)
    take = rng.choice(len(test), size=min(n_examples, len(test)), replace=False)
    sub = test.iloc[take].reset_index(drop=True)

    # Build a loader for these example_ids via full test loader filter is hard;
    # instead reconstruct batches from make_loaders and match example_id.
    target_idx = inter_idx  # evaluate interaction labels
    # Use interonly checkpoints when available (n_targets=8)
    models = {}
    for arm, rd in run_dirs.items():
        nt = 8 if "interonly" in str(rd) else None
        models[arm] = _load_model(rd, device, n_targets=nt)

    train_loader, _, test_loader, _, info = make_loaders(
        scenario_dir,
        batch_size=32,
        target_idx=inter_idx if any("interonly" in str(rd) for rd in run_dirs.values()) else None,
    )
    # Collect test tensors keyed by example_id
    by_eid: dict[int, dict] = {}
    # Need example_id in batch — dataset has it but collate drops it.
    # Rebuild from dataset directly.
    from dataset import BenchmarkDataset, build_vocab, collate_batch, load_scenario_dir

    examples_all, labels, _, _ = load_scenario_dir(scenario_dir)
    vocab = build_vocab(examples_all)
    ds = BenchmarkDataset(
        examples_all,
        labels,
        "test",
        vocab,
        target_idx=inter_idx if any("interonly" in str(rd) for rd in run_dirs.values()) else None,
    )
    eid_to_idx = {int(ds.examples.iloc[i]["example_id"]): i for i in range(len(ds))}

    curves = {arm: {str(a): [] for a in CF_AGES} for arm in models}
    curves["oracle"] = {str(a): [] for a in CF_AGES}
    for row in sub.itertuples(index=False):
        eid = int(row.example_id)
        if eid not in eid_to_idx:
            continue
        item = ds[eid_to_idx[eid]]
        batch = collate_batch([item])
        sig = _signals_from_example(row)
        for a in CF_AGES:
            # Oracle on interaction labels only
            logits, probs, _, _ = compute_target_logits(
                age=float(a),
                signals=sig,
                specs=specs,
                scenario=scenario,
                theta0=float(meta["theta0"]),
                beta=float(meta["beta_true"]),
                noise=np.zeros(len(specs)),
            )
            curves["oracle"][str(a)].append(float(probs[inter_idx].mean()))
            for arm, model in models.items():
                p = _model_probs_at_age(
                    model,
                    batch,
                    a,
                    device,
                    inter_idx if model.head.out_features > len(inter_idx) else None,
                )
                curves[arm][str(a)].append(float(p.mean()))

    summary = {"scenario": scenario, "n_examples": len(sub), "ages": CF_AGES, "mean_p": {}}
    for name, c in curves.items():
        summary["mean_p"][name] = {
            str(a): float(np.mean(c[str(a)])) if c[str(a)] else float("nan") for a in CF_AGES
        }
        # Slope approx (17 vs 2)
        p2 = summary["mean_p"][name].get("2.0", np.nan)
        p17 = summary["mean_p"][name].get("17.0", np.nan)
        summary["mean_p"][name]["delta_17_minus_2"] = float(p17 - p2) if np.isfinite(p2) else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, style in (
        ("oracle", "k--"),
        ("temporal_only", "-"),
        ("age_temporal", "-"),
    ):
        if name not in summary["mean_p"]:
            continue
        ys = [summary["mean_p"][name][str(a)] for a in CF_AGES]
        ax.plot(CF_AGES, ys, style, marker="o", label=name)
    ax.set_xlabel(r"Counterfactual cutoff age $a_*$")
    ax.set_ylabel("Mean predicted P (interaction labels)")
    ax.set_title(f"Figure 14 — Counterfactual age sensitivity ({scenario})")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["mean_p"], indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenario-dir", type=Path, default=None)
    ap.add_argument("--data-seed", type=int, default=DATA_SEED)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = DEFAULT_OUTPUT_DIR / "runs" / "controlled"
    sdir = args.scenario_dir or (
        DEFAULT_OUTPUT_DIR / "data" / f"seed{args.data_seed}" / "controlled" / "S2"
    )
    run_dirs = {
        "no_age": root / "S2_no_age_d20260922_m0",
        "temporal_only": root / "followup_S2_temporal_only_d20260922_m0_interonly",
        "age_temporal": root / "followup_S2_age_temporal_d20260922_m0_interonly",
    }
    # Fallback if followup missing
    for k, p in list(run_dirs.items()):
        if not (p / "model.pt").exists():
            alt = root / f"S2_{k}_d20260922_m0"
            if (alt / "model.pt").exists():
                run_dirs[k] = alt

    follow = DEFAULT_RESULTS_DIR / "followup"
    follow.mkdir(parents=True, exist_ok=True)
    run_age_probe(
        scenario_dir=sdir,
        run_dirs=run_dirs,
        device=device,
        out_json=follow / "followup_age_probe.json",
        fig_path=DEFAULT_RESULTS_DIR / "figures" / "fig13_age_probe",
    )
    # Counterfactual: prefer interonly temporal/age models
    cf_dirs = {
        "temporal_only": run_dirs["temporal_only"],
        "age_temporal": run_dirs["age_temporal"],
    }
    run_counterfactual(
        scenario_dir=sdir,
        run_dirs=cf_dirs,
        device=device,
        scenario="S2",
        out_json=follow / "counterfactual_age_sensitivity.json",
        fig_path=DEFAULT_RESULTS_DIR / "figures" / "fig14_counterfactual_age",
    )
    # Also S3
    s3 = sdir.parent / "S3"
    if s3.exists():
        cf3 = {
            "temporal_only": root / "followup_S3_temporal_only_d20260922_m0_interonly",
            "age_temporal": root / "followup_S3_age_temporal_d20260922_m0_interonly",
        }
        for k, p in list(cf3.items()):
            if not (p / "model.pt").exists():
                cf3[k] = root / f"S3_{k}_d20260922_m0"
        run_counterfactual(
            scenario_dir=s3,
            run_dirs=cf3,
            device=device,
            scenario="S3",
            out_json=follow / "counterfactual_age_sensitivity_S3.json",
            fig_path=DEFAULT_RESULTS_DIR / "figures" / "fig14_counterfactual_age_S3",
        )


if __name__ == "__main__":
    main()
