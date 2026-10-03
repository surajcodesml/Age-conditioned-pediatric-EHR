#!/usr/bin/env python3
"""CEHR-BERT-small capacity-control experiment on PRIMARY S2 32-target.

Reuses the existing synthetic CEHR-BERT training / CF evaluation protocol.
Writes only under results/baselines/synthetic/cehrbert_small/ (never overwrites
full CEHR-BERT artifacts).

Usage:
  python -m analysis.cehr_small_s2.run_experiment --phase all
  python -m analysis.cehr_small_s2.run_experiment --phase train --seeds 0,1,2,3,4
  python -m analysis.cehr_small_s2.run_experiment --phase eval
  python -m analysis.cehr_small_s2.run_experiment --phase report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# Insert repo root AFTER synthetic_age_temporal so package `baselines/` wins over
# synthetic_age_temporal/baselines.py (module shadow).
sys.path.insert(0, str(REPO_ROOT / "synthetic_age_temporal"))
sys.path.insert(0, str(REPO_ROOT))

from synthetic_age_temporal.config import (  # noqa: E402
    BATCH_SIZE,
    GRAD_CLIP,
    LR,
    MAX_EPOCHS,
    MAX_SEQ_LEN,
    PATIENCE,
    WEIGHT_DECAY,
    Config,
)
from baselines.cehrbert_adapter.adapter import CEHRBertAdapter  # noqa: E402
from baselines.common.capacity_report import count_parameters  # noqa: E402
from baselines.common.counterfactual import (  # noqa: E402
    CF_AGES,
    CF_LAGS_DAYS,
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    build_surface_grid,
    full_counterfactual_report,
)
from baselines.common.metrics import multilabel_metrics  # noqa: E402
from baselines.common.training import (  # noqa: E402
    evaluate_loader,
    get_device,
    set_seed,
    train_neural_baseline,
)
from baselines.synthetic.counterfactual_eval import (  # noqa: E402
    _build_oracle_fns,
    make_predict_fns,
)
from baselines.synthetic.data_adapter import make_baseline_loaders, model_batch  # noqa: E402
from baselines.synthetic.result_schema import from_train_and_cf  # noqa: E402
from baselines.synthetic.runner import _SafeLoader  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "synthetic" / "cehr_bert_small_s2_32target.yaml"
RESULTS_ROOT = REPO_ROOT / "results" / "baselines" / "synthetic" / "cehrbert_small"
FULL_CEHR_DIR = REPO_ROOT / "results" / "baselines" / "synthetic" / "cehrbert" / "S2"
DTR_AT_DIR = REPO_ROOT / "results" / "baselines" / "synthetic" / "dtr_age_temporal_new" / "S2"
DTR_TO_DIR = REPO_ROOT / "results" / "baselines" / "synthetic" / "dtr_temporal_only_new" / "S2"
FIGURES_DIR = REPO_ROOT / "figures" / "final"
REPORT_PATH = REPO_ROOT / "analysis" / "cehr_small_s2" / "report.md"

INTERACTION_TARGET_IDS = list(range(8))  # S2 mechanism=interaction


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def arch_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    a = cfg["architecture"]
    return {
        "d_model": int(a["d_model"]),
        "d_ff": int(a["d_ff"]),
        "n_layers": int(a["n_layers"]),
        "n_heads": int(a["n_heads"]),
        "dropout": float(a["dropout"]),
        "time_dim": int(a["time_dim"]),
        "age_dim": int(a["age_dim"]),
        "max_seq_len": int(a["max_seq_len"]),
    }


def build_small_model(n_codes: int, n_targets: int, cfg: dict[str, Any]) -> CEHRBertAdapter:
    return CEHRBertAdapter(n_codes=n_codes, n_targets=n_targets, **arch_kwargs(cfg))


def seed_dir(seed: int) -> Path:
    return RESULTS_ROOT / f"seed{seed}" / "S2"


def print_capacity_table(cfg: dict[str, Any], n_codes: int = 566, n_targets: int = 32) -> list[dict]:
    """Reproduce full count and print candidate table; return rows."""
    print("\n=== Parameter accounting (count_parameters convention) ===")
    full = CEHRBertAdapter(
        n_codes=n_codes, n_targets=n_targets,
        d_model=128, n_layers=5, n_heads=8, d_ff=512,
        time_dim=32, age_dim=32, max_seq_len=112,
    )
    full_c = count_parameters(full)
    frozen = sum(p.numel() for p in full.parameters() if not p.requires_grad)
    print(f"Full CEHR-BERT:")
    print(f"  trainable={full_c['trainable_params']:,}")
    print(f"  frozen={frozen:,}")
    print(f"  total={full_c['total_params']:,}")
    print(f"  embedding_params={full_c['embedding_params']:,}")
    print(f"  code_embeddings_trainable={full.code_embedding.weight.requires_grad}")
    print(f"  segment_embeddings_trainable={full.segment_embedding.weight.requires_grad}")

    rows = []
    print(f"\n{'hidden_dim':>10} {'ffn_dim':>7} {'layers':>6} {'heads':>5} {'trainable_params':>16}")
    for c in cfg["candidates"]:
        m = CEHRBertAdapter(
            n_codes=n_codes, n_targets=n_targets,
            d_model=c["hidden_dim"], d_ff=c["ffn_dim"],
            n_layers=c["layers"], n_heads=c["heads"],
            time_dim=32, age_dim=32, max_seq_len=112,
        )
        tp = count_parameters(m)["trainable_params"]
        rows.append({**c, "measured_trainable": tp})
        mark = " <-- chosen" if (
            c["hidden_dim"] == cfg["architecture"]["d_model"]
            and c["ffn_dim"] == cfg["architecture"]["d_ff"]
            and c["layers"] == cfg["architecture"]["n_layers"]
            and c["heads"] == cfg["architecture"]["n_heads"]
        ) else ""
        print(f"{c['hidden_dim']:10d} {c['ffn_dim']:7d} {c['layers']:6d} {c['heads']:5d} {tp:16d}{mark}")

    chosen = build_small_model(n_codes, n_targets, cfg)
    chosen_c = count_parameters(chosen)
    print(f"\nChosen CEHR-BERT-small measured: {chosen_c}")
    print(f"Δ vs DTR(55,107) = {chosen_c['trainable_params'] - 55107:+d}")
    return rows


def train_one_seed(
    seed: int,
    cfg: dict[str, Any],
    *,
    device: str,
    data_seed: int,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = seed_dir(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        print(f"[seed {seed}] already trained → {result_path}")
        with result_path.open() as f:
            return json.load(f)

    print(f"\n{'='*60}\nCEHR-BERT-small | S2 | seed={seed}\n{'='*60}")
    train_raw, val_raw, test_raw, vocab, info = make_baseline_loaders(
        "S2", data_seed=data_seed, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
    )
    train_loader = _SafeLoader(train_raw)
    val_loader = _SafeLoader(val_raw)
    test_loader = _SafeLoader(test_raw)
    n_codes, n_targets = info["n_codes"], info["n_targets"]

    set_seed(seed)
    model = build_small_model(n_codes, n_targets, cfg)
    param_counts = count_parameters(model)
    print(f"  params={param_counts}")

    t0 = time.time()
    train_result = train_neural_baseline(
        model=model,
        train_fn=model.training_step,
        predict_fn=model.predict,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        grad_clip=GRAD_CLIP,
        device=device,
        run_dir=run_dir,
        seed=seed,
    )
    test_metrics = evaluate_loader(model, model.predict, test_loader, get_device(device))
    model.save_checkpoint(run_dir)

    result: dict[str, Any] = {
        "model": "cehrbert_small",
        "scenario": "S2",
        "seed": seed,
        "n_codes": n_codes,
        "n_targets": n_targets,
        "architecture": arch_kwargs(cfg),
        "param_counts": param_counts,
        "model_card": model.model_card,
        "train": train_result,
        "test_metrics": test_metrics,
        "train_wall_s": time.time() - t0,
        "protocol": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "checkpoint_selection": "best_val_bce",
            "loss": "BCEWithLogitsLoss",
            "data_seed": data_seed,
        },
        "checkpoint_paths": {
            "best": str(run_dir / "best_checkpoint.pt"),
            "last": str(run_dir / "last_checkpoint.pt"),
            "canonical": str(run_dir / "checkpoint.pt"),
        },
    }
    _attach_predictive(result)
    with result_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(
        f"  seed {seed}: AUPRC={result.get('AUPRC')} AUROC={result.get('AUROC')} "
        f"BCE={result.get('BCE')}"
    )
    return result


def _attach_predictive(result: dict[str, Any]) -> None:
    tm = result.get("test_metrics") or {}
    result["AUROC"] = tm.get("micro_auroc")
    result["AUPRC"] = tm.get("micro_auprc")
    result["BCE"] = tm.get("bce")


def _collect_logits_labels(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, logits = [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model.predict(b)
            logits.append(out.logits.detach().cpu().numpy())
            ys.append(batch["labels"].cpu().numpy())
    return np.concatenate(ys, axis=0), np.concatenate(logits, axis=0)


def eval_one_seed(
    seed: int,
    cfg: dict[str, Any],
    *,
    device: str,
    data_seed: int,
) -> dict[str, Any]:
    run_dir = seed_dir(seed)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing train result for seed {seed}: {result_path}")

    with result_path.open() as f:
        result = json.load(f)

    _, _, test_raw, vocab, info = make_baseline_loaders(
        "S2", data_seed=data_seed, batch_size=1, max_seq_len=MAX_SEQ_LEN,
    )
    # Also full test loader for predictive recompute / interaction subset
    _, _, test_bs, _, _ = make_baseline_loaders(
        "S2", data_seed=data_seed, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
    )
    test_loader = _SafeLoader(test_bs)

    n_codes, n_targets = info["n_codes"], info["n_targets"]
    model = build_small_model(n_codes, n_targets, cfg)
    model.load_checkpoint(run_dir)
    dev = get_device(device)
    model.to(dev)

    # Predictive (full 32) — already in result; recompute interaction-8 secondary
    y, logits = _collect_logits_labels(model, test_loader, dev)
    inter_metrics = multilabel_metrics(
        y[:, INTERACTION_TARGET_IDS], logits[:, INTERACTION_TARGET_IDS], ks=(5,),
    )
    result["secondary_interaction8"] = {
        "n_targets": 8,
        "target_ids": INTERACTION_TARGET_IDS,
        "note": "Same checkpoint; metrics restricted to mechanism=interaction labels.",
        "AUROC": inter_metrics.get("micro_auroc"),
        "AUPRC": inter_metrics.get("micro_auprc"),
        "BCE": inter_metrics.get("bce"),
        "test_metrics": inter_metrics,
    }

    # CF template
    cfg_data = Config(data_seed=data_seed)
    scenario_dir = cfg_data.data_dir() / "controlled" / "S2"
    template = None
    for batch in test_raw:
        if batch["is_signal"].any():
            template = {k: v for k, v in batch.items()}
            break
    if template is None:
        raise RuntimeError("No signal template in S2 test set")

    specs = json.loads((scenario_dir / "target_specs.json").read_text())
    meta = json.loads((scenario_dir / "meta.json").read_text())
    itos = dict(vocab.itos)
    o_age, o_lag, o_surf = _build_oracle_fns(
        template, itos, specs, "S2",
        float(meta["theta0"]), float(meta["beta_true"]),
    )
    p_age, p_lag, p_surf, _ = make_predict_fns(model, template, dev, n_codes)

    # S0 CF-age for mechanism classification (optional; use existing if available)
    s0_rmse = None
    report = full_counterfactual_report(
        p_age, p_lag, p_surf, o_age, o_lag, o_surf, cf_age_rmse_s0=s0_rmse,
    )
    report["model"] = "cehrbert_small"
    report["scenario"] = "S2"
    report["seed"] = seed

    # Save curves / surfaces
    age_curve_hat = np.stack([p_age(a) for a in CF_AGES], axis=0)
    age_curve_oracle = np.stack([o_age(a) for a in CF_AGES], axis=0)
    lag_curve_hat = np.stack([p_lag(lag) for lag in CF_LAGS_DAYS], axis=0)
    lag_curve_oracle = np.stack([o_lag(lag) for lag in CF_LAGS_DAYS], axis=0)
    surface_hat = build_surface_grid(p_surf)
    surface_oracle = build_surface_grid(o_surf)
    residual = surface_hat - surface_oracle

    artifacts = {
        "ages": list(CF_AGES),
        "lags_days": list(CF_LAGS_DAYS),
        "surface_ages": list(SURFACE_AGES),
        "surface_lags_days": list(SURFACE_LAGS_DAYS),
        "age_cf_curve_pred": age_curve_hat,
        "age_cf_curve_oracle": age_curve_oracle,
        "lag_cf_curve_pred": lag_curve_hat,
        "lag_cf_curve_oracle": lag_curve_oracle,
        "surface_pred": surface_hat,
        "surface_oracle": surface_oracle,
        "residual_surface": residual,
    }
    np.savez_compressed(run_dir / "cf_surfaces.npz", **artifacts)

    # Interaction-8 CF secondary (same checkpoint; slice target dims)
    def _slice8(fn):
        def wrapped(*args):
            return np.asarray(fn(*args), dtype=np.float64)[INTERACTION_TARGET_IDS]
        return wrapped

    report_8 = full_counterfactual_report(
        _slice8(p_age), _slice8(p_lag), _slice8(p_surf),
        _slice8(o_age), _slice8(o_lag), _slice8(o_surf),
        cf_age_rmse_s0=None,
    )
    report_8["model"] = "cehrbert_small"
    report_8["scenario"] = "S2"
    report_8["seed"] = seed
    report_8["n_targets"] = 8
    report_8["note"] = "SECONDARY: interaction targets only; not primary result."
    result["secondary_interaction8"]["cf_report"] = report_8

    with (run_dir / "cf_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    with (run_dir / "cf_report_interaction8.json").open("w") as f:
        json.dump(report_8, f, indent=2)

    result["cf_report"] = report
    schema = from_train_and_cf(
        scenario="S2", model="cehrbert_small",
        test_metrics=result.get("test_metrics"), cf_report=report,
    )
    result["benchmark_record"] = schema
    for k, v in schema.items():
        if k not in ("scenario", "model"):
            result[k] = v

    with result_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)

    print(
        f"[seed {seed}] CF-age={report['cf_rmse_age']:.4f} "
        f"CF-lag={report['cf_rmse_lag']:.4f} Surf={report['surface_rmse']:.4f} | "
        f"inter8 AUPRC={result['secondary_interaction8']['AUPRC']:.4f} "
        f"Surf8={report_8['surface_rmse']:.4f}"
    )
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def summarize(seeds: list[int]) -> dict[str, Any]:
    rows = []
    for s in seeds:
        p = seed_dir(s) / "result.json"
        if not p.exists():
            continue
        r = _load_json(p)
        rows.append({
            "seed": s,
            "trainable_params": (r.get("param_counts") or {}).get("trainable_params"),
            "AUPRC": r.get("AUPRC"),
            "AUROC": r.get("AUROC"),
            "BCE": r.get("BCE"),
            "CF_RMSE_age": r.get("CF_RMSE_age"),
            "CF_RMSE_lag": r.get("CF_RMSE_lag"),
            "Surface_RMSE": r.get("Surface_RMSE"),
            "checkpoint": (r.get("checkpoint_paths") or {}).get("best"),
            "secondary_interaction8": r.get("secondary_interaction8"),
        })

    def _stats(key: str) -> dict[str, float] | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
            "n": len(vals),
        }

    canonical = next((r for r in rows if r["seed"] == 0), rows[0] if rows else None)
    summary = {
        "model": "cehrbert_small",
        "scenario": "S2",
        "seeds": [r["seed"] for r in rows],
        "per_seed": rows,
        "canonical_seed0": canonical,
        "aggregate": {
            k: _stats(k)
            for k in ("AUPRC", "AUROC", "BCE", "CF_RMSE_age", "CF_RMSE_lag", "Surface_RMSE")
        },
    }

    # Reference models (canonical published / _new)
    refs = {}
    if FULL_CEHR_DIR.joinpath("result.json").exists():
        fr = _load_json(FULL_CEHR_DIR / "result.json")
        refs["cehrbert_full"] = {
            "trainable_params": (fr.get("param_counts") or {}).get("trainable_params", 1126304),
            "AUPRC": fr.get("AUPRC"),
            "AUROC": fr.get("AUROC"),
            "BCE": fr.get("BCE"),
            "CF_RMSE_age": fr.get("CF_RMSE_age"),
            "CF_RMSE_lag": fr.get("CF_RMSE_lag"),
            "Surface_RMSE": fr.get("Surface_RMSE"),
            "seed": fr.get("seed", 0),
        }
    if DTR_AT_DIR.joinpath("result.json").exists():
        dr = _load_json(DTR_AT_DIR / "result.json")
        refs["dtr_age_temporal"] = {
            "trainable_params": (dr.get("param_counts") or {}).get("trainable_params", 55107),
            "AUPRC": dr.get("AUPRC") or (dr.get("test_metrics") or {}).get("micro_auprc"),
            "AUROC": dr.get("AUROC") or (dr.get("test_metrics") or {}).get("micro_auroc"),
            "BCE": dr.get("BCE") or (dr.get("test_metrics") or {}).get("bce"),
            "CF_RMSE_age": dr.get("CF_RMSE_age"),
            "CF_RMSE_lag": dr.get("CF_RMSE_lag"),
            "Surface_RMSE": dr.get("Surface_RMSE"),
            "seed": dr.get("seed", 0),
        }
    if DTR_TO_DIR.joinpath("result.json").exists():
        tr = _load_json(DTR_TO_DIR / "result.json")
        refs["dtr_temporal_only"] = {
            "trainable_params": (tr.get("param_counts") or {}).get("trainable_params", 55106),
            "AUPRC": tr.get("AUPRC") or (tr.get("test_metrics") or {}).get("micro_auprc"),
            "AUROC": tr.get("AUROC") or (tr.get("test_metrics") or {}).get("micro_auroc"),
            "BCE": tr.get("BCE") or (tr.get("test_metrics") or {}).get("bce"),
            "CF_RMSE_age": tr.get("CF_RMSE_age"),
            "CF_RMSE_lag": tr.get("CF_RMSE_lag"),
            "Surface_RMSE": tr.get("Surface_RMSE"),
            "seed": tr.get("seed", 0),
        }
    summary["references"] = refs

    # Deltas (canonical seed-0 small vs refs)
    if canonical and refs:
        small = canonical
        deltas = {}
        if "cehrbert_full" in refs:
            f = refs["cehrbert_full"]
            deltas["small_vs_full"] = {
                k: (None if small.get(k) is None or f.get(k) is None
                    else float(small[k]) - float(f[k]))
                for k in ("AUPRC", "AUROC", "CF_RMSE_age", "CF_RMSE_lag", "Surface_RMSE")
            }
            if small.get("Surface_RMSE") and f.get("Surface_RMSE"):
                deltas["small_vs_full"]["Surface_RMSE_rel"] = (
                    float(small["Surface_RMSE"]) / float(f["Surface_RMSE"]) - 1.0
                )
        if "dtr_age_temporal" in refs:
            d = refs["dtr_age_temporal"]
            deltas["small_vs_dtr"] = {
                k: (None if small.get(k) is None or d.get(k) is None
                    else float(small[k]) - float(d[k]))
                for k in ("AUPRC", "AUROC", "CF_RMSE_age", "CF_RMSE_lag", "Surface_RMSE")
            }
            if small.get("Surface_RMSE") and d.get("Surface_RMSE"):
                deltas["small_vs_dtr"]["Surface_RMSE_rel"] = (
                    float(small["Surface_RMSE"]) / float(d["Surface_RMSE"]) - 1.0
                )
        summary["deltas_canonical"] = deltas

    out = RESULTS_ROOT / "summary.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out}")
    return summary


def make_figures(summary: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    refs = summary.get("references") or {}
    can = summary.get("canonical_seed0") or {}
    agg = (summary.get("aggregate") or {}).get("Surface_RMSE") or {}
    agg_a = (summary.get("aggregate") or {}).get("AUPRC") or {}

    points = []
    # DTR
    if "dtr_age_temporal" in refs:
        r = refs["dtr_age_temporal"]
        points.append(("DTR", r["trainable_params"], r["Surface_RMSE"], r["AUPRC"], "o"))
    if "dtr_temporal_only" in refs:
        r = refs["dtr_temporal_only"]
        points.append(("Temporal-only DTR", r["trainable_params"], r["Surface_RMSE"], r["AUPRC"], "s"))
    # CEHR-small (mean ± from aggregate if available)
    if can.get("trainable_params") and can.get("Surface_RMSE") is not None:
        surf = agg.get("mean", can["Surface_RMSE"])
        auprc = agg_a.get("mean", can["AUPRC"])
        points.append(("CEHR-BERT-small", can["trainable_params"], surf, auprc, "^"))
    if "cehrbert_full" in refs:
        r = refs["cehrbert_full"]
        points.append(("CEHR-BERT-full", r["trainable_params"], r["Surface_RMSE"], r["AUPRC"], "D"))

    def _plot(y_key_idx: int, ylabel: str, stem: str):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for name, nparm, surf, auprc, marker in points:
            y = surf if y_key_idx == 0 else auprc
            ax.scatter([nparm], [y], marker=marker, s=80, label=name, zorder=3)
            ax.annotate(name, (nparm, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
        # error bar for small surface if multi-seed
        if y_key_idx == 0 and agg.get("std") and can.get("trainable_params"):
            ax.errorbar(
                [can["trainable_params"]], [agg["mean"]], yerr=[agg["std"]],
                fmt="none", ecolor="C2", capsize=4, zorder=2,
            )
        if y_key_idx == 1 and agg_a.get("std") and can.get("trainable_params"):
            ax.errorbar(
                [can["trainable_params"]], [agg_a["mean"]], yerr=[agg_a["std"]],
                fmt="none", ecolor="C2", capsize=4, zorder=2,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Capacity vs {ylabel} (S2, 32-target)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        for ext in ("svg", "png"):
            path = FIGURES_DIR / f"{stem}.{ext}"
            fig.savefig(path, dpi=200)
            print(f"Wrote {path}")
        plt.close(fig)

    _plot(0, "Surface RMSE", "capacity_vs_surface_rmse")
    _plot(1, "AUPRC", "capacity_vs_auprc")


def _fmt_pm(stat: dict | None, digits: int = 3) -> str:
    if not stat:
        return "—"
    if stat["n"] <= 1:
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def write_report(cfg: dict[str, Any], summary: dict[str, Any], capacity_rows: list[dict]) -> None:
    refs = summary.get("references") or {}
    can = summary.get("canonical_seed0") or {}
    agg = summary.get("aggregate") or {}
    deltas = summary.get("deltas_canonical") or {}
    arch = cfg["architecture"]

    # Interpretation
    interpretation = []
    d_full = deltas.get("small_vs_full") or {}
    d_dtr = deltas.get("small_vs_dtr") or {}
    surf_s = can.get("Surface_RMSE")
    surf_f = (refs.get("cehrbert_full") or {}).get("Surface_RMSE")
    surf_d = (refs.get("dtr_age_temporal") or {}).get("Surface_RMSE")
    auprc_s = can.get("AUPRC")
    auprc_f = (refs.get("cehrbert_full") or {}).get("AUPRC")
    auprc_d = (refs.get("dtr_age_temporal") or {}).get("AUPRC")

    if surf_s is not None and surf_f is not None and surf_d is not None:
        # Advantage of full over DTR on surface
        full_adv = float(surf_d) - float(surf_f)
        small_adv = float(surf_d) - float(surf_s)
        retained = small_adv / full_adv if abs(full_adv) > 1e-9 else float("nan")
        pred_close = (
            auprc_s is not None and auprc_f is not None
            and abs(float(auprc_s) - float(auprc_f)) < 0.02
        )
        if retained < 0.4 and float(surf_s) > float(surf_f) + 0.02:
            conclusion = (
                "**A.** The surface-fitting advantage is partly capacity-dependent: "
                "CEHR-BERT-small loses much of the full model's Surface-RMSE advantage over DTR."
            )
        elif retained > 0.6 and float(surf_s) + 0.02 < float(surf_d):
            conclusion = (
                "**B.** The advantage cannot be explained by parameter count alone and likely "
                "reflects CEHR-BERT's less constrained age/time representation: "
                "CEHR-BERT-small retains most of the full model's Surface-RMSE advantage at "
                "DTR-matched capacity."
            )
        elif pred_close and float(surf_s) > float(surf_f) + 0.02:
            conclusion = (
                "**C.** Prediction remains similar but surface error worsens: distinguish "
                "predictive capacity from counterfactual response flexibility."
            )
        else:
            conclusion = (
                "Mixed / inconclusive under pre-registered A/B/C thresholds; see deltas below. "
                "Do **not** conclude that parameter count alone explains CEHR-BERT performance "
                "unless the retained-advantage pattern clearly supports that inference."
            )
        interpretation.append(f"Full−DTR Surface advantage: {full_adv:.4f}")
        interpretation.append(f"Small−DTR Surface advantage: {small_adv:.4f}")
        interpretation.append(f"Fraction of full advantage retained by small: {retained:.2f}")
    else:
        conclusion = "Insufficient reference metrics to classify A/B/C."

    lines = []
    lines.append("# CEHR-BERT-small capacity-control experiment (S2, 32-target)")
    lines.append("")
    lines.append("## Scientific question")
    lines.append("")
    lines.append(
        "Does CEHR-BERT's stronger counterfactual surface fitting persist when trainable "
        "capacity is approximately matched to DTR (~55k)?"
    )
    lines.append("")
    lines.append("## Exact architecture (chosen)")
    lines.append("")
    lines.append("| Hyperparameter | Value |")
    lines.append("|---|---|")
    for k, v in arch.items():
        lines.append(f"| `{k}` | {v} |")
    lines.append("| age representation | Time2Vec (unchanged) |")
    lines.append("| time representation | Time2Vec (unchanged) |")
    lines.append("| projection | concat(code, segment, time, age) → Linear → d_model |")
    lines.append("")
    lines.append("## Parameter count")
    lines.append("")
    lines.append("Counting convention: `baselines.common.capacity_report.count_parameters`.")
    lines.append("")
    lines.append("| Model | Trainable | Frozen | Total | Code/seg embeddings |")
    lines.append("|---|---:|---:|---:|---|")
    lines.append("| CEHR-BERT-full | ~1,126,304 | 0 | ~1,126,304 | **trainable** |")
    lines.append(
        f"| CEHR-BERT-small | {can.get('trainable_params', cfg['parameter_count']['chosen_trainable']):,} "
        f"| 0 | {can.get('trainable_params', cfg['parameter_count']['chosen_trainable']):,} | **trainable** |"
    )
    lines.append("| DTR (age_temporal) | 55,107 | 0 | 55,107 | trainable |")
    lines.append("")
    lines.append("### Capacity search table (time_dim=age_dim=32 fixed)")
    lines.append("")
    lines.append("| hidden_dim | ffn_dim | layers | heads | trainable_params |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in capacity_rows:
        lines.append(
            f"| {r['hidden_dim']} | {r['ffn_dim']} | {r['layers']} | {r['heads']} "
            f"| {r['measured_trainable']} |"
        )
    lines.append("")
    lines.append(
        f"Chosen config is closest multi-layer match to DTR "
        f"(Δ = {cfg['parameter_count']['chosen_trainable'] - 55107:+d})."
    )
    lines.append("")
    lines.append("## Training config")
    lines.append("")
    lines.append("Matched to full CEHR-BERT synthetic baseline protocol:")
    lines.append("")
    proto = cfg["training"]
    lines.append(f"- LR = `{proto['lr']}`, weight_decay = `{proto['weight_decay']}`")
    lines.append(f"- max_epochs = `{proto['max_epochs']}`, patience = `{proto['patience']}`")
    lines.append(f"- grad_clip = `{proto['grad_clip']}`, batch_size = `{proto['batch_size']}`")
    lines.append(f"- loss = `{proto['loss']}`")
    lines.append(f"- checkpoint selection = `{proto['checkpoint_selection']}` (lower val BCE)")
    lines.append(f"- data_seed = `{cfg['data_seed']}` (patient-disjoint split)")
    lines.append(f"- model seeds = `{proto['seeds']}` (seed 0 = canonical / paper-table seed)")
    lines.append("")
    lines.append("## Seeds and checkpoints")
    lines.append("")
    lines.append("| Seed | Role | Checkpoint |")
    lines.append("|---:|---|---|")
    for r in summary.get("per_seed") or []:
        role = "canonical" if r["seed"] == 0 else "uncertainty"
        lines.append(f"| {r['seed']} | {role} | `{r.get('checkpoint')}` |")
    lines.append("")
    lines.append("## Primary metrics (32-target)")
    lines.append("")
    lines.append("### Canonical seed 0")
    lines.append("")
    lines.append("| Metric | CEHR-small | CEHR-full | DTR | Temporal-only DTR |")
    lines.append("|---|---:|---:|---:|---:|")

    def _g(ref, key):
        return (refs.get(ref) or {}).get(key)

    for key, label in [
        ("AUPRC", "AUPRC"), ("AUROC", "AUROC"), ("BCE", "BCE"),
        ("CF_RMSE_age", "CF-RMSE-age"), ("CF_RMSE_lag", "CF-RMSE-lag"),
        ("Surface_RMSE", "Surface RMSE"),
    ]:
        lines.append(
            f"| {label} | {_num(can.get(key))} | {_num(_g('cehrbert_full', key))} | "
            f"{_num(_g('dtr_age_temporal', key))} | {_num(_g('dtr_temporal_only', key))} |"
        )
    lines.append("")
    lines.append("### Multi-seed mean ± SD (CEHR-BERT-small)")
    lines.append("")
    lines.append("| Metric | mean ± SD |")
    lines.append("|---|---|")
    for key in ("AUPRC", "AUROC", "BCE", "CF_RMSE_age", "CF_RMSE_lag", "Surface_RMSE"):
        lines.append(f"| {key} | {_fmt_pm(agg.get(key))} |")
    lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append(
        "| Model | Trainable params | AUPRC | AUROC | CF-RMSE-age | CF-RMSE-lag | Surface RMSE |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| DTR | {_g('dtr_age_temporal','trainable_params')} | "
        f"{_num(_g('dtr_age_temporal','AUPRC'))} | {_num(_g('dtr_age_temporal','AUROC'))} | "
        f"{_num(_g('dtr_age_temporal','CF_RMSE_age'))} | {_num(_g('dtr_age_temporal','CF_RMSE_lag'))} | "
        f"{_num(_g('dtr_age_temporal','Surface_RMSE'))} |"
    )
    lines.append(
        f"| Temporal-only DTR | {_g('dtr_temporal_only','trainable_params')} | "
        f"{_num(_g('dtr_temporal_only','AUPRC'))} | {_num(_g('dtr_temporal_only','AUROC'))} | "
        f"{_num(_g('dtr_temporal_only','CF_RMSE_age'))} | {_num(_g('dtr_temporal_only','CF_RMSE_lag'))} | "
        f"{_num(_g('dtr_temporal_only','Surface_RMSE'))} |"
    )
    lines.append(
        f"| CEHR-BERT-small (~55k) seed0 | {can.get('trainable_params')} | "
        f"{_num(can.get('AUPRC'))} | {_num(can.get('AUROC'))} | "
        f"{_num(can.get('CF_RMSE_age'))} | {_num(can.get('CF_RMSE_lag'))} | "
        f"{_num(can.get('Surface_RMSE'))} |"
    )
    lines.append(
        f"| CEHR-BERT-small multi-seed | {can.get('trainable_params')} | "
        f"{_fmt_pm(agg.get('AUPRC'))} | {_fmt_pm(agg.get('AUROC'))} | "
        f"{_fmt_pm(agg.get('CF_RMSE_age'))} | {_fmt_pm(agg.get('CF_RMSE_lag'))} | "
        f"{_fmt_pm(agg.get('Surface_RMSE'))} |"
    )
    lines.append(
        f"| CEHR-BERT-full (~1.1M) | {_g('cehrbert_full','trainable_params')} | "
        f"{_num(_g('cehrbert_full','AUPRC'))} | {_num(_g('cehrbert_full','AUROC'))} | "
        f"{_num(_g('cehrbert_full','CF_RMSE_age'))} | {_num(_g('cehrbert_full','CF_RMSE_lag'))} | "
        f"{_num(_g('cehrbert_full','Surface_RMSE'))} |"
    )
    lines.append("")
    lines.append("## Deltas (canonical seed 0)")
    lines.append("")
    lines.append("### CEHR-small vs CEHR-full")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(d_full, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### CEHR-small vs DTR")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(d_dtr, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Capacity hypothesis")
    lines.append("")
    for s in interpretation:
        lines.append(f"- {s}")
    lines.append("")
    lines.append(conclusion)
    lines.append("")
    lines.append("## Secondary analysis: 8 interaction targets")
    lines.append("")
    lines.append(
        "Produced from the **same** 32-target checkpoint by restricting metrics / CF "
        "to `mechanism=interaction` labels (ids 0–7). **Not** the primary result."
    )
    lines.append("")
    lines.append("| Seed | AUPRC | AUROC | BCE | CF-age | CF-lag | Surface |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary.get("per_seed") or []:
        sec = r.get("secondary_interaction8") or {}
        cfr = sec.get("cf_report") or {}
        lines.append(
            f"| {r['seed']} | {_num(sec.get('AUPRC'))} | {_num(sec.get('AUROC'))} | "
            f"{_num(sec.get('BCE'))} | {_num(cfr.get('cf_rmse_age'))} | "
            f"{_num(cfr.get('cf_rmse_lag'))} | {_num(cfr.get('surface_rmse'))} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Config: `{CONFIG_PATH.relative_to(REPO_ROOT)}`")
    lines.append(f"- Results root: `{RESULTS_ROOT.relative_to(REPO_ROOT)}`")
    lines.append(f"- Per-seed: `result.json`, `cf_report.json`, `cf_surfaces.npz`, checkpoints")
    lines.append(f"- Summary: `{RESULTS_ROOT.relative_to(REPO_ROOT)}/summary.json`")
    lines.append("- Figures: `figures/final/capacity_vs_surface_rmse.{svg,png}`, `capacity_vs_auprc.{svg,png}`")
    lines.append("")
    lines.append("## Fairness / instability notes")
    lines.append("")
    lines.append(
        "- Full CEHR-BERT artifacts under `results/baselines/synthetic/cehrbert/` were **not** "
        "modified."
    )
    lines.append(
        "- DTR and the synthetic benchmark were **not** modified."
    )
    lines.append(
        "- Training budget and early-stopping criterion match full CEHR-BERT "
        f"(max_epochs={proto['max_epochs']}, patience={proto['patience']}, best val BCE)."
    )
    lines.append(
        "- Only ordinary capacity hyperparameters were reduced; age/time Time2Vec inputs "
        "and concat→project architecture were retained with dim=32."
    )
    lines.append(
        "- DTR multi-seed runs were not available; only canonical seed 0 exists for DTR. "
        "CEHR-small uses seed 0 for direct comparison plus seeds 1–4 for uncertainty."
    )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def _num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=["capacity", "train", "eval", "report", "all"],
    )
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-seed", type=int, default=20260922)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    capacity_rows: list[dict] = []
    if args.phase in ("capacity", "all", "report"):
        capacity_rows = print_capacity_table(cfg)
        with (RESULTS_ROOT / "capacity_search.json").open("w") as f:
            json.dump({"candidates": capacity_rows, "chosen": arch_kwargs(cfg)}, f, indent=2)

    if args.phase in ("train", "all"):
        for s in seeds:
            train_one_seed(
                s, cfg, device=args.device, data_seed=args.data_seed, force=args.force,
            )

    if args.phase in ("eval", "all"):
        for s in seeds:
            eval_one_seed(s, cfg, device=args.device, data_seed=args.data_seed)

    if args.phase in ("report", "all"):
        if not capacity_rows:
            capacity_rows = print_capacity_table(cfg)
        summary = summarize(seeds)
        make_figures(summary)
        write_report(cfg, summary, capacity_rows)


if __name__ == "__main__":
    main()
