#!/usr/bin/env python3
"""Build ICLR final-results artifacts for Developmental Temporal Retrieval.

READ-ONLY w.r.t. training / data generation. Writes only under
``analysis/final_results/`` and ``figures/final/``.

May run evaluation forward passes from existing checkpoints.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
# Last insert wins search order: keep repo root ahead of synthetic_age_temporal
# so the ``baselines/`` package is not shadowed by ``baselines.py``.
sys.path.insert(0, str(REPO / "synthetic_age_temporal"))
sys.path.insert(0, str(REPO))

from baselines.common.counterfactual import (  # noqa: E402
    SURFACE_AGES,
    SURFACE_LAGS_DAYS,
    build_surface_grid,
    cf_rmse_age,
    surface_rmse,
)
from baselines.common.training import get_device  # noqa: E402
from baselines.synthetic.counterfactual_eval import (  # noqa: E402
    _build_oracle_fns,
    make_predict_fns,
)
from baselines.synthetic.data_adapter import make_dtr_baseline_loaders  # noqa: E402
from baselines.synthetic.runner import build_model, dtr_arm_dirname  # noqa: E402
from synthetic_age_temporal.config import Config  # noqa: E402
from synthetic_age_temporal.dataset import make_loaders  # noqa: E402
from train_dtr import ablations, predict  # noqa: E402

OUT = REPO / "analysis" / "final_results"
FIG = REPO / "figures" / "final"
SYN = REPO / "results" / "baselines" / "synthetic"
DATA_SEED = 20260922
MODEL_SEED = 0
SCENARIOS = ("S0", "S1", "S2", "S3", "S5")
ARMS = ("age_temporal", "temporal_only")

# Paper primary S2 32-target table (rounded) for provenance checks.
PRIMARY_S2 = {
    "DTR": {
        "AUPRC": 0.577,
        "AUROC": 0.761,
        "CF_RMSE_age": 0.142,
        "CF_RMSE_lag": 0.166,
        "Surface_RMSE": 0.201,
        "result_key": "dtr_age_temporal_new",
    },
    "Temporal-only DTR": {
        "AUPRC": 0.569,
        "AUROC": 0.758,
        "CF_RMSE_age": 0.177,
        "CF_RMSE_lag": 0.184,
        "Surface_RMSE": 0.236,
        "result_key": "dtr_temporal_only_new",
    },
    "Count+LightGBM": {
        "AUPRC": 0.520,
        "AUROC": 0.729,
        "CF_RMSE_age": 0.193,
        "CF_RMSE_lag": 0.220,
        "Surface_RMSE": 0.207,
        "result_key": "count_lightgbm",
    },
    "RETAIN": {
        "AUPRC": 0.435,
        "AUROC": 0.675,
        "CF_RMSE_age": 0.223,
        "CF_RMSE_lag": 0.193,
        "Surface_RMSE": 0.236,
        "result_key": "retain",
    },
    "BEHRT": {
        "AUPRC": 0.508,
        "AUROC": 0.723,
        "CF_RMSE_age": 0.189,
        "CF_RMSE_lag": 0.186,
        "Surface_RMSE": 0.217,
        "result_key": "behrt",
    },
    "CEHR-BERT": {
        "AUPRC": 0.582,
        "AUROC": 0.764,
        "CF_RMSE_age": 0.098,
        "CF_RMSE_lag": 0.111,
        "Surface_RMSE": 0.131,
        "result_key": "cehrbert",
    },
}

DISPLAY = {
    "dtr_age_temporal_new": "DTR",
    "dtr_temporal_only_new": "Temporal-only DTR",
    "cehrbert": "CEHR-BERT",
    "behrt": "BEHRT",
    "retain": "RETAIN",
    "count_lightgbm": "Count+LightGBM",
}


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -40.0, 40.0)))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / stem}.{{png,svg}}")


def load_result(arm_dir: Path) -> dict[str, Any]:
    return json.loads((arm_dir / "result.json").read_text())


def data_meta(scenario: str) -> dict[str, Any]:
    d = Config(data_seed=DATA_SEED).data_dir() / "controlled" / scenario
    meta = json.loads((d / "meta.json").read_text())
    splits = json.loads((d / "splits.json").read_text())
    specs = json.loads((d / "target_specs.json").read_text())
    meta["_path"] = str(d)
    meta["_split_counts"] = {k: len(v) for k, v in splits.items()}
    meta["_mechanism_counts"] = dict(Counter(s["mechanism"] for s in specs))
    meta["_interaction_idx"] = [
        i for i, s in enumerate(specs) if s["mechanism"] == "interaction"
    ]
    meta["_age_only_idx"] = [i for i, s in enumerate(specs) if s["mechanism"] == "age_only"]
    meta["_specs"] = specs
    return meta


def round_match(got: float | None, expected: float, places: int = 3) -> bool:
    if got is None or (isinstance(got, float) and math.isnan(got)):
        return False
    return round(float(got), places) == expected


# ---------------------------------------------------------------------------
# 1. Provenance
# ---------------------------------------------------------------------------
def write_provenance() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    discrepancies: list[str] = []

    for scenario in SCENARIOS:
        meta = data_meta(scenario)
        for arm in ARMS:
            name = dtr_arm_dirname(arm, "_new")
            arm_dir = SYN / name / scenario
            r = load_result(arm_dir)
            mc = r.get("model_card") or {}
            row = {
                "scenario": scenario,
                "arm": name,
                "architecture": r.get("architecture") or mc.get("architecture"),
                "checkpoint": str(arm_dir / "best_checkpoint.pt"),
                "canonical_checkpoint": str(arm_dir / "checkpoint.pt"),
                "result_json": str(arm_dir / "result.json"),
                "cf_report": str(arm_dir / "cf_report.json"),
                "history_json": str(arm_dir / "history.json"),
                "config_path": (
                    str(arm_dir / "config.json")
                    if (arm_dir / "config.json").exists()
                    else "NONE (hyperparams embedded in result.json / runner defaults; "
                    "no separate config.json for synthetic _new runs)"
                ),
                "model_seed": r.get("seed", MODEL_SEED),
                "data_seed": DATA_SEED,
                "data_dir": meta["_path"],
                "n_targets": meta["n_targets"],
                "mechanism_counts": meta["_mechanism_counts"],
                "n_examples": meta["n_examples"],
                "split_counts": meta["_split_counts"],
                "beta_true": meta["beta_true"],
                "theta0_true": meta["theta0"],
                "trainable_params": mc.get("trainable_params"),
                "embedding_params": mc.get("embedding_params"),
                "hidden_size": mc.get("hidden_size"),
                "AUPRC": r.get("AUPRC"),
                "AUROC": r.get("AUROC"),
                "BCE": r.get("BCE"),
                "beta_hat": r.get("beta_hat"),
                "theta0_hat": r.get("theta0_hat"),
                "delta_BCE_beta0": r.get("delta_BCE_beta0"),
                "delta_BCE_age_shuffle": r.get("delta_BCE_age_shuffle"),
                "CF_RMSE_age": r.get("CF_RMSE_age"),
                "CF_RMSE_lag": r.get("CF_RMSE_lag"),
                "Surface_RMSE": r.get("Surface_RMSE"),
                "lambda_corr": r.get("lambda_corr"),
                "param_Surface_RMSE": r.get("param_Surface_RMSE"),
                "mechanism_classification": r.get("mechanism_classification"),
                "S5_Surface_RMSE_acute": r.get("S5_Surface_RMSE_acute"),
                "S5_Surface_RMSE_intermediate": r.get("S5_Surface_RMSE_intermediate"),
                "S5_Surface_RMSE_chronic": r.get("S5_Surface_RMSE_chronic"),
                "S5_Surface_RMSE_mean": r.get("S5_Surface_RMSE_mean"),
                "persistence_order_correct": r.get("persistence_order_correct"),
            }
            rows.append(row)

    # Primary table verification
    table_checks = []
    for label, exp in PRIMARY_S2.items():
        key = exp["result_key"]
        r = load_result(SYN / key / "S2")
        got = {
            "AUPRC": r.get("AUPRC"),
            "AUROC": r.get("AUROC"),
            "CF_RMSE_age": r.get("CF_RMSE_age"),
            "CF_RMSE_lag": r.get("CF_RMSE_lag"),
            "Surface_RMSE": r.get("Surface_RMSE"),
        }
        ok = {
            m: round_match(got[m], exp[m], 3)
            for m in ("AUPRC", "AUROC", "CF_RMSE_age", "CF_RMSE_lag", "Surface_RMSE")
        }
        table_checks.append({"model": label, "expected": exp, "got": got, "match_3dp": ok})
        for m, matched in ok.items():
            if not matched:
                discrepancies.append(
                    f"S2 primary table mismatch for {label}.{m}: "
                    f"table={exp[m]} artifact={got[m]}"
                )

    # Known artifact inconsistencies (do not silently reconcile)
    all_new = json.loads((SYN / "all_results_new.json").read_text())
    if all(v.get("AUPRC") is None for v in all_new.values()):
        discrepancies.append(
            "results/baselines/synthetic/all_results_new.json has null predictive/"
            "CF metrics for all scenarios; authoritative numbers live in per-arm "
            "result.json / cf_report.json / dtr_new_mechanism_summary.json / "
            "dtr_new_results_table.csv."
        )

    for row in rows:
        if row["scenario"] == "S2" and row["arm"] == "dtr_age_temporal_new":
            # Classification vs CF magnitudes
            if row["mechanism_classification"] == "NO_MECHANISM_RECOVERY" and (
                row["Surface_RMSE"] is not None and row["Surface_RMSE"] < 0.25
            ):
                discrepancies.append(
                    "S2 dtr_age_temporal_new has Surface_RMSE="
                    f"{row['Surface_RMSE']:.3f} (partial threshold=0.25) but "
                    "mechanism_classification=NO_MECHANISM_RECOVERY because the "
                    "classifier also requires S0 CF-RMSE-age ≤ 0.05; observed S0 "
                    f"CF-RMSE-age for this arm is "
                    f"{next(x['CF_RMSE_age'] for x in rows if x['scenario']=='S0' and x['arm']=='dtr_age_temporal_new'):.4f}."
                )

    # Config absence
    for scenario in SCENARIOS:
        for arm in ARMS:
            name = dtr_arm_dirname(arm, "_new")
            if not (SYN / name / scenario / "config.json").exists():
                # already noted generically once
                pass
    if not any((SYN / dtr_arm_dirname(a, "_new") / "S2" / "config.json").exists() for a in ARMS):
        discrepancies.append(
            "No per-run config.json under synthetic dtr_*_new/; training used "
            "baselines.synthetic.runner defaults (seed=0, data_seed=20260922, "
            "batch_size=32, max_epochs=25, Content-Persistence DTR d_model=64)."
        )

    # Predictions / CF outputs: probability surfaces are not persisted; only metrics.
    discrepancies.append(
        "No per-patient prediction tensors or age×lag surface arrays are saved under "
        "dtr_*_new/; only scalar CF metrics in cf_report.json / result.json. "
        "Surfaces in figures/final/ are regenerated by forward pass from "
        "best_checkpoint.pt on the same held-out CF template protocol."
    )

    payload = {
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "primary_protocol": "32-target controlled S2 (interaction=8 + non-interaction=24)",
        "runs": rows,
        "primary_s2_table_checks": table_checks,
        "discrepancies": discrepancies,
        "mechanism_summary": str(SYN / "dtr_new_mechanism_summary.json"),
        "results_table_csv": str(SYN / "dtr_new_results_table.csv"),
        "report_md": str(SYN / "dtr_new_REPORT.md"),
    }
    (OUT / "result_provenance.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# Result provenance — Content-Persistence DTR (`*_new`)",
        "",
        "Primary synthetic comparison: **32-target S2** protocol "
        "(8 interaction + 6 temporal_only + 6 age_only + 6 content_only + 6 null).",
        "The 8-interaction-only evaluation is secondary/appendix and is not used here "
        "as the main table.",
        "",
        f"- Data seed: `{DATA_SEED}`",
        f"- Model seed: `{MODEL_SEED}`",
        f"- Data root: `synthetic_age_temporal/outputs/data/seed{DATA_SEED}/controlled/{{S}}`",
        "- Architecture: Content-Persistence DTR (mass-preserving / raw-additive)",
        "- Arms: `dtr_age_temporal_new`, `dtr_temporal_only_new`",
        "",
        "## Per-run table",
        "",
        "| Scenario | Arm | Checkpoint | Seed | n_targets | splits (tr/va/te) | params | AUPRC | AUROC | CF-age | CF-lag | Surf | β̂ | θ̂₀ |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        sc = row["split_counts"]
        splits = f"{sc['train']}/{sc['val']}/{sc['test']}"
        lines.append(
            f"| {row['scenario']} | {row['arm']} | `{row['checkpoint']}` | "
            f"{row['model_seed']} | {row['n_targets']} | {splits} | "
            f"{row['trainable_params']} | "
            f"{_fmt(row['AUPRC'])} | {_fmt(row['AUROC'])} | "
            f"{_fmt(row['CF_RMSE_age'])} | {_fmt(row['CF_RMSE_lag'])} | "
            f"{_fmt(row['Surface_RMSE'])} | {_fmt(row['beta_hat'])} | "
            f"{_fmt(row['theta0_hat'])} |"
        )

    lines += [
        "",
        "## Evaluation artifacts",
        "",
        "For each arm/scenario:",
        "- `best_checkpoint.pt` / `checkpoint.pt` / `last_checkpoint.pt`",
        "- `result.json` (train history + test metrics + mechanism fields)",
        "- `cf_report.json` (CF-RMSE-age/lag, Surface RMSE, mechanism class)",
        "- `history.json`",
        "",
        "Aggregates:",
        f"- `{SYN / 'dtr_new_mechanism_summary.json'}`",
        f"- `{SYN / 'dtr_new_results_table.csv'}`",
        f"- `{SYN / 'cf_summary_S*_new.json'}`",
        "",
        "## Primary S2 table verification (3 d.p.)",
        "",
    ]
    for chk in table_checks:
        ok = all(chk["match_3dp"].values())
        lines.append(
            f"- **{chk['model']}**: {'MATCH' if ok else 'MISMATCH'} — "
            f"artifact AUPRC/AUROC/CF-age/CF-lag/Surf = "
            f"{_fmt(chk['got']['AUPRC'])}/{_fmt(chk['got']['AUROC'])}/"
            f"{_fmt(chk['got']['CF_RMSE_age'])}/{_fmt(chk['got']['CF_RMSE_lag'])}/"
            f"{_fmt(chk['got']['Surface_RMSE'])}"
        )

    lines += ["", "## Discrepancies (not silently reconciled)", ""]
    if discrepancies:
        for d in discrepancies:
            lines.append(f"- {d}")
    else:
        lines.append("- None.")

    lines += [
        "",
        "## Notes",
        "",
        "- Patient splits are example-level controlled splits from `splits.json` "
        f"(same sizes across S0–S5 for seed {DATA_SEED}): "
        f"{rows[0]['split_counts']}.",
        "- Frozen text/code embeddings: none for DTR (code embeddings are trainable).",
        "- NCH leakage-corrected matched `_new` pair is incomplete at provenance time "
        "(see `nch_final_analysis.md`).",
        "",
    ]
    (OUT / "result_provenance.md").write_text("\n".join(lines))
    print("wrote result_provenance.md")
    return payload


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


# ---------------------------------------------------------------------------
# 2. Functional diagnostics
# ---------------------------------------------------------------------------
@torch.no_grad()
def mean_abs_logit_changes(
    model,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Mean |Δlogit| under age-shuffle and β=0 on the given loader."""
    base = predict(model, loader, device)
    base_logits = base["logits"]

    # Age shuffle (same RNG as train_dtr.ablations)
    batches = []
    ages = []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batches.append(b)
        ages.append(b["age"].detach().cpu().numpy())
    flat = np.concatenate(ages)
    shuf = np.random.default_rng(0).permutation(flat)
    ptr = 0
    sh_logits = []
    for b in batches:
        bsz = b["age"].size(0)
        age = torch.tensor(shuf[ptr : ptr + bsz], dtype=torch.float32, device=device)
        ptr += bsz
        out = model(
            enc_code_ids=b["enc_code_ids"],
            enc_code_mask=b["enc_code_mask"],
            enc_tau=b["enc_tau"],
            enc_padding_mask=b["enc_padding_mask"],
            age=age,
        )
        sh_logits.append(out.detach().cpu().numpy())
    sh_logits_a = np.concatenate(sh_logits)

    saved = model.zero_all_betas_()
    b0_logits = []
    for b in batches:
        out = model(
            enc_code_ids=b["enc_code_ids"],
            enc_code_mask=b["enc_code_mask"],
            enc_tau=b["enc_tau"],
            enc_padding_mask=b["enc_padding_mask"],
            age=b["age"],
        )
        b0_logits.append(out.detach().cpu().numpy())
    model.restore_betas_(saved)
    b0_logits_a = np.concatenate(b0_logits)

    return {
        "mean_abs_logit_change_age_shuffle": float(np.mean(np.abs(sh_logits_a - base_logits))),
        "mean_abs_logit_change_beta0": float(np.mean(np.abs(b0_logits_a - base_logits))),
    }


def lambda_curve(inner_model, device: torch.device) -> dict[str, list[float]]:
    ages = np.arange(0, 19, 1.0)
    with torch.no_grad():
        lam = (
            inner_model.lambda_of(
                torch.tensor(ages, dtype=torch.float32, device=device)
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return {"ages": ages.tolist(), "lambda": [float(x) for x in lam]}


def s0_cf_decomposition(
    model,
    device: torch.device,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Explain nontrivial S0 CF-RMSE-age without changing the metric."""
    scenario = "S0"
    cfg = Config(data_seed=DATA_SEED)
    scenario_dir = cfg.data_dir() / "controlled" / scenario
    _, _, test_loader, vocab, info = make_loaders(scenario_dir, batch_size=1)
    itos = dict(vocab.itos)
    # Token template for oracle
    template = None
    for batch in test_loader:
        if batch["is_signal"].any():
            template = {k: v for k, v in batch.items()}
            break
    if template is None:
        raise RuntimeError("No oracle CF template for S0")
    _, _, dtr_test, _, _ = make_dtr_baseline_loaders(
        scenario, data_seed=DATA_SEED, batch_size=1
    )
    dtr_template = None
    for batch in dtr_test:
        if (~batch["enc_padding_mask"]).any():
            dtr_template = {k: v for k, v in batch.items()}
            break
    if dtr_template is None:
        raise RuntimeError("No DTR CF template for S0")
    specs = meta["_specs"]
    theta0 = float(meta["theta0"])
    beta = float(meta["beta_true"])
    o_age, _, o_surf = _build_oracle_fns(template, itos, specs, scenario, theta0, beta)
    p_age, _, p_surf, _ = make_predict_fns(model, dtr_template, device, info["n_codes"])

    inter = np.array(meta["_interaction_idx"], dtype=int)
    age_only = np.array(meta["_age_only_idx"], dtype=int)
    all_idx = np.arange(meta["n_targets"])

    def _subset_rmse(predict_fn, oracle_fn, idxs: np.ndarray) -> float:
        def p_wrap(a):
            return np.asarray(predict_fn(a), dtype=np.float64)[idxs]

        def o_wrap(a):
            return np.asarray(oracle_fn(a), dtype=np.float64)[idxs]

        return cf_rmse_age(p_wrap, o_wrap)

    # Age-head magnitude
    age_w = model._model.age_head.weight.detach().cpu().numpy()
    age_b = model._model.age_head.bias.detach().cpu().numpy()
    beta_hat = float(model._model.beta.detach().cpu().reshape(-1)[0])
    theta0_hat = float(model._model.theta0.detach().cpu().reshape(-1)[0])

    # Oracle age sensitivity on interaction vs age_only (lag fixed in template)
    ages = (2.0, 5.0, 9.0, 13.0, 17.0)
    o_grid = np.stack([o_age(a) for a in ages], axis=0)
    p_grid = np.stack([p_age(a) for a in ages], axis=0)
    oracle_age_std = {
        "all": float(o_grid.std(axis=0).mean()),
        "interaction": float(o_grid[:, inter].std(axis=0).mean()),
        "age_only": float(o_grid[:, age_only].std(axis=0).mean()),
    }
    model_age_std = {
        "all": float(p_grid.std(axis=0).mean()),
        "interaction": float(p_grid[:, inter].std(axis=0).mean()),
        "age_only": float(p_grid[:, age_only].std(axis=0).mean()),
    }

    # Zero age-head residual CF (diagnostic only)
    saved_w = model._model.age_head.weight.data.clone()
    saved_b = model._model.age_head.bias.data.clone()
    model._model.age_head.weight.data.zero_()
    model._model.age_head.bias.data.zero_()
    p_age_no_ah, _, _, _ = make_predict_fns(model, dtr_template, device, info["n_codes"])
    cf_no_age_head = {
        "all": _subset_rmse(p_age_no_ah, o_age, all_idx),
        "interaction": _subset_rmse(p_age_no_ah, o_age, inter),
        "age_only": _subset_rmse(p_age_no_ah, o_age, age_only),
    }
    model._model.age_head.weight.data.copy_(saved_w)
    model._model.age_head.bias.data.copy_(saved_b)

    cf_full = {
        "all": _subset_rmse(p_age, o_age, all_idx),
        "interaction": _subset_rmse(p_age, o_age, inter),
        "age_only": _subset_rmse(p_age, o_age, age_only),
    }

    explanation = {
        "beta_hat": beta_hat,
        "theta0_hat": theta0_hat,
        "beta_true": beta,
        "age_head_weight_l2": float(np.linalg.norm(age_w)),
        "age_head_bias_l2": float(np.linalg.norm(age_b)),
        "mean_abs_age_head_weight": float(np.mean(np.abs(age_w))),
        "oracle_age_std_across_cf_ages": oracle_age_std,
        "model_age_std_across_cf_ages": model_age_std,
        "cf_rmse_age_by_mechanism_subset": cf_full,
        "cf_rmse_age_with_age_head_zeroed": cf_no_age_head,
        "interpretation": [],
    }
    interp = explanation["interpretation"]
    interp.append(
        "(c) CF-RMSE-age is defined over all 32 targets, including 6 age_only "
        "targets with genuine oracle age main effects (gamma≠0). "
        f"Subset CF-RMSE-age: interaction={cf_full['interaction']:.4f}, "
        f"age_only={cf_full['age_only']:.4f}, all={cf_full['all']:.4f}."
    )
    interp.append(
        "(a) DTR has an explicit age main-effect head f_age(z) shared by both arms; "
        f"||W_age||_2={explanation['age_head_weight_l2']:.3f}. "
        "Zeroing age_head changes CF-RMSE-age "
        f"all {cf_full['all']:.4f}→{cf_no_age_head['all']:.4f}, "
        f"age_only {cf_full['age_only']:.4f}→{cf_no_age_head['age_only']:.4f}."
    )
    interp.append(
        f"(e) β̂={beta_hat:.4g}≈0 and β_true=0, so this is not genuine "
        "age×lag interaction sensitivity via the gate pathway."
    )
    interp.append(
        "(d) Metric definition: CF-RMSE-age = RMS over ages of mean-squared "
        "error across targets between counterfactual predictions and oracle; "
        "it does not isolate the β pathway."
    )
    interp.append(
        "(b) Implicit age-in-content is possible but secondary here: controlled "
        "synthetic codes are not age-labeled; residual CF after zeroing age_head "
        f"(all={cf_no_age_head['all']:.4f}) can still reflect history/content "
        "mismatch vs oracle, not necessarily content-encoded age."
    )
    return explanation


def compute_diagnostics(device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    s0_extra: dict[str, Any] = {}

    for scenario in SCENARIOS:
        meta = data_meta(scenario)
        _, val_raw, test_raw, _, info = make_dtr_baseline_loaders(
            scenario, data_seed=DATA_SEED, batch_size=32
        )
        for arm in ARMS:
            name = dtr_arm_dirname(arm, "_new")
            arm_dir = SYN / name / scenario
            result = load_result(arm_dir)
            model = build_model(
                "dtr",
                info["n_codes"],
                info.get("n_types", 11),
                info["n_targets"],
                arm=arm,
            )
            model.load_checkpoint(arm_dir)
            model.to(device)
            model.eval()

            theta0_hat = float(model._model.theta0.detach().cpu().reshape(-1)[0])
            beta_hat = float(model._model.beta.detach().cpu().reshape(-1)[0])
            lam = lambda_curve(model._model, device)

            # Ablations + logit changes on TEST (held-out), matching requested patients.
            # Existing delta_BCE_* in result.json used VAL; we recompute on TEST and
            # also report the stored VAL numbers for provenance.
            abl = ablations(model._model, test_raw, device) if arm == "age_temporal" else None
            logit_chg = (
                mean_abs_logit_changes(model._model, test_raw, device)
                if arm == "age_temporal"
                else {
                    "mean_abs_logit_change_age_shuffle": None,
                    "mean_abs_logit_change_beta0": None,
                }
            )

            z = (np.array(lam["ages"]) - 9.0) / 9.0
            lam_true = _softplus(float(meta["theta0"]) + float(meta["beta_true"]) * z)

            row = {
                "scenario": scenario,
                "arm": name,
                "beta_true": float(meta["beta_true"]),
                "theta0_true": float(meta["theta0"]),
                "beta_hat": beta_hat,
                "theta0_hat": theta0_hat,
                "lambda_ages": lam["ages"],
                "lambda_hat": lam["lambda"],
                "lambda_true": [float(x) for x in lam_true],
                "AUPRC": result.get("AUPRC"),
                "AUROC": result.get("AUROC"),
                "BCE": result.get("BCE"),
                "CF_RMSE_age": result.get("CF_RMSE_age"),
                "CF_RMSE_lag": result.get("CF_RMSE_lag"),
                "Surface_RMSE": result.get("Surface_RMSE"),
                "delta_BCE_age_shuffle_val_stored": result.get("delta_BCE_age_shuffle"),
                "delta_BCE_beta0_val_stored": result.get("delta_BCE_beta0"),
                "delta_BCE_age_shuffle_test": (
                    float(abl["delta_bce_shuffle_age"]) if abl else None
                ),
                "delta_BCE_beta0_test": float(abl["delta_bce_beta0"]) if abl else None,
                **logit_chg,
                "lambda_corr": result.get("lambda_corr"),
                "param_Surface_RMSE": result.get("param_Surface_RMSE"),
                "S5_Surface_RMSE_acute": result.get("S5_Surface_RMSE_acute"),
                "S5_Surface_RMSE_intermediate": result.get("S5_Surface_RMSE_intermediate"),
                "S5_Surface_RMSE_chronic": result.get("S5_Surface_RMSE_chronic"),
                "S5_Surface_RMSE_mean": result.get("S5_Surface_RMSE_mean"),
                "persistence_order_correct": result.get("persistence_order_correct"),
                "n_targets": meta["n_targets"],
                "split_counts": meta["_split_counts"],
                "trainable_params": (result.get("model_card") or {}).get("trainable_params"),
            }

            # S5 decay proxy from lambda curve shape is weak for global β; use
            # stored group Surface RMSE as primary S5 diagnostics.
            if scenario == "S5" and arm == "age_temporal":
                # Black-box decay proxy: mean(p_short - p_long) over ages from CF surface
                # regenerated below in surface cache; placeholder filled later if needed.
                row["s5_decay_proxy_note"] = (
                    "Class-specific Surface RMSE from stored result.json; "
                    "ordering acute>intermediate>chronic expected for decay rate via "
                    "lower long-lag relevance for acute."
                )

            rows.append(row)

            if scenario == "S0" and arm == "age_temporal":
                s0_extra = s0_cf_decomposition(model, device, meta)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Persist
    diag = {"seed_model": MODEL_SEED, "data_seed": DATA_SEED, "rows": rows, "s0_analysis": s0_extra}
    (OUT / "dtr_functional_diagnostics.json").write_text(
        json.dumps(diag, indent=2, default=str)
    )

    # CSV (flatten lambda)
    fieldnames = [
        "scenario",
        "arm",
        "beta_true",
        "theta0_true",
        "beta_hat",
        "theta0_hat",
        "AUPRC",
        "AUROC",
        "BCE",
        "CF_RMSE_age",
        "CF_RMSE_lag",
        "Surface_RMSE",
        "delta_BCE_age_shuffle_val_stored",
        "delta_BCE_beta0_val_stored",
        "delta_BCE_age_shuffle_test",
        "delta_BCE_beta0_test",
        "mean_abs_logit_change_age_shuffle",
        "mean_abs_logit_change_beta0",
        "lambda_corr",
        "param_Surface_RMSE",
        "S5_Surface_RMSE_acute",
        "S5_Surface_RMSE_intermediate",
        "S5_Surface_RMSE_chronic",
        "S5_Surface_RMSE_mean",
        "persistence_order_correct",
        "trainable_params",
        "lambda_hat_json",
    ]
    with (OUT / "dtr_functional_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["lambda_hat_json"] = json.dumps(row["lambda_hat"])
            w.writerow(out)

    # Markdown
    lines = [
        "# DTR functional mechanism diagnostics (`*_new`)",
        "",
        "Held-out patients: controlled test split "
        f"(n={rows[0]['split_counts']['test']}, data_seed={DATA_SEED}).",
        "Stored ΔBCE_* in result.json were computed on **validation** "
        "(see `baselines/synthetic/eval_dtr_mechanism_new.py`); this report also "
        "recomputes ΔBCE and mean |Δlogit| on the **test** set.",
        "",
        "| Scenario | Arm | β̂ | θ̂₀ | AUPRC | Surf | CF-age | shuffleΔBCE(val/test) | β=0ΔBCE(val/test) | |Δlogit| shuffle | |Δlogit| β=0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['arm']} | {_fmt(row['beta_hat'],4)} | "
            f"{_fmt(row['theta0_hat'],4)} | {_fmt(row['AUPRC'])} | "
            f"{_fmt(row['Surface_RMSE'])} | {_fmt(row['CF_RMSE_age'])} | "
            f"{_fmt(row['delta_BCE_age_shuffle_val_stored'],4)}/"
            f"{_fmt(row['delta_BCE_age_shuffle_test'],4)} | "
            f"{_fmt(row['delta_BCE_beta0_val_stored'],4)}/"
            f"{_fmt(row['delta_BCE_beta0_test'],4)} | "
            f"{_fmt(row['mean_abs_logit_change_age_shuffle'],4)} | "
            f"{_fmt(row['mean_abs_logit_change_beta0'],4)} |"
        )

    lines += ["", "## S0 CF-RMSE-age investigation", ""]
    if s0_extra:
        for s in s0_extra.get("interpretation", []):
            lines.append(f"- {s}")
        lines += [
            "",
            "### Quantitative decomposition",
            "",
            "```json",
            json.dumps(
                {
                    k: s0_extra[k]
                    for k in (
                        "beta_hat",
                        "age_head_weight_l2",
                        "oracle_age_std_across_cf_ages",
                        "model_age_std_across_cf_ages",
                        "cf_rmse_age_by_mechanism_subset",
                        "cf_rmse_age_with_age_head_zeroed",
                    )
                },
                indent=2,
            ),
            "```",
            "",
            "**Conclusion:** Nontrivial S0 CF-RMSE-age is primarily explained by "
            "(c) averaging over non-interaction targets (especially age_only) and "
            "(a)/(d) the age main-effect head + CF metric definition — **not** by "
            "false β-pathway interaction sensitivity. The official all-target "
            "CF-RMSE-age metric is left unchanged.",
        ]
    (OUT / "dtr_functional_diagnostics.md").write_text("\n".join(lines))
    print("wrote dtr_functional_diagnostics.*")
    return diag


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------
def _select_templates(scenario: str, device: torch.device):
    cfg = Config(data_seed=DATA_SEED)
    scenario_dir = cfg.data_dir() / "controlled" / scenario
    _, _, test_loader, vocab, info = make_loaders(scenario_dir, batch_size=1)
    itos = dict(vocab.itos)
    template = None
    for batch in test_loader:
        if batch["is_signal"].any():
            template = {k: v for k, v in batch.items()}
            break
    if template is None:
        raise RuntimeError(f"No CF template for {scenario}")
    _, _, dtr_test, _, _ = make_dtr_baseline_loaders(
        scenario, data_seed=DATA_SEED, batch_size=1
    )
    dtr_template = None
    for batch in dtr_test:
        if (~batch["enc_padding_mask"]).any():
            dtr_template = {k: v for k, v in batch.items()}
            break
    specs = json.loads((scenario_dir / "target_specs.json").read_text())
    meta = json.loads((scenario_dir / "meta.json").read_text())
    return template, dtr_template, vocab, info, specs, meta, itos


def _load_baseline_checkpoint(model: Any, model_dir: Path) -> None:
    """Load checkpoint, resizing code embeddings if adapter vocab convention drifted."""
    model_dir = Path(model_dir)
    ckpt_path = model_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "best_checkpoint.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "code_embedding.weight" in state and hasattr(
        model, "code_embedding"
    ):
        vs = int(state["code_embedding.weight"].shape[0])
        if model.code_embedding.weight.shape[0] != vs:
            d = model.code_embedding.embedding_dim
            model.code_embedding = torch.nn.Embedding(vs, d, padding_idx=0)
            model.vocab_size = vs
            if hasattr(model, "cls_id"):
                model.cls_id = min(int(getattr(model, "cls_id", vs - 1)), vs - 1)
    model.load_state_dict(state)


def build_model_surface(
    model_name: str,
    scenario: str,
    device: torch.device,
    template,
    dtr_template,
    info,
    specs,
    meta,
    itos,
) -> tuple[np.ndarray, float]:
    """Return mean-target probability surface [n_ages, n_lags] and Surface RMSE."""
    o_age, o_lag, o_surf = _build_oracle_fns(
        template,
        itos,
        specs,
        scenario,
        float(meta["theta0"]),
        float(meta["beta_true"]),
    )
    if model_name == "oracle":
        grid = build_surface_grid(o_surf)  # [A,L,T]
        return grid.mean(axis=-1), 0.0

    result_path = SYN / model_name / scenario / "result.json"
    saved = json.loads(result_path.read_text()) if result_path.exists() else {}
    n_codes = int(saved.get("n_codes", info["n_codes"]))
    n_types = int(saved.get("n_types") or info.get("n_types", 11))
    n_targets = int(saved.get("n_targets", info["n_targets"]))

    if model_name.startswith("dtr_"):
        arm = "age_temporal" if "age_temporal" in model_name else "temporal_only"
        model = build_model("dtr", n_codes, n_types, n_targets, arm=arm)
        model.load_checkpoint(SYN / model_name / scenario)
        cf_template = dtr_template
    else:
        model = build_model(model_name, n_codes, n_types, n_targets)
        _load_baseline_checkpoint(model, SYN / model_name / scenario)
        cf_template = template
    if hasattr(model, "to"):
        model.to(device)
    model.eval()
    _, _, p_surf, _ = make_predict_fns(model, cf_template, device, n_codes)
    grid = build_surface_grid(p_surf)
    srmse = surface_rmse(p_surf, o_surf)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grid.mean(axis=-1), float(srmse)


def regenerate_surfaces(device: torch.device) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    for scenario in ("S2", "S3"):
        template, dtr_template, vocab, info, specs, meta, itos = _select_templates(
            scenario, device
        )
        surfaces = {}
        rmses = {}
        for name in (
            "oracle",
            "cehrbert",
            "dtr_temporal_only_new",
            "dtr_age_temporal_new",
        ):
            print(f"  surface {scenario} {name}...")
            surf, srmse = build_model_surface(
                name,
                scenario,
                device,
                template,
                dtr_template,
                info,
                specs,
                meta,
                itos,
            )
            surfaces[name] = surf
            rmses[name] = srmse
            # Prefer stored Surface RMSE for non-oracle to match table exactly
            if name != "oracle":
                stored = load_result(SYN / name / scenario).get("Surface_RMSE")
                if stored is not None:
                    rmses[name] = float(stored)
        cache[scenario] = {
            "surfaces": {k: v.tolist() for k, v in surfaces.items()},
            "surface_rmse": rmses,
            "ages": list(SURFACE_AGES),
            "lags": list(SURFACE_LAGS_DAYS),
        }
        # also keep arrays
        cache[scenario]["_arrays"] = surfaces
    (OUT / "surface_grids_s2_s3.json").write_text(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "_arrays"} for k, v in cache.items()},
            indent=2,
        )
    )
    return cache


# ---------------------------------------------------------------------------
# Capacity table
# ---------------------------------------------------------------------------
def write_capacity_table() -> list[dict[str, Any]]:
    rows = []
    specs = {
        "dtr_age_temporal_new": {
            "display": "DTR",
            "explicit_age_input": True,
            "explicit_lag_input": True,
            "explicit_age_x_lag_pathway": True,
        },
        "dtr_temporal_only_new": {
            "display": "Temporal-only DTR",
            "explicit_age_input": True,  # age main-effect head still present
            "explicit_lag_input": True,
            "explicit_age_x_lag_pathway": False,
        },
        "cehrbert": {
            "display": "CEHR-BERT",
            "explicit_age_input": True,
            "explicit_lag_input": True,
            "explicit_age_x_lag_pathway": False,
        },
        "behrt": {
            "display": "BEHRT",
            "explicit_age_input": True,
            "explicit_lag_input": False,
            "explicit_age_x_lag_pathway": False,
        },
        "retain": {
            "display": "RETAIN",
            "explicit_age_input": False,
            "explicit_lag_input": True,
            "explicit_age_x_lag_pathway": False,
        },
        "count_lightgbm": {
            "display": "Count+LightGBM",
            "explicit_age_input": True,
            "explicit_lag_input": False,
            "explicit_age_x_lag_pathway": False,
        },
    }
    for key, meta in specs.items():
        r = load_result(SYN / key / "S2")
        mc = r.get("model_card") or {}
        pc = r.get("param_counts") or {}
        trainable = mc.get("trainable_params", pc.get("trainable_params"))
        total = pc.get("total_params", trainable)
        emb = mc.get("embedding_params", pc.get("embedding_params", 0))
        # Frozen: none recorded; embeddings counted in trainable for neural models
        frozen = 0
        if isinstance(trainable, str):
            # LightGBM
            trainable_n = None
            frozen = None
            total_n = None
            emb = None
        else:
            trainable_n = int(trainable) if trainable is not None else None
            total_n = int(total) if total is not None else trainable_n
            emb = int(emb) if emb is not None else 0
        rows.append(
            {
                "model": meta["display"],
                "trainable_params": trainable_n if trainable_n is not None else trainable,
                "frozen_params": frozen if frozen is not None else "N/A",
                "total_params": total_n if total_n is not None else total,
                "embedding_params_trainable": emb,
                "hidden_dim": mc.get("hidden_size") or mc.get("d_emb"),
                "layers": mc.get("layers"),
                "heads": mc.get("heads"),
                "explicit_age_input": meta["explicit_age_input"],
                "explicit_lag_input": meta["explicit_lag_input"],
                "explicit_age_x_lag_pathway": meta["explicit_age_x_lag_pathway"],
                "notes": (
                    "Code/segment embeddings are trainable (not frozen) for neural "
                    "baselines. DTR temporal_only freezes β=0 but still has f_age(z)."
                    if key != "count_lightgbm"
                    else "Tree ensemble; param count N/A under NN definition."
                ),
            }
        )

    with (OUT / "model_capacity.csv").open("w", newline="") as f:
        fields = [
            "model",
            "trainable_params",
            "frozen_params",
            "total_params",
            "hidden_dim",
            "layers",
            "heads",
            "explicit_age_input",
            "explicit_lag_input",
            "explicit_age_x_lag_pathway",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    (OUT / "model_capacity.json").write_text(json.dumps(rows, indent=2, default=str))
    print("wrote model_capacity.csv")
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_surface_comparison(surface_cache: dict[str, Any]) -> None:
    apply_style()
    names = [
        ("oracle", "Oracle"),
        ("cehrbert", "CEHR-BERT"),
        ("dtr_temporal_only_new", "Temporal-only DTR"),
        ("dtr_age_temporal_new", "DTR"),
    ]
    # S2 only primary
    sc = "S2"
    arrays = surface_cache[sc]["_arrays"]
    rmses = surface_cache[sc]["surface_rmse"]
    ages = surface_cache[sc]["ages"]
    lags = surface_cache[sc]["lags"]

    preds = [arrays[k] for k, _ in names]
    vmin = float(min(a.min() for a in preds))
    vmax = float(max(a.max() for a in preds))
    oracle = arrays["oracle"]
    residuals = [np.abs(arrays[k] - oracle) for k, _ in names[1:]]
    rmax = float(max(r.max() for r in residuals)) if residuals else 1.0

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), constrained_layout=True)
    lag_labs = [str(int(x)) if x != 0 else "0" for x in lags]
    for j, (key, title) in enumerate(names):
        ax = axes[0, j]
        im0 = ax.imshow(
            arrays[key],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        srmse = rmses[key]
        ax.set_title(f"{title}\nSurface RMSE={srmse:.3f}", fontsize=7.5)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels(lag_labs, fontsize=5.5, rotation=45)
        ax.set_yticks(range(0, len(ages), 3))
        ax.set_yticklabels([str(int(ages[i])) for i in range(0, len(ages), 3)])
        if j == 0:
            ax.set_ylabel("Age (y)")
        ax.set_xlabel("Lag (d)")

        axr = axes[1, j]
        if key == "oracle":
            axr.axis("off")
            axr.text(0.5, 0.5, "Oracle\n(reference)", ha="center", va="center", fontsize=8)
        else:
            im1 = axr.imshow(
                np.abs(arrays[key] - oracle),
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=rmax,
                interpolation="nearest",
            )
            axr.set_title(r"|pred − oracle|", fontsize=7.5)
            axr.set_xticks(range(len(lags)))
            axr.set_xticklabels(lag_labs, fontsize=5.5, rotation=45)
            axr.set_yticks(range(0, len(ages), 3))
            axr.set_yticklabels([str(int(ages[i])) for i in range(0, len(ages), 3)])
            if j == 1:
                axr.set_ylabel("Age (y)")
            axr.set_xlabel("Lag (d)")

    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02, label="Mean P")
    fig.colorbar(im1, ax=axes[1, 1:].tolist(), fraction=0.025, pad=0.02, label="|ΔP|")
    save_fig(fig, "synthetic_surface_comparison_32target")

    # Optional S2+S3
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), constrained_layout=True)
    all_preds = []
    for sc in ("S2", "S3"):
        for k, _ in names:
            all_preds.append(surface_cache[sc]["_arrays"][k])
    vmin = float(min(a.min() for a in all_preds))
    vmax = float(max(a.max() for a in all_preds))
    for i, sc in enumerate(("S2", "S3")):
        arrays = surface_cache[sc]["_arrays"]
        rmses = surface_cache[sc]["surface_rmse"]
        for j, (key, title) in enumerate(names):
            ax = axes[i, j]
            im = ax.imshow(
                arrays[key],
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"{sc} {title}\nRMSE={rmses[key]:.3f}", fontsize=7)
            ax.set_xticks(range(len(lags)))
            ax.set_xticklabels(lag_labs, fontsize=5, rotation=45)
            ax.set_yticks(range(0, len(ages), 6))
            ax.set_yticklabels([str(int(ages[t])) for t in range(0, len(ages), 6)])
            if j == 0:
                ax.set_ylabel(f"{sc}\nAge")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="Mean P")
    save_fig(fig, "synthetic_surface_comparison_s2_s3")


def fig_functional_recovery(diag: dict[str, Any]) -> None:
    apply_style()
    scenarios = ["S0", "S1", "S2", "S3"]
    by = {(r["scenario"], r["arm"]): r for r in diag["rows"]}
    beta_true = [data_meta(s)["beta_true"] for s in scenarios]
    beta_hat = [by[(s, "dtr_age_temporal_new")]["beta_hat"] for s in scenarios]
    shuf = [
        by[(s, "dtr_age_temporal_new")]["delta_BCE_age_shuffle_val_stored"] for s in scenarios
    ]
    b0 = [by[(s, "dtr_age_temporal_new")]["delta_BCE_beta0_val_stored"] for s in scenarios]
    surf_at = [by[(s, "dtr_age_temporal_new")]["Surface_RMSE"] for s in scenarios]
    surf_to = [by[(s, "dtr_temporal_only_new")]["Surface_RMSE"] for s in scenarios]
    dsurf = [t - a for t, a in zip(surf_to, surf_at)]  # positive => AT better

    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, beta_true, "o--", color="#718096", label=r"$\beta_{\mathrm{true}}$", ms=4)
    ax.plot(x, beta_hat, "s-", color="#0B6E4F", label=r"$\hat\beta$", ms=4)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(r"$\beta$")
    ax.set_title("A  Parameter")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    ax.plot(x, shuf, "o-", color="#2B6CB0", ms=4)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(r"age-shuffle $\Delta$BCE")
    ax.set_title("B  Age shuffle")

    ax = axes[2]
    ax.plot(x, b0, "o-", color="#C45C26", ms=4)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(r"$\beta{=}0$ $\Delta$BCE")
    ax.set_title(r"C  $\beta{=}0$")

    ax = axes[3]
    ax.plot(x, dsurf, "o-", color="#0B6E4F", ms=4, label=r"$\Delta$Surf (TO−AT)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(r"$\Delta$ Surface RMSE")
    ax.set_title("D  vs temporal-only")

    save_fig(fig, "synthetic_functional_recovery")


def fig_prediction_vs_surface(capacity_rows: list[dict[str, Any]]) -> None:
    apply_style()
    cap = {r["model"]: r for r in capacity_rows}
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
    colors = {
        "DTR": "#0B6E4F",
        "Temporal-only DTR": "#C45C26",
        "CEHR-BERT": "#2B6CB0",
        "BEHRT": "#6B46C1",
        "RETAIN": "#C05621",
        "Count+LightGBM": "#718096",
    }
    for label, exp in PRIMARY_S2.items():
        key = exp["result_key"]
        r = load_result(SYN / key / "S2")
        x = float(r["AUPRC"])
        y = float(r["Surface_RMSE"])
        params = cap.get(label, {}).get("trainable_params")
        ax.scatter(x, y, s=36, color=colors.get(label, "k"), zorder=3)
        annot = label
        if isinstance(params, int):
            annot = f"{label}\n({params/1e3:.0f}k)" if params >= 1000 else f"{label}\n({params})"
        ax.annotate(
            annot,
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6,
        )
    ax.set_xlabel("AUPRC (↑)")
    ax.set_ylabel("Surface RMSE (↓)")
    ax.set_title("S2: prediction vs counterfactual fidelity")
    save_fig(fig, "s2_prediction_vs_surface")


def fig_dtr_gain(diag: dict[str, Any]) -> None:
    apply_style()
    scenarios = ["S0", "S1", "S2", "S3", "S5"]
    by = {(r["scenario"], r["arm"]): r for r in diag["rows"]}
    metrics = [
        ("CF_RMSE_age", "CF-RMSE-age"),
        ("CF_RMSE_lag", "CF-RMSE-lag"),
        ("Surface_RMSE", "Surface RMSE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True, sharey=False)
    x = np.arange(len(scenarios))
    for ax, (key, title) in zip(axes, metrics):
        delta = [
            by[(s, "dtr_temporal_only_new")][key] - by[(s, "dtr_age_temporal_new")][key]
            for s in scenarios
        ]
        ax.axhline(0, color="k", lw=0.7, alpha=0.6)
        ax.plot(x, delta, "o-", color="#0B6E4F", ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.set_title(title)
        ax.set_ylabel(r"$\Delta$ (TO − AT)")
    axes[0].set_xlabel("Scenario")
    save_fig(fig, "dtr_gain_across_scenarios")

    # S5 persistence
    at = by[("S5", "dtr_age_temporal_new")]
    to = by[("S5", "dtr_temporal_only_new")]
    groups = ["acute", "intermediate", "chronic"]
    at_v = [
        at["S5_Surface_RMSE_acute"],
        at["S5_Surface_RMSE_intermediate"],
        at["S5_Surface_RMSE_chronic"],
    ]
    to_v = [
        to["S5_Surface_RMSE_acute"],
        to["S5_Surface_RMSE_intermediate"],
        to["S5_Surface_RMSE_chronic"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), constrained_layout=True)
    x = np.arange(len(groups))
    w = 0.35
    axes[0].bar(x - w / 2, to_v, w, label="Temporal-only", color="#C45C26")
    axes[0].bar(x + w / 2, at_v, w, label="DTR", color="#0B6E4F")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("Surface RMSE")
    axes[0].set_title("S5 class-specific Surface RMSE")
    axes[0].legend(frameon=False, fontsize=6)

    # Decay proxy: use inverse of surface fidelity gap is wrong.
    # Use stored group RMSE ordering: lower RMSE for chronic if model captures persistence.
    # Also plot mean lambda from age_temporal as a single global curve note.
    axes[1].plot(x, to_v, "o--", color="#C45C26", label="Temporal-only")
    axes[1].plot(x, at_v, "s-", color="#0B6E4F", label="DTR")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylabel("Surface RMSE")
    axes[1].set_title("Expected: acute hardest if miss fast decay")
    axes[1].legend(frameon=False, fontsize=6)
    # Annotate order
    def order_ok(vals):
        # persistence recovery uses decay score order, not RMSE order.
        return at.get("persistence_order_correct")

    axes[1].text(
        0.02,
        0.98,
        f"persistence_order_correct(AT)={at.get('persistence_order_correct')}\n"
        f"persistence_order_correct(TO)={to.get('persistence_order_correct')}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=6,
    )
    save_fig(fig, "s5_persistence_recovery")


def fig_param_vs_functional(diag: dict[str, Any]) -> None:
    """Original Transformer vs factorized DTR: parameter vs functional reliance."""
    apply_style()
    # Sources (existing metrics only)
    xf = REPO / (
        "synthetic_age_temporal/outputs/runs/controlled/"
        "arch_S2_age_temporal_d20260922_m0_interonly/metrics.json"
    )
    # Prefer the encounter-level factorized run with strong functional reliance as
    # an older positive control, plus current baseline _new factorized.
    old_factorized = REPO / (
        "synthetic_age_temporal/outputs/runs/dtr/"
        "controlled_S2_aggcmp_dtr_age_temporal_raw_additive_m0/metrics.json"
    )
    points = []
    if xf.exists():
        m = json.loads(xf.read_text())
        points.append(
            {
                "name": "Transformer DTR (S2)",
                "lambda_corr": m["recovery"]["corr_lambda"],
                "beta_err": abs(m["recovery"]["beta_hat"] - m["recovery"]["beta_true"])
                / abs(m["recovery"]["beta_true"]),
                "delta_bce_beta0": m["ablations"]["delta_bce_beta0"],
                "delta_bce_shuffle": m["ablations"]["delta_bce_shuffle_age"],
                "color": "#6B46C1",
            }
        )
    if old_factorized.exists():
        m = json.loads(old_factorized.read_text())
        points.append(
            {
                "name": "Factorized DTR (canonical S2)",
                "lambda_corr": m["recovery"]["corr_lambda"],
                "beta_err": abs(m["recovery"]["beta_hat"] - m["recovery"]["beta_true"])
                / abs(m["recovery"]["beta_true"]),
                "delta_bce_beta0": m["ablations"]["delta_bce_beta0"],
                "delta_bce_shuffle": m["ablations"]["delta_bce_shuffle_age"],
                "color": "#2B6CB0",
            }
        )
    # Current baseline factorized (main table)
    row = next(
        r
        for r in diag["rows"]
        if r["scenario"] == "S2" and r["arm"] == "dtr_age_temporal_new"
    )
    beta_true = abs(row["beta_true"])
    points.append(
        {
            "name": "Factorized DTR (baseline `_new` S2)",
            "lambda_corr": row.get("lambda_corr"),
            "beta_err": abs(row["beta_hat"] - row["beta_true"]) / beta_true,
            "delta_bce_beta0": row["delta_BCE_beta0_val_stored"],
            "delta_bce_shuffle": row["delta_BCE_age_shuffle_val_stored"],
            "color": "#0B6E4F",
        }
    )
    # S0 negative control
    row0 = next(
        r
        for r in diag["rows"]
        if r["scenario"] == "S0" and r["arm"] == "dtr_age_temporal_new"
    )
    points.append(
        {
            "name": "DTR `_new` S0 (neg. control)",
            "lambda_corr": row0.get("lambda_corr"),
            "beta_err": abs(row0["beta_hat"] - 0.0),  # true beta=0; use |beta_hat|
            "delta_bce_beta0": row0["delta_BCE_beta0_val_stored"],
            "delta_bce_shuffle": row0["delta_BCE_age_shuffle_val_stored"],
            "color": "#718096",
            "x_is_abs_beta": True,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), constrained_layout=True)
    for ax, ykey, ylab in [
        (axes[0], "delta_bce_beta0", r"$\beta{=}0$ $\Delta$BCE"),
        (axes[1], "delta_bce_shuffle", r"age-shuffle $\Delta$BCE"),
    ]:
        for p in points:
            if p.get("x_is_abs_beta"):
                # skip S0 on lambda-corr axis plot using beta_err panel only via left?
                x = p["beta_err"]
                xlab_note = True
            else:
                x = p["lambda_corr"]
                xlab_note = False
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            # Use lambda_corr on both panels for comparable models; S0 uses |beta_hat| only on left annotation
            if p.get("x_is_abs_beta"):
                continue
            ax.scatter(x, p[ykey], s=40, color=p["color"], zorder=3)
            ax.annotate(p["name"], (x, p[ykey]), textcoords="offset points", xytext=(3, 3), fontsize=5.5)
        ax.set_xlabel(r"$\lambda$-curve corr. with oracle")
        ax.set_ylabel(ylab)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    # Add S0 as open marker at left using beta_err secondary annotation text
    axes[0].text(
        0.02,
        0.02,
        f"S0 |β̂|={row0['beta_hat']:.4f}, "
        f"β=0 ΔBCE={row0['delta_BCE_beta0_val_stored']:.2e}",
        transform=axes[0].transAxes,
        fontsize=6,
        va="bottom",
    )
    axes[0].set_title("Parameter curve vs β=0 reliance")
    axes[1].set_title("Parameter curve vs age-shuffle reliance")
    save_fig(fig, "parameter_vs_functional_recovery")

    (OUT / "parameter_vs_functional_points.json").write_text(
        json.dumps(points, indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# NCH skip
# ---------------------------------------------------------------------------
def write_nch_skip() -> None:
    to = REPO / "results/baselines/nch/dtr_temporal_only_new"
    at = REPO / "results/baselines/nch/dtr_age_temporal_new"
    lines = [
        "# NCH final analysis",
        "",
        "## Status: SKIPPED — leakage-corrected matched `_new` pair incomplete",
        "",
        "Checked:",
        f"- `dtr_age_temporal_new` exists: **{at.exists()}**",
        f"- `dtr_temporal_only_new/result.json` exists: **{(to / 'result.json').exists()}**",
        f"- `dtr_temporal_only_new` history epochs: "
        f"{len(json.loads((to/'history.json').read_text())) if (to/'history.json').exists() else 0}",
        "",
        "Obsolete NCH runs (`dtr_age_temporal`, `dtr_no_interaction`) were **not** used, "
        "because the temporal-only arm previously benefited from leakage and is not a "
        "valid matched comparator for the final paper.",
        "",
        "No NCH figures were written under `figures/final/nch_*`.",
        "",
    ]
    (OUT / "nch_final_analysis.md").write_text("\n".join(lines))
    print("wrote nch_final_analysis.md (skip)")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(diag: dict[str, Any], capacity_rows: list[dict[str, Any]]) -> None:
    by = {(r["scenario"], r["arm"]): r for r in diag["rows"]}
    s2_at = by[("S2", "dtr_age_temporal_new")]
    s2_to = by[("S2", "dtr_temporal_only_new")]
    lines = [
        "# Figure analysis report (ICLR DTR)",
        "",
        "Primary synthetic protocol: **32-target S2**. "
        "Models: Content-Persistence `dtr_age_temporal_new` / `dtr_temporal_only_new`.",
        "",
        "---",
        "",
        "## Figure: `synthetic_surface_comparison_32target`",
        "",
        "- **Data source:** Forward-pass age×lag surfaces from `best_checkpoint.pt` "
        "for CEHR-BERT, Temporal-only DTR, DTR; oracle from controlled S2 generator. "
        "Grids: ages 0–18, lags (0,7,30,90,180,365,730). Same CF template protocol as "
        "`baselines/synthetic/counterfactual_eval.py`. Surface RMSE titles use stored "
        "result.json values.",
        "- **Plotted:** Mean predicted probability over 32 targets; second row absolute "
        "residual vs oracle.",
        f"- **Quantitative:** DTR Surface RMSE={s2_at['Surface_RMSE']:.3f}; "
        f"Temporal-only={s2_to['Surface_RMSE']:.3f}; "
        f"CEHR-BERT={load_result(SYN/'cehrbert'/'S2')['Surface_RMSE']:.3f}.",
        "- **Supported:** Visual mismatch between predictive baselines and oracle "
        "age×lag structure; DTR improves on temporal-only but does not dominate CEHR-BERT "
        "on Surface RMSE.",
        "- **Not supported:** Claiming DTR uniquely recovers the oracle surface; "
        "CEHR-BERT has lower Surface RMSE on this protocol.",
        "- **Caption:** Predicted age×lag response surfaces on the primary 32-target S2 "
        "benchmark (shared color scales). DTR reduces Surface RMSE relative to "
        "temporal-only but CEHR-BERT remains closest to the oracle surface among "
        "compared models.",
        "- **Recommendation:** Main paper.",
        "",
        "## Figure: `synthetic_surface_comparison_s2_s3`",
        "",
        "- **Data source:** Same protocol for S2 and S3.",
        f"- **Quantitative:** S3 DTR Surface RMSE={by[('S3','dtr_age_temporal_new')]['Surface_RMSE']:.3f} "
        f"vs temporal-only {by[('S3','dtr_temporal_only_new')]['Surface_RMSE']:.3f} "
        "(near-zero gain).",
        "- **Supported:** Cross-scenario visual comparison.",
        "- **Not supported:** Strong S3 mechanism advantage for DTR over temporal-only.",
        "- **Caption:** Age×lag surfaces for S2 (β_true=-2.5) and S3 (sign-flipped β) "
        "under the 32-target protocol.",
        "- **Recommendation:** Appendix.",
        "",
        "## Figure: `synthetic_functional_recovery`",
        "",
        "- **Data source:** `dtr_functional_diagnostics` / stored mechanism fields.",
        f"- **Quantitative:** β̂ S0→S3 = "
        + ", ".join(
            f"{s}:{by[(s,'dtr_age_temporal_new')]['beta_hat']:.3f}" for s in ["S0", "S1", "S2", "S3"]
        )
        + f"; S2 β=0 ΔBCE={s2_at['delta_BCE_beta0_val_stored']:.4f}, "
        f"shuffle ΔBCE={s2_at['delta_BCE_age_shuffle_val_stored']:.4f}.",
        "- **Supported:** β̂ near 0 on S0 and moves toward the signed interaction on "
        "S2/S3; functional ΔBCE is small on S0/S1 and larger on S2/S3, but absolute "
        "β=0 ΔBCE remains modest (≤0.01 on S2).",
        "- **Not supported:** Strong claim of functional mechanism recovery solely from "
        "β̂ sign matching; intervention ΔBCE magnitudes are small relative to canonical "
        "factorized DTR runs with ΔBCE_β0≈0.09.",
        "- **Caption:** Across S0–S3, DTR’s β̂ tracks the presence/sign of the planted "
        "interaction while age-shuffle and β=0 ΔBCE remain near zero without an "
        "interaction and increase when one exists—yet functional reliance is partial.",
        "- **Recommendation:** Main paper (with cautious wording).",
        "",
        "## Figure: `s2_prediction_vs_surface`",
        "",
        "- **Data source:** Primary S2 result.json for six models; capacity from "
        "`model_capacity.csv`.",
        "- **Plotted:** AUPRC vs Surface RMSE with parameter annotations.",
        "- **Supported:** Predictive ranking ≠ counterfactual surface ranking "
        "(CEHR-BERT best surface; DTR/CEHR similar AUPRC).",
        "- **Not supported:** Equating AUPRC gains with mechanism fidelity.",
        "- **Caption:** On the 32-target S2 benchmark, models with similar predictive "
        "AUPRC can differ substantially in age×lag Surface RMSE, separating prediction "
        "from counterfactual fidelity.",
        "- **Recommendation:** Main paper.",
        "",
        "## Figure: `dtr_gain_across_scenarios`",
        "",
        "- **Data source:** CF metrics from `dtr_*_new` result.json.",
        "- **Plotted:** Δ = temporal-only error − DTR error for CF-age/lag/Surface.",
        f"- **Quantitative:** S2 ΔSurface={s2_to['Surface_RMSE']-s2_at['Surface_RMSE']:.3f}; "
        f"S0 ΔSurface={by[('S0','dtr_temporal_only_new')]['Surface_RMSE']-by[('S0','dtr_age_temporal_new')]['Surface_RMSE']:.3f}; "
        f"S3 ΔSurface={by[('S3','dtr_temporal_only_new')]['Surface_RMSE']-by[('S3','dtr_age_temporal_new')]['Surface_RMSE']:.3f}.",
        "- **Supported:** Clearest CF gains on S2/S5; near-zero or mixed on S0/S1/S3.",
        "- **Not supported:** Uniform improvement across all scenarios.",
        "- **Caption:** Age-temporal DTR improves counterfactual errors over temporal-only "
        "primarily when an age×lag interaction is planted (S2/S5).",
        "- **Recommendation:** Main paper or appendix depending on space.",
        "",
        "## Figure: `s5_persistence_recovery`",
        "",
        "- **Data source:** S5 class-specific Surface RMSE in result.json.",
        f"- **Quantitative:** DTR acute/inter/chronic="
        f"{s2_at and by[('S5','dtr_age_temporal_new')]['S5_Surface_RMSE_acute']:.3f}/"
        f"{by[('S5','dtr_age_temporal_new')]['S5_Surface_RMSE_intermediate']:.3f}/"
        f"{by[('S5','dtr_age_temporal_new')]['S5_Surface_RMSE_chronic']:.3f}; "
        f"order_ok={by[('S5','dtr_age_temporal_new')]['persistence_order_correct']}.",
        "- **Supported:** DTR lowers class-wise Surface RMSE vs temporal-only; "
        "persistence order flag true for both arms under the black-box surface test.",
        "- **Not supported:** Claiming unique recovery of heterogeneous persistence "
        "parameters (global β model).",
        "- **Caption:** On S5, age-temporal DTR improves class-specific Surface RMSE "
        "relative to temporal-only while preserving the black-box persistence ordering check.",
        "- **Recommendation:** Appendix (S5).",
        "",
        "## Figure: `parameter_vs_functional_recovery`",
        "",
        "- **Data source:** Existing Transformer S2 metrics "
        "(`arch_S2_age_temporal_.../metrics.json`), canonical factorized DTR metrics, "
        "and baseline `_new` diagnostics. No fabricated points.",
        "- **Supported:** High λ-correlation can coexist with weak β=0/age-shuffle ΔBCE "
        "(Transformer and baseline `_new`); canonical factorized run shows both high "
        "corr and large functional ΔBCE.",
        "- **Not supported:** Equating parameter-curve recovery with functional reliance.",
        "- **Caption:** Lambda-curve agreement with the oracle does not imply functional "
        "reliance on the age×lag pathway; β=0 and age-shuffle ΔBCE separate apparent "
        "parameter recovery from mechanism use.",
        "- **Recommendation:** Main paper (methods/results distinction) or appendix.",
        "",
        "## NCH figures",
        "",
        "- **Skipped.** See `nch_final_analysis.md`. Leakage-corrected matched "
        "`dtr_age_temporal_new` + finished `dtr_temporal_only_new` results are not present.",
        "",
        "## Capacity",
        "",
        "- See `model_capacity.csv`. Counting convention: "
        "`baselines.common.capacity_report.count_parameters` / stored `model_card`. "
        "**No frozen text/code embeddings** in these synthetic baselines; embedding "
        "parameters are included in trainable counts.",
        "",
        "## S0 caveat (do not reinterpret as false interaction)",
        "",
    ]
    for s in (diag.get("s0_analysis") or {}).get("interpretation", []):
        lines.append(f"- {s}")
    lines += [
        "",
        "## Overall wording guidance",
        "",
        "Prefer: DTR shows **partial** alignment with the planted age×lag pathway "
        "(signed β̂, modest intervention ΔBCE, improved CF errors vs temporal-only on S2/S5). "
        "Avoid: unqualified “mechanism recovery” for the baseline `_new` checkpoint, "
        "given small β=0 ΔBCE and CEHR-BERT’s superior Surface RMSE.",
        "",
    ]
    (OUT / "figure_analysis_report.md").write_text("\n".join(lines))
    print("wrote figure_analysis_report.md")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    device = get_device("cuda")
    print("device", device)

    print("=== 1 provenance ===")
    write_provenance()

    print("=== 2 diagnostics ===")
    diag = compute_diagnostics(device)

    print("=== 3 surfaces ===")
    surface_cache = regenerate_surfaces(device)

    print("=== 8 capacity ===")
    capacity_rows = write_capacity_table()

    print("=== figures ===")
    fig_surface_comparison(surface_cache)
    fig_functional_recovery(diag)
    fig_prediction_vs_surface(capacity_rows)
    fig_dtr_gain(diag)
    fig_param_vs_functional(diag)

    print("=== 9 NCH ===")
    write_nch_skip()

    print("=== 10 report ===")
    write_report(diag, capacity_rows)
    print("DONE")


if __name__ == "__main__":
    main()
