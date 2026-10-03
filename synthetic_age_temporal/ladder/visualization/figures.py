"""Figures from saved surfaces and the summary table. Does not run models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ladder import REPO_ROOT
from ladder.evaluation.aggregate import iter_ladder_runs

REFERENCE_SURFACES = (
    REPO_ROOT / "analysis" / "final_results" / "surface_grids_s2_s3.json"
)


def _lag_labels(lags: np.ndarray) -> list[str]:
    labels = []
    for lag in lags:
        days = float(lag)
        if days == 0:
            labels.append("0")
        elif abs(days - 365) < 1:
            labels.append("1y")
        elif abs(days - 730) < 1:
            labels.append("2y")
        else:
            labels.append(str(int(days)))
    return labels


def _mean_surface(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    models, oracles = [], []
    ages = lags = None
    for run_dir in run_dirs:
        with np.load(run_dir / "mechanism_outputs.npz") as blob:
            models.append(np.asarray(blob["surface_model"], dtype=np.float64).mean(axis=-1))
            oracles.append(np.asarray(blob["surface_oracle"], dtype=np.float64).mean(axis=-1))
            ages = np.asarray(blob["surface_ages"], dtype=np.float64)
            lags = np.asarray(blob["surface_lags"], dtype=np.float64)
    return np.mean(models, axis=0), np.mean(oracles, axis=0), ages, lags


def _collect_panels(root: Path) -> dict[str, list[dict]]:
    grouped: dict[tuple, list[Path]] = {}
    for experiment, scenario, arm, _seed, run_dir in iter_ladder_runs(root):
        if scenario not in {"S2", "S3"}:
            continue
        if not (run_dir / "mechanism_outputs.npz").exists():
            continue
        grouped.setdefault((experiment, scenario, arm), []).append(run_dir)
    panels: dict[str, list[dict]] = {"S2": [], "S3": []}
    experiments = sorted({key[0] for key in grouped})
    for scenario in ("S2", "S3"):
        for experiment in experiments:
            at_key = (experiment, scenario, "age_temporal")
            to_key = (experiment, scenario, "temporal_only")
            if at_key not in grouped or to_key not in grouped:
                continue
            at, oracle, ages, lags = _mean_surface(grouped[at_key])
            to, _oracle_to, _, _ = _mean_surface(grouped[to_key])
            panels[scenario].append({
                "name": experiment,
                "oracle": oracle,
                "temporal_only": to,
                "age_temporal": at,
                "residual": at - oracle,
                "ages": ages,
                "lags": lags,
            })
    if REFERENCE_SURFACES.exists():
        blob = json.loads(REFERENCE_SURFACES.read_text())
        for scenario in ("S2", "S3"):
            block = blob.get(scenario)
            if not block:
                continue
            surfaces = block["surfaces"]
            oracle = np.asarray(surfaces["oracle"], dtype=np.float64)
            panels[scenario].insert(0, {
                "name": "E00_dtr_age_temporal_new",
                "oracle": oracle,
                "temporal_only": np.asarray(surfaces["dtr_temporal_only_new"], dtype=np.float64),
                "age_temporal": np.asarray(surfaces["dtr_age_temporal_new"], dtype=np.float64),
                "residual": np.asarray(surfaces["dtr_age_temporal_new"], dtype=np.float64) - oracle,
                "ages": np.asarray(block["ages"], dtype=np.float64),
                "lags": np.asarray(block["lags"], dtype=np.float64),
                "cehrbert": np.asarray(surfaces["cehrbert"], dtype=np.float64),
            })
    return panels


def _scales(panels: dict[str, list[dict]]) -> tuple[float, float, float]:
    probs = []
    residuals = []
    for rows in panels.values():
        for row in rows:
            probs.extend([row["oracle"], row["temporal_only"], row["age_temporal"]])
            if "cehrbert" in row:
                probs.append(row["cehrbert"])
            residuals.append(row["residual"])
    if not probs:
        return 0.0, 1.0, 1.0
    stacked = np.stack([np.asarray(p) for p in probs])
    resid = np.stack([np.asarray(r) for r in residuals])
    limit = float(max(np.max(np.abs(resid)), 1e-6))
    return float(stacked.min()), float(stacked.max()), limit


def plot_heatmaps(root: Path, fig_dir: Path) -> None:
    panels = _collect_panels(root)
    vmin, vmax, rlim = _scales(panels)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for scenario, rows in panels.items():
        if not rows:
            continue
        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, 4, figsize=(12, 2.4 * n_rows), squeeze=False)
        for i, row in enumerate(rows):
            grids = [
                (row["oracle"], "Oracle", "viridis", vmin, vmax),
                (row["temporal_only"], "Temporal-only", "viridis", vmin, vmax),
                (row["age_temporal"], "Age-temporal", "viridis", vmin, vmax),
                (row["residual"], "Age-temporal − oracle", "coolwarm", -rlim, rlim),
            ]
            for j, (grid, title, cmap, lo, hi) in enumerate(grids):
                ax = axes[i, j]
                im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_title(f"{row['name']}: {title}", fontsize=8)
                lags = _lag_labels(row["lags"])
                ax.set_xticks(range(len(lags)))
                ax.set_xticklabels(lags, fontsize=7, rotation=45)
                ages = row["ages"]
                idx = list(range(0, len(ages), 3))
                ax.set_yticks(idx)
                ax.set_yticklabels([str(int(ages[k])) for k in idx], fontsize=7)
                if j == 0:
                    ax.set_ylabel("Age (years)")
                ax.set_xlabel("Lag")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{scenario} mean P(Y | age, lag). Shared probability scale [{vmin:.3f}, {vmax:.3f}]; "
            f"shared residual scale ±{rlim:.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"heatmap_{scenario}.png", dpi=160)
        fig.savefig(fig_dir / f"heatmap_{scenario}.svg")
        plt.close(fig)


def _summary_lookup(root: Path) -> dict[tuple, dict]:
    path = root / "architecture_ladder_summary.csv"
    if not path.exists():
        return {}
    import csv
    out = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out[(row["experiment"], row["scenario"], row["arm"], row["metric"])] = row
    return out


def _bar(ax, labels, means, stds, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)


def plot_metric_bars(root: Path, fig_dir: Path) -> None:
    table = _summary_lookup(root)
    if not table:
        return
    experiments = []
    for key in table:
        if key[0].startswith("E0") or key[0].startswith("CEHR"):
            continue
        if key[0] not in experiments:
            experiments.append(key[0])
    # Keep ladder experiments plus the current DTR reference on the mechanism plots.
    order = ["E00_dtr_age_temporal_new", "CEHR-BERT", *experiments]
    fig_dir.mkdir(parents=True, exist_ok=True)

    def series(scenario, arm, metric):
        labels, means, stds = [], [], []
        for name in order:
            row = table.get((name, scenario, arm, metric))
            if row is None and name == "CEHR-BERT":
                row = table.get((name, scenario, "cehrbert", metric))
            if row is None or row.get("mean", "") == "":
                continue
            labels.append(name.replace("E00_dtr_age_temporal_new", "E00"))
            means.append(float(row["mean"]))
            stds.append(float(row["std"]) if row.get("std") not in ("", None) else 0.0)
        return labels, means, stds

    mech_metrics = [
        ("surface_rmse", "Surface RMSE"),
        ("cf_rmse_age", "CF-RMSE age"),
        ("cf_rmse_lag", "CF-RMSE lag"),
        ("delta_bce_beta0", "beta=0 ΔBCE"),
        ("delta_bce_age_shuffle", "age-shuffle ΔBCE"),
    ]
    fig, axes = plt.subplots(1, len(mech_metrics), figsize=(3.2 * len(mech_metrics), 4.2), squeeze=False)
    for ax, (metric, title) in zip(axes[0], mech_metrics):
        labels, means, stds = series("S2", "age_temporal", metric)
        if labels:
            _bar(ax, labels, means, stds, f"S2 {title}", title)
    fig.tight_layout()
    fig.savefig(fig_dir / "mechanism_metrics_s2.png", dpi=160)
    fig.savefig(fig_dir / "mechanism_metrics_s2.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, metric, title in zip(axes, ("auroc", "auprc", "bce"), ("AUROC", "AUPRC", "BCE")):
        labels, means, stds = series("S2", "age_temporal", metric)
        if labels:
            _bar(ax, labels, means, stds, f"S2 age-temporal {title}", title)
    fig.tight_layout()
    fig.savefig(fig_dir / "predictive_metrics_s2.png", dpi=160)
    fig.savefig(fig_dir / "predictive_metrics_s2.svg")
    plt.close(fig)

    neg_metrics = [
        ("abs_beta_mean", "|beta|"),
        ("delta_bce_beta0", "beta=0 ΔBCE"),
        ("delta_bce_age_shuffle", "age-shuffle ΔBCE"),
        ("surface_rmse", "Surface RMSE"),
        ("cf_rmse_age", "CF-RMSE age"),
    ]
    fig, axes = plt.subplots(2, len(neg_metrics), figsize=(3.1 * len(neg_metrics), 7), squeeze=False)
    for row_i, scenario in enumerate(("S0", "S1")):
        for ax, (metric, title) in zip(axes[row_i], neg_metrics):
            labels, means, stds = series(scenario, "age_temporal", metric)
            if labels:
                _bar(ax, labels, means, stds, f"{scenario} {title}", title)
    fig.tight_layout()
    fig.savefig(fig_dir / "negative_controls_s0_s1.png", dpi=160)
    fig.savefig(fig_dir / "negative_controls_s0_s1.svg")
    plt.close(fig)


def plot_lambda(root: Path, fig_dir: Path) -> None:
    grouped: dict[tuple, list[Path]] = {}
    for experiment, scenario, arm, _seed, run_dir in iter_ladder_runs(root):
        if scenario not in {"S2", "S3"} or arm != "age_temporal":
            continue
        grouped.setdefault((scenario, experiment), []).append(run_dir)
    if not grouped:
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, scenario in zip(axes, ("S2", "S3")):
        truth = None
        ages = None
        for (scen, experiment), runs in sorted(grouped.items()):
            if scen != scenario:
                continue
            curves = []
            for run_dir in runs:
                with np.load(run_dir / "mechanism_outputs.npz") as blob:
                    if "lambda_age" not in blob.files:
                        continue
                    curves.append(np.asarray(blob["lambda_age"], dtype=np.float64))
                    ages = np.asarray(blob["surface_ages"], dtype=np.float64)
                    truth = np.asarray(blob["lambda_true_age"], dtype=np.float64)
            if not curves or ages is None:
                continue
            stack = np.stack(curves)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            ax.plot(ages, mean, label=experiment)
            ax.fill_between(ages, mean - std, mean + std, alpha=0.15)
        if truth is not None and ages is not None:
            ax.plot(ages, truth, color="black", linestyle="--", label="oracle lambda(a)")
        ax.set_title(f"{scenario} lambda(a)")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("lambda")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "lambda_age.png", dpi=160)
    fig.savefig(fig_dir / "lambda_age.svg")
    plt.close(fig)

    # E06 rho is not lambda(a). Plot it separately when present.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    drew = False
    for ax, scenario in zip(axes, ("S2", "S3")):
        runs = grouped.get((scenario, "E06_integrated_hazard"), [])
        curves = []
        ages = None
        for run_dir in runs:
            with np.load(run_dir / "mechanism_outputs.npz") as blob:
                if "rho_age" not in blob.files:
                    continue
                curves.append(np.asarray(blob["rho_age"], dtype=np.float64))
                ages = np.asarray(blob["surface_ages"], dtype=np.float64)
        if not curves or ages is None:
            ax.set_axis_off()
            continue
        drew = True
        stack = np.stack(curves)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(ages, mean, color="darkred")
        ax.fill_between(ages, mean - std, mean + std, color="darkred", alpha=0.15)
        ax.set_title(f"{scenario} E06 rho(a)")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("rho")
    if drew:
        fig.tight_layout()
        fig.savefig(fig_dir / "e06_rho_age.png", dpi=160)
        fig.savefig(fig_dir / "e06_rho_age.svg")
    plt.close(fig)


def plot_run_curves(root: Path) -> None:
    for _experiment, _scenario, _arm, _seed, run_dir in iter_ladder_runs(root):
        npz = run_dir / "mechanism_outputs.npz"
        if not npz.exists():
            continue
        with np.load(npz) as blob:
            ages = np.asarray(blob["surface_ages"], dtype=np.float64)
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            if "lambda_age" in blob.files:
                ax.plot(ages, np.asarray(blob["lambda_age"]), label="learned")
                if "lambda_true_age" in blob.files:
                    ax.plot(ages, np.asarray(blob["lambda_true_age"]), linestyle="--", label="oracle")
                ax.set_ylabel("lambda")
            elif "rho_age" in blob.files:
                ax.plot(ages, np.asarray(blob["rho_age"]), color="darkred", label="rho")
                ax.set_ylabel("rho")
            ax.set_xlabel("Age (years)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            out = run_dir / "plots"
            out.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out / "lambda_or_rho.png", dpi=120)
            plt.close(fig)


def make_all_figures(root: Path) -> Path:
    fig_dir = Path(root) / "figures"
    plot_heatmaps(root, fig_dir)
    plot_metric_bars(root, fig_dir)
    plot_lambda(root, fig_dir)
    plot_run_curves(root)
    return fig_dir
