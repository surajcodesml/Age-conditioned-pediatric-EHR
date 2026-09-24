"""Future S5 figure hooks (do not generate final figures from the runner).

Call ``prepare_s5_persistence_curve_data`` / ``plot_s5_persistence_curves``
after baseline training + counterfactual eval to produce the paper figure
comparing true vs predicted persistence curves for acute / intermediate /
chronic history.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from baselines.synthetic.data_adapter import S5_GROUP_THETA, S5_PERSISTENCE_GROUPS
from baselines.synthetic.s5_eval import (
    make_controlled_oracle_surface_fn,
    make_controlled_predict_surface_fn,
)


def prepare_s5_persistence_curve_data(
    predict_fn: Callable,
    batch,
    itos: dict[int, str],
    specs: list[dict[str, Any]],
    theta0: float,
    beta: float,
    *,
    age: float = 9.0,
    lags_days: tuple[float, ...] = (7.0, 30.0, 90.0, 180.0, 365.0, 730.0),
) -> dict[str, Any]:
    """Collect true vs predicted mean probability vs lag per persistence group.

    Returns a serializable dict suitable for later plotting — no files written.
    """
    n_targets = int(batch["labels"].shape[-1])
    curves: dict[str, Any] = {
        "age": age,
        "lags_days": list(lags_days),
        "groups": {},
        "group_theta": dict(S5_GROUP_THETA),
        "persistence_groups": {g: list(c) for g, c in S5_PERSISTENCE_GROUPS.items()},
    }
    for group in ("acute", "intermediate", "chronic"):
        o_fn = make_controlled_oracle_surface_fn(
            group, specs, theta0, beta, scenario="S5",
        )
        p_fn = make_controlled_predict_surface_fn(
            predict_fn, group, itos, n_targets,
        )
        true_curve, pred_curve = [], []
        for lag in lags_days:
            true_curve.append(float(np.mean(o_fn(age, lag))))
            pred_curve.append(float(np.mean(p_fn(age, lag))))
        curves["groups"][group] = {
            "true_mean_prob": true_curve,
            "pred_mean_prob": pred_curve,
            "theta": S5_GROUP_THETA[group],
        }
    return curves


def plot_s5_persistence_curves(
    curve_data: dict[str, Any],
    out_path: Path | str | None = None,
    *,
    show: bool = False,
):
    """Render true vs predicted persistence curves (optional; not called by runner).

    Intentionally not invoked during implementation-only benchmark setup.
    """
    import matplotlib.pyplot as plt

    lags = curve_data["lags_days"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    styles = {
        "acute": ("#c0392b", "-"),
        "intermediate": ("#2980b9", "--"),
        "chronic": ("#27ae60", "-."),
    }
    for group, (color, ls) in styles.items():
        g = curve_data["groups"][group]
        ax.plot(lags, g["true_mean_prob"], color=color, ls=ls, lw=2.0,
                label=f"{group} true (θ={g['theta']})")
        ax.plot(lags, g["pred_mean_prob"], color=color, ls=ls, lw=1.5, alpha=0.55,
                marker="o", markersize=3, label=f"{group} pred")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Mean predicted probability")
    ax.set_title(
        f"S5 persistence curves (age={curve_data['age']})\n"
        "acute decays fastest → chronic slowest"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        fig.savefig(out_path.with_suffix(".svg"))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
