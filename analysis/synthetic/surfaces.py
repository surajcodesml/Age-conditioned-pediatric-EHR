"""Age × lag probability surfaces, P̂(Y | a, τ), mean over targets.

Default source is the forward-pass grid already stored in
``analysis/final_results/surface_grids_s2_s3.json`` (same protocol as the
32-target S2 counterfactual surfaces). ``recompute=True`` rebuilds the grids
from checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CACHE_DIR = Path(__file__).resolve().parent / "cache"
SOURCE_JSON = REPO / "analysis" / "final_results" / "surface_grids_s2_s3.json"

MODEL_KEYS = (
    "oracle",
    "dtr_age_temporal_new",
    "dtr_temporal_only_new",
    "cehrbert",
)


def _from_source_json(scenario: str) -> dict[str, Any]:
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(SOURCE_JSON)
    blob = json.loads(SOURCE_JSON.read_text())
    if scenario not in blob:
        raise KeyError(f"{scenario} not in {SOURCE_JSON}")
    block = blob[scenario]
    surfaces = {k: np.asarray(block["surfaces"][k], dtype=np.float64) for k in MODEL_KEYS}
    return {
        "scenario": scenario,
        "ages": np.asarray(block["ages"], dtype=np.float64),
        "lags": np.asarray(block["lags"], dtype=np.float64),
        "surfaces": surfaces,
        "surface_rmse": {k: float(v) for k, v in block.get("surface_rmse", {}).items()},
        "source": str(SOURCE_JSON),
    }


def _recompute(scenario: str, device: str) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(REPO / "synthetic_age_temporal"))
    from analysis.final_results.build_final_paper_artifacts import (  # noqa: WPS433
        build_model_surface,
        _select_templates,
    )

    dev = torch.device(device)
    template, dtr_template, _vocab, info, specs, meta, itos = _select_templates(scenario, dev)
    surfaces = {}
    rmses = {}
    for name in MODEL_KEYS:
        print(f"surface {scenario} {name}...", flush=True)
        grid, srmse = build_model_surface(
            name, scenario, dev, template, dtr_template, info, specs, meta, itos
        )
        surfaces[name] = np.asarray(grid, dtype=np.float64)
        rmses[name] = float(srmse)
    from baselines.common.counterfactual import SURFACE_AGES, SURFACE_LAGS_DAYS

    return {
        "scenario": scenario,
        "ages": np.asarray(SURFACE_AGES, dtype=np.float64),
        "lags": np.asarray(SURFACE_LAGS_DAYS, dtype=np.float64),
        "surfaces": surfaces,
        "surface_rmse": rmses,
        "source": "recomputed",
    }


def _cache_path(scenario: str) -> Path:
    return CACHE_DIR / f"surfaces_{scenario}.npz"


def save_cache(pack: dict[str, Any]) -> Path:
    path = _cache_path(pack["scenario"])
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"p_{k}": v for k, v in pack["surfaces"].items()}
    np.savez_compressed(
        path,
        ages=pack["ages"],
        lags=pack["lags"],
        source=np.asarray(pack["source"]),
        **arrays,
    )
    meta = {
        "scenario": pack["scenario"],
        "surface_rmse": pack["surface_rmse"],
        "source": pack["source"],
        "models": list(pack["surfaces"]),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load_cache(scenario: str) -> dict[str, Any]:
    path = _cache_path(scenario)
    z = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    surfaces = {k: np.asarray(z[f"p_{k}"], dtype=np.float64) for k in MODEL_KEYS}
    return {
        "scenario": scenario,
        "ages": np.asarray(z["ages"], dtype=np.float64),
        "lags": np.asarray(z["lags"], dtype=np.float64),
        "surfaces": surfaces,
        "surface_rmse": meta.get("surface_rmse", {}),
        "source": meta.get("source", str(path)),
    }


def get_surfaces(scenario: str = "S2", *, recompute: bool = False, device: str = "cpu") -> dict[str, Any]:
    path = _cache_path(scenario)
    if path.exists() and path.with_suffix(".json").exists() and not recompute:
        print(f"using cached surfaces {path}")
        return load_cache(scenario)
    if not recompute and SOURCE_JSON.exists():
        pack = _from_source_json(scenario)
    else:
        pack = _recompute(scenario, device)
    save_cache(pack)
    return pack


def probability_limits(pack: dict[str, Any]) -> tuple[float, float]:
    """Shared color limits over oracle, DTR, temporal-only, and CEHR-BERT.

    Panels are not normalized independently.
    """
    stacked = np.stack([pack["surfaces"][k] for k in MODEL_KEYS], axis=0)
    return float(stacked.min()), float(stacked.max())


def residuals(pack: dict[str, Any]) -> dict[str, np.ndarray]:
    oracle = pack["surfaces"]["oracle"]
    return {
        "dtr": pack["surfaces"]["dtr_age_temporal_new"] - oracle,
        "cehrbert": pack["surfaces"]["cehrbert"] - oracle,
    }


def residual_limit(pack: dict[str, Any]) -> float:
    """Shared symmetric limit for DTR and CEHR-BERT residuals."""
    res = residuals(pack)
    return float(max(np.max(np.abs(res["dtr"])), np.max(np.abs(res["cehrbert"])), 1e-6))
