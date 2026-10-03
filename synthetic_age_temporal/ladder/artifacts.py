"""Checkpoint, manifest, and prediction IO. Predictions are write-once."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from ladder import REPO_ROOT


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_directory(
    root: Path,
    experiment_id: str,
    scenario: str,
    arm: str,
    seed: int,
) -> Path:
    return Path(root) / experiment_id / scenario.lower() / arm / f"seed_{seed}"


def stage_a_directory(root: Path, experiment_id: str, scenario: str, seed: int) -> Path:
    return Path(root) / experiment_id / scenario.lower() / "_stage_a" / f"seed_{seed}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_config_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    cfg: dict[str, Any],
    n_codes: int,
    n_targets: int,
    age_temporal: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": cfg,
            "n_codes": int(n_codes),
            "n_targets": int(n_targets),
            "age_temporal": bool(age_temporal),
            "architecture": getattr(model, "architecture", cfg.get("architecture")),
        },
        path,
    )


def write_predictions(
    path: Path,
    *,
    example_id: np.ndarray,
    patient_id: list[str],
    age: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    logits_beta0: np.ndarray,
    logits_age_shuffle: np.ndarray,
) -> None:
    """Write the test-set prediction table. Refuses to overwrite an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    def column(array: np.ndarray) -> pa.Array:
        values = np.asarray(array, dtype=np.float32)
        return pa.array([row.tolist() for row in values], type=pa.list_(pa.float32()))

    table = pa.table(
        {
            "example_id": pa.array(np.asarray(example_id, dtype=np.int64)),
            "patient_id": pa.array([str(x) for x in patient_id]),
            "age": pa.array(np.asarray(age, dtype=np.float32)),
            "split": pa.array(["test"] * len(patient_id)),
            "labels": column(labels),
            "logits": column(logits),
            "logits_beta0": column(logits_beta0),
            "logits_age_shuffle": column(logits_age_shuffle),
        }
    )
    pq.write_table(table, path)


def read_predictions(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)

    def matrix(name: str) -> np.ndarray:
        rows = table.column(name).to_pylist()
        return np.asarray(rows, dtype=np.float64)

    return {
        "example_id": np.asarray(table.column("example_id").to_pylist(), dtype=np.int64),
        "patient_id": [str(x) for x in table.column("patient_id").to_pylist()],
        "age": np.asarray(table.column("age").to_pylist(), dtype=np.float64),
        "labels": matrix("labels"),
        "logits": matrix("logits"),
        "logits_beta0": matrix("logits_beta0"),
        "logits_age_shuffle": matrix("logits_age_shuffle"),
    }
