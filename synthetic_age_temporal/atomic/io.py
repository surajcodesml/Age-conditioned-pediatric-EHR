"""Write-once prediction tables for the atomic follow-up."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_predictions(
    path: Path,
    *,
    example_id: np.ndarray,
    patient_id: list[str],
    age: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    logits_beta0: np.ndarray,
    logits_full_age_shuffle: np.ndarray,
    logits_gate_age_shuffle: np.ndarray,
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    def column(array: np.ndarray) -> pa.Array:
        values = np.asarray(array, dtype=np.float32)
        return pa.array([row.tolist() for row in values], type=pa.list_(pa.float32()))

    table = pa.table({
        "example_id": pa.array(np.asarray(example_id, dtype=np.int64)),
        "patient_id": pa.array([str(x) for x in patient_id]),
        "age": pa.array(np.asarray(age, dtype=np.float32)),
        "split": pa.array(["test"] * len(patient_id)),
        "labels": column(labels),
        "logits": column(logits),
        "logits_beta0": column(logits_beta0),
        "logits_full_age_shuffle": column(logits_full_age_shuffle),
        "logits_gate_age_shuffle": column(logits_gate_age_shuffle),
    })
    pq.write_table(table, path)


def read_predictions(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)

    def matrix(name: str) -> np.ndarray:
        return np.asarray(table.column(name).to_pylist(), dtype=np.float64)

    return {
        "example_id": np.asarray(table.column("example_id").to_pylist(), dtype=np.int64),
        "patient_id": [str(x) for x in table.column("patient_id").to_pylist()],
        "age": np.asarray(table.column("age").to_pylist(), dtype=np.float64),
        "labels": matrix("labels"),
        "logits": matrix("logits"),
        "logits_beta0": matrix("logits_beta0"),
        "logits_full_age_shuffle": matrix("logits_full_age_shuffle"),
        "logits_gate_age_shuffle": matrix("logits_gate_age_shuffle"),
    }
