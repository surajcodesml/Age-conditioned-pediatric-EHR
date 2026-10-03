"""Load and merge architecture-ladder configs. Seeds and schedules are predefined."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ladder import CONFIG_DIR

EXPERIMENT_FILES = (
    "e01_direct.yaml",
    "e02_staged.yaml",
    "e03_mass.yaml",
    "e04_channels.yaml",
    "e05_mixture.yaml",
    "e06_integrated_hazard.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return data


def resolve_config(experiment_file: str | Path, *, config_dir: Path | None = None) -> dict[str, Any]:
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    base = load_yaml(directory / "base.yaml")
    exp_path = Path(experiment_file)
    if not exp_path.is_file():
        exp_path = directory / experiment_file
    exp = load_yaml(exp_path)
    merged = dict(base)
    merged.update(exp)
    merged["config_path"] = str(exp_path.resolve())
    merged["base_config_path"] = str((directory / "base.yaml").resolve())
    merged["config_hash"] = config_hash(merged)
    return merged


def config_hash(cfg: dict[str, Any]) -> str:
    payload = {k: v for k, v in cfg.items() if k not in {"config_hash"}}
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def experiment_configs(config_dir: Path | None = None) -> list[dict[str, Any]]:
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return [resolve_config(name, config_dir=directory) for name in EXPERIMENT_FILES]
