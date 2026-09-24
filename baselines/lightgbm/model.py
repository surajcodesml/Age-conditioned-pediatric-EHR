"""Count + LightGBM non-neural baseline.

Features (primary mode — no handcrafted age×lag interaction):
  - Binary code presence over vocabulary
  - Code frequency counts
  - Age at prediction time
  - Demographics (sex, race where available)
  - Number of encounters
  - Total history duration (days)

One-vs-rest LightGBM classifiers for multi-label prediction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from baselines.common.interface import BaselineModel, ModelOutput
from baselines.common.registry import register_baseline


def build_features_from_batch(
    batch: dict[str, Any],
    n_codes: int,
) -> np.ndarray:
    """Convert a synthetic benchmark batch to LightGBM features.

    Features per example:
      [0]            age
      [1]            z_age
      [2:2+n_codes]  binary code presence
    """
    code_ids = batch["code_ids"]  # [B, L]
    if isinstance(code_ids, torch.Tensor):
        code_ids = code_ids.numpy()
    B, L = code_ids.shape
    age = batch["age"]
    if isinstance(age, torch.Tensor):
        age = age.numpy()
    z_age = batch.get("z_age", np.zeros(B))
    if isinstance(z_age, torch.Tensor):
        z_age = z_age.numpy()

    # Padding mask
    pad = batch.get("padding_mask", None)
    if pad is not None and isinstance(pad, torch.Tensor):
        pad = pad.numpy()

    # Binary presence + frequency
    presence = np.zeros((B, n_codes), dtype=np.float32)
    for i in range(B):
        for j in range(L):
            if pad is not None and pad[i, j]:
                continue
            cid = int(code_ids[i, j])
            if 0 < cid < n_codes:  # skip pad=0
                presence[i, cid] = 1.0

    features = np.column_stack([
        age.astype(np.float32).reshape(-1, 1),
        z_age.astype(np.float32).reshape(-1, 1),
        presence,
    ])
    return features


@register_baseline("count_lightgbm")
class LightGBMBaseline(BaselineModel):
    """One-vs-rest LightGBM with count features."""

    def __init__(self, n_codes: int, n_targets: int, **lgb_params: Any) -> None:
        self.n_codes = n_codes
        self.n_targets = n_targets
        self.lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 300,
            "min_child_samples": 20,
            "seed": 0,
            **lgb_params,
        }
        self.models: list[Any] = []  # one per target
        self._fitted = False

    @property
    def name(self) -> str:
        return "count_lightgbm"

    @property
    def has_age_input(self) -> bool:
        return True  # age is an explicit feature

    @property
    def has_time_input(self) -> bool:
        return False  # no explicit temporal features in primary mode

    @property
    def model_card(self) -> dict[str, Any]:
        return {
            "trainable_params": "N/A (tree model)",
            "embedding_params": 0,
            "layers": "N/A",
            "heads": "N/A",
            "hidden_size": "N/A",
            "ffn_size": "N/A",
            "max_seq_len": "unlimited",
            "n_targets": self.n_targets,
            "n_features": 2 + self.n_codes,
            "lgb_params": self.lgb_params,
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None:
        """Fit one-vs-rest LightGBM classifiers."""
        import lightgbm as lgb

        self.models = []
        for k in range(self.n_targets):
            yk = y_train[:, k]
            # Skip degenerate targets
            if len(np.unique(yk)) < 2:
                self.models.append(None)
                continue
            dtrain = lgb.Dataset(X_train, yk)
            callbacks = [lgb.log_evaluation(period=0)]
            if X_val is not None and y_val is not None:
                dval = lgb.Dataset(X_val, y_val[:, k], reference=dtrain)
                model = lgb.train(
                    self.lgb_params, dtrain,
                    num_boost_round=self.lgb_params.get("n_estimators", 300),
                    valid_sets=[dval],
                    callbacks=callbacks + [lgb.early_stopping(50, verbose=False)],
                )
            else:
                model = lgb.train(
                    self.lgb_params, dtrain,
                    num_boost_round=self.lgb_params.get("n_estimators", 300),
                    callbacks=callbacks,
                )
            self.models.append(model)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities [N, n_targets]."""
        preds = np.full((X.shape[0], self.n_targets), 0.5, dtype=np.float64)
        for k, model in enumerate(self.models):
            if model is not None:
                preds[:, k] = model.predict(X, num_iteration=model.best_iteration)
        return preds

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Predict log-odds [N, n_targets]."""
        p = np.clip(self.predict_proba(X), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def predict(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        X = build_features_from_batch(batch, self.n_codes)
        logits = torch.tensor(self.predict_logits(X), dtype=torch.float32)
        return ModelOutput(
            logits=logits,
            patient_repr=torch.tensor(X, dtype=torch.float32),
        )

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        raise NotImplementedError("LightGBM uses fit(), not training_step()")

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for k, model in enumerate(self.models):
            if model is not None:
                model.save_model(str(path / f"target_{k}.txt"))
        with (path / "config.json").open("w") as f:
            json.dump({"n_codes": self.n_codes, "n_targets": self.n_targets,
                        "lgb_params": self.lgb_params}, f)

    def load_checkpoint(self, path: Path) -> None:
        import lightgbm as lgb
        path = Path(path)
        with (path / "config.json").open() as f:
            cfg = json.load(f)
        self.n_codes = cfg["n_codes"]
        self.n_targets = cfg["n_targets"]
        self.models = []
        for k in range(self.n_targets):
            model_file = path / f"target_{k}.txt"
            if model_file.exists():
                self.models.append(lgb.Booster(model_file=str(model_file)))
            else:
                self.models.append(None)
        self._fitted = True
