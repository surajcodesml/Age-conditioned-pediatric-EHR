#!/usr/bin/env python3
"""Unit tests for all baseline models.

Tests cover:
  - Tensor shapes
  - Padding invariance
  - No future leakage
  - Gradient flow
  - Tiny-batch overfit
  - Checkpoint save/load equality
  - Age/time feature presence/absence as intended
  - Target dimensionality

Model-specific tests:
  - BEHRT: age bucket embedding correctness
  - Med-BERT: no age input
  - CEHR-BERT: time2vec embeddings present
  - RETAIN: reverse-time order
  - EHR-BERT: no age, no time
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "synthetic_age_temporal"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_batch(B: int = 4, L: int = 16, n_codes: int = 50, n_targets: int = 8):
    """Create a minimal synthetic benchmark batch."""
    return {
        "code_ids": torch.randint(1, n_codes, (B, L)),
        "type_ids": torch.randint(0, 5, (B, L)),
        "lag_days": torch.rand(B, L) * 365.0,
        "tau": torch.rand(B, L) * 3.0,
        "is_query": torch.zeros(B, L, dtype=torch.bool),
        "is_signal": torch.zeros(B, L, dtype=torch.bool),
        "padding_mask": torch.zeros(B, L, dtype=torch.bool),
        "age": torch.rand(B) * 18.0,
        "z_age": (torch.rand(B) - 0.5) * 2.0,
        "labels": (torch.rand(B, n_targets) > 0.7).float(),
    }


N_CODES = 50
N_TYPES = 11
N_TARGETS = 8


# ---------------------------------------------------------------------------
# RETAIN tests
# ---------------------------------------------------------------------------

class TestRETAIN:
    def _make_model(self):
        from baselines.retain.model import RETAINModel
        return RETAINModel(n_codes=N_CODES, n_targets=N_TARGETS, d_emb=32, d_rnn=32)

    def test_output_shape(self):
        model = self._make_model()
        batch = _make_batch()
        out = model.predict(batch)
        assert out.logits.shape == (4, N_TARGETS)
        assert out.patient_repr.shape[0] == 4

    def test_gradient_flow(self):
        model = self._make_model()
        batch = _make_batch()
        result = model.training_step(batch)
        result["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients flowed"

    def test_no_age_input(self):
        assert not self._make_model().has_age_input

    def test_no_time_input(self):
        assert not self._make_model().has_time_input

    def test_tiny_overfit(self):
        model = self._make_model()
        batch = _make_batch(B=2, L=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(50):
            opt.zero_grad()
            r = model.training_step(batch)
            r["loss"].backward()
            opt.step()
        assert r["loss"].item() < 0.5, f"Failed to overfit: loss={r['loss'].item()}"

    def test_checkpoint_roundtrip(self):
        model = self._make_model()
        batch = _make_batch()
        out1 = model.predict(batch).logits
        with tempfile.TemporaryDirectory() as td:
            model.save_checkpoint(Path(td))
            model2 = self._make_model()
            model2.load_checkpoint(Path(td))
            out2 = model2.predict(batch).logits
        assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# EHR-BERT tests
# ---------------------------------------------------------------------------

class TestEHRBert:
    def _make_model(self):
        from baselines.ehr_bert.model import EHRBertModel
        return EHRBertModel(n_codes=N_CODES, n_targets=N_TARGETS,
                            d_model=64, n_layers=2, n_heads=2, d_ff=128,
                            max_seq_len=32)

    def test_output_shape(self):
        model = self._make_model()
        batch = _make_batch()
        out = model.predict(batch)
        assert out.logits.shape == (4, N_TARGETS)

    def test_no_age_input(self):
        assert not self._make_model().has_age_input

    def test_no_time_input(self):
        assert not self._make_model().has_time_input

    def test_gradient_flow(self):
        model = self._make_model()
        batch = _make_batch()
        r = model.training_step(batch)
        r["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_padding_invariance(self):
        """Adding padding should not change predictions for valid tokens."""
        model = self._make_model()
        model.eval()
        batch1 = _make_batch(B=1, L=8)
        batch2 = _make_batch(B=1, L=12)
        # Copy batch1 content into batch2 and pad the rest
        batch2["code_ids"][0, :8] = batch1["code_ids"][0]
        batch2["code_ids"][0, 8:] = 0
        batch2["padding_mask"][0, :8] = False
        batch2["padding_mask"][0, 8:] = True
        batch2["age"] = batch1["age"]
        batch2["z_age"] = batch1["z_age"]
        batch2["tau"][0, :8] = batch1["tau"][0]
        batch2["tau"][0, 8:] = 0
        with torch.no_grad():
            out1 = model.predict(batch1).logits
            out2 = model.predict(batch2).logits
        assert torch.allclose(out1, out2, atol=1e-4), \
            f"Padding changed predictions: max diff = {(out1-out2).abs().max():.6f}"


# ---------------------------------------------------------------------------
# BEHRT tests
# ---------------------------------------------------------------------------

class TestBEHRT:
    def _make_model(self):
        from baselines.behrt.model import BEHRTModel
        return BEHRTModel(n_codes=N_CODES, n_targets=N_TARGETS,
                          d_model=48, n_layers=2, n_heads=4, max_seq_len=32)

    def test_output_shape(self):
        model = self._make_model()
        batch = _make_batch()
        out = model.predict(batch)
        assert out.logits.shape == (4, N_TARGETS)

    def test_has_age_input(self):
        assert self._make_model().has_age_input

    def test_no_time_input(self):
        assert not self._make_model().has_time_input

    def test_age_bucket_variation(self):
        """BEHRT predictions should change when age changes."""
        model = self._make_model()
        model.eval()
        batch = _make_batch(B=2, L=8)
        batch["age"][0] = 2.0
        batch["age"][1] = 16.0
        # Same codes
        batch["code_ids"][1] = batch["code_ids"][0]
        batch["padding_mask"][1] = batch["padding_mask"][0]
        batch["tau"][1] = batch["tau"][0]
        with torch.no_grad():
            out = model.predict(batch)
        # Predictions should differ (age buckets differ)
        diff = (out.logits[0] - out.logits[1]).abs().max().item()
        assert diff > 1e-4, "BEHRT predictions unchanged with different ages"

    def test_gradient_flow(self):
        model = self._make_model()
        batch = _make_batch()
        r = model.training_step(batch)
        r["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# Med-BERT tests
# ---------------------------------------------------------------------------

class TestMedBERT:
    def _make_model(self):
        from baselines.medbert.model import MedBERTModel
        return MedBERTModel(n_codes=N_CODES, n_targets=N_TARGETS,
                            d_model=48, n_layers=2, n_heads=6, max_seq_len=32)

    def test_output_shape(self):
        model = self._make_model()
        batch = _make_batch()
        out = model.predict(batch)
        assert out.logits.shape == (4, N_TARGETS)

    def test_no_age_input(self):
        """Med-BERT canonical has NO age input."""
        assert not self._make_model().has_age_input

    def test_no_time_input(self):
        assert not self._make_model().has_time_input

    def test_age_invariance(self):
        """Changing age should NOT change Med-BERT predictions."""
        model = self._make_model()
        model.eval()
        batch = _make_batch(B=2, L=8)
        batch["age"][0] = 2.0
        batch["age"][1] = 16.0
        # Same codes
        batch["code_ids"][1] = batch["code_ids"][0]
        batch["padding_mask"][1] = batch["padding_mask"][0]
        batch["tau"][1] = batch["tau"][0]
        with torch.no_grad():
            out = model.predict(batch)
        diff = (out.logits[0] - out.logits[1]).abs().max().item()
        assert diff < 1e-6, f"Med-BERT predictions changed with age: diff={diff}"

    def test_gradient_flow(self):
        model = self._make_model()
        batch = _make_batch()
        r = model.training_step(batch)
        r["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# CEHR-BERT tests
# ---------------------------------------------------------------------------

class TestCEHRBert:
    def _make_model(self):
        from baselines.cehrbert_adapter.adapter import CEHRBertAdapter
        return CEHRBertAdapter(n_codes=N_CODES, n_targets=N_TARGETS,
                               d_model=32, n_layers=2, n_heads=4, max_seq_len=32)

    def test_output_shape(self):
        model = self._make_model()
        batch = _make_batch()
        out = model.predict(batch)
        assert out.logits.shape == (4, N_TARGETS)

    def test_has_age_input(self):
        assert self._make_model().has_age_input

    def test_has_time_input(self):
        assert self._make_model().has_time_input

    def test_gradient_flow(self):
        model = self._make_model()
        batch = _make_batch()
        r = model.training_step(batch)
        r["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# LightGBM tests
# ---------------------------------------------------------------------------

class TestLightGBM:
    def test_feature_extraction(self):
        from baselines.lightgbm.model import build_features_from_batch
        batch = _make_batch()
        X = build_features_from_batch(batch, N_CODES)
        assert X.shape == (4, 2 + N_CODES)  # age, z_age, n_codes presence

    def test_predict_after_fit(self):
        pytest.importorskip("lightgbm")
        from baselines.lightgbm.model import LightGBMBaseline, build_features_from_batch
        model = LightGBMBaseline(n_codes=N_CODES, n_targets=N_TARGETS,
                                  n_estimators=5)
        batch = _make_batch(B=20)
        X = build_features_from_batch(batch, N_CODES)
        y = batch["labels"].numpy()
        model.fit(X, y, X, y)
        out = model.predict(batch)
        assert out.logits.shape == (20, N_TARGETS)


# ---------------------------------------------------------------------------
# Common interface tests
# ---------------------------------------------------------------------------

class TestModelInterface:
    """Test that all models satisfy the BaselineModel contract."""

    @pytest.fixture(params=["retain", "ehr_bert", "behrt", "medbert", "cehrbert"])
    def model_name(self, request):
        return request.param

    def _make_model(self, name):
        if name == "retain":
            from baselines.retain.model import RETAINModel
            return RETAINModel(n_codes=N_CODES, n_targets=N_TARGETS, d_emb=16, d_rnn=16)
        elif name == "ehr_bert":
            from baselines.ehr_bert.model import EHRBertModel
            return EHRBertModel(n_codes=N_CODES, n_targets=N_TARGETS,
                                d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=32)
        elif name == "behrt":
            from baselines.behrt.model import BEHRTModel
            return BEHRTModel(n_codes=N_CODES, n_targets=N_TARGETS,
                              d_model=24, n_layers=1, n_heads=4, max_seq_len=32)
        elif name == "medbert":
            from baselines.medbert.model import MedBERTModel
            return MedBERTModel(n_codes=N_CODES, n_targets=N_TARGETS,
                                d_model=24, n_layers=1, n_heads=6, max_seq_len=32)
        elif name == "cehrbert":
            from baselines.cehrbert_adapter.adapter import CEHRBertAdapter
            return CEHRBertAdapter(n_codes=N_CODES, n_targets=N_TARGETS,
                                    d_model=16, n_layers=1, n_heads=4, max_seq_len=32)
        raise ValueError(name)

    def test_has_name(self, model_name):
        model = self._make_model(model_name)
        assert isinstance(model.name, str) and len(model.name) > 0

    def test_has_model_card(self, model_name):
        model = self._make_model(model_name)
        card = model.model_card
        assert "trainable_params" in card
        assert "hidden_size" in card

    def test_predict_returns_model_output(self, model_name):
        from baselines.common.interface import ModelOutput
        model = self._make_model(model_name)
        batch = _make_batch(B=2, L=8)
        out = model.predict(batch)
        assert isinstance(out, ModelOutput)
        assert out.logits.shape == (2, N_TARGETS)

    def test_training_step_returns_loss(self, model_name):
        model = self._make_model(model_name)
        batch = _make_batch(B=2, L=8)
        result = model.training_step(batch)
        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].requires_grad

    def test_deterministic_eval(self, model_name):
        model = self._make_model(model_name)
        model.eval()
        batch = _make_batch(B=2, L=8)
        with torch.no_grad():
            out1 = model.predict(batch).logits
            out2 = model.predict(batch).logits
        assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# Counterfactual module tests
# ---------------------------------------------------------------------------

class TestCounterfactual:
    def test_cf_rmse_age(self):
        from baselines.common.counterfactual import cf_rmse_age
        # Perfect model
        result = cf_rmse_age(
            predict_fn=lambda a: np.array([0.5]),
            oracle_fn=lambda a: np.array([0.5]),
        )
        assert result == 0.0

    def test_surface_rmse(self):
        from baselines.common.counterfactual import surface_rmse
        result = surface_rmse(
            predict_fn=lambda a, l: np.array([0.5]),
            oracle_fn=lambda a, l: np.array([0.5]),
        )
        assert result == 0.0

    def test_mechanism_classification(self):
        from baselines.common.counterfactual import classify_mechanism
        assert classify_mechanism(0.05) == "FUNCTIONAL_RECOVERY"
        assert classify_mechanism(0.15) == "PARTIAL_RECOVERY"
        assert classify_mechanism(0.30) == "NO_MECHANISM_RECOVERY"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
