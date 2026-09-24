#!/usr/bin/env python3
"""Tests for S5 heterogeneous-persistence baseline benchmark support.

Confirms:
  1. S5 loads through every baseline adapter
  2. Patient splits remain identical across S0–S5
  3. Persistence labels never enter model inputs
  4. Oracle S5 surfaces differ across acute/intermediate/chronic
  5. Counterfactual evaluation can compute group-specific surface RMSE
  6. Persistence ordering is computed from predictions, not model internals
  7. Existing S0–S3 mechanism thresholds / classification are unchanged
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "synthetic_age_temporal"))

from baselines.synthetic.data_adapter import (
    BENCHMARK_SCENARIOS,
    CORE_SCENARIOS,
    EVAL_ONLY_KEYS,
    MODEL_INPUT_KEYS,
    S5_PERSISTENCE_GROUPS,
    assert_no_eval_leakage,
    load_splits,
    make_baseline_loaders,
    model_batch,
    resolve_scenarios,
    scenario_dir,
)
from baselines.synthetic.runner import ALL_MODELS, build_model
from baselines.synthetic.result_schema import from_train_and_cf
from baselines.synthetic import s5_eval
from baselines.common.counterfactual import (
    FUNCTIONAL_SURFACE_RMSE,
    PARTIAL_SURFACE_RMSE,
    classify_mechanism,
    full_counterfactual_report,
)

DATA_SEED = 20260922


def _s5_available() -> bool:
    return scenario_dir("S5", DATA_SEED).exists()


pytestmark = pytest.mark.skipif(
    not _s5_available(),
    reason="S5 controlled dataset not found",
)


# ---------------------------------------------------------------------------
# 1. S5 loads through every baseline adapter
# ---------------------------------------------------------------------------

class TestS5LoadsThroughAdapters:
    def test_make_baseline_loaders_s5(self):
        train, val, test, vocab, info = make_baseline_loaders(
            "S5", data_seed=DATA_SEED, batch_size=4,
        )
        assert info["n_targets"] > 0
        assert info["n_codes"] > 0
        assert info["adapter"]["is_s5"] is True
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        batch = next(iter(train))
        assert "labels" in batch
        assert "code_ids" in batch

    @pytest.mark.parametrize("model_name", [
        "count_lightgbm", "retain", "ehr_bert", "behrt", "medbert", "cehrbert", "dtr",
    ])
    def test_each_baseline_accepts_s5_batch(self, model_name):
        _, _, test, vocab, info = make_baseline_loaders(
            "S5", data_seed=DATA_SEED, batch_size=2,
        )
        raw = next(iter(test))
        batch = model_batch(raw)
        n_codes, n_types, n_targets = info["n_codes"], info["n_types"], info["n_targets"]

        if model_name == "dtr":
            model = build_model("dtr", n_codes, n_types, n_targets, arm="age_temporal")
        else:
            model = build_model(model_name, n_codes, n_types, n_targets)

        if model_name == "count_lightgbm":
            from baselines.lightgbm.model import build_features_from_batch
            X = build_features_from_batch(batch, n_codes)
            assert X.shape[0] == batch["labels"].shape[0]
            # Unfitted LightGBM: just verify feature path; training_step N/A
            return

        out = model.predict(batch)
        assert out.logits.shape == (batch["labels"].shape[0], n_targets)
        step = model.training_step(batch)
        assert "loss" in step
        assert torch.isfinite(step["loss"])


# ---------------------------------------------------------------------------
# 2. Patient splits remain identical
# ---------------------------------------------------------------------------

class TestIdenticalSplits:
    def test_splits_identical_across_benchmark_scenarios(self):
        ref = load_splits("S0", DATA_SEED)
        for scen in BENCHMARK_SCENARIOS:
            other = load_splits(scen, DATA_SEED)
            for split in ("train", "val", "test"):
                assert other[split] == ref[split], (
                    f"Split {split} differs between S0 and {scen}"
                )

    def test_s5_split_sizes_match_meta(self):
        splits = load_splits("S5", DATA_SEED)
        meta = json.loads((scenario_dir("S5", DATA_SEED) / "meta.json").read_text())
        for split, n in meta["split_sizes"].items():
            assert len(splits[split]) == n


# ---------------------------------------------------------------------------
# 3. Persistence labels never enter model inputs
# ---------------------------------------------------------------------------

class TestNoPersistenceLeakage:
    def test_model_batch_strips_is_signal(self):
        _, _, test, _, _ = make_baseline_loaders("S5", data_seed=DATA_SEED, batch_size=2)
        raw = next(iter(test))
        assert "is_signal" in raw  # present in raw loader for eval
        clean = model_batch(raw)
        assert "is_signal" not in clean
        assert_no_eval_leakage(clean)
        assert set(clean.keys()) <= MODEL_INPUT_KEYS

    def test_eval_only_keys_blocked(self):
        bad = {
            "code_ids": torch.zeros(1, 2, dtype=torch.long),
            "labels": torch.zeros(1, 2),
            "persistence_group": torch.zeros(1),
            "is_signal": torch.zeros(1, 2, dtype=torch.bool),
        }
        with pytest.raises(AssertionError):
            assert_no_eval_leakage(bad)

    def test_s5_persistence_groups_are_eval_only_constant(self):
        # Groups exist for evaluation but are not in MODEL_INPUT_KEYS
        assert "persistence_group" not in MODEL_INPUT_KEYS
        assert "is_signal" not in MODEL_INPUT_KEYS
        assert set(S5_PERSISTENCE_GROUPS) == {"acute", "intermediate", "chronic"}

    def test_baseline_forward_signatures_ignore_persistence(self):
        """Models must not declare persistence / is_signal parameters."""
        _, _, test, _, info = make_baseline_loaders("S5", data_seed=DATA_SEED, batch_size=1)
        n_codes, n_types, n_targets = info["n_codes"], info["n_types"], info["n_targets"]
        forbidden_names = {
            "persistence_group", "persistence_class", "is_signal",
            "true_lambda", "decay_parameters",
        }
        for name in ("retain", "behrt", "medbert", "ehr_bert", "cehrbert"):
            model = build_model(name, n_codes, n_types, n_targets)
            src = inspect.getsource(model.forward) if hasattr(model, "forward") else ""
            for bad in forbidden_names:
                assert bad not in src, f"{name}.forward references {bad}"


# ---------------------------------------------------------------------------
# 4. Oracle S5 surfaces differ across groups
# ---------------------------------------------------------------------------

def _s5_multigroup_template():
    _, _, test, vocab, info = make_baseline_loaders(
        "S5", data_seed=DATA_SEED, batch_size=1,
    )
    itos = dict(vocab.itos)
    template = s5_eval.find_multigroup_template(test, itos, min_groups=3)
    assert template is not None, "No multi-group S5 test patient found"
    return template, vocab, info, itos


class TestOracleSurfacesDiffer:
    def test_oracle_group_surfaces_differ(self):
        template, vocab, info, itos = _s5_multigroup_template()
        specs = info["specs"]
        meta = info["meta"]
        result = s5_eval.oracle_group_surfaces_differ(
            template,
            itos,
            specs,
            float(meta["theta0"]),
            float(meta["beta_true"]),
        )
        assert result["differ"] is True
        for v in result["pairwise_rmse"].values():
            assert v > 1e-4


# ---------------------------------------------------------------------------
# 5. Group-specific surface RMSE computable
# ---------------------------------------------------------------------------

class TestGroupSurfaceRMSE:
    def test_group_surface_rmse_with_oracle_as_predictor(self):
        """Oracle-as-predictor → near-zero group Surface RMSE."""
        template, vocab, info, itos = _s5_multigroup_template()
        specs = info["specs"]
        meta = info["meta"]
        theta0, beta = float(meta["theta0"]), float(meta["beta_true"])

        ages = (5.0, 9.0)
        lags = (7.0, 90.0, 365.0)

        def predict_batch(b):
            from synthetic_age_temporal.ground_truth import ExampleSignals, compute_target_logits
            codes = b["code_ids"][0].cpu().numpy()
            pad = b["padding_mask"][0].cpu().numpy()
            lags_arr = b["lag_days"][0].cpu().numpy()
            is_q = b["is_query"][0].cpu().numpy()
            sig_codes, sig_lags = [], []
            for j in range(len(codes)):
                if pad[j] or is_q[j]:
                    continue
                name = itos.get(int(codes[j]))
                if name and str(name).startswith("SYN_SIGNAL_"):
                    sig_codes.append(str(name))
                    sig_lags.append(float(lags_arr[j]))
            sig = ExampleSignals(
                codes=np.array(sig_codes, dtype=object),
                lag_days=np.asarray(sig_lags, dtype=np.float64),
                tau=np.log1p(np.asarray(sig_lags, dtype=np.float64) / 7.0)
                if len(sig_lags) else np.zeros(0),
                times=np.array([], dtype="datetime64[ns]"),
            )
            _, probs, _, _ = compute_target_logits(
                age=float(b["age"][0]),
                signals=sig,
                specs=specs,
                scenario="S5",
                theta0=theta0,
                beta=beta,
                noise=np.zeros(len(specs)),
            )
            return probs

        rmses = s5_eval.group_surface_rmses(
            predict_batch, template, itos, specs, theta0, beta,
            ages=ages, lags_days=lags,
        )
        for g in ("acute", "intermediate", "chronic", "mean"):
            assert g in rmses
            assert rmses[g] < 1e-6, f"{g} RMSE={rmses[g]}"


# ---------------------------------------------------------------------------
# 6. Ordering from predictions, not internals
# ---------------------------------------------------------------------------

class TestPersistenceOrderingFromPredictions:
    def test_order_from_oracle_surfaces_is_correct(self):
        template, vocab, info, itos = _s5_multigroup_template()
        specs = info["specs"]
        meta = info["meta"]
        theta0, beta = float(meta["theta0"]), float(meta["beta_true"])

        def predict_batch(b):
            from synthetic_age_temporal.ground_truth import ExampleSignals, compute_target_logits
            codes = b["code_ids"][0].cpu().numpy()
            pad = b["padding_mask"][0].cpu().numpy()
            lags_arr = b["lag_days"][0].cpu().numpy()
            is_q = b["is_query"][0].cpu().numpy()
            sig_codes, sig_lags = [], []
            for j in range(len(codes)):
                if pad[j] or is_q[j]:
                    continue
                name = itos.get(int(codes[j]))
                if name and str(name).startswith("SYN_SIGNAL_"):
                    sig_codes.append(str(name))
                    sig_lags.append(float(lags_arr[j]))
            sig = ExampleSignals(
                codes=np.array(sig_codes, dtype=object),
                lag_days=np.asarray(sig_lags, dtype=np.float64),
                tau=np.log1p(np.asarray(sig_lags, dtype=np.float64) / 7.0)
                if len(sig_lags) else np.zeros(0),
                times=np.array([], dtype="datetime64[ns]"),
            )
            _, probs, _, _ = compute_target_logits(
                age=float(b["age"][0]), signals=sig, specs=specs, scenario="S5",
                theta0=theta0, beta=beta, noise=np.zeros(len(specs)),
            )
            return probs

        order = s5_eval.persistence_order_correct_from_surfaces(
            predict_batch, template, itos,
        )
        assert order["persistence_order_correct"] is True
        d = order["decay_proxy"]
        assert d["acute"] > d["intermediate"] > d["chronic"]

    def test_ordering_function_does_not_read_model_internals(self):
        src = inspect.getsource(s5_eval.persistence_order_correct_from_surfaces)
        assert "persistence_projection" not in src
        assert "lambda_of" not in src
        src_proxy = inspect.getsource(s5_eval.lag_decay_proxy)
        assert "cos" in src_proxy or "persistence" in src_proxy.lower()


# ---------------------------------------------------------------------------
# 7. S0–S3 mechanism behavior unchanged
# ---------------------------------------------------------------------------

class TestS0S3Unchanged:
    def test_core_scenarios_constant(self):
        assert CORE_SCENARIOS == ("S0", "S1", "S2", "S3")
        assert "S5" not in CORE_SCENARIOS
        assert "S6" not in BENCHMARK_SCENARIOS

    def test_resolve_scenarios(self):
        assert resolve_scenarios("core") == list(CORE_SCENARIOS)
        assert resolve_scenarios("all") == list(BENCHMARK_SCENARIOS)
        assert resolve_scenarios("S5") == ["S5"]

    def test_mechanism_thresholds_unchanged(self):
        assert FUNCTIONAL_SURFACE_RMSE == 0.10
        assert PARTIAL_SURFACE_RMSE == 0.25
        assert classify_mechanism(0.05) == "FUNCTIONAL_RECOVERY"
        assert classify_mechanism(0.15) == "PARTIAL_RECOVERY"
        assert classify_mechanism(0.30) == "NO_MECHANISM_RECOVERY"

    def test_s0_s3_cf_report_has_null_s5_fields(self):
        report = full_counterfactual_report(
            lambda a: np.array([0.5]),
            lambda l: np.array([0.5]),
            lambda a, l: np.array([0.5]),
            lambda a: np.array([0.5]),
            lambda l: np.array([0.5]),
            lambda a, l: np.array([0.5]),
        )
        assert report["S5_Surface_RMSE_acute"] is None
        assert report["persistence_order_correct"] is None
        assert report["mechanism_classification"] in {
            "FUNCTIONAL_RECOVERY", "PARTIAL_RECOVERY", "NO_MECHANISM_RECOVERY",
        }

    def test_s5_classification_labels_distinct(self):
        assert s5_eval.classify_heterogeneous_persistence(0.05, True) == (
            "HETEROGENEOUS_PERSISTENCE_RECOVERED"
        )
        assert s5_eval.classify_heterogeneous_persistence(0.05, False) == (
            "PARTIAL_HETEROGENEOUS_PERSISTENCE_RECOVERY"
        )
        assert s5_eval.classify_heterogeneous_persistence(0.15, True) == (
            "PARTIAL_HETEROGENEOUS_PERSISTENCE_RECOVERY"
        )
        assert s5_eval.classify_heterogeneous_persistence(0.30, True) == (
            "NO_HETEROGENEOUS_PERSISTENCE_RECOVERY"
        )

    def test_schema_nulls_s5_fields_for_core(self):
        rec = from_train_and_cf(
            scenario="S2",
            model="behrt",
            test_metrics={"micro_auroc": 0.7, "micro_auprc": 0.4, "bce": 0.5},
            cf_report={
                "cf_rmse_age": 0.1,
                "cf_rmse_lag": 0.1,
                "surface_rmse": 0.1,
                "mechanism_classification": "FUNCTIONAL_RECOVERY",
                "S5_Surface_RMSE_acute": 0.01,  # should be wiped
                "persistence_order_correct": True,
            },
        )
        assert rec["S5_Surface_RMSE_acute"] is None
        assert rec["persistence_order_correct"] is None
        assert rec["AUROC"] == 0.7
        assert rec["mechanism_classification"] == "FUNCTIONAL_RECOVERY"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
