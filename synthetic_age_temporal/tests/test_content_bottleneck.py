"""Smoke tests for E01 + generator content helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAT))
sys.path.insert(0, str(ROOT))

from baselines.common.training import set_seed  # noqa: E402
from content_bottleneck.generator_content import (  # noqa: E402
    load_weight_matrix,
    oracle_content_pre_decay,
)
from content_bottleneck.models_e01 import build_e01  # noqa: E402
from config import all_signal_codes  # noqa: E402
import json


def test_e01_shapes_and_pi_free():
    set_seed(0)
    model = build_e01({"d_model": 64, "dropout": 0.0}, 40, 8, age_temporal=True)
    batch = {
        "enc_code_ids": torch.randint(1, 40, (2, 3, 4)),
        "enc_code_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "enc_tau": torch.rand(2, 3) + 0.1,
        "enc_padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        "age": torch.tensor([4.0, 12.0]),
    }
    with torch.no_grad():
        model.base.beta.fill_(-1.5)
    parts = model(
        batch["enc_code_ids"], batch["enc_code_mask"], batch["enc_tau"],
        batch["enc_padding_mask"], batch["age"], return_parts=True,
    )
    assert parts["logits"].shape == (2, 8)
    assert parts["a"].shape == (2, 3, 8)
    assert parts["contrib"].shape == (2, 3, 8)
    # age must not enter content projections
    other = model(
        batch["enc_code_ids"], batch["enc_code_mask"], batch["enc_tau"],
        batch["enc_padding_mask"], batch["age"], return_parts=True,
        gate_age=torch.tensor([0.0, 18.0]),
    )
    assert torch.allclose(parts["a"], other["a"])
    assert torch.allclose(parts["e"], other["e"])
    assert not torch.allclose(parts["g"], other["g"])


def test_generator_w_both_signs():
    specs = json.loads(
        (SAT / "outputs/data/seed20260922/controlled/S2/target_specs.json").read_text()
    )
    W, signals, mechs = load_weight_matrix(specs)
    assert W.shape == (32, 12)
    assert list(signals) == list(all_signal_codes())
    assert (W > 0).any() and (W < 0).any()
    mem = torch.zeros(12)
    mem[0] = 1.0
    pre = oracle_content_pre_decay(mem.numpy(), specs, W)
    assert pre.shape == (32,)


if __name__ == "__main__":
    test_e01_shapes_and_pi_free()
    print("OK test_e01_shapes_and_pi_free")
    test_generator_w_both_signs()
    print("OK test_generator_w_both_signs")
    print("All content-bottleneck smoke tests passed.")
