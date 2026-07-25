#!/usr/bin/env python3
"""``model_new`` -- a clean implementation of developmental kernel modulation (DKM).

Self-contained. Imports nothing from ``model/``, ``model_ablation/`` or ``finetune/``, and
modifies none of them.

Layout:

    arms.py          --arm -> ArmConfig, invariant checks
    basis.py         Chebyshev T_1..T_5 on tau_tilde; the only Chebyshev evaluation
    age_encoding.py  LogAgeFourier, CoefficientGenerator, AgeConditioner, band stats
    encoder.py       TimeAwareAttention, EncoderBlock, Encoder, mask helpers
    pooling.py       AttentionPooling
    model.py         DKMModel -- one class, arm-gated
    data.py          pretrain dataset + collate + tau_max + corpus stats
    data_finetune.py fine-tune dataset + collate
    optim.py         build_param_groups from module-declared sets
    diagnostics.py   the sole owner of all logging and JSON
    train.py         pretraining
    train_finetune.py
    preflight.py     Phase 10 review checkpoint

Every module except ``diagnostics`` is runnable as ``python -m model_new.<mod>`` for its own
smoke test; those smoke tests route their output through ``diagnostics``.
"""

from __future__ import annotations

__version__ = "1.0.0"

from model_new.arms import ARMS, ArmConfig, resolve_arm

__all__ = ["ARMS", "ArmConfig", "resolve_arm", "__version__"]
