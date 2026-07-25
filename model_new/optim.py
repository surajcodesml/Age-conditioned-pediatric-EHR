#!/usr/bin/env python3
"""Optimizer parameter groups, built from **module-declared sets** (D6).

The legacy failure was ``Adam(model.parameters(), lr)``: one group, so the age pathway
trained at the backbone learning rate. A later fix used string matching on parameter names
and a parameter that should have been in the age group sat in the backbone group at a
3,400x smaller learning rate, so the mechanism never trained. An explicit declaration
cannot be broken by a rename, so membership here comes only from

    age      = union of module.age_parameters()   over modules that define it
    head     = union of module.head_parameters()  over modules that define it
    backbone = every other trainable parameter

and there is no name-string matching anywhere in this file.

A note on what to expect at step 0: the generator's final layer is zero-initialised, so its
*first* layer receives exactly zero gradient at step 0 (dL/dW1 is proportional to W2^T = 0)
and the pathway warms up over the first few hundred steps. That is expected -- do not
perturb the init to "fix" it. It is also why the acceptance signal is parameter **drift**
rather than gradient norm: under Adam's second-moment normalisation a tiny gradient still
produces a full-size step, so gradient norm is uninformative here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["build_param_groups", "GROUP_NAMES"]

GROUP_NAMES = ("backbone", "age", "head")


def build_param_groups(model: nn.Module, lr_backbone: float, lr_age: float,
                       lr_head: float) -> tuple[list[dict], dict]:
    """-> (param_groups for the optimizer, a report dict for config.json).

    HARD: the three groups are pairwise disjoint, their union is exactly the trainable set,
    and ``age`` is empty for ``vanilla`` and non-empty for every other arm.
    """
    age = list(model.age_parameters()) if hasattr(model, "age_parameters") else []
    head = list(model.head_parameters()) if hasattr(model, "head_parameters") else []

    age_ids = {id(p) for p in age}
    head_ids = {id(p) for p in head}
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable}
    backbone = [p for p in trainable if id(p) not in age_ids and id(p) not in head_ids]
    backbone_ids = {id(p) for p in backbone}

    overlaps = {
        "age&head": age_ids & head_ids,
        "age&backbone": age_ids & backbone_ids,
        "head&backbone": head_ids & backbone_ids,
    }
    for label, ids in overlaps.items():
        if ids:
            raise AssertionError(f"[INV-GROUPS] groups {label} overlap on {len(ids)} tensors")

    union = age_ids | head_ids | backbone_ids
    if union != trainable_ids:
        raise AssertionError(
            f"[INV-GROUPS] group union has {len(union)} tensors but the trainable set has "
            f"{len(trainable_ids)}: {len(trainable_ids - union)} missing, "
            f"{len(union - trainable_ids)} extra (a declared parameter is frozen or detached)"
        )
    for p in age:
        if not p.requires_grad:
            raise AssertionError("[INV-GROUPS] a declared age parameter has requires_grad=False")

    arm = getattr(getattr(model, "cfg", None), "arm", None)
    if arm == "vanilla" and age:
        raise AssertionError(f"[INV-GROUPS] arm=vanilla must have an empty age group, got {len(age)}")
    if arm is not None and arm != "vanilla" and not age:
        raise AssertionError(f"[INV-GROUPS] arm={arm} must have a non-empty age group")

    groups = [
        {"params": backbone, "lr": float(lr_backbone), "name": "backbone"},
        {"params": age, "lr": float(lr_age), "name": "age"},
        {"params": head, "lr": float(lr_head), "name": "head"},
    ]
    report = {
        "lr_backbone": float(lr_backbone),
        "lr_age": float(lr_age),
        "lr_head": float(lr_head),
        "n_tensors": {g["name"]: len(g["params"]) for g in groups},
        "n_params": {g["name"]: sum(p.numel() for p in g["params"]) for g in groups},
    }
    return groups, report


def _smoke() -> None:
    from model_new import diagnostics
    from model_new.arms import ARMS
    from model_new.model import DKMModel

    lines = []
    table = torch.randn(12, 16)
    for arm in ARMS:
        m = DKMModel(num_codes=10, embedding_table=table, arm=arm, d_model=8, age_hidden=4,
                     demo_hidden=4)
        _, rep = build_param_groups(m, 1e-4, 1e-3, 1e-3)
        lines.append(f"{arm:<16} tensors={rep['n_tensors']}  params={rep['n_params']}")
    diagnostics.print_block("optim.py smoke  (INV-GROUPS)", lines)


if __name__ == "__main__":
    _smoke()
