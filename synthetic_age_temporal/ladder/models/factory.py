"""Build a ladder model from a resolved config."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from ladder.models.channels import ContentChannelDTR
from ladder.models.direct import DirectEvidenceDTR
from ladder.models.integrated import IntegratedHazardDTR
from ladder.models.mixture import PersistenceMixtureDTR


def build_model(
    cfg: dict[str, Any],
    n_codes: int,
    n_targets: int,
    *,
    age_temporal: bool,
) -> nn.Module:
    architecture = cfg["architecture"]
    common = dict(
        n_codes=n_codes,
        n_targets=n_targets,
        d_model=int(cfg["d_model"]),
        dropout=float(cfg["dropout"]),
        age_temporal=age_temporal,
        lambda_init=float(cfg["lambda_init"]),
    )
    if architecture == "direct":
        return DirectEvidenceDTR(
            **common,
            aggregation=str(cfg.get("aggregation", "raw_additive")),
            mass_eps=float(cfg.get("mass_eps", 1e-6)),
        )
    if architecture == "channels":
        return ContentChannelDTR(**common, n_channels=int(cfg["n_channels"]))
    if architecture == "mixture":
        return PersistenceMixtureDTR(**common, n_components=int(cfg["n_components"]))
    if architecture == "integrated_hazard":
        return IntegratedHazardDTR(**common, knots=tuple(cfg["knots"]))
    raise ValueError(f"Unknown architecture {architecture}")
