#!/usr/bin/env python3
"""Vanilla TALE-EHR backbone for the SHARED pretrain (ablation package).

This is the ``vanilla`` arm of :class:`TALEEHRAblation` with age conditioning off.
It is a thin subclass so the pretrained state_dict keys are byte-for-byte the same
as every arm's non-age parameters, letting all four fine-tune arms load the one
shared backbone with ``strict=False`` (age keys are simply absent here and added
fresh at fine-tune).

Demographics fix (INV-demo) is inherited: ``demo_dim == 2`` (sex, race only); age
is a separate ``age_years`` field and never enters ``demo_proj``.
"""

from __future__ import annotations

from pathlib import Path

from model_ablation.tale_ehr_age import TALEEHRAblation


class TALEEHR(TALEEHRAblation):
    def __init__(
        self,
        embedding_path: str | Path,
        num_codes: int,
        d_model: int = 256,
        poly_degree: int = 5,
        demo_hidden: int = 64,
        age_emb_dim: int = 32,
        age_hidden_dim: int = 64,
    ) -> None:
        super().__init__(
            embedding_path=embedding_path,
            num_codes=num_codes,
            arm="vanilla",
            d_model=d_model,
            poly_degree=poly_degree,
            demo_hidden=demo_hidden,
            age_emb_dim=age_emb_dim,
            age_hidden_dim=age_hidden_dim,
        )
