#!/usr/bin/env python3
"""Four-arm age-conditioning ablation model (single class, ``--arm``-gated).

One class, both age pathways present-or-absent by :class:`arms.ArmConfig`:

  arm             kernel Delta-alpha        additive embed delta     age fed to Fourier
  ----            ------------------        --------------------     ------------------
  vanilla         0 (no params)             0 (no params)            (unused)
  random_constant MLP(phi(const_age))       0 (no params)            constant
  additive        0 (no params)             MLP(phi(real_age))       real
  kernel          MLP(phi(real_age))        0 (no params)            real

Invariants (asserted at runtime by ``assert_arm_invariants`` and in the forward
debug path):
  (2) Delta-alpha exactly zero in vanilla and additive.
  (3) additive embed delta exactly zero in vanilla and kernel.
  (INV-demo) age never enters through ``demographics`` (which is [B, L, 2] = sex, race,
  ``demo_dim == 2``); age arrives only via the separate ``age_years`` field.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_ablation.age_embedding import AdditiveAgeEmbedding, FourierAgeEmbedding
from model_ablation.arms import RANDOM_CONSTANT_AGE_YEARS, ArmConfig, resolve_arm
from model_ablation.time_aware_attention_age import (
    AgeConditionedMultiScaleTemporalAggregation,
    AgeConditionedTimeAwareAttention,
)

DEMO_DIM = 2  # sex, race only. Age is a SEPARATE field and never enters demo_proj.


class TALEEHRAblation(nn.Module):
    def __init__(
        self,
        embedding_path: str | Path,
        num_codes: int,
        arm: str = "vanilla",
        d_model: int = 256,
        poly_degree: int = 5,
        demo_hidden: int = 64,
        age_emb_dim: int = 32,
        age_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cfg: ArmConfig = resolve_arm(arm)
        self.embedding_path = Path(embedding_path)
        self.num_codes = int(num_codes)
        self.d_model = int(d_model)
        self.demo_dim = DEMO_DIM
        self.demo_hidden = int(demo_hidden)
        self.age_emb_dim = int(age_emb_dim)

        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {self.embedding_path}")
        emb_obj = torch.load(self.embedding_path, map_location="cpu")
        embedding_table = emb_obj["embeddings"].float()
        if embedding_table.ndim != 2 or embedding_table.shape[1] != 1024:
            raise ValueError(f"Expected embedding table [N+2, 1024], got {tuple(embedding_table.shape)}")
        self.register_buffer("embedding_table", embedding_table, persistent=True)
        self.embedding_dim = 1024

        self.time_aware_attention = AgeConditionedTimeAwareAttention(
            embedding_dim=self.embedding_dim,
            d_model=self.d_model,
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            age_hidden_dim=age_hidden_dim,
            age_conditioning_mode=self.cfg.age_conditioning_mode,
        )
        self.temporal_aggregation = AgeConditionedMultiScaleTemporalAggregation(
            d_model=self.d_model,
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            age_hidden_dim=age_hidden_dim,
            age_conditioning_mode=self.cfg.age_conditioning_mode,
        )
        # Additive age pathway (parameters exist only in the additive arm).
        self.additive_age_emb = AdditiveAgeEmbedding(
            in_dim=age_emb_dim,
            hidden_dim=age_hidden_dim,
            out_dim=self.embedding_dim,
            enabled=self.cfg.additive_embed,
        )
        # Fourier embedding for the additive pathway (fixed, shared banding).
        self.additive_fourier = FourierAgeEmbedding(num_frequencies=age_emb_dim // 2)

        # demo_proj consumes sex/race ONLY (demo_dim == 2). Age cannot enter here.
        self.demo_proj = nn.Sequential(nn.Linear(self.demo_dim, self.demo_hidden), nn.GELU())
        self.history_proj = nn.Identity()
        predictor_in = self.d_model + self.demo_hidden

        self.code_predictor = nn.Sequential(
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, self.num_codes),
        )
        with torch.no_grad():
            self.code_predictor[-1].bias.fill_(-7.0)

    # ---- age-source helpers ------------------------------------------------
    def _kernel_age_years(self, age_years: torch.Tensor) -> torch.Tensor:
        """Age (years) fed to the kernel Fourier embedding, per arm."""
        if self.cfg.age_source == "constant":
            return torch.full_like(age_years, RANDOM_CONSTANT_AGE_YEARS)
        # real / none: real age is passed; when mode == 'none' the MLP returns zeros
        # regardless, so the (ignored) age value is harmless.
        return age_years.clamp(min=0.0)

    def _additive_age_years(self, age_years: torch.Tensor) -> torch.Tensor:
        return age_years.clamp(min=0.0)

    # ---- invariant checks --------------------------------------------------
    @torch.no_grad()
    def assert_arm_invariants(self, age_years: torch.Tensor) -> None:
        af_kernel = self.time_aware_attention.age_emb(self._kernel_age_years(age_years))
        delta_alpha = self.time_aware_attention.temporal_weight.coefficient_delta(af_kernel)
        af_add = self.additive_fourier(self._additive_age_years(age_years))
        add_delta = self.additive_age_emb(af_add)

        if self.cfg.arm in {"vanilla", "additive"}:
            nz = int(torch.count_nonzero(delta_alpha))
            assert nz == 0, f"[INV-2] arm={self.cfg.arm}: Delta-alpha must be exactly 0, found {nz} nonzeros"
        if self.cfg.arm in {"vanilla", "kernel"}:
            nz = int(torch.count_nonzero(add_delta))
            assert nz == 0, f"[INV-3] arm={self.cfg.arm}: additive delta must be exactly 0, found {nz} nonzeros"

    def age_pathway_param_count(self) -> int:
        """Trainable parameters in this arm's active age pathway (attention side)."""
        kernel = self.time_aware_attention.temporal_weight.age_coeff_gen.num_pathway_params()
        additive = self.additive_age_emb.num_pathway_params()
        return kernel + additive

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_repr_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        code_indices = batch["code_indices"]
        delta_t = batch["delta_t"]
        attention_mask = batch["attention_mask"]
        demographics = batch["demographics"]
        if demographics.shape[-1] != self.demo_dim:
            raise AssertionError(
                f"[INV-demo] demographics must have {self.demo_dim} channels (sex, race); "
                f"got {demographics.shape[-1]} -> age is leaking into demographics/demo_proj."
            )
        if "age_years" not in batch:
            raise AssertionError("[INV-demo] age must be supplied via batch['age_years'], not demographics.")
        age_years = batch["age_years"]  # [B, L]

        code_embeddings = self.embedding_table[code_indices]

        # Additive age pathway: delta added to code embeddings before mlp_q/k/v.
        af_add = self.additive_fourier(self._additive_age_years(age_years))
        add_delta = self.additive_age_emb(af_add)                       # [B, L, embedding_dim]
        add_delta = add_delta * attention_mask.unsqueeze(-1).to(add_delta.dtype)
        code_embeddings = code_embeddings + add_delta

        # Kernel age pathway: Fourier features fed to the temporal-weight generator.
        af_kernel = self.time_aware_attention.age_emb(self._kernel_age_years(age_years))

        e = self.time_aware_attention(code_embeddings, delta_t, attention_mask, af_kernel)

        if return_repr_only:
            demo_features = self.demo_proj(demographics)
            return {"h_repr": e, "demo_features": demo_features}

        b = code_indices.shape[0]
        batch_idx = torch.arange(b, device=code_indices.device)
        lengths = attention_mask.sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)

        # Aggregation is modulated by a SINGLE current age per sequence (age at the
        # last valid event), matching its delta-to-current construction -> features
        # are [B, age_emb_dim] and Delta-alpha is [B, D+1].
        age_current = age_years[batch_idx, last_idx]
        af_kernel_agg = self.temporal_aggregation.age_emb(self._kernel_age_years(age_current))
        h = self.temporal_aggregation(e, batch["timestamps_days"], attention_mask, af_kernel_agg)

        demo_last = demographics[batch_idx, last_idx]
        combined = torch.cat([self.history_proj(h), self.demo_proj(demo_last)], dim=-1)
        code_logits = self.code_predictor(combined)
        return {"code_logits": code_logits, "h": h}


def _count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
