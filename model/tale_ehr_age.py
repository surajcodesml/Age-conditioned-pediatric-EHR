#!/usr/bin/env python3
"""TALE-EHR model with age-conditioned polynomial temporal decay."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.tale_ehr import TALEEHR
from model.time_aware_attention_age import (
    AgeConditionedMultiScaleTemporalAggregation,
    AgeConditionedTimeAwareAttention,
)


class TALEEHRAge(nn.Module):
    def __init__(
        self,
        embedding_path: str | Path,
        num_codes: int,
        d_model: int = 256,
        poly_degree: int = 5,
        demo_dim: int = 3,
        demo_hidden: int = 64,
        age_conditioning_mode: str = "real",
        age_emb_dim: int = 32,
        age_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_path = Path(embedding_path)
        self.num_codes = int(num_codes)
        self.d_model = int(d_model)
        self.demo_dim = int(demo_dim)
        self.demo_hidden = int(demo_hidden)

        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {self.embedding_path}")

        emb_obj = torch.load(self.embedding_path, map_location="cpu")
        if "embeddings" not in emb_obj:
            raise ValueError("bge_embeddings.pt missing key 'embeddings'")
        embedding_table = emb_obj["embeddings"].float()
        if embedding_table.ndim != 2 or embedding_table.shape[1] != 1024:
            raise ValueError(f"Expected embedding table [N+2, 1024], got {tuple(embedding_table.shape)}")
        if embedding_table.shape[0] < self.num_codes + 2:
            raise ValueError(
                f"Embedding rows ({embedding_table.shape[0]}) < num_codes+2 ({self.num_codes + 2})"
            )
        self.register_buffer("embedding_table", embedding_table, persistent=True)

        self.time_aware_attention = AgeConditionedTimeAwareAttention(
            embedding_dim=1024,
            d_model=self.d_model,
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            age_hidden_dim=age_hidden_dim,
            age_conditioning_mode=age_conditioning_mode,
        )
        self.temporal_aggregation = AgeConditionedMultiScaleTemporalAggregation(
            d_model=self.d_model,
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            age_hidden_dim=age_hidden_dim,
            age_conditioning_mode=age_conditioning_mode,
        )

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
        self.time_params_predictor = nn.Sequential(
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, self.demo_hidden),
            nn.GELU(),
            nn.Linear(self.demo_hidden, 2),
        )

        with torch.no_grad():
            final_code_layer = self.code_predictor[-1]
            final_code_layer.bias.fill_(-7.0)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_repr_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        code_indices = batch["code_indices"]
        delta_t = batch["delta_t"]
        attention_mask = batch["attention_mask"]
        timestamps_days = batch["timestamps_days"]
        demographics = batch["demographics"]
        age_years = demographics[..., 0]
        debug_sample = bool(torch.rand(1, device=code_indices.device).item() < 0.005)

        code_embeddings = self.embedding_table[code_indices]
        e = self.time_aware_attention(
            code_embeddings,
            delta_t,
            attention_mask,
            age_years,
            debug_sample=debug_sample,
        )
        h = self.temporal_aggregation(
            e,
            timestamps_days,
            attention_mask,
            age_years,
            debug_sample=debug_sample,
        )

        if return_repr_only:
            demo_features = self.demo_proj(demographics)
            return {"h_repr": e, "demo_features": demo_features}

        b = code_indices.shape[0]
        device = code_indices.device
        lengths = attention_mask.sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        demo_last = demographics[torch.arange(b, device=device), last_idx]

        h_proj = self.history_proj(h)
        d_proj = self.demo_proj(demo_last)
        combined = torch.cat([h_proj, d_proj], dim=-1)

        code_logits = self.code_predictor(combined)
        if debug_sample:
            with torch.no_grad():
                print(
                    f"[logits] mean={float(code_logits.mean()):.3f} "
                    f"std={float(code_logits.std(unbiased=False)):.3f} "
                    f"min={float(code_logits.min()):.3f} "
                    f"max={float(code_logits.max()):.3f} "
                    f"sigmoid_mean={float(torch.sigmoid(code_logits).mean()):.5f}",
                    flush=True,
                )
        time_params = self.time_params_predictor(combined)
        if debug_sample:
            with torch.no_grad():
                print(
                    f"[time_params] mean={float(time_params.mean()):.3f} "
                    f"std={float(time_params.std(unbiased=False)):.3f}",
                    flush=True,
                )
        return {"code_logits": code_logits, "time_params": time_params, "h": h}


def _count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    b, l = 2, 10
    num_codes = 100
    emb_rows = num_codes + 2

    with tempfile.TemporaryDirectory() as td:
        emb_path = Path(td) / "bge_embeddings.pt"
        emb = torch.randn(emb_rows, 1024, dtype=torch.float32) * 0.01
        torch.save({"code_ids": [f"C{i}" for i in range(emb_rows)], "embeddings": emb}, emb_path)

        attention_mask = torch.tensor([[True] * 10, [True] * 7 + [False] * 3], dtype=torch.bool)
        timestamps_days = torch.stack(
            [torch.linspace(0.0, 9.0, l), torch.tensor([2, 4, 5, 7, 9, 12, 14, 0, 0, 0], dtype=torch.float32)]
        )
        delta_t = torch.log1p(torch.abs(timestamps_days.unsqueeze(2) - timestamps_days.unsqueeze(1)) / 7.0)
        pair_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
        delta_t = delta_t * pair_mask.float()
        code_indices = torch.randint(2, emb_rows, (b, l), dtype=torch.long)
        code_indices[1, 7:] = 0
        demographics = torch.randn(b, l, 3, dtype=torch.float32)
        demographics[~attention_mask] = 0.0

        model_age = TALEEHRAge(
            embedding_path=emb_path,
            num_codes=num_codes,
            d_model=64,
            poly_degree=5,
            demo_hidden=32,
            age_conditioning_mode="real",
            age_emb_dim=32,
            age_hidden_dim=64,
        )
        batch = {
            "code_indices": code_indices,
            "delta_t": delta_t,
            "attention_mask": attention_mask,
            "timestamps_days": timestamps_days,
            "demographics": demographics,
        }
        out = model_age(batch)
        assert out["code_logits"].shape == (b, num_codes)
        assert out["time_params"].shape == (b, 2)
        assert out["h"].shape == (b, 64)
        assert torch.isfinite(out["code_logits"]).all()
        assert torch.isfinite(out["time_params"]).all()
        assert torch.isfinite(out["h"]).all()

        model_base = TALEEHR(
            embedding_path=emb_path,
            num_codes=num_codes,
            d_model=64,
            poly_degree=5,
            demo_hidden=32,
        )
        model_none = TALEEHRAge(
            embedding_path=emb_path,
            num_codes=num_codes,
            d_model=64,
            poly_degree=5,
            demo_hidden=32,
            age_conditioning_mode="none",
            age_emb_dim=32,
            age_hidden_dim=64,
        )
        model_none_perturbed = TALEEHRAge(
            embedding_path=emb_path,
            num_codes=num_codes,
            d_model=64,
            poly_degree=5,
            demo_hidden=32,
            age_conditioning_mode="none",
            age_emb_dim=32,
            age_hidden_dim=64,
        )

        # Explicit shared-parameter copy gives deterministic parity independent of construction order/RNG.
        with torch.no_grad():
            model_none.time_aware_attention.mlp_q.load_state_dict(model_base.time_aware_attention.mlp_q.state_dict())
            model_none.time_aware_attention.mlp_k.load_state_dict(model_base.time_aware_attention.mlp_k.state_dict())
            model_none.time_aware_attention.mlp_v.load_state_dict(model_base.time_aware_attention.mlp_v.state_dict())
            model_none.time_aware_attention.temporal_weight.coefficients.copy_(
                model_base.time_aware_attention.temporal_weight.coefficients
            )
            model_none.temporal_aggregation.q_base.copy_(model_base.temporal_aggregation.q_base)
            model_none.temporal_aggregation.temporal_weight.coefficients.copy_(
                model_base.temporal_aggregation.temporal_weight.coefficients
            )
            model_none.demo_proj.load_state_dict(model_base.demo_proj.state_dict())
            model_none.history_proj.load_state_dict(model_base.history_proj.state_dict())
            model_none.code_predictor.load_state_dict(model_base.code_predictor.state_dict())
            model_none.time_params_predictor.load_state_dict(model_base.time_params_predictor.state_dict())
            model_none_perturbed.load_state_dict(model_none.state_dict())
            model_none_perturbed.time_aware_attention.temporal_weight.age_coeff_gen.mlp[-1].weight.copy_(
                torch.randn_like(model_none_perturbed.time_aware_attention.temporal_weight.age_coeff_gen.mlp[-1].weight)
            )
            model_none_perturbed.time_aware_attention.temporal_weight.age_coeff_gen.mlp[-1].bias.copy_(
                torch.randn_like(model_none_perturbed.time_aware_attention.temporal_weight.age_coeff_gen.mlp[-1].bias)
            )
            model_none_perturbed.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].weight.copy_(
                torch.randn_like(model_none_perturbed.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].weight)
            )
            model_none_perturbed.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].bias.copy_(
                torch.randn_like(model_none_perturbed.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].bias)
            )

        out_base = model_none(batch)
        out_none = model_none_perturbed(batch)
        assert torch.allclose(out_base["code_logits"], out_none["code_logits"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(out_base["time_params"], out_none["time_params"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(out_base["h"], out_none["h"], atol=1e-5, rtol=1e-5)

        model_sens = TALEEHRAge(
            embedding_path=emb_path,
            num_codes=num_codes,
            d_model=64,
            poly_degree=5,
            demo_hidden=32,
            age_conditioning_mode="real",
            age_emb_dim=32,
            age_hidden_dim=64,
        )
        with torch.no_grad():
            model_sens.time_aware_attention.temporal_weight.age_coeff_gen.mlp[-1].weight.zero_()
            model_sens.time_aware_attention.temporal_weight.age_coeff_gen.mlp[-1].bias.zero_()
            model_sens.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].weight.copy_(
                torch.randn_like(model_sens.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].weight) * 1.0
            )
            model_sens.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].bias.copy_(
                torch.randn_like(model_sens.temporal_aggregation.temporal_weight.age_coeff_gen.mlp[-1].bias) * 1.0
            )
        batch_young = dict(batch)
        batch_old = dict(batch)
        batch_young["demographics"] = demographics.clone()
        batch_old["demographics"] = demographics.clone()
        batch_young["demographics"][..., 0] = 2.0
        batch_old["demographics"][..., 0] = 65.0
        out_young = model_sens(batch_young)
        out_old = model_sens(batch_old)
        assert (out_young["code_logits"] - out_old["code_logits"]).abs().max().detach().item() > 1e-3
        assert (out_young["h"] - out_old["h"]).abs().max().detach().item() > 1e-7

        trainable_base = _count_trainable(model_base)
        trainable_age = _count_trainable(model_age)
        delta = trainable_age - trainable_base
        print(f"Trainable parameter delta vs baseline: {delta}")
        assert delta == 5004

        print("Smoke test passed.")
