#!/usr/bin/env python3
"""TALE-EHR pretraining model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.time_aware_attention import MultiScaleTemporalAggregation, TimeAwareAttention
except ModuleNotFoundError:
    from time_aware_attention import MultiScaleTemporalAggregation, TimeAwareAttention


class TALEEHR(nn.Module):
    def __init__(
        self,
        embedding_path: str | Path,
        num_codes: int,
        d_model: int = 256,
        poly_degree: int = 5,
        demo_dim: int = 3,
        demo_hidden: int = 64,
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

        self.time_aware_attention = TimeAwareAttention(
            embedding_dim=1024,
            d_model=self.d_model,
            poly_degree=poly_degree,
        )
        self.temporal_aggregation = MultiScaleTemporalAggregation(
            d_model=self.d_model,
            poly_degree=poly_degree,
        )

        #self.demo_proj = nn.Sequential(nn.Linear(self.demo_dim, self.demo_hidden), nn.GELU())
        #self.history_proj = nn.Sequential(nn.Linear(self.d_model, self.demo_hidden), nn.GELU())
        #predictor_in = self.demo_hidden * 2
        
        self.demo_proj = nn.Sequential(nn.Linear(self.demo_dim, self.demo_hidden), nn.GELU())
        self.history_proj = nn.Identity()  # don't compress history
        predictor_in = self.d_model + self.demo_hidden  # 256 + 64 = 320

        self.code_predictor = nn.Sequential(
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, self.num_codes),
        )
        self.intensity_predictor = nn.Sequential(
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, predictor_in),
            nn.GELU(),
            nn.Linear(predictor_in, self.demo_hidden),
            nn.GELU(),
            nn.Linear(self.demo_hidden, 1),
        )
        
        # Initialize final code-predictor bias to the log-odds of the marginal
        # positive rate (~0.04% = 1/30635 * mean_codes_per_visit ≈ 0.0004).
        # This gives the model a free baseline; gradients then push
        # patient-specific deviations on top of the bias instead of having to
        # learn the marginal from scratch (which causes collapse).
        with torch.no_grad():
            final_code_layer = self.code_predictor[-1]
            final_code_layer.bias.fill_(-7.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        code_indices = batch["code_indices"]
        delta_t = batch["delta_t"]
        attention_mask = batch["attention_mask"]
        timestamps_days = batch["timestamps_days"]
        demographics = batch["demographics"]
        debug_sample = bool(torch.rand(1, device=code_indices.device).item() < 0.005)

        code_embeddings = self.embedding_table[code_indices]
        e = self.time_aware_attention(code_embeddings, delta_t, attention_mask, debug_sample=debug_sample)
        h = self.temporal_aggregation(e, timestamps_days, attention_mask, debug_sample=debug_sample)

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
        intensity = self.intensity_predictor(combined).squeeze(-1)
        if debug_sample:
            with torch.no_grad():
                print(
                    f"[intensity] mean={float(intensity.mean()):.3f} "
                    f"std={float(intensity.std(unbiased=False)):.3f}",
                    flush=True,
                )
        return {"code_logits": code_logits, "intensity": intensity, "h": h}


def _count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    torch.manual_seed(0)
    b, l = 2, 10
    num_codes = 100
    emb_rows = num_codes + 2

    with tempfile.TemporaryDirectory() as td:
        emb_path = Path(td) / "bge_embeddings.pt"
        emb = torch.randn(emb_rows, 1024, dtype=torch.float32) * 0.01
        torch.save({"code_ids": [f"C{i}" for i in range(emb_rows)], "embeddings": emb}, emb_path)

        model = TALEEHR(
            embedding_path=emb_path,
            num_codes=num_codes,
            d_model=64,
            poly_degree=5,
            demo_hidden=32,
        )

        attention_mask = torch.tensor([[True] * 10, [True] * 7 + [False] * 3], dtype=torch.bool)
        timestamps_days = torch.stack(
            [torch.linspace(0.0, 9.0, l), torch.tensor([2, 4, 5, 7, 9, 12, 14, 0, 0, 0], dtype=torch.float32)]
        )
        delta_t = torch.log1p(torch.abs(timestamps_days.unsqueeze(2) - timestamps_days.unsqueeze(1)))
        pair_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
        delta_t = delta_t * pair_mask.float()

        # BGE-aligned ids: PAD=0, UNK=1, real>=2
        code_indices = torch.randint(2, emb_rows, (b, l), dtype=torch.long)
        code_indices[1, 7:] = 0
        demographics = torch.randn(b, l, 3, dtype=torch.float32)
        demographics[~attention_mask] = 0.0

        out = model(
            {
                "code_indices": code_indices,
                "delta_t": delta_t,
                "attention_mask": attention_mask,
                "timestamps_days": timestamps_days,
                "demographics": demographics,
            }
        )
        print("code_logits:", tuple(out["code_logits"].shape))
        print("intensity:", tuple(out["intensity"].shape))
        print("h:", tuple(out["h"].shape))
        assert out["code_logits"].shape == (b, num_codes)
        assert out["intensity"].shape == (b,)
        assert out["h"].shape == (b, 64)
        assert torch.isfinite(out["code_logits"]).all()
        assert torch.isfinite(out["intensity"]).all()
        assert torch.isfinite(out["h"]).all()

        total, trainable = _count_params(model)
        frozen = model.embedding_table.numel()
        print(f"Parameters total={total:,}, trainable={trainable:,}, frozen_embedding={frozen:,}")
        print("Smoke test passed.")
