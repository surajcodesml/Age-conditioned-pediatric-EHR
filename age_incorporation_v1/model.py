"""Dataset-independent longitudinal Transformer with four age-incorporation arms.

Every arm instantiates the same modules (identical parameter count). Only the
forward equations change.

    base_i = E_code(code_i) + E_type(type_i) + W_time(time_norm_i)
    z_age_i = tanh(AgeEnc(age_at_event_norm_i))

    no_age:           x_i = base_i ;                      head_age = 0
    late_age:         x_i = base_i ;                      head_age = age_index_norm
    additive_age:     x_i = base_i + z_age_i ;            head_age = age_index_norm
    conditioned_age:  x_i = base_i * (1 + z_age_i) ;      head_age = age_index_norm
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import ARMS


class AgeEncoder(nn.Module):
    """Shared event-age encoder: Linear(1,32) → GELU → Linear(32,128), last layer zero-init."""

    def __init__(self, hidden: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, age_norm: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(age_norm))


class AgeIncorporationModel(nn.Module):
    def __init__(
        self,
        arm: str,
        n_codes: int,
        n_types: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.10,
        age_hidden: int = 32,
        head_hidden: int = 64,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.d_model = d_model

        self.code_embedding = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(n_types, d_model, padding_idx=0)
        self.time_projection = nn.Linear(1, d_model)
        self.age_encoder = AgeEncoder(age_hidden, d_model)
        self.pre_ln = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_kwargs = {"num_layers": n_layers}
        try:
            self.encoder = nn.TransformerEncoder(
                enc_layer, enable_nested_tensor=False, **encoder_kwargs
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, **encoder_kwargs)

        self.head = nn.Sequential(
            nn.Linear(d_model + 1, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def encode_events(
        self,
        code_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_norm: torch.Tensor,
        age_event_norm: torch.Tensor,
    ) -> torch.Tensor:
        base = (
            self.code_embedding(code_ids)
            + self.type_embedding(type_ids)
            + self.time_projection(time_norm.unsqueeze(-1))
        )
        if self.arm in ("additive_age", "conditioned_age"):
            z_age = self.age_encoder(age_event_norm.unsqueeze(-1))
            if self.arm == "additive_age":
                x = base + z_age
            else:
                x = base * (1.0 + z_age)
        else:
            x = base
        return self.pre_ln(x)

    def forward(
        self,
        code_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_norm: torch.Tensor,
        age_event_norm: torch.Tensor,
        padding_mask: torch.Tensor,
        index_age_norm: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encode_events(code_ids, type_ids, time_norm, age_event_norm)
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        if self.arm == "no_age":
            head_age = torch.zeros_like(index_age_norm)
        else:
            head_age = index_age_norm
        logits = self.head(torch.cat([pooled, head_age.unsqueeze(-1)], dim=-1))
        return logits.squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
