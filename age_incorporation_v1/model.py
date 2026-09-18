"""Dataset-independent longitudinal Transformer with five age-incorporation arms.

Every arm instantiates the same modules (identical parameter count). Only the
forward equations change.

    base_i = E_code(code_i) + E_type(type_i) + W_time(time_norm_i)
    z_age_i = tanh(AgeEnc(age_at_event_norm_i))

    no_age:           x_i = base_i ;                      head_age = 0
    late_age:         x_i = base_i ;                      head_age = age_index_norm
    additive_age:     x_i = base_i + z_age_i ;            head_age = age_index_norm
    conditioned_age:  x_i = base_i * (1 + z_age_i) ;      head_age = age_index_norm
    dkm_age:          x_i = base_i ;                      head_age = age_index_norm
                      + attention logits -= lambda(a_i) * tau_ij
    shared_decay:     x_i = base_i ;                      head_age = age_index_norm
                      + attention logits -= lambda_shared * tau_ij
                      (same as dkm_age but lambda does NOT depend on age)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ARMS, DKM_PROBE_AGES


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


class AgeLambdaGenerator(nn.Module):
    """DKM query-age generator: Linear(1,32) → GELU → Linear(32,1), last layer zero-init."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, age_norm: torch.Tensor) -> torch.Tensor:
        return self.net(age_norm.unsqueeze(-1)).squeeze(-1)


def _time_denom(age_scale_years: float) -> float:
    return math.log1p(age_scale_years * 365.25)


def days_from_time_norm(time_norm: torch.Tensor, age_scale_years: float) -> torch.Tensor:
    """Invert the dataset time_norm = log1p(days) / log1p(18*365.25)."""
    denom = _time_denom(age_scale_years)
    return torch.expm1(time_norm.clamp(0.0, 1.0) * denom)


def pairwise_tau(time_norm: torch.Tensor, age_scale_years: float) -> torch.Tensor:
    days = days_from_time_norm(time_norm, age_scale_years)
    delta = (days.unsqueeze(-1) - days.unsqueeze(-2)).abs()
    tau = torch.log1p(delta) / _time_denom(age_scale_years)
    return tau.clamp(0.0, 1.0)


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
        age_scale_years: float = 18.0,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.d_model = d_model
        self.n_heads = n_heads
        self.age_scale_years = age_scale_years
        self._dkm_cache: dict[str, torch.Tensor] | None = None

        self.code_embedding = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(n_types, d_model, padding_idx=0)
        self.time_projection = nn.Linear(1, d_model)
        self.age_encoder = AgeEncoder(age_hidden, d_model)
        self.pre_ln = nn.LayerNorm(d_model)

        self.age_lambda_generator = AgeLambdaGenerator(hidden=age_hidden)
        self.lambda_base_raw = nn.Parameter(torch.zeros(1))

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

    def dkm_lambda(self, age_event_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.age_lambda_generator(age_event_norm)
        if self.arm == "shared_decay":
            # Age cannot modify lambda; delta is computed but not used
            lam = F.softplus(self.lambda_base_raw).expand_as(delta)
            return lam, delta
        lam = F.softplus(self.lambda_base_raw + delta)
        return lam, delta

    def lambda_at_ages(self, ages_years: list[float] | tuple[float, ...], device: torch.device) -> dict[str, float]:
        age = torch.tensor(list(ages_years), dtype=torch.float32, device=device)
        age_norm = (age / self.age_scale_years).clamp(0.0, 1.0)
        lam, _ = self.dkm_lambda(age_norm)
        return {str(a): float(v) for a, v in zip(ages_years, lam.detach().cpu().tolist())}

    def _dkm_attn_bias(
        self,
        time_norm: torch.Tensor,
        age_event_norm: torch.Tensor,
    ) -> torch.Tensor:
        lam, delta = self.dkm_lambda(age_event_norm)
        tau = pairwise_tau(time_norm, self.age_scale_years)
        bias_bt = -lam.unsqueeze(-1) * tau
        bsz, seq = age_event_norm.shape
        attn_mask = (
            bias_bt.unsqueeze(1)
            .expand(bsz, self.n_heads, seq, seq)
            .contiguous()
            .view(bsz * self.n_heads, seq, seq)
        )
        self._dkm_cache = {
            "lambda": lam.detach(),
            "delta_lambda": delta.detach(),
            "tau": tau.detach(),
            "attn_bias": bias_bt.detach(),
            "attn_mask": attn_mask.detach(),
        }
        return attn_mask

    def _encode_dkm(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the shared encoder with temporal bias, using explicit softmax attention.

        SDPA on this stack produced NaNs in eval; ``need_weights=True`` uses
        baddbmm + softmax, matching ``score_ij - λ_i τ_ij`` then pad + softmax.
        """
        pad_f = torch.zeros(padding_mask.shape, dtype=x.dtype, device=x.device)
        pad_f = pad_f.masked_fill(padding_mask, float("-inf"))
        output = x
        for layer in self.encoder.layers:
            if not layer.norm_first:
                raise RuntimeError("dkm_age expects Pre-LayerNorm Transformer layers")
            y = layer.norm1(output)
            attn_out, _ = layer.self_attn(
                y,
                y,
                y,
                attn_mask=attn_mask,
                key_padding_mask=pad_f,
                need_weights=True,
                is_causal=False,
            )
            output = output + layer.dropout1(attn_out)
            output = output + layer._ff_block(layer.norm2(output))
        if self.encoder.norm is not None:
            output = self.encoder.norm(output)
        return output

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
        if self.arm in ("dkm_age", "shared_decay"):
            attn_bias = self._dkm_attn_bias(time_norm, age_event_norm)
            h = self._encode_dkm(x, attn_bias, padding_mask)
        else:
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


@torch.no_grad()
def dkm_lambda_curve(model: AgeIncorporationModel, device: torch.device) -> dict[str, float]:
    return model.lambda_at_ages(DKM_PROBE_AGES, device)


def verify_dkm_batch(
    model: AgeIncorporationModel,
    batch: dict,
    device: torch.device,
) -> dict[str, object]:
    """Smoke checks: shapes, masking, gradients, init lambda nearly constant."""
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(
        batch["code_ids"].to(device),
        batch["type_ids"].to(device),
        batch["time_norm"].to(device),
        batch["age_event_norm"].to(device),
        batch["padding_mask"].to(device),
        batch["index_age_norm"].to(device),
    )
    loss = logits.square().mean()
    loss.backward()
    cache = model._dkm_cache
    assert cache is not None
    bsz, seq = batch["code_ids"].shape
    bias = cache["attn_bias"]
    attn_mask = cache["attn_mask"]
    tau = cache["tau"]
    lam = cache["lambda"]
    pad = batch["padding_mask"].to(device)
    diag = torch.diagonal(tau, dim1=-2, dim2=-1)
    checks = {
        "logits_shape": tuple(logits.shape),
        "lambda_shape": tuple(lam.shape),
        "tau_shape": tuple(tau.shape),
        "attn_bias_shape": tuple(bias.shape),
        "attn_mask_shape": tuple(attn_mask.shape),
        "expected_bias_shape": (bsz, seq, seq),
        "expected_mask_shape": (bsz * model.n_heads, seq, seq),
        "bias_shape_ok": tuple(bias.shape) == (bsz, seq, seq),
        "mask_shape_ok": tuple(attn_mask.shape) == (bsz * model.n_heads, seq, seq),
        "tau_in_01": bool(tau.min().item() >= -1e-6 and tau.max().item() <= 1.0 + 1e-6),
        "tau_diag_zero": bool(diag.abs().max().item() < 1e-6),
        "lambda_positive": bool(lam.min().item() > 0),
        "padding_mask_bool": bool(pad.dtype == torch.bool),
        "age_generator_has_grad": any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in model.age_lambda_generator.parameters()
        ),
        "lambda_base_has_grad": model.lambda_base_raw.grad is not None,
    }
    curve = dkm_lambda_curve(model, device)
    vals = list(curve.values())
    checks["lambda_curve_init"] = curve
    checks["lambda_curve_nearly_constant"] = max(vals) - min(vals) < 1e-5
    g2 = 0.0
    for p in model.age_lambda_generator.parameters():
        if p.grad is not None:
            g2 += float(p.grad.detach().pow(2).sum())
    checks["age_generator_grad_norm"] = g2 ** 0.5
    model.zero_grad(set_to_none=True)
    return checks
