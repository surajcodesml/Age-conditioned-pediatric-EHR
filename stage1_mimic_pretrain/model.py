#!/usr/bin/env python3
"""Minimal age-conditioned temporal attention (no Fourier, no Chebyshev).

    s_ij^(h) = q_i^(h)⊤ k_j^(h) / sqrt(d_h) - [λ0 + β z(a_i)] τ_ij

λ0 and β are single scalars shared across attention heads (never per-head).
They are applied only inside Transformer self-attention unless
``pool_temporal_bias=True`` (off by default).

``no_interaction`` freezes β ≡ 0; ``age_temporal`` trains both.
``temporal_only`` is accepted as an alias of ``no_interaction``.
"""
from __future__ import annotations

import math
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_new.data import tau_from_timestamps
from model_new.encoder import build_key_mask, build_pair_mask
from stage1_mimic_pretrain.config import (
    DEFAULT_N_HEADS,
    HEAD_FINAL_BIAS_PRETRAIN,
    MIMIC_AGE_MEAN_YEARS,
    MIMIC_AGE_STD_YEARS,
    PROBE_AGES_YEARS,
    age_transform_spec,
    resolve_arm,
    tau_transform_spec,
)

__all__ = [
    "AgeTemporalBias",
    "TimeAwareAttention",
    "EncoderBlock",
    "Encoder",
    "AttentionPooling",
    "PredictionHead",
    "MinimalDKMModel",
]


class AgeTemporalBias(nn.Module):
    """λ(a) = λ0 + β z(a), z(a) = (a − μ)/σ. Shared across attention heads."""

    def __init__(self, arm: str, age_mean: float, age_sd: float) -> None:
        super().__init__()
        self.arm = resolve_arm(arm)
        if not (age_sd > 0):
            raise ValueError(f"age_sd must be > 0, got {age_sd}")
        self.lambda0 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        if self.arm == "no_interaction":
            self.beta.requires_grad_(False)
        self.register_buffer("age_mean", torch.tensor(float(age_mean), dtype=torch.float32),
                             persistent=True)
        self.register_buffer("age_sd", torch.tensor(float(age_sd), dtype=torch.float32),
                             persistent=True)

    def z_of(self, age_years: torch.Tensor) -> torch.Tensor:
        return (age_years - self.age_mean) / self.age_sd.clamp_min(1e-6)

    def lambda_of(self, age_years: torch.Tensor) -> torch.Tensor:
        """λ(a) = λ0 + β z(a). β is identically 0 under ``no_interaction``."""
        return self.lambda0 + self.beta * self.z_of(age_years)

    def pairwise_bias(self, tau: torch.Tensor, query_age_years: torch.Tensor) -> torch.Tensor:
        """``-λ(a_i) τ_ij`` with shape ``[B, L, L]``, broadcast-ready for heads.

        ``query_age_years`` is ``[B, L]``; row i is conditioned on query age a_i.
        """
        lam = self.lambda_of(query_age_years)  # [B, L]
        return -lam.unsqueeze(-1) * tau

    def key_bias(self, tau_to_now: torch.Tensor, age_last: torch.Tensor) -> torch.Tensor:
        """Pooling-site bias ``-λ(a_n) τ_to_now``, shape ``[B, L]``."""
        lam = self.lambda_of(age_last)  # [B]
        return -lam.unsqueeze(-1) * tau_to_now

    def lambda_at_ages(self, ages_years=PROBE_AGES_YEARS) -> dict[str, float]:
        device = self.lambda0.device
        age = torch.tensor(list(ages_years), dtype=torch.float32, device=device)
        lam = self.lambda_of(age)
        return {str(a): float(v) for a, v in zip(ages_years, lam.detach().cpu().tolist())}

    def age_parameters(self) -> list[nn.Parameter]:
        out = [self.lambda0]
        if self.beta.requires_grad:
            out.append(self.beta)
        return out


class TimeAwareAttention(nn.Module):
    """Content multi-head attention plus the shared age-temporal bias."""

    def __init__(self, d_in: int, d_model: int, *, n_heads: int = DEFAULT_N_HEADS,
                 temporal: AgeTemporalBias, use_out_proj: bool = False) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_in, self.d_model, self.n_heads = int(d_in), int(d_model), int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.mlp_q = nn.Sequential(nn.Linear(self.d_in, self.d_model), nn.GELU())
        self.mlp_k = nn.Sequential(nn.Linear(self.d_in, self.d_model), nn.GELU())
        self.mlp_v = nn.Sequential(nn.Linear(self.d_in, self.d_model), nn.GELU())
        self.out_proj = nn.Linear(self.d_model, self.d_model) if use_out_proj else None
        # Hold the shared bias by reference so λ0/β are not registered twice.
        self._bias = [temporal]

    @property
    def temporal(self) -> AgeTemporalBias:
        return self._bias[0]

    def forward(self, x: torch.Tensor, tau: torch.Tensor, attention_mask: torch.Tensor,
                age_years: torch.Tensor, *, need_diagnostics: bool = False):
        b, l, _ = x.shape
        q = self.mlp_q(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.mlp_k(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.mlp_v(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)

        content = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, L, L]
        bias = self.temporal.pairwise_bias(tau, age_years)           # [B, L, L]
        scores = content + bias.unsqueeze(1)                         # broadcast over H

        pair_mask = build_pair_mask(attention_mask).unsqueeze(1)
        scores = scores.masked_fill(~pair_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(~pair_mask, 0.0)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, l, self.d_model)
        if self.out_proj is not None:
            out = self.out_proj(out)
        if need_diagnostics:
            return out, {
                "attn": attn,
                "content_logits": content,
                "temporal_bias": bias,
                "pair_mask": pair_mask.squeeze(1),
            }
        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_in: int, d_model: int, *, temporal: AgeTemporalBias,
                 n_heads: int = DEFAULT_N_HEADS, use_residual: bool = True, use_layernorm: bool = True,
                 use_ffn: bool = True, ffn_mult: int = 4, use_out_proj: bool = False) -> None:
        super().__init__()
        self.d_in, self.d_model = int(d_in), int(d_model)
        self.use_residual = bool(use_residual)
        self.use_layernorm = bool(use_layernorm)
        self.use_ffn = bool(use_ffn)
        self.attn = TimeAwareAttention(d_in, d_model, n_heads=n_heads, temporal=temporal,
                                       use_out_proj=use_out_proj)
        self.ln_attn = nn.LayerNorm(self.d_in) if self.use_layernorm else None
        self.shortcut = (
            nn.Linear(self.d_in, self.d_model, bias=False)
            if (self.use_residual and self.d_in != self.d_model) else None
        )
        if self.use_ffn:
            self.ln_ffn = nn.LayerNorm(self.d_model) if self.use_layernorm else None
            self.ffn = nn.Sequential(
                nn.Linear(self.d_model, ffn_mult * self.d_model),
                nn.GELU(),
                nn.Linear(ffn_mult * self.d_model, self.d_model),
            )
        else:
            self.ln_ffn, self.ffn = None, None

    def forward(self, x, tau, attention_mask, age_years, *, need_diagnostics: bool = False):
        h = self.ln_attn(x) if self.ln_attn is not None else x
        extras = None
        if need_diagnostics:
            a, extras = self.attn(h, tau, attention_mask, age_years, need_diagnostics=True)
        else:
            a = self.attn(h, tau, attention_mask, age_years)
        if self.use_residual:
            res = self.shortcut(x) if self.shortcut is not None else x
            x = res + a
        else:
            x = a
        if self.use_ffn:
            h = self.ln_ffn(x) if self.ln_ffn is not None else x
            f = self.ffn(h)
            x = x + f if self.use_residual else f
        return (x, extras) if need_diagnostics else x


class Encoder(nn.Module):
    def __init__(self, d_in: int, d_model: int, *, temporal: AgeTemporalBias,
                 n_layers: int = 1, **block_kwargs) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = int(n_layers)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_in if i == 0 else d_model, d_model, temporal=temporal, **block_kwargs)
            for i in range(self.n_layers)
        ])

    def forward(self, x, tau, attention_mask, age_years, *, need_diagnostics: bool = False):
        extras = None
        for i, blk in enumerate(self.blocks):
            if need_diagnostics and i == 0:
                x, extras = blk(x, tau, attention_mask, age_years, need_diagnostics=True)
            else:
                x = blk(x, tau, attention_mask, age_years)
        return (x, extras) if need_diagnostics else x


class AttentionPooling(nn.Module):
    """Single-query pooling over encoded events.

    Default (Stage-1): relevance = q_base · e only. Age-conditioned temporal
    bias is **not** applied here, so pooling is identical across arms.

    ``use_temporal_bias=True`` restores the previous ``-λ(a_n) τ_to_now`` term
    for ablation only; it is off for main runs.
    """

    def __init__(self, d_model: int, *, temporal: AgeTemporalBias | None = None,
                 use_temporal_bias: bool = False) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_temporal_bias = bool(use_temporal_bias)
        self.q_base = nn.Parameter(torch.zeros(self.d_model))
        self._bias = [temporal]
        if self.use_temporal_bias and temporal is None:
            raise ValueError("use_temporal_bias=True requires a temporal module")
        self.reset_raw_parameters_(torch.Generator().manual_seed(0))

    @property
    def temporal(self) -> AgeTemporalBias | None:
        return self._bias[0]

    def reset_raw_parameters_(self, gen: torch.Generator) -> None:
        with torch.no_grad():
            self.q_base.copy_(torch.randn(self.d_model, generator=gen) * 0.02)

    @staticmethod
    def last_valid_index(attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.bool().sum(dim=1).long()
        if bool((lengths == 0).any()):
            bad = int((lengths == 0).nonzero()[0])
            raise ValueError(f"zero-length sequence at batch row {bad}: pooling is undefined")
        return lengths - 1

    def forward(self, e, tau_to_now, attention_mask, age_last, *, need_weights: bool = False):
        relevance = torch.einsum("d,bld->bl", self.q_base, e)
        bias = None
        if self.use_temporal_bias:
            bias = self.temporal.key_bias(tau_to_now, age_last)
            scores = relevance + bias
        else:
            scores = relevance
        keep = build_key_mask(attention_mask)
        scores = scores.masked_fill(~keep, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~keep, 0.0)
        h = torch.einsum("bl,bld->bd", attn, e)
        if need_weights:
            return h, attn, bias
        return h


class PredictionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int,
                 final_bias: float = HEAD_FINAL_BIAS_PRETRAIN) -> None:
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.final_bias = float(final_bias)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.apply_final_bias_()

    @torch.no_grad()
    def apply_final_bias_(self) -> None:
        self.net[-1].bias.fill_(self.final_bias)

    def head_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class MinimalDKMModel(nn.Module):
    """Frozen BGE embeddings → encoder → attention pooling → demo concat → code head."""

    def __init__(
        self,
        *,
        num_codes: int,
        embedding_path: str | Path | None = None,
        embedding_table: torch.Tensor | None = None,
        arm: str = "age_temporal",
        seed: int = 0,
        d_model: int = 256,
        n_layers: int = 1,
        n_heads: int = DEFAULT_N_HEADS,
        use_residual: bool = True,
        use_layernorm: bool = True,
        use_ffn: bool = True,
        ffn_mult: int = 4,
        demo_dim: int = 9,
        demo_channels: tuple[str, ...] = (),
        race_encoding: str = "one_hot",
        demo_hidden: int = 64,
        age_mean: float = MIMIC_AGE_MEAN_YEARS,
        age_sd: float = MIMIC_AGE_STD_YEARS,
        task: str = "pretrain",
        pool_temporal_bias: bool = False,
    ) -> None:
        super().__init__()
        arm = resolve_arm(arm)
        if task not in {"pretrain", "classification"}:
            raise ValueError(f"task must be 'pretrain' or 'classification', got {task!r}")
        self.arm = arm
        self.num_codes = int(num_codes)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.use_residual = bool(use_residual)
        self.use_layernorm = bool(use_layernorm)
        self.use_ffn = bool(use_ffn)
        self.ffn_mult = int(ffn_mult)
        self.demo_dim = int(demo_dim)
        self.demo_channels = tuple(demo_channels)
        self.race_encoding = str(race_encoding)
        self.demo_hidden = int(demo_hidden)
        self.seed = int(seed)
        self.task = task
        self.pool_temporal_bias = bool(pool_temporal_bias)

        self.temporal = AgeTemporalBias(arm, age_mean, age_sd)
        # Keep demographic-channel moments as buffers too (same μ, σ; R1 is unchanged).
        self.register_buffer("age_mean", torch.tensor(float(age_mean), dtype=torch.float32),
                             persistent=True)
        self.register_buffer("age_sd", torch.tensor(float(age_sd), dtype=torch.float32),
                             persistent=True)

        table = self._load_embedding_table(embedding_path, embedding_table)
        if table.shape[0] != self.num_codes + 2:
            raise AssertionError(
                f"embedding_table.shape[0] must equal len(code_vocab)+2 = "
                f"{self.num_codes + 2}, got {table.shape[0]}")
        self.register_buffer("embedding_table", table.float(), persistent=True)
        self.embedding_table.requires_grad_(False)
        self.embedding_dim = int(table.shape[1])

        self.encoder = Encoder(
            self.embedding_dim, self.d_model, temporal=self.temporal,
            n_layers=self.n_layers, n_heads=self.n_heads,
            use_residual=self.use_residual, use_layernorm=self.use_layernorm,
            use_ffn=self.use_ffn, ffn_mult=self.ffn_mult,
            use_out_proj=self.n_heads > 1,
        )
        self.pooling = AttentionPooling(
            self.d_model, temporal=self.temporal,
            use_temporal_bias=self.pool_temporal_bias,
        )
        self.demo_proj = nn.Sequential(nn.Linear(self.demo_dim, self.demo_hidden), nn.GELU())
        self.head_in = self.d_model + self.demo_hidden
        out_dim = self.num_codes if task == "pretrain" else 1
        self.head = PredictionHead(
            self.head_in, out_dim, hidden_dim=self.head_in,
            final_bias=HEAD_FINAL_BIAS_PRETRAIN if task == "pretrain" else 0.0,
        )
        self.reinit_non_age_parameters_(self.seed)

    @staticmethod
    def _load_embedding_table(path, table) -> torch.Tensor:
        if table is not None:
            return table.detach().clone()
        if path is None:
            raise ValueError("one of embedding_path / embedding_table is required")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"missing embedding file: {p}")
        obj = torch.load(p, map_location="cpu", weights_only=False)
        t = obj["embeddings"] if isinstance(obj, dict) else obj
        if t.ndim != 2:
            raise ValueError(f"expected a 2-D embedding table, got {tuple(t.shape)}")
        return t

    def _param_generator(self, name: str) -> torch.Generator:
        h = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
        return torch.Generator().manual_seed((int(self.seed) * 1_000_003 + h) % (2 ** 63 - 1))

    @torch.no_grad()
    def reinit_non_age_parameters_(self, seed: int) -> None:
        """Deterministic backbone init so both arms share identical non-temporal weights."""
        self.seed = int(seed)
        age_ids = {id(p) for p in self.age_parameters()}
        owners: dict[str, tuple[nn.Module, str, nn.Parameter]] = {}
        for mname, mod in self.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                owners[f"{mname}.{pname}" if mname else pname] = (mod, pname, p)
        for full in sorted(owners):
            mod, pname, p = owners[full]
            if id(p) in age_ids or not p.requires_grad:
                continue
            gen = self._param_generator(full)
            if isinstance(mod, nn.LayerNorm):
                p.fill_(1.0) if pname == "weight" else p.zero_()
            elif isinstance(mod, nn.Linear):
                if pname == "weight":
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(p)
                    bound = math.sqrt(6.0 / (fan_in + fan_out))
                    p.uniform_(-bound, bound, generator=gen)
                else:
                    p.zero_()
        self.pooling.reset_raw_parameters_(self._param_generator("pooling"))
        self.head.apply_final_bias_()

    def age_parameters(self) -> list[nn.Parameter]:
        return self.temporal.age_parameters()

    def head_parameters(self) -> list[nn.Parameter]:
        return self.head.head_parameters()

    def standardize_demo_age(self, demo: torch.Tensor) -> torch.Tensor:
        out = demo.clone()
        out[..., 0] = (out[..., 0] - self.age_mean) / self.age_sd.clamp_min(1e-6)
        return out

    def parameter_report(self) -> dict[str, int]:
        age_ids = {id(p) for p in self.age_parameters()}
        head_ids = {id(p) for p in self.head_parameters()}
        backbone = sum(p.numel() for p in self.parameters()
                       if p.requires_grad and id(p) not in age_ids and id(p) not in head_ids)
        age = sum(p.numel() for p in self.age_parameters())
        head = sum(p.numel() for p in self.head_parameters())
        return {
            "backbone": backbone,
            "age": age,
            "head": head,
            "frozen_embedding": int(self.embedding_table.numel()),
            "total_trainable": backbone + age + head,
            "lambda0_trainable": int(self.temporal.lambda0.requires_grad),
            "beta_trainable": int(self.temporal.beta.requires_grad),
        }

    def config_dict(self) -> dict:
        return {
            "arm": self.arm,
            "attention_equation": (
                "s_ij^(h) = q_i^(h)^T k_j^(h) / sqrt(d_h) - [lambda0 + beta * z(a_i)] * tau_ij"
            ),
            "lambda0_beta": "single scalars shared across all attention heads (not per-head)",
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_head": self.d_model // self.n_heads,
            "use_residual": self.use_residual,
            "use_layernorm": self.use_layernorm,
            "use_ffn": self.use_ffn,
            "ffn_mult": self.ffn_mult,
            "use_out_proj": self.n_heads > 1,
            "demo_dim": self.demo_dim,
            "demo_channels": list(self.demo_channels),
            "demo_hidden": self.demo_hidden,
            "race_encoding": self.race_encoding,
            "age_transform": age_transform_spec(float(self.age_mean), float(self.age_sd)),
            "tau_transform": tau_transform_spec(),
            "embedding_dim": self.embedding_dim,
            "masking": "padding_only",
            "pooling": ("attention_with_optional_lambda_beta"
                        if self.pool_temporal_bias else
                        "attention_relevance_only (no lambda/beta)"),
            "pool_temporal_bias": self.pool_temporal_bias,
            "experimental_difference": "self-attention lambda(a)*tau only; pooling identical across arms",
            "head_in": self.head_in,
            "head_out": self.head.out_dim,
            "head_final_bias": self.head.final_bias,
            "task": self.task,
            "obsolete_disabled": [
                "LogAgeFourier / LinearAgeFourier",
                "ChebyshevKernel / polynomial temporal basis",
                "AgeConditioner MLP / coefficient generator",
                "per-head lambda/beta",
            ],
        }

    def _check_batch(self, batch: dict[str, torch.Tensor]) -> None:
        if "age_years" not in batch:
            raise AssertionError("age must arrive as batch['age_years']")
        demo = batch["demographics"]
        if demo.shape[-1] != self.demo_dim:
            raise AssertionError(
                f"demographics must have {self.demo_dim} channels, got {demo.shape[-1]}")
        for key in ("code_indices", "timestamps_days", "attention_mask"):
            if key not in batch:
                raise AssertionError(f"batch is missing required key {key!r}")
        if "lengths" in batch:
            if not bool(torch.equal(batch["lengths"].to(batch["attention_mask"].device),
                                    batch["attention_mask"].sum(dim=1).long())):
                raise AssertionError("batch['lengths'] disagrees with attention_mask")

    def forward(self, batch: dict[str, torch.Tensor], *,
                age_years_for_bias: torch.Tensor | None = None,
                need_diagnostics: bool = False) -> dict:
        """``age_years_for_bias`` overrides only the temporal-bias ages (shuffle test).

        Demographic channel 0 is **not** overwritten, so ΔL_shuffle isolates the
        age × temporal interaction from the existing R1 demographic age feature.
        """
        self._check_batch(batch)
        code_indices = batch["code_indices"]
        attention_mask = batch["attention_mask"]
        age_years = batch["age_years"]
        age_bias = age_years if age_years_for_bias is None else age_years_for_bias
        demographics = batch["demographics"]

        tau, tau_to_now = tau_from_timestamps(
            batch["timestamps_days"], attention_mask, batch.get("lengths"))
        x = self.embedding_table[code_indices]
        extras = None
        if need_diagnostics:
            e, extras = self.encoder(x, tau, attention_mask, age_bias, need_diagnostics=True)
        else:
            e = self.encoder(x, tau, attention_mask, age_bias)

        last = self.pooling.last_valid_index(attention_mask)
        rows = torch.arange(code_indices.shape[0], device=code_indices.device)
        age_last = age_bias[rows, last]
        if need_diagnostics:
            h, pool_attn, pool_bias = self.pooling(
                e, tau_to_now, attention_mask, age_last, need_weights=True)
        else:
            h = self.pooling(e, tau_to_now, attention_mask, age_last)
            pool_attn = pool_bias = None

        demo_last = self.standardize_demo_age(demographics[rows, last])
        logits = self.head(torch.cat([h, self.demo_proj(demo_last)], dim=-1))

        out = {
            "h": h,
            "age_last": age_last,
            "lambda0": self.temporal.lambda0.detach(),
            "beta": self.temporal.beta.detach(),
        }
        if self.task == "pretrain":
            out["code_logits"] = logits
        else:
            out["logits"] = logits.squeeze(-1)
        if need_diagnostics:
            out.update({
                "pool_attn": pool_attn,
                "pool_bias": pool_bias,
                "e": e,
                "tau": tau,
                "tau_to_now": tau_to_now,
            })
            if extras is not None:
                out.update(extras)
        return out


def build_param_groups(model: MinimalDKMModel, lr_backbone: float, lr_age: float,
                       lr_head: float) -> tuple[list[dict], dict]:
    age = list(model.age_parameters())
    head = list(model.head_parameters())
    age_ids = {id(p) for p in age}
    head_ids = {id(p) for p in head}
    trainable = [p for p in model.parameters() if p.requires_grad]
    backbone = [p for p in trainable if id(p) not in age_ids and id(p) not in head_ids]
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
