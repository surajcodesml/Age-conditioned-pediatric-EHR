#!/usr/bin/env python3
"""Time-aware attention, the encoder block, and the encoder stack.

Figure 1A of the draft shows a transformer encoder stack (LayerNorm -> time-aware
self-attention -> Add & Norm -> FFN -> Add & Norm, repeated N times). The legacy code is a
single time-aware attention operation with no residual, no LayerNorm, no FFN and no
stacking. These are not the same model and the difference is load-bearing: residual
connections give the model a route around attention entirely, so a deeper stack can dilute
whatever the temporal kernel contributes, while a single un-normalised attention layer is a
weak encoder that may underfit in a way that masks any arm difference.

Both are therefore available. ``n_layers``, ``use_residual``, ``use_layernorm`` and
``use_ffn`` are config parameters recorded in ``config.json``. With ``n_layers=1`` and all
three components disabled, this reduces exactly to the legacy single-attention encoder.
The temporal kernel is applied at **every** layer, conditioned on the same per-query age.

Masking is **padding only** (D4). The pretraining target visit lies outside the input
window, so bidirectional attention within the window is not leakage, and the legacy
``torch.tril`` contradicted the draft.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_new.age_encoding import AgeConditioner
from model_new.basis import DEFAULT_S, ChebyshevKernel

__all__ = ["build_pair_mask", "build_key_mask", "TimeAwareAttention", "EncoderBlock", "Encoder"]


# --------------------------------------------------------------------------- #
# Masking -- one helper, used by both the encoder and the pooling site.        #
# --------------------------------------------------------------------------- #
def build_pair_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, L] -> [B, L, L]`` padding-only pair mask with the diagonal forced ``True``.

    Without the diagonal, a padded row is all ``-inf`` and ``softmax`` returns ``NaN``. The
    downstream ``masked_fill`` happens to repair both the forward value and the gradient --
    ``masked_fill``'s backward fills rather than multiplies -- but relying on that is
    fragile. Forcing the diagonal makes ``softmax`` well-defined everywhere. ``INV-NAN``
    remains a test regardless.
    """
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()
    pair = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
    l = attention_mask.shape[1]
    eye = torch.eye(l, dtype=torch.bool, device=attention_mask.device)
    return pair | eye.unsqueeze(0)


def build_key_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, L] -> [B, L]`` key mask for the single-query pooling site, with position 0
    forced ``True`` so an all-padding row cannot produce an all ``-inf`` softmax."""
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.bool()
    keep = attention_mask.clone()
    keep[:, 0] = True
    return keep


class TimeAwareAttention(nn.Module):
    """``q,k,v = MLP(x)``; ``scores = q k^T / sqrt(d_head) + log_w``; padding mask; softmax.

    ``log_w`` is injected **directly** in log space (draft Eq. 3), not through
    ``logsigmoid`` as the legacy code did. ``||alpha||_1`` bounds the bias magnitude since
    ``|T_k| <= 1``; it is monitored but not regularised. A kernel that grows to dominate the
    QK term is a finding, not a bug.
    """

    def __init__(self, d_in: int, d_model: int, *, n_heads: int = 1, s: int = DEFAULT_S,
                 tau_max: float = 6.5, generator_mode: str = "none", age_M: int = 16,
                 age_p_min: float = 0.15, age_p_max: float = 6.0, age_hidden: int = 64,
                 gen_final_bias: bool = False, center_delta_alpha: bool = False,
                 use_out_proj: bool = False) -> None:
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

        self.kernel = ChebyshevKernel(s=s, tau_max=tau_max)
        self.alpha_base = nn.Parameter(torch.zeros(s))
        self.age = AgeConditioner(
            out_dim=s, M=age_M, p_min=age_p_min, p_max=age_p_max, hidden_dim=age_hidden,
            mode=generator_mode, final_bias=gen_final_bias,
            center_delta_alpha=center_delta_alpha,
        )

    # -- declared parameter sets (D6): membership, never name matching ------- #
    def age_parameters(self) -> list[nn.Parameter]:
        return self.age.age_parameters()

    def reset_raw_parameters_(self, gen: torch.Generator) -> None:
        """Raw (non-Linear, non-LayerNorm) parameters owned directly by this module."""
        with torch.no_grad():
            self.alpha_base.zero_()

    def alpha(self, age_years: torch.Tensor) -> torch.Tensor:
        """``alpha_base + Delta-alpha(a)`` -> ``[B, L, s]``."""
        return self.alpha_base + self.age(age_years)

    def forward(self, x: torch.Tensor, tau: torch.Tensor, attention_mask: torch.Tensor,
                age_years: torch.Tensor, *, need_weights: bool = False):
        b, l, _ = x.shape
        q = self.mlp_q(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        k = self.mlp_k(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)
        v = self.mlp_v(x).view(b, l, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, L, L]

        # alpha[..., k:k+1] is [B, L, 1] and broadcasts along the KEY axis, so row i is
        # conditioned on the query age a_i. Conditioning on the key age would place a
        # different kernel shape in every entry of one softmax row, making the weights
        # incomparable.
        log_w = self.kernel(tau, self.alpha(age_years))  # [B, L, L]
        scores = scores + log_w.unsqueeze(1)

        pair_mask = build_pair_mask(attention_mask).unsqueeze(1)  # [B, 1, L, L]
        scores = scores.masked_fill(~pair_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(~pair_mask, 0.0)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, l, self.d_model)
        if self.out_proj is not None:
            out = self.out_proj(out)
        if need_weights:
            return out, attn, log_w
        return out


class EncoderBlock(nn.Module):
    """Pre-LayerNorm block: ``x = x + Attn(LN(x))``, ``x = x + FFN(LN(x))``.

    Each of ``use_residual`` / ``use_layernorm`` / ``use_ffn`` is individually disableable.
    When the attention sub-block changes width (layer 0 maps the 1024-d frozen embeddings to
    ``d_model``) the residual uses a bias-free linear shortcut.
    """

    def __init__(self, d_in: int, d_model: int, *, use_residual: bool = True,
                 use_layernorm: bool = True, use_ffn: bool = True, ffn_mult: int = 4,
                 **attn_kwargs) -> None:
        super().__init__()
        self.d_in, self.d_model = int(d_in), int(d_model)
        self.use_residual = bool(use_residual)
        self.use_layernorm = bool(use_layernorm)
        self.use_ffn = bool(use_ffn)

        self.attn = TimeAwareAttention(d_in, d_model, **attn_kwargs)
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

    def age_parameters(self) -> list[nn.Parameter]:
        return self.attn.age_parameters()

    def forward(self, x: torch.Tensor, tau: torch.Tensor, attention_mask: torch.Tensor,
                age_years: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x) if self.ln_attn is not None else x
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
        return x


class Encoder(nn.Module):
    """``n_layers`` blocks. The kernel is applied at every layer with the same per-query age."""

    def __init__(self, d_in: int, d_model: int, *, n_layers: int = 1, **block_kwargs) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = int(n_layers)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_in if i == 0 else d_model, d_model, **block_kwargs)
            for i in range(self.n_layers)
        ])

    def age_parameters(self) -> list[nn.Parameter]:
        return [p for blk in self.blocks for p in blk.age_parameters()]

    def kernel_sites(self) -> list[tuple[str, TimeAwareAttention]]:
        return [(f"encoder_layer{i}", blk.attn) for i, blk in enumerate(self.blocks)]

    def forward(self, x: torch.Tensor, tau: torch.Tensor, attention_mask: torch.Tensor,
                age_years: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, tau, attention_mask, age_years)
        return x


def _smoke() -> None:
    from model_new import diagnostics
    from model_new.data import pairwise_tau   # the one lag convention; never re-derived here

    torch.manual_seed(0)
    b, l, d_in, d_model = 3, 7, 64, 32
    x = torch.randn(b, l, d_in)
    t = torch.rand(b, l).cumsum(dim=1) * 30.0
    mask = torch.ones(b, l, dtype=torch.bool)
    mask[1, 4:] = False
    mask[2, 1:] = False
    tau = pairwise_tau(t, mask)
    ages = torch.rand(b, l) * 80.0

    lines = []
    for legacy in (True, False):
        enc = Encoder(d_in, d_model, n_layers=1, use_residual=not legacy,
                      use_layernorm=not legacy, use_ffn=not legacy, generator_mode="real")
        out = enc(x, tau, mask, ages)
        n = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        lines.append(f"{'legacy block ' if legacy else 'standard block'}: out={tuple(out.shape)} "
                     f"finite={bool(torch.isfinite(out).all())} params={n:,}")

    enc = Encoder(d_in, d_model, n_layers=1, use_residual=False, use_layernorm=False,
                  use_ffn=False, generator_mode="real")
    # The generator's final layer is zero-initialised, so Delta-alpha is identically zero and
    # age cannot move anything at init. Wake the pathway up before probing INV-QUERY.
    with torch.no_grad():
        enc.blocks[0].attn.age.generator.mlp[-1].weight.normal_(0.0, 0.5)
    with torch.no_grad():
        base = enc(x, tau, mask, ages)
        bumped = ages.clone()
        bumped[:, 2] += 5.0
        diff = (enc(x, tau, mask, bumped) - base).abs().amax(dim=-1)  # [B, L]
    changed = (diff > 0).float().sum(dim=0)
    lines.append(f"INV-QUERY rows changed by perturbing age[:, 2]: "
                 f"{[int(c) for c in changed]}  (expect nonzero only at index 2)")
    lines.append(f"pair mask diagonal always True: "
                 f"{bool(build_pair_mask(mask).diagonal(dim1=-2, dim2=-1).all())}")
    diagnostics.print_block("encoder.py smoke", lines)


if __name__ == "__main__":
    _smoke()
