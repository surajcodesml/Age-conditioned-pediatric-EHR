#!/usr/bin/env python3
"""Attention pooling over the encoded event sequence -- the second kernel site.

A single query at the present. ``a_n`` is the age at the last valid event and ``tau_to_now``
the lag from each event to that point (computed once, in the collate):

    relevance = q_base . e
    log_w     = ChebyshevKernel(tau_to_now, alpha(a_n))
    scores    = relevance + log_w
    h         = softmax(scores over valid) @ e

D5 -- the legacy pooling used ``scores = relevance * w`` on a **signed** relevance, so a
kernel factor in (0, 1) *raised* attention on any event with a negative relevance score.
The fix is the same log-space injection the attention site uses.

This site owns a separate ``alpha_base`` and a separate generator instance from the encoder
sites: the draft states the two sites share form and nothing else.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_new.age_encoding import AgeConditioner
from model_new.basis import DEFAULT_S, ChebyshevKernel
from model_new.encoder import build_key_mask

__all__ = ["AttentionPooling"]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, *, s: int = DEFAULT_S, tau_max: float = 6.5,
                 generator_mode: str = "none", age_M: int = 16, age_p_min: float = 0.15,
                 age_p_max: float = 6.0, age_hidden: int = 64, gen_final_bias: bool = False,
                 center_delta_alpha: bool = False) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.q_base = nn.Parameter(torch.zeros(self.d_model))
        self.kernel = ChebyshevKernel(s=s, tau_max=tau_max)
        self.alpha_base = nn.Parameter(torch.zeros(s))
        self.age = AgeConditioner(
            out_dim=s, M=age_M, p_min=age_p_min, p_max=age_p_max, hidden_dim=age_hidden,
            mode=generator_mode, final_bias=gen_final_bias,
            center_delta_alpha=center_delta_alpha,
        )
        self.reset_raw_parameters_(torch.Generator().manual_seed(0))

    def age_parameters(self) -> list[nn.Parameter]:
        return self.age.age_parameters()

    def reset_raw_parameters_(self, gen: torch.Generator) -> None:
        with torch.no_grad():
            self.alpha_base.zero_()
            self.q_base.copy_(torch.randn(self.d_model, generator=gen) * 0.02)

    @staticmethod
    def last_valid_index(attention_mask: torch.Tensor) -> torch.Tensor:
        """Index of the last valid event per row. Raises on length-zero sequences rather
        than letting ``lengths - 1`` wrap to the end of the sequence."""
        lengths = attention_mask.bool().sum(dim=1).long()
        if bool((lengths == 0).any()):
            bad = int((lengths == 0).nonzero()[0])
            raise ValueError(f"zero-length sequence at batch row {bad}: pooling is undefined")
        return lengths - 1

    def alpha(self, age_last: torch.Tensor) -> torch.Tensor:
        """``alpha_base + Delta-alpha(a_n)`` -> ``[B, s]``."""
        return self.alpha_base + self.age(age_last)

    def forward(self, e: torch.Tensor, tau_to_now: torch.Tensor, attention_mask: torch.Tensor,
                age_last: torch.Tensor, *, need_weights: bool = False):
        relevance = torch.einsum("d,bld->bl", self.q_base, e)
        log_w = self.kernel(tau_to_now, self.alpha(age_last))  # [B, L]
        scores = relevance + log_w

        keep = build_key_mask(attention_mask)
        scores = scores.masked_fill(~keep, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~keep, 0.0)

        h = torch.einsum("bl,bld->bd", attn, e)
        if need_weights:
            return h, attn, log_w
        return h


def _smoke() -> None:
    from model_new import diagnostics

    torch.manual_seed(0)
    b, l, d = 4, 9, 16
    e = torch.randn(b, l, d)
    tau_to_now = torch.rand(b, l) * 6.0
    mask = torch.ones(b, l, dtype=torch.bool)
    mask[1, 5:] = False
    ages = torch.tensor([0.5, 3.0, 40.0, 70.0])

    pool = AttentionPooling(d, generator_mode="real")
    with torch.no_grad():
        h, attn, log_w = pool(e, tau_to_now, mask, ages, need_weights=True)

    lines = [
        f"h shape                    : {tuple(h.shape)}",
        f"attn rows sum to 1         : {bool(torch.allclose(attn.sum(-1), torch.ones(b), atol=1e-6))}",
        f"attn zero on padding       : {float(attn[~mask].abs().max()):.3e}",
        f"log_w at init max|.|       : {float(log_w.abs().max()):.3e}  (expect 0)",
        f"last valid index           : {[int(i) for i in pool.last_valid_index(mask)]}",
    ]
    empty = mask.clone()
    empty[2] = False
    try:
        pool.last_valid_index(empty)
        lines.append("zero-length sequence        : NOT RAISED  <-- bug")
    except ValueError as exc:
        lines.append(f"zero-length sequence raises: {exc}")
    diagnostics.print_block("pooling.py smoke", lines)


if __name__ == "__main__":
    _smoke()
