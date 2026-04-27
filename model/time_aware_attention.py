#!/usr/bin/env python3
"""Time-aware attention and multi-scale temporal aggregation (TALE-EHR, Sec. 3.1–3.2)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolynomialTemporalWeight(nn.Module):
    """Learnable polynomial in t passed through sigmoid (Eq. 4 temporal factor w(t))."""

    def __init__(self, poly_degree: int = 5) -> None:
        super().__init__()
        self.poly_degree = int(poly_degree)
        coeffs = torch.zeros(self.poly_degree + 1)
        coeffs[0] = 0.5
        self.coefficients = nn.Parameter(coeffs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        powers = [torch.ones_like(t)]
        cur = t
        for _ in range(self.poly_degree):
            powers.append(cur)
            cur = cur * t
        poly = torch.zeros_like(t)
        for k in range(self.poly_degree + 1):
            poly = poly + self.coefficients[k] * powers[k]
        return torch.sigmoid(poly)


class TimeAwareAttention(nn.Module):
    """Single-head time-aware attention over events (Eq. 3–4)."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        d_model: int = 256,
        poly_degree: int = 5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.mlp_q = nn.Sequential(nn.Linear(embedding_dim, d_model), nn.GELU())
        self.mlp_k = nn.Sequential(nn.Linear(embedding_dim, d_model), nn.GELU())
        self.mlp_v = nn.Sequential(nn.Linear(embedding_dim, d_model), nn.GELU())
        self.temporal_weight = PolynomialTemporalWeight(poly_degree)

    def forward(
        self,
        code_embeddings: torch.Tensor,
        delta_t: torch.Tensor,
        attention_mask: torch.Tensor,
        debug_sample: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            code_embeddings: [B, L, embedding_dim]
            delta_t: [B, L, L] log1p pairwise |tj - tk| in days (0 for padding)
            attention_mask: [B, L] bool, True = real event
        Returns:
            E: [B, L, d_model]
        """
        q = self.mlp_q(code_embeddings)
        k = self.mlp_k(code_embeddings)
        v = self.mlp_v(code_embeddings)

        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale
        w = self.temporal_weight(delta_t)
        scores = scores * w

        b, l, _ = scores.shape
        device, dtype = scores.device, scores.dtype
        causal = torch.tril(torch.ones((l, l), device=device, dtype=torch.bool))
        pad_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        full_mask = pad_mask

        scores = scores.masked_fill(~full_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(~full_mask, 0.0)
        if debug_sample:
            with torch.no_grad():
                eps = 1e-8
                attn_valid = attn[attention_mask]
                if attn_valid.numel() > 0:
                    entropy = -(attn_valid * torch.log(attn_valid.clamp_min(eps))).sum(dim=-1)
                    max_w = attn_valid.max(dim=-1).values
                    print(
                        f"[attn] entropy_mean={float(entropy.mean()):.3f} "
                        f"max_w_mean={float(max_w.mean()):.3f} "
                        f"collapse_frac={float((max_w > 0.9).float().mean()):.3f}",
                        flush=True,
                    )

                w_valid = w[full_mask]
                if w_valid.numel() > 0:
                    print(
                        f"[w(t)] mean={float(w_valid.mean()):.3f} "
                        f"std={float(w_valid.std(unbiased=False)):.3f} "
                        f"low_frac={float((w_valid < 0.1).float().mean()):.3f} "
                        f"high_frac={float((w_valid > 0.9).float().mean()):.3f}",
                        flush=True,
                    )

        e = torch.bmm(attn, v)
        if debug_sample:
            with torch.no_grad():
                e_valid = e[attention_mask]
                if e_valid.numel() > 0:
                    print(
                        f"[e_out] mean_abs={float(e_valid.abs().mean()):.3f} "
                        f"std={float(e_valid.std(unbiased=False)):.3f} "
                        f"dead_frac={float((e_valid.abs() < 0.01).float().mean()):.3f}",
                        flush=True,
                    )
        return e


class MultiScaleTemporalAggregation(nn.Module):
    """Aggregate event representations using semantic relevance and temporal weighting (Sec. 3.2)."""

    def __init__(self, d_model: int = 256, poly_degree: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.q_base = nn.Parameter(torch.randn(d_model) * 0.02)
        self.temporal_weight = PolynomialTemporalWeight(poly_degree)

    def forward(
        self,
        e: torch.Tensor,
        timestamps_days: torch.Tensor,
        attention_mask: torch.Tensor,
        debug_sample: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            e: [B, L, d_model]
            timestamps_days: [B, L] raw days
            attention_mask: [B, L] bool
        Returns:
            h: [B, d_model]
        """
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        b = e.shape[0]
        batch_idx = torch.arange(b, device=e.device)
        t_current = timestamps_days[batch_idx, lengths - 1]

        delta_to_current = torch.abs(t_current.unsqueeze(1) - timestamps_days)
        #log_delta = torch.log1p(delta_to_current)
        log_delta = torch.log1p(delta_to_current / 7.0)  # match collator's units

        relevance = torch.einsum("d, bld -> bl", self.q_base, e)
        w = self.temporal_weight(log_delta)

        scores = relevance * w
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(~attention_mask, 0.0)
        if debug_sample:
            with torch.no_grad():
                eps = 1e-8
                lengths_f = lengths.clamp(min=1).to(dtype=alpha.dtype)
                alpha_peak_idx = alpha.argmax(dim=-1).to(dtype=alpha.dtype)
                peak_rel_pos = alpha_peak_idx / lengths_f
                
                # Per-row entropy: alpha is [B, L], compute entropy along L then mean over B
                row_entropy = -(alpha * torch.log(alpha.clamp_min(eps))).sum(dim=-1)  # [B]
                print(
                    f"[agg_alpha] entropy={float(row_entropy.mean()):.3f} "
                    f"peak_rel_pos={float(peak_rel_pos.mean()):.3f}",
                    flush=True,
                )

        h = torch.einsum("bl, bld -> bd", alpha, e)
        if debug_sample:
            with torch.no_grad():
                print(
                    f"[h_repr] mean_abs={float(h.abs().mean()):.3f} "
                    f"std={float(h.std(unbiased=False)):.3f}",
                    flush=True,
                )
        return h


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    bsz, seq_len, emb_dim, d_model = 2, 10, 1024, 256

    code_embeddings = torch.randn(bsz, seq_len, emb_dim)
    attention_mask = torch.tensor(
        [[True] * 10, [True] * 7 + [False] * 3],
        dtype=torch.bool,
    )
    timestamps_days = torch.stack(
        [
            torch.linspace(0.0, 9.0, seq_len),
            torch.cat([torch.linspace(10.0, 16.0, 7), torch.zeros(3)]),
        ]
    )

    delta_t = torch.log1p(torch.abs(timestamps_days.unsqueeze(2) - timestamps_days.unsqueeze(1)))
    pair_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    delta_t = delta_t * pair_mask.float()

    attn_mod = TimeAwareAttention(emb_dim, d_model, poly_degree=5)
    agg_mod = MultiScaleTemporalAggregation(d_model, poly_degree=5)

    e = attn_mod(code_embeddings, delta_t, attention_mask)
    h = agg_mod(e, timestamps_days, attention_mask)

    print("E shape:", tuple(e.shape), "dtype:", e.dtype)
    print("h shape:", tuple(h.shape), "dtype:", h.dtype)
    assert e.shape == (bsz, seq_len, d_model)
    assert h.shape == (bsz, d_model)
    assert torch.isfinite(e).all() and torch.isfinite(h).all()

    tw = PolynomialTemporalWeight(5)
    t_sample = torch.linspace(0.0, 3.0, 100)
    with torch.no_grad():
        w_out = tw(t_sample)
    print("PolynomialTemporalWeight output min/max:", float(w_out.min()), float(w_out.max()))
    assert float(w_out.min()) > 0.0 and float(w_out.max()) < 1.0

    print("PolynomialTemporalWeight params:", _count_params(tw))
    print("TimeAwareAttention params:", _count_params(attn_mod))
    print("MultiScaleTemporalAggregation params:", _count_params(agg_mod))
    print("Smoke test passed.")
