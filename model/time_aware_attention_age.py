#!/usr/bin/env python3
"""Age-conditioned time-aware attention."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.age_embedding import AgeCoefficientGenerator, FourierAgeEmbedding
from model.time_aware_attention import TimeAwareAttention


def _forward_with_causal_mask(
    baseline: TimeAwareAttention,
    code_embeddings: torch.Tensor,
    delta_t: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference baseline forward with explicit causal masking."""
    q = baseline.mlp_q(code_embeddings)
    k = baseline.mlp_k(code_embeddings)
    v = baseline.mlp_v(code_embeddings)
    scale = 1.0 / math.sqrt(baseline.d_model)
    scores = torch.bmm(q, k.transpose(-1, -2)) * scale
    poly = baseline.temporal_weight.poly_value(delta_t)
    w = torch.sigmoid(poly)
    if baseline.kernel_injection == "multiplicative":
        scores = scores * w
    else:  # "additive_logspace"
        scores = scores + F.logsigmoid(poly)
    _, l, _ = scores.shape
    causal = torch.tril(torch.ones((l, l), device=scores.device, dtype=torch.bool))
    pad_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
    full_mask = pad_mask & causal.unsqueeze(0)
    scores = scores.masked_fill(~full_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn = attn.masked_fill(~full_mask, 0.0)
    return torch.bmm(attn, v)


class AgeConditionedPolynomialWeight(nn.Module):
    """Polynomial temporal decay with query-age-conditioned coefficients."""

    def __init__(
        self,
        poly_degree: int = 5,
        age_emb_dim: int = 32,
        hidden_dim: int = 64,
        age_conditioning_mode: str = "real",
    ) -> None:
        super().__init__()
        self.poly_degree = int(poly_degree)
        coeffs = torch.zeros(self.poly_degree + 1)
        coeffs[0] = 0.5
        self.coefficients = nn.Parameter(coeffs)
        self.age_coeff_gen = AgeCoefficientGenerator(
            in_dim=age_emb_dim,
            hidden_dim=hidden_dim,
            out_dim=self.poly_degree + 1,
            mode=age_conditioning_mode,
        )

    def _poly(self, delta_t: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        alpha_delta = self.age_coeff_gen(age_features)
        alpha = self.coefficients + alpha_delta

        powers = [torch.ones_like(delta_t)]
        cur = delta_t
        for _ in range(self.poly_degree):
            powers.append(cur)
            cur = cur * delta_t

        poly = torch.zeros_like(delta_t)
        for k in range(self.poly_degree + 1):
            poly = poly + alpha[..., k : k + 1] * powers[k]
        return poly

    def poly_value(self, delta_t: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        return self._poly(delta_t, age_features)

    def forward(self, delta_t: torch.Tensor, age_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._poly(delta_t, age_features))


class AgeConditionedMultiScaleTemporalAggregation(nn.Module):
    """Age-conditioned temporal aggregation with current-age coefficients."""

    def __init__(
        self,
        d_model: int = 256,
        poly_degree: int = 5,
        age_emb_dim: int = 32,
        age_hidden_dim: int = 64,
        age_conditioning_mode: str = "real",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.q_base = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.temporal_weight = AgeConditionedPolynomialWeight(
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            hidden_dim=age_hidden_dim,
            age_conditioning_mode=age_conditioning_mode,
        )
        self.age_emb = FourierAgeEmbedding(num_frequencies=age_emb_dim // 2)

    def forward(
        self,
        e: torch.Tensor,
        timestamps_days: torch.Tensor,
        attention_mask: torch.Tensor,
        age_years: torch.Tensor,
        debug_sample: bool = False,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        b = e.shape[0]
        batch_idx = torch.arange(b, device=e.device)
        t_current = timestamps_days[batch_idx, lengths - 1]
        age_current = age_years[batch_idx, lengths - 1].clamp(min=0.0)
        age_features = self.age_emb(age_current)

        delta_to_current = torch.abs(t_current.unsqueeze(1) - timestamps_days)
        log_delta = torch.log1p(delta_to_current / 7.0)

        relevance = torch.einsum("d, bld -> bl", self.q_base, e)
        w = self.temporal_weight(log_delta, age_features)

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

                row_entropy = -(alpha * torch.log(alpha.clamp_min(eps))).sum(dim=-1)
                print(
                    f"[agg_alpha] entropy={float(row_entropy.mean()):.3f} "
                    f"peak_rel_pos={float(peak_rel_pos.mean()):.3f}",
                    flush=True,
                )
                w_valid = w[attention_mask]
                if w_valid.numel() > 0:
                    print(
                        f"[agg_w(t)] mean={float(w_valid.mean()):.3f} "
                        f"std={float(w_valid.std(unbiased=False)):.3f} "
                        f"low_frac={float((w_valid < 0.1).float().mean()):.3f} "
                        f"high_frac={float((w_valid > 0.9).float().mean()):.3f}",
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


class AgeConditionedTimeAwareAttention(nn.Module):
    """Single-head attention with age-conditioned polynomial temporal weighting."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        d_model: int = 256,
        poly_degree: int = 5,
        age_emb_dim: int = 32,
        age_hidden_dim: int = 64,
        age_conditioning_mode: str = "real",
        kernel_injection: str = "additive_logspace",
    ) -> None:
        super().__init__()
        if kernel_injection not in {"multiplicative", "additive_logspace"}:
            raise ValueError(
                f"kernel_injection must be 'multiplicative' or 'additive_logspace', got {kernel_injection!r}"
            )
        self.kernel_injection = kernel_injection
        self.embedding_dim = int(embedding_dim)
        self.d_model = int(d_model)

        self.mlp_q = nn.Sequential(nn.Linear(self.embedding_dim, self.d_model), nn.GELU())
        self.mlp_k = nn.Sequential(nn.Linear(self.embedding_dim, self.d_model), nn.GELU())
        self.mlp_v = nn.Sequential(nn.Linear(self.embedding_dim, self.d_model), nn.GELU())
        self.temporal_weight = AgeConditionedPolynomialWeight(
            poly_degree=poly_degree,
            age_emb_dim=age_emb_dim,
            hidden_dim=age_hidden_dim,
            age_conditioning_mode=age_conditioning_mode,
        )
        self.age_emb = FourierAgeEmbedding(num_frequencies=age_emb_dim // 2)

    def forward(
        self,
        code_embeddings: torch.Tensor,
        delta_t: torch.Tensor,
        attention_mask: torch.Tensor,
        age_years: torch.Tensor,
        debug_sample: bool = False,
    ) -> torch.Tensor:
        q = self.mlp_q(code_embeddings)
        k = self.mlp_k(code_embeddings)
        v = self.mlp_v(code_embeddings)
        age_features = self.age_emb(age_years.clamp(min=0.0))

        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.bmm(q, k.transpose(-1, -2)) * scale
        poly = self.temporal_weight.poly_value(delta_t, age_features)
        w = torch.sigmoid(poly)
        if self.kernel_injection == "multiplicative":
            scores = scores * w
        else:  # "additive_logspace"
            scores = scores + F.logsigmoid(poly)

        _, l, _ = scores.shape
        device = scores.device
        causal = torch.tril(torch.ones((l, l), device=device, dtype=torch.bool))
        pad_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        full_mask = pad_mask & causal.unsqueeze(0)

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
                qk_stats = self.attention_qk_norm_stats(code_embeddings, attention_mask)
                print(
                    f"[attn_qk] q_norm_mean={qk_stats['q_norm_mean']:.3f} "
                    f"q_norm_std={qk_stats['q_norm_std']:.3f} "
                    f"k_norm_mean={qk_stats['k_norm_mean']:.3f} "
                    f"k_norm_std={qk_stats['k_norm_std']:.3f}",
                    flush=True,
                )
                e_valid = e[attention_mask]
                if e_valid.numel() > 0:
                    print(
                        f"[e_out] mean_abs={float(e_valid.abs().mean()):.3f} "
                        f"std={float(e_valid.std(unbiased=False)):.3f} "
                        f"dead_frac={float((e_valid.abs() < 0.01).float().mean()):.3f}",
                        flush=True,
                    )
        return e

    @torch.no_grad()
    def attention_qk_norm_stats(
        self,
        code_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Per-token L2 norms of attention-path q/k over valid (non-pad) positions."""
        q = self.mlp_q(code_embeddings)
        k = self.mlp_k(code_embeddings)
        if not attention_mask.any():
            return {
                "q_norm_mean": float("nan"),
                "q_norm_std": float("nan"),
                "k_norm_mean": float("nan"),
                "k_norm_std": float("nan"),
            }
        q_norms = q[attention_mask].float().norm(dim=-1)
        k_norms = k[attention_mask].float().norm(dim=-1)
        return {
            "q_norm_mean": float(q_norms.mean()),
            "q_norm_std": float(q_norms.std(unbiased=False)),
            "k_norm_mean": float(k_norms.mean()),
            "k_norm_std": float(k_norms.std(unbiased=False)),
        }


def _make_inputs(
    bsz: int = 2,
    seq_len: int = 10,
    embedding_dim: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    code_embeddings = torch.randn(bsz, seq_len, embedding_dim)
    attention_mask = torch.tensor(
        [[True] * seq_len, [True] * (seq_len - 3) + [False] * 3],
        dtype=torch.bool,
    )
    timestamps_days = torch.stack(
        [
            torch.linspace(0.0, float(seq_len - 1), seq_len),
            torch.cat([torch.linspace(10.0, 16.0, seq_len - 3), torch.zeros(3)]),
        ]
    )
    delta_t = torch.log1p(torch.abs(timestamps_days.unsqueeze(2) - timestamps_days.unsqueeze(1)) / 7.0)
    pair_mask = attention_mask.unsqueeze(2) & attention_mask.unsqueeze(1)
    delta_t = delta_t * pair_mask.float()
    return code_embeddings, delta_t, attention_mask


if __name__ == "__main__":
    torch.manual_seed(0)
    bsz, seq_len, emb_dim, d_model = 2, 10, 1024, 256
    age_emb_dim = 32
    age_hidden_dim = 64
    code_embeddings, delta_t, attention_mask = _make_inputs(
        bsz=bsz, seq_len=seq_len, embedding_dim=emb_dim
    )

    poly_mod = AgeConditionedPolynomialWeight(
        poly_degree=5,
        age_emb_dim=age_emb_dim,
        hidden_dim=age_hidden_dim,
        age_conditioning_mode="real",
    )
    dt_3d = torch.rand(3, 7, 7)
    age_feat_3d = torch.randn(3, 7, age_emb_dim)
    w_3d = poly_mod(dt_3d, age_feat_3d)
    assert w_3d.shape == (3, 7, 7)
    dt_2d = torch.rand(3, 7)
    age_feat_2d = torch.randn(3, age_emb_dim)
    w_2d = poly_mod(dt_2d, age_feat_2d)
    assert w_2d.shape == (3, 7)

    baseline = TimeAwareAttention(embedding_dim=emb_dim, d_model=d_model, poly_degree=5)
    age_real = AgeConditionedTimeAwareAttention(
        embedding_dim=emb_dim,
        d_model=d_model,
        poly_degree=5,
        age_emb_dim=age_emb_dim,
        age_hidden_dim=age_hidden_dim,
        age_conditioning_mode="real",
    )
    with torch.no_grad():
        age_real.mlp_q.load_state_dict(baseline.mlp_q.state_dict())
        age_real.mlp_k.load_state_dict(baseline.mlp_k.state_dict())
        age_real.mlp_v.load_state_dict(baseline.mlp_v.state_dict())
        age_real.temporal_weight.coefficients.copy_(baseline.temporal_weight.coefficients)

    age_years_mid = torch.full((bsz, seq_len), 50.0)
    out_base = _forward_with_causal_mask(baseline, code_embeddings, delta_t, attention_mask)
    out_real = age_real(code_embeddings, delta_t, attention_mask, age_years_mid)
    assert torch.allclose(out_base, out_real, atol=1e-5, rtol=1e-5)

    age_none = AgeConditionedTimeAwareAttention(
        embedding_dim=emb_dim,
        d_model=d_model,
        poly_degree=5,
        age_emb_dim=age_emb_dim,
        age_hidden_dim=age_hidden_dim,
        age_conditioning_mode="none",
    )
    with torch.no_grad():
        age_none.mlp_q.load_state_dict(baseline.mlp_q.state_dict())
        age_none.mlp_k.load_state_dict(baseline.mlp_k.state_dict())
        age_none.mlp_v.load_state_dict(baseline.mlp_v.state_dict())
        age_none.temporal_weight.coefficients.copy_(baseline.temporal_weight.coefficients)
        age_none.temporal_weight.age_coeff_gen.mlp[-1].weight.copy_(
            torch.randn_like(age_none.temporal_weight.age_coeff_gen.mlp[-1].weight)
        )
        age_none.temporal_weight.age_coeff_gen.mlp[-1].bias.copy_(
            torch.randn_like(age_none.temporal_weight.age_coeff_gen.mlp[-1].bias)
        )
    out_none = age_none(code_embeddings, delta_t, attention_mask, age_years_mid)
    assert torch.allclose(out_base, out_none, atol=1e-6, rtol=1e-6)

    age_real_sens = AgeConditionedTimeAwareAttention(
        embedding_dim=emb_dim,
        d_model=d_model,
        poly_degree=5,
        age_emb_dim=age_emb_dim,
        age_hidden_dim=age_hidden_dim,
        age_conditioning_mode="real",
    )
    with torch.no_grad():
        age_real_sens.mlp_q.load_state_dict(baseline.mlp_q.state_dict())
        age_real_sens.mlp_k.load_state_dict(baseline.mlp_k.state_dict())
        age_real_sens.mlp_v.load_state_dict(baseline.mlp_v.state_dict())
        age_real_sens.temporal_weight.coefficients.copy_(baseline.temporal_weight.coefficients)
        age_real_sens.temporal_weight.age_coeff_gen.mlp[-1].weight.copy_(
            torch.randn_like(age_real_sens.temporal_weight.age_coeff_gen.mlp[-1].weight) * 0.1
        )
        age_real_sens.temporal_weight.age_coeff_gen.mlp[-1].bias.copy_(
            torch.randn_like(age_real_sens.temporal_weight.age_coeff_gen.mlp[-1].bias) * 0.1
        )
    out_young = age_real_sens(
        code_embeddings, delta_t, attention_mask, torch.full((bsz, seq_len), 2.0)
    )
    out_old = age_real_sens(
        code_embeddings, delta_t, attention_mask, torch.full((bsz, seq_len), 65.0)
    )
    assert (out_young - out_old).abs().max().detach().item() > 1e-3

    age_rand = AgeConditionedTimeAwareAttention(
        embedding_dim=emb_dim,
        d_model=d_model,
        poly_degree=5,
        age_emb_dim=age_emb_dim,
        age_hidden_dim=age_hidden_dim,
        age_conditioning_mode="random_constant",
    )
    with torch.no_grad():
        age_rand.mlp_q.load_state_dict(baseline.mlp_q.state_dict())
        age_rand.mlp_k.load_state_dict(baseline.mlp_k.state_dict())
        age_rand.mlp_v.load_state_dict(baseline.mlp_v.state_dict())
        age_rand.temporal_weight.coefficients.copy_(baseline.temporal_weight.coefficients)
        age_rand.temporal_weight.age_coeff_gen.mlp[-1].weight.copy_(
            torch.randn_like(age_rand.temporal_weight.age_coeff_gen.mlp[-1].weight) * 0.1
        )
        age_rand.temporal_weight.age_coeff_gen.mlp[-1].bias.copy_(
            torch.randn_like(age_rand.temporal_weight.age_coeff_gen.mlp[-1].bias) * 0.1
        )
    out_rand_young = age_rand(
        code_embeddings, delta_t, attention_mask, torch.full((bsz, seq_len), 2.0)
    )
    out_rand_old = age_rand(
        code_embeddings, delta_t, attention_mask, torch.full((bsz, seq_len), 65.0)
    )
    assert torch.allclose(out_rand_young, out_rand_old, atol=1e-6, rtol=1e-6)

    age_grad = AgeConditionedTimeAwareAttention(
        embedding_dim=emb_dim,
        d_model=d_model,
        poly_degree=5,
        age_emb_dim=age_emb_dim,
        age_hidden_dim=age_hidden_dim,
        age_conditioning_mode="real",
    )
    age_grad.zero_grad(set_to_none=True)
    out_grad = age_grad(code_embeddings, delta_t, attention_mask, age_years_mid)
    loss = out_grad.sum()
    loss.backward()
    grad = age_grad.temporal_weight.age_coeff_gen.mlp[-1].weight.grad
    assert grad is not None
    assert float(grad.norm()) > 0.0

    out_shape = age_real(code_embeddings, delta_t, attention_mask, age_years_mid)
    assert out_shape.shape == (bsz, seq_len, d_model)
    assert torch.isfinite(out_shape).all()

    agg = AgeConditionedMultiScaleTemporalAggregation(
        d_model=d_model,
        poly_degree=5,
        age_emb_dim=age_emb_dim,
        age_hidden_dim=age_hidden_dim,
        age_conditioning_mode="real",
    )
    with torch.no_grad():
        agg.temporal_weight.age_coeff_gen.mlp[-1].weight.copy_(
            torch.randn_like(agg.temporal_weight.age_coeff_gen.mlp[-1].weight) * 0.1
        )
        agg.temporal_weight.age_coeff_gen.mlp[-1].bias.copy_(
            torch.randn_like(agg.temporal_weight.age_coeff_gen.mlp[-1].bias) * 0.1
        )
    h_young = agg(e=out_shape, timestamps_days=torch.arange(seq_len).float().repeat(bsz, 1), attention_mask=attention_mask, age_years=torch.full((bsz, seq_len), 2.0))
    h_old = agg(e=out_shape, timestamps_days=torch.arange(seq_len).float().repeat(bsz, 1), attention_mask=attention_mask, age_years=torch.full((bsz, seq_len), 65.0))
    assert h_young.shape == (bsz, d_model)
    assert (h_young - h_old).abs().max().detach().item() > 1e-3

    print("Smoke test passed.")
