"""Content-Persistence Developmental Temporal Retrieval (DTR) — locked canonical.

Architecture (locked):
  v_m = f_enc(C_m)                         # encounter content only
  u_m = qᵀ k_m                             # content relevance (no age/τ)
  θ_m = θ₀ + rᵀ v_m                        # content-dependent persistence
  λ_m(a*) = softplus(θ_m + β z(a*))        # developmental gate scale
  g_m = exp[-λ_m(a*) τ_m]
  w_m = exp(u_m) g_m
  h = Σ_m w_m v_m                          # raw additive aggregation
  ℓ = f_history(h) + f_age(z(a*)) + b      # structurally additive head

Age modifies historical relevance ONLY through β z(a*) inside λ_m.

NOTE: Multi-horizon supervision is currently experimental and is not part of
the locked canonical architecture because naive joint training with content
persistence inverted beta on the synthetic S2 benchmark.
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AGE_CENTER, AGE_SCALE, PROBE_AGES

# Canonical defaults (locked).
CANONICAL_AGGREGATION = "raw_additive"
CANONICAL_NUM_CONTENT_QUERIES = 1
CANONICAL_AGE_CONDITIONING = "linear_softplus"
CANONICAL_BETA_SCOPE = "global"
CANONICAL_PERSISTENCE_PROJECTION = "linear"

# U_MAX clamp on content scores before exp(u). Preserves ordering for typical
# scores; prevents overflow of exp(u) without replacing raw-additive Σ w v
# with softmax. Documented in forward().
CONTENT_SCORE_EXP_CLAMP = 20.0

AGGREGATIONS = ("raw_additive", "weighted_mean_plus_log_mass")


def z_of_age(age: torch.Tensor) -> torch.Tensor:
    return (age - AGE_CENTER) / AGE_SCALE


class EncounterEncoder(nn.Module):
    """f_enc(C_m): DeepSets mean-pool over codes → MLP. No age/τ."""

    def __init__(self, n_codes: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.code_emb = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.enc_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, enc_code_ids: torch.Tensor, enc_code_mask: torch.Tensor) -> torch.Tensor:
        e = self.code_emb(enc_code_ids)
        mask = enc_code_mask.to(e.dtype).unsqueeze(-1)
        summed = (e * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp(min=1.0)
        return self.enc_mlp(summed / denom)


class _GateFacade:
    """TemporalGate-compatible surface over Content-Persistence DTR parameters."""

    def __init__(self, parent: "DevelopmentalTemporalRetrieval") -> None:
        self._p = parent

    @property
    def theta0(self) -> nn.Parameter:
        return self._p.theta0

    @property
    def beta(self) -> nn.Parameter:
        return self._p.beta

    @property
    def age_temporal(self) -> bool:
        return self._p.age_temporal

    def z_of(self, age: torch.Tensor) -> torch.Tensor:
        return self._p.z_of(age)

    def lambda_of(
        self, age: torch.Tensor, theta: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Legacy TemporalGate: `theta` replaces θ₀ entirely.
        return self._p.lambda_of(age, theta=theta)

    def gate(
        self,
        age: torch.Tensor,
        tau: torch.Tensor,
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lam = self.lambda_of(age, theta=theta)
        if lam.dim() < tau.dim():
            lam = lam.unsqueeze(-1)
        return torch.exp(-lam * tau)

    def age_parameters(self) -> list[nn.Parameter]:
        return self._p.age_parameters()


class DevelopmentalTemporalRetrieval(nn.Module):
    """Content-Persistence DTR (canonical production architecture).

    Scientific roles:
      encounter_encoder        — what happened in encounter m (no age/τ)
      content_query/content_key — clinical content relevance u_m
      persistence_projection   — how persistent this content type is (rᵀ v_m)
      theta0                   — global baseline persistence
      beta                     — developmental age × persistence (global)
      history_head / age_head  — separated age main effect vs history
    """

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 64,
        age_temporal: bool = True,
        aggregation: str = CANONICAL_AGGREGATION,
        dropout: float = 0.0,
        max_codes_per_encounter: int = 32,
        content_persistence: bool = True,
        content_dependent_persistence: bool | None = None,
        multi_query_K: int = CANONICAL_NUM_CONTENT_QUERIES,
        persistence_projection: str = CANONICAL_PERSISTENCE_PROJECTION,
        temporal_aggregation: str | None = None,
        age_conditioning: str = CANONICAL_AGE_CONDITIONING,
        beta_scope: str = CANONICAL_BETA_SCOPE,
        num_content_queries: int | None = None,
    ) -> None:
        super().__init__()
        if temporal_aggregation is not None:
            aggregation = temporal_aggregation
        if content_dependent_persistence is not None:
            content_persistence = content_dependent_persistence
        if num_content_queries is not None:
            multi_query_K = num_content_queries

        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation}")
        if persistence_projection != "linear":
            raise ValueError("canonical persistence_projection must be 'linear'")
        if age_conditioning != "linear_softplus":
            raise ValueError("canonical age_conditioning must be 'linear_softplus'")
        if beta_scope != "global":
            raise ValueError("canonical beta_scope must be 'global'")
        if multi_query_K != 1:
            raise ValueError(
                "canonical Content-Persistence DTR uses num_content_queries=1; "
                "use DTRMultiQuery for the experimental multi-query ablation"
            )

        self.aggregation = aggregation
        self.temporal_aggregation = aggregation
        self.age_temporal = bool(age_temporal)
        self.d_model = d_model
        self.max_codes_per_encounter = max_codes_per_encounter
        self.content_persistence = bool(content_persistence)
        self.content_dependent_persistence = self.content_persistence
        self.multi_query_K = 1
        self.num_content_queries = 1
        self.persistence_projection_type = persistence_projection
        self.age_conditioning = age_conditioning
        self.beta_scope = beta_scope

        # --- Encounter encoder f_enc(C_m): content only ---
        self.encounter_encoder = EncounterEncoder(n_codes, d_model, dropout=dropout)

        # --- Content relevance: u_m = qᵀ k_m ---
        self.content_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.content_key = nn.Linear(d_model, d_model, bias=False)

        # --- Content persistence: θ_m = θ₀ + rᵀ v_m ---
        self.persistence_projection = nn.Linear(d_model, 1, bias=True)
        # Near-zero init so θ_m ≈ θ₀ at step 0 (close to global-persistence DTR).
        nn.init.zeros_(self.persistence_projection.weight)
        nn.init.zeros_(self.persistence_projection.bias)

        # --- Developmental parameters ---
        self.theta0 = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        if not self.age_temporal:
            self.beta.requires_grad_(False)

        # --- Structurally additive prediction heads ---
        hist_in = d_model + (1 if aggregation == "weighted_mean_plus_log_mass" else 0)
        self.history_head = nn.Sequential(
            nn.Linear(hist_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_targets),
        )
        self.age_head = nn.Linear(1, n_targets)
        self.bias = nn.Parameter(torch.zeros(n_targets))
        self._gate_facade = _GateFacade(self)
        self._cache: dict[str, torch.Tensor] | None = None

    # ---- Compatibility shims (older attribute names) ----
    @property
    def code_emb(self) -> nn.Embedding:
        return self.encounter_encoder.code_emb

    @property
    def enc_mlp(self) -> nn.Sequential:
        return self.encounter_encoder.enc_mlp

    @property
    def f_history(self) -> nn.Sequential:
        return self.history_head

    @property
    def f_age(self) -> nn.Linear:
        return self.age_head

    @property
    def q(self) -> nn.Parameter:
        return self.content_query

    @property
    def W_r(self) -> nn.Linear:
        return self.persistence_projection

    @property
    def W_k(self) -> nn.Linear:
        return self.content_key

    @property
    def W_v(self) -> nn.Module:
        """Locked architecture uses v_m directly; Identity for legacy callers."""
        return nn.Identity()

    @property
    def gate(self) -> _GateFacade:
        """Legacy TemporalGate-compatible facade (theta0/beta/lambda/gate)."""
        return self._gate_facade

    def encode_encounters(
        self,
        enc_code_ids: torch.Tensor,
        enc_code_mask: torch.Tensor,
    ) -> torch.Tensor:
        """DeepSets mean-pool of code embeddings → MLP. No age/τ."""
        return self.encounter_encoder(enc_code_ids, enc_code_mask)

    def z_of(self, age: torch.Tensor) -> torch.Tensor:
        return z_of_age(age)

    def persistence_offset(self, v: torch.Tensor) -> torch.Tensor:
        """rᵀ v_m (+ bias). Scalar per encounter. No age/τ input."""
        if not self.content_persistence:
            return torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype)
        return self.persistence_projection(v).squeeze(-1)

    def lambda_of(
        self,
        age: torch.Tensor,
        theta: torch.Tensor | None = None,
        persistence_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """λ = softplus(θ₀ + offset + β z(a*)). Positive by construction."""
        z = self.z_of(age)
        if theta is not None:
            th = theta
        elif persistence_offset is not None:
            th = self.theta0 + persistence_offset
        else:
            th = self.theta0
        # Broadcast age over encounter dim when th is [B, M]
        if th.dim() > z.dim():
            z = z.view(z.shape + (1,) * (th.dim() - z.dim()))
        if self.age_temporal:
            return F.softplus(th + self.beta * z)
        if th.dim() > 0 and th.shape != z.shape:
            return F.softplus(th)
        return F.softplus(th).expand_as(z) if th.shape != z.shape else F.softplus(th)

    def temporal_gate(
        self,
        age: torch.Tensor,
        tau: torch.Tensor,
        persistence_offset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (g_m, λ_m) with g = exp(-λ τ)."""
        lam = self.lambda_of(age, persistence_offset=persistence_offset)
        if lam.dim() < tau.dim():
            lam = lam.unsqueeze(-1)
        g = torch.exp(-lam * tau)
        return g, lam

    def forward(
        self,
        enc_code_ids: torch.Tensor,
        enc_code_mask: torch.Tensor,
        enc_tau: torch.Tensor,
        enc_padding_mask: torch.Tensor,  # True = pad encounter
        age: torch.Tensor,
        return_parts: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # Encounter content representation — no age / lag / cutoff.
        v = self.encode_encounters(enc_code_ids, enc_code_mask)  # [B, M, d]
        hist = ~enc_padding_mask  # [B, M]
        hist_f = hist.to(v.dtype)

        # Content relevance (no age, no τ)
        k = self.content_key(v)
        # u_m = qᵀ k_m / sqrt(d)  (scaled dot-product; scale is constant)
        scale = math.sqrt(k.size(-1))
        u = torch.einsum("bmd,d->bm", k, self.content_query) / scale  # [B, M]
        u = u.masked_fill(~hist, 0.0)

        # Content-dependent persistence offset (no age, no τ)
        theta_content = self.persistence_offset(v)  # [B, M]
        theta_content = theta_content * hist_f
        theta_m = self.theta0 + theta_content

        z = self.z_of(age)
        if self.age_temporal:
            lam = F.softplus(theta_m + self.beta * z.unsqueeze(-1))
        else:
            lam = F.softplus(theta_m)
        lam = lam * hist_f + (1.0 - hist_f) * 1.0  # unused pads → finite

        # Temporal gate
        g = torch.exp(-lam * enc_tau) * hist_f

        # ----------------------------------------------------------------
        # Numerical stability for w = exp(u) g  (raw additive, NOT softmax)
        # Clamp content score before exp to avoid overflow. This does not
        # renormalize across encounters; padded positions stay at weight 0.
        # Equivalent when u <= CONTENT_SCORE_EXP_CLAMP:
        #   w = exp(u) * g
        # ----------------------------------------------------------------
        u_clamped = u.clamp(max=CONTENT_SCORE_EXP_CLAMP)
        w = torch.exp(u_clamped) * g
        w = w * hist_f

        # Raw additive history: h = Σ_m w_m v_m
        weighted = (w.unsqueeze(-1) * v).sum(dim=1)  # [B, d]
        M = w.sum(dim=1, keepdim=True)  # [B, 1]

        if self.aggregation == "raw_additive":
            h_hist = weighted
        else:
            # Legacy ablation option (not canonical).
            h_bar = weighted / (M + 1e-6)
            log_mass = torch.log1p(M)
            h_hist = torch.cat([h_bar, log_mass], dim=-1)

        z1 = z.unsqueeze(-1)
        history_logit = self.history_head(h_hist)
        age_logit = self.age_head(z1)
        total = history_logit + age_logit + self.bias

        self._cache = {
            "u": u.detach(),
            "g": g.detach(),
            "w": w.detach(),
            "M": M.detach(),
            "lambda": lam.detach(),
            "theta_m": theta_m.detach(),
            "theta_content": theta_content.detach(),
            "persistence_offset": theta_content.detach(),
            "h_hist": h_hist.detach(),
            "history_logit": history_logit.detach(),
            "age_logit": age_logit.detach(),
            "v": v.detach(),
            "content_mag": u.abs().mean().detach(),
            "gate_mag": (-torch.log(g.clamp_min(1e-12)) * hist_f).sum().detach()
            / hist_f.sum().clamp_min(1.0),
        }
        if return_parts:
            return {
                "logits": total,
                "history_logit": history_logit,
                "age_logit": age_logit,
                "h_hist": h_hist,
                "M": M,
                "g": g,
                "u": u,
                "w": w,
                "lambda": lam,
                "theta_m": theta_m,
                "theta_content": theta_content,
                "v": v,
            }
        return total

    def age_parameters(self) -> list[nn.Parameter]:
        return [p for p in (self.theta0, self.beta) if p.requires_grad]

    def zero_all_betas_(self):
        saved = {"beta": self.beta.detach().clone()}
        with torch.no_grad():
            self.beta.zero_()
        return saved

    def restore_betas_(self, saved):
        with torch.no_grad():
            self.beta.copy_(saved["beta"])

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model": "Content-Persistence DTR",
            "content_dependent_persistence": self.content_dependent_persistence,
            "persistence_projection": self.persistence_projection_type,
            "temporal_aggregation": self.temporal_aggregation,
            "num_content_queries": self.num_content_queries,
            "age_conditioning": self.age_conditioning,
            "beta_scope": self.beta_scope,
            "age_temporal": self.age_temporal,
            "content_score_exp_clamp": CONTENT_SCORE_EXP_CLAMP,
        }


# Backward-compatible alias name.
ContentPersistenceDTR = DevelopmentalTemporalRetrieval


def build_dtr(
    *,
    age_temporal: bool,
    n_codes: int,
    n_targets: int,
    d_model: int = 64,
    aggregation: str = CANONICAL_AGGREGATION,
    content_persistence: bool = True,
    multi_query_K: int = 1,
    **kwargs: Any,
) -> DevelopmentalTemporalRetrieval:
    return DevelopmentalTemporalRetrieval(
        n_codes=n_codes,
        n_targets=n_targets,
        d_model=d_model,
        age_temporal=age_temporal,
        aggregation=aggregation,
        content_persistence=content_persistence,
        multi_query_K=multi_query_K,
        **kwargs,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_arm_init_check(
    at: DevelopmentalTemporalRetrieval,
    to: DevelopmentalTemporalRetrieval,
    batch: dict[str, torch.Tensor],
    atol: float = 1e-6,
) -> float:
    """Verify temporal-only and age-temporal logits match at init (β=0)."""
    at.eval()
    to.eval()
    with torch.no_grad():
        # Force identical β=0
        at.beta.zero_()
        to.beta.zero_()
        la = at(
            batch["enc_code_ids"],
            batch["enc_code_mask"],
            batch["enc_tau"],
            batch["enc_padding_mask"],
            batch["age"],
        )
        lt = to(
            batch["enc_code_ids"],
            batch["enc_code_mask"],
            batch["enc_tau"],
            batch["enc_padding_mask"],
            batch["age"],
        )
    return float((la - lt).abs().max().item())


def load_legacy_dtr_checkpoint(
    model: DevelopmentalTemporalRetrieval,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Migrate older DTR checkpoints into Content-Persistence DTR.

    Transfers:
      code_emb.*/enc_mlp.* → encounter_encoder.*,
      W_k→content_key, q→content_query,
      W_r→persistence_projection (if present), gate.theta0→theta0,
      gate.beta→beta, f_history→history_head, f_age→age_head, bias.

    If persistence_projection weights are missing, they remain at the
    zero initialization and a warning is emitted (not silently ignored).
    """
    key_map = {
        "W_k.weight": "content_key.weight",
        "W_r.weight": "persistence_projection.weight",
        "W_r.bias": "persistence_projection.bias",
        "gate.theta0": "theta0",
        "gate.beta": "beta",
        "f_history.0.weight": "history_head.0.weight",
        "f_history.0.bias": "history_head.0.bias",
        "f_history.2.weight": "history_head.2.weight",
        "f_history.2.bias": "history_head.2.bias",
        "f_age.weight": "age_head.weight",
        "f_age.bias": "age_head.bias",
    }
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k == "q":
            if v.dim() == 2:
                remapped["content_query"] = v[0].clone()
            else:
                remapped["content_query"] = v.clone()
            continue
        if k.startswith("code_emb."):
            remapped["encounter_encoder." + k] = v
            continue
        if k.startswith("enc_mlp."):
            remapped["encounter_encoder." + k] = v
            continue
        nk = key_map.get(k, k)
        if nk.startswith("W_v.") or "multi_query" in nk:
            continue
        remapped[nk] = v

    missing_persist = [
        k
        for k in ("persistence_projection.weight", "persistence_projection.bias")
        if k not in remapped
    ]
    if missing_persist:
        warnings.warn(
            "Legacy DTR checkpoint missing persistence_projection parameters "
            f"{missing_persist}; leaving zero initialization "
            "(θ_m ≈ θ₀). Use load_legacy_dtr_checkpoint explicitly.",
            stacklevel=2,
        )

    incompatible = model.load_state_dict(remapped, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


# ---------------------------------------------------------------------------
# Experimental / ablation variants (NOT part of the locked canonical model)
# ---------------------------------------------------------------------------


class DTRContentPersistence(DevelopmentalTemporalRetrieval):
    """Alias kept for older experiment scripts; identical to canonical DTR."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("content_persistence", True)
        kwargs.setdefault("aggregation", CANONICAL_AGGREGATION)
        super().__init__(*args, **kwargs)


class DTRMultiQuery(nn.Module):
    """Experimental multi-query ablation — NOT part of the locked architecture."""

    def __init__(
        self,
        n_codes: int,
        n_targets: int,
        d_model: int = 64,
        age_temporal: bool = True,
        aggregation: str = CANONICAL_AGGREGATION,
        dropout: float = 0.0,
        content_persistence: bool = True,
        multi_query_K: int = 4,
        num_queries: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        warnings.warn(
            "DTRMultiQuery is experimental and not part of the locked "
            "Content-Persistence DTR architecture.",
            stacklevel=2,
        )
        K = num_queries if num_queries is not None else multi_query_K
        self.num_queries = K
        self.multi_query_K = K
        self.aggregation = aggregation
        self.age_temporal = age_temporal
        self.d_model = d_model
        self.content_persistence = content_persistence

        self.base = DevelopmentalTemporalRetrieval(
            n_codes=n_codes,
            n_targets=n_targets,
            d_model=d_model,
            age_temporal=age_temporal,
            aggregation="raw_additive",  # base unused for head
            dropout=dropout,
            content_persistence=content_persistence,
            multi_query_K=1,
        )
        # Replace single query with multi-query parameter.
        self.Q = nn.Parameter(torch.randn(K, d_model) * 0.02)
        hist_in = (d_model + (1 if aggregation == "weighted_mean_plus_log_mass" else 0)) * K
        self.f_history = nn.Sequential(
            nn.Linear(hist_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_targets),
        )
        self.f_age = self.base.age_head
        self.bias = self.base.bias
        self.gate = self.base
        self._cache: dict[str, torch.Tensor] | None = None

    def encode_encounters(self, enc_code_ids, enc_code_mask):
        return self.base.encode_encounters(enc_code_ids, enc_code_mask)

    def forward(self, enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age, return_parts=False, **kwargs):
        v = self.base.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        hist_f = hist.to(v.dtype)

        k = self.base.content_key(v)
        u = torch.matmul(k, self.Q.T) / math.sqrt(k.size(-1))  # [B, M, K]
        u = u.masked_fill(~hist.unsqueeze(-1), 0.0)

        theta_content = self.base.persistence_offset(v) * hist_f
        theta_m = self.base.theta0 + theta_content
        z = self.base.z_of(age)
        if self.age_temporal:
            lam = F.softplus(theta_m + self.base.beta * z.unsqueeze(-1))
        else:
            lam = F.softplus(theta_m)
        g = torch.exp(-lam * enc_tau) * hist_f

        w = torch.exp(u.clamp(max=CONTENT_SCORE_EXP_CLAMP)) * g.unsqueeze(-1)
        w = w * hist.unsqueeze(-1).to(w.dtype)

        M = w.sum(dim=1)  # [B, K]
        weighted = torch.einsum("bmk,bmd->bkd", w, v)  # [B, K, d]

        if self.aggregation == "raw_additive":
            h_hist = weighted.reshape(weighted.size(0), -1)
        else:
            h_bar = weighted / (M.unsqueeze(-1) + 1e-6)
            log_mass = torch.log1p(M)
            h_hist = torch.cat([h_bar.reshape(weighted.size(0), -1), log_mass], dim=-1)

        history_logit = self.f_history(h_hist)
        age_logit = self.f_age(z.unsqueeze(-1))
        total = history_logit + age_logit + self.bias
        self._cache = {"u": u.detach(), "g": g.detach(), "w": w.detach(), "M": M.detach(), "lambda": lam.detach()}
        if return_parts:
            return {"logits": total, "history_logit": history_logit, "age_logit": age_logit, "h_hist": h_hist, "M": M, "g": g, "u": u, "w": w}
        return total

    def age_parameters(self):
        return self.base.age_parameters()

    def zero_all_betas_(self):
        return self.base.zero_all_betas_()

    def restore_betas_(self, saved):
        return self.base.restore_betas_(saved)

    @property
    def theta0(self):
        return self.base.theta0

    @property
    def beta(self):
        return self.base.beta

    @property
    def persistence_projection(self):
        return self.base.persistence_projection
