#!/usr/bin/env python3
"""Age -> Fourier features -> coefficient deltas.

D7 -- the legacy band. The legacy embedding placed Fourier periods from 1/12 y to 200 y on
**linear** age. Measured on that band, ``||psi(a) - psi(a+g)||`` reaches 90% of its
asymptote by a gap of 7.4 months, the minimum pairwise distance over a 0.25 y grid on
[0, 90] is 3.35 against a maximum of 8.00, and ``||dpsi/da||`` is flat in age. That is a
near-orthogonal *hash* of age, not a smooth developmental coordinate: nothing about it says
that 6 months and 9 months are closer than 6 months and 40 years, so the cheapest thing the
generator can learn on top of it is a constant ``Delta-alpha`` -- which ``alpha_base``
absorbs, giving exactly the age-invariant-offset failure that was observed.

The fix is to run the same construction over **log-age** ``u = log1p(a)``, with periods
log-spaced in u-units. Resolution is then allocated by developmental rate rather than
uniformly in calendar time.

Nothing in this module prints. ``characterize_band`` returns numbers; ``diagnostics`` owns
the formatting.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

__all__ = [
    "LogAgeFourier",
    "LinearAgeFourier",
    "CoefficientGenerator",
    "AgeConditioner",
    "characterize_band",
    "DEFAULT_M",
    "DEFAULT_P_MIN",
    "DEFAULT_P_MAX",
    "REFERENCE_AGE_GRID",
]

DEFAULT_M = 16
DEFAULT_P_MIN = 0.15
DEFAULT_P_MAX = 6.0

# Fixed reference grid for --center_delta_alpha. Registered as a buffer so it serialises.
REFERENCE_AGE_GRID = (0.0, 90.0, 181)  # start, stop, n  -> 0.5 y spacing


class _FourierBase(nn.Module):
    """Shared machinery. Subclasses supply the age -> phase-coordinate transform."""

    def __init__(self, M: int, p_min: float, p_max: float) -> None:
        super().__init__()
        self.M = int(M)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        if self.M <= 0:
            raise ValueError(f"M must be > 0, got {M}")
        if not (0 < self.p_min <= self.p_max):
            raise ValueError(f"need 0 < p_min <= p_max, got {p_min}, {p_max}")
        # D12: the emitted feature dim is 2M and must be even.
        if self.embedding_dim % 2 != 0:
            raise AssertionError(f"[D12] age_emb_dim must be even, got {self.embedding_dim}")
        periods = torch.exp(
            torch.linspace(math.log(self.p_max), math.log(self.p_min), steps=self.M,
                           dtype=torch.float32)
        )
        # Persistent + requires_grad False: restored from checkpoint, never rebuilt (INV-FROZEN).
        self.register_buffer("frequencies", 1.0 / periods, persistent=True)
        self.register_buffer("periods", periods, persistent=True)
        self.frequencies.requires_grad_(False)
        self.periods.requires_grad_(False)

    @property
    def embedding_dim(self) -> int:
        return 2 * self.M

    def coordinate(self, age_years: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, age_years: torch.Tensor) -> torch.Tensor:
        u = self.coordinate(age_years)
        angles = 2.0 * math.pi * u.unsqueeze(-1) * self.frequencies.to(u.dtype)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class LogAgeFourier(_FourierBase):
    """``u = log1p(clamp(age_years, min=0))``, periods log-spaced in u-units.

    Defaults ``p_min=0.15``, ``p_max=6.0``, ``M=16`` -> 32 dims. ``u`` spans
    ``[0, log1p(90) = 4.51]``. ``p_max = 6.0 > 4.51`` gives one sub-cycle component that
    acts as a monotone global coordinate; ``p_min = 0.15`` in u-units is about 2.7 months of
    age at ``a = 0.5`` but about 6.2 years at ``a = 40``. These are a starting point, not a
    result -- ``characterize_band`` reports what they actually buy.
    """

    def __init__(self, M: int = DEFAULT_M, p_min: float = DEFAULT_P_MIN,
                 p_max: float = DEFAULT_P_MAX) -> None:
        super().__init__(M, p_min, p_max)

    def coordinate(self, age_years: torch.Tensor) -> torch.Tensor:
        return torch.log1p(age_years.clamp(min=0.0))


class LinearAgeFourier(_FourierBase):
    """The legacy band (1/12 y .. 200 y on linear age). Kept **only** as the reference column
    in ``characterize_band``; no model path constructs it."""

    def __init__(self, M: int = DEFAULT_M, p_min: float = 1.0 / 12.0,
                 p_max: float = 200.0) -> None:
        super().__init__(M, p_min, p_max)

    def coordinate(self, age_years: torch.Tensor) -> torch.Tensor:
        return age_years.clamp(min=0.0)


class CoefficientGenerator(nn.Module):
    """``psi(a) -> Delta-alpha``. ``Linear(2M -> 64) -> GELU -> Linear(64 -> s)``.

    The final layer weight is zero-initialised, so ``Delta-alpha == 0`` at init and every
    arm starts from the same kernel.

    The final layer **bias is omitted by default**. A free bias makes a constant
    ``Delta-alpha`` the cheapest descent direction, and a constant ``Delta-alpha`` is fully
    absorbed by ``alpha_base`` -- the age-invariant-offset failure. This is a mild
    expressivity restriction, not a correctness requirement; ``--gen_final_bias`` re-enables
    it.

    Modes:
      ``real``            -- receives ``psi(a)``.
      ``random_constant`` -- receives a fixed random vector (seed 0) expanded to the shape of
                             the age features. Capacity-matched to ``real``; carries no real
                             age signal.
      ``none``            -- the MLP is **not constructed**; the arm has zero age parameters.
    """

    MODES = ("real", "random_constant", "none")

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, mode: str = "real",
                 final_bias: bool = False, random_seed: int = 0) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode = mode
        self.final_bias = bool(final_bias)

        if mode == "none":
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.out_dim, bias=self.final_bias),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            if self.final_bias:
                nn.init.zeros_(self.mlp[-1].bias)

        if mode == "random_constant":
            # Drawn from a dedicated generator so it never perturbs the global RNG stream
            # (INV-ZERO-A depends on arms sharing an identical backbone init).
            gen = torch.Generator().manual_seed(int(random_seed))
            vec = torch.randn(self.in_dim, generator=gen, dtype=torch.float32)
            self.register_buffer("random_vector", vec, persistent=True)
            self.random_vector.requires_grad_(False)

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, in_dim={self.in_dim}, out_dim={self.out_dim}, "
                f"final_bias={self.final_bias}")

    def forward(self, age_features: torch.Tensor) -> torch.Tensor:
        if age_features.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected trailing dim {self.in_dim}, got {tuple(age_features.shape)}"
            )
        if self.mode == "none" or self.mlp is None:
            return torch.zeros(
                age_features.shape[:-1] + (self.out_dim,),
                dtype=age_features.dtype, device=age_features.device,
            )
        if self.mode == "random_constant":
            age_features = self.random_vector.to(age_features.dtype).expand_as(age_features)
        return self.mlp(age_features)

    def age_parameters(self) -> list[nn.Parameter]:
        """Declared membership for the ``age`` optimizer group (D6). Not name matching."""
        return [p for p in self.parameters() if p.requires_grad]


class AgeConditioner(nn.Module):
    """One kernel site's age pathway: its own Fourier band and its own generator.

    The two kernel sites (encoder, pooling) share *form* and nothing else -- each owns a
    separate instance, per the draft.
    """

    def __init__(self, out_dim: int, M: int = DEFAULT_M, p_min: float = DEFAULT_P_MIN,
                 p_max: float = DEFAULT_P_MAX, hidden_dim: int = 64, mode: str = "real",
                 final_bias: bool = False, center_delta_alpha: bool = False,
                 random_seed: int = 0) -> None:
        super().__init__()
        self.fourier = LogAgeFourier(M=M, p_min=p_min, p_max=p_max)
        self.generator = CoefficientGenerator(
            in_dim=self.fourier.embedding_dim, out_dim=out_dim, hidden_dim=hidden_dim,
            mode=mode, final_bias=final_bias, random_seed=random_seed,
        )
        self.out_dim = int(out_dim)
        self.center_delta_alpha = bool(center_delta_alpha)
        start, stop, n = REFERENCE_AGE_GRID
        grid = torch.linspace(start, stop, n, dtype=torch.float32)
        self.register_buffer("reference_ages", grid, persistent=True)
        self.reference_ages.requires_grad_(False)

    @property
    def mode(self) -> str:
        return self.generator.mode

    @property
    def age_emb_dim(self) -> int:
        return self.fourier.embedding_dim

    def forward(self, age_years: torch.Tensor) -> torch.Tensor:
        """``age_years [...] -> Delta-alpha [..., out_dim]``."""
        delta = self.generator(self.fourier(age_years))
        if self.center_delta_alpha and self.generator.mode != "none":
            ref = self.generator(self.fourier(self.reference_ages.to(age_years.device)))
            delta = delta - ref.mean(dim=0)
        return delta

    def age_parameters(self) -> list[nn.Parameter]:
        return self.generator.age_parameters()


# --------------------------------------------------------------------------- #
# 2.3 Band characterisation -- MEASURE, not assert.                            #
# Only one HARD failure: psi must be injective on the grid.                    #
# --------------------------------------------------------------------------- #
def characterize_band(psi: Callable[[torch.Tensor], torch.Tensor], *, M: int,
                      grid_step: float = 0.25, age_max: float = 90.0,
                      sat_age: float = 5.0, adult_min: float = 18.0) -> dict:
    """Geometry of an age embedding. All quantities are reported, none are asserted --
    except injectivity on the grid, which is HARD."""
    with torch.no_grad():
        n = int(round(age_max / grid_step)) + 1
        ages = torch.linspace(0.0, age_max, n, dtype=torch.float64)
        feats = psi(ages).double()  # [n, 2M]

        d = torch.cdist(feats, feats)
        eye = torch.eye(n, dtype=torch.bool)
        min_pairwise = float(d.masked_fill(eye, float("inf")).min())
        max_possible = 2.0 * math.sqrt(M)

        # Saturation gap at sat_age: smallest g where ||psi(a) - psi(a+g)|| first reaches 90%
        # of its asymptote. The asymptote is taken ANALYTICALLY as sqrt(2M) -- the expected
        # distance between two independent uniform phases, since
        # ||psi(a)-psi(a')||^2 = sum_m 2(1 - cos(2 pi (u-u')/p_m)). Using an analytic value
        # keeps the number independent of the gap grid; the empirical far-range mean is
        # reported alongside so the two can be compared.
        gaps = torch.linspace(0.0, age_max - sat_age, 20001, dtype=torch.float64)[1:]
        a0 = psi(torch.tensor([sat_age], dtype=torch.float64)).double()
        far = psi(sat_age + gaps).double()
        dist = (far - a0).norm(dim=-1)
        asymptote = math.sqrt(2.0 * M)
        empirical_asymptote = float(dist[len(dist) // 2 :].mean())
        reached = (dist >= 0.9 * asymptote).nonzero()
        sat_gap_years = float(gaps[int(reached[0])]) if len(reached) else float("nan")

        # ||dpsi/da|| by central difference.
        def grad_norm(a: float, h: float = 1e-4) -> float:
            pts = torch.tensor([a - h, a + h], dtype=torch.float64)
            f = psi(pts).double()
            return float((f[1] - f[0]).norm() / (2 * h))

        g_half, g_forty = grad_norm(0.5), grad_norm(40.0)

        adult = ages >= adult_min
        adult_feats = feats[adult]
        da = torch.cdist(adult_feats, adult_feats)
        n_ad = adult_feats.shape[0]
        adult_min_pairwise = float(
            da.masked_fill(torch.eye(n_ad, dtype=torch.bool), float("inf")).min()
        )
        nearest_adult, nearest_adult_ratio = {}, {}
        for a in (0.5, 2.0, 5.0, 10.0):
            fa = psi(torch.tensor([a], dtype=torch.float64)).double()
            v = float((adult_feats - fa).norm(dim=-1).min())
            nearest_adult[str(a)] = v
            # The interpretable form: how far a pediatric age sits from the adult manifold,
            # measured in units of how far adults sit from each other. A ratio near 1 means
            # a child is no more distinguishable from an adult than two adults are from each
            # other -- i.e. the embedding is a hash of age, not a developmental coordinate.
            nearest_adult_ratio[str(a)] = v / adult_min_pairwise if adult_min_pairwise > 0 else float("inf")

    if not (min_pairwise > 0.0):
        raise AssertionError(
            f"[BAND] psi is not injective on the {grid_step} y grid: "
            f"min pairwise distance = {min_pairwise:.3e}"
        )

    return {
        "M": M,
        "grid_step_years": grid_step,
        "age_max_years": age_max,
        "saturation_gap_years_at_a5": sat_gap_years,
        "saturation_gap_months_at_a5": sat_gap_years * 12.0,
        "saturation_asymptote_analytic": asymptote,
        "saturation_asymptote_empirical": empirical_asymptote,
        "min_pairwise_distance": min_pairwise,
        "max_possible_distance": max_possible,
        "grad_norm_at_0.5": g_half,
        "grad_norm_at_40": g_forty,
        "grad_ratio_0.5_over_40": g_half / g_forty if g_forty > 0 else float("inf"),
        "adult_min_pairwise_distance": adult_min_pairwise,
        "nearest_adult_distance": nearest_adult,
        "nearest_adult_over_adult_min": nearest_adult_ratio,
    }


def _smoke() -> None:
    from model_new import diagnostics

    log_band = LogAgeFourier()
    lin_band = LinearAgeFourier()
    got = characterize_band(log_band, M=log_band.M)
    ref = characterize_band(lin_band, M=lin_band.M)
    diagnostics.print_band_characterization(got, ref)

    cond = AgeConditioner(out_dim=5, mode="real")
    ages = torch.tensor([[0.25, 1.0, 40.0], [70.0, 3.0, 8.0]])
    with torch.no_grad():
        delta = cond(ages)
    diagnostics.print_block(
        "age_encoding.py smoke",
        [
            f"psi dim                    : {cond.age_emb_dim}",
            f"Delta-alpha shape          : {tuple(delta.shape)}",
            f"Delta-alpha at init max|.| : {float(delta.abs().max()):.3e}  (expect 0)",
            f"age params (real)          : {sum(p.numel() for p in cond.age_parameters())}",
            f"age params (none)          : "
            f"{sum(p.numel() for p in AgeConditioner(out_dim=5, mode='none').age_parameters())}",
        ],
    )


if __name__ == "__main__":
    _smoke()
