#!/usr/bin/env python3
"""``--arm`` -> resolved behaviour. The single front-end for the four-arm experiment.

Every arm shares backbone code, seed, data, schedule, optimizer settings and masking.
The arm flag is the only difference, and what it selects is **where the coefficient
generator's output is delivered** -- not whether age is available to the model.

    arm              demo feature (R1)   kernel Delta-alpha (R2)   concat to h (R3)   age into psi
    ---------------  ------------------  ------------------------  -----------------  -----------------
    vanilla          yes                 identically 0, no params  --                 unused
    random_constant  yes                 generator(fixed rand vec) --                 bypassed
    additive         yes                 identically 0, no params  generator(psi(a_n)) real, last event
    kernel           yes                 generator(psi(a_i))       --                 real, per query

R1 is present in every arm and is **not** the experimental variable. Age stays in the
demographic vector because that is the route age already has, and it is the one DKM has to
improve on; removing it would make ``vanilla`` age-blind and the baseline a strawman.

``random_constant``'s generator can only ever produce a single learned constant, which the
free ``alpha_base`` absorbs. It should therefore track ``vanilla`` closely: a large gap
between them indicates a bug, not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ARMS",
    "ArmConfig",
    "resolve_arm",
    "assert_arm_invariants",
    "KERNEL_INJECTION",
    "MASKING",
    "POOLING",
]

ARMS = ("vanilla", "random_constant", "additive", "kernel")

# Locked for every arm; recorded in config.json.
KERNEL_INJECTION = "log_space_additive"
MASKING = "padding_only"
POOLING = "attention"


@dataclass(frozen=True)
class ArmConfig:
    """Resolved internal behaviour for one arm."""

    arm: str
    # CoefficientGenerator.mode at the two kernel sites: real | random_constant | none.
    # "none" means the module is NOT CONSTRUCTED, so the arm carries zero age parameters.
    kernel_generator_mode: str
    # Whether a generator output is concatenated to the pooled representation h.
    additive_head: bool

    kernel_injection: str = KERNEL_INJECTION
    masking: str = MASKING
    pooling: str = POOLING

    @property
    def has_kernel_age_params(self) -> bool:
        return self.kernel_generator_mode != "none"

    @property
    def uses_real_age(self) -> bool:
        return self.kernel_generator_mode == "real" or self.additive_head


def resolve_arm(arm: str) -> ArmConfig:
    if arm not in ARMS:
        raise ValueError(f"--arm must be one of {ARMS}, got {arm!r}")
    if arm == "vanilla":
        return ArmConfig(arm, kernel_generator_mode="none", additive_head=False)
    if arm == "random_constant":
        return ArmConfig(arm, kernel_generator_mode="random_constant", additive_head=False)
    if arm == "additive":
        return ArmConfig(arm, kernel_generator_mode="none", additive_head=True)
    if arm == "kernel":
        return ArmConfig(arm, kernel_generator_mode="real", additive_head=False)
    raise AssertionError("unreachable")


def assert_arm_invariants(cfg: ArmConfig, *, center_delta_alpha: bool) -> None:
    """Structural checks that do not need a constructed model. HARD."""
    if cfg.arm not in ARMS:
        raise AssertionError(f"[INV-ARM] unknown arm {cfg.arm!r}")
    if cfg.kernel_generator_mode not in {"real", "random_constant", "none"}:
        raise AssertionError(f"[INV-ARM] bad generator mode {cfg.kernel_generator_mode!r}")
    if cfg.arm in {"vanilla", "additive"} and cfg.has_kernel_age_params:
        raise AssertionError(f"[INV-ARM] arm={cfg.arm} must carry no kernel-side age parameters")
    if cfg.arm in {"vanilla", "random_constant", "kernel"} and cfg.additive_head:
        raise AssertionError(f"[INV-ARM] arm={cfg.arm} must not concatenate to h")
    if center_delta_alpha and cfg.arm == "random_constant":
        # The random_constant generator sees a constant input, so its Delta-alpha is constant
        # over the reference grid and centering makes it EXACTLY zero -- collapsing the
        # capacity control onto vanilla. See INVARIANTS.md.
        raise AssertionError(
            "[INV-ARM] --center_delta_alpha collapses random_constant onto vanilla "
            "(constant Delta-alpha minus its own mean is exactly 0); refusing to run."
        )
