#!/usr/bin/env python3
"""``--arm`` -> internal behavior resolution (the single front-end).

Design decision (config-driven conditional instantiation, not strategy objects):
one model class holds BOTH age pathways as sub-modules; a resolved
:class:`ArmConfig` sets a handful of booleans/enums that gate them. Rationale:

  * Only four arms, and the two pathways are small and static. Strategy/polymorphic
    objects would scatter the non-negotiable invariants (Delta-alpha == 0 in
    vanilla/additive; additive-delta == 0 in vanilla/kernel; param-count ordering)
    across classes and make them hard to assert in one place.
  * ``random_constant`` must be *architecturally identical* to ``kernel`` and differ
    only in the age input. A single class with a ``age_source`` switch guarantees
    that by construction; two strategy classes would risk drift.
  * It reuses the frozen enums rather than paralleling them: ``--arm`` maps onto the
    existing ``age_conditioning_mode`` values ({real, random_constant, none}) plus a
    single new ``additive_embed`` boolean. ``kernel_injection`` is LOCKED to
    ``additive_logspace`` for every arm.
"""

from __future__ import annotations

from dataclasses import dataclass

ARMS = ("vanilla", "random_constant", "additive", "kernel")

# Locked for every arm.
KERNEL_INJECTION = "additive_logspace"

# Fixed constant age (years) fed to the kernel pathway in the random_constant arm.
# A plausible mid-childhood value; its exact value is irrelevant to the control's
# purpose (capacity matched, no *real* age signal).
RANDOM_CONSTANT_AGE_YEARS = 7.0


@dataclass(frozen=True)
class ArmConfig:
    """Resolved internal behavior for one arm."""

    arm: str
    # Kernel pathway (AgeCoefficientGenerator.mode): real | random_constant | none.
    age_conditioning_mode: str
    # Whether the additive code-embedding pathway is active.
    additive_embed: bool
    # Where the age fed to the Fourier embedding comes from: real | constant | none.
    age_source: str
    kernel_injection: str = KERNEL_INJECTION

    @property
    def uses_real_age(self) -> bool:
        return self.age_source == "real"


def resolve_arm(arm: str) -> ArmConfig:
    if arm not in ARMS:
        raise ValueError(f"--arm must be one of {ARMS}, got {arm!r}")
    if arm == "vanilla":
        return ArmConfig(arm, age_conditioning_mode="none", additive_embed=False, age_source="none")
    if arm == "random_constant":
        # Same architecture as kernel; age input is a fixed constant.
        return ArmConfig(arm, age_conditioning_mode="random_constant", additive_embed=False, age_source="constant")
    if arm == "additive":
        return ArmConfig(arm, age_conditioning_mode="none", additive_embed=True, age_source="real")
    if arm == "kernel":
        return ArmConfig(arm, age_conditioning_mode="real", additive_embed=False, age_source="real")
    raise AssertionError("unreachable")
