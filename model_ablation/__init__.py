"""Self-contained four-arm age-conditioning ablation.

This package is deliberately isolated from the frozen ``model/`` and ``finetune/``
packages: it imports nothing from them, so prior results cannot leak into (or be
perturbed by) the ablation. Duplication of dataset / training code is intentional.

Arms (single ``--arm`` front-end, see :class:`arms.ArmConfig`):
  vanilla          - base temporal kernel, no age anywhere.
  random_constant  - kernel age-pathway fed a fixed constant age (capacity control).
  additive         - real age injected additively into code embeddings; kernel Delta-alpha = 0.
  kernel           - real age modulates the temporal-kernel coefficients; no additive injection.

Locked design choices: additive-logspace kernel injection in ALL arms; no
multiplicative; no QK normalization; no time/Weibull loss.
"""
