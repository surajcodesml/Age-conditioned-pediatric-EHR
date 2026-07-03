# `model_ablation/` — four-arm age-conditioning ablation

Self-contained. Imports **nothing** from `model/` or `finetune/` (verified:
`grep -rE "from model[. ]|import model\b" model_ablation/` returns nothing). The
frozen trees are provenance for prior results and cannot affect these.

## Arms (single `--arm` front-end)

| arm | kernel Δα | additive embed | age → Fourier |
|---|---|---|---|
| `vanilla` | 0 (no params) | 0 (no params) | — |
| `random_constant` | MLP(φ(**const** age)) | 0 | constant (7y) |
| `additive` | 0 | MLP(φ(**real** age)) → +code_emb | real |
| `kernel` | MLP(φ(**real** age)) → Δcoeffs | 0 | real |

Locked for every arm: **additive-logspace** kernel injection (`scores + logsigmoid(poly)`);
no multiplicative; no QK-normalization; no time/Weibull loss. Temporal polynomial
uses a **Chebyshev** basis (degree 5) on `x = 2·log1p(Δt/7)/6.5 − 1`.

## Gating design: config-driven conditional instantiation (not strategy objects)

`arms.resolve_arm(arm) → ArmConfig(age_conditioning_mode, additive_embed, age_source)`
is the single front-end; it maps `--arm` onto the **existing** enum
(`age_conditioning_mode ∈ {real, random_constant, none}`) plus one new boolean
`additive_embed`. `TALEEHRAblation` reads that config and builds only the pathway an
arm uses. Rationale: with four arms and two tiny pathways, one class + one config
keeps every non-negotiable invariant assertable **in one place**
(`assert_arm_invariants`, `age_pathway_param_count`) and guarantees
`random_constant` is architecturally identical to `kernel` by construction. Strategy
objects would scatter those invariants across classes and invite drift.

## Demographics leak fix (INV-demo)

Age is a **separate** `age_years` [B,L] field; `demographics` is [B,L,2] = (sex, race);
`demo_dim == 2`. Age can no longer enter `demo_proj` in any arm (asserted at forward).
A legacy `demo_dim=3` pretrain checkpoint is adapted at load by dropping the age input
column of `demo_proj` (see `model_finetune._adapt_legacy_demo_proj`). Note: the spec's
`demo_last[..., 1:]` referred to the old 3-col layout; here age is removed upstream so
demographics is 2-col (sex, race) outright — a stronger guarantee than slicing.

## Usage

```bash
# Verification (run this first; reports param counts, invariants, gradient liveness)
conda run -n ehr python model_ablation/verify_arms.py \
    --pretrained_ckpt checkpoints/run_20260427_152603/best_pretrain.pt \
    --tensorized_dir data/finetune/heart_failure_tensorized   # CHD stand-in until built

# Shared vanilla pretrain (one backbone for all arms)
conda run -n ehr python model_ablation/train.py --tensorized_dir data/processed/tensorized

# Arm fine-tune (only --arm varies; seed/hparams identical across arms)
conda run -n ehr python model_ablation/train_finetune.py --arm {vanilla|random_constant|additive|kernel} \
    --pretrained_ckpt <shared_backbone.pt> --tensorized_dir <CHD_tensorized_dir>
```

**CHD data:** "CHD" = the `heart_malformations` cohort, which is **not built yet**
(`data/finetune/heart_malformations/` is empty). Build that cohort + tensorize it,
then pass its dir. `heart_failure_tensorized` is used only as a stand-in to exercise
the verification harness.
