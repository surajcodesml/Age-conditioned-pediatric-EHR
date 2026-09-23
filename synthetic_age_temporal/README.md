# Synthetic age × temporal benchmark

Semi-synthetic pediatric EHR benchmark built on **Synthea trajectories** with a
**known softplus age × temporal ground-truth mechanism**.

## Purpose

Validate whether an age-conditioned temporal attention model:

1. detects that an age × lag interaction exists;
2. recovers its direction and magnitude;
3. functionally uses it for prediction;
4. remains inert when no interaction exists (S0/S1).

This is **not** a generic Synthea data dump. Background Synthea events provide
realistic longitudinal history; controlled `SYN_SIGNAL_*` events carry the
identifiable mechanism.

## Layout

```text
synthetic_age_temporal/
  config.py              # z(a), τ, softplus λ, scenarios, training defaults
  generate_synthea.py    # reuse / optional regenerate Synthea cohort
  ground_truth.py        # λ(a), R(a,τ), target logits, calibration, oracle
  build_benchmark.py     # splits, signal injection, labels, oracle gate
  dataset.py             # model-visible tensors (no GT leakage)
  model.py               # prediction-time softplus age×temporal Transformer
  baselines.py           # Y ~ age+lag vs age+lag+age×lag
  train.py               # matched arms
  evaluate.py            # metrics, ablations, λ/surface recovery
  plots.py               # fig1–fig6 (PNG+SVG)
  run_benchmark.py       # end-to-end orchestration
  tests/test_sanity.py   # automated invariant tests
  report.md              # regenerated results report
```

## Equations

$$
z(a)=\frac{a-9}{9},\quad
\tau=\log(1+\Delta t/7),\quad
\lambda(a)=\mathrm{softplus}(\theta_0+\beta z(a)),\quad
R(a,\tau)=e^{-\lambda(a)\tau}
$$

Prediction-time attention:

$$
s_{*j}^{(h)}=\frac{q_*^{(h)\top}k_j^{(h)}}{\sqrt{d_h}}-\lambda(a_*)\tau_{*j}
$$

## Scenarios

| ID | β_true | Meaning |
|----|--------|---------|
| S0 | 0 | temporal relevance, no age interaction |
| S1 | 0 | age main effect only |
| S2 | <0 | developmental: younger decays faster |
| S3 | >0 | reversed interaction |

## Quick start

```bash
cd synthetic_age_temporal
# Reuse existing ~10k sep1-exp Synthea cohort (no regeneration)
/home/suraj/miniconda3/envs/ehr/bin/python run_benchmark.py --quick --epochs 8

# Full controlled + full-realism cohorts, strength sweep
/home/suraj/miniconda3/envs/ehr/bin/python run_benchmark.py --strength-sweep
```

Unit tests only:

```bash
/home/suraj/miniconda3/envs/ehr/bin/python tests/test_sanity.py
```

## Arms

| Arm | Behavior |
|-----|----------|
| `no_age` | no age, no temporal bias |
| `age_only` | late-fusion age; no age×lag bias |
| `temporal_only` | λ = softplus(θ0) |
| `age_temporal` | λ(a*) = softplus(θ0 + β z(a*)) |
| `historical_age` | optional: λ(a_j) at event age |

## Scientific constraints

- Do not tune the generator until the proposed model wins.
- No future timestamps; cutoff age/time only.
- Ground-truth mechanism fields are never model inputs.
- No Fourier / Chebyshev / per-head β.
- No sign constraint on learned β.
- Oracle gate must pass for S2/S3 before Transformer training.

## Provenance

Default cohort: `synthea/sep1-exp/output/full/processed` (~10k patients,
ages 0–18, Synthea commit `aa0772fb…`). Production MIMIC/NCH pipelines are
not modified.
