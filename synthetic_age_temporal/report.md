# Synthetic age × temporal benchmark report

Semi-synthetic pediatric EHR benchmark using Synthea trajectories with a
known softplus age × temporal ground-truth mechanism.

## Core success criterion

> The model recovers and functionally uses the known age × temporal
> mechanism when it exists, and remains inert when it does not.

Do **not** declare success solely from validation AUPRC. Classification below
combines predictive gain, functional ablations, and λ(a) / surface recovery.

---

## Synthea cohort generation

| Field | Value |
|---|---|
| synthea_version | `master-branch-latest` |
| synthea_commit | `aa0772fb5e92e48a776c51508c00eddc0d9d27ff` |
| reused_existing_cohort | **yes** (`synthea/sep1-exp/output/full/processed`) |
| geography | Massachusetts |
| reference_date | 20260101 |
| generation_seed (by stratum) | infant=202601011, early_childhood=202601012, school_age=202601013, adolescent=202601014 |
| n_patients | 10,000 |
| age range | 0–18 |
| background events | 3,275,125 (source=`synthea` only; prior SIGNAL_A/B dropped) |

### Age distribution

mean=7.01, std=5.69, min≈0.003, max≈17.99; strata ≈2.5k each (<1, 1–5, 6–11, 12–17).

### Event counts (background)

observation 2.16M · procedure 0.38M · immunization 0.30M · encounter 0.24M · condition 0.14M · medication 62k.

---

## Dataset construction

### Patient splits (seed=20260922)

| Split | Fraction | Notes |
|---|---|---|
| train | 70% | patient-disjoint |
| val | 15% | |
| test | 15% | |

### Cohorts

| Cohort | Filter | n (controlled build) |
|---|---|---|
| **controlled** | age ≥ 2y; max signal lag ≤ 730d | 7,007 |
| **full** | ages 0–18; lags up to lifetime | 10,000 |

### Signal-event injection

- 12 codes: `SYN_SIGNAL_A` … `SYN_SIGNAL_L`
- Lags (days): 7, 30, 90, 180, 365, 730 (±10% jitter)
- 4–8 signals / example; treated as ordinary categorical events
- Occurrence frequency not engineered to reveal Y

### Prediction cutoffs

- `t*` = Synthea index/reference date
- `a*` = age at cutoff
- History restricted to `t_j < t*` (strict)

---

## Target-generation equations

$$
z(a)=\frac{a-9}{9},\qquad
\tau=\log\!\bigl(1+\Delta t_{\mathrm{days}}/7\bigr)
$$

$$
\lambda_{\mathrm{true}}(a)=\mathrm{softplus}(\theta_0+\beta_{\mathrm{true}} z(a)),\qquad
R(a,\tau)=\exp\!\bigl[-\lambda_{\mathrm{true}}(a)\,\tau\bigr]
$$

**Interaction targets**

$$
\eta_k=b_k+\gamma_k z(a_*)+\alpha\sum_j w_{kj} R(a_*,\tau_j)+\epsilon,\quad
Y_k\sim\mathrm{Bernoulli}(\sigma(\eta_k))
$$

with relevance scale $\alpha=2.5$, noise $\epsilon\sim\mathcal N(0,0.35^2)$,
prevalence calibrated to ≈0.25.

**Other mechanism classes** (32 labels total): temporal-only (fixed $\lambda_0$),
age-only, content-only, null. Every target records `target_mechanism_type`.

### Scenario parameters

| Scenario | $\theta_0$ | $\beta_{\mathrm{true}}$ | Notes |
|---|---|---|---|
| S0 | 0 | **0** | temporal relevance; no age×lag |
| S1 | 0 | **0** | age main effect only ($\gamma$ path) |
| S2 | 0 | **−2.5** (medium) | developmental; strength sweep {1.2, 2.5, 4.0} |
| S3 | 0 | **+2.5** | reversed interaction |

Calibration tables live in each scenario’s `calibration.json`
(λ at ages 1/5/10/15/18; R at 30–730d).

---

## Oracle validation (controlled)

Gate: S2/S3 must show clear interaction signal **before** Transformer training.

| Scenario | Oracle micro-AUROC | ΔBCE age-shuffle | ΔBCE lag-shuffle | ΔBCE no-interaction | Ready |
|---|---|---|---|---|---|
| S0 | 0.755 | ≈0.01 | — | 0 | yes |
| S1 | 0.770 | — | — | 0 | yes (remove-age hurts) |
| S2 | 0.787 | **0.102** | **0.070** | **0.044** | yes |
| S3 | 0.763 | **0.076** | **0.051** | **0.031** | yes |

Full-realism cohort also oracle-ready (S0–S3 AUROC 0.77–0.81).

---

## Statistical baselines (controlled, interaction labels)

`Y ~ z + Σ_c exp(−λ₀τ)` vs `Y ~ … + Σ_c z·τ·exp(−λ₀τ)` (train fit, test eval).

| Scenario | mean ΔAUROC (interaction − base) | frac Δ>0 |
|---|---|---|
| S0 | **−0.001** | 0.125 |
| S1 | **+0.000** | 0.50 |
| S2 | **+0.027** | **1.00** |
| S3 | **+0.013** | **1.00** |

Identifiability holds: interaction features help only when $\beta_{\mathrm{true}}\neq 0$.

---

## Neural model configuration

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| n_heads | 4 |
| n_layers | 1 |
| shared θ₀, β | yes (not per-head) |
| λ | softplus(θ₀ + β z(a*)) |
| loss | BCEWithLogits |
| optimizer | AdamW (temporal params 10× LR) |
| max_seq_len | 96 |
| matched across arms | splits, dims, epochs, early stopping, init |

Arms: `no_age`, `age_only`, `temporal_only`, `age_temporal` (+ optional `historical_age`).

---

## Predictive metrics (controlled, smoke / mid-length runs)

Micro-AUROC on all 32 labels (8-epoch matched suite unless noted):

| Scenario | no_age | age_only | temporal_only | age_temporal |
|---|---|---|---|---|
| S0 | 0.685 | 0.688 | **0.734** | **0.734** |
| S1 | 0.706 | 0.709 | **0.748** | **0.748** |
| S2 (≤20 ep) | 0.736 | 0.737 | 0.764 | **0.765** |
| S3 (≤20 ep) | 0.711 | 0.713 | 0.738 | **0.738** |

Interaction-label AUROC (S2, 15 ep): temporal_only 0.877 · age_temporal **0.880**.

---

## Mechanism recovery

### Learned β

| Scenario | β_true | β̂ (age_temporal) | sign match |
|---|---|---|---|
| S0 | 0 | ≈ −0.08 | inert (≈0) |
| S1 | 0 | ≈ +0.03 | inert (≈0) |
| S2 | −2.5 | ≈ **−0.64** | **yes** |
| S3 | +2.5 | ≈ **+0.24** | **yes** |

Magnitude is underestimated in these runs; direction is recovered.

### Learned vs true λ(a)

S2: corr(λ̂, λ_true) ≈ **0.98**, RMSE_λ ≈ 0.65, RMSE_surface ≈ 0.22  
S3: corr ≈ **0.97**

### Functional ablations (age_temporal)

| Scenario | ΔBCE shuffle-age | ΔBCE β=0 | ΔBCE constant-age |
|---|---|---|---|
| S0 | ≈ 0 | ≈ 0 | ≈ 0 |
| S2 | +0.004 (weak) | +0.002 (weak) | — |
| S3 | ≈ 0 | ≈ 0 | — |

Functional use is **weaker than oracle / GLM baselines** at current training length —
models recover the **shape** of λ(a) and the **sign** of β, but have not yet
translated that into large predictive or ablation gaps vs `temporal_only`.

---

## Interaction-strength sweep (S2)

Built for $|\beta|\in\{1.2,2.5,4.0\}$. Figures: `results/figures/fig6_*`.
Expect larger shuffle degradation and easier recovery as $|\beta|$ increases;
re-run `run_benchmark.py --strength-sweep` for full multi-seed tables.

---

## Sanity tests

`tests/test_sanity.py` — all required invariants (disjoint splits, no future
events, z-anchors, λ>0, S0 β=0, S2/S3 opposite signs, relevance monotone in
lag, S2/S3 decay ordering, GT absent from tensors, β=0 ablation, gradients to
θ₀/β, prediction-time cutoff age, no NaN/Inf): **passed**.

---

## Figures

| Figure | Path |
|---|---|
| 1 mechanism | `results/figures/fig1_benchmark_mechanism.{png,svg}` |
| 2 true vs learned decay | `results/figures/fig2_decay_S2*.{png,svg}` |
| 3 age×lag surfaces | `results/figures/fig3_surfaces_S2*.{png,svg}` |
| 4 performance by scenario | `results/figures/fig4_performance_by_scenario.{png,svg}` |
| 5 mechanism ablations | `results/figures/fig5_mechanism_ablations.{png,svg}` |
| 6 strength sensitivity | `results/figures/fig6_interaction_strength.{png,svg}` |

---

## Scenario classification

| Setting | Classification | Rationale |
|---|---|---|
| controlled/S0/age_temporal | **MECHANISM RECOVERED** | β̂≈0; matches temporal_only; shuffle≈0 |
| controlled/S1/age_temporal | **MECHANISM RECOVERED** | no invented interaction; age_only ≥ no_age |
| controlled/S2/age_temporal | **PARTIALLY RECOVERED** | correct sign + λ curve; weak ablation / small AUROC gain |
| controlled/S3/age_temporal | **PARTIALLY RECOVERED** | correct **reversed** sign + λ curve; weak functional gap |

### Failure modes observed

1. **Magnitude under-recovery**: $|\hat\beta|\ll|\beta_{\mathrm{true}}|$ within ≤20 epochs.
2. **Dilution**: 24/32 labels lack the interaction, so micro-AUROC gaps vs
   `temporal_only` stay small even when interaction labels move.
3. **Ablation soft**: shuffle-age / β=0 BCE deltas are far below oracle deltas —
   the network still leans on content embeddings.

Recommended next steps (without retuning the generator to favor the model):
longer training, interaction-label-weighted loss for diagnostics only,
multi-seed `5×5` final tables, and the optional S4 nonlinear age probe.

---

## Reproducibility

```bash
cd synthetic_age_temporal
/home/suraj/miniconda3/envs/ehr/bin/python run_benchmark.py --quick
/home/suraj/miniconda3/envs/ehr/bin/python tests/test_sanity.py outputs/data/seed20260922
```

Production MIMIC/NCH preprocessing was **not** modified.

---

## Follow-up mechanism investigation

Purpose: determine whether partial recovery of the global age×temporal
mechanism is due to **sequence truncation**, **heterogeneous-label dilution**,
**optimization**, or **insufficient temporal-model capacity** — without
redesigning the generator. Generator, splits, targets, and ground truth are unchanged.

### 1. Signal visibility audit (exact model truncation)

Pipeline matches `BenchmarkDataset`: keep all signal events, subsample ≤48
background events, then keep the newest `max_seq_len−1=95` tokens.

- `max_seq_len=96`, `max_background=48`
- mean injected signals before truncation: 6.00
- mean retained after truncation: 6.00
- frac examples with ≥1 visible signal: **1.000**
- frac with **all** signals visible: **1.000**
- frac examples truncated: **0.000**

P(signal visible | lag) — controlled cohort:

| lag | P(visible) |
|---|---|
| 7d | 1.000 |
| 30d | 1.000 |
| 90d | 1.000 |
| 180d | 1.000 |
| 365d | 1.000 |
| 730d | 1.000 |

Figure: `results/figures/fig7_signal_visibility_L96.{png,svg}`

**Conclusion:** long-lag signals are fully visible to the network under the
current tensorization. No sequence-length sweep was required.

### 2. Full oracle vs visible oracle

| Scenario | full ΔBCE age-shuf (inter) | visible ΔBCE age-shuf (inter) | full AUROC | visible AUROC |
|---|---|---|---|---|
| S0 | 0.0000 | 0.0000 | 0.755 | 0.755 |
| S1 | 0.0420 | 0.0420 | 0.770 | 0.770 |
| S2 | **0.4653** | **0.4653** | 0.787 | 0.787 |
| S3 | **0.2494** | **0.2494** | 0.763 | 0.763 |

Figure: `results/figures/fig8_full_vs_visible_oracle.{png,svg}`

> Visible oracle ≡ full oracle. Sequence truncation is **not** an
> input-information bottleneck. The neural gap vs oracle (ΔBCE shuffle ≪ 0.46)
> is therefore a **model / objective** issue, not missing tokens.

### 3–4. Interaction-label-only + longer convergence (dilution test)

Diagnostic only (not a clinical recipe): train/evaluate on the **8 interaction
labels only**. Same histories, splits, architecture, optimizer family.

| Run | β̂ | sign | corr λ | ΔBCE shuffle | ΔBCE β=0 | AUROC | epochs |
|---|---|---|---|---|---|---|---|
| S2 temporal_only (inter) | 0.000 | — | — | 0.000 | 0.000 | 0.912 | 18 |
| S2 age_temporal (inter, tracked) | **−0.580** | True | 0.980 | 0.0065 | 0.0028 | **0.913** | 21 |
| S2 age_temporal (arch) | −0.643 | True | 0.982 | 0.0072 | 0.0029 | 0.913 | 22 |
| S3 temporal_only (inter) | 0.000 | — | — | 0.000 | 0.000 | 0.836 | 21 |
| S3 age_temporal (inter, tracked) | **+0.325** | True | 0.974 | 0.0020 | 0.0004 | 0.836 | 21 |

Convergence (S2 interaction-only, fig10): $|\hat\beta|$ rises then **plateaus
near 0.6 ≪ 2.5**; shuffle / β=0 deltas grow only to ~0.01 and stop; validation
AUPRC converges **before** any strong functional recovery.

**Dilution hypothesis: REJECTED.** Restricting supervision to interaction labels
does **not** produce meaningful predictive gain over `temporal_only`
(ΔAUROC ≈ 0.001) nor oracle-scale ablations. Heterogeneous-label dilution is
not the primary failure mode.

**Optimization vs model limitation:** trajectories show a plateau, not slow
monotonic climb toward $\beta_{\mathrm{true}}$. Issue is classified as a
**model/objective limitation** (content pathway dominates; parametric
age×temporal bias under-used), not merely insufficient epochs.

Figures: `fig9_global_dilution`, `fig10_convergence`.

### 5–8. Per-head age-conditioned kernels

$$
\lambda_h(a)=\mathrm{softplus}(\theta_{0,h}+\beta_h z(a)),\quad h=1\ldots 4
$$

(+ matched `temporal_only_per_head` with $\beta_h\equiv 0$). Prediction-cutoff
age $a_*$; softplus; no Fourier / Chebyshev / age MLP / per-target kernels.

#### Matched AUROC (arch runs)

| Scenario | temporal_only | age_temporal | temporal_only_per_head | age_temporal_per_head |
|---|---|---|---|---|
| S0 (all labels) | 0.734 | 0.734 | 0.734 | 0.734 |
| S1 (all labels) | 0.748 | 0.748 | 0.749 | 0.749 |
| S2 (inter only) | 0.913 | 0.913 | 0.913 | 0.913 |
| S3 (inter only) | 0.840 | 0.841 | 0.840 | 0.842 |

#### Mechanism diagnostics

| Scenario | model | mean β̂ | β vector | ΔBCE shuffle | ΔBCE β=0 |
|---|---|---|---|---|---|
| S0 | per-head | −0.07 | [−0.14,−0.06,+0.03,−0.11] | ≈0 | ≈0 |
| S1 | per-head | +0.05 | mixed small | ≈0 | ≈0 |
| S2 | global | −0.64 | — | 0.007 | 0.003 |
| S2 | per-head | −0.49 | [−0.10,−0.10,−0.57,−1.20] | 0.006 | 0.003 |
| S3 | global | +0.18 | — | 0.001 | 0.000 |
| S3 | per-head | +0.18 | [+0.18,+0.36,+0.05,+0.13] | 0.001 | 0.000 |

Falsification behavior:
- **S0/S1:** interaction path inert (AUROC matched to temporal-only; shuffle≈0).
- **S2:** correct negative direction (global and per-head).
- **S3:** **reverses** to positive β̂ (required; no fixed-sign bias).

Per-head does **not** improve interaction-label AUROC or ablations vs global.
One S2 head reaches $\beta_h\approx-1.2$ (closer to truth) without changing
predictions — further evidence the content encoder, not $\lambda_h\tau$,
carries most of the decision.

Figures: `fig11_global_vs_perhead`, `fig12_per_head_lambda`.

### 9. Dilution quantification (all 32 vs interaction-only)

| Setting (S2 age_temporal) | \|β̂\| | RMSE_λ | RMSE_surface | ΔBCE β=0 | ΔBCE shuffle | AUROC |
|---|---|---|---|---|---|---|
| all 32 labels (prior ≤20 ep) | ~0.64 | ~0.67 | ~0.23 | ~0.002 | ~0.004 | ~0.76 (all) |
| interaction only | ~0.58–0.64 | (corr≈0.98) | — | ~0.003 | ~0.007 | 0.913 |

Interaction-only slightly increases ablation deltas but nowhere near the
visible-oracle interaction ΔBCE of **0.465**. Dilution is a minor contributor
at most.

### 10–12. Weighted loss / seq-len / multi-seed

- Interaction-weighted loss: **not used** as a main solution (diagnostic
  interaction-only already covers the dilution question).
- Sequence-length follow-up (96/192/384): **skipped** — visibility audit showed
  P(visible)=1 at all lags.
- Multi-seed 5× grid: **not run**. Neither global nor per-head met the
  section-14 “mechanism validated” bar; expanding seeds would estimate
  variance of a still-non-functional recovery.

### Section-14 checklist (candidate = global or per-head age_temporal)

| Criterion | Status |
|---|---|
| 1. S0 inert | **pass** |
| 2. S1 no invented interaction | **pass** |
| 3. S2 developmental direction | **pass** (sign) |
| 4. S3 reversed direction | **pass** (sign) |
| 5. Interaction AUROC > temporal-only | **fail** (Δ≈0) |
| 6. Age shuffle hurts S2/S3 | **fail** (ΔBCE ≪ oracle) |
| 7. β=0 hurts S2/S3 | **fail** (ΔBCE ≈ 0.003) |
| 8. λ(a) / surface ≈ truth | **partial** (corr≈0.98; \|β\| under-recovered) |
| 9. Reproducible across seeds | not stress-tested (architecture not selected) |
| 10. Not just more parameters | per-head (+6 temporal params) does not help |

### Final interpretation

**AGE × TEMPORAL MECHANISM STILL NOT FUNCTIONALLY RECOVERED**

Evidence-based explanation:

1. **Not input visibility** — all injected lags survive tensorization; visible
   oracle = full oracle (S2 interaction shuffle ΔBCE = 0.465).
2. **Not label dilution** — interaction-only training still yields ≈0 AUROC
   gain over temporal-only and tiny ablations.
3. **Not solved by per-head kernels** — $H=4$ softplus kernels recover the
   correct S2/S3 signs (and reverse on S3) but do not improve functional use.
4. **What does recover:** parametric $\mathrm{sign}(\hat\beta)$ and the
   *shape* of $\lambda(a)$ (corr≈0.98). What does **not:** predictive
   dependence on the age×temporal bias path.

Most likely remaining bottleneck: the **content self-attention + embedding
pathway** can already approximate lag-dependent relevance without routing
gradient through $\mathrm{softplus}(\theta_0+\beta z)$, so the dedicated
kernel remains under-identified for prediction despite being statistically
identifiable (GLM baselines) and oracle-strong.

Recommended next scientific steps (still without retuning the generator):
kernel-only / content-ablated probes, stronger inductive bias tying
prediction to $\lambda(a)\tau$ (e.g. frozen content early), or a
diagnostic where content is deliberately uninformative — none of which
were in scope for this follow-up.

Reproduce:

```bash
cd synthetic_age_temporal
/home/suraj/miniconda3/envs/ehr/bin/python audit_visibility.py
/home/suraj/miniconda3/envs/ehr/bin/python run_followup.py --skip-multiseed --skip-seqlen
```

## Mechanism-identifiability and factorized architecture investigation

Goal: identify and remove the architectural bypass that allows prediction without
functionally using age×time. Generator / splits / S0–S3 ground truth unchanged.

### 1. Age-decoding probe (A1)

Linear probe on frozen representations → cutoff age $a_*$ (TRAIN fit, TEST eval).
Artifacts: `results/followup_age_probe.json`, fig13.

| Arm | site | MAE (y) | RMSE | R² | band acc |
|---|---|---|---|---|---|
| no_age | pre_pool | 0.780 | 1.034 | 0.952 | 0.981 |
| no_age | pooled | 0.802 | 1.081 | 0.948 | 0.981 |
| no_age | head_mean | 0.916 | 1.215 | 0.934 | 0.971 |
| temporal_only | pre_pool | 0.752 | 1.017 | 0.954 | 0.985 |
| temporal_only | pooled | 0.704 | 0.990 | 0.956 | 0.985 |
| temporal_only | head_mean | 0.866 | 1.172 | 0.938 | 0.971 |
| age_temporal | pre_pool | 0.731 | 1.007 | 0.955 | 0.987 |
| age_temporal | pooled | 0.689 | 0.973 | 0.958 | 0.986 |
| age_temporal | head_mean | 0.848 | 1.128 | 0.943 | 0.977 |

**Interpretation:** high predictability from `no_age` / `temporal_only` (R²≈0.95) implies
the content/history pathway reconstructs developmental age — age leakage / bypass.

### 2. Counterfactual age sensitivity (A2)

Fixed history; vary only $a_*\in\{2,5,9,13,17\}$. Artifacts: `results/counterfactual_age_sensitivity.json`, fig14.

| Model | mean ΔP(17−2) |
|---|---|
| temporal_only | 0.0000 |
| age_temporal | -0.0183 |
| oracle | 0.3590 |

`temporal_only` is flat (no age input). `age_temporal` Transformer barely moves vs oracle —
predictions do not route through the age×time gate despite representations encoding age.

### 3. Background-content ablation (A3)

**Decision: `BYPASS CONFIRMED: CONTENT PATH ENCODES AGE/DEVELOPMENTAL STATE`**

- full shuffle ΔBCE = 0.0072
- signal-only shuffle ΔBCE = 0.1743

| Variant | arm | AUROC | AUPRC | β̂ | shuffle ΔBCE | β=0 ΔBCE | age-probe R² |
|---|---|---|---|---|---|---|---|
| full | temporal_only | 0.9130 | 0.8319 | 0.000 | 0.0000 | 0.0000 | 0.955 |
| full | age_temporal | 0.9130 | 0.8337 | -0.643 | 0.0072 | 0.0029 | 0.955 |
| signal_only | temporal_only | 0.8174 | 0.5862 | 0.000 | 0.0000 | 0.0000 | -0.077 |
| signal_only | age_temporal | 0.8807 | 0.7411 | -4.468 | 0.1743 | 0.1571 | 0.578 |
| bg_randomized | temporal_only | 0.8631 | 0.7122 | 0.000 | 0.0000 | 0.0000 | 0.357 |
| bg_randomized | age_temporal | 0.8724 | 0.7534 | 1.654 | 0.0730 | 0.0291 | 0.563 |

Removing Synthea background restores large functional interaction sensitivity
(signal-only age_temporal: ΔAUROC≈0.06 vs temporal_only; shuffle ΔBCE≈0.17).
Figure: `fig15_background_ablation`.

### 4. M1 kernel-only (no Transformer)

Primary aggregation: **additive** $\sum_j g_j e_j$. Matched $\beta{=}0$ control = temporal_only.

| Aggregation | passed | ΔAUROC | ΔAUPRC | ΔBCE shuffle | ΔBCE β=0 | corr λ | β̂ |
|---|---|---|---|---|---|---|---|
| additive | True | 0.0207 | 0.0353 | 0.2258 | 0.0554 | 0.999 | -1.916 |
| softmax | False | -0.0019 | -0.0052 | 0.1597 | 0.0001 | -0.961 | 0.042 |
| additive_mass | True | 0.0176 | 0.0366 | 0.2028 | 0.0482 | 1.000 | -1.926 |

**SOFTMAX NORMALIZATION DESTROYS ABSOLUTE TEMPORAL EVIDENCE MAGNITUDE**

Additive and additive_mass pass the mechanism gate; softmax fails sign/β=0/λ correlation.
Ground truth uses a SUM of weighted relevance — softmax cancels absolute mass.

### 5. Additive vs normalized aggregation

Figure 19. Finding above stands: prefer unnormalized additive evidence for this mechanism.

### 6. M2 — content relevance × temporal gate

Passed: **True**. ΔAUROC=0.0094, ΔAUPRC=0.0191,
shuffle ΔBCE=0.1302, β=0 ΔBCE=0.0440,
β̂=-1.785, corr λ=1.000.
Content score $u_j=q^\top k_j$ does not see age/τ; age×time only via $g_j$.

### 7. M3 — encounter encoder + developmental temporal retrieval

Passed: **True** (same gate metrics as M2 under event-as-encounter encoding).
β̂=-1.785.
Selected as real-EHR candidate architecture (global β, additive aggregation).

### 8. S0–S3 validation (M3)

| Scenario | β_true | β̂ | AUROC | shuffle ΔBCE | β=0 ΔBCE | gate |
|---|---|---|---|---|---|---|
| S0 | 0.0 | -0.007 | 0.7315 | 0.0041 | 0.0000 | n/a |
| S1 | 0.0 | 0.063 | 0.7431 | 0.0134 | 0.0002 | n/a |
| S2 | -2.5 | -1.785 | 0.9180 | 0.1302 | 0.0440 | True |
| S3 | 2.5 | 1.792 | 0.8531 | 0.1397 | 0.0601 | True |

S0/S1: $\hat\beta\approx 0$, small shuffle. S2: $\hat\beta<0$ + functional sensitivity. S3: $\hat\beta>0$ (falsification).

### 9. Final selected architecture

**M3** = encounter content encoder (no age/τ) + developmental temporal retrieval
$g_m=\exp[-\lambda(a_*)\tau_{*m}]$ with $\lambda=\mathrm{softplus}(\theta_0+\beta z)$, **additive** aggregation.
Do not use softmax temporal normalization for this mechanism.

Figure 16 ladder (GLM / M1 / M2 / M3 / Transformer):
- GLM: ΔAUROC=0.0275, shuffle=n/a, β0=n/a
- M1_additive: ΔAUROC=0.0207, shuffle=0.2258, β0=0.0554
- M2: ΔAUROC=0.0094, shuffle=0.1302, β0=0.0440
- M3: ΔAUROC=0.0094, shuffle=0.1302, β0=0.0440
- Transformer: ΔAUROC=0.0002, shuffle=0.0065, β0=0.0028

### 10. Functional mechanism recovery

Yes — under factorized additive architectures (M1→M3). The Transformer bypass was
background-content age encoding + softmax-style normalization that destroys absolute
temporal evidence mass. Removing the bypass recovers sign, $\lambda(a)$, and ablation sensitivity.

**MECHANISM FUNCTIONALLY RECOVERED — READY FOR MULTI-SEED**

Figures: fig13–fig19 under `results/figures/`.
Artifacts: `results/followup/`, `outputs/runs/controlled/factorized/`, `results/followup_age_probe.json`,
`results/counterfactual_age_sensitivity.json`.

## Final Developmental Temporal Retrieval validation

### 1. Final architecture

Encounter DeepSets encoder (no age/τ) → content relevance $u_m=q^\top k_m$ →
developmental gate $g_m=\exp[-\lambda(a_*)\tau_m]$ →
**raw_additive** aggregation → $\ell=f_{\mathrm{history}}(h)+f_{\mathrm{age}}(z)+b$.

Preferred `weighted_mean_plus_log_mass` did not clear the predictive-gain gate; `raw_additive` selected. Diagnosis: weighted_mean_plus_log_mass failed predictive ΔAUROC/AUPRC gate (got ΔAUROC=0.0040, ΔAUPRC=0.0087); raw_additive selected as validated encounter-level aggregation.

### 2. Encounter construction

Controlled encounters/patient mean=24.74; codes/encounter mean=2.83; signal encounters mean=5.90.
Full: encounters mean=20.61.

### 3. Aggregation validation

- raw_additive passed: True
- weighted_mean_plus_log_mass passed: False
- selected: `raw_additive`

### 4. Controlled S0–S3

| Scenario | ΔAUROC | ΔAUPRC | β̂ | shuffle ΔBCE | β=0 ΔBCE | corr λ | passed |
|---|---|---|---|---|---|---|---|
| S0 | 0.0027 | 0.0008 | -0.034 | 0.0054 | 0.0000 | n/a | True |
| S1 | 0.0000 | -0.0000 | 0.058 | 0.0098 | 0.0002 | n/a | True |
| S2 | 0.0106 | 0.0238 | -1.912 | 0.1926 | 0.0917 | 1.000 | True |
| S3 | 0.0257 | 0.0420 | 2.092 | 0.1502 | 0.0735 | 1.000 | True |

### 5. Full-realism

| Scenario | ΔAUROC | β̂ | shuffle ΔBCE | β=0 ΔBCE | passed |
|---|---|---|---|---|---|
| S0 | -0.0002 | -0.045 | 0.0091 | 0.0000 | True |
| S1 | 0.0003 | -0.117 | 0.0252 | 0.0002 | True |
| S2 | 0.0237 | -2.225 | 0.2702 | 0.1463 | True |
| S3 | 0.0113 | 1.786 | 0.1601 | 0.0699 | True |

### 6. Mechanism recovery

Controlled S2 β̂=-1.912, corr λ=1.000, surface RMSE=0.0457.
Controlled S3 β̂=2.092.
Full S2 β̂=-2.225, shuffle=0.2702.

### 7. Counterfactual test

- temporal-only history invariant: True
- age-temporal history varies via gate: True
- structural identifiability OK: True

### 8. Strength sensitivity

- |β|=1.2: β̂=-1.158, ΔAUROC=0.0142, shuffle=0.0693
- |β|=2.5: β̂=-1.912, ΔAUROC=0.0106, shuffle=0.1926
- |β|=4.0: β̂=-3.349, ΔAUROC=0.0189, shuffle=0.2512

### 9. Architecture comparison

| Model | ΔAUROC | shuffle ΔBCE | β=0 ΔBCE | recovered? |
|---|---|---|---|---|
| Original Transformer | 0.0002 | 0.0065 | 0.0028 | False |
| GLM interaction | 0.0275 | n/a | n/a | True |
| M1 additive | 0.0207 | 0.2258 | 0.0554 | True |
| M2 | 0.0094 | 0.1302 | 0.0440 | True |
| event-M3 | 0.0094 | 0.1302 | 0.0440 | True |
| Final encounter-level DTR | 0.0106 | 0.1926 | 0.0917 | True |

### 10. Paper-ready figure/table index

Figures: `results/figures/final/figA`–`figH` (PNG+SVG).
Tables/artifacts: `results/final_validation/`.

### 11. Final verdict

**FINAL SYNTHETIC ARCHITECTURE VALIDATED — READY FOR MIMIC IMPLEMENTATION**

#### Paper-level claims

- claim1_identifiable: **SUPPORTED**
- claim2_transformer_bypass: **SUPPORTED**
- claim3_factorization_restores: **SUPPORTED**
- claim4_mass_preserving: **SUPPORTED**
- claim5_full_realism: **SUPPORTED**
- claim6_falsifiable: **SUPPORTED**


## Model improvement experiments
### Experiment 1: Content-dependent temporal persistence
Result: Improved S5 AUPRC — **promoted to locked canonical architecture** (see below).

### Experiment 2: Multi-query / target-aware content retrieval
Result: Failed to improve S6 AUPRC — rejected.

### Experiment 3: Multi-horizon temporal supervision
Result: Improved mean forecasting AUPRC in isolation, but **naive joint training with content persistence inverted β on S2** (β̂≈+1.36 when β_true=-2.5). Multi-horizon remains experimental only; not part of the locked canonical model.

RECOMMENDED FOR MIMIC:
Content-Persistence DTR + single-task training (`train_dtr.py`)

REJECTED / EXPERIMENTAL:
- Multi-query retrieval (rejected)
- Multi-horizon joint supervision with content persistence (experimental; mechanism sign failure)


## Canonical Content-Persistence DTR implementation

Locked architecture (canonical production model):

$$
v_m = f_{\mathrm{enc}}(C_m)
$$

$$
u_m = q^\top k_m
$$

$$
\theta_m = \theta_0 + r^\top v_m
$$

$$
\lambda_m(a_*) = \operatorname{softplus}(\theta_m + \beta z(a_*))
$$

$$
g_m = \exp[-\lambda_m(a_*)\tau_m],\quad
w_m = \exp(u_m)\,g_m,\quad
h = \sum_m w_m v_m
$$

$$
\ell = f_{\mathrm{history}}(h) + f_{\mathrm{age}}(z(a_*)) + b
$$

Raw additive aggregation only. No softmax over encounters. Single global β. Linear persistence projection. Separated age main-effect head.

### Files changed

- `synthetic_age_temporal/model_dtr.py` — canonical Content-Persistence DTR
- `synthetic_age_temporal/train_dtr.py` — canonical trainer + persistence diagnostics
- `synthetic_age_temporal/tests/test_dtr.py` — 20 required contract tests (+ extras)
- `synthetic_age_temporal/train_multi_horizon.py` — marked experimental; API aligned
- `synthetic_age_temporal/validate_canonical_dtr.py` — lightweight S0–S3 / S5 regression
- `dataset_dtr.py` — unchanged (model inputs already sufficient; S5 groups are eval-only)

### Initialization contract

- `beta = 0` for both arms at init
- `persistence_projection` weights and bias initialized to **0** so θ_m ≈ θ₀ at step 0
- Gradients still reach `persistence_projection` from the first step
- Matched arms share identical weights; `max_abs_logit_diff < 1e-6` before training

### Matched arm definition

| Arm | λ_m | β |
|---|---|---|
| `temporal_only` | softplus(θ₀ + rᵀv_m) | frozen at 0 |
| `age_temporal` | softplus(θ₀ + rᵀv_m + β z(a*)) | trainable |

Otherwise identical modules (encoder, content relevance, persistence projection, heads).

### Numerical stability

`w = exp(u)·g` uses `u.clamp(max=20)` before `exp` (`CONTENT_SCORE_EXP_CLAMP=20`). This caps overflow without softmax renormalization; for typical scores the quantity is unchanged.

### Checkpoint compatibility

`load_legacy_dtr_checkpoint(...)` migrates older keys (`gate.theta0/beta`, `W_r`, `W_k`, `q`, bare `code_emb`/`enc_mlp`, `f_history`/`f_age`). Missing `persistence_projection.*` triggers an explicit warning and leaves zero init (θ_m ≈ θ₀). Not silently ignored.

### Config fields saved per run

```text
content_dependent_persistence: true
persistence_projection: linear
temporal_aggregation: raw_additive
num_content_queries: 1
age_conditioning: linear_softplus
beta_scope: global
```

### Tests passed

All 29 tests in `tests/test_dtr.py`, including the 20 required contract checks (β=0 arm identity, gradients to β/θ₀/persistence, pad masking, raw additive, no softmax, S2/S3 sign capacity, no GT leakage, matched init, etc.).

### Regression results (`results/canonical_dtr/regression_summary.json`)

| Scenario | β̂ | Expectation | Pass |
|---|---|---|---|
| S0 | +0.009 | β≈0, β=0 ablation ≈0 | Yes |
| S1 | +0.075 | ordinary age ok; interaction inert | Yes |
| S2 | −2.358 | β<0; age-shuffle / β=0 hurt | Yes |
| S3 | +2.065 | β>0 | Yes |

S2 smoke (8 epochs): β̂=−1.92, AUROC=0.922.

S5 heterogeneous persistence:

- Content-Persistence AUPRC **0.817** > global-persistence baseline **0.812**
- Learned group offsets: acute **+1.11** > intermediate **+0.20** > chronic **−0.53** (correct ordering)

### Multi-horizon note

Multi-horizon supervision is currently experimental and is not part of the locked canonical architecture because naive joint training with content persistence inverted beta on the synthetic S2 benchmark.

### Verdict

**CANONICAL CONTENT-PERSISTENCE DTR IMPLEMENTATION READY**
