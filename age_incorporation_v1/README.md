# Developmental Age Incorporation Benchmark v1

Five ways of putting **developmental age** into the **same** longitudinal Transformer, evaluated on the sep1-exp Synthea semi-synthetic benchmark (S0 / S1 / S2).

This folder is self-contained. It does not import or modify `model/`, `model_new/`, `model_ablation/`, or `finetune/`.

```
age_incorporation_v1/
  config.py           shared hyperparameters
  dataset.py          Synthea benchmark loader (swap later for NCH / ECHO / PIC)
  model.py            dataset-independent Transformer + 5 arms
  train.py            AdamW + val-AUPRC early stopping
  evaluate.py         AUPRC / AUROC / BCE, overall and by age group
  run_experiment.py   smoke / full / dkm / single-run CLI
  README.md           this file
  outputs/            run artifacts (gitignored)
```

---

## Logic flow

```
sep1-exp patients.parquet + events.parquet
        │  existing split, y_S0/y_S1/y_S2, pre-index events only
        ▼
dataset.py
  • vocab from TRAIN codes/types only (<PAD>=0, <UNK>=1)
  • chronological sort, keep 1024 most recent events
  • time_norm, age_at_event_norm, index_age_norm
        ▼
model.py  (arm ∈ {no_age, late_age, additive_age, conditioned_age, dkm_age})
  base_i → arm-specific x_i → LayerNorm → Transformer → masked mean
  → concat head_age scalar → Linear head → logit
  dkm_age only: attention logits -= λ(a_i) · τ_ij
        ▼
train.py  BCEWithLogitsLoss, checkpoint = best validation AUPRC
        ▼
evaluate.py  test AUPRC / AUROC / BCE overall and <1, 1–5, 6–11, 12–17
```

Age groups are **analysis only**. They are never model inputs.

---

## Data used (not regenerated)

| File | Role |
|---|---|
| `synthea/sep1-exp/output/full/processed/patients.parquet` | split, `age_at_index`, `y_S0/S1/S2`, age group |
| `synthea/sep1-exp/output/full/processed/events.parquet` | codes, types, `time_before_index_days`, `age_at_event` |

Model inputs: `event_code`, `event_type`, `time_before_index`, and (additive/conditioned only) `age_at_event`. Index age goes to the prediction head except in `no_age`. `dkm_age` also uses query-event age inside attention.

Not used as features: DOB, calendar timestamps, `p_S*`, `has_SIGNAL_*`, `age_at_SIGNAL_*`, generation stratum.

---

## Event representation (all arms)

`d_model = 128`.

```
time_norm_i = clip( log1p(days_before_index_i) / log1p(18 × 365.25) , 0, 1 )
age_norm    = clip( age_years / 18 , 0, 1 )

base_i = Embedding_code(code_i)
       + Embedding_type(type_i)
       + Linear_1→128(time_norm_i)
```

No Fourier / Chebyshev / custom temporal kernels except the simplified DKM decay below.

Shared event-age encoder (always constructed; used only by additive and conditioned):

```
AgeEnc: Linear(1,32) → GELU → Linear(32,128)
        last Linear is zero-initialized
z_age_i = tanh(AgeEnc(age_at_event_norm_i))
```

At initialization `z_age_i = 0`, so additive and conditioned start as `base_i`.

Shared DKM generator (always constructed; used only by `dkm_age`):

```
AgeLambda: Linear(1,32) → GELU → Linear(32,1)
           last Linear is zero-initialized
λ_base_raw  scalar parameter, init 0
```

---

## Arm equations

After `x_i` is formed, **every** arm applies the same `LayerNorm(128)`, then a 2-layer Transformer encoder (`heads=4`, `ff=256`, dropout `0.10`, GELU, `norm_first=True`, padding mask only, **not** causal). Patient vector = masked mean of valid positions.

Head input is always 129-D: `[patient_vector ; head_age]`.

```
head: Linear(129,64) → GELU → Dropout(0.10) → Linear(64,1)
```

| Arm | Event mixing | `head_age` | Extra | What it tests |
|---|---|---|---|---|
| **no_age** | `x_i = base_i` | `0` | — | history without age |
| **late_age** | `x_i = base_i` | `age_index / 18` | — | conventional H1: current age at the head only |
| **additive_age** | `x_i = base_i + z_age_i` | `age_index / 18` | — | additive event-level age |
| **conditioned_age** | `x_i = base_i ⊙ (1 + z_age_i)` | `age_index / 18` | — | multiplicative developmental conditioning |
| **dkm_age** | `x_i = base_i` | `age_index / 18` | attention `score_ij -= λ(a_i)·τ_ij` | age-dependent temporal discounting |

All five objects contain the same modules, so **parameter count is identical**. Forward behavior is the only difference. `age_at_event` is ignored in `no_age` and `late_age` embeddings; in `dkm_age` it controls `λ(a_i)` only. Current age is zeroed only in `no_age`.

No second β/shift path and no full FiLM in this version.

### Simplified DKM (`dkm_age`)

Query age (not index age) sets a single shared decay rate, reused by every head and both layers:

```
age_norm_i = clip(age_at_event_i / 18, 0, 1)
Δλ_i       = AgeLambda(age_norm_i)
λ_i        = softplus(λ_base_raw + Δλ_i)

days_i     = expm1(time_norm_i · log1p(18 × 365.25))
τ_ij       = clip( log1p(|days_i − days_j|) / log1p(18 × 365.25) , 0, 1 )

score_ij   = q_iᵀ k_j / √d_head  −  λ_i · τ_ij
```

Then the usual padding mask and softmax. Larger `λ(a_i)` discounts temporally distant keys faster. Age is **not** added to the event embedding.

Primary contrast: `dkm_age` vs `late_age`. Both see current index age at the head; `dkm_age` adds only age-dependent temporal weighting.

S2 is an age × event-meaning interaction, **not** an age-dependent temporal-decay process. A null S2 result does not disprove the DKM temporal hypothesis.

---

## Training (identical for every arm and task)

| | |
|---|---|
| Loss | BCEWithLogitsLoss (no class weights) |
| Optim | AdamW, lr `3e-4`, weight decay `1e-2` |
| Batch / epochs | 32 / 30 |
| Grad clip | 1.0 |
| Early stop | validation **AUPRC**, patience 5 |
| Seeds | 0–4; the same seed is reset before each arm |
| Checkpoint | `checkpoint_best.pt` (highest val AUPRC; used for test) and `checkpoint_last.pt` (final epoch, including early stop) |

`dkm_age` runs also write `dkm_diagnostics.json`: `λ_base`, `λ(a)` at ages 0, 1, 3, 5, 8, 12, 15, 17, mean/SD of `Δλ`, and age-generator gradient norm.

---

## Commands

Smoke test (S2, four baseline arms, seed 0, 2 epochs):

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --smoke
```

Full 60-run baseline matrix (3 tasks × 4 arms × 5 seeds). Do not rerun after the first complete pass:

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --full
```

DKM correctness smoke (S2, `dkm_age`, seed 0, 2 epochs):

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --smoke-dkm
```

DKM arm only (3 tasks × 5 seeds = 15 runs), appended to `summary_full.csv`:

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --dkm
```

One run:

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py \
  --task S2 --arm dkm_age --seed 0
```

---

## Design choices

- **New folder, new code** so later NCH/ECHO/PIC loaders only replace `dataset.py` if they emit the same batch keys (`code_ids`, `type_ids`, `time_norm`, `age_event_norm`, `padding_mask`, `index_age_norm`, `labels`, `age_group`).
- **One model class** so arms are comparable (same init, same param count). DKM parameters exist on every arm and are unused outside `dkm_age`.
- **Masked mean pooling** on purpose: no CLS, no attention pooling, no age-conditioned pooling.
- **Truncation = last 1024 chronological events**, including SIGNAL_A/B if they fall in that window. Not tuned to keep signals.
- **S0/S1/S2 share architecture and hyperparameters.** Differences should come from how age is incorporated, not from per-task tuning.
- Do not treat “conditioned_age wins S2” or “dkm_age wins S2” as an implementation requirement.
- Do not change the model based on `dkm_age` results.

Each run directory `outputs/{task}_{arm}_seed{seed}/` contains:

- `history.json` — rewritten after every epoch: train BCE, val AUPRC/AUROC/BCE, val metrics by age group, `is_best`
- `metrics.json` — full run summary (same history plus test metrics from the **best** checkpoint)
- `config.json`
- `checkpoint_best.pt`, `checkpoint_last.pt`
- `dkm_age` only: `dkm_diagnostics.json`

Intended contrasts (interpretation, not acceptance tests):

- S0: `dkm_age` vs `no_age` / `late_age` — does temporal conditioning introduce a spurious gain?
- S1: `dkm_age` vs `late_age` — age is only a main effect; temporal conditioning is not expected to help
- S2: `dkm_age` vs `late_age` / `additive_age` / `conditioned_age` — S2 is event-meaning × age, not generated temporal decay

---

## S4: True Age-Dependent Temporal Decay

### Scientific question

Does developmental age change how rapidly historical clinical information loses predictive relevance?

Unlike S2 (age × event-meaning interaction), **S4 contains a true AGE × TIME interaction** in the data-generating process.

### Data generation

Eligible patients: ≥90 days pre-index history (9,407 of 10,000). Original split assignments are preserved.

For each eligible patient, inject:
- 1 `TEMP_QUERY` event 1 day before index
- 6 historical temporal-signal events (`TEMP_POS` or `TEMP_NEG`, each with probability 0.5)

Each signal's gap before `TEMP_QUERY` is sampled log-uniformly between 7 and 90 days (common support across all ages — no age–time confound).

### True temporal mechanism

```
λ_true(a) = 0.5 + 2.5 · exp(−a / 4.0)

τ_k = log1p(gap_days_k) / log1p(18 × 365.25)    clipped to [0,1]
weight_k = exp(−λ_true(a_query) · τ_k)

temporal_score = Σ_k polarity_k · weight_k / √6

score_S4 = intercept_S4 + 2.0 · temporal_score + noise
    noise ~ N(0, 0.25)

p_S4 = sigmoid(score_S4)
y_S4 ~ Bernoulli(p_S4)
```

`intercept_S4` is calibrated to ~22% prevalence. No other parameters are tuned.

### Difference from S2

| | S2 | S4 |
|---|---|---|
| Interaction | age × event meaning | age × temporal gap × polarity |
| What age modulates | Which code is protective vs. risky | How fast historical signals decay |
| DKM-relevant | No (S2 is not a temporal-decay task) | Yes (S4 directly tests the DKM hypothesis) |

### `shared_decay` control arm

`shared_decay` uses the same temporal attention mechanism as `dkm_age`, but **age cannot modify λ**:

```
shared_decay:   λ = softplus(λ_base)                      # age-independent
dkm_age:        λ(a) = softplus(λ_base + Δλ(a))          # age-dependent
```

Otherwise identical. The key comparison:

- **`dkm_age` > `shared_decay`**: evidence that age-conditioned temporal weighting helps when the true process has age-dependent decay
- **`shared_decay` ≈ `dkm_age`**: age-conditioned decay is not being exploited
- **`additive_age` / `conditioned_age` match DKM**: generic event-level age interactions may suffice; DKM may not provide a useful inductive bias

### Commands

Build S4 data (one-time, ~15 min):
```bash
conda run -n ehr python age_incorporation_v1/build_s4.py
```

S4 smoke test (6 arms × seed 0 × 2 epochs):
```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --smoke-s4
```

Full S4 run (6 arms × 5 seeds = 30 runs, seed-major order):
```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --s4
```

### Interpretation

DKM diagnostics compare learned λ(a) with the known λ_true(a) = 0.5 + 2.5·exp(−a/4). This mechanism-recovery comparison is as important as predictive performance.

### S4 results (5 seeds, mean ± SD)

Test metrics from `outputs/summary_s4.csv`. Previous S0–S2 rows in `outputs/summary_full.csv` were not overwritten.

| Arm | AUPRC | AUROC | AUPRC &lt;1 | AUPRC 1–5 | AUPRC 6–11 | AUPRC 12–17 |
|---|---|---|---|---|---|---|
| no_age | 0.4965 ± 0.0019 | 0.7916 ± 0.0015 | 0.3557 ± 0.0084 | 0.4198 ± 0.0043 | 0.5365 ± 0.0067 | 0.5900 ± 0.0060 |
| late_age | 0.4962 ± 0.0019 | 0.7915 ± 0.0017 | 0.3554 ± 0.0082 | 0.4200 ± 0.0044 | 0.5349 ± 0.0060 | 0.5905 ± 0.0064 |
| additive_age | 0.4978 ± 0.0037 | 0.7915 ± 0.0016 | 0.3547 ± 0.0082 | 0.4213 ± 0.0050 | 0.5349 ± 0.0073 | 0.5893 ± 0.0065 |
| conditioned_age | 0.4983 ± 0.0032 | 0.7920 ± 0.0013 | 0.3549 ± 0.0093 | 0.4200 ± 0.0042 | 0.5339 ± 0.0065 | 0.5902 ± 0.0052 |
| shared_decay | 0.5002 ± 0.0039 | 0.7917 ± 0.0011 | 0.3571 ± 0.0085 | 0.4205 ± 0.0071 | 0.5373 ± 0.0054 | 0.5917 ± 0.0044 |
| dkm_age | 0.4979 ± 0.0043 | 0.7918 ± 0.0009 | 0.3572 ± 0.0077 | 0.4189 ± 0.0048 | 0.5368 ± 0.0057 | 0.5922 ± 0.0040 |

Paired seed-wise differences (`dkm_age − other`):

| Contrast | ΔAUPRC | ΔAUROC |
|---|---|---|
| dkm_age − shared_decay | −0.0024 ± 0.0053 | +0.0001 ± 0.0005 |
| dkm_age − late_age | +0.0017 ± 0.0046 | +0.0003 ± 0.0021 |
| dkm_age − additive_age | +0.0000 ± 0.0055 | +0.0003 ± 0.0020 |
| dkm_age − conditioned_age | −0.0004 ± 0.0039 | −0.0002 ± 0.0020 |

**Primary finding:** `dkm_age` ≈ `shared_decay`. Age-conditioned decay is not being exploited beyond a generic learned temporal decay. All six arms sit near the oracle AUROC (~0.79); polarity count alone already reaches 0.778, so the remaining age × time increment is small.

**Mechanism recovery:** learned `λ(a)` does **not** recover `λ_true(a) = 0.5 + 2.5 e^{−a/4}` (which falls from 3.0 at age 0 to 0.54 at age 17). `dkm_age` stays nearly flat around 0.60; `shared_decay` is a constant ~0.68. See `outputs/s4_lambda_curve_dkm_age.png`.
