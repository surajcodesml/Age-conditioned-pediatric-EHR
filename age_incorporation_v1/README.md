# Developmental Age Incorporation Benchmark v1

Four ways of putting **developmental age** into the **same** longitudinal Transformer, evaluated on the sep1-exp Synthea semi-synthetic benchmark (S0 / S1 / S2).

This folder is self-contained. It does not import or modify `model/`, `model_new/`, `model_ablation/`, or `finetune/`.

```
age_incorporation_v1/
  config.py           shared hyperparameters
  dataset.py          Synthea benchmark loader (swap later for NCH / ECHO / PIC)
  model.py            dataset-independent Transformer + 4 arms
  train.py            AdamW + val-AUPRC early stopping
  evaluate.py         AUPRC / AUROC / BCE, overall and by age group
  run_experiment.py   smoke / full / single-run CLI
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
model.py  (arm ∈ {no_age, late_age, additive_age, conditioned_age})
  base_i → arm-specific x_i → LayerNorm → Transformer → masked mean
  → concat head_age scalar → Linear head → logit
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

Model inputs: `event_code`, `event_type`, `time_before_index`, and (additive/conditioned only) `age_at_event`. Index age goes to the prediction head except in `no_age`.

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

No Fourier / Chebyshev / custom temporal attention.

Shared event-age encoder (always constructed; used only by additive and conditioned):

```
AgeEnc: Linear(1,32) → GELU → Linear(32,128)
        last Linear is zero-initialized
z_age_i = tanh(AgeEnc(age_at_event_norm_i))
```

At initialization `z_age_i = 0`, so additive and conditioned start as `base_i`.

---

## Arm equations

After `x_i` is formed, **every** arm applies the same `LayerNorm(128)`, then a 2-layer Transformer encoder (`heads=4`, `ff=256`, dropout `0.10`, GELU, `norm_first=True`, padding mask only, **not** causal). Patient vector = masked mean of valid positions.

Head input is always 129-D: `[patient_vector ; head_age]`.

```
head: Linear(129,64) → GELU → Dropout(0.10) → Linear(64,1)
```

| Arm | Event mixing | `head_age` | What it tests |
|---|---|---|---|
| **no_age** | `x_i = base_i` | `0` | history without age |
| **late_age** | `x_i = base_i` | `age_index / 18` | conventional H1: current age at the head only |
| **additive_age** | `x_i = base_i + z_age_i` | `age_index / 18` | additive event-level age |
| **conditioned_age** | `x_i = base_i ⊙ (1 + z_age_i)` | `age_index / 18` | multiplicative developmental conditioning |

All four objects contain the same modules, so **parameter count is identical**. Forward behavior is the only difference. `age_at_event` is ignored in `no_age` and `late_age`. Current age is zeroed only in `no_age`.

No second β/shift path and no full FiLM in this version.

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

---

## Commands

Smoke test (S2, four arms, seed 0, 2 epochs):

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --smoke
```

Full 60-run matrix (3 tasks × 4 arms × 5 seeds):

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py --full
```

One run:

```bash
conda run -n ehr python age_incorporation_v1/run_experiment.py \
  --task S2 --arm conditioned_age --seed 0
```

---

## Design choices

- **New folder, new code** so later NCH/ECHO/PIC loaders only replace `dataset.py` if they emit the same batch keys (`code_ids`, `type_ids`, `time_norm`, `age_event_norm`, `padding_mask`, `index_age_norm`, `labels`, `age_group`).
- **One model class** so arms are comparable (same init, same param count).
- **Masked mean pooling** on purpose: no CLS, no attention pooling, no age-conditioned pooling.
- **Truncation = last 1024 chronological events**, including SIGNAL_A/B if they fall in that window. Not tuned to keep signals.
- **S0/S1/S2 share architecture and hyperparameters.** Differences should come from how age is incorporated, not from per-task tuning.
- Do not treat “conditioned_age wins S2” as an implementation requirement.

Each run directory `outputs/{task}_{arm}_seed{seed}/` contains:

- `history.json` — rewritten after every epoch: train BCE, val AUPRC/AUROC/BCE, val metrics by age group, `is_best`
- `metrics.json` — full run summary (same history plus test metrics from the **best** checkpoint)
- `config.json`
- `checkpoint_best.pt`, `checkpoint_last.pt`

Intended contrasts (interpretation, not acceptance tests):

- S0: does age injection help when age is not in the DGP?
- S1: `late_age` vs `no_age` — can current age recover a main effect?
- S2: `conditioned_age` vs `late_age` — does event-level conditioning beat current age?
- S2: `conditioned_age` vs `additive_age` — multiply vs add?
