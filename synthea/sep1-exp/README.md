# Age-signal pediatric EHR benchmark (1 Sep 2026)

Semi-synthetic pediatric dataset for testing **how developmental age should enter a longitudinal EHR model**. Stock Synthea supplies realistic background events. Two artificial codes (`SIGNAL_A`, `SIGNAL_B`) and three labels (`y_S0`, `y_S1`, `y_S2`) are injected afterward so the age relationship is known.

This folder is self-contained. It does **not** use the custom obesity/T2D/OSA/asthma modules, and it does **not** overwrite `synthea/output/` or `data/synthea/`.

```bash
bash synthea/sep1-exp/generate_age_benchmark.sh full
conda run -n ehr python synthea/sep1-exp/build_age_benchmark.py --mode full
```

Pilot (`40` patients/stratum) is the same command with `pilot` instead of `full`.

---

## Layout

```
synthea/sep1-exp/
  README.md                          this file
  age_benchmark_config.json          all generation / label knobs
  age_benchmark.properties           Synthea export overrides
  generate_age_benchmark.sh          four age-stratum Synthea runs
  build_age_benchmark.py             signals, labels, splits, validation
  age_benchmark_validation_report.md snapshot of the full-run report
  age_benchmark_s2_curves.{png,csv}  ground-truth S2 age × event curves
  output/                            generated data (gitignored)
    pilot|full/raw/<stratum>/csv/    Synthea CSVs
    pilot|full/processed/            patient/event tables + report
```

---

## Overall flow

```
1. Synthea (stock modules, 4 age runs)
        ↓  CSV: patients, encounters, conditions, meds, procedures, obs, immunizations
2. Union strata, define index date = 2026-01-01
        ↓  keep only events with timestamp < index
3. Inject SIGNAL_A / SIGNAL_B (independent, ~50%, one each, random historical time)
        ↓
4. Score three logits on the SAME patients
        S0: events only
        S1: events + current age
        S2: event × age-at-occurrence
        ↓  Bernoulli labels; intercepts calibrated to ~22% prevalence
5. Patient-level 70/15/15 split, stratified by developmental age group
6. Validation report (coverage, history, signal balance, H0/H1/H2, leakage)
```

Later models see **only pre-index events**. Labels live on the patient table. Do not feed `y_*`, `p_*`, or `has_SIGNAL_*` as input features.

---

## Knobs that were turned

All values live in `age_benchmark_config.json` except Synthea exporter flags in `age_benchmark.properties`. Coefficients were **not** searched against any model.

### Synthea population

| Knob | Value | Why |
|---|---|---|
| Engine | `/home/suraj/Git/synthea` (unmodified) | reuse existing clone |
| Custom modules (`-d`) | **not loaded** | background noise only |
| Geography | Massachusetts | one fixed location |
| Reference date `-r` | `20260101` | reproducible “today” |
| End date `-e` | `20260101` | **required**; default end time is the machine clock |
| Alive only | `true` | every index date equals the reference date |
| Overflow `-o` | `false` | do not keep extra deceased records |
| History | `exporter.years_of_history = 0` | keep birth→index events (default 10 years would drop early childhood) |
| FHIR / billing | off | CSV only |
| CSV tables | patients, encounters, conditions, medications, procedures, observations, immunizations | enough to rebuild timelines |
| Strata | 4 separate runs, ~2500 living patients each | balanced developmental coverage |

Age flags (Synthea’s integer max is exclusive after `(int)` cast):

| Group | `-a` | Seed |
|---|---|---|
| `<1` | `0-0` | 202601011 |
| 1–5 | `1-6` | 202601012 |
| 6–11 | `6-12` | 202601013 |
| 12–17 | `12-18` | 202601014 |

### Index time

- Living patient: `index_date = 2026-01-01`
- Age at index = (index − DOB) / 365.25
- Developmental group from that age: `[0,1)`, `[1,6)`, `[6,12)`, `[12,18)`
- Events with `timestamp >= index` are dropped

### Synthetic signals

| Knob | Value |
|---|---|
| Codes | `SIGNAL_A`, `SIGNAL_B` (look like ordinary event codes) |
| Probability | 0.5 each, independent |
| Max per patient | one A, one B |
| Time rule | uniform among the patient’s distinct **pre-index** event timestamps |
| Seed | `20260101` |
| `source` column | `synthea` vs `synthetic_signal` |

Occurrence does **not** depend on age. Age at occurrence is recorded and used only in S2.

### Labels (same cohort, three tasks)

Logit noise: `ε ~ N(0, 0.25)`, seed `20260102`. Then `y ~ Bernoulli(sigmoid(score))`.

**S0 — event signal, no age (negative control)**

```
score = intercept + 1.0·I(A) + (−1.0)·I(B) + ε
```

**S1 — age main effect**

```
score = intercept + 1.0·I(A) + (−1.0)·I(B) + 0.6·z(age_at_index) + ε
```

`z` is cohort mean/std of age at index (saved in processed config). Event effects do not change with age.

**S2 — age × event interaction**

```
g(a) = sigmoid((a − 8.5) / 2.5)     # a = age when that signal occurred
score = intercept
        + 1.0 · I(A) · (1 − 2g(age_A))
        − 1.0 · I(B) · (1 − 2g(age_B))
        + ε
```

A raises risk at young occurrence ages and reverses after ~8.5y; B is the opposite. Current age is **not** in this equation.

Only **intercepts** were calibrated (binary search) to target prevalence **0.22**:

| Task | intercept |
|---|---|
| S0 | −1.416 |
| S1 | −1.499 |
| S2 | −1.389 |

### Splits

- 70% train / 15% val / 15% test
- Patient-level, one split shared by S0/S1/S2
- Stratified by developmental age group
- Seed `42`

---

## Generated data features

Full run: **10,000 patients**, **3,285,207 events** (3,275,125 Synthea + 10,082 signals).

### Patient table (`output/full/processed/patients.{csv,parquet}`)

| Column | Meaning |
|---|---|
| `patient_id` | Synthea UUID |
| `date_of_birth` | DOB |
| `index_date` | 2026-01-01 |
| `age_at_index` | years |
| `developmental_age_group` | `<1` / `1-5` / `6-11` / `12-17` |
| `generation_stratum` | which Synthea run produced the patient |
| `split` | train / val / test |
| `has_SIGNAL_A`, `has_SIGNAL_B` | injection flags (not model inputs) |
| `age_at_SIGNAL_A`, `age_at_SIGNAL_B` | years, NaN if absent |
| `y_S0`, `y_S1`, `y_S2` | binary labels |
| `p_S0`, `p_S1`, `p_S2` | generating probabilities (not model inputs) |

### Event table (`output/full/processed/events.{csv,parquet}`)

| Column | Meaning |
|---|---|
| `patient_id` | |
| `event_timestamp` | |
| `age_at_event` | years since birth |
| `time_before_index_days` | always **&gt; 0** |
| `event_code` | `ENC_*` `COND_*` `MED_*` `OBS_*` `PROC_*` `IMM_*` or `SIGNAL_A`/`SIGNAL_B` |
| `event_type` | encounter, condition, medication, observation, procedure, immunization, synthetic_signal |
| `source` | `synthea` or `synthetic_signal` |

### Realized coverage (full)

| Group | n | events/patient (median) | history span (median years) |
|---|---:|---|---|
| `<1` | 2,493 | 78 | 0.44 |
| 1–5 | 2,508 | 230 | 3.43 |
| 6–11 | 2,503 | 390 | 9.01 |
| 12–17 | 2,496 | 584 | 14.62 |

**2,500/2,500 adolescents** have at least one event at age &lt; 1 year (full history retained).

SIGNAL_A/B rates ≈ 50% in every age group. A and B are independent (joint cells ≈ 0.25). Median SIGNAL_A is ~758 days before index.

Label prevalence: S0 21.9%, S1 22.1%, S2 22.3%. S0 is flat across age; S1 rises with current age (0.12 → 0.35); S2 is not solved by age (AUROC 0.49) or by A−B alone (0.60).

Splits: 6,999 / 1,498 / 1,503, disjoint, all age groups in every split.

---

## What this is for

Four future model configurations (not implemented here):

1. No Age  
2. Late Age  
3. Additive Age  
4. Conditioned Age  

on three tasks:

| Label | Intended mechanism |
|---|---|
| S0 | clinical events matter; age should not |
| S1 | age shifts baseline risk; event meaning is constant |
| S2 | the same event means different things at different developmental ages |

S2 uses **age when the signal occurred**, not age at prediction. Empirically, an adolescent’s `SIGNAL_A` often happened in childhood, so tables by *current* age show attenuation rather than a full crossover. The ground-truth curves vs *occurrence* age do cross at 8.5 years (`age_benchmark_s2_curves.png`).

---

## What was deliberately not done

- No custom Synthea disease modules
- No extra synthetic diseases or high-dimensional latents
- No tuning of β, interaction strength, or signal rate to make a model win
- No overwrite of the earlier 0–25 custom-module cohort
- No Transformer training
