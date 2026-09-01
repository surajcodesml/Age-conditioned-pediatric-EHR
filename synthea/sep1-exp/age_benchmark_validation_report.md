# Age-signal pediatric benchmark — validation report

- mode: `full`
- reference date: `20260101`
- geography: `Massachusetts`
- patients: **10,000**
- all acceptance criteria passed: **True**

## Files

- `patients_parquet`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/patients.parquet`
- `patients_csv`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/patients.csv`
- `events_parquet`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/events.parquet`
- `events_csv`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/events.csv`
- `config`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/benchmark_config.json`
- `s2_curves_csv`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/s2_ground_truth_curves.csv`
- `s2_curves_png`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/s2_ground_truth_curves.png`
- `report`: `/home/suraj/Git/Age-conditioned-pediatric-EHR/synthea/sep1-exp/output/full/processed/validation_report.md`

## A. Age coverage

| group | n |
|---|---:|
| <1 | 2,493 |
| 1-5 | 2,508 |
| 6-11 | 2,503 |
| 12-17 | 2,496 |

## B. Longitudinal history

- events per patient: 311.50 (IQR 119.75–486.00)
- history duration (years, max−min event age): 5.98 (IQR 1.00–11.87)

| group | events/patient | duration (years) |
|---|---|---|
| <1 | 78.00 (IQR 46.00–97.00) | 0.44 (IQR 0.10–0.69) |
| 1-5 | 230.00 (IQR 177.00–272.00) | 3.43 (IQR 2.00–4.22) |
| 6-11 | 390.00 (IQR 347.00–438.00) | 9.01 (IQR 7.00–10.04) |
| 12-17 | 583.50 (IQR 526.00–659.00) | 14.62 (IQR 13.11–16.14) |

### Complete-history check (adolescents)

- adolescents: 2500
- with an event at age < 1y: 2500 (1.000)
- with an event at age < 6y: 2500
- adolescent event-age min/max: 0.000 / 17.944
- median event-age span (years): 14.604
- history_complete: True

## C. SIGNAL_A / SIGNAL_B rates

- overall A: 0.506
- overall B: 0.502

| group | SIGNAL_A | SIGNAL_B |
|---|---:|---:|
| <1 | 0.493 | 0.502 |
| 1-5 | 0.516 | 0.490 |
| 6-11 | 0.509 | 0.517 |
| 12-17 | 0.505 | 0.500 |

## D. Label prevalence

- intercepts: S0=-1.4157, S1=-1.4989, S2=-1.3892
- target prevalence: 0.22

| | overall | <1 | 1–5 | 6–11 | 12–17 |
|---|---:|---:|---:|---:|---:|
| S0 | 0.219 | 0.208 | 0.227 | 0.216 | 0.225 |
| S1 | 0.221 | 0.122 | 0.172 | 0.244 | 0.347 |
| S2 | 0.223 | 0.231 | 0.238 | 0.211 | 0.214 |

## E. H0 / H1 / H2 sanity

AUROC of simple scores vs labels (not a trained model):

| task | age only | A only | B only | A−B |
|---|---:|---:|---:|---:|
| S0 | 0.506 | 0.615 | 0.389 | 0.670 |
| S1 | 0.644 | 0.613 | 0.397 | 0.665 |
| S2 | 0.486 | 0.564 | 0.433 | 0.600 |

- S0 prevalence spread across age groups: 0.019
- S0 max within-(A,B) cell age-group prevalence range: 0.039

S1 prevalence by age group: 0.122, 0.172, 0.244, 0.347

### Event risk differences Δ = P(y=1|signal) − P(y=1|no signal)

**S0**

| group | ΔA | ΔB |
|---|---:|---:|
| <1 | +0.168 | -0.162 |
| 1-5 | +0.159 | -0.163 |
| 6-11 | +0.139 | -0.133 |
| 12-17 | +0.166 | -0.145 |

**S1**

| group | ΔA | ΔB |
|---|---:|---:|
| <1 | +0.095 | -0.085 |
| 1-5 | +0.139 | -0.127 |
| 6-11 | +0.177 | -0.153 |
| 12-17 | +0.190 | -0.223 |

**S2**

| group | ΔA | ΔB |
|---|---:|---:|
| <1 | +0.135 | -0.160 |
| 1-5 | +0.107 | -0.149 |
| 6-11 | +0.078 | -0.104 |
| 12-17 | +0.027 | +0.033 |

## F. Leakage

- events on/after index: 0
- train/val overlap: 0
- train/test overlap: 0
- val/test overlap: 0
- synthetic codes: ['SIGNAL_A', 'SIGNAL_B']
- label-like event codes: []

Split counts: {'train': 6999, 'val': 1498, 'test': 1503}

## Ground-truth S2 curves

`g(a) = sigmoid((a − 8.5) / 2.5)`; effect_A = +(1−2g), effect_B = −(1−2g).
See `s2_ground_truth_curves.csv` / `.png`.

## Acceptance criteria

| criterion | pass |
|---|---|
| balanced_age_groups | True |
| complete_history | True |
| signal_independent_of_age | True |
| S0_no_age_effect | True |
| S1_age_main_effect | True |
| S2_interaction | True |
| S2_not_age_only | True |
| S2_not_AB_only | True |
| labels_stochastic | True |
| prevalence_ok | True |
| splits_leak_free | True |

## Regeneration

```bash
bash synthea/sep1-exp/generate_age_benchmark.sh full
conda run -n ehr python synthea/sep1-exp/build_age_benchmark.py --mode full
```

Coefficients (not searched): {"beta_A": 1.0, "beta_B": -1.0, "beta_age": 0.6, "interaction_strength": 1.0, "g_center_years": 8.5, "g_scale_years": 2.5}

