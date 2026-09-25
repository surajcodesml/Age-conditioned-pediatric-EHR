# Synthetic benchmark facts

## Dataset

- Synthea version `master-branch-latest`, commit `aa0772fb5e92e48a776c51508c00eddc0d9d27ff`, Massachusetts, reference date 20260101.
- Source cohort: 10000 patients, age 0.003 to 17.993 years, 3275125 Synthea-source background events.
- Controlled cohort used for every run in this package: 7007 patients with age at cutoff at least 2 years.
- Full-realism cohort size is 10000. It was not used for these final runs.
- Patient-disjoint split, seed 20260922, fractions 0.70 / 0.15 / 0.15: train 4906, validation 1070, test 1031.
- The same patient ids are used for S0, S1, S2, S3, and S5.
- Signal codes `SYN_SIGNAL_A` through `SYN_SIGNAL_L`.
- Signals per patient: integer from 4 through 8, lags in {7, 30, 90, 180, 365, 730} days with multiplicative jitter in [0.9, 1.1], kept inside the patient's lifetime and at most 730 days.
- Background history: the newest 64 Synthea events with timestamp strictly before the cutoff. Sequence models then use maximum length 96.
- Prediction cutoff: the patient index date. An event is historical only if its timestamp is before that cutoff.
- 32 targets: 8 interaction, 6 temporal-only, 6 age-only, 6 content-only, 6 null.

## Ground-truth mechanism

\[
z(a)=\frac{a-9}{9}
\]

\[
\tau=\log(1+\Delta t/7)
\]

\[
\lambda=\mathrm{softplus}(\theta+\beta z(a)),\qquad R=\exp(-\lambda\tau)
\]

Interaction and temporal targets enter the logit through a scaled weighted sum of event relevance (scale 2.5). Age-only targets use an age main effect and ignore lag. Null targets ignore history. Label noise standard deviation is 0.35. Target prevalence is calibrated to 0.25 on the training split.

## Scenarios

### S0
`beta_true = 0`, `theta0 = 0`, `has_interaction = false`. Interaction targets use the temporal-only path with fixed `lambda = 0.8`. Age does not modify decay. Age-only targets still use coefficient 1.0 times their own gamma, because `age_main_effect` is 0 and the generator then falls back to 1.0 for age-only targets.

### S1
`beta_true = 0`, `has_interaction = false`, `age_main_effect = 1.2`. Temporal relevance stays age-independent (`lambda = 0.8`). The age main effect is `1.2 * gamma * z(a)` on interaction-path and age-only targets.

### S2
`beta_true = -2.5`, `theta0 = 0`, `has_interaction = true`, `age_main_effect = 0`. Developmental interaction: younger age gives faster decay.

### S3
`beta_true = +2.5`, `theta0 = 0`, `has_interaction = true`. Reversed interaction: older age gives faster decay.

### S5
Same developmental beta as S2 (`beta_true = -2.5`) with signal-group baseline persistence on interaction targets:

- acute `SYN_SIGNAL_A`–`D`: `theta = 1`
- intermediate `SYN_SIGNAL_E`–`H`: `theta = 0`
- chronic `SYN_SIGNAL_I`–`L`: `theta = -1`

`lambda` for an interaction-target event is `softplus(theta_group + beta * z(a))`. Temporal-only targets still use fixed `lambda = 0.8`.

Saved `meta.json` beta values match these constants for S0–S3 and S5.
