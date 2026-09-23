# Figure results summary

## Checkpoints (validation micro-AUPRC selection)

| Arm | Path | Epoch | Val micro-AUPRC | λ₀ | β | age μ/σ |
|---|---|---:|---:|---:|---:|---|
| MIMIC ADKM | `stage1_mimic_pretrain/run/adkm_s0/epoch_005.pt` | 5 | (from history.json: 0.4665) | -1.4984 | -0.1802 | 63.336/16.575 |
| MIMIC NINT | `stage1_mimic_pretrain/run/nint_s0/epoch_005.pt` | 5 | (from history.json: 0.4708) | -1.4484 | 0.0000 | 63.336/16.575 |
| NCH ADKM | `stage2_nch/run/adkm_nch_s0/checkpoint_best_auprc.pt` | 4 | 0.17091525796592097 | -1.0229 | 0.0924 | 9/9 |
| NCH NINT | `stage2_nch/run/nint_nch_s0/checkpoint_best_auprc.pt` | 4 | 0.16619003801360127 | -1.0702 | 0.0000 | 9/9 |

## Quantities plotted

### `mimic_lambda_age`
- **y:** λ(a) = λ₀ + β z(a) from `AgeTemporalBias.lambda_of`.
- **z(a):** (a − μ)/σ with frozen MIMIC train event-level μ, σ.
- **x:** age in years over observed MIMIC event-age range.
- **Background:** density histogram of event ages sampled from MIMIC train NPZ shards.

### `nch_age_lag_heatmap`
- **Cell:** K(a, Δt) = −λ(a)·τ(Δt), τ = log1p(|Δt|/7) via `lag_to_tau`, λ via `lambda_of`
  (verified equal to `pairwise_bias` on a probe).
- Ages 0–18 y; lags log-spaced 1 d → 10 y.
- Panels: ADKM, NINT (shared color scale), ADKM−NINT.

### `nch_subgroup_performance`
- Held-out NCH test set; epoch-4 ADKM/NINT `checkpoint_best_auprc.pt`.
- Metric: **micro-AUPRC** (project `_safe_auprc`), on codes with ≥1 positive in the test set.
- Strata: developmental age `<1`, `1–5`, `6–11`, `12–17`; history `<3 months`, `3–12 months`, `1–3 years`, `>3 years` (by available lookback span).
- Patient-level bootstrap 95% CI (n=100, seed=0); paired Δ = ADKM − NINT.
- Subgroups with <20 patients are flagged and CI omitted.

## Subgroup sample sizes

See `nch_subgroup_performance.csv`.

## Outputs

- `/home/suraj/Git/Age-conditioned-pediatric-EHR/figures/results/mimic_lambda_age.png` / `.svg`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/figures/results/nch_age_lag_heatmap.png` / `.svg`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/figures/results/nch_subgroup_performance.png` / `.svg`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/figures/results/nch_subgroup_performance.csv` / `.json`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/figures/results/raw/` plotting data and metadata

## Assumptions / exclusions

- No retraining; preprocessing unchanged.
- MIMIC age histogram is a random sample of train shards (not the full 405M events).
- Heatmap lag axis is log10-scaled for display; values use the model τ on raw days.
- Micro-AUPRC bootstrap uses the active test-set code subset for tractability; never-positive codes in the full test set are omitted (cannot contribute TPs).

### Numerical results

```
stratum_type     stratum  adkm_micro_auprc  adkm_ci_lo  adkm_ci_hi  nint_micro_auprc  nint_ci_lo  nint_ci_hi     delta  delta_ci_lo  delta_ci_hi  n_windows  n_patients  flagged_small
         age          <1          0.155701    0.137418    0.178116          0.153993    0.134205    0.177783  0.001707    -0.001756     0.005886       3706         264          False
         age         1-5          0.206172    0.168588    0.232969          0.204390    0.168391    0.232960  0.001782    -0.000967     0.005286      14038         407          False
         age        6-11          0.148744    0.122243    0.185808          0.149060    0.122561    0.185822 -0.000316    -0.004688     0.004442       7283         283          False
         age       12-17          0.105117    0.077667    0.137113          0.105083    0.077691    0.139616  0.000035    -0.002975     0.005106       3120         128          False
     history   <3 months          0.118523    0.101685    0.135768          0.116103    0.099815    0.136112  0.002420    -0.002647     0.007340       1974         505          False
     history 3–12 months          0.179238    0.157117    0.197225          0.177420    0.157873    0.198169  0.001818    -0.005152     0.007331       4120         354          False
     history   1–3 years          0.204815    0.165696    0.244766          0.202153    0.163573    0.243358  0.002661    -0.000038     0.006024       9336         373          False
     history    >3 years          0.157198    0.133109    0.188644          0.158084    0.135305    0.189349 -0.000886    -0.004015     0.003153      12717         330          False
```

