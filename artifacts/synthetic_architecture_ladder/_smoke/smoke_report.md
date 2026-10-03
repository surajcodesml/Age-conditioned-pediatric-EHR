# Synthetic architecture ladder

Metrics and figures were computed from saved predictions and mechanism arrays.
Models were not rerun for the tables or plots.

## Protocol

- Dataset: existing controlled Synthea scenarios S0–S3, patient split seed 20260922.
- Seeds, fixed before fitting: 0, 1, 2, 3, 4. This is the existing `cehrbert_small` seed list. The current DTR reference itself was fit only at seed 0.
- Training budget matches `dtr_age_temporal_new`: AdamW, lr 3e-4, weight decay 0.01, batch 32, 25 epochs, patience 5, minimum 12 epochs, gradient clip 1, d_model 64, dropout 0.
- E02 is the only run that changes the optimizer.
- E06 is an extension. The S2 oracle was generated with current-age λ(a), not an integrated hazard, so E06 is not expected to beat E01 on oracle recovery by construction.
- Surface RMSE, CF-RMSE_age, and CF-RMSE_lag use `baselines.common.counterfactual` on the saved grids.
- β=0 ΔBCE and age-shuffle ΔBCE are test-set BCE(counterfactual) − BCE(original). The age shuffle uses NumPy seed 0, the same seed as the existing DTR ablation.
- For mixture models, λ(a) compared with the oracle is the unweighted mean of λ_k(a). That reduction was fixed in the metric code before the runs.
- Seed-level intervals in the summary CSV are normal approximations mean ± 1.96·sd/√n. Each run's `metrics.json` also stores a 200-draw patient bootstrap of the BCE deltas.

Artifact root: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke`

## E01_direct

Hypothesis: Retrieval and readout complexity in the current DTR is the main bottleneck for explicit age×lag recovery.

Single change: Remove content query, exp(u), content-dependent persistence, and the nonlinear history MLP. Direct linear readout. Weak lambda initialization.

Parameters added: none beyond theta0, beta, linear W_history, linear W_age, and bias. Parameters removed: content query/key, exp(u), persistence projection r·v + b_r, history MLP.

Training: Same trainer as the current DTR baseline: one AdamW group, learning rate 3e-4, weight decay 1e-2, 25 epochs, patience 5, minimum 12 epochs, batch size 32, gradient clip 1.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/E01_direct`

- S0 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S0 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S2 age_temporal: BCE 0.7637 (n=1); AUROC 0.5049 (n=1); AUPRC 0.2491 (n=1); Surface RMSE 0.3839 (n=1); CF-RMSE_age 0.3878 (n=1); CF-RMSE_lag 0.3653 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0008 (n=1); β 0.0006 (n=1); |β| 0.0006 (n=1); λ RMSE 1.1559 (n=1); λ correlation -0.9634 (n=1)
- S2 temporal_only: BCE 0.7637 (n=1); AUROC 0.5049 (n=1); AUPRC 0.2491 (n=1); Surface RMSE 0.3839 (n=1); CF-RMSE_age 0.3878 (n=1); CF-RMSE_lag 0.3653 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0008 (n=1); β 0.0000 (n=1); |β| 0.0000 (n=1); λ RMSE 1.1559 (n=1); λ correlation —
- S3 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S3 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E00 current DTR: Observed difference +0.1826 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_age versus E00 current DTR: Observed difference +0.2455 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_lag versus E00 current DTR: Observed difference +0.1997 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal bce versus E00 current DTR: Observed difference +0.3051 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auroc versus E00 current DTR: Observed difference -0.2557 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auprc versus E00 current DTR: Observed difference -0.3274 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_beta0 versus E00 current DTR: Observed difference -0.0069 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_age_shuffle versus E00 current DTR: Observed difference -0.0192 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S0 CF-RMSE_age versus E00 current DTR: Not compared; a value is missing.
- S0 |β| is —.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E02_staged

Hypothesis: A staged optimizer, with architecture held fixed at E01, improves mechanism recovery.

Single change: No architectural change relative to E01.

Parameters added: none. Parameters removed: none.

Training: Stage A trains beta=0. The checkpoint is cloned into both arms. Stage B freezes the encoder and readout and trains theta0/beta for 5 epochs at 10x learning rate and zero weight decay. Stage C unfreezes all parameters and fine-tunes with the same temporal parameter group.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/E02_staged`

- S0 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S0 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S2 age_temporal: BCE 0.7401 (n=1); AUROC 0.5031 (n=1); AUPRC 0.2495 (n=1); Surface RMSE 0.3709 (n=1); CF-RMSE_age 0.3738 (n=1); CF-RMSE_lag 0.3511 (n=1); β=0 ΔBCE 0.0001 (n=1); age-shuffle ΔBCE 0.0010 (n=1); β 0.0119 (n=1); |β| 0.0119 (n=1); λ RMSE 1.1555 (n=1); λ correlation -0.9626 (n=1)
- S2 temporal_only: BCE 0.7402 (n=1); AUROC 0.5032 (n=1); AUPRC 0.2496 (n=1); Surface RMSE 0.3709 (n=1); CF-RMSE_age 0.3738 (n=1); CF-RMSE_lag 0.3511 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0009 (n=1); β 0.0000 (n=1); |β| 0.0000 (n=1); λ RMSE 1.1551 (n=1); λ correlation —
- S3 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S3 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0130 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_age versus E01: Observed difference -0.0140 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference -0.0142 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal bce versus E01: Observed difference -0.0236 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auroc versus E01: Observed difference -0.0018 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auprc versus E01: Observed difference +0.0004 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference +0.0001 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0001 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S0 CF-RMSE_age versus E01: Not compared; a value is missing.
- S0 |β| is —.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E03_mass

Hypothesis: Separating history composition from surviving evidence mass improves stability and recovery.

Single change: Replace only the sum g·v aggregation by normalized composition plus log1p(evidence mass).

Parameters added: log1p(M) feature on the linear readout. Parameters removed: raw unnormalized sum as the sole history vector.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/E03_mass`

- S0 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S0 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S2 age_temporal: BCE 0.7212 (n=1); AUROC 0.5079 (n=1); AUPRC 0.2455 (n=1); Surface RMSE 0.3556 (n=1); CF-RMSE_age 0.3528 (n=1); CF-RMSE_lag 0.3317 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0020 (n=1); β 0.0006 (n=1); |β| 0.0006 (n=1); λ RMSE 1.1559 (n=1); λ correlation -0.9634 (n=1)
- S2 temporal_only: BCE 0.7212 (n=1); AUROC 0.5079 (n=1); AUPRC 0.2455 (n=1); Surface RMSE 0.3556 (n=1); CF-RMSE_age 0.3528 (n=1); CF-RMSE_lag 0.3317 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0020 (n=1); β 0.0000 (n=1); |β| 0.0000 (n=1); λ RMSE 1.1559 (n=1); λ correlation —
- S3 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S3 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0283 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_age versus E01: Observed difference -0.0350 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference -0.0336 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal bce versus E01: Observed difference -0.0425 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auroc versus E01: Observed difference +0.0029 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auprc versus E01: Observed difference -0.0036 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference -0.0000 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0011 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S0 CF-RMSE_age versus E01: Not compared; a value is missing.
- S0 |β| is —.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E04_channels

Hypothesis: Extra content channels help while the developmental decay stays a single shared lambda(a).

Single change: Add 4 content channels c=sigmoid(q_h^T v) with one shared lambda(a). Queries do not receive age or lag.

Parameters added: 4 content query vectors. Parameters removed: none from the E01 mechanism.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/E04_channels`

- S0 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S0 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S2 age_temporal: BCE 0.6704 (n=1); AUROC 0.4906 (n=1); AUPRC 0.2479 (n=1); Surface RMSE 0.3560 (n=1); CF-RMSE_age 0.3476 (n=1); CF-RMSE_lag 0.3265 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0027 (n=1); β 0.0006 (n=1); |β| 0.0006 (n=1); λ RMSE 1.1560 (n=1); λ correlation -0.9634 (n=1)
- S2 temporal_only: BCE 0.6704 (n=1); AUROC 0.4906 (n=1); AUPRC 0.2479 (n=1); Surface RMSE 0.3560 (n=1); CF-RMSE_age 0.3476 (n=1); CF-RMSE_lag 0.3265 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0027 (n=1); β 0.0000 (n=1); |β| 0.0000 (n=1); λ RMSE 1.1559 (n=1); λ correlation —
- S3 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S3 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0279 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_age versus E01: Observed difference -0.0402 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference -0.0388 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal bce versus E01: Observed difference -0.0933 (favorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auroc versus E01: Observed difference -0.0144 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auprc versus E01: Observed difference -0.0012 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference -0.0000 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0019 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S0 CF-RMSE_age versus E01: Not compared; a value is missing.
- S0 |β| is —.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E05_mixture

Hypothesis: A small content-dependent mixture of timescales recovers the age×lag surface better than one global decay.

Single change: K=3 content-only mixture of developmental rates. First model with content-specific timescales.

Parameters added: linear content mixture, theta_k, beta_k. Parameters removed: single global theta0/beta.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/E05_mixture`

- S0 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S0 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S2 age_temporal: BCE 0.7663 (n=1); AUROC 0.4990 (n=1); AUPRC 0.2404 (n=1); Surface RMSE 0.4050 (n=1); CF-RMSE_age 0.3969 (n=1); CF-RMSE_lag 0.3791 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0092 (n=1); β —; |β| 0.0006 (n=1); λ RMSE 1.1559 (n=1); λ correlation -0.9634 (n=1)
- S2 temporal_only: BCE 0.7663 (n=1); AUROC 0.4991 (n=1); AUPRC 0.2404 (n=1); Surface RMSE 0.4050 (n=1); CF-RMSE_age 0.3969 (n=1); CF-RMSE_lag 0.3791 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0092 (n=1); β —; |β| 0.0000 (n=1); λ RMSE 1.1559 (n=1); λ correlation —
- S3 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S3 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference +0.0210 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_age versus E01: Observed difference +0.0091 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference +0.0138 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal bce versus E01: Observed difference +0.0025 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auroc versus E01: Observed difference -0.0059 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auprc versus E01: Observed difference -0.0087 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference -0.0000 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0084 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S0 CF-RMSE_age versus E01: Not compared; a value is missing.
- S0 |β| is —.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E06_integrated_hazard

Hypothesis: An integrated hazard from event age to prediction age is a separate developmental parameterization.

Single change: Replace tau-multiplied lambda(a) by the integral of a 4-knot positive piecewise-linear hazard from event age to prediction age. Temporal-only is a constant hazard.

Parameters added: knot coefficients of rho(a). Parameters removed: softplus lambda(a) times tau.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/E06_integrated_hazard`

- S0 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S0 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S1 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S2 age_temporal: BCE 0.8030 (n=1); AUROC 0.5057 (n=1); AUPRC 0.2497 (n=1); Surface RMSE 0.3985 (n=1); CF-RMSE_age 0.4084 (n=1); CF-RMSE_lag 0.3839 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0008 (n=1); β —; |β| 0.0006 (n=1); λ RMSE —; λ correlation —
- S2 temporal_only: BCE 0.8031 (n=1); AUROC 0.5057 (n=1); AUPRC 0.2497 (n=1); Surface RMSE 0.3985 (n=1); CF-RMSE_age 0.4084 (n=1); CF-RMSE_lag 0.3839 (n=1); β=0 ΔBCE 0.0000 (n=1); age-shuffle ΔBCE 0.0008 (n=1); β —; |β| 0.0000 (n=1); λ RMSE —; λ correlation —
- S3 age_temporal: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —
- S3 temporal_only: BCE —; AUROC —; AUPRC —; Surface RMSE —; CF-RMSE_age —; CF-RMSE_lag —; β=0 ΔBCE —; age-shuffle ΔBCE —; β —; |β| —; λ RMSE —; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference +0.0146 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_age versus E01: Observed difference +0.0207 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference +0.0186 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal bce versus E01: Observed difference +0.0393 (unfavorable if lower is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auroc versus E01: Observed difference +0.0008 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal auprc versus E01: Observed difference +0.0006 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference +0.0000 (favorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference -0.0000 (unfavorable if higher is better). Both sides are single-seed, so variability across seeds is not estimated.
- S0 CF-RMSE_age versus E01: Not compared; a value is missing.
- S0 |β| is —.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

E06-specific constraint: a worse S2 surface than E01 is not evidence against the integrated hazard as a model of a different mechanism. The oracle labels were not generated from that integral.

## Compact comparison

AUROC and AUPRC are S2 age-temporal micro averages. S0 false interaction is S0 age-temporal CF-RMSE_age. Cells are seed mean ± sample standard deviation.

| Experiment | Single change tested | S2 Surface RMSE | S2 β=0 ΔBCE | S2 shuffle ΔBCE | S0 false interaction | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E00_dtr_age_temporal_new | current DTR age_temporal (seed 0 reference) | 0.2013 (n=1) | 0.0069 (n=1) | 0.0201 (n=1) | 0.0986 (n=1) | 0.7606 (n=1) | 0.5765 (n=1) |
| E00_dtr_temporal_only_new | current DTR temporal_only (seed 0 reference) | 0.2361 (n=1) | — | — | 0.0841 (n=1) | 0.7577 (n=1) | 0.5690 (n=1) |
| E00_legacy_dtr_age_temporal | legacy unsuffixed dtr_age_temporal | 0.2266 (n=1) | — | — | 0.0769 (n=1) | 0.7650 (n=1) | 0.5835 (n=1) |
| E00_legacy_dtr_temporal_only | legacy unsuffixed dtr_temporal_only | 0.2251 (n=1) | — | — | 0.0740 (n=1) | 0.7642 (n=1) | 0.5825 (n=1) |
| CEHR-BERT | published CEHR-BERT, all 32 targets | 0.1308 (n=1) | — | — | 0.0597 (n=1) | 0.7636 (n=1) | 0.5818 (n=1) |
| CEHR-BERT-small | published CEHR-BERT-small, S2 seeds 0–4 | 0.1645 ± 0.0117 | — | — | — | 0.7390 ± 0.0088 | 0.5410 ± 0.0134 |
| E01_direct | Remove content query, exp(u), content-dependent persistence, and the nonlinear history MLP. Direct linear readout. Weak lambda initialization. | 0.3839 (n=1) | 0.0000 (n=1) | 0.0008 (n=1) | — | 0.5049 (n=1) | 0.2491 (n=1) |
| E02_staged | No architectural change relative to E01. | 0.3709 (n=1) | 0.0001 (n=1) | 0.0010 (n=1) | — | 0.5031 (n=1) | 0.2495 (n=1) |
| E03_mass | Replace only the sum g·v aggregation by normalized composition plus log1p(evidence mass). | 0.3556 (n=1) | 0.0000 (n=1) | 0.0020 (n=1) | — | 0.5079 (n=1) | 0.2455 (n=1) |
| E04_channels | Add 4 content channels c=sigmoid(q_h^T v) with one shared lambda(a). Queries do not receive age or lag. | 0.3560 (n=1) | 0.0000 (n=1) | 0.0027 (n=1) | — | 0.4906 (n=1) | 0.2479 (n=1) |
| E05_mixture | K=3 content-only mixture of developmental rates. First model with content-specific timescales. | 0.4050 (n=1) | 0.0000 (n=1) | 0.0092 (n=1) | — | 0.4990 (n=1) | 0.2404 (n=1) |
| E06_integrated_hazard | Replace tau-multiplied lambda(a) by the integral of a 4-knot positive piecewise-linear hazard from event age to prediction age. Temporal-only is a constant hazard. | 0.3985 (n=1) | 0.0000 (n=1) | 0.0008 (n=1) | — | 0.5057 (n=1) | 0.2497 (n=1) |

## Figure paths

- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/figures/heatmap_S2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/figures/heatmap_S3.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/figures/lambda_age.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/figures/e06_rho_age.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/figures/mechanism_metrics_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/figures/predictive_metrics_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/figures/negative_controls_s0_s1.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/architecture_ladder_metrics.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/architecture_ladder_summary.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/_smoke/architecture_ladder_summary.json`

