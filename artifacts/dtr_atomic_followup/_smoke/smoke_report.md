# Atomic DTR follow-up

Comparisons are paired by seed against C00. Tables were computed from saved predictions and mechanism arrays.

## Protocol

- C00 retrains the current Content-Persistence DTR on seeds 0–4 with matched initialization and the same data order in both arms.
- Accept/reject decisions use C00, not the earlier single-seed dtr_age_temporal_new result.
- Training budget matches that reference: AdamW, lr 3e-4, weight decay 0.01, batch 32, 25 epochs, patience 5, minimum 12 epochs.
- C01 is the only optimizer change. Stage B trains theta0/beta for 5 epochs at 10× learning rate and zero weight decay, with the rest frozen.
- delta_bce_gate_age_shuffle shuffles age only inside the temporal gate. The additive age head keeps the true age.
- delta_bce_full_age_shuffle is the previous whole-model age shuffle.
- Gate recovery for content-dependent models is the forward gate on signal encounters versus the oracle gate at the same age and tau. Mixture models use g_eff = sum_k pi_k exp(-lambda_k tau), not the mean of lambda_k.
- C04 also stores a global gate surface because its lambda does not depend on content.
- Matched-arm deltas are BCE(temporal_only) − BCE(age_temporal), and age_temporal minus temporal_only for AUROC and AUPRC.

Artifact root: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke`

## Paired differences versus C00

Age-temporal arm unless the metric is a matched-arm delta. Cells are mean ± standard deviation of seed-paired (candidate − C00).

| Experiment | S2 surface RMSE | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 β=0 ΔBCE | S2 BCE | S2 AUPRC | S2 matched BCE TO−AT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01_staged_current | -0.0026 (n=1) | -0.0003 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | -0.0031 (n=1) | +0.0002 (n=1) | +0.0000 (n=1) |
| C02_mass_current | -0.0060 (n=1) | +0.0000 (n=1) | -0.0000 (n=1) | -0.0000 (n=1) | +0.0015 (n=1) | +0.0008 (n=1) | -0.0000 (n=1) |
| C03_weak_lambda_init | +0.0035 (n=1) | +0.1872 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | +0.0028 (n=1) | -0.0002 (n=1) | +0.0000 (n=1) |
| C04_no_content_persistence | +0.0000 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | -0.0000 (n=1) | +0.0000 (n=1) |
| C05_shared_beta_mixture | +0.0000 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | -0.0000 (n=1) | +0.0000 (n=1) |
| C06_component_beta_mixture | +0.0000 (n=1) | +0.0000 (n=1) | -0.0000 (n=1) | +0.0000 (n=1) | +0.0000 (n=1) | -0.0000 (n=1) | +0.0000 (n=1) |

## Levels on C00 and each candidate

| Experiment | S2 surface | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 β=0 ΔBCE | S0 gate-shuffle ΔBCE | S0 β=0 ΔBCE | S2 β mean | S3 β mean | S2 AUROC | S2 AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00_current_dtr | 0.3493 | 0.2898 | 0.0000 | 0.0000 | — | — | 0.0006 | — | 0.4901 | 0.2414 |
| C01_staged_current | 0.3467 | 0.2895 | 0.0000 | 0.0000 | — | — | 0.0116 | — | 0.4903 | 0.2416 |
| C02_mass_current | 0.3433 | 0.2898 | 0.0000 | 0.0000 | — | — | 0.0006 | — | 0.4902 | 0.2422 |
| C03_weak_lambda_init | 0.3528 | 0.4770 | 0.0000 | 0.0000 | — | — | 0.0006 | — | 0.4932 | 0.2412 |
| C04_no_content_persistence | 0.3493 | 0.2898 | 0.0000 | 0.0000 | — | — | 0.0006 | — | 0.4901 | 0.2414 |
| C05_shared_beta_mixture | 0.3493 | 0.2898 | 0.0000 | 0.0000 | — | — | 0.0006 | — | 0.4901 | 0.2414 |
| C06_component_beta_mixture | 0.3493 | 0.2898 | 0.0000 | 0.0000 | — | — | 0.0006 | — | 0.4901 | 0.2414 |

## C01_staged_current

Hypothesis: Staged optimization improves the current architecture.

Single change: No architectural change. Stage A trains beta=0, then both arms clone that checkpoint. Stage B trains theta0/beta with the encoder and readout frozen. Stage C jointly fine-tunes.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/C01_staged_current`

- S2 age_temporal surface_rmse: -0.0026 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: -0.0003 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: -0.0018 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_lag: -0.0022 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: -0.0031 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: +0.0002 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: +0.0002 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0000 (n=1). descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: —. missing.
- S0 age_temporal delta_bce_beta0: —. missing.
- S0 age_temporal gate_signal_rmse: —. missing.
- S3 age_temporal surface_rmse: —. missing.
- S3 age_temporal gate_signal_rmse: —. missing.
- Observed S2 β mean 0.0116; S3 β mean —. S3 should reverse the S2 developmental sign.

Supported interpretation: at least one S2/S3 gate or prediction-surface comparison improves beyond seed noise, and S2 BCE/AUPRC do not worsen beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C02_mass_current

Hypothesis: Keeping evidence mass beside the normalized history improves the current DTR.

Single change: Replace h=sum w v by concat(sum(w v)/(M+eps), log1p(M)). The history MLP stays; only its first input dimension grows by one.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/C02_mass_current`

- S2 age_temporal surface_rmse: -0.0060 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: +0.0005 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_lag: -0.0035 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0015 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: +0.0001 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: +0.0008 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: -0.0000 (n=1). descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: —. missing.
- S0 age_temporal delta_bce_beta0: —. missing.
- S0 age_temporal gate_signal_rmse: —. missing.
- S3 age_temporal surface_rmse: —. missing.
- S3 age_temporal gate_signal_rmse: —. missing.
- Observed S2 β mean 0.0006; S3 β mean —. S3 should reverse the S2 developmental sign.

Supported interpretation: a recovery metric moves beyond seed noise, and S2 predictive BCE or AUPRC also worsens beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C03_weak_lambda_init

Hypothesis: Initializing effective lambda at 0.1 instead of softplus(0) lets long-range evidence survive early training.

Single change: Set theta0 so softplus(theta0) = 0.1. Persistence intercept stays 0, so the initial effective lambda is 0.1.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/C03_weak_lambda_init`

- S2 age_temporal surface_rmse: +0.0035 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: +0.1872 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: +0.0112 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_lag: +0.0098 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0028 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: +0.0030 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0002 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0000 (n=1). descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: —. missing.
- S0 age_temporal delta_bce_beta0: —. missing.
- S0 age_temporal gate_signal_rmse: —. missing.
- S3 age_temporal surface_rmse: —. missing.
- S3 age_temporal gate_signal_rmse: —. missing.
- Observed S2 β mean 0.0006; S3 β mean —. S3 should reverse the S2 developmental sign.

Supported interpretation: no S2/S3 gate or prediction-surface gain is larger than the paired seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C04_no_content_persistence

Hypothesis: Removing r·v + b_r lets the developmental slope be the only persistence mechanism.

Single change: Drop the content persistence offset. Keep content query, exp(u), raw aggregation, history MLP, and the current optimizer.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/C04_no_content_persistence`

- S2 age_temporal surface_rmse: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_lag: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0000 (n=1). descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: —. missing.
- S0 age_temporal delta_bce_beta0: —. missing.
- S0 age_temporal gate_signal_rmse: —. missing.
- S3 age_temporal surface_rmse: —. missing.
- S3 age_temporal gate_signal_rmse: —. missing.
- Observed S2 β mean 0.0006; S3 β mean —. S3 should reverse the S2 developmental sign.

Supported interpretation: no S2/S3 gate or prediction-surface gain is larger than the paired seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C05_shared_beta_mixture

Hypothesis: A few content-specific baseline timescales help when they share one developmental slope.

Single change: Replace r·v + b_r with K=3 mixture weights from content only and lambda_k=softplus(theta_k + beta z). beta is one shared scalar.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/C05_shared_beta_mixture`

- S2 age_temporal surface_rmse: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_lag: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0000 (n=1). descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: —. missing.
- S0 age_temporal delta_bce_beta0: —. missing.
- S0 age_temporal gate_signal_rmse: —. missing.
- S3 age_temporal surface_rmse: —. missing.
- S3 age_temporal gate_signal_rmse: —. missing.
- Observed S2 β mean 0.0006; S3 β mean —. S3 should reverse the S2 developmental sign.

Supported interpretation: no S2/S3 gate or prediction-surface gain is larger than the paired seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C06_component_beta_mixture

Hypothesis: Separate developmental slopes across mixture components are required for the E05-style gain.

Single change: Identical to C05 except each component has its own beta_k.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/C06_component_beta_mixture`

- S2 age_temporal surface_rmse: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_lag: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0000 (n=1). favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0000 (n=1). unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0000 (n=1). descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: —. missing.
- S0 age_temporal delta_bce_beta0: —. missing.
- S0 age_temporal gate_signal_rmse: —. missing.
- S3 age_temporal surface_rmse: —. missing.
- S3 age_temporal gate_signal_rmse: —. missing.
- Observed S2 β mean 0.0006; S3 β mean —. S3 should reverse the S2 developmental sign.

Supported interpretation: no S2/S3 gate or prediction-surface gain is larger than the paired seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## Changes with evidence for a later combination stage

This list is not a license to combine the changes yet. A name appears when a paired S2 or S3 surface or gate RMSE improvement exceeds seed noise and S2 BCE/AUPRC do not show a supported worsening.

- C01_staged_current

## Figure paths

- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/figures/prediction_surface_S2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/figures/prediction_surface_S3.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/figures/gate_signal_rmse_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/figures/lambda_curves.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/figures/negative_controls_s0.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/atomic_followup_metrics.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/atomic_followup_summary.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/atomic_followup_summary.json`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/_smoke/paired_delta_vs_C00.csv`

