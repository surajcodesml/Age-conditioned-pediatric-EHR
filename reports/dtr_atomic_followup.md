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

Artifact root: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup`

## Paired differences versus C00

Age-temporal arm unless the metric is a matched-arm delta. Cells are mean ± standard deviation of seed-paired (candidate − C00).

| Experiment | S2 surface RMSE | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 β=0 ΔBCE | S2 BCE | S2 AUPRC | S2 matched BCE TO−AT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01_staged_current | -0.0221 ± 0.0081 | -0.0436 ± 0.0028 | +0.0128 ± 0.0013 | +0.0055 ± 0.0010 | -0.0030 ± 0.0006 | +0.0055 ± 0.0014 | -0.0001 ± 0.0010 |
| C02_mass_current | -0.0162 ± 0.0126 | +0.0248 ± 0.0296 | -0.0047 ± 0.0016 | -0.0031 ± 0.0011 | +0.0047 ± 0.0019 | -0.0102 ± 0.0033 | -0.0020 ± 0.0003 |
| C03_weak_lambda_init | -0.0045 ± 0.0169 | -0.0078 ± 0.0010 | +0.0019 ± 0.0007 | +0.0007 ± 0.0004 | -0.0007 ± 0.0005 | +0.0015 ± 0.0006 | +0.0000 ± 0.0003 |
| C04_no_content_persistence | +0.0242 ± 0.0144 | -0.0033 ± 0.0023 | +0.0032 ± 0.0011 | +0.0021 ± 0.0006 | +0.0011 ± 0.0006 | -0.0026 ± 0.0012 | +0.0008 ± 0.0004 |
| C05_shared_beta_mixture | +0.0242 ± 0.0144 | -0.0033 ± 0.0023 | +0.0032 ± 0.0011 | +0.0021 ± 0.0006 | +0.0011 ± 0.0006 | -0.0026 ± 0.0012 | +0.0008 ± 0.0004 |
| C06_component_beta_mixture | +0.0242 ± 0.0144 | -0.0033 ± 0.0023 | +0.0032 ± 0.0011 | +0.0021 ± 0.0006 | +0.0011 ± 0.0006 | -0.0026 ± 0.0012 | +0.0008 ± 0.0004 |

## Levels on C00 and each candidate

| Experiment | S2 surface | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 β=0 ΔBCE | S0 gate-shuffle ΔBCE | S0 β=0 ΔBCE | S2 β mean | S3 β mean | S2 AUROC | S2 AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00_current_dtr | 0.1968 ± 0.0137 | 0.1929 ± 0.0062 | 0.0103 ± 0.0020 | 0.0070 ± 0.0017 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | -0.5840 ± 0.0545 | 0.5970 ± 0.0329 | 0.7589 ± 0.0013 | 0.5762 ± 0.0017 |
| C01_staged_current | 0.1747 ± 0.0110 | 0.1493 ± 0.0055 | 0.0230 ± 0.0032 | 0.0124 ± 0.0019 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | -1.0059 ± 0.0433 | 0.8860 ± 0.0515 | 0.7613 ± 0.0010 | 0.5817 ± 0.0013 |
| C02_mass_current | 0.1806 ± 0.0120 | 0.2177 ± 0.0342 | 0.0056 ± 0.0023 | 0.0039 ± 0.0015 | -0.0000 ± 0.0000 | -0.0000 ± 0.0000 | -0.6356 ± 0.1133 | 0.6127 ± 0.0345 | 0.7558 ± 0.0024 | 0.5660 ± 0.0042 |
| C03_weak_lambda_init | 0.1923 ± 0.0074 | 0.1851 ± 0.0064 | 0.0122 ± 0.0024 | 0.0077 ± 0.0016 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | -0.6693 ± 0.0568 | 0.6830 ± 0.0239 | 0.7594 ± 0.0010 | 0.5777 ± 0.0012 |
| C04_no_content_persistence | 0.2211 ± 0.0281 | 0.1897 ± 0.0081 | 0.0134 ± 0.0031 | 0.0090 ± 0.0022 | 0.0001 ± 0.0001 | 0.0000 ± 0.0000 | -0.5676 ± 0.0676 | 0.5489 ± 0.0572 | 0.7579 ± 0.0012 | 0.5736 ± 0.0017 |
| C05_shared_beta_mixture | 0.2211 ± 0.0281 | 0.1897 ± 0.0081 | 0.0134 ± 0.0031 | 0.0090 ± 0.0022 | 0.0001 ± 0.0001 | 0.0000 ± 0.0000 | -0.5676 ± 0.0676 | 0.5489 ± 0.0572 | 0.7579 ± 0.0012 | 0.5736 ± 0.0017 |
| C06_component_beta_mixture | 0.2211 ± 0.0281 | 0.1897 ± 0.0081 | 0.0134 ± 0.0031 | 0.0090 ± 0.0022 | 0.0001 ± 0.0001 | 0.0000 ± 0.0000 | -0.5676 ± 0.0676 | 0.5489 ± 0.0572 | 0.7579 ± 0.0012 | 0.5736 ± 0.0017 |

## C01_staged_current

Hypothesis: Staged optimization improves the current architecture.

Single change: No architectural change. Stage A trains beta=0, then both arms clone that checkpoint. Stage B trains theta0/beta with the encoder and readout frozen. Stage C jointly fine-tunes.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/C01_staged_current`

- S2 age_temporal surface_rmse: -0.0221 ± 0.0081. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: -0.0436 ± 0.0028. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: -0.0114 ± 0.0117. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal cf_rmse_lag: -0.0099 ± 0.0087. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0128 ± 0.0013. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0055 ± 0.0010. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: -0.0030 ± 0.0006. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: +0.0023 ± 0.0010. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: +0.0055 ± 0.0014. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: -0.0001 ± 0.0010. descriptive; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal delta_bce_gate_age_shuffle: -0.0000 ± 0.0000. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal delta_bce_beta0: -0.0000 ± 0.0000. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal gate_signal_rmse: -0.0226 ± 0.0188. favorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal surface_rmse: -0.0046 ± 0.0067. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S3 age_temporal gate_signal_rmse: -0.0246 ± 0.0114. favorable; |mean paired difference| exceeds the seed standard deviation.
- Observed S2 β mean -1.0059 ± 0.0433; S3 β mean 0.8860 ± 0.0515. S3 should reverse the S2 developmental sign.

Supported interpretation: at least one S2/S3 gate or prediction-surface comparison improves beyond seed noise, and S2 BCE/AUPRC do not worsen beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C02_mass_current

Hypothesis: Keeping evidence mass beside the normalized history improves the current DTR.

Single change: Replace h=sum w v by concat(sum(w v)/(M+eps), log1p(M)). The history MLP stays; only its first input dimension grows by one.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/C02_mass_current`

- S2 age_temporal surface_rmse: -0.0162 ± 0.0126. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: +0.0248 ± 0.0296. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal cf_rmse_age: +0.0011 ± 0.0083. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal cf_rmse_lag: +0.0008 ± 0.0186. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal delta_bce_gate_age_shuffle: -0.0047 ± 0.0016. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: -0.0031 ± 0.0011. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0047 ± 0.0019. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: -0.0031 ± 0.0014. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0102 ± 0.0033. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: -0.0020 ± 0.0003. descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: -0.0000 ± 0.0000. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_beta0: -0.0000 ± 0.0000. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal gate_signal_rmse: +0.0175 ± 0.0208. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S3 age_temporal surface_rmse: -0.0208 ± 0.0268. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S3 age_temporal gate_signal_rmse: +0.0143 ± 0.0174. unfavorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- Observed S2 β mean -0.6356 ± 0.1133; S3 β mean 0.6127 ± 0.0345. S3 should reverse the S2 developmental sign.

Supported interpretation: a recovery metric moves beyond seed noise, and S2 predictive BCE or AUPRC also worsens beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C03_weak_lambda_init

Hypothesis: Initializing effective lambda at 0.1 instead of softplus(0) lets long-range evidence survive early training.

Single change: Set theta0 so softplus(theta0) = 0.1. Persistence intercept stays 0, so the initial effective lambda is 0.1.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/C03_weak_lambda_init`

- S2 age_temporal surface_rmse: -0.0045 ± 0.0169. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal gate_signal_rmse: -0.0078 ± 0.0010. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: +0.0119 ± 0.0086. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_lag: -0.0048 ± 0.0132. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0019 ± 0.0007. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0007 ± 0.0004. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: -0.0007 ± 0.0005. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: +0.0004 ± 0.0006. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal auprc: +0.0015 ± 0.0006. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0000 ± 0.0003. descriptive; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal delta_bce_gate_age_shuffle: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal delta_bce_beta0: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal gate_signal_rmse: -0.0148 ± 0.0200. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S3 age_temporal surface_rmse: -0.0171 ± 0.0169. favorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal gate_signal_rmse: -0.0157 ± 0.0110. favorable; |mean paired difference| exceeds the seed standard deviation.
- Observed S2 β mean -0.6693 ± 0.0568; S3 β mean 0.6830 ± 0.0239. S3 should reverse the S2 developmental sign.

Supported interpretation: at least one S2/S3 gate or prediction-surface comparison improves beyond seed noise, and S2 BCE/AUPRC do not worsen beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C04_no_content_persistence

Hypothesis: Removing r·v + b_r lets the developmental slope be the only persistence mechanism.

Single change: Drop the content persistence offset. Keep content query, exp(u), raw aggregation, history MLP, and the current optimizer.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/C04_no_content_persistence`

- S2 age_temporal surface_rmse: +0.0242 ± 0.0144. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: -0.0033 ± 0.0023. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: -0.0083 ± 0.0126. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal cf_rmse_lag: +0.0216 ± 0.0175. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0032 ± 0.0011. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0021 ± 0.0006. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0011 ± 0.0006. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: -0.0011 ± 0.0003. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0026 ± 0.0012. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0008 ± 0.0004. descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal delta_bce_beta0: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal gate_signal_rmse: -0.0783 ± 0.0160. favorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal surface_rmse: +0.0122 ± 0.0094. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal gate_signal_rmse: -0.0415 ± 0.0067. favorable; |mean paired difference| exceeds the seed standard deviation.
- Observed S2 β mean -0.5676 ± 0.0676; S3 β mean 0.5489 ± 0.0572. S3 should reverse the S2 developmental sign.

Supported interpretation: a recovery metric moves beyond seed noise, and S2 predictive BCE or AUPRC also worsens beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C05_shared_beta_mixture

Hypothesis: A few content-specific baseline timescales help when they share one developmental slope.

Single change: Replace r·v + b_r with K=3 mixture weights from content only and lambda_k=softplus(theta_k + beta z). beta is one shared scalar.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/C05_shared_beta_mixture`

- S2 age_temporal surface_rmse: +0.0242 ± 0.0144. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: -0.0033 ± 0.0023. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: -0.0083 ± 0.0126. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal cf_rmse_lag: +0.0216 ± 0.0175. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0032 ± 0.0011. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0021 ± 0.0006. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0011 ± 0.0006. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: -0.0011 ± 0.0003. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0026 ± 0.0012. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0008 ± 0.0004. descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal delta_bce_beta0: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal gate_signal_rmse: -0.0783 ± 0.0160. favorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal surface_rmse: +0.0122 ± 0.0094. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal gate_signal_rmse: -0.0415 ± 0.0067. favorable; |mean paired difference| exceeds the seed standard deviation.
- Observed S2 β mean -0.5676 ± 0.0676; S3 β mean 0.5489 ± 0.0572. S3 should reverse the S2 developmental sign.

Supported interpretation: a recovery metric moves beyond seed noise, and S2 predictive BCE or AUPRC also worsens beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## C06_component_beta_mixture

Hypothesis: Separate developmental slopes across mixture components are required for the E05-style gain.

Single change: Identical to C05 except each component has its own beta_k.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/C06_component_beta_mixture`

- S2 age_temporal surface_rmse: +0.0242 ± 0.0144. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal gate_signal_rmse: -0.0033 ± 0.0023. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal cf_rmse_age: -0.0083 ± 0.0126. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S2 age_temporal cf_rmse_lag: +0.0216 ± 0.0175. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_gate_age_shuffle: +0.0032 ± 0.0011. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal delta_bce_beta0: +0.0021 ± 0.0006. favorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal bce: +0.0011 ± 0.0006. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auroc: -0.0011 ± 0.0003. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 age_temporal auprc: -0.0026 ± 0.0012. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S2 matched delta_bce_to_minus_at: +0.0008 ± 0.0004. descriptive; |mean paired difference| exceeds the seed standard deviation.
- S0 age_temporal delta_bce_gate_age_shuffle: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal delta_bce_beta0: +0.0000 ± 0.0000. favorable; unresolved because the seed standard deviation is at least as large as the mean paired difference.
- S0 age_temporal gate_signal_rmse: -0.0783 ± 0.0160. favorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal surface_rmse: +0.0122 ± 0.0094. unfavorable; |mean paired difference| exceeds the seed standard deviation.
- S3 age_temporal gate_signal_rmse: -0.0415 ± 0.0067. favorable; |mean paired difference| exceeds the seed standard deviation.
- Observed S2 β mean -0.5676 ± 0.0676; S3 β mean 0.5489 ± 0.0572. S3 should reverse the S2 developmental sign.

Supported interpretation: a recovery metric moves beyond seed noise, and S2 predictive BCE or AUPRC also worsens beyond seed noise.
Unresolved where the paired seed standard deviation is at least as large as the mean difference.

## Changes with evidence for a later combination stage

This list is not a license to combine the changes yet. A name appears when a paired S2 or S3 surface or gate RMSE improvement exceeds seed noise and S2 BCE/AUPRC do not show a supported worsening.

- C01_staged_current
- C03_weak_lambda_init

## Figure paths

- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/figures/prediction_surface_S2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/figures/prediction_surface_S3.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/figures/gate_signal_rmse_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/figures/lambda_curves.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/figures/negative_controls_s0.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/atomic_followup_metrics.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/atomic_followup_summary.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/atomic_followup_summary.json`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_atomic_followup/paired_delta_vs_C00.csv`

