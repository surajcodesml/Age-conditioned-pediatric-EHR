# High-impact DTR follow-up

Baseline is C01_staged_current. D00 is an oracle-gate ceiling diagnostic. D01/D02 change content retrieval only, with staged optimization held fixed.

## Protocol

- Seeds 0–4, scenarios S0–S3, matched arms, same training budget as C01.
- D00 replaces the learned gate with generator `lambda_true(a)`. No trainable theta/beta.
- D01: H=4 content heads, d_head=16, shared developmental gate, total history width 64.
- D02: same heads as D01 with `beta_h = beta_global + centered delta_h`, identical to D01 at init.
- Gate-only age shuffle keeps the additive age head on true age.
- Head specialization: query cosine, content-score correlation, contribution norms, head ablation ΔBCE.

Artifact root: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke`

## Levels

| Experiment | S2 surface | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 BCE | S2 AUPRC | S0 gate-shuffle ΔBCE |
| --- | --- | --- | --- | --- | --- | --- |
| C01_staged_current | 0.1747 ± 0.0110 | 0.1493 ± 0.0055 | 0.0230 ± 0.0032 | 0.4557 ± 0.0006 | 0.5817 ± 0.0013 | 0.0000 ± 0.0000 |
| D00_oracle_gate | 0.3504 | 0.0000 | -0.0014 | 0.7478 | 0.2412 | — |
| D01_multihead_shared | 0.3387 | 0.2871 | 0.0000 | 0.7466 | 0.2414 | — |
| D02_multihead_dev | 0.3387 | 0.2871 | 0.0000 | 0.7466 | 0.2414 | — |

## Paired deltas versus C01

| Experiment | S2 surface | S2 gate RMSE | S2 BCE | S2 AUPRC | S2 gate-shuffle ΔBCE |
| --- | --- | --- | --- | --- | --- |
| D00_oracle_gate | +0.1596 | -0.1513 | +0.2926 | -0.3415 | -0.0224 |
| D01_multihead_shared | +0.1479 | +0.1358 | +0.2914 | -0.3412 | -0.0211 |
| D02_multihead_dev | +0.1479 | +0.1358 | +0.2914 | -0.3412 | -0.0211 |

## D00_oracle_gate

Hypothesis: With a perfect temporal gate, the remaining C01 content encoder/retrieval/readout sets the prediction ceiling.

Single change: Replace the learned developmental gate with the generator lambda_true(a). No trainable theta/beta. Everything else matches C01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/D00_oracle_gate`

- S2 age_temporal surface_rmse vs C01: +0.1596. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal gate_signal_rmse vs C01: -0.1513. favorable; |mean| exceeds seed sd.
- S2 age_temporal bce vs C01: +0.2926. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auroc vs C01: -0.2738. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auprc vs C01: -0.3415. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_gate_age_shuffle vs C01: -0.0224. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_beta0 vs C01: -0.0132. unfavorable; |mean| exceeds seed sd.
- S0 age_temporal delta_bce_gate_age_shuffle vs C01: —. missing.
- S3 age_temporal surface_rmse vs C01: —. missing.
- S3 age_temporal beta_mean vs C01: —. missing.

## D01_multihead_shared

Hypothesis: Four content retrieval heads with fixed total width improve recovery while sharing one developmental gate.

Single change: Replace the single content query/key with H=4 heads of width 16. Shared lambda/gate across heads. Concatenate head histories to width 64.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/D01_multihead_shared`

- S2 age_temporal surface_rmse vs C01: +0.1479. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal gate_signal_rmse vs C01: +0.1358. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal bce vs C01: +0.2914. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auroc vs C01: -0.2701. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auprc vs C01: -0.3412. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_gate_age_shuffle vs C01: -0.0211. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_beta0 vs C01: -0.0116. unfavorable; |mean| exceeds seed sd.
- S0 age_temporal delta_bce_gate_age_shuffle vs C01: —. missing.
- S3 age_temporal surface_rmse vs C01: —. missing.
- S3 age_temporal beta_mean vs C01: —. missing.
- S2 effective_n_heads: 4.0000
- S2 head ablation ΔBCE[0]: 0.0015
- S2 head ablation ΔBCE[1]: 0.0012
- S2 head ablation ΔBCE[2]: 0.0008
- S2 head ablation ΔBCE[3]: 0.0013

## D02_multihead_dev

Hypothesis: After multi-head content specialization, head-specific centered developmental slopes help further.

Single change: Same multi-head content as D01. beta_h = beta_global + centered delta_h. Identical to D01 at initialization.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/D02_multihead_dev`

- S2 age_temporal surface_rmse vs C01: +0.1479. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal gate_signal_rmse vs C01: +0.1358. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal bce vs C01: +0.2914. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auroc vs C01: -0.2701. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auprc vs C01: -0.3412. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_gate_age_shuffle vs C01: -0.0211. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_beta0 vs C01: -0.0116. unfavorable; |mean| exceeds seed sd.
- S0 age_temporal delta_bce_gate_age_shuffle vs C01: —. missing.
- S3 age_temporal surface_rmse vs C01: —. missing.
- S3 age_temporal beta_mean vs C01: —. missing.
- S2 effective_n_heads: 4.0000
- S2 head ablation ΔBCE[0]: 0.0015
- S2 head ablation ΔBCE[1]: 0.0012
- S2 head ablation ΔBCE[2]: 0.0008
- S2 head ablation ΔBCE[3]: 0.0013

## Decision

Observed result (mechanical):
- D00 vs C01 S2 surface: +0.1596
- D00 vs C01 S2 BCE: +0.2926
- D01 vs C01 S2 surface: +0.1479
- D01 vs C01 S2 BCE: +0.2914

Supported interpretation: C. both remain limiting.
Unresolved where paired seed sd is at least as large as the mean difference, and wherever heads fail to specialize.

## Figure paths

- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/figures/prediction_surface_S2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/figures/prediction_surface_S3.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/figures/surface_rmse_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/figures/D01_multihead_shared_query_cosine.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/figures/D01_multihead_shared_content_score_corr.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/figures/D02_multihead_dev_query_cosine.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/figures/d02_lambda_h.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/high_impact_followup_metrics.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/_smoke/paired_delta_vs_C01.csv`

