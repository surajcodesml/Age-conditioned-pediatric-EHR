# High-impact DTR follow-up

Baseline is C01_staged_current. D00 is an oracle-gate ceiling diagnostic. D01/D02 change content retrieval only, with staged optimization held fixed.

## Protocol

- Seeds 0–4, scenarios S0–S3, matched arms, same training budget as C01.
- D00 replaces the learned gate with generator `lambda_true(a)`. No trainable theta/beta.
- D01: H=4 content heads, d_head=16, shared developmental gate, total history width 64.
- D02: same heads as D01 with `beta_h = beta_global + centered delta_h`, identical to D01 at init.
- Gate-only age shuffle keeps the additive age head on true age.
- Head specialization: query cosine, content-score correlation, contribution norms, head ablation ΔBCE.

Artifact root: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup`

## Levels

| Experiment | S2 surface | S2 gate RMSE | S2 gate-shuffle ΔBCE | S2 BCE | S2 AUPRC | S0 gate-shuffle ΔBCE |
| --- | --- | --- | --- | --- | --- | --- |
| C01_staged_current | 0.1747 ± 0.0110 | 0.1493 ± 0.0055 | 0.0230 ± 0.0032 | 0.4557 ± 0.0006 | 0.5817 ± 0.0013 | 0.0000 ± 0.0000 |
| D00_oracle_gate | 0.2097 ± 0.0329 | 0.0000 ± 0.0000 | 0.0787 ± 0.0065 | 0.4605 ± 0.0009 | 0.5736 ± 0.0014 | 0.0000 ± 0.0000 |
| D01_multihead_shared | 0.1821 ± 0.0194 | 0.1501 ± 0.0029 | 0.0229 ± 0.0022 | 0.4551 ± 0.0006 | 0.5829 ± 0.0012 | -0.0000 ± 0.0000 |
| D02_multihead_dev | 0.1657 ± 0.0235 | 0.1544 ± 0.0158 | 0.0570 ± 0.0050 | 0.4504 ± 0.0004 | 0.5914 ± 0.0012 | 0.0007 ± 0.0006 |

## Paired deltas versus C01

| Experiment | S2 surface | S2 gate RMSE | S2 BCE | S2 AUPRC | S2 gate-shuffle ΔBCE |
| --- | --- | --- | --- | --- | --- |
| D00_oracle_gate | +0.0350 ± 0.0286 | -0.1493 ± 0.0055 | +0.0048 ± 0.0003 | -0.0080 ± 0.0009 | +0.0557 ± 0.0039 |
| D01_multihead_shared | +0.0074 ± 0.0210 | +0.0008 ± 0.0042 | -0.0006 ± 0.0007 | +0.0013 ± 0.0014 | -0.0001 ± 0.0023 |
| D02_multihead_dev | -0.0090 ± 0.0244 | +0.0051 ± 0.0191 | -0.0054 ± 0.0005 | +0.0097 ± 0.0013 | +0.0340 ± 0.0040 |

## D00_oracle_gate

Hypothesis: With a perfect temporal gate, the remaining C01 content encoder/retrieval/readout sets the prediction ceiling.

Single change: Replace the learned developmental gate with the generator lambda_true(a). No trainable theta/beta. Everything else matches C01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/D00_oracle_gate`

- S2 age_temporal surface_rmse vs C01: +0.0350 ± 0.0286. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal gate_signal_rmse vs C01: -0.1493 ± 0.0055. favorable; |mean| exceeds seed sd.
- S2 age_temporal bce vs C01: +0.0048 ± 0.0003. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auroc vs C01: -0.0072 ± 0.0006. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal auprc vs C01: -0.0080 ± 0.0009. unfavorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_gate_age_shuffle vs C01: +0.0557 ± 0.0039. favorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_beta0 vs C01: +0.0211 ± 0.0032. favorable; |mean| exceeds seed sd.
- S0 age_temporal delta_bce_gate_age_shuffle vs C01: -0.0000 ± 0.0000. unfavorable; unresolved vs seed sd.
- S3 age_temporal surface_rmse vs C01: +0.0225 ± 0.0181. unfavorable; |mean| exceeds seed sd.
- S3 age_temporal beta_mean vs C01: +1.6140 ± 0.0515. descriptive; |mean| exceeds seed sd.

## D01_multihead_shared

Hypothesis: Four content retrieval heads with fixed total width improve recovery while sharing one developmental gate.

Single change: Replace the single content query/key with H=4 heads of width 16. Shared lambda/gate across heads. Concatenate head histories to width 64.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/D01_multihead_shared`

- S2 age_temporal surface_rmse vs C01: +0.0074 ± 0.0210. unfavorable; unresolved vs seed sd.
- S2 age_temporal gate_signal_rmse vs C01: +0.0008 ± 0.0042. unfavorable; unresolved vs seed sd.
- S2 age_temporal bce vs C01: -0.0006 ± 0.0007. favorable; unresolved vs seed sd.
- S2 age_temporal auroc vs C01: +0.0011 ± 0.0012. favorable; unresolved vs seed sd.
- S2 age_temporal auprc vs C01: +0.0013 ± 0.0014. favorable; unresolved vs seed sd.
- S2 age_temporal delta_bce_gate_age_shuffle vs C01: -0.0001 ± 0.0023. unfavorable; unresolved vs seed sd.
- S2 age_temporal delta_bce_beta0 vs C01: +0.0000 ± 0.0021. favorable; unresolved vs seed sd.
- S0 age_temporal delta_bce_gate_age_shuffle vs C01: -0.0000 ± 0.0000. unfavorable; unresolved vs seed sd.
- S3 age_temporal surface_rmse vs C01: +0.0036 ± 0.0216. unfavorable; unresolved vs seed sd.
- S3 age_temporal beta_mean vs C01: -0.0442 ± 0.0765. descriptive; unresolved vs seed sd.
- S2 effective_n_heads: 3.9947 ± 0.0059
- S2 head ablation ΔBCE[0]: 0.0306 ± 0.0066
- S2 head ablation ΔBCE[1]: 0.0256 ± 0.0085
- S2 head ablation ΔBCE[2]: 0.0227 ± 0.0037
- S2 head ablation ΔBCE[3]: 0.0302 ± 0.0153

## D02_multihead_dev

Hypothesis: After multi-head content specialization, head-specific centered developmental slopes help further.

Single change: Same multi-head content as D01. beta_h = beta_global + centered delta_h. Identical to D01 at initialization.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/D02_multihead_dev`

- S2 age_temporal surface_rmse vs C01: -0.0090 ± 0.0244. favorable; unresolved vs seed sd.
- S2 age_temporal gate_signal_rmse vs C01: +0.0051 ± 0.0191. unfavorable; unresolved vs seed sd.
- S2 age_temporal bce vs C01: -0.0054 ± 0.0005. favorable; |mean| exceeds seed sd.
- S2 age_temporal auroc vs C01: +0.0056 ± 0.0012. favorable; |mean| exceeds seed sd.
- S2 age_temporal auprc vs C01: +0.0097 ± 0.0013. favorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_gate_age_shuffle vs C01: +0.0340 ± 0.0040. favorable; |mean| exceeds seed sd.
- S2 age_temporal delta_bce_beta0 vs C01: +0.0174 ± 0.0016. favorable; |mean| exceeds seed sd.
- S0 age_temporal delta_bce_gate_age_shuffle vs C01: +0.0007 ± 0.0006. favorable; |mean| exceeds seed sd.
- S3 age_temporal surface_rmse vs C01: -0.0071 ± 0.0207. favorable; unresolved vs seed sd.
- S3 age_temporal beta_mean vs C01: -0.0231 ± 0.1623. descriptive; unresolved vs seed sd.
- S2 effective_n_heads: 3.9951 ± 0.0050
- S2 head ablation ΔBCE[0]: 0.0327 ± 0.0081
- S2 head ablation ΔBCE[1]: 0.0290 ± 0.0113
- S2 head ablation ΔBCE[2]: 0.0476 ± 0.0190
- S2 head ablation ΔBCE[3]: 0.0439 ± 0.0111

## Oracle-gate ceiling vs C01 and CEHR-BERT (S2 age_temporal)

| Model | BCE | AUROC | AUPRC | Surface RMSE | Gate RMSE |
| --- | --- | --- | --- | --- | --- |
| C01_staged_current | 0.4557 ± 0.0006 | 0.7613 ± 0.0010 | 0.5817 ± 0.0013 | 0.1747 ± 0.0110 | 0.1493 ± 0.0055 |
| D00_oracle_gate | 0.4605 ± 0.0009 | 0.7540 ± 0.0009 | 0.5736 ± 0.0014 | 0.2097 ± 0.0329 | 0.0000 ± 0.0000 |
| CEHR-BERT (published, n=1) | — | 0.7636 | 0.5818 | 0.1308 | — |
| CEHR-BERT-small (seeds 0–4) | — | 0.7390 ± 0.0088 | 0.5410 ± 0.0134 | 0.1645 ± 0.0117 | — |

D00 drives gate RMSE to zero by construction, but S2 BCE/AUPRC/surface do not improve over C01 (paired BCE +0.0048, AUPRC −0.0080, surface +0.0350). Relative to CEHR-BERT's surface RMSE 0.1308, D00 (0.2097) remains far above the published Transformer surface.

## Multi-head specialization

D01/D02 keep all four heads active (effective_n_heads ≈ 4; each head ablation raises S2 BCE by ~0.02–0.05), but encounter content scores are highly correlated across heads (mean |off-diagonal corr| ≈ 0.73–0.82 on S2/S3). Query cosine off-diagonals are only modest (~0.17–0.22). Multi-head capacity therefore does not produce clearly distinct content-relevance directions.

D01 vs C01 is essentially null on S2 predictive and mechanism metrics (BCE -0.0006 ± 0.0007, surface +0.0074 ± 0.0210, AUPRC +0.0013 ± 0.0014). Do not retain multi-head content retrieval as a supported improvement over C01.

D02 improves S2 BCE (-0.0054 ± 0.0005) but does so with large head-specific delta_h on S2 while gate RMSE does not improve (+0.0051 ± 0.0191). S0 gate-shuffle ΔBCE vs C01 is +0.0007 ± 0.0006 (small false interaction). Because the oracle uses one shared developmental slope, do not retain head-specific betas.

## Decision

Observed result (mechanical):
- D00 vs C01 S2 surface: +0.0350 ± 0.0286
- D00 vs C01 S2 BCE: +0.0048 ± 0.0003
- D00 vs C01 S2 AUPRC: -0.0080 ± 0.0009
- D00 vs C01 S2 gate RMSE: -0.1493 ± 0.0055
- D01 vs C01 S2 surface: +0.0074 ± 0.0210
- D01 vs C01 S2 BCE: -0.0006 ± 0.0007

**Supported interpretation: B. content retrieval is the main bottleneck.**

With a perfect Synthea oracle gate, D00 still fails to beat C01 on S2 BCE/AUPRC/surface and remains well above CEHR-BERT surface quality. The remaining ceiling is therefore content representation/retrieval/readout, not lambda learning. D01 did not break that ceiling; heads stay content-correlated. D02's predictive gains come from large unsupported beta_h separation and are not retained.

Retain C01_staged_current as the working baseline. Do not retain D01 or D02.

## Figure paths

- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/prediction_surface_S2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/prediction_surface_S3.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/surface_rmse_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/D01_multihead_shared_query_cosine.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/D01_multihead_shared_content_score_corr.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/D02_multihead_dev_query_cosine.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/D02_multihead_dev_content_score_corr.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/figures/d02_lambda_h.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/high_impact_followup_metrics.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/dtr_high_impact_followup/paired_delta_vs_C01.csv`

