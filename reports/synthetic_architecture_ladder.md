# Synthetic architecture ladder

Metrics and figures were computed from saved predictions and mechanism arrays.
Models were not rerun for the tables or plots.

## Protocol

- Dataset: existing controlled Synthea scenarios S0–S3, patient split seed 20260922.
- Seeds, fixed before fitting: 0, 1, 2, 3, 4. This is the existing `cehrbert_small` seed list. The current DTR reference itself was fit only at seed 0.
- Training budget matches `dtr_age_temporal_new`: AdamW, lr 3e-4, weight decay 0.01, batch 32, 25 epochs, patience 5, minimum 12 epochs, gradient clip 1, d_model 64, dropout 0.
- Encounter tensors are precomputed and padded to the split maximum. Pad positions are masked, and the shuffle still comes from the trainer seed. Independent seeds run two at a time on one GPU. Batch size, learning rate, epoch budget, and seeds are unchanged.
- E02 is the only run that changes the optimizer.
- E06 is an extension. The S2 oracle was generated with current-age λ(a), not an integrated hazard, so E06 is not expected to beat E01 on oracle recovery by construction.
- Surface RMSE, CF-RMSE_age, and CF-RMSE_lag use `baselines.common.counterfactual` on the saved grids.
- β=0 ΔBCE and age-shuffle ΔBCE are test-set BCE(counterfactual) − BCE(original). The age shuffle uses NumPy seed 0, the same seed as the existing DTR ablation.
- For mixture models, λ(a) compared with the oracle is the unweighted mean of λ_k(a). That reduction was fixed in the metric code before the runs.
- Seed-level intervals in the summary CSV are normal approximations mean ± 1.96·sd/√n. Each run's `metrics.json` also stores a 200-draw patient bootstrap of the BCE deltas.

Artifact root: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder`

## E01_direct

Hypothesis: Retrieval and readout complexity in the current DTR is the main bottleneck for explicit age×lag recovery.

Single change: Remove content query, exp(u), content-dependent persistence, and the nonlinear history MLP. Direct linear readout. Weak lambda initialization.

Parameters added: none beyond theta0, beta, linear W_history, linear W_age, and bias. Parameters removed: content query/key, exp(u), persistence projection r·v + b_r, history MLP.

Training: Same trainer as the current DTR baseline: one AdamW group, learning rate 3e-4, weight decay 1e-2, 25 epochs, patience 5, minimum 12 epochs, batch size 32, gradient clip 1.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/E01_direct`

- S0 age_temporal: BCE 0.5031 ± 0.0004; AUROC 0.7098 ± 0.0007; AUPRC 0.4763 ± 0.0015; Surface RMSE 0.1973 ± 0.0188; CF-RMSE_age 0.1865 ± 0.0187; CF-RMSE_lag 0.1776 ± 0.0194; β=0 ΔBCE 0.0003 ± 0.0001; age-shuffle ΔBCE 0.0083 ± 0.0018; β 0.1095 ± 0.0406; |β| 0.1095 ± 0.0406; λ RMSE 0.4728 ± 0.0027; λ correlation —
- S0 temporal_only: BCE 0.5031 ± 0.0004; AUROC 0.7097 ± 0.0006; AUPRC 0.4762 ± 0.0015; Surface RMSE 0.1980 ± 0.0190; CF-RMSE_age 0.1871 ± 0.0192; CF-RMSE_lag 0.1782 ± 0.0198; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0080 ± 0.0019; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.4720 ± 0.0028; λ correlation —
- S1 age_temporal: BCE 0.4966 ± 0.0007; AUROC 0.7252 ± 0.0012; AUPRC 0.4963 ± 0.0017; Surface RMSE 0.2194 ± 0.0110; CF-RMSE_age 0.2005 ± 0.0184; CF-RMSE_lag 0.1789 ± 0.0150; β=0 ΔBCE 0.0009 ± 0.0003; age-shuffle ΔBCE 0.0104 ± 0.0012; β 0.1944 ± 0.0428; |β| 0.1944 ± 0.0428; λ RMSE 0.4857 ± 0.0039; λ correlation —
- S1 temporal_only: BCE 0.4968 ± 0.0006; AUROC 0.7251 ± 0.0011; AUPRC 0.4955 ± 0.0016; Surface RMSE 0.2175 ± 0.0109; CF-RMSE_age 0.1989 ± 0.0177; CF-RMSE_lag 0.1783 ± 0.0148; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0092 ± 0.0012; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.4849 ± 0.0040; λ correlation —
- S2 age_temporal: BCE 0.4723 ± 0.0008; AUROC 0.7471 ± 0.0012; AUPRC 0.5502 ± 0.0013; Surface RMSE 0.2540 ± 0.0090; CF-RMSE_age 0.2053 ± 0.0138; CF-RMSE_lag 0.1990 ± 0.0113; β=0 ΔBCE 0.0011 ± 0.0003; age-shuffle ΔBCE 0.0110 ± 0.0014; β -0.2975 ± 0.0407; |β| 0.2975 ± 0.0407; λ RMSE 1.0714 ± 0.0043; λ correlation 0.9791 ± 0.0019
- S2 temporal_only: BCE 0.4733 ± 0.0008; AUROC 0.7466 ± 0.0012; AUPRC 0.5481 ± 0.0013; Surface RMSE 0.2553 ± 0.0089; CF-RMSE_age 0.2045 ± 0.0134; CF-RMSE_lag 0.1984 ± 0.0120; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0105 ± 0.0013; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.0968 ± 0.0013; λ correlation —
- S3 age_temporal: BCE 0.5033 ± 0.0006; AUROC 0.7162 ± 0.0011; AUPRC 0.4943 ± 0.0018; Surface RMSE 0.1853 ± 0.0136; CF-RMSE_age 0.1580 ± 0.0104; CF-RMSE_lag 0.1318 ± 0.0036; β=0 ΔBCE 0.0034 ± 0.0005; age-shuffle ΔBCE 0.0139 ± 0.0023; β 0.5216 ± 0.0428; |β| 0.5216 ± 0.0428; λ RMSE 1.0680 ± 0.0060; λ correlation 0.9881 ± 0.0014
- S3 temporal_only: BCE 0.5054 ± 0.0005; AUROC 0.7148 ± 0.0012; AUPRC 0.4892 ± 0.0015; Surface RMSE 0.1810 ± 0.0108; CF-RMSE_age 0.1502 ± 0.0073; CF-RMSE_lag 0.1314 ± 0.0042; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0085 ± 0.0025; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.1091 ± 0.0016; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E00 current DTR: Observed difference +0.0527 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal cf_rmse_age versus E00 current DTR: Observed difference +0.0631 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal cf_rmse_lag versus E00 current DTR: Observed difference +0.0334 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal bce versus E00 current DTR: Observed difference +0.0137 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auroc versus E00 current DTR: Observed difference -0.0134 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auprc versus E00 current DTR: Observed difference -0.0264 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_beta0 versus E00 current DTR: Observed difference -0.0059 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_age_shuffle versus E00 current DTR: Observed difference -0.0091 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S0 CF-RMSE_age versus E00 current DTR: Observed difference +0.0879 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S0 |β| is 0.1095 ± 0.0406.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E02_staged

Hypothesis: A staged optimizer, with architecture held fixed at E01, improves mechanism recovery.

Single change: No architectural change relative to E01.

Parameters added: none. Parameters removed: none.

Training: Stage A trains beta=0. The checkpoint is cloned into both arms. Stage B freezes the encoder and readout and trains theta0/beta for 5 epochs at 10x learning rate and zero weight decay. Stage C unfreezes all parameters and fine-tunes with the same temporal parameter group.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/E02_staged`

- S0 age_temporal: BCE 0.4910 ± 0.0004; AUROC 0.7198 ± 0.0009; AUPRC 0.5105 ± 0.0010; Surface RMSE 0.2193 ± 0.0163; CF-RMSE_age 0.2066 ± 0.0123; CF-RMSE_lag 0.2023 ± 0.0157; β=0 ΔBCE -0.0000 ± 0.0000; age-shuffle ΔBCE 0.0064 ± 0.0013; β -0.0277 ± 0.0192; |β| 0.0284 ± 0.0180; λ RMSE 0.1937 ± 0.0049; λ correlation —
- S0 temporal_only: BCE 0.4909 ± 0.0004; AUROC 0.7198 ± 0.0009; AUPRC 0.5106 ± 0.0009; Surface RMSE 0.2192 ± 0.0161; CF-RMSE_age 0.2064 ± 0.0123; CF-RMSE_lag 0.2021 ± 0.0153; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0065 ± 0.0013; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.1942 ± 0.0050; λ correlation —
- S1 age_temporal: BCE 0.4881 ± 0.0004; AUROC 0.7340 ± 0.0008; AUPRC 0.5227 ± 0.0008; Surface RMSE 0.2142 ± 0.0055; CF-RMSE_age 0.1732 ± 0.0115; CF-RMSE_lag 0.1734 ± 0.0087; β=0 ΔBCE 0.0001 ± 0.0002; age-shuffle ΔBCE 0.0096 ± 0.0015; β 0.0172 ± 0.0242; |β| 0.0190 ± 0.0224; λ RMSE 0.2645 ± 0.0021; λ correlation —
- S1 temporal_only: BCE 0.4882 ± 0.0004; AUROC 0.7339 ± 0.0008; AUPRC 0.5226 ± 0.0009; Surface RMSE 0.2140 ± 0.0058; CF-RMSE_age 0.1732 ± 0.0116; CF-RMSE_lag 0.1734 ± 0.0087; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0095 ± 0.0015; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.2641 ± 0.0025; λ correlation —
- S2 age_temporal: BCE 0.4661 ± 0.0007; AUROC 0.7537 ± 0.0007; AUPRC 0.5631 ± 0.0014; Surface RMSE 0.2532 ± 0.0073; CF-RMSE_age 0.1983 ± 0.0087; CF-RMSE_lag 0.1996 ± 0.0110; β=0 ΔBCE 0.0041 ± 0.0011; age-shuffle ΔBCE 0.0158 ± 0.0015; β -0.5165 ± 0.0790; |β| 0.5165 ± 0.0790; λ RMSE 0.9416 ± 0.0228; λ correlation 0.9854 ± 0.0024
- S2 temporal_only: BCE 0.4680 ± 0.0004; AUROC 0.7531 ± 0.0008; AUPRC 0.5593 ± 0.0007; Surface RMSE 0.2581 ± 0.0085; CF-RMSE_age 0.1983 ± 0.0105; CF-RMSE_lag 0.2006 ± 0.0116; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0105 ± 0.0010; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.0182 ± 0.0055; λ correlation —
- S3 age_temporal: BCE 0.4975 ± 0.0007; AUROC 0.7232 ± 0.0011; AUPRC 0.5107 ± 0.0020; Surface RMSE 0.1864 ± 0.0152; CF-RMSE_age 0.1459 ± 0.0059; CF-RMSE_lag 0.1322 ± 0.0022; β=0 ΔBCE 0.0049 ± 0.0009; age-shuffle ΔBCE 0.0156 ± 0.0018; β 0.4874 ± 0.0428; |β| 0.4874 ± 0.0428; λ RMSE 0.9547 ± 0.0128; λ correlation 0.9847 ± 0.0013
- S3 temporal_only: BCE 0.4999 ± 0.0004; AUROC 0.7220 ± 0.0010; AUPRC 0.5049 ± 0.0012; Surface RMSE 0.1812 ± 0.0115; CF-RMSE_age 0.1337 ± 0.0038; CF-RMSE_lag 0.1315 ± 0.0010; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0084 ± 0.0023; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.0161 ± 0.0042; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0008 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal cf_rmse_age versus E01: Observed difference -0.0070 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference +0.0006 (unfavorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal bce versus E01: Observed difference -0.0062 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auroc versus E01: Observed difference +0.0065 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auprc versus E01: Observed difference +0.0129 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference +0.0030 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0048 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S0 CF-RMSE_age versus E01: Observed difference +0.0201 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S0 |β| is 0.0284 ± 0.0180.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E03_mass

Hypothesis: Separating history composition from surviving evidence mass improves stability and recovery.

Single change: Replace only the sum g·v aggregation by normalized composition plus log1p(evidence mass).

Parameters added: log1p(M) feature on the linear readout. Parameters removed: raw unnormalized sum as the sole history vector.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/E03_mass`

- S0 age_temporal: BCE 0.4964 ± 0.0007; AUROC 0.7205 ± 0.0013; AUPRC 0.4948 ± 0.0018; Surface RMSE 0.1745 ± 0.0075; CF-RMSE_age 0.1402 ± 0.0123; CF-RMSE_lag 0.1487 ± 0.0076; β=0 ΔBCE 0.0006 ± 0.0001; age-shuffle ΔBCE 0.0071 ± 0.0010; β 0.3878 ± 0.1395; |β| 0.3878 ± 0.1395; λ RMSE 0.3285 ± 0.0069; λ correlation —
- S0 temporal_only: BCE 0.4969 ± 0.0007; AUROC 0.7199 ± 0.0014; AUPRC 0.4936 ± 0.0018; Surface RMSE 0.1745 ± 0.0076; CF-RMSE_age 0.1401 ± 0.0128; CF-RMSE_lag 0.1484 ± 0.0077; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0064 ± 0.0008; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.3239 ± 0.0062; λ correlation —
- S1 age_temporal: BCE 0.4915 ± 0.0002; AUROC 0.7341 ± 0.0010; AUPRC 0.5081 ± 0.0009; Surface RMSE 0.1963 ± 0.0091; CF-RMSE_age 0.1692 ± 0.0109; CF-RMSE_lag 0.1576 ± 0.0049; β=0 ΔBCE 0.0005 ± 0.0001; age-shuffle ΔBCE 0.0110 ± 0.0023; β 0.4053 ± 0.1101; |β| 0.4053 ± 0.1101; λ RMSE 0.3446 ± 0.0064; λ correlation —
- S1 temporal_only: BCE 0.4920 ± 0.0001; AUROC 0.7337 ± 0.0010; AUPRC 0.5063 ± 0.0005; Surface RMSE 0.1962 ± 0.0092; CF-RMSE_age 0.1696 ± 0.0112; CF-RMSE_lag 0.1575 ± 0.0050; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0106 ± 0.0024; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.3425 ± 0.0046; λ correlation —
- S2 age_temporal: BCE 0.4753 ± 0.0006; AUROC 0.7468 ± 0.0007; AUPRC 0.5414 ± 0.0013; Surface RMSE 0.2252 ± 0.0111; CF-RMSE_age 0.1669 ± 0.0125; CF-RMSE_lag 0.1819 ± 0.0091; β=0 ΔBCE 0.0016 ± 0.0002; age-shuffle ΔBCE 0.0156 ± 0.0021; β 0.6012 ± 0.0856; |β| 0.6012 ± 0.0856; λ RMSE 1.0792 ± 0.0104; λ correlation -0.9256 ± 0.0062
- S2 temporal_only: BCE 0.4767 ± 0.0006; AUROC 0.7472 ± 0.0008; AUPRC 0.5378 ± 0.0012; Surface RMSE 0.2256 ± 0.0113; CF-RMSE_age 0.1720 ± 0.0162; CF-RMSE_lag 0.1833 ± 0.0096; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0117 ± 0.0016; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.0106 ± 0.0038; λ correlation —
- S3 age_temporal: BCE 0.4988 ± 0.0004; AUROC 0.7239 ± 0.0008; AUPRC 0.5032 ± 0.0016; Surface RMSE 0.1661 ± 0.0059; CF-RMSE_age 0.1424 ± 0.0072; CF-RMSE_lag 0.1165 ± 0.0010; β=0 ΔBCE 0.0002 ± 0.0001; age-shuffle ΔBCE 0.0091 ± 0.0008; β -0.2724 ± 0.0525; |β| 0.2724 ± 0.0525; λ RMSE 1.0543 ± 0.0068; λ correlation -0.9472 ± 0.0034
- S3 temporal_only: BCE 0.4989 ± 0.0004; AUROC 0.7244 ± 0.0009; AUPRC 0.5027 ± 0.0018; Surface RMSE 0.1664 ± 0.0059; CF-RMSE_age 0.1446 ± 0.0073; CF-RMSE_lag 0.1164 ± 0.0012; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0088 ± 0.0007; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.0232 ± 0.0028; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0288 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal cf_rmse_age versus E01: Observed difference -0.0384 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference -0.0172 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal bce versus E01: Observed difference +0.0030 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auroc versus E01: Observed difference -0.0003 (unfavorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal auprc versus E01: Observed difference -0.0088 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference +0.0005 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0046 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S0 CF-RMSE_age versus E01: Observed difference -0.0463 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S0 |β| is 0.3878 ± 0.1395.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E04_channels

Hypothesis: Extra content channels help while the developmental decay stays a single shared lambda(a).

Single change: Add 4 content channels c=sigmoid(q_h^T v) with one shared lambda(a). Queries do not receive age or lag.

Parameters added: 4 content query vectors. Parameters removed: none from the E01 mechanism.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/E04_channels`

- S0 age_temporal: BCE 0.5027 ± 0.0007; AUROC 0.7098 ± 0.0011; AUPRC 0.4771 ± 0.0015; Surface RMSE 0.1896 ± 0.0127; CF-RMSE_age 0.1802 ± 0.0110; CF-RMSE_lag 0.1723 ± 0.0107; β=0 ΔBCE 0.0002 ± 0.0001; age-shuffle ΔBCE 0.0071 ± 0.0014; β 0.0962 ± 0.0405; |β| 0.0962 ± 0.0405; λ RMSE 0.4627 ± 0.0033; λ correlation —
- S0 temporal_only: BCE 0.5027 ± 0.0007; AUROC 0.7098 ± 0.0011; AUPRC 0.4771 ± 0.0014; Surface RMSE 0.1898 ± 0.0126; CF-RMSE_age 0.1804 ± 0.0112; CF-RMSE_lag 0.1720 ± 0.0105; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0069 ± 0.0013; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.4618 ± 0.0033; λ correlation —
- S1 age_temporal: BCE 0.4955 ± 0.0003; AUROC 0.7267 ± 0.0002; AUPRC 0.4988 ± 0.0006; Surface RMSE 0.2190 ± 0.0054; CF-RMSE_age 0.1928 ± 0.0113; CF-RMSE_lag 0.1772 ± 0.0097; β=0 ΔBCE 0.0007 ± 0.0003; age-shuffle ΔBCE 0.0100 ± 0.0023; β 0.1723 ± 0.0361; |β| 0.1723 ± 0.0361; λ RMSE 0.4770 ± 0.0020; λ correlation —
- S1 temporal_only: BCE 0.4957 ± 0.0003; AUROC 0.7267 ± 0.0002; AUPRC 0.4982 ± 0.0007; Surface RMSE 0.2179 ± 0.0056; CF-RMSE_age 0.1919 ± 0.0112; CF-RMSE_lag 0.1772 ± 0.0095; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0092 ± 0.0021; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 0.4759 ± 0.0021; λ correlation —
- S2 age_temporal: BCE 0.4709 ± 0.0002; AUROC 0.7488 ± 0.0007; AUPRC 0.5531 ± 0.0006; Surface RMSE 0.2506 ± 0.0074; CF-RMSE_age 0.2066 ± 0.0078; CF-RMSE_lag 0.1957 ± 0.0070; β=0 ΔBCE 0.0017 ± 0.0003; age-shuffle ΔBCE 0.0111 ± 0.0016; β -0.3739 ± 0.0361; |β| 0.3739 ± 0.0361; λ RMSE 1.0556 ± 0.0043; λ correlation 0.9822 ± 0.0014
- S2 temporal_only: BCE 0.4721 ± 0.0003; AUROC 0.7477 ± 0.0008; AUPRC 0.5508 ± 0.0009; Surface RMSE 0.2523 ± 0.0087; CF-RMSE_age 0.2063 ± 0.0099; CF-RMSE_lag 0.1953 ± 0.0081; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0092 ± 0.0022; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.0915 ± 0.0018; λ correlation —
- S3 age_temporal: BCE 0.5022 ± 0.0003; AUROC 0.7179 ± 0.0006; AUPRC 0.4968 ± 0.0016; Surface RMSE 0.1786 ± 0.0145; CF-RMSE_age 0.1522 ± 0.0153; CF-RMSE_lag 0.1285 ± 0.0015; β=0 ΔBCE 0.0035 ± 0.0005; age-shuffle ΔBCE 0.0141 ± 0.0018; β 0.5241 ± 0.0432; |β| 0.5241 ± 0.0432; λ RMSE 1.0596 ± 0.0057; λ correlation 0.9880 ± 0.0014
- S3 temporal_only: BCE 0.5043 ± 0.0002; AUROC 0.7165 ± 0.0008; AUPRC 0.4916 ± 0.0011; Surface RMSE 0.1754 ± 0.0119; CF-RMSE_age 0.1460 ± 0.0136; CF-RMSE_lag 0.1286 ± 0.0023; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0086 ± 0.0016; β 0.0000 ± 0.0000; |β| 0.0000 ± 0.0000; λ RMSE 1.1029 ± 0.0012; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0034 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal cf_rmse_age versus E01: Observed difference +0.0013 (unfavorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference -0.0033 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal bce versus E01: Observed difference -0.0014 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auroc versus E01: Observed difference +0.0016 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auprc versus E01: Observed difference +0.0029 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference +0.0006 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0001 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S0 CF-RMSE_age versus E01: Observed difference -0.0062 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S0 |β| is 0.0962 ± 0.0405.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E05_mixture

Hypothesis: A small content-dependent mixture of timescales recovers the age×lag surface better than one global decay.

Single change: K=3 content-only mixture of developmental rates. First model with content-specific timescales.

Parameters added: linear content mixture, theta_k, beta_k. Parameters removed: single global theta0/beta.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/E05_mixture`

- S0 age_temporal: BCE 0.5002 ± 0.0021; AUROC 0.7138 ± 0.0029; AUPRC 0.4841 ± 0.0046; Surface RMSE 0.1902 ± 0.0159; CF-RMSE_age 0.1771 ± 0.0174; CF-RMSE_lag 0.1702 ± 0.0180; β=0 ΔBCE 0.0003 ± 0.0002; age-shuffle ΔBCE 0.0074 ± 0.0019; β —; |β| 0.1322 ± 0.0112; λ RMSE 0.5303 ± 0.0053; λ correlation —
- S0 temporal_only: BCE 0.4993 ± 0.0012; AUROC 0.7149 ± 0.0019; AUPRC 0.4864 ± 0.0021; Surface RMSE 0.1894 ± 0.0148; CF-RMSE_age 0.1759 ± 0.0118; CF-RMSE_lag 0.1685 ± 0.0171; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0073 ± 0.0019; β —; |β| 0.0000 ± 0.0000; λ RMSE 0.5295 ± 0.0037; λ correlation —
- S1 age_temporal: BCE 0.4957 ± 0.0028; AUROC 0.7265 ± 0.0036; AUPRC 0.4988 ± 0.0072; Surface RMSE 0.2094 ± 0.0104; CF-RMSE_age 0.1955 ± 0.0108; CF-RMSE_lag 0.1746 ± 0.0119; β=0 ΔBCE 0.0013 ± 0.0006; age-shuffle ΔBCE 0.0115 ± 0.0020; β —; |β| 0.2155 ± 0.0423; λ RMSE 0.5262 ± 0.0065; λ correlation —
- S1 temporal_only: BCE 0.4928 ± 0.0009; AUROC 0.7301 ± 0.0014; AUPRC 0.5066 ± 0.0020; Surface RMSE 0.1999 ± 0.0064; CF-RMSE_age 0.1948 ± 0.0126; CF-RMSE_lag 0.1613 ± 0.0121; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0101 ± 0.0022; β —; |β| 0.0000 ± 0.0000; λ RMSE 0.5371 ± 0.0025; λ correlation —
- S2 age_temporal: BCE 0.4649 ± 0.0003; AUROC 0.7541 ± 0.0002; AUPRC 0.5660 ± 0.0002; Surface RMSE 0.2097 ± 0.0110; CF-RMSE_age 0.1733 ± 0.0151; CF-RMSE_lag 0.1787 ± 0.0098; β=0 ΔBCE 0.0131 ± 0.0016; age-shuffle ΔBCE 0.0359 ± 0.0031; β —; |β| 0.5215 ± 0.0354; λ RMSE 1.0845 ± 0.0053; λ correlation 0.9995 ± 0.0003
- S2 temporal_only: BCE 0.4701 ± 0.0004; AUROC 0.7501 ± 0.0004; AUPRC 0.5546 ± 0.0013; Surface RMSE 0.2371 ± 0.0097; CF-RMSE_age 0.2049 ± 0.0026; CF-RMSE_lag 0.1732 ± 0.0120; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0100 ± 0.0023; β —; |β| 0.0000 ± 0.0000; λ RMSE 1.1167 ± 0.0032; λ correlation —
- S3 age_temporal: BCE 0.4981 ± 0.0008; AUROC 0.7223 ± 0.0014; AUPRC 0.5086 ± 0.0015; Surface RMSE 0.1548 ± 0.0065; CF-RMSE_age 0.1311 ± 0.0118; CF-RMSE_lag 0.1168 ± 0.0068; β=0 ΔBCE 0.0080 ± 0.0010; age-shuffle ΔBCE 0.0201 ± 0.0023; β —; |β| 0.4670 ± 0.0480; λ RMSE 1.0961 ± 0.0014; λ correlation 0.9991 ± 0.0004
- S3 temporal_only: BCE 0.5036 ± 0.0008; AUROC 0.7174 ± 0.0011; AUPRC 0.4941 ± 0.0028; Surface RMSE 0.1671 ± 0.0108; CF-RMSE_age 0.1430 ± 0.0126; CF-RMSE_lag 0.1188 ± 0.0050; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0094 ± 0.0008; β —; |β| 0.0000 ± 0.0000; λ RMSE 1.1244 ± 0.0037; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0443 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal cf_rmse_age versus E01: Observed difference -0.0321 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference -0.0203 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal bce versus E01: Observed difference -0.0074 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auroc versus E01: Observed difference +0.0070 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auprc versus E01: Observed difference +0.0158 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference +0.0120 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference +0.0249 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S0 CF-RMSE_age versus E01: Observed difference -0.0093 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S0 |β| is 0.1322 ± 0.0112.

Supported interpretation is limited to the comparisons above.
Where the seed standard deviation is at least as large as the mean difference, whether the change helps is unresolved.

## E06_integrated_hazard

Hypothesis: An integrated hazard from event age to prediction age is a separate developmental parameterization.

Single change: Replace tau-multiplied lambda(a) by the integral of a 4-knot positive piecewise-linear hazard from event age to prediction age. Temporal-only is a constant hazard.

Parameters added: knot coefficients of rho(a). Parameters removed: softplus lambda(a) times tau.

Training: Identical to E01.

Artifacts: `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/E06_integrated_hazard`

- S0 age_temporal: BCE 0.5107 ± 0.0005; AUROC 0.6996 ± 0.0009; AUPRC 0.4551 ± 0.0015; Surface RMSE 0.2100 ± 0.0194; CF-RMSE_age 0.1963 ± 0.0231; CF-RMSE_lag 0.1915 ± 0.0218; β=0 ΔBCE 0.0079 ± 0.0006; age-shuffle ΔBCE 0.0081 ± 0.0019; β —; |β| 0.7428 ± 0.0080; λ RMSE —; λ correlation —
- S0 temporal_only: BCE 0.5151 ± 0.0007; AUROC 0.6937 ± 0.0010; AUPRC 0.4423 ± 0.0018; Surface RMSE 0.2100 ± 0.0201; CF-RMSE_age 0.1931 ± 0.0228; CF-RMSE_lag 0.1886 ± 0.0231; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0084 ± 0.0020; β —; |β| 0.0000 ± 0.0000; λ RMSE —; λ correlation —
- S1 age_temporal: BCE 0.5035 ± 0.0006; AUROC 0.7158 ± 0.0009; AUPRC 0.4766 ± 0.0015; Surface RMSE 0.2168 ± 0.0118; CF-RMSE_age 0.2237 ± 0.0183; CF-RMSE_lag 0.1833 ± 0.0150; β=0 ΔBCE 0.0072 ± 0.0006; age-shuffle ΔBCE 0.0094 ± 0.0012; β —; |β| 0.6933 ± 0.0155; λ RMSE —; λ correlation —
- S1 temporal_only: BCE 0.5068 ± 0.0006; AUROC 0.7111 ± 0.0011; AUPRC 0.4670 ± 0.0015; Surface RMSE 0.2149 ± 0.0118; CF-RMSE_age 0.2172 ± 0.0185; CF-RMSE_lag 0.1807 ± 0.0148; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0095 ± 0.0012; β —; |β| 0.0000 ± 0.0000; λ RMSE —; λ correlation —
- S2 age_temporal: BCE 0.4785 ± 0.0007; AUROC 0.7404 ± 0.0011; AUPRC 0.5374 ± 0.0008; Surface RMSE 0.2521 ± 0.0084; CF-RMSE_age 0.2173 ± 0.0151; CF-RMSE_lag 0.1958 ± 0.0113; β=0 ΔBCE 0.0038 ± 0.0004; age-shuffle ΔBCE 0.0106 ± 0.0015; β —; |β| 0.4982 ± 0.0133; λ RMSE —; λ correlation —
- S2 temporal_only: BCE 0.4808 ± 0.0007; AUROC 0.7374 ± 0.0011; AUPRC 0.5327 ± 0.0009; Surface RMSE 0.2524 ± 0.0083; CF-RMSE_age 0.2198 ± 0.0158; CF-RMSE_lag 0.1978 ± 0.0113; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0107 ± 0.0014; β —; |β| 0.0000 ± 0.0000; λ RMSE —; λ correlation —
- S3 age_temporal: BCE 0.5084 ± 0.0006; AUROC 0.7104 ± 0.0012; AUPRC 0.4814 ± 0.0015; Surface RMSE 0.1761 ± 0.0103; CF-RMSE_age 0.1602 ± 0.0088; CF-RMSE_lag 0.1334 ± 0.0050; β=0 ΔBCE 0.0041 ± 0.0002; age-shuffle ΔBCE 0.0096 ± 0.0024; β —; |β| 0.5154 ± 0.0265; λ RMSE —; λ correlation —
- S3 temporal_only: BCE 0.5104 ± 0.0005; AUROC 0.7073 ± 0.0013; AUPRC 0.4765 ± 0.0014; Surface RMSE 0.1740 ± 0.0092; CF-RMSE_age 0.1604 ± 0.0087; CF-RMSE_lag 0.1319 ± 0.0054; β=0 ΔBCE 0.0000 ± 0.0000; age-shuffle ΔBCE 0.0087 ± 0.0025; β —; |β| 0.0000 ± 0.0000; λ RMSE —; λ correlation —

### Comparison with E01 and the current DTR

- S2 age-temporal surface_rmse versus E01: Observed difference -0.0019 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal cf_rmse_age versus E01: Observed difference +0.0120 (unfavorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal cf_rmse_lag versus E01: Observed difference -0.0032 (favorable). The absolute difference does not exceed the larger seed standard deviation.
- S2 age-temporal bce versus E01: Observed difference +0.0062 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auroc versus E01: Observed difference -0.0067 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal auprc versus E01: Observed difference -0.0128 (unfavorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_beta0 versus E01: Observed difference +0.0027 (favorable). The absolute difference exceeds the larger seed standard deviation.
- S2 age-temporal delta_bce_age_shuffle versus E01: Observed difference -0.0004 (unfavorable). The absolute difference does not exceed the larger seed standard deviation.
- S0 CF-RMSE_age versus E01: Observed difference +0.0099 (unfavorable). The absolute difference does not exceed the larger seed standard deviation.
- S0 |β| is 0.7428 ± 0.0080.

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
| E01_direct | Remove content query, exp(u), content-dependent persistence, and the nonlinear history MLP. Direct linear readout. Weak lambda initialization. | 0.2540 ± 0.0090 | 0.0011 ± 0.0003 | 0.0110 ± 0.0014 | 0.1865 ± 0.0187 | 0.7471 ± 0.0012 | 0.5502 ± 0.0013 |
| E02_staged | No architectural change relative to E01. | 0.2532 ± 0.0073 | 0.0041 ± 0.0011 | 0.0158 ± 0.0015 | 0.2066 ± 0.0123 | 0.7537 ± 0.0007 | 0.5631 ± 0.0014 |
| E03_mass | Replace only the sum g·v aggregation by normalized composition plus log1p(evidence mass). | 0.2252 ± 0.0111 | 0.0016 ± 0.0002 | 0.0156 ± 0.0021 | 0.1402 ± 0.0123 | 0.7468 ± 0.0007 | 0.5414 ± 0.0013 |
| E04_channels | Add 4 content channels c=sigmoid(q_h^T v) with one shared lambda(a). Queries do not receive age or lag. | 0.2506 ± 0.0074 | 0.0017 ± 0.0003 | 0.0111 ± 0.0016 | 0.1802 ± 0.0110 | 0.7488 ± 0.0007 | 0.5531 ± 0.0006 |
| E05_mixture | K=3 content-only mixture of developmental rates. First model with content-specific timescales. | 0.2097 ± 0.0110 | 0.0131 ± 0.0016 | 0.0359 ± 0.0031 | 0.1771 ± 0.0174 | 0.7541 ± 0.0002 | 0.5660 ± 0.0002 |
| E06_integrated_hazard | Replace tau-multiplied lambda(a) by the integral of a 4-knot positive piecewise-linear hazard from event age to prediction age. Temporal-only is a constant hazard. | 0.2521 ± 0.0084 | 0.0038 ± 0.0004 | 0.0106 ± 0.0015 | 0.1963 ± 0.0231 | 0.7404 ± 0.0011 | 0.5374 ± 0.0008 |

## Figure paths

- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/figures/heatmap_S2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/figures/heatmap_S3.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/figures/lambda_age.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/figures/e06_rho_age.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/figures/mechanism_metrics_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/figures/predictive_metrics_s2.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/figures/negative_controls_s0_s1.png`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/architecture_ladder_metrics.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/architecture_ladder_summary.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/artifacts/synthetic_architecture_ladder/architecture_ladder_summary.json`

