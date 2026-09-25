# Content-Persistence DTR mechanism summary

These numbers come from `canonical_regression` metrics.json. Labels are interaction targets only.

## S0 / S1
- S0 learned beta 0.0086 (true 0). Age-shuffle ΔBCE 0.0020. beta=0 ΔBCE 0.0000.
- S0 AUPRC 0.705 versus temporal-only 0.706 (delta -0.0009).
- S1 learned beta 0.0751 (true 0). Age-shuffle ΔBCE 0.0148. beta=0 ΔBCE 0.0003.
- S1 AUPRC 0.682 versus temporal-only 0.687 (delta -0.0049).
- On both negative controls, beta stays near 0 and the beta=0 ablation changes BCE by less than 0.001.

## S2
- beta_true -2.5, beta_hat -2.3585, sign match True.
- lambda correlation 0.9976, lambda RMSE 0.4301, gate Surface RMSE 0.1251.
- age-shuffle ΔBCE 0.3083; beta=0 ΔBCE 0.1284.
- AUPRC 0.857 versus temporal-only 0.831 (delta 0.0259).

## S3
- beta_true +2.5, beta_hat 2.0651, sign match True.
- lambda correlation 0.9983, lambda RMSE 0.5667, gate Surface RMSE 0.1414.
- age-shuffle ΔBCE 0.1384; beta=0 ΔBCE 0.0786.
- AUPRC 0.746 versus temporal-only 0.710 (delta 0.0361).

## S5
- learned developmental beta -2.3558 (true -2.5), sign match True.
- learned persistence offsets acute/intermediate/chronic: 1.1088 / 0.1955 / -0.5349.
- offset order acute > intermediate > chronic: True.
- Group-specific universal surface RMSE was not saved for this run.
- AUPRC 0.817 versus Global-Persistence DTR 0.812 (delta 0.0045).
- No matched temporal-only Content-Persistence checkpoint was saved for S5.

## Previous DTR on the shared full-label benchmark
These arms are the transformer global-kernel DTR trained with the baselines. They are not Content-Persistence DTR.

- S0: beta_hat -0.0131, sign match True, AUPRC 0.520 vs temporal-only 0.522 (delta -0.0024), Surface RMSE 0.1377, class PARTIAL_RECOVERY.
- S1: beta_hat 0.0221, sign match True, AUPRC 0.534 vs temporal-only 0.534 (delta -0.0000), Surface RMSE 0.1943, class NO_MECHANISM_RECOVERY.
- S2: beta_hat -0.1284, sign match True, AUPRC 0.584 vs temporal-only 0.582 (delta 0.0010), Surface RMSE 0.2266, class NO_MECHANISM_RECOVERY.
- S3: beta_hat 0.0417, sign match True, AUPRC 0.532 vs temporal-only 0.530 (delta 0.0023), Surface RMSE 0.1757, class NO_MECHANISM_RECOVERY.
- S5: beta_hat -0.1817, sign match True, AUPRC 0.598 vs temporal-only 0.598 (delta -0.0001), Surface RMSE 0.1921, class NO_HETEROGENEOUS_PERSISTENCE_RECOVERY.
