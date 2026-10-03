# Content-Persistence DTR `_new` synthetic results (seed 0)

Architecture: mass-preserving Content-Persistence DTR. Legacy `dtr_*` dirs untouched.

## Per-arm metrics

| Scenario | Arm | AUROC | AUPRC | BCE | θ₀ | β | β=0 ΔBCE | age-shuffle ΔBCE | CF-RMSE age | CF-RMSE lag | Surface RMSE | Class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S0 | dtr_temporal_only_new | 0.7334 | 0.5297 | 0.4807 | -0.0582 | 0.0000 | — | — | 0.0841 | 0.1153 | 0.1498 | PARTIAL_RECOVERY |
| S0 | dtr_age_temporal_new | 0.7304 | 0.5259 | 0.4825 | -0.0590 | 0.0019 | 0.0000 | 0.0078 | 0.0986 | 0.1201 | 0.1387 | PARTIAL_RECOVERY |
| S1 | dtr_temporal_only_new | 0.7399 | 0.5370 | 0.4814 | -0.0929 | 0.0000 | — | — | 0.1406 | 0.1206 | 0.1687 | NO_MECHANISM_RECOVERY |
| S1 | dtr_age_temporal_new | 0.7419 | 0.5387 | 0.4804 | -0.0794 | 0.0265 | 0.0000 | 0.0100 | 0.1328 | 0.1201 | 0.1701 | NO_MECHANISM_RECOVERY |
| S2 | dtr_temporal_only_new | 0.7577 | 0.5690 | 0.4625 | -0.1093 | 0.0000 | — | — | 0.1774 | 0.1841 | 0.2361 | NO_MECHANISM_RECOVERY |
| S2 | dtr_age_temporal_new | 0.7606 | 0.5765 | 0.4586 | -0.0882 | -0.6151 | 0.0069 | 0.0201 | 0.1422 | 0.1656 | 0.2013 | NO_MECHANISM_RECOVERY |
| S3 | dtr_temporal_only_new | 0.7300 | 0.5216 | 0.4917 | -0.1000 | 0.0000 | — | — | 0.1317 | 0.0887 | 0.1474 | NO_MECHANISM_RECOVERY |
| S3 | dtr_age_temporal_new | 0.7316 | 0.5293 | 0.4889 | -0.1052 | 0.5880 | 0.0059 | 0.0176 | 0.1317 | 0.0905 | 0.1463 | NO_MECHANISM_RECOVERY |
| S5 | dtr_temporal_only_new | 0.7677 | 0.5828 | 0.4625 | -0.0781 | 0.0000 | — | — | 0.2808 | 0.2513 | 0.2186 | PARTIAL_HETEROGENEOUS_PERSISTENCE_RECOVERY |
| S5 | dtr_age_temporal_new | 0.7716 | 0.5871 | 0.4588 | -0.0737 | -0.4796 | 0.0045 | 0.0215 | 0.2538 | 0.2190 | 0.1826 | PARTIAL_HETEROGENEOUS_PERSISTENCE_RECOVERY |

## age_temporal_new − temporal_only_new

| Scenario | ΔAUROC | ΔAUPRC | ΔBCE | ΔCF-age | ΔCF-lag | ΔSurface | β̂ (AT) | β=0 ΔBCE | shuffle ΔBCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | -0.0030 | -0.0038 | 0.0019 | 0.0145 | 0.0048 | -0.0111 | 0.0019 | 0.0000 | 0.0078 |
| S1 | 0.0020 | 0.0017 | -0.0010 | -0.0077 | -0.0005 | 0.0013 | 0.0265 | 0.0000 | 0.0100 |
| S2 | 0.0029 | 0.0076 | -0.0039 | -0.0351 | -0.0185 | -0.0348 | -0.6151 | 0.0069 | 0.0201 |
| S3 | 0.0016 | 0.0076 | -0.0028 | 0.0001 | 0.0018 | -0.0011 | 0.5880 | 0.0059 | 0.0176 |
| S5 | 0.0038 | 0.0043 | -0.0037 | -0.0270 | -0.0323 | -0.0359 | -0.4796 | 0.0045 | 0.0215 |

## S5 persistence (age_temporal_new)

- age_temporal_new: acute=0.1867, intermediate=0.1989, chronic=0.1491, mean=0.1782, order_ok=True
- temporal_only_new: acute=0.2041, intermediate=0.2020, chronic=0.1805, mean=0.1955, order_ok=True
- Δ mean Surface RMSE (AT−TO) = -0.0173
