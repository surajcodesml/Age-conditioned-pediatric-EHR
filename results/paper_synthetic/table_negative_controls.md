# Negative controls

false_interaction_flag uses the implemented S0 rule: CF-RMSE age > 0.05 (`S0_MAX_CF_AGE_RMSE`).
No separate S1 threshold is implemented. S1 values are reported without a new cutoff.

| Model | S0 CF-RMSE age | S0 Surface RMSE | S1 CF-RMSE age | S1 Surface RMSE | false interaction (S0 rule) |
|---|---:|---:|---:|---:|---|
| Previous DTR (age×temporal) | 0.0769 | 0.1377 | 0.1173 | 0.1943 | true |
| Previous DTR (temporal-only) | 0.0740 | 0.1415 | 0.1172 | 0.1943 | true |
| Previous DTR (age-only) | 0.1079 | 0.1245 | 0.1833 | 0.1870 | true |
| Previous DTR (no age) | 0.1271 | 0.1284 | 0.1826 | 0.1929 | true |
| Count + LightGBM | 0.1332 | 0.1271 | 0.1827 | 0.1863 | true |
| RETAIN | 0.1504 | 0.1562 | 0.2259 | 0.2177 | true |
| EHR-BERT | 0.1297 | 0.1352 | 0.1870 | 0.2061 | true |
| Med-BERT | 0.1571 | 0.1520 | 0.2076 | 0.1942 | true |
| BEHRT | 0.1481 | 0.1419 | 0.1775 | 0.1975 | true |
| CEHR-BERT | 0.0597 | 0.0754 | 0.0995 | 0.1110 | true |
