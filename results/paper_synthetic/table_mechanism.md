# Mechanism recovery (universal counterfactual metrics)

Primary comparison is Surface RMSE, CF-RMSE age, and CF-RMSE lag on the shared prediction-probability grid.
Lower is better. Classification thresholds are the implemented values: Surface RMSE < 0.10 functional, < 0.25 partial.
beta and lambda are reported only for previous DTR arms, from saved checkpoint parameters. Black-box baselines are NA.

| Model | S2 CF-RMSE age ↓ | S2 CF-RMSE lag ↓ | S2 Surface RMSE ↓ | S2 class | S3 CF-RMSE age ↓ | S3 CF-RMSE lag ↓ | S3 Surface RMSE ↓ | S3 class |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Previous DTR (age×temporal) | 0.1967 | 0.1651 | 0.2266 | NO_MECHANISM_RECOVERY | 0.1440 | 0.1071 | 0.1757 | NO_MECHANISM_RECOVERY |
| Previous DTR (temporal-only) | 0.1780 | 0.1671 | 0.2251 | NO_MECHANISM_RECOVERY | 0.1492 | 0.1072 | 0.1860 | NO_MECHANISM_RECOVERY |
| Previous DTR (age-only) | 0.1870 | 0.1744 | 0.2127 | NO_MECHANISM_RECOVERY | 0.1404 | 0.1110 | 0.1524 | NO_MECHANISM_RECOVERY |
| Previous DTR (no age) | 0.1947 | 0.1616 | 0.2210 | NO_MECHANISM_RECOVERY | 0.1615 | 0.1137 | 0.1740 | NO_MECHANISM_RECOVERY |
| Count + LightGBM | 0.1926 | 0.2197 | 0.2071 | NO_MECHANISM_RECOVERY | 0.1478 | 0.1265 | 0.1587 | NO_MECHANISM_RECOVERY |
| RETAIN | 0.2226 | 0.1931 | 0.2363 | NO_MECHANISM_RECOVERY | 0.2228 | 0.2004 | 0.2173 | NO_MECHANISM_RECOVERY |
| EHR-BERT | 0.2109 | 0.1947 | 0.2339 | NO_MECHANISM_RECOVERY | 0.1778 | 0.1101 | 0.1740 | NO_MECHANISM_RECOVERY |
| Med-BERT | 0.1941 | 0.1962 | 0.2344 | NO_MECHANISM_RECOVERY | 0.1887 | 0.1240 | 0.1831 | NO_MECHANISM_RECOVERY |
| BEHRT | 0.1894 | 0.1860 | 0.2173 | NO_MECHANISM_RECOVERY | 0.1513 | 0.1188 | 0.1615 | NO_MECHANISM_RECOVERY |
| CEHR-BERT | 0.0983 | 0.1108 | 0.1308 | NO_MECHANISM_RECOVERY | 0.1079 | 0.0891 | 0.1326 | NO_MECHANISM_RECOVERY |

## Lowest and second-lowest Surface RMSE

- S2: lowest CEHR-BERT (0.1308); second Count + LightGBM (0.2071)
- S3: lowest CEHR-BERT (0.1326); second Previous DTR (age-only) (0.1524)
