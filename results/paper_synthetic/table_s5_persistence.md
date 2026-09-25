# S5 heterogeneous persistence (shared full-label protocol)

Surface RMSE columns are prediction-probability errors on the group-controlled grids.
Persistence ordering uses the implemented surface decay-proxy rule (acute > intermediate > chronic).

| Model | AUROC ↑ | AUPRC ↑ | CF-RMSE age ↓ | CF-RMSE lag ↓ | Acute ↓ | Intermediate ↓ | Chronic ↓ | Mean ↓ | Order correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Previous DTR (age×temporal) | 0.779 | 0.598 | 0.1679 | 0.0973 | 0.3071 | 0.2863 | 0.2095 | 0.2676 | no |
| Previous DTR (temporal-only) | 0.779 | 0.598 | 0.1825 | 0.0951 | 0.2563 | 0.2531 | 0.2602 | 0.2565 | no |
| Previous DTR (age-only) | 0.748 | 0.541 | 0.2208 | 0.1245 | 0.3600 | 0.2604 | 0.3398 | 0.3201 | no |
| Previous DTR (no age) | 0.749 | 0.542 | 0.2234 | 0.1115 | 0.2840 | 0.2548 | 0.2015 | 0.2468 | no |
| Count + LightGBM | 0.737 | 0.526 | 0.1372 | 0.1071 | 0.1774 | 0.1484 | 0.1031 | 0.1430 | no |
| RETAIN | 0.671 | 0.420 | 0.2444 | 0.1708 | 0.3074 | 0.2141 | 0.2988 | 0.2735 | no |
| EHR-BERT | 0.738 | 0.520 | 0.2263 | 0.1308 | 0.1892 | 0.2020 | 0.1612 | 0.1841 | no |
| Med-BERT | 0.735 | 0.521 | 0.2240 | 0.0989 | 0.2112 | 0.1997 | 0.1627 | 0.1912 | no |
| BEHRT | 0.735 | 0.522 | 0.1448 | 0.0896 | 0.1933 | 0.1586 | 0.1179 | 0.1566 | no |
| CEHR-BERT | 0.779 | 0.601 | 0.0673 | 0.1116 | 0.1509 | 0.1351 | 0.1406 | 0.1422 | yes |

Best AUPRC: CEHR-BERT (0.601); second Previous DTR (temporal-only) (0.598).
Lowest mean persistence Surface RMSE: CEHR-BERT (0.1422); second Count + LightGBM (0.1430).

