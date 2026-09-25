# Predictive comparison (shared full-label protocol)

Rows are models trained on all 32 targets. AUPRC is the primary metric. AUROC is secondary.
Content-Persistence DTR is omitted here because its saved metrics use interaction labels only.

| Model | S0 AUROC | S0 AUPRC | S1 AUROC | S1 AUPRC | S2 AUROC | S2 AUPRC | S3 AUROC | S3 AUPRC | S5 AUROC | S5 AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| Previous DTR (age×temporal) | 0.734 | 0.520 | 0.749 | 0.534 | 0.765 | 0.584 | 0.735 | 0.532 | 0.779 | 0.598 |
| Previous DTR (temporal-only) | 0.735 | 0.522 | 0.749 | 0.534 | 0.764 | 0.582 | 0.734 | 0.530 | 0.779 | 0.598 |
| Previous DTR (age-only) | 0.685 | 0.422 | 0.708 | 0.457 | 0.739 | 0.544 | 0.711 | 0.497 | 0.748 | 0.541 |
| Previous DTR (no age) | 0.684 | 0.424 | 0.709 | 0.460 | 0.736 | 0.537 | 0.707 | 0.492 | 0.749 | 0.542 |
| Count + LightGBM | 0.676 | 0.404 | 0.695 | 0.440 | 0.729 | 0.520 | 0.706 | 0.488 | 0.737 | 0.526 |
| RETAIN | 0.632 | 0.359 | 0.633 | 0.358 | 0.675 | 0.435 | 0.607 | 0.357 | 0.671 | 0.420 |
| EHR-BERT | 0.678 | 0.413 | 0.697 | 0.443 | 0.731 | 0.524 | 0.702 | 0.482 | 0.738 | 0.520 |
| Med-BERT | 0.673 | 0.409 | 0.695 | 0.443 | 0.727 | 0.515 | 0.700 | 0.479 | 0.735 | 0.521 |
| BEHRT | 0.646 | 0.372 | 0.696 | 0.438 | 0.723 | 0.508 | 0.701 | 0.479 | 0.735 | 0.522 |
| CEHR-BERT | 0.730 | 0.520 | 0.748 | 0.539 | 0.764 | 0.582 | 0.738 | 0.534 | 0.779 | 0.601 |

## Best and second-best AUPRC

- S0: best Previous DTR (temporal-only) (0.522); second CEHR-BERT (0.520)
- S1: best CEHR-BERT (0.539); second Previous DTR (temporal-only) (0.534)
- S2: best Previous DTR (age×temporal) (0.584); second Previous DTR (temporal-only) (0.582)
- S3: best CEHR-BERT (0.534); second Previous DTR (age×temporal) (0.532)
- S5: best CEHR-BERT (0.601); second Previous DTR (temporal-only) (0.598)

## Best and second-best AUROC

- S0: best Previous DTR (temporal-only) (0.735); second Previous DTR (age×temporal) (0.734)
- S1: best Previous DTR (temporal-only) (0.749); second Previous DTR (age×temporal) (0.749)
- S2: best Previous DTR (age×temporal) (0.765); second Previous DTR (temporal-only) (0.764)
- S3: best CEHR-BERT (0.738); second Previous DTR (age×temporal) (0.735)
- S5: best Previous DTR (temporal-only) (0.779); second Previous DTR (age×temporal) (0.779)

Count + LightGBM on S2 has `smoke=true` in result.json. The loader is not subsetted; the value is the saved test metric.

