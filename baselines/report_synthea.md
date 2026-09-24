# Synthetic Benchmark Results

This document contains the automated evaluation results for all baselines
across the Synthea S0–S3 scenarios.

## 1. Predictive Performance (Test AUROC)

| Model | S0 (No Interaction) | S1 (Age-only) | S2 (Age×Temporal) | S3 (Non-linear) |
|-------|---------------------|---------------|-------------------|-----------------|
| Count + LightGBM | 0.676 | 0.695 | 0.729 | 0.706 |
| RETAIN | 0.632 | 0.633 | 0.675 | 0.607 |
| EHR-BERT | 0.678 | 0.697 | 0.731 | 0.702 |
| Med-BERT | 0.673 | 0.695 | 0.727 | 0.700 |
| BEHRT | 0.646 | 0.696 | 0.723 | 0.701 |
| CEHR-BERT | 0.730 | 0.748 | 0.764 | 0.738 |
| DTR (Temporal) | 0.684 | 0.709 | 0.736 | 0.707 |
| DTR (Age) | 0.685 | 0.708 | 0.739 | 0.711 |
| DTR (Age×Temporal) | 0.734 | 0.749 | 0.765 | 0.735 |

## 2. Mechanism Recovery (S2: Age×Temporal)

Evaluation of how accurately models capture the true data-generating mechanism
(the temporal interaction surface λ(a) over age).

| Model | CF-RMSE (Age) | Surface RMSE | Recovery Classification |
|-------|---------------|--------------|--------------------------|
| Count + LightGBM | 0.2286 | 0.2259 | NO MECHANISM RECOVERY |
| RETAIN | 0.1942 | 0.1946 | NO MECHANISM RECOVERY |
| EHR-BERT | 0.2206 | 0.2207 | NO MECHANISM RECOVERY |
| Med-BERT | 0.2171 | 0.2172 | NO MECHANISM RECOVERY |
| BEHRT | 0.2423 | 0.2421 | NO MECHANISM RECOVERY |
| CEHR-BERT | 0.2796 | 0.2639 | NO MECHANISM RECOVERY |
| DTR (Temporal) | 0.2322 | 0.2324 | NO MECHANISM RECOVERY |
| DTR (Age) | 0.2302 | 0.2303 | NO MECHANISM RECOVERY |
| DTR (Age×Temporal) | 0.2850 | 0.2382 | NO MECHANISM RECOVERY |

## Summary

The DTR (Age×Temporal) model should uniquely demonstrate `FUNCTIONAL RECOVERY`
of the true mechanism, while maintaining top-tier predictive performance.