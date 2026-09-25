# Main paper vs appendix

## Main paper

- One short cohort/scenario table from `benchmark_summary.md`: controlled n=7007, the split, z(a), tau, and S0–S3/S5 definitions including S5 group thetas.
- One comparison table: shared-protocol AUPRC (primary) with Surface RMSE for S2 and S3. Use `table_predictive` and `table_mechanism`. Do not place Content-Persistence interaction-only AUPRC in that table.
- Figure 3 (Surface RMSE on S2 and S3) or Figure 6 (Surface RMSE across S0–S3). Figure 1 can sit with the benchmark definition.

## Appendix

- Full S0–S5 AUROC/AUPRC/BCE table.
- CF-RMSE age, CF-RMSE lag, and mechanism class for every model.
- S5 acute/intermediate/chronic surface errors and the ordering flag.
- Negative-control CF-RMSE values and the 0.05 S0 rule.
- Content-Persistence DTR beta, lambda, ablations, and the S5 offset ordering, labeled interaction-only.
- Previous DTR checkpoint beta and the age×temporal versus temporal-only deltas on the shared protocol.
- Global-Persistence versus Content-Persistence S5 deltas.
- Count + LightGBM S2 `smoke=true` flag.

Do not put architecture-search, multi-horizon, or multi-query experiments in the main benchmark. Those runs are not part of this completed comparison.
