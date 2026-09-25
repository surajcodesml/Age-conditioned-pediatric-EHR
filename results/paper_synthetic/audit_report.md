# Audit

1. Same patient splits for each scenario: **PASS**. SHA-256 of `splits.json` is identical for S0, S1, S2, S3, and S5 (750e8b767774). Sizes are train 4906, val 1070, test 1031.
2. Test metrics from the intended best checkpoint: **PASS**. Neural training loads the minimum-validation-BCE state before the test evaluation and writes it to `best_checkpoint.pt`. For S5, `checkpoint.pt` has a different file hash from `best_checkpoint.pt`, and the parameter tensors match (max absolute difference 0) for every neural model checked. Count + LightGBM S2 has `smoke=true` in `result.json`; the LightGBM path still iterates the full train, validation, and test loaders.
3. Model names consistent: **PASS**. See `model_name_map.json`. Content-Persistence DTR and Previous DTR (age×temporal) are different runs.
4. S0–S3 definitions unchanged: **PASS**. Saved `meta.json` beta/theta0 match `SCENARIO_SPECS` (S0/S1 beta 0, S2 beta -2.5, S3 beta +2.5, theta0 0).
5. S5 uses the heterogeneous-persistence generator: **PASS**. `meta.json` scenario S5, beta -2.5, theta0 0. Group thetas in `ground_truth.compute_target_logits` are acute 1, intermediate 0, chronic -1 for `SYN_SIGNAL_A`–`L`.
6. Baselines do not receive oracle/signal metadata: **PASS** on the training path. `model_batch` keeps only model-input keys and the runner wraps loaders with that strip. This audit did not re-execute training.
7. Counterfactual grids identical across models: **PASS**. Every saved `cf_report` records the same thresholds ['{"functional": 0.1, "partial": 0.25, "s0_max_cf_age": 0.05}']. Grids are the shared constants CF ages (2, 5, 9, 13, 17), CF lags (7, 30, 90, 180, 365, 730), surface ages 0–18, surface lags (0, 7, 30, 90, 180, 365, 730).
8. Directions: **PASS**. Tables mark AUROC and AUPRC as higher-better and RMSE columns as lower-better.
9. No validation metric reported as test: **PASS**. Shared AUROC/AUPRC/BCE are `test_metrics`. Content-Persistence AUROC/AUPRC/BCE are the `test` block written after the best weights are loaded.
10. Missing metrics marked rather than inferred: **PASS**. Universal CF fields for Content-Persistence are NA. Beta and lambda are NA for non-DTR baselines. S5 temporal-only Content-Persistence deltas are NA. Bootstrap is marked unavailable. Tensor mismatches: none.

Caveat: Content-Persistence gate Surface RMSE and the shared prediction-probability Surface RMSE are different quantities. They are stored with `surface_definition` and are not ranked against each other.
