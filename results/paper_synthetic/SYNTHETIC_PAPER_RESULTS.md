# Synthetic paper results

## 1. Benchmark cohort

Controlled Synthea cohort, 7007 patients, age at cutoff at least 2 years, from a 10000-patient pediatric source (Massachusetts, Synthea commit `aa0772fb5e92e48a776c51508c00eddc0d9d27ff`, 3275125 background events). Patient-disjoint split 4906 / 1070 / 1031, identical across S0–S3 and S5. The 10000-patient full-realism cohort was not used for these runs.

## 2. Scenario definitions

`z(a)=(a-9)/9`, `tau=log(1+Δt/7)`, `lambda=softplus(theta+beta z(a))`, relevance `exp(-lambda tau)`.

- S0: beta 0, no age×time interaction.
- S1: beta 0, age main effect 1.2, no age×time interaction.
- S2: beta -2.5, developmental interaction.
- S3: beta +2.5, reversed interaction.
- S5: beta -2.5 plus group thetas acute 1, intermediate 0, chronic -1.

## 3. Models compared

Shared full-label protocol (all 32 targets), seed 0: Previous DTR age×temporal, temporal-only, age-only, and no-age; Count + LightGBM; RETAIN; EHR-BERT; Med-BERT; BEHRT; CEHR-BERT.

Separate interaction-only protocol: Content-Persistence DTR, its temporal-only control on S0–S3, and Global-Persistence DTR on S5. These predictive numbers are not on the same label set as the shared table.

The directory `results/baselines/synthetic/dtr/` has no completed run. `_smoke` runs are excluded.

## 4. Predictive results

On the shared protocol, S2 AUPRC is highest for Previous DTR (age×temporal) (0.584), then Previous DTR (temporal-only) (0.582). Previous DTR (age×temporal) S2 AUPRC is 0.584.

S3 AUPRC is highest for CEHR-BERT (0.534), then Previous DTR (age×temporal) (0.532).

S5 AUPRC is highest for CEHR-BERT (0.601), then Previous DTR (temporal-only) (0.598). Previous DTR (age×temporal) S5 AUPRC is 0.598.

Content-Persistence DTR interaction-only S2 AUPRC is 0.857 versus its temporal-only control 0.831.

## 5. Mechanism-recovery results

Shared-protocol Surface RMSE on S2 is lowest for CEHR-BERT (0.1308). Previous DTR (age×temporal) is 0.2266, class NO_MECHANISM_RECOVERY. Saved beta is -0.1284 against true -2.5 (sign match True).

S3 lowest Surface RMSE is CEHR-BERT (0.1326).

No shared-protocol model is classified FUNCTIONAL_RECOVERY or PARTIAL_RECOVERY. The implemented cutoffs are Surface RMSE < 0.10 and < 0.25.

Content-Persistence DTR, on its own gate surface, has S2 beta_hat -2.3585, lambda correlation 0.9976, gate Surface RMSE 0.1251. S3 beta_hat is 2.0651 with sign match True.

## 6. S5 heterogeneous persistence

On the shared protocol, every model has persistence_order_correct = false. Previous DTR (age×temporal) mean group Surface RMSE is 0.2676 (acute 0.3071, intermediate 0.2863, chronic 0.2095).

Content-Persistence learned offsets are acute 1.109, intermediate 0.196, chronic -0.535. The acute > intermediate > chronic order holds. AUPRC versus Global-Persistence DTR differs by +0.0045 on interaction labels. Universal group surface RMSE was not saved for that pair. A matched temporal-only Content-Persistence S5 run was not saved.

## 7. Counterfactual findings

S0 false-interaction flags (CF-RMSE age > 0.05) : Previous DTR (age×temporal), Previous DTR (temporal-only), Previous DTR (age-only), Previous DTR (no age), Count + LightGBM, RETAIN, EHR-BERT, Med-BERT, BEHRT, CEHR-BERT.

Content-Persistence beta on S0/S1 is 0.0086 / 0.0751, and the beta=0 ablation changes BCE by 0.000007 / 0.000300.

## 8. Key findings

See `key_findings.md`. The predictive gap between Previous DTR (age×temporal) and Previous DTR (temporal-only) on the shared protocol is small. Content-Persistence recovers the sign of beta on S2 and S3 inside its interaction-only runs. Shared-protocol surface error stays in the NO_MECHANISM_RECOVERY range for every model.

## 9. Candidate paper tables

- `table_predictive.md` / `.tex` for full-label AUROC and AUPRC.
- `table_mechanism.md` / `.tex` for S2/S3 counterfactual error.
- `benchmark_summary.md` for the scenario definitions.
- Appendix: `table_s5_persistence`, `table_negative_controls`, `dtr_deltas.csv`, `dtr_mechanism_summary.md`.

## 10. Candidate paper figures

- `figures/fig1_benchmark_mechanism`
- `figures/fig2_predictive_auprc` (S2, S3, S5; five scenarios were too crowded)
- `figures/fig3_surface_rmse`
- `figures/fig4_s2_counterfactual_surface` (oracle, Previous DTR, lowest-S2-Surface-RMSE external baseline)
- `figures/fig5_s5_persistence_curves`
- `figures/fig6_negative_controls`

## 11. Caveats

- One seed (0). Bootstrap intervals are unavailable from saved artifacts.
- Content-Persistence predictive metrics use interaction labels only.
- Gate Surface RMSE and prediction-probability Surface RMSE are not interchangeable.
- Count + LightGBM S2 is marked `smoke=true` in its result file.
- Previous DTR beta and lambda were read from `best_checkpoint.pt`. Age-shuffle and beta=0 ablations were not saved for that arm.
- S5 `checkpoint.pt` bytes differ from `best_checkpoint.pt`, and the tensors match.

SYNTHETIC BENCHMARK RESULTS CONSOLIDATED

## Generated files

- `SYNTHETIC_PAPER_RESULTS.md`
- `all_results.csv`
- `all_results.json`
- `audit_report.md`
- `benchmark_summary.json`
- `benchmark_summary.md`
- `bootstrap_comparisons.csv`
- `dtr_deltas.csv`
- `dtr_mechanism_summary.json`
- `dtr_mechanism_summary.md`
- `figures/fig1_benchmark_mechanism.png`
- `figures/fig1_benchmark_mechanism.svg`
- `figures/fig2_predictive_auprc.png`
- `figures/fig2_predictive_auprc.svg`
- `figures/fig3_surface_rmse.png`
- `figures/fig3_surface_rmse.svg`
- `figures/fig4_s2_counterfactual_surface.png`
- `figures/fig4_s2_counterfactual_surface.svg`
- `figures/fig5_s5_persistence_curves.png`
- `figures/fig5_s5_persistence_curves.svg`
- `figures/fig6_negative_controls.png`
- `figures/fig6_negative_controls.svg`
- `key_findings.md`
- `model_name_map.json`
- `paper_selection.md`
- `run_manifest.csv`
- `table_mechanism.csv`
- `table_mechanism.md`
- `table_mechanism.tex`
- `table_negative_controls.csv`
- `table_negative_controls.md`
- `table_predictive.csv`
- `table_predictive.md`
- `table_predictive.tex`
- `table_s5_persistence.csv`
- `table_s5_persistence.md`
- `table_s5_persistence.tex`
