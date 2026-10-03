# Figure analysis report (ICLR DTR)

Primary synthetic protocol: **32-target S2**. Models: Content-Persistence `dtr_age_temporal_new` / `dtr_temporal_only_new`.

---

## Figure: `synthetic_surface_comparison_32target`

- **Data source:** Forward-pass age×lag surfaces from `best_checkpoint.pt` for CEHR-BERT, Temporal-only DTR, DTR; oracle from controlled S2 generator. Grids: ages 0–18, lags (0,7,30,90,180,365,730). Same CF template protocol as `baselines/synthetic/counterfactual_eval.py`. Surface RMSE titles use stored result.json values.
- **Plotted:** Mean predicted probability over 32 targets; second row absolute residual vs oracle.
- **Quantitative:** DTR Surface RMSE=0.201; Temporal-only=0.236; CEHR-BERT=0.131.
- **Supported:** Visual mismatch between predictive baselines and oracle age×lag structure; DTR improves on temporal-only but does not dominate CEHR-BERT on Surface RMSE.
- **Not supported:** Claiming DTR uniquely recovers the oracle surface; CEHR-BERT has lower Surface RMSE on this protocol.
- **Caption:** Predicted age×lag response surfaces on the primary 32-target S2 benchmark (shared color scales). DTR reduces Surface RMSE relative to temporal-only but CEHR-BERT remains closest to the oracle surface among compared models.
- **Recommendation:** Main paper.

## Figure: `synthetic_surface_comparison_s2_s3`

- **Data source:** Same protocol for S2 and S3.
- **Quantitative:** S3 DTR Surface RMSE=0.146 vs temporal-only 0.147 (near-zero gain).
- **Supported:** Cross-scenario visual comparison.
- **Not supported:** Strong S3 mechanism advantage for DTR over temporal-only.
- **Caption:** Age×lag surfaces for S2 (β_true=-2.5) and S3 (sign-flipped β) under the 32-target protocol.
- **Recommendation:** Appendix.

## Figure: `synthetic_functional_recovery`

- **Data source:** `dtr_functional_diagnostics` / stored mechanism fields.
- **Quantitative:** β̂ S0→S3 = S0:0.002, S1:0.027, S2:-0.615, S3:0.588; S2 β=0 ΔBCE=0.0069, shuffle ΔBCE=0.0201.
- **Supported:** β̂ near 0 on S0 and moves toward the signed interaction on S2/S3; functional ΔBCE is small on S0/S1 and larger on S2/S3, but absolute β=0 ΔBCE remains modest (≤0.01 on S2).
- **Not supported:** Strong claim of functional mechanism recovery solely from β̂ sign matching; intervention ΔBCE magnitudes are small relative to canonical factorized DTR runs with ΔBCE_β0≈0.09.
- **Caption:** Across S0–S3, DTR’s β̂ tracks the presence/sign of the planted interaction while age-shuffle and β=0 ΔBCE remain near zero without an interaction and increase when one exists—yet functional reliance is partial.
- **Recommendation:** Main paper (with cautious wording).

## Figure: `s2_prediction_vs_surface`

- **Data source:** Primary S2 result.json for six models; capacity from `model_capacity.csv`.
- **Plotted:** AUPRC vs Surface RMSE with parameter annotations.
- **Supported:** Predictive ranking ≠ counterfactual surface ranking (CEHR-BERT best surface; DTR/CEHR similar AUPRC).
- **Not supported:** Equating AUPRC gains with mechanism fidelity.
- **Caption:** On the 32-target S2 benchmark, models with similar predictive AUPRC can differ substantially in age×lag Surface RMSE, separating prediction from counterfactual fidelity.
- **Recommendation:** Main paper.

## Figure: `dtr_gain_across_scenarios`

- **Data source:** CF metrics from `dtr_*_new` result.json.
- **Plotted:** Δ = temporal-only error − DTR error for CF-age/lag/Surface.
- **Quantitative:** S2 ΔSurface=0.035; S0 ΔSurface=0.011; S3 ΔSurface=0.001.
- **Supported:** Clearest CF gains on S2/S5; near-zero or mixed on S0/S1/S3.
- **Not supported:** Uniform improvement across all scenarios.
- **Caption:** Age-temporal DTR improves counterfactual errors over temporal-only primarily when an age×lag interaction is planted (S2/S5).
- **Recommendation:** Main paper or appendix depending on space.

## Figure: `s5_persistence_recovery`

- **Data source:** S5 class-specific Surface RMSE in result.json.
- **Quantitative:** DTR acute/inter/chronic=0.187/0.199/0.149; order_ok=True.
- **Supported:** DTR lowers class-wise Surface RMSE vs temporal-only; persistence order flag true for both arms under the black-box surface test.
- **Not supported:** Claiming unique recovery of heterogeneous persistence parameters (global β model).
- **Caption:** On S5, age-temporal DTR improves class-specific Surface RMSE relative to temporal-only while preserving the black-box persistence ordering check.
- **Recommendation:** Appendix (S5).

## Figure: `parameter_vs_functional_recovery`

- **Data source:** Existing Transformer S2 metrics (`arch_S2_age_temporal_.../metrics.json`), canonical factorized DTR metrics, and baseline `_new` diagnostics. No fabricated points.
- **Supported:** High λ-correlation can coexist with weak β=0/age-shuffle ΔBCE (Transformer and baseline `_new`); canonical factorized run shows both high corr and large functional ΔBCE.
- **Not supported:** Equating parameter-curve recovery with functional reliance.
- **Caption:** Lambda-curve agreement with the oracle does not imply functional reliance on the age×lag pathway; β=0 and age-shuffle ΔBCE separate apparent parameter recovery from mechanism use.
- **Recommendation:** Main paper (methods/results distinction) or appendix.

## NCH figures

- **Skipped.** See `nch_final_analysis.md`. Leakage-corrected matched `dtr_age_temporal_new` + finished `dtr_temporal_only_new` results are not present.

## Capacity

- See `model_capacity.csv`. Counting convention: `baselines.common.capacity_report.count_parameters` / stored `model_card`. **No frozen text/code embeddings** in these synthetic baselines; embedding parameters are included in trainable counts.

## S0 caveat (do not reinterpret as false interaction)

- (e) Ruled out as β-pathway false interaction: β̂≈0.0019, β_true=0, β=0 ΔBCE≈0.
- (a) Direct age main-effect head: oracle age-std on interaction targets is 0.0 but model is 0.042; zeroing `f_age` reduces interaction CF-RMSE-age 0.094→0.065 (while worsening age_only).
- (c) All-target averaging includes age_only targets with genuine oracle age effects.
- (d) CF-RMSE-age does not isolate the age×lag gate.
- (b) Implicit age-in-content is secondary; residual CF after zeroing `f_age` remains.
- Full write-up: `dtr_functional_diagnostics.md`.

## Overall wording guidance

Prefer: DTR shows **partial** alignment with the planted age×lag pathway (signed β̂, modest intervention ΔBCE, improved CF errors vs temporal-only on S2/S5). Avoid: unqualified “mechanism recovery” for the baseline `_new` checkpoint, given small β=0 ΔBCE and CEHR-BERT’s superior Surface RMSE.
