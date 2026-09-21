# NCH Stage-2 preprocessing v2 report

Versioned under `artifacts/nch_stage2/v2/` (v1 audit untouched).

## Answers

1. **Procedure mismatch:** Most OOV is CPT/HCPCS Level I ({'CPT_HCPCS_LEVEL_I': 255459, 'ICD10_PCS': 19198, 'ICD9_PCS': 10847, 'NCH_LOCAL_SHX': 686}); ICD-PCS already maps to CCS_*. Exact event-weighted coverage 15.1% → format-normalized 15.1%.
2. **Grouping recovery:** CPT chapter grouping (not AMA CCS Svcs/Proc — unavailable) covers 99.7% events as CPT_CHAPTER_* (new namespace). Stage-1-compatible normalized recovery remains 15.1%.
3. **Stage-1 CCS compatible?** NO for CPT chapters vs CCS_*: Stage-1 CCS_* are ICD procedure CCS. MIMIC/NCH share 4 CPT chapters at coarse level.
4. **Medication ladder:** see `medication_mapping/medication_ladder_report.json`
5. **Med OOV pediatric vs formulation:** `{"unique_oov_cuis": 1667, "formulation_mismatch_cuis_ingredient_in_stage1": 2, "genuinely_absent_or_unresolvable_cuis": 1665, "tty_among_oov": {"missing": 1135941, "Semantic Clinical Drug": 726001, "Precise Ingredient": 163751, "Ingredient": 40041, "Generic Pack": 637, "Multiple Ingredients": 15}}`
6. **Missing RxCUI:** `{"method": "exact normalized name match to rxcui_name_cache.json only (no fuzzy)", "n_missing_rxcui_events": 1135941, "n_exact_name_mapped": 0, "n_mapped_into_stage1_vocab": 0, "unique_missing_generic_names": 270, "unique_missing_descr_names": 1378, "unique_generics_exact_matchable_to_rxnorm_cache": 32, "unique_descr_exact_matchable_to_rxnorm_cache": 2, "examples": [], "note": "Many missing-RxCUI rows are NCH compound/custom strings (e.g. LET gel, human milk) that have no exact RxNorm string match; fuzzy matching is disallowed."}`
7. **Extended frozen-BGE valid?** `True`
8. **Additional data preservable:** `{"proc_chapter_pct": 99.74003284531256, "med_L2_any_pct": 37.353800446634324, "sample_encode_status": "complete"}`
9. **Truncation:** `{"percentiles_raw_length": {"0.5": 211.0, "0.75": 583.0, "0.9": 1768.800000000001, "0.95": 3269.5999999999967, "0.99": 8575.359999999984}, "n_gt_1024": 545, "pct_gt_1024": 15.69248488338612, "by_age_band": [{"age_band": "<1", "n": 195, "pct_gt_1024": 14.358974358974358, "median_len": 203.0, "median_duration_before": 93.80667824074074, "median_duration_retained": 76.66315972222222, "median_frac_retained": 1.0, "median_earliest_age_before_y": 0.0054757015742642025, "median_earliest_age_after_y": 0…`
10. **Index leakage:** `{"psg_encounter_events_in_clean_table": 0, "passed": true}`
11–12. **Age extrapolation:** `{"sign_change_across_0_90": false, "pediatric_lambda_magnitude_max": 1.3855146534733793, "adult_support_approx": {"mean": 63.33600997924805, "sd": 16.574804306030273, "approx_range_years": [30.1864013671875, 96.4856185913086]}, "extrapolation_z_at_age_0": -3.8212221881983264, "extrapolation_z_at_age_10": -3.2178968146153766, "beta_near_zero": false, "interpretation_hint": "pathological pediatric extrapolation risk"}`
13. **Incident OSA:** `{"n": 705, "prevalence": 0.20287769784172663}`
14. **AHI:** `{"n_local_tsv_files": 20, "n_local_edf_files": 21, "n_studies_with_valid_ahi": 20, "n_studies_attempted": 20, "missingness_note": "SleepBank Sleep_Data locally contains only ~20 EDF/TSV pairs; cohort-wide AHI is NOT available from Health_Data tables. Do not fabricate labels for the remaining studies.", "derivation": "AHI = count(apnea|hypopnea annotations) / sleep_hours; sleep_hours from stage annotation durations when present.", "pediatric_severity_bins": [{"name": "normal", "lo": 0.0, "hi": 1.0}, {"name": "mild", "lo": 1.0, "hi": 5.0}, {"name": "moderate", "lo": 5.0, "hi": 10.0}, {"name": "severe", "lo": 10.0, "hi": Infinity}]}`
15. **Splits:** `{"train": 2433, "val": 522, "test": 520}`
16. **Ready variants:** ['diagnoses_only', 'dx_proc_strict', 'dx_proc_normalized', 'dx_med_normalized', 'dx_proc_med_strict', 'dx_proc_med_normalized', 'extended_bge_offline_sample']

## Notes

- AHRQ CCS for Services and Procedures was **not** available (AMA gate); used CPT chapters with distinct `CPT_CHAPTER_*` namespace.
- Stage-1 OOV contract restored: retain as UNK (id 1), do not drop.
- No Stage-2 finetuning launched.

