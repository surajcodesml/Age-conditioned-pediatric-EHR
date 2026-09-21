# NCH Stage-2 preprocessing v2 — final answers

Artifacts: `artifacts/nch_stage2/v2/` (v1 untouched). Stage-1 checkpoints used read-only.

## 1–3 Procedures
Most OOV is CPT/HCPCS Level I ({'CPT_HCPCS_LEVEL_I': 255459, 'ICD10_PCS': 19198, 'ICD9_PCS': 10847, 'NCH_LOCAL_SHX': 686}); ICD-PCS already maps to CCS_*. Exact event-weighted coverage 15.1% → format-normalized 15.1%.
CPT chapter grouping (not AMA CCS Svcs/Proc — unavailable) covers 99.7% events as CPT_CHAPTER_* (new namespace). Stage-1-compatible normalized recovery remains 15.1%.
NO for CPT chapters vs CCS_*: Stage-1 CCS_* are ICD procedure CCS. MIMIC/NCH share 4 CPT chapters at coarse level.

## 4–6 Medications
| Level | Event-weighted % | Events recovered |
|---|---:|---:|
| L0 exact RxCUI | 31.9 | 966968 |
| L1 brand-neutral → Stage-1 | 38.8 | 1177436 |
| L2 ingredient → Stage-1 | 31.9 | 966968 |
| L2 ingredient sets (incl. new) | 37.4 | 1133073 |
| L3 NCH thera/pharm class | 62.6 | 1897413 |

OOV CUIs: {'unique_oov_cuis': 1667, 'formulation_mismatch_cuis_ingredient_in_stage1': 2, 'genuinely_absent_or_unresolvable_cuis': 1665, 'tty_among_oov': {'missing': 1135941, 'Semantic Clinical Drug': 726001, 'Precise Ingredient': 163751, 'Ingredient': 40041, 'Generic Pack': 637, 'Multiple Ingredients': 15}}

Missing RxCUI: {'method': 'exact normalized name match to rxcui_name_cache.json only (no fuzzy)', 'n_missing_rxcui_events': 1135941, 'n_exact_name_mapped': 0, 'n_mapped_into_stage1_vocab': 0, 'unique_missing_generic_names': 270, 'unique_missing_descr_names': 1378, 'unique_generics_exact_matchable_to_rxnorm_cache': 32, 'unique_descr_exact_matchable_to_rxnorm_cache': 2, 'examples': [], 'note': 'Many missing-RxCUI rows are NCH compound/custom strings (e.g. LET gel, human milk) that have no exact RxNorm string match; fuzzy matching is disallowed.'}

## 7–8 Extended BGE
Technically valid: **True**. Preserve: {'proc_chapter_pct': 99.74003284531256, 'med_L2_any_pct': 37.353800446634324, 'sample_encode_status': 'complete'}

## 9 Truncation
15.7% sequences >1024 (n=545). p50/p90/p99 lengths = 211/1769/8575.
Adolescent early-history: {'pct_truncated': 12.134831460674157, 'median_earliest_age_before_y': 5.368925393566051, 'median_earliest_age_after_y': 6.225872689938399, 'highlight': 'Adolescent truncation does not strongly shift earliest retained age'}

## 10 Leakage
{'psg_encounter_events_in_clean_table': 0, 'passed': True}
Washout (analysis only): {'window_24h': {'n_events': 108, 'n_patients': 57}, 'window_7d': {'n_events': 532, 'n_patients': 244}, 'note': 'Analysis only; these windows are NOT removed in v2 primary sequences.'}

## 11–12 Age extrapolation
{'sign_change_across_0_90': False, 'pediatric_lambda_magnitude_max': 1.3855146534733793, 'adult_support_approx': {'mean': 63.33600997924805, 'sd': 16.574804306030273, 'approx_range_years': [30.1864013671875, 96.4856185913086]}, 'extrapolation_z_at_age_0': -3.8212221881983264, 'extrapolation_z_at_age_10': -3.2178968146153766, 'beta_near_zero': False, 'interpretation_hint': 'pathological pediatric extrapolation risk'}

## 13–15 Labels & splits
Incident OSA: {'n': 705, 'prevalence': 0.20287769784172663}
AHI: {'n_local_tsv_files': 20, 'n_local_edf_files': 21, 'n_studies_with_valid_ahi': 20, 'n_studies_attempted': 20, 'missingness_note': 'SleepBank Sleep_Data locally contains only ~20 EDF/TSV pairs; cohort-wide AHI is NOT available from Health_Data tables. Do not fabricate labels for the remaining studies.', 'derivation': 'AHI = count(apnea|hypopnea annotations) / sleep_hours; sleep_hours from stage annotation durations when present.', 'pediatric_severity_bins': [{'name': 'normal', 'lo': 0.0, 'hi': 1.0}, {'name': 'mild', 'lo': 1.0, 'hi': 5.0}, {'name': 'moderate', 'lo': 5.0, 'hi': 10.0}, {'name': 'severe', 'lo': 10.0, 'hi': inf}]}
Splits: {'train': 2433, 'val': 522, 'test': 520}

## 16 Ready representations
['diagnoses_only', 'dx_proc_strict', 'dx_proc_normalized', 'dx_med_normalized', 'dx_proc_med_strict', 'dx_proc_med_normalized', 'extended_bge_offline_sample']

Cohort: {'primary_rule': 'index_age_years < 18', 'pediatric_all_studies': {'n_studies': 3761, 'n_patients': 3475}, 'pediatric_first_study': {'n_studies': 3475, 'n_patients': 3475}, 'adult_all_studies': {'n_studies': 223, 'n_patients': 204}, 'adult_first_study': {'n_studies': 204, 'n_patients': 204}}
Temporal cleaned: {'n': 3473, 'history_duration_days': {'min': 1.3181712962962964, 'p50': 1890.8529513888889, 'p95': 4428.203261574074, 'p99': 5278.505187499994, 'max': 6374.786261574074}, 'max_years': 17.453213584049486, 'extreme_120y_gone': True}
