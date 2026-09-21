# NCH Stage-2 preprocessing validation

## Temporal leakage

- Assert `max(input_event_time) < index_time`
- Violations: **0**
- Passed: True

## Ordering

- Violations: **0** (passed=True)

## Vocabulary

- `|V|` = 30635; PAD=0 UNK model id=1; unk_vocab_index=30635
- Out-of-range token IDs: **0**
- Max observed vocab id: 26726
- MIMIC token IDs were not rewritten.

## Temporal recompute spot-check

- Passed: True
- patient 17740 study 4801: raw_span=58.7708d τ=2.240266 lag_to_tau=2.240266 match=True
- patient 1564 study 514: raw_span=1946.8195d τ=5.631631 lag_to_tau=5.631631 match=True
- patient 310 study 20791: raw_span=43093.7578d τ=8.725386 lag_to_tau=8.725386 match=True
- patient 10726 study 12454: raw_span=3076.8369d τ=6.088020 lag_to_tau=6.088020 match=True
- patient 6436 study 9721: raw_span=4409.8174d τ=6.447265 lag_to_tau=6.447265 match=True
- patient 5584 study 7180: raw_span=1355.8215d τ=5.271402 lag_to_tau=5.271402 match=True
- patient 826 study 4513: raw_span=1008.7750d τ=4.977497 lag_to_tau=4.977497 match=True
- patient 13366 study 17521: raw_span=1683.2125d τ=5.486699 lag_to_tau=5.486699 match=True

## Age spot-check (DOB vs recorded)

- Max |Δ| vs DOB: 0.0 days; passed=True
- band <1 patient 91: index_age_days=27.83 recomputed=27.830381944444444 sleep_study_col=27.0
- band <1 patient 127: index_age_days=22.78 recomputed=22.778622685185184 sleep_study_col=22.0
- band <1 patient 136: index_age_days=358.78 recomputed=358.77856481481484 sleep_study_col=358.0
- band 1-5 patient 7: index_age_days=775.76 recomputed=775.7642476851852 sleep_study_col=775.0
- band 1-5 patient 22: index_age_days=1730.78 recomputed=1730.784849537037 sleep_study_col=1730.0
- band 1-5 patient 31: index_age_days=1146.79 recomputed=1146.7896412037037 sleep_study_col=1146.0
- band 6-11 patient 1: index_age_days=3338.85 recomputed=3338.849178240741 sleep_study_col=3338.0
- band 6-11 patient 28: index_age_days=2252.81 recomputed=2252.813425925926 sleep_study_col=2252.0
- band 6-11 patient 37: index_age_days=3506.84 recomputed=3506.844351851852 sleep_study_col=3506.0
- band 12-17 patient 10: index_age_days=4782.78 recomputed=4782.778657407407 sleep_study_col=4782.0
- band 12-17 patient 16: index_age_days=4408.85 recomputed=4408.845497685185 sleep_study_col=4408.0
- band 12-17 patient 25: index_age_days=6543.77 recomputed=6543.769166666667 sleep_study_col=6543.0
- band >=18 patient 121: index_age_days=6754.89 recomputed=6754.8879050925925 sleep_study_col=6754.0
- band >=18 patient 259: index_age_days=6664.85 recomputed=6664.847824074074 sleep_study_col=6664.0
- band >=18 patient 271: index_age_days=13713.77 recomputed=13713.768275462962 sleep_study_col=13713.0

## Pediatric cohort (first-study, dx+procedure sequences)

| band | patients | studies | median events | median history (d) | mean z(a) | hist≥1y |
|---|---:|---:|---:|---:|---:|---:|
| <1 | 195 | 195 | 41.0 | 92.80239583333334 | -3.7972825612234975 | 0.0 |
| 1-5 | 1187 | 1187 | 113.0 | 1004.8705902777778 | -3.607046996150568 | 0.8525695029486099 |
| 6-11 | 1201 | 1201 | 141.0 | 2593.2507060185185 | -3.2896398192492686 | 0.9217318900915903 |
| 12-17 | 890 | 890 | 164.0 | 3424.662210648148 | -2.925051907964811 | 0.9179775280898876 |
| >=18 | 198 | 198 | 321.0 | 3597.8297627314814 | -2.5083173349211187 | 0.9595959595959596 |

- Samples with ≥1 history event: 3671 / 3671
- Samples with ≥10 events: 3596
- Samples with ≥30d history: 3592
- Samples with ≥1y history: 3126

## Missing / invalid values

{
  "n_rows": 5187791,
  "missing_patient_id": 0,
  "missing_time": 0,
  "missing_code": 1136847,
  "n_redacted": 6460,
  "n_impossible_age": 2707,
  "n_measurement": 332569,
  "handling": {
    "measurements": "excluded from sequence tokens (not in Stage-1 portable vocab)",
    "oov_clinical_codes": "retained in canonical_events; dropped from encoder input",
    "missing_times": "cannot enter a time-ordered history",
    "redacted": "not converted into clinical concepts"
  }
}

## Patient splits

Splits are not assigned in this preprocessing pass. When they are, split on patient_id so no patient appears in more than one split (including the all-study cohort).
