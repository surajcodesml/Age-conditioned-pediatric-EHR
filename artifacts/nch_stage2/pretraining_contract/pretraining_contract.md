# Stage-1 MIMIC pretraining contract

Source of truth: implementation (`stage1_mimic_pretrain/`, `model_new/data.py`,
`preprocessing/build_event_table.py`, `preprocessing/rollup_and_describe.py`)
plus the already-written `adkm_s0` config. The MIMIC vocabulary was **not** regenerated.

- Active run: `stage1_mimic_pretrain.train --arm age_temporal --run_name adkm_s0 --seed 0`
- Git commit recorded by the run: `6778b5527b82e03f2648e223ea023a7ba56f28e5`

## Event representation

Model batch (`model_new.data.pretrain_collate`):

- `code_indices`: [B, L] int64; PAD=0, UNK=1, real = vocab_index + 2
- `timestamps_days`: [B, L] float64; days from the patient's first event; padded 0.0
- `age_years`: [B, L] float32; age_at_event_days / 365.25; padded 0.0
- `demographics`: [B, L, 9] float32; last-event vector is consumed by the head
- `attention_mask`: [B, L] bool
- `lengths`: [B] int64

- Timestamp on the event table: timestamp_days = (event_time - min_event_time_of_patient) / 86400
- Age at event: `age_at_event_days = (event_time - dob)/86400; years = days/365.25`
- Token string: `rolled code_id with type prefix (PHE_, CCS_, RXN_, ...)`
- Model id: `PAD=0, UNK=1, real=vocab_id+2 (model_new.data._pad_common)`

Modalities in the frozen vocabulary (counts are unique tokens):

- `ICD10_`: 16092
- `RXN_`: 3720
- `HCPCS_`: 2364
- `CHART_`: 2311
- `PHE_`: 1748
- `ICD9_`: 1590
- `LAB_`: 976
- `DRG_`: 846
- `INPUT_`: 327
- `CCS_`: 231
- `PROC10_`: 188
- `ICUPROC_`: 159
- `OUTPUT_`: 71
- `NDC_`: 12

## Vocabulary

- Path: `/home/suraj/Git/Age-conditioned-pediatric-EHR/data/processed/code_vocab.json`
- Size `|V|` = **30635**
- Token→ID: JSON `JSON object {token: int}; keys sorted alphabetically at BGE build time`
- Special: PAD model id **0**, UNK model id **1** (`unk_vocab_index = 30635` in shards)
- Frozen text encoder: `BAAI/bge-m3` dim 1024, 30637 rows (PAD/UNK zero vectors)
- Rare-code filter: none at vocab build; vocab = every code that received a description after rollup
- OOV: codes missing from code_vocab.json map to unk_vocab_index = V, then model id 1
- Code-system is encoded in the token prefix: `True`
- ICD normalization: `strip().upper().replace('.', '')` (`preprocessing.rollup_and_describe.normalize_code`)

## Temporal preprocessing

- Order: subject_id, event_time, code_id (parquet); within a visit: event_time, code_id (tensorize)
- Ties: stable secondary key code_id (and original index at window selection via lexsort)
- **τ** = `tau_ij = log1p(|t_i - t_j| / c) with c = 7 days` implemented by `model_new.data.lag_to_tau`
- `log1p`: True; `c` = 7.0 days
- Chebyshev / `[-1,1]` rescale: **False**
- `tau_max` used by Stage-1 model: **False**
- Recorded MIMIC `tau_max` (diagnostic / INV-TMAX): `None`
- Truncation: newest 1024 events
- Masking: padding-only, not causal

**NCH must reuse MIMIC `WEEK_DAYS=7` and the recorded `tau_max` for diagnostics;**
it must not estimate a new `c` or `tau_max`. The live kernel does not clip τ.

## Age

- `age_at_event_days = (event_time - dob)/86400; years = days/365.25`
- z(a) mean = `63.33601047086567` (run) / `63.33601047086648` (config default)
- z(a) sd = `16.574804662350303` / `16.574804662346914`
- No age clipping, no age `log1p`.
- Age enters attention as λ(a)=λ0+β z(a_i) and the head as z-scored last-event demo channel 0.

## Sequence construction

- one sample per (patient, next hadm-visit V_{m+1}) with ≥1 prior event
- History: `timestamp < target_time; ties go to the target (INV-HORIZON)`
- Splits: patient-level quintile-stratified 70/10/20, seed 42 (build_splits.py)

## Stage-1 objective vs Stage-2 encoder inputs

Pretraining-only: next-visit multi-hot BCE over `|V|`.

Stage-2 must reproduce the encoder inputs listed in the JSON field
`stage1_objective.encoder_inputs_stage2_must_reproduce`.

## Traceability

- **build_event_table**: `preprocessing/build_event_table.py` (main)
- **rollup**: `preprocessing/rollup_and_describe.py` (['normalize_code', 'create_phecode_maps', 'create_ccs_maps', 'build_rollup'])
- **splits**: `preprocessing/build_splits.py` (assign_splits)
- **embeddings_and_vocab**: `preprocessing/compute_bge_embeddings.py` (main)
- **tensorize**: `model_new/tensorize_pretrain.py` (['build_subject_payload', 'tensorize_bucket', 'rebuild_split'])
- **dataset**: `model_new/data.py` (TensorizedPretrainDataset)
- **model**: `stage1_mimic_pretrain/model.py` (MinimalDKMModel)
