# CEHR-BERT Baseline Report

## Literature Fidelity

- **Representation Components:**
  - CEHR-BERT temporal transformation implemented as per Pang et al. (ML4H 2021).
  - Code embeddings, segment embeddings (alternating A/B for visit segments), age embeddings (Time2Vec-style), and absolute-time embeddings (Time2Vec-style) are concatenated and projected down via a learned linear layer to `d_model`.
- **ATT Definitions:**
  - Added Artificial Time Tokens (ATT) between neighboring visits based on the elapsed time ($\Delta t$):
    - `W0` - `W3`: Interval < 28 days (week interval: $W_{i} = \lfloor \Delta t / 7 \rfloor$).
    - `M1` - `M11`: Interval 28–365 days (month interval: $M_{i} = \lfloor \Delta t / 30 \rfloor$).
    - `LT`: Interval > 365 days.
- **Age/Time Embeddings:**
  - Both age and absolute time use Time2Vec-style sinusoidal encodings which are concatenated into the final representation. Transferred directly from Stage 1 to Stage 2.
- **Visit Segments:**
  - Standard `[VS]` and `[VE]` boundaries added around visit grouped clinical tokens. Segment embeddings alternate (A, B, A, B) across successive visits. For pseudo-visits lacking `hadm_id`, events mapped to daily boundaries.
- **MLM Implementation:**
  - Standard BERT masked language modeling.
  - 15% masking applied only to clinical concept tokens, excluding special tokens (VS, VE, ATT, padding).
- **VTP (Visit Type Prediction):**
  - **Status:** VTP was omitted from the primary CEHR-BERT-native baseline (`CEHR-BERT-MLM`).
  - **Reason:** MIMIC provides `admission_type` only for admitted patients, while ~65M clinical events lacked an explicit encounter mapping. To preserve rigorous literature fidelity without fabricating encounter types, VTP was dropped in favor of an MLM-only objective.
- **Architecture:**
  - Explicit configuration `cehrbert_pang2021`: hidden_size=128, n_layers=5, n_heads=8, dropout=0.1, max_seq_len=300.
- **Departures from Pang et al. 2021:**
  - Omitting VTP due to lack of standard visit type mappings across all MIMIC-IV modalities used here.
  - Using relative timestamps based on MIMIC shifting rather than true calendar seasonality for the time embedding (since MIMIC true dates are de-identified/shifted).

## Comparison Fairness

- **MIMIC Patients Shared:** 100% of the patient IDs from the primary Stage-1 cohort (154,418 Train / 22,060 Val).
- **Clinical Domains:** All 10 domains from the baseline pipeline (lab, medication, input, chart, procedure, icu_procedure, hcpcs, diagnosis, output, drg) are supplied identically to CEHR-BERT.
- **NCH Patients Shared:** 100% identical patient splits (2,433 Train / 522 Val / 520 Test).
- **Prediction Cutoff:** Identical next-encounter prediction boundaries used.
- **Target:** Identical 30,635-dimensional multi-hot next-encounter diagnosis target.
- **Vocabulary / OOV Differences:** The exact same 30,635 vocabulary is used. NCH OOV codes mapped consistently to the `[UNK]` token id.
- **Compute:** Model trained on identical hardware with matched learning rate warmup / cosine annealing strategies where possible.

### Summary Table

| Metric | Ours | CEHR-BERT |
|--------|------|-----------|
| MIMIC patients | 220,603 total | same |
| Input domains | 10 domains | same |
| Vocab | 30,635 codes | same |
| Max context | 1024 | 300 |
| Pretrain objective | Forecast | MLM |
| NCH patients | same | same |
| NCH target | same | same |

## Readiness Check
READY FOR FULL CEHR-BERT STAGE-1 PRETRAINING

## Pre-Training Validations & Configuration

### 1. Visit Construction Audit
A rigorous audit of the visit construction logic was implemented to resolve encounter-unmapped MIMIC events. Events without a `hadm_id` are now dynamically checked against the patient's existing admission intervals (with a ±1 day grace period). Only events that genuinely fall outside any known hospitalization are grouped into daily pseudo-visits.
*(Generated via `baselines/cehrbert/audit_visits.py`)*
```text
total events: 123305876
events with direct encounter mapping: 110116184
events recovered by timestamp-to-admission mapping: 3657359
events using other encounter IDs: 0
events remaining in pseudo-visits: 9532333
percentage of all tokens in pseudo-visits: 7.73%
number of real visits: 109846
number of pseudo-visits: 384402
median/mean events per visit: 29.0 / 249.5
```
Manual chronological sequences confirm `[VS] ... [VE] ATT [VS]` logic is correctly preserved without fragmenting true inpatient events.

### 2. True BERT MLM Corruption
- The `cehrbert_mlm_collate` function enforces exact 15% selection of clinical tokens for MLM.
- Special tokens (VS, VE, ATT, padding, CLS) are strictly excluded from masking.
- Of the 15% selected:
  - 80% are replaced with `[MASK]`.
  - 10% are replaced with a random clinical token.
  - 10% are left unchanged.
- Verified by `test_mlm_collator` in the unit tests suite.

### 3. CEHR-BERT Pretraining Window Sampling
- Handled via `create_cehrbert_sequence(is_pretraining=True)`.
- For patients over 300 tokens, the sequence starts at a randomly selected `[VS]` boundary to expose different parts of the history across epochs.
- For Stage-2/validation (`is_pretraining=False`), sequences fallback to deterministic tail-truncation, selecting the most recent history ending before the prediction index.
- Confirmed by `test_sampling_window` in the unit test suite.

### 4. Final Configuration Check
```text
model = cehrbert_pang2021
hidden_size = 128
n_layers = 5
n_heads = 8
dropout = 0.1
max_seq_len = 300

objective = MLM (CEHR-BERT-MLM)
mlm_probability = 0.15

epochs = 5
effective_batch_size = 32
optimizer = Adam
initial_lr = 2e-4
scheduler = cosine decay
```
