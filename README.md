# Age-Conditioned Pediatric EHR

Continuous developmental age conditioning of temporal attention weighting for pediatric EHR trajectory prediction. Built as a reimplementation and extension of TALE-EHR (Yu et al., 2025, arXiv:2507.14847).

## Environment and Data Requirements

- Python environment: `conda` env named `ehr`
- Core dependencies: `duckdb`, `pyarrow`, `pandas`, `torch`, `FlagEmbedding`, `requests`
- MIMIC-IV v3.1 (PhysioNet credentialed access required)
  - Symlink: `data/raw/mimiciv-3.1 -> /path/to/physionet.org/files/mimiciv/3.1`
  - Required subfolders: `hosp/`, `icu/`
- MIMIC-IV-ED v2.2 (optional, for ED event sources)
- Hardware: 32GB RAM, GPU for BGE embeddings (ROCm or CUDA)

## Repository Structure

```
Age-conditioned-pediatric-EHR/
├── preprocessing/
│   ├── build_event_table.py      # Step 1: raw event extraction
│   ├── rollup_and_describe.py    # Step 2: code rollup + descriptions
│   ├── build_splits.py           # Step 3: patient-level splits
│   └── compute_bge_embeddings.py # Step 4: BGE-m3 embeddings
├── model/                        # TALE-EHR pretraining stack
├── data/
│   ├── raw/                      # Symlink to MIMIC-IV v3.1
│   └── processed/                # Pipeline outputs
└── tests/
```

---

## 1. Preprocessing

Run in order. Each step depends on the previous output.

### Step 1: `build_event_table.py`

Extracts clinical events from 10 MIMIC-IV source tables into a unified parquet with continuous timestamps.

**Source tables:** `diagnoses_icd`, `procedures_icd`, `prescriptions`, `labevents`, `chartevents`, `drgcodes`, `inputevents`, `outputevents`, `procedureevents`, `hcpcsevents`. ED tables (`diagnosis`, `medrecon`, `pyxis`) included if MIMIC-IV-ED is available.

**Output:** `data/processed/patient_events_{full|test}.parquet`, `data/processed/code_vocab_raw.json`

**Notes:**
- Uses persistent DuckDB with 24GB memory limit and spill-to-disk
- Filters: ≥5 events per patient, >1 unique timestamp
- Computes `timestamp_days` (relative days from patient's first event, t=0) and `age_at_event_days`

```bash
conda run -n ehr python preprocessing/build_event_table.py           # full
conda run -n ehr python preprocessing/build_event_table.py --test_mode  # 1000 patients
```

### Step 2: `rollup_and_describe.py`

Rolls up raw codes to clinical ontologies and builds text descriptions for BGE embedding.

**Rollup mapping:**
- Diagnoses: `ICD9_`/`ICD10_` → `PHE_` (PheCode via PheWAS catalog)
- Procedures: `PROC9_`/`PROC10_` → `CCS_` (HCUP procedure CCS)
- HCPCS: `HCPCS_` → `CCS_` (attempted via CCS crosswalk)
- Medications: `NDC_` → `RXN_` (RxNorm via bulk RXNSAT.RRF, no API calls)
- Passthrough (no rollup): `LAB_`, `CHART_`, `DRG_`, `INPUT_`, `OUTPUT_`, `ICUPROC_`

**Output:** `data/processed/patient_events_rolled_{full|test}.parquet`, `data/processed/code_descriptions.json`

**Notes:**
- Uses persistent DuckDB with 24GB memory limit
- RxNorm mappings cached in `data/processed/mappings/ndc_rxnorm_cache.json`; reused on subsequent runs
- Use `--force-rxnorm` to re-download and re-parse RxNorm files

```bash
conda run -n ehr python preprocessing/rollup_and_describe.py           # full
conda run -n ehr python preprocessing/rollup_and_describe.py --test_mode  # test
```

### Step 3: `build_splits.py`

Patient-level train/val/test split (70/10/20), stratified by event-count quintiles. Seed=42.

**Output:** `data/processed/{train,val,test}_events.parquet`

```bash
conda run -n ehr python preprocessing/build_splits.py           # full
conda run -n ehr python preprocessing/build_splits.py --test_mode  # test
```

### Step 4: `compute_bge_embeddings.py`

Encodes code descriptions with BGE-m3 into frozen 1024-dim embeddings. PAD (index 0) and UNK (index 1) get zero vectors. Strict offset: `code_vocab[code] == bge_index[code] - 2`.

**Output:** `data/processed/bge_embeddings.pt`, `data/processed/code_vocab.json`

```bash
conda run -n ehr python preprocessing/compute_bge_embeddings.py --force
```

### Test mode workflow

Run all four steps with `--test_mode` first to validate before full runs:

```bash
conda run -n ehr python preprocessing/build_event_table.py --test_mode
conda run -n ehr python preprocessing/rollup_and_describe.py --test_mode
conda run -n ehr python preprocessing/build_splits.py --test_mode
conda run -n ehr python preprocessing/compute_bge_embeddings.py --force
```

---

## 2. TALE-EHR Baseline (Implemented)

Core TALE-EHR pretraining components are now implemented end-to-end:

- `model/dataset.py`
  - `EHRDataset` builds visit-level pretraining samples from split parquet files
  - `ehr_collate` pads variable-length sequences and returns:
    - `code_indices` (BGE-aligned ids: PAD=0, UNK=1, real=2+)
    - `timestamps_days`, `delta_t`, `attention_mask`, `demographics`
    - `target_codes` (multi-label next-visit targets), `target_time_gap`
- `model/time_aware_attention.py`
  - `PolynomialTemporalWeight`
  - `TimeAwareAttention` (single-head, causal, time-weighted attention)
  - `MultiScaleTemporalAggregation`
- `model/tale_ehr.py`
  - Full TALE-EHR model with frozen BGE table, temporal encoder, demographic/history projections,
    code predictor `f`, and intensity predictor `g`
- `model/train.py`
  - Pretraining loop (Algorithm 1 phase): selectable code loss (`bce` default, optional `focal`) +
    temporal point-process loss, AMP support, grad clipping, checkpointing, dry-run mode, and resume support

### Quick sanity checks

```bash
# Dataset + collator smoke test
conda run -n ehr python model/dataset.py --test_mode --max_rows 200000

# Time-aware attention smoke test
conda run -n ehr python model/time_aware_attention.py

# Full model smoke test
conda run -n ehr python model/tale_ehr.py

# Pretraining dry run (3 train steps, 1 val step)
conda run -n ehr python model/train.py \
  --data_dir data/processed \
  --embedding_path data/processed/bge_embeddings.pt \
  --vocab_path data/processed/code_vocab.json \
  --epochs 1 --batch_size 2 --max_rows 50000 --dry_run --num_workers 0
```

### Pretraining (full)

```bash
conda run -n ehr python model/train.py \
  --data_dir data/processed \
  --embedding_path data/processed/bge_embeddings.pt \
  --vocab_path data/processed/code_vocab.json \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 8 \
  --use_tensorized \
  --device cuda \
  --code_loss bce \
  --gamma_loss 500.0 \
  --bce_pos_weight 0.0 \
  --run_name run_full_bce_g500
```

### Resume an interrupted run

```bash
conda run -n ehr python -u model/train.py \
  --epochs 10 \
  --batch_size 32 \
  --num_workers 8 \
  --use_tensorized \
  --data_dir data/processed \
  --device cuda \
  --code_loss bce \
  --gamma_loss 500.0 \
  --bce_pos_weight 0.0 \
  --run_name run_20260427_152603 \
  --resume_from checkpoints/run_20260427_152603/epoch_003.pt
```

Notes:
- If `val_events.parquet` is missing, training falls back to `test_events.parquet` for validation.
- `delta_t` is the primary memory bottleneck; reduce `--batch_size` first if OOM occurs.
- Each run writes:
  - `checkpoints/<run_name>/train.log` (epoch summary lines)
  - `checkpoints/<run_name>/console.log` (all stdout/stderr, including diagnostics prints)
- On resume, logs append (they are not overwritten), and training restarts at `checkpoint_epoch + 1`.

### Minimal diagnostic + stability notes

- Code-prediction collapse was mitigated by making `bce` the default code loss and increasing `--gamma_loss` default to `500.0` so code-loss gradients are not starved by time loss.
- Optional diagnostics in `model/time_aware_attention.py` and `model/tale_ehr.py` are sampled (~0.5% forward passes) and print compact stats for attention entropy/collapse, temporal weights, representation scale/dead activations, logits, and intensity.
- Pause safely with `Ctrl+C`; avoid hard kill (`kill -9`). Resume from the latest completed `epoch_XXX.pt`.

## 3. Age-Conditioned Extension (TODO)

- [ ] Developmental age embedding
- [ ] Age-conditioned polynomial coefficients: `w(Δt, eₐ) = σ(Σ αₖ(eₐ)·log(1+Δt)ᵏ)`
- [ ] Evaluation on PIC dataset
- [ ] Cross-dataset transfer: MIMIC-IV → PIC

## 4. GrowSmart Clinical System (TODO)

- [ ] Pediatric-stratified RAG
- [ ] Dual-decoder NLG (clinician-facing + family-facing)

---

## Output Summary

After full preprocessing, key artifacts in `data/processed/`:

| File | Description |
|---|---|
| `patient_events_full.parquet` | Raw unified events (626M rows) |
| `patient_events_rolled_full.parquet` | After code rollup |
| `code_descriptions.json` | Code → text description |
| `code_vocab.json` | Code → integer index (0-indexed, no PAD/UNK) |
| `bge_embeddings.pt` | `[N+2, 1024]` frozen embeddings |
| `{train,val,test}_events.parquet` | Patient-level splits |
| `mappings/` | Downloaded mapping files and caches |

## References

- Yu et al., "Time-Aware Attention for Enhanced EHR Modeling", arXiv:2507.14847, 2025
- Johnson et al., "MIMIC-IV", Scientific Data, 2023
- Chen et al., "BGE M3-Embedding", arXiv:2402.03216, 2024
- Gupta et al., "An Extensive Data Processing Pipeline for MIMIC-IV", ML4H, 2022