# PIC (Paediatric Intensive Care) pipeline documentation

PIC v1.1.0 is preprocessed to match the MIMIC-IV event schema so existing `model/`, `finetune/dataset.py`, and tensorization logic run unchanged. All PIC-specific code lives under `preprocessing/PIC/` and `finetune/PIC/`; parent MIMIC scripts are not modified.

**Environment:** `conda` env `ehr` for all commands below (`conda run --no-capture-output -n ehr ...`).

**Raw data (default):** `/home/suraj/Data/PIC/physionet.org/files/picdb/1.1.0/V1.1.0/`

---

## 1. Data preprocessing

Run in order from the repo root.

| Step | Script | Main outputs |
|------|--------|----------------|
| 1 | `preprocessing/PIC/build_event_table_pic.py` | `data/processed/pic/patient_events_pic.parquet`, `code_vocab_raw_pic.json`, `pic_lab_loinc_map.json` |
| 2 | `preprocessing/PIC/rollup_and_describe_pic.py` | `patient_events_rolled_pic.parquet`, `code_descriptions_pic.json` |
| 3 | `preprocessing/PIC/build_splits_pic.py` | `train_events.parquet`, `val_events.parquet`, `test_events.parquet` (70/10/20, stratified by event count) |
| 4 | `preprocessing/PIC/compute_bge_embeddings_pic.py` | `code_vocab_pic.json`, `bge_embeddings_pic.pt` |

```bash
python -m preprocessing.PIC.build_event_table_pic
python -m preprocessing.PIC.rollup_and_describe_pic
python -m preprocessing.PIC.build_splits_pic
python -m preprocessing.PIC.compute_bge_embeddings_pic
```

**PIC-specific handling (step 1):**

- `ITEMID` read as VARCHAR; diagnoses use WHO ICD-10 in `aux_code` for PheCode mapping (Chinese `ICD10_CODE_CN` in `code_id`).
- Race set to `'UNKNOWN'` (UNK bucket in `encode_race`).
- DuckDB CSV quoting fixed for embedded commas in lab/microbiology text.
- Cohort filters aligned with MIMIC: ≥5 events, >1 distinct timestamp, admission present.

**Rollup (step 2):** Diagnoses → PheCodes via two-pass ICD-10 lookup (full code, then 3-char prefix). Other modalities keep native codes + English descriptions.

**Shared helpers:** `preprocessing/PIC/_shared.py` re-exports `create_phecode_maps`, split logic, and BGE embedding driver from parent `preprocessing/` (read-only).

---

## 2. Fine-tuning cohorts

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `finetune/PIC/prep_clip_age.py` | Clip negative `age_at_event_days` to 0 in split + rolled parquets |
| 1 | `finetune/PIC/build_cohorts_pic.py` | Build task cohorts + diagnosis filtered event parquets |
| 2 | `finetune/PIC/test_cohort_leakage_pic.py` | Leakage checks, length-only AUROC, dataset smoke test |

```bash
python -m finetune.PIC.prep_clip_age
python -m finetune.PIC.build_cohorts_pic
python -m finetune.PIC.test_cohort_leakage_pic
```

**Cohort contract:** `subject_id`, `label` (0/1), `last_event_idx` — same as MIMIC `DiseaseClassificationDataset`.

**Settings:** `OBS_WINDOW_DAYS = 1.0`; index admission = first admission; `last_event_idx` matches dataset ordering `(timestamp_days, event_time, code_id)`.

**Tasks:**

| Task | Label | Notes |
|------|-------|--------|
| `mortality` | In-hospital death | Full split events |
| `los_gt7` | ICU LOS > 7 days | Full split events |
| `pneumonia` | Target codes in index admission | Filtered events (targets removed from input) |
| `heart_malformations` | Same | PheCode family `PHE_747.*` only; ICD-10 Q20–Q28 fallback |

**Diagnosis targets:** Derived from PheWAS map + vocab; pneumonia restricted to `PHE_480.*`, heart to `PHE_747.*` (off-family codes logged with descriptions).

**Outputs:** `data/processed/pic/cohorts/{task}_{split}_cohort.parquet`; diagnosis tasks also `{task}_{split}_events.parquet`.

---

## 3. Fine-tuning

### Setup and tensorization

```bash
python -m finetune.PIC.setup_finetune_dirs
# Per task:
python -m finetune.PIC.build_disease_tensors_pic \
  --cohort_dir data/processed/pic/finetune/<task>/cohort \
  --events_parquet data/processed/pic/finetune/<task>/events.parquet \
  --vocab_path data/processed/pic/code_vocab_pic.json \
  --out_dir data/tensorized/pic/<task>
```

`setup_finetune_dirs` writes `data/processed/pic/finetune/<task>/cohort/` and `events.parquet` (symlink to rolled events for mortality/LOS; union of filtered splits for diagnosis tasks).

`build_disease_tensors_pic` writes NPZ shards with `hadm_id` and `n_events_in_window` (required by `TensorizedDiseaseClassificationDataset`).

### Training (`finetune/PIC/train_pic.py`)

Warm-starts the pretrained backbone onto PIC via `PICTALEEHRClassifier` (`finetune/PIC/model_pic.py`): MIMIC temporal/attention weights transfer; **PIC BGE table** (`bge_embeddings_pic.pt`, 2200 rows) replaces MIMIC embeddings.

**Pretrained checkpoints:**

| Arm | Checkpoint |
|-----|------------|
| TALE-EHR (vanilla) | `checkpoints/run_20260427_152603/best_pretrain.pt` |
| Age-conditioned | `checkpoints/age_real_202605112156/epoch_012.pt` |

**Default optimization (10 epochs):**

| Group | LR | Parameters |
|-------|-----|------------|
| kernel | `1e-3` | `age_coeff_gen` (age arm) or `temporal_weight.coefficients` (TALE-EHR) |
| backbone_slow | `1e-5` | Remaining backbone |
| head | `1e-3` | Classifier |

**Run directories:** `checkpoints/finetune/PIC/<task>_{age|vanilla}_lr_10eph/` (`best.pt`, `history.json`, `console.log`, `test_predictions.parquet`, `decay_alpha.json`).

```bash
python -m finetune.PIC.train_pic \
  --disease pneumonia \
  --pretrained_ckpt checkpoints/age_real_202605112156/epoch_012.pt \
  --cohort_dir data/processed/pic/finetune/pneumonia/cohort \
  --tensorized_dir data/tensorized/pic/pneumonia \
  --run_dir checkpoints/finetune/PIC/pneumonia_age_lr_10eph \
  --epochs 10 --lr_kernel 1e-3 --lr_backbone 1e-5 --lr_head 1e-3 \
  --batch_size 64 --num_workers 4 --prefetch_factor 2
```

### Orchestration (`run_all_finetune.sh`)

From repo root:

```bash
chmod +x run_all_finetune.sh
./run_all_finetune.sh
```

Runs PIC setup → tensorization (skip if shards exist) → all four tasks × both arms. Logs: `checkpoints/finetune/PIC/run_all_<timestamp>.log`. Uses `HIP_VISIBLE_DEVICES=0` by default.

**Note:** MIMIC `los_gt7` training block is commented out in the current script; enable separately if needed (use `batch_size 16` on 30GB RAM to avoid DataLoader OOM).

---

## 4. Evaluation and analysis

### Age-stratified test metrics (`finetune/PIC/age_stratified_eval.py`)

Developmental bands (years): neonate (0–1/12), infant, toddler, preschool, school, adolescent (12–18).

Per task: AUROC/AUPRC/ECE per band for TALE-EHR and age-conditioned; bootstrap CIs; paired ΔAUROC (age − vanilla).

```bash
python -m finetune.PIC.age_stratified_eval
```

**Outputs:** `results/pic/age_stratified/<task>.csv`, `<task>_auroc_by_band.png`

Uses `test_preds.parquet` or builds it from `test_predictions.parquet` / fresh inference.

### Kernel mechanism plots (`finetune/PIC/age_kernel_viz.py`)

Decay \(w(\Delta t, a)\) from fine-tuned checkpoints (attention + aggregation modules).

```bash
python -m finetune.PIC.age_kernel_viz
```

**Outputs:** `results/pic/age_kernel/<task>_{family,coeffs,sensitivity,horizon}.png`, `<task>_kernel.npz`

### Mechanism link (`finetune/PIC/age_mechanism_link.py`)

Spearman correlation between per-band ΔAUROC and kernel deviation \(\|\Delta\alpha\|\).

```bash
python -m finetune.PIC.age_mechanism_link
```

**Outputs:** `results/pic/age_mechanism_link/<task>_mechanism_link.{png,csv}`

### Decay comparison figures (`finetune/PIC/plot_pic_decay_comparison.py`)

TALE-EHR vs age-conditioned decay for pneumonia and heart malformations (attention module, fine-tuned `*_lr_10eph` checkpoints).

```bash
python -m finetune.PIC.plot_pic_decay_comparison
```

**Outputs:** `figures/pic_decay/pic_decay_tale_ehr.png`, `pic_decay_age_conditioned.png`, `pic_decay_combined.png` (+ PDF)

---

## 5. Directory map

```
data/processed/pic/
  train|val|test_events.parquet
  patient_events_rolled_pic.parquet
  code_vocab_pic.json, bge_embeddings_pic.pt, code_descriptions_pic.json
  cohorts/                          # flat cohort + filtered events
  finetune/<task>/cohort/, events.parquet

data/tensorized/pic/<task>/{train,val,test}/shard_*.npz

checkpoints/finetune/PIC/
  <task>_{age|vanilla}_lr_10eph/best.pt

results/pic/
  age_stratified/, age_kernel/, age_mechanism_link/

figures/pic_decay/
```

---

## 6. Implementation notes

- **PIC vocab size:** 2198 codes; BGE table shape `[2200, 1024]` (PAD/UNK + codes). Collate maps vocab indices to rows `+2`.
- **Age conditioning in fine-tune:** Gradients reach **attention** `age_coeff_gen`; aggregation `age_coeff_gen` is in the fast LR group but off the classification path (`return_repr_only` uses attention output `e` only).
- **No edits to** `finetune/train.py`, `finetune/model.py`, or MIMIC preprocessing scripts; PIC paths are selected only in `finetune/PIC/*`.

---

## 7. Script index

| Path | Role |
|------|------|
| `preprocessing/PIC/build_event_table_pic.py` | Raw PIC → long event table |
| `preprocessing/PIC/rollup_and_describe_pic.py` | PheCode rollup + descriptions |
| `preprocessing/PIC/build_splits_pic.py` | Train/val/test splits |
| `preprocessing/PIC/compute_bge_embeddings_pic.py` | BGE-m3 embeddings |
| `finetune/PIC/prep_clip_age.py` | Age clipping |
| `finetune/PIC/build_cohorts_pic.py` | Task cohorts |
| `finetune/PIC/test_cohort_leakage_pic.py` | Cohort verification |
| `finetune/PIC/setup_finetune_dirs.py` | Finetune directory layout |
| `finetune/PIC/build_disease_tensors_pic.py` | NPZ tensorization |
| `finetune/PIC/model_pic.py` | PIC classifier + backbone transfer |
| `finetune/PIC/train_pic.py` | Fine-tuning loop |
| `run_all_finetune.sh` | Batch PIC fine-tuning |
| `finetune/PIC/age_stratified_eval.py` | Band-stratified metrics |
| `finetune/PIC/age_kernel_viz.py` | Kernel diagnostics |
| `finetune/PIC/age_mechanism_link.py` | Outcome vs kernel link |
| `finetune/PIC/plot_pic_decay_comparison.py` | Decay comparison figures |
| `finetune/PIC/pic_age_eval_common.py` | Shared eval constants/loaders |
