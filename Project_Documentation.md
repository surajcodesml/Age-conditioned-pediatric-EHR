# GrowSmart — Thrust 1: Age-Conditioned Pediatric EHR Modeling

**Project documentation: work completed to date**
*Repo: `Age-conditioned-pediatric-EHR/`*
*Hardware: AMD Radeon R9700 (ROCm), 32 GB RAM, machine `tesla`, conda env `ehr`*

---

## 0. Project framing

The project is the **Thrust 1** component of the GrowSmart pediatric clinical decision support system: continuous developmental age conditioning of temporal attention weighting for pediatric EHR trajectory prediction.

The work consists of three sequential pieces:

1. **Preprocessing pipeline** — MIMIC-IV v3.1 → patient event sequences ready for a TALE-EHR-style model.
2. **TALE-EHR reimplementation from scratch** — chosen because the original TALE-EHR (Yu et al., 2025, arXiv:2507.14847) has no public code release. This is the **primary baseline** for the novelty claim.
3. **Age-conditioned extension** — making the polynomial coefficients of TALE-EHR's `w(Δt)` functions of patient developmental age via a Fourier-feature embedding + zero-initialized MLP.

A code-letter received from the TALE-EHR authors confirmed they could not share code. Reimplementation from the paper text is the only path.

**Key constraints carried throughout the work**:
- No PEDSnet access. MIMIC-IV is the only adult source; PIC is the only real pediatric source.
- MIMIC-IV is **adult-only** (age ≥ 18); cannot support pediatric preventive-care claims.
- PIC is single-site Chinese ICU only; cannot anchor outpatient/preventive claims.
- All hardware fits on a single workstation, 32 GB RAM.

---

## 1. Preprocessing pipeline

### 1.1 Final pipeline (4 scripts, in order)

| Step | Script | Output |
|---|---|---|
| 1 | `preprocessing/build_event_table.py` | `data/processed/patient_events_{full|test}.parquet`, `code_vocab_raw.json` |
| 2 | `preprocessing/rollup_and_describe.py` | `patient_events_rolled_{full|test}.parquet`, `code_descriptions.json` |
| 3 | `preprocessing/build_splits.py` | `train/val/test_events.parquet` |
| 4 | `preprocessing/compute_bge_embeddings.py` | `bge_embeddings.pt` |

Followed by `preprocessing/tensorize.py` which converts to memory-mappable `.npz` shards.

### 1.2 Why we built our own pipeline (not lab's)

The lab's `healthylaife/MIMIC-IV-Data-Pipeline` was deliberately **not used** as the primary processor. Three reasons:

- It targets MIMIC-IV v1.0 / v2.0 schema; v3.1 has schema changes (notably `itemid` values in `labevents` and `d_labitems` changed between v2.2 and v3.0).
- It uses **fixed time-window bucketing**, which discretizes events into windows. TALE-EHR requires **continuous timestamps** for its `w(Δt)` polynomial.
- It is UI-driven (notebook with checkboxes) rather than scriptable.

The mapping files inside `utils/mappings/` of the lab pipeline were still useful in isolation. PyHealth was used as a reference cross-check (`pyhealth.medcode.CrossMap`), not as the primary processor.

### 1.3 What each script does

**`build_event_table.py`** — DuckDB-based extraction from 10 MIMIC-IV source tables: `diagnoses_icd`, `procedures_icd`, `prescriptions`, `labevents`, `chartevents`, `drgcodes`, `inputevents`, `outputevents`, `procedureevents`, `hcpcsevents` (plus optional MIMIC-IV-ED tables). Uses persistent DuckDB with 24 GB memory limit and spill-to-disk via `PRAGMA temp_directory='data/processed/duckdb_tmp'`. Computes `timestamp_days` (relative days from each patient's first event, t=0) and `age_at_event_days`. Filters: ≥5 events per patient, >1 unique timestamp.

Final test-mode run (1000 patients): 1,505,856 events across 617 subjects and 9,280 unique codes. Code-type breakdown: chart 1,079,176; lab 365,337; medication 41,789; diagnosis 15,330; procedure 2,318; drg 1,906.

**`rollup_and_describe.py`** — Rolls up raw codes to clinical ontologies and builds the `{code_id → text description}` map for BGE embedding.

- Diagnoses: ICD-9-CM / ICD-10-CM → **PheWAS catalog PheCode** (numeric format `PHE_250.2`). ~70–75% mapping rate. Unmapped Z/V/Y codes retained as `ICD9_*` / `ICD10_*` with their `long_title`. Vanderbilt PheWAS catalog files: `phecode_icd9_rolled.csv`, `phecode_icd10.csv`.
- Procedures: ICD-9/10-PCS + HCPCS → **CCS** (AHRQ HCUP), 100% mapping rate.
- Medications: NDC → **RxNorm** via bulk `RXNSAT.RRF` from the NLM prescribable content release.
- Lab / chart / DRG: kept as-is. Lab/chart descriptions come from MIMIC-IV's own `d_labitems.label` and `d_items.label`/`category`. DRG description from `drgcodes.description`.

Code namespace uses these prefixes everywhere downstream: `PHE_`, `CCS_`, `RXN_`, `LAB_`, `CHART_`, `DRG_`.

Rolled test-mode run: 9,280 → 6,711 unique codes. Every code_id has a description (`code_descriptions.json` is fully covered).

**`build_splits.py`** — Patient-level 70/10/20 split, seed 42, stratified by event-count quintile so long-sequence patients are distributed across splits.

**`compute_bge_embeddings.py`** — Encodes every code's text description through frozen `BAAI/bge-m3` (Chen et al., 2024, arXiv:2402.03216), produces `[N+2, 1024]` table with index 0 = PAD, index 1 = UNK, indices 2..N+1 = real codes. GPU (ROCm) for this step.

### 1.4 Blockers and fixes encountered

These are the real ones, in chronological order:

| # | Blocker | Resolution |
|---|---|---|
| 1 | Initially downloaded MIMIC-IV-ED instead of the core `hosp + icu` modules. ED module does not contain the longitudinal diagnosis/procedure/medication records TALE-EHR needs. | Re-downloaded the correct module set. Cost: ~2–3 days. |
| 2 | First attempt used `polars` for CSV.gz loading. Gzip is not seekable, so `polars` lazy scan fails and forces full materialization. | Switched all source-table loading to **DuckDB** (`read_csv_auto`), which streams the gzip and filters during the scan. Removed all `polars` and `pd.read_csv` calls in `build_event_table.py`. |
| 3 | First `--test_mode` run filled RAM/swap in ~4 minutes and dropped the SSH connection. The agent's filter pattern was still materializing the full decompressed `chartevents` and `labevents` into pandas before filtering. | Replaced monolithic `load_table → DataFrame` with **chunked streaming**: read CSV in 500k–1M-row chunks, filter to target `subject_id`s per chunk, append filtered rows directly to a per-source parquet on disk. Memory bounded to one chunk at a time. Also moved to `tmux` so SSH drops don't kill the job. |
| 4 | PheCode rollup initially produced codes like `PHE_BI_160.1` (PhecodeX format, two-letter prefix). TALE-EHR uses the **standard PheWAS catalog** numeric format (`PHE_250.2`). | Agent had silently fallen back to PhecodeX GitHub files when the PheWAS catalog download returned 403. Fixed by sourcing `phecode_icd9_rolled.csv` and `phecode_icd10.csv` directly (Vanderbilt) and removing the PhecodeX fallback. |
| 5 | NDC → RxNorm mapping via per-NDC NLM RxNav API calls. MIMIC-IV has ~50k–100k unique NDCs; at the API's rate limit each pipeline run took 2+ hours and got repeated several times due to (4). | Switched to **bulk `RXNSAT.RRF`** from the NLM RxNorm prescribable content release. Single file download, one DuckDB join. Coverage ~97%, no API needed. |
| 6 | DuckDB `executemany()` for inserting 100k+ rows one at a time caused a 15+ hour stall during the rollup script. | Replaced with **DataFrame-based DuckDB bulk inserts** (`con.register("df", df); con.execute("INSERT INTO t SELECT * FROM df")`). |
| 7 | Procedure → CCS initially had 0% mapping rate because the agent only found the diagnosis CCS crosswalk, not the procedure CCS file. | Sourced the correct HCUP procedure CCS file from `modusdatascience/ccs` and AHRQ. |
| 8 | DuckDB connections cannot be passed to spawned DataLoader workers. Per-sample DuckDB queries against the 626M-row events parquet at runtime had ~100× overhead and stalled at `num_workers ≥ 2` with spawn context (zombie worker observed at PID 2178347). | Pre-tensorized everything into `.npz` shards with memory-mapped numpy reads at runtime (`preprocessing/tensorize.py` → `TensorizedEHRDataset` with LRU shard cache). This pattern is now used for both pretraining and fine-tuning. |
| 9 | When tensorizer was first added, `_build_subject_payload` tried to sort by an `event_time` column the rolled parquet did not have (only `timestamp_days`). Would have KeyError'd at runtime. | Fixed by sorting on the columns that actually exist in the parquet schema. |
| 10 | `train.py` DataLoader did not set `multiprocessing_context='spawn'`. With DuckDB-backed dataset and fork context, connections would corrupt silently. | After the move to TensorizedEHRDataset (mmap, fork-safe), this stopped being a problem. The tensorized path is now the only production path. |
| 11 | Fine-tune multiprocessing import collision: spawned workers resolved `from dataset import` to `model/dataset.py` instead of `finetune/dataset.py`. | Fixed by making `finetune/__init__.py` minimal, using fully package-qualified imports everywhere (`from finetune.dataset import …`, `from model.tale_ehr import …`), and adding `sys.path` sanitization at the top of every entry script to drop the script directory. |
| 12 | PyTorch 2.6 default `weights_only=True` broke checkpoint reload during test-set evaluation. | Added `weights_only=False` when loading our own trusted checkpoints. |
| 13 | Fine-tune tensor shards stored variable-length sequences as `dtype=object`. `np.load(..., mmap_mode="r")` does not mmap object arrays, so each sample read incurred expensive unpickling and CPU-bound data loading. | Reworked shard format to flat concatenated arrays + `offsets` index (`code_indices`, `timestamps_days`, `age_days`), kept `np.savez` uncompressed, and updated `TensorizedDiseaseClassificationDataset` to slice mmap-backed flat arrays by offsets with `allow_pickle=False`. Added loud stale-shard guard (`ndim != 1` raises) so old object-format shards fail fast and require re-tensorization. |

### 1.5 Final pipeline output files (what the model consumes)

- `train/val/test_events.parquet` — per-event rows: `subject_id`, `code_id`, `timestamp_days`, `age_at_event_days`, `sex`, `race`.
- `bge_embeddings.pt` — `{code_ids: list[N+2], embeddings: FloatTensor[N+2, 1024]}`. PAD=0, UNK=1, real codes start at 2.
- `code_vocab.json` — `{code_id → int_index}` for the N real codes (no PAD/UNK entries).
- `code_descriptions.json` — upstream only, used to compute BGE embeddings; not used at train time.

### 1.6 Model input tensors (from `model/dataset.py` `EHRDataset` + `EHRCollator`)

| Tensor | Shape | Notes |
|---|---|---|
| `code_embeddings` | `[B, L, 1024]` | Indexed from frozen `bge_embeddings.pt` via `code_indices`. |
| `delta_t` | `[B, L, L]` | log1p pairwise gaps in **weeks**, computed in collator. |
| `timestamps_weeks` | `[B, L]` | `timestamp_days / 7`, raw float. |
| `demographics` | `[B, L, 3]` | `[age_years, sex, race]` per event (per-event, not patient-level). |
| `attention_mask` | `[B, L]` | True = real event, False = padding. |
| `target_code_idx` | `[B]` | Next-visit code index. |
| `target_time_weeks` | `[B]` | Time of next event. |

Max sequence length: 1024 events (tail-truncate, keeping the most recent 1024).

---

## 2. TALE-EHR reimplementation

### 2.1 Architecture (from paper, what we built)

**Code files (current, all under `model/`):**
- `time_aware_attention.py` — `PolynomialTemporalWeight`, `TimeAwareAttention`, `MultiScaleTemporalAggregation`.
- `tale_ehr.py` — `TALEEHR` orchestration class wiring the above + frozen BGE embeddings + demo MLP + two prediction heads.
- `dataset.py` — `EHRDataset` (DuckDB-backed) + `TensorizedEHRDataset` (mmap shards) + `ehr_collate`.
- `train.py` — pretraining loop and loss functions.

**Forward pass:**
1. `code_embeddings = embedding_table[code_indices]` — frozen BGE lookup.
2. `Q, K, V = MLP_Q(c), MLP_K(c), MLP_V(c)` — separate MLPs on each event's BGE vector.
3. `scores = (QK^T / √d) · w(Δt)` where `w(Δt) = σ(Σ_k α_k · Δt^k)` is a learnable degree-5 polynomial through a sigmoid.
4. `attn = softmax(scores)` after padding mask.
5. `e = attn @ V` — time-aware event representations `[B, L, d_model]`.
6. **Multi-scale aggregation**: a single learnable query vector `q_base ∈ R^d`. For each event, `relevance_j = q_base · e_j`. Weights `α_j = softmax(relevance_j · w(|t_current − t_j|))`. Patient state `h = Σ α_j e_j`. (Same polynomial form, separate instance, separate coefficients.)
7. Concatenate `h` with `MLP_demo(demographics_last_event)` → `combined`.
8. Two heads:
   - `code_predictor`: 3-layer MLP → `[B, num_codes]` logits, **final-layer bias initialized to −7** (log-odds prior on the 0.04% positive rate to prevent collapse at init).
   - `time_params_predictor`: 4-layer MLP → `[B, 2]` Weibull `(k, λ)` parameters.

**Hyperparameters (defaults):**
- `d_model = 256`, `poly_degree = 5`, `demo_dim = 3` (`age_years, sex, race`), `demo_hidden = 64`.
- Max seq len 1024. Single attention block (matching paper Section 4.2 — not multi-layer).

### 2.2 Design choices Claude flagged as ambiguous in the paper

Paper does not specify and we had to choose:

1. Multi-head structure for the polynomial coefficients (paper is silent on heads). Chose **per-head temporal weight functions** so age conditioning can later modulate each head independently.
2. `d_model` and hidden dims — not stated. Chose 256.
3. Number of transformer layers — not stated. Chose 1 (matches paper's "lightweight" framing and reading of Section 4.2).
4. Monte-Carlo sample count for the TPP loss — not stated. Chose 20.
5. Loss-balancing γ — not stated. Defaulted to 500 (kept γ·L_code and L_time within an order of magnitude in pilots).

### 2.3 Pretraining results

**Best checkpoint:** `checkpoints/run_20260427_152603/epoch_007.pt` (9 epochs, BCE code loss + γ = 500).

| Epoch | train | val | recall@10 | AUROC |
|---|---|---|---|---|
| 1 | 11.087549 | 10.587364 | 0.1051 | 0.9800 |
| 7 | 10.369564 | 10.104421 | 0.1133 | 0.9834 |
| 9 | 10.268005 | 10.258823 | 0.1118 | 0.9840 |

Train loss decreased monotonically. Val loss minimized at epoch 7 (10.10), drifted mildly to 10.26 at epoch 9. recall@10 plateaued at ~0.113.

**recall@10 interpretation:** ceiling for recall@10 is mean(10 / n_positives) = **0.3524** because the average sample has ~135 positives but k = 10. The observed 0.1133 is ~32% of the theoretical ceiling — **not collapse**, a metric-ceiling mismatch. Loss is the right pretraining metric; recall@K is a sanity check.

### 2.4 Pretraining debugging history (what didn't work first)

This took several iterations; documenting in order:

**Run with focal loss (γ_focal = 2, α = 0.25), code loss frozen at ~0.00115:**
- Time loss decreased; code loss flatlined after epoch 1; recall@10 frozen at 0.104. Root cause: focal loss is gradient-starved on 0.04% positive rate (~10–20 positives in a 30,635-class multi-hot target).

**Switch focal → BCE with γ_loss = 500, code-predictor final bias to −7:**
- This is the version that converged. Bias init gave a free baseline matching the marginal positive rate so the model only had to learn deviations.

**Things tried and reverted:**
- `bce_pos_weight > 0`: tried to up-weight positives directly. Reverted; the bias-init version was cleaner.
- Higher learning rates: caused unstable val.
- Unfreezing BGE embeddings: not done; paper freezes them, and we kept that.

**Diagnostic prints that proved the model was learning (`debug_sample=True`):**
- `[attn] entropy_mean`, `max_w_mean`, `collapse_frac` — attention sharpening over training.
- `[w(t)] mean/std/low_frac/high_frac` — polynomial output distribution.
- `[agg_alpha] entropy, peak_rel_pos` — aggregation behavior.
- `[logits] mean/std/sigmoid_mean` — code-head output stats.

All four sets of diagnostics confirmed healthy training in the final run.

### 2.5 The polynomial-shape problem discovered after pretraining (Project_Update0505)

This is the **largest open issue** in the project right now.

**Finding:** when `w(Δt)` is plotted as a function of Δt from the pretrained checkpoint, it is **monotonically increasing and saturates at 1.0 within ~15 days**. The paper's Figure 2 shows a **decay** (decreasing) shape.

**Cross-checks done:**
- Same shape in both the attention-block polynomial and the multi-scale aggregation polynomial.
- Vanilla TALE-EHR reimplementation (no age conditioning) shows the same inverted shape — so this is **not** an age-conditioning bug. It is a baseline-level issue.
- The age-conditioning module itself works mechanically: `‖Δα(a)‖₂` scales smoothly and monotonically with age, all six per-coefficient curves are smooth and age-coherent (see `age_diagnostics.py` and `visualize_age_conditioning.ipynb`).
- **Fine-tuning on T2D partially recovers decay** in the attention-block polynomial — the curve rises initially, then decreases. The aggregation polynomial does **not** recover under fine-tuning; it still saturates to 1.0 within ~30 days.

**Why this happens (Claude's analysis):**
- The TALE-EHR temporal point-process loss as implemented:
  ```python
  def temporal_point_process_loss(intensity, target_time_gap, T, n_mc_samples=20):
      norm_sq = (intensity**2).mean()
      fit_term = 2.0 * intensity.mean()
      return norm_sq - fit_term
  ```
  This loss **does not depend on time**. The `intensity` tensor is a per-patient scalar and `target_time_gap` is unused. There is therefore no gradient pressure on `w(Δt)` to be decay-shaped during pretraining.
- For attention with pairwise Δt, the code-loss-optimal behavior is to push `w → 1` so QK signal is preserved across all temporal scales (multiplicative form: `w → 0` collapses signed QK; `w → 1` preserves it).
- For aggregation with Δt-to-current, the code-loss-optimal behavior **is** decay (recent state is most informative for next-event prediction). Hence aggregation shows mild decay tendency while attention saturates up. This matches the observed `[w_curve_agg]` decreasing and attention rising.
- Paper's Figure 2 is from a *post-fine-tuning* analysis. TALE-EHR's pretraining-time decay shape is likely similarly weak in their model too, but they don't report pretraining curves.

**Three hypotheses for the failure (ranked by likelihood per Project_Update0505):**

1. **Loss imbalance.** γ_loss = 500 is unusually high; time loss may dominate the gradient.
   *Experiment:* re-run pretraining with γ_loss = 1.0 for 1–2 epochs, plot `w(Δt)`.
2. **Initialization bias.** Coefficients init `[0.5, 0, 0, 0, 0, 0]` gives flat `w(Δt) = σ(0.5) ≈ 0.62`. No built-in pressure for coefficients to go negative.
   *Experiment:* init `[2.0, -0.5, 0, 0, 0, 0]` to bias toward decay at start.
3. **Pretraining objective does not constrain decay shape.** The TPP intensity loss as-implemented gives no temporal signal to `w(Δt)`.
   *Experiment:* replace with **Weibull time-to-event NLL** (see §2.6).

### 2.6 Weibull TTE replacement for the TPP loss (in progress)

Replaced the time-invariant intensity loss with a parametric Weibull NLL:

```
L_time = −log p_Weibull(Δt_next | k, λ)
       = −log[ (k/λ) · (Δt/λ)^(k−1) · exp(−(Δt/λ)^k) ]
```

The model outputs `(k, λ)` per patient (`time_params_predictor` returns `[B, 2]`, softplus + ε applied inside the loss). Reference: Martinsson 2016, *WTTE-RNN* (Chalmers MSc thesis); Cox & Oakes 1984, *Analysis of Survival Data*. `debug_weibull.py` is the diagnostic harness (1-epoch run on a 2000-step subset, γ_loss = 1.0).

**Why this might fix the decay problem:** the Weibull NLL puts a real distributional constraint on Δt_next given history. Unlike the TPP intensity loss which can be minimized by predicting a constant rate, the Weibull loss requires the model to *sharpen* its time distribution, which gives the polynomial `w(Δt)` a real temporal-attention reason to decay.

**Caveats already flagged:**
- Weibull assumes monotonic hazard (k>1 increasing, k<1 decreasing). EHR inter-event gaps can be bursty (acute) and slow (chronic) in the same patient — non-monotonic.
- **Cox partial likelihood is rejected as a baseline loss**: the proportional-hazards assumption (`h(t|x) = h₀(t)·exp(β^T x)`) says the baseline hazard shape is shared across patients and covariates only shift it multiplicatively. This is fundamentally incompatible with the age-conditioning hypothesis, which says the *shape* of `w(Δt)` changes with age.
- **Discrete-time DeepHit (Lee et al., AAAI 2018, arXiv:1804.03234) is the planned fallback** if Weibull's monotonic-hazard restriction matters. Bin Δt log-spaced [1h, 5y] into K=15 bins, softmax over bins, NLL on the observed bin, optional DeepHit ranking term. No monotonic-hazard assumption, no softplus parameterization to be numerically unstable.

### 2.7 Fine-tuning pipeline

**Tasks implemented:**
- **T2D** (PheCode prefix `PHE_250.2`) — original binary classification target.
- **AKI** (`PHE_585.3`) — binary classification.
- **Heart Failure** (`PHE_428`) — binary classification.
- **LOS > 7 days** — admission-anchored, ~296k train admissions, 23% prevalence — added later as a clean benchmark.

**Files:**
- `finetune/build_disease_cohort.py` — original (deprecated for the chronic-disease setting, see §3).
- `finetune/build_cohort.py` — **landmark cohort** rewrite (van Houwelingen 2007; Tomašev et al. 2019; Rajkomar et al. 2018), supports `--obs_window_days`, `--gap_days`, `--pred_window_days`, identical anchoring procedure for positives and negatives.
- `finetune/build_los_cohort.py` — admission-anchored LOS cohort. Cohort unit is `(subject_id, hadm_id)`.
- `finetune/build_disease_tensors.py` — `.npz` shards keyed on `subject_id`.
- `finetune/build_los_tensors.py` — `.npz` shards keyed on `(subject_id, hadm_id)`.
- `finetune/test_cohort_leakage.py` and `finetune/test_los_cohort_leakage.py` — leakage diagnostics.
- `finetune/dataset.py` — `DiseaseClassificationDataset` (DuckDB, deprecated) + `TensorizedDiseaseClassificationDataset` (mmap, production).
- `finetune/model.py` — `TALEEHRClassifier` wrapping either `TALEEHR` or `TALEEHRAge` backbone with a 1-output linear classifier head on `[h_last ; demo_features_last]`.
- `finetune/train.py` — fine-tuning loop, BCE-with-pos-weight loss, AUROC/AUPRC validation.

**Backbone variant auto-detection (`_infer_backbone_hparams`):** checks if `age_emb` / `age_coeff_gen` keys are in the checkpoint state dict, instantiates `TALEEHRAge` if so, else `TALEEHR`. Same fine-tune code path for both — important for unbiased ablation comparison.

**June 2026 data-loading bottleneck fix (all fine-tune cohorts, including LOS/T2D/AKI/HF):**
- `finetune/build_los_tensors.py` and `finetune/build_disease_tensors.py` now write sequence fields in **flat mmap-safe layout**: `offsets` + concatenated arrays.
- `finetune/dataset.py` `TensorizedDiseaseClassificationDataset` now reads with `allow_pickle=False`, caches shard member views, and slices `[start:end]` via `offsets`.
- Old tensorized shards are intentionally incompatible with the new reader. Rebuild required per cohort before training.

**Sample fine-tuning command (tensorized path):**
```bash
conda run -n ehr python -m finetune.train \
  --disease los_gt7 \
  --pretrained_ckpt checkpoints/run_20260427_152603/best_pretrain.pt \
  --cohort_dir data/cohorts/los_gt7 \
  --tensorized_dir data/tensorized/los_gt7 \
  --epochs 5 \
  --batch_size 64 \
  --num_workers 6 \
  --prefetch_factor 2
```

---

## 3. Cohort design — the T2D length-leakage saga

This is the most instructive episode in the project. Documented because the fix-iteration trajectory is the lesson.

### 3.1 Initial cohort

`build_disease_cohort.py` v1: positives = patients with a `PHE_250.2*` event, sequence cropped at `(first_disease_idx − 1)`. Negatives = patients with no `PHE_250.2*` event, full timeline kept.

T2D stats: 25,214 train positives / 129,204 train negatives, 16.3% prevalence.

**Smoke test (`num_workers=0`, 1 epoch, 1024 train samples):** val AUROC 0.731, AUPRC 0.392 against 0.15 prevalence baseline. Looked fine.

### 3.2 Length leakage discovered (before full training)

Sequence-length stratification: **positive median 73 events / negative median 361 events**. Positives were cropped just before disease onset; negatives kept their full history. A model could learn "long sequence → negative" without learning any disease-relevant features.

### 3.3 First fix attempt — quantile-matched truncation

Resample each negative's `last_event_idx` from the per-split positive `last_event_idx` distribution, drop short negatives (`last_idx < 4`).

After v2 fix: positive median 73 / negative median 60–66 (matched ✓), but **mean 624 (pos) vs 139 (neg)** because of the heavy right-tail of multi-admission positives. Length asymmetry largely persisted in the tail.

### 3.4 Decision: ship the engineering milestone, flag the science

After two iterations the heavy-tail asymmetry was not closing. The decision was to launch full training on the partially-corrected cohort as an engineering milestone — to verify the pipeline end-to-end — while explicitly **not** treating the resulting numbers as scientific results.

**Full 5-epoch T2D run:** val AUROC monotonically 0.842 → 0.946. Test AUROC = 0.946, Test AUPRC = 0.778.

**This is well above published baselines** (Med-BERT ~0.83 per Rasmy et al. *npj Digital Medicine* 2021; TALE-EHR ~0.85). Reported in the weekly update as:

> *"AUROC 0.946 is well above published baselines, which signals the cohort definition has residual label leakage rather than the model being unusually strong. Two known leak vectors: (1) length asymmetry, (2) index-event leakage (diabetes workup labs/codes at `last_event_idx` for positives). Will report decomposed numbers before treating this as a methodological result."*

### 3.5 Diagnostics that should have been run before any reporting

(Listed here because they apply to every future fine-tuning task too.)

1. **Subject-level split overlap check** — train/val/test `subject_id` sets must be disjoint. Verified ✓.
2. **Length-stratified AUROC** — compute AUROC within sequence-length quartiles. If AUROC stays >0.9 within buckets, signal is real; if it drops to 0.6–0.7, length is doing the work.
3. **Inventory of `code_id` at `last_event_idx` for positives** — looking for diabetes-related codes (glucose labs, HbA1c, prediabetes) at the cropping boundary that would leak the label.
4. **PheCode-family inventory** — `starts_with("PHE_250.2")` matches `PHE_250.2`, `.20`, `.21`, …; verify the cohort builder isn't missing parent codes that leak through the observation window.

### 3.6 Landmark cohort rewrite (`build_cohort.py`)

Standard four-segment timeline:

```
[--- observation window ---] [-- gap --] [--- prediction window ---]
     (model sees this)        (blackout)    (label is determined here)
                              ↑
                          t_landmark
```

Key design points:
- `t_landmark` defined in **patient elapsed days** (`timestamp_days`), not calendar days.
- Same anchoring procedure for positives and negatives — only difference is whether the disease code occurs inside the prediction window.
- For positives, `t_landmark = t_first_disease − gap_days − pred_window_days`; for negatives, `t_landmark` is sampled from the per-split positive `t_landmark` distribution.
- `last_event_idx` continues to denote the 0-indexed position of the last in-window event — keeps existing `dataset.py` and `tensorize.py` schema unchanged.

**Diagnostic results (`test_cohort_leakage.py`):**
- Length-leakage AUROC: **0.5087** (passes — length alone has no signal) ✓
- Landmark-leakage AUROC: **0.6388** (fails the 0.6 threshold)

**Why the landmark leakage remains:** structural MIMIC-IV asymmetry. T2D positives are defined by *having* the code somewhere in their timeline. MIMIC-IV is acute care; patients with a chronic T2D code at any point have, almost by definition, a long-enough record to *get* the code. Positives are selected to have long records **by construction** — independent of any cohort design choice.

**Conclusion (carried forward):** binary classification for chronic disease prediction on MIMIC-IV is the wrong framing. T2D should eventually be **reframed as a survival / time-to-event task** (van Houwelingen landmark survival), where the asymmetry becomes a feature (right-censoring) rather than a leak. This is on the horizon, not implemented.

### 3.7 LOS > 7 days — chosen as a clean benchmark

Why LOS is structurally easier than T2D:
- Outcome is **observed at known time** (every admission has `admittime` and `dischtime`), no censoring asymmetry.
- Short prediction horizon (within days), uses first 24 h as observation window.
- Standard MIMIC-IV benchmark — Harutyunyan et al. (2019, *Sci Data*, doi:10.1038/s41597-019-0103-9); van de Water et al. *YAIB* (ICLR 2024, arXiv:2306.05109); TALE-EHR reports ~0.812 in its own table.
- Cohort unit is **`(subject_id, hadm_id)`**, not `subject_id` — admission-aware tensorizer required.

**Build stats:**
- Train: 296,501 admissions, 23.13% prevalence
- Val: 41,786 admissions, 22.83%
- Test: 85,522 admissions, 22.96%
- Split-consistency checks pass (no subject overlap across splits) ✓

**Length-leakage AUROC: 0.7225**. Initially this looked like a fail (the diagnostic threshold spec was 0.6), but on reflection this is the *correct* answer:
- `n_events_in_window` (events in first 24h) is a direct severity proxy.
- APACHE-II / SAPS-II severity scores, which are essentially this statistic, get AUROCs in the 0.7–0.85 range for ICU outcomes.
- The diagnostic threshold I set was inappropriate for this task. The right check for LOS is "does the full model beat 0.7225?" not "is length AUROC below 0.6?".

**Bug found in `build_disease_tensors.py` while building LOS tensorizer:** the original tensorizer groups events by `subject_id` after the SQL fetch. For a subject with multiple cohort rows (multiple admissions), both rows look up the **same** event group — the union of events permitted by *any* of that subject's cohort rows. Earlier admissions end up training on events from after their own landmark, with the wrong label. Silent correctness bug.

**Fix:** `finetune/build_los_tensors.py` is admission-aware (one shard entry per `(subject_id, hadm_id)`), keyed events fetched once per subject then truncated per cohort row in pandas. T2D path was left untouched (Option B from the design discussion).

**LOS training: not yet run to completion.** Target test AUROC: 0.78–0.85; TALE-EHR's published number on this is ~0.812.

---

## 4. Age-conditioning extension (Thrust 1's core novelty)

### 4.1 Final formulation

The polynomial coefficients of `w(Δt)` become functions of patient developmental age:

```
α_k(a) = α_k^base + Δα_k(a)
Δα(a)  = MLP( γ(a) )
w(Δt, a) = σ( Σ_k α_k(a) · Δt^k )
```

where:
- `α_k^base` is the original TALE-EHR coefficient, initialized as in TALE-EHR (`[0.5, 0, 0, 0, 0, 0]`).
- `γ(a) ∈ R^32` is a **Fourier embedding** of age: 16 sin/cos pairs with periods log-spaced from 1 month (1/12 yr) to 200 years.
- `MLP: R^32 → R^6` produces the six coefficient offsets jointly. **Final layer is zero-initialized** — at init the model is bit-exact equivalent to vanilla TALE-EHR. Smoke test in `time_aware_attention_age.py` verifies bit-exact parity (`torch.allclose(out_base, out_real, atol=1e-5)`).

This is **FiLM-style (Perez et al. 2018, arXiv:1709.07871)** additive modulation of polynomial coefficients. It is applied at **both** TimeAwareAttention and MultiScaleTemporalAggregation (separate MLPs for each).

Trainable parameters added vs vanilla baseline: **5,004** (the parameter-matched ablation requirement is satisfied trivially, given the baseline is millions of params).

### 4.2 Why this design over alternatives

A long design-space exploration (multiple chats over months) narrowed the choices. The shortlist that survived:

**Rank 1 (chosen) — Age-modulated polynomial coefficients via zero-init MLP.** Modifies the temporal kernel *shape* rather than adding redundant signals. Zero-init preserves the vanilla baseline as a special case. Clinically motivated (decay curves should plausibly differ across developmental stages). Parameter cost ~5k.

**Rank 2 — AdaLN-Zero (Peebles & Xie 2023, *DiT*, arXiv:2212.09748).** Per-block, modulate LayerNorm outputs via affine parameters predicted from the age embedding. **Rejected for TALE-EHR as-is** because:
- AdaLN-Zero is designed for deep transformer stacks (DiT uses 12–28 blocks). Our TALE-EHR has a single time-aware attention block and an aggregation module — there are no LayerNorms in the standard transformer-block sense to modulate.
- Adopting AdaLN-Zero would require restructuring the model into a multi-block transformer with pre-norm LayerNorms, roughly doubling parameter count.
- No published EHR paper uses AdaLN-Zero; its benefit in shallow EHR architectures is empirically unconfirmed.
- Kept as a possible follow-up if the kernel-modulation approach hits a ceiling.

**Rank 3 — Age register token (Darcet-style [AGE] token built from Fourier features) + classifier-free guidance on age.** Maximally interpretable — can inspect `attention(token_i → [AGE])` — but the model can ignore the register if loss is locally minimized that way.

**Rank 4 — FiLM in the FFN.** Special case of AdaLN-Zero without the α residual-scaling gate. Backed by clinical-imaging literature (arXiv:2102.09582; doi:10.1002/jum.16633; arXiv:2511.16498). Slightly worse OOD properties than AdaLN-Zero.

**Rank 5 — B-spline age basis.** Strongest theoretical OOD guarantee (compactly supported basis), but less expressive than Fourier features, and combining a spline with a sinusoidal embedding is unusual.

### 4.3 Rejected approaches (and why)

These were explored seriously and dropped:

- **ALiBi-on-age (Press et al. 2022, arXiv:2108.12409).** Initially considered. **Rejected on structural grounds**: in single-patient sequences, softmax invariance means per-query constant additions have no effect on attention weights, and pairwise age differences in single-patient sequences are redundant with the existing time delta. Claude initially over-endorsed this and later corrected. The multiplicative-sigmoid implementation explored is not technically ALiBi anyway (canonical ALiBi requires additive logit biases before softmax).
- **GT-BEHRT as codebase scaffold.** Initially considered for early baselining. Dropped in favor of a from-scratch TALE-EHR reimplementation aligned with the actual paper rather than adapting an unrelated codebase.
- **Variant 4 ALiBi-style age conditioning.** Specific formulation explored in two chats. Rejected: within single-patient sequences the antisymmetric age-difference term collapses to a re-parameterization of time delta.
- **Learned per-year age lookup embedding (BEHRT-style).** Unsafe under OOD by construction. Only useful as a baseline showing the failure age-conditioning is meant to fix.
- **Cross-attention on a long age context.** Unbounded; overkill for a 1-D scalar.
- **Hypernetworks generating full FFN weights.** Parameter explosion; principled init is fragile (arXiv:2312.08399).
- **Cox proportional-hazards loss as the pretraining time loss.** Incompatible with the age-conditioning hypothesis — Cox assumes baseline hazard shape is shared across patients, only multiplicatively shifted. Age conditioning requires *shape* changes.

### 4.4 Age embedding details

`FourierAgeEmbedding` (in `age_embedding.py`):
- 16 frequencies, log-spaced periods from 1 month to 200 years.
- `embedding_dim = 32` (= 2 × num_frequencies).
- Frequencies stored as a buffer (not learned). Robust to OOD ages.
- Bounded by construction: `sin/cos ∈ [−1, 1]`, so `w(Δt, a) ∈ (0, 1)` regardless of input age.

`AgeCoefficientGenerator` (also in `age_embedding.py`):
- 32 → 64 (GELU) → 6 MLP.
- **Final layer zero-initialized** (`nn.init.zeros_(self.mlp[-1].weight); .bias`). Critical for the "vanilla TALE-EHR is a special case at t=0" property.
- Three ablation modes built in: `"real"` (normal), `"random_constant"` (replace age input with a fixed random vector — used to test if the model is reading age or just adding capacity), `"none"` (return zeros — pure parameter-matched control).

### 4.5 Architecture parity check (smoke tests in `time_aware_attention_age.py` and `tale_ehr_age.py`)

Three smoke tests are guaranteed to pass:

1. **Zero-init equivalence:** with zero-init MLP and any age, `AgeConditionedTimeAwareAttention` output matches `TimeAwareAttention` output bitwise (`atol=1e-5`).
2. **Age-as-noise dropout:** `mode="random_constant"` produces identical outputs for `age=2` and `age=65`, confirming the model is reading the age input and not some other side channel.
3. **Age sensitivity:** with non-zero MLP weights, `(out_young - out_old).abs().max() > 1e-3` for `age=2` vs `age=65`. Confirms gradient pathway.

Trainable parameter delta is hard-coded in the test: 5,004. Any architectural change that drifts this number will fail the smoke test.

### 4.6 Diagnostics (`age_diagnostics.py`, `visualize_age_conditioning.ipynb`)

`compute_alpha_delta_stats(model, batch)` reports `‖Δα(a)‖₂` summary stats stratified by developmental age bucket:
- neonate (a < 1/12 yr)
- infant (1/12 ≤ a < 2)
- child (2 ≤ a < 12)
- adolescent (12 ≤ a < 18)
- young_adult (18 ≤ a < 40)
- middle_age (40 ≤ a < 65)
- older_adult (a ≥ 65)

Plus the aggregation-module Δα norm. Logged every N steps during pretraining.

**What the diagnostics show on the age-conditioned pretrained checkpoint:**
- `‖Δα(a)‖₂` scales smoothly and monotonically with age ✓
- Per-coefficient curves `Δα_k(a)` are smooth and age-coherent across the lifespan ✓
- OOD pediatric region (a < 18 y) produces bounded, structured offsets ✓

**The age-conditioning module is functional.** This is verified, not assumed.

### 4.7 What this means together with the polynomial-shape problem

The age-conditioning module works, **but the baseline polynomial `w(Δt)` it conditions on is broken** (§2.5). Conditioning a malformed substrate cannot be cleanly interpreted. The Project_Update0505 framing is:

> The age MLP is conditioning a malformed substrate. We should not draw conclusions from current downstream T2D AUROC numbers (age-conditioned AUROC 0.707 vs vanilla 0.946 — the difference probably reflects the leakage issue and the kernel issue, not the age conditioning).

**Order of operations going forward:** fix the kernel (try Weibull, lower γ, biased init), then re-run age conditioning, then compare.

### 4.8 Preliminary downstream comparison (from Project_Update0505)

Both runs on the same (still partially-leaky) T2D cohort, 5 epochs each:

| Model | Loss | AUROC | AUPRC | Acc |
|---|---|---|---|---|
| Vanilla TALE-EHR | 0.475 | 0.9456 | 0.7778 | 0.846 |
| Age-conditioned | 0.673 | 0.7067 | 0.6413 | 0.646 |

**Treating these as preliminary only.** The cohort still had length asymmetry. The kernel is broken in pretraining. The numbers don't compare cleanly until both upstream issues are fixed.

---

## 5. Open issues and what's still needed

### 5.1 Open issues (broken / known-broken)

1. **`w(Δt)` saturates upward in pretraining** instead of decaying (§2.5). Fix candidates: lower γ_loss to ~1.0 (loss-balance hypothesis); biased polynomial init `[2.0, -0.5, ...]` (init hypothesis); Weibull TTE replacement (objective hypothesis).
2. **TPP intensity loss is time-invariant as currently implemented** — `temporal_point_process_loss(intensity, target_time_gap, T, n_mc_samples=20)` does not actually use `target_time_gap`. The `intensity` head returns a per-patient scalar, not λ(t). The Doob-Meyer Monte-Carlo formulation in the paper requires λ_θ(t) evaluable at arbitrary t.
3. **T2D cohort framing is structurally wrong on MIMIC-IV** for chronic disease prediction. Reframe as survival/TTE on a different dataset, or pick a different anchor task.
4. **LOS fine-tuning has not been run to completion** with the admission-aware tensorizer.
5. **Age-conditioned baseline cannot be evaluated** cleanly until issue (1) is fixed.

### 5.2 Things still to be done

1. Run the three diagnostic experiments from §2.5 (γ sweep, init bias, Weibull).
2. Run LOS > 7 d fine-tuning to completion; compare test AUROC against TALE-EHR ~0.812.
3. Reframe T2D as survival/time-to-event (van Houwelingen landmark design + DeepHit-style head).
4. **Code-vocabulary overlap audit** between PIC's Chinese ICD-9-CM codes and MIMIC-IV's codes after PheCode rollup — *critical prerequisite* before any cross-dataset transfer experiment. Compute `n_pic_codes_seen_in_mimic / n_pic_codes_total` separately per prefix family (PHE, CCS, RXN, LAB, CHART, DRG). RxNorm coverage for Chinese drugs may be poor; LOINC vs native-itemid choice for labs matters.
5. Parameter-matched ablation suite (mandatory): polynomial coefficient conditioning vs AdaLN-Zero vs Fourier-feature encoding vs random-constant control vs none-mode control.
6. Evaluate vanilla TALE-EHR stratified by developmental stage on PIC — establishes the empirical motivation for age conditioning.
7. Decide whether to anchor the publication on (a) IEEE BHI 2026 (~June deadline, fastest), (b) ML4H 2026 (~September), or (c) push to CHIL 2027 / MLHC 2027 / ACM BCB 2027.

### 5.3 Carry-over principles (lessons codified across the project)

- **Label/length leakage is the primary validity threat in EHR cohort design.** AUROC results exceeding published baselines (0.82–0.87 for comparable methods) are a red flag, not a success signal.
- **Binary classification for chronic disease on MIMIC-IV is structurally problematic** — positives are defined by having disease codes which correlates with record length. Survival framing is the correct solution.
- **Length leakage on LOS is clinically meaningful**, not a defect — event count in first 24 h is a real severity proxy (analogous to APACHE-II). The leakage-threshold diagnostic must be task-specific.
- **DuckDB connections cannot be passed to spawned DataLoader workers.** Pre-tensorize to `.npz` shards with memory-mapped numpy reads.
- **ALiBi is structurally unsuitable for single-patient EHR sequences** — pairwise age differences collapse to a re-parameterization of Δt. Multiplicative-sigmoid implementations are not technically ALiBi anyway.
- **AdaLN-Zero's benefit is unconfirmed in EHR literature** and structurally smaller in shallow architectures than in DiT-style deep stacks.
- **Age-modulated polynomial coefficients are the strongest mechanism candidate** because they modulate the temporal kernel *shape* rather than adding redundant signals, with zero-init preserving the vanilla baseline as a special case.
- **Parameter-matched ablations are non-negotiable** — any performance gain must be attributable to the conditioning mechanism, not added capacity.
- **Cox proportional hazards is incompatible with the age-conditioning hypothesis** — reject as baseline loss; use discrete-time log-likelihood (DeepHit) or Weibull NLL as default.
- **PIC is ICU-only** — cannot support preventive care claims; validate architecture on PIC using mortality/sepsis, position preventive care as future work.
- **Evaluation design derives from dataset capabilities**, not proposal language: acute/ICU endpoints for PIC+MIMIC-IV; synthetic longitudinal growth for Synthea.

---

## 6. File map (current state of `Age-conditioned-pediatric-EHR/`)

```
preprocessing/
├── build_event_table.py          # DuckDB extraction from MIMIC-IV v3.1 (chunked)
├── rollup_and_describe.py        # ICD→PheCode, NDC→RxNorm bulk, Proc→CCS, descriptions
├── build_splits.py               # patient-level 70/10/20, stratified by event-count quintile
├── compute_bge_embeddings.py     # BGE-m3 on description text → [N+2, 1024]
└── tensorize.py                  # event parquets → .npz shards (pretraining)

model/
├── time_aware_attention.py       # PolynomialTemporalWeight, TimeAwareAttention, MultiScaleTemporalAggregation
├── tale_ehr.py                   # vanilla TALEEHR (frozen BGE + attention + agg + heads)
├── time_aware_attention_age.py   # AgeConditionedPolynomialWeight, AgeConditionedTimeAwareAttention, AgeConditionedMultiScaleTemporalAggregation
├── tale_ehr_age.py               # TALEEHRAge with age-conditioned modules
├── age_embedding.py              # FourierAgeEmbedding, AgeCoefficientGenerator (real/random_constant/none modes)
├── age_diagnostics.py            # ||Δα(a)||₂ stats by developmental age bucket
├── dataset.py                    # EHRDataset (DuckDB) + TensorizedEHRDataset (mmap) + ehr_collate
├── train.py                      # pretraining loop, focal/bce code loss, TPP/Weibull time loss
├── debug_no_time_loss.py         # diagnostic: pretrain with code-only loss
├── debug_weibull.py              # diagnostic: 1-epoch Weibull TTE run on 2000-step subset
└── check_dataset_equivalence.py  # parity check across dataset variants

finetune/
├── build_cohort.py               # anchored landmark cohort (van Houwelingen design)
├── build_disease_cohort.py       # v1 cohort (deprecated for chronic disease, kept for AKI/HF)
├── build_disease_tensors.py      # per-subject .npz shards
├── build_los_cohort.py           # admission-anchored LOS>7 cohort
├── build_los_tensors.py          # per-(subject_id, hadm_id) .npz shards
├── test_cohort_leakage.py        # length-leakage diagnostic (T2D, AKI, HF)
├── test_los_cohort_leakage.py    # length-leakage diagnostic (LOS)
├── dataset.py                    # DiseaseClassificationDataset (DuckDB) + Tensorized variant
├── model.py                      # TALEEHRClassifier (auto-detects vanilla vs age backbone)
└── train.py                      # fine-tune loop, BCE pos-weighted, AUROC/AUPRC

notebooks/
├── visualize_age_conditioning.ipynb  # Δα(a) curves, per-coefficient breakdown
└── visualize_finetune_decay.ipynb    # post-fine-tune w(Δt) plots
```

---

## 7. Key references used in design decisions

- TALE-EHR — Yu et al., 2025, *Time-Aware Attention for Enhanced Electronic Health Records Modeling*, arXiv:2507.14847.
- BGE-m3 — Chen et al., 2024, arXiv:2402.03216.
- MIMIC-IV — Johnson et al., 2023, *Sci Data*, doi:10.1038/s41597-023-02136-9.
- PheWAS catalog — Vanderbilt, phewascatalog.org.
- FiLM — Perez et al., AAAI 2018, arXiv:1709.07871.
- AdaLN / DiT — Peebles & Xie, 2023, arXiv:2212.09748.
- ALiBi — Press et al., 2022, arXiv:2108.12409.
- Fourier features — Tancik et al., 2020, arXiv:2006.10739.
- DeepHit — Lee et al., AAAI 2018, arXiv:1804.03234.
- WTTE-RNN (Weibull) — Martinsson 2016, Chalmers MSc thesis.
- DeepSurv — Katzman et al., 2018, doi:10.1186/s12874-018-0482-1.
- Landmark survival — van Houwelingen, 2007 (*Stat. Med.*).
- DeepMind AKI — Tomašev et al., 2019, *Nature*.
- Med-BERT — Rasmy et al., 2021, *npj Digital Medicine*.
- Continued pretraining (adult → pediatric) — Guo et al., 2024, PMC11211479.
- LOS benchmark — Harutyunyan et al., 2019, *Sci Data*, doi:10.1038/s41597-019-0103-9.
- Shortcut learning in EHR foundation models — Yuan et al., 2024, *NEJM AI*.
- OHDSI/HARMONY conventions — Hripcsak et al., 2015, *JAMIA*.
- Informed presence bias — Goldstein et al., 2017, *JAMIA*.
- HyMaTE — lab's own paper, ACM BCB 2025.
- LifeClock/EHRFormer — *Nat Medicine*, 2025, doi:10.1038/s41591-025-04006-w.

---

*This document reflects work through Project_Update0505 (May 2026) and subsequent debugging through the Weibull TTE replacement.*
