# `model_new/` readiness for PIC fine-tuning

Generated alongside `audit.json` (`python -m model_new.pic_audit`). Every claim below is
anchored to a function name and, where it is a measurement, to a field in `audit.json`.

**Verdict in one line.** The data path, the model path and the invariant machinery are
ready and were exercised end to end. Two things are not: the fine-tune path has no
implementation of either vocabulary-transfer route (`DECISION D3`), and the kernel has
essentially no usable degrees of freedom on the 24 h PIC shards (`DECISION D1`). Neither is
a code defect; both are decisions.

---

## 1. `data_finetune.py`

| question | answer |
|---|---|
| does the dataset read PIC shards as-is? | **Yes.** `TensorizedFinetuneDataset._load_shard` reads `offsets, code_indices, timestamps_days, age_days, subject_id, sex, race, label, unk_vocab_index` and optionally `hadm_id`. Every one is present in all 12 PIC shards (`A1_schema.per_split.*.missing_required == []`). |
| does it emit `age_years` as a separate `[B, L]` tensor (D3)? | **Yes.** `data._pad_common` builds `age_years = age_days / DAYS_PER_YEAR` masked by validity and returns it as its own key; `_build_demographics` *additionally* copies it into channel 0. `DKMModel._check_batch` raises if `age_years` is absent. No module reads age out of `demographics`. |
| does it use float64 differencing (Fix D)? | **Yes, for the arithmetic.** `_pad_common` allocates `ts_np` as `float64`; `data.pairwise_tau` casts to `.double()` before differencing and back to float32 only after `log1p`. **Storage is float32** in both the shard and `__getitem__`'s cast — irreducible, and harmless here: PIC timestamps are ≤ 1.0 d, where the float32 ulp is ~5 ms. |
| does it build the padding-only mask via `encoder.build_key_mask` (D4)? | **Indirectly, and correctly.** The collate emits `attention_mask` (and `lengths`); the masks are built at the point of use — `encoder.build_pair_mask` in `TimeAwareAttention.forward`, `encoder.build_key_mask` in `AttentionPooling.forward`. There is no `tril` anywhere in `model_new/`. |
| unconsumed fields | `n_events_in_window` is present in every PIC shard and is **not** read by `model_new` (the legacy `finetune/dataset.py` returned it in every item and put it in the batch). It equals `diff(offsets)` exactly, so nothing is lost. `hadm_id` is read and carried through the collate but used only for provenance. |
| missing fields | none. There is no `split` field — the split is the containing directory, which is how both trees have always done it. |

**One real gap.** `TensorizedFinetuneDataset.__getitem__` returns `code_indices` unchanged
and `unk_vocab_index` straight from the shard (2198, the PIC vocabulary size). There is no
reindexing hook. That is `DECISION D3` and is discussed in §6.

## 2. `model.py` with `task='classification'`

- **The head sits on pooled `h`.** `DKMModel.forward` runs `Encoder → AttentionPooling →
  concat([h, demo_proj(demo_last)] (+ `age_delta` for `additive`)) → head`, identically for
  both tasks; only `out_dim` differs (`num_codes` vs `1`). **There is no
  `return_repr_only` path anywhere in `model_new/`** — D9 is structurally impossible, not
  merely avoided.
- **The pooling-site age generator is on the gradient path.** `AttentionPooling.forward`
  computes `alpha = alpha_base + self.age(age_last)` and injects `relevance + log w`; `h`
  is the head's input, so `∂loss/∂(pooling age params) ≠ 0` by construction. This is the
  legacy defect in its second disguise — fast-LR group, no gradient — and it cannot recur
  here, because `optim.build_param_groups` takes the age group from
  `module.age_parameters()` and `DKMModel.forward` has exactly one route from pooling to
  the loss. The smoke run in §7 confirms it empirically: the `additive_head` and pooling
  generators report `frac_nonzero_grad_gt_0 = 1.0`.
- `task` is validated in `__init__` (`{"pretrain", "classification"}`) and recorded in
  `config_dict()["task"]`, which `eval_finetune.build_model` asserts on.

## 3. `train_finetune.py`

| question | before this pass | now |
|---|---|---|
| `τ_max` from the checkpoint, asserted not recomputed (D8, §11.3)? | Yes — `resolve_tau_max` + a post-load equality check. | Unchanged, plus `assert_frozen_constants` re-checks every kernel site's buffer against `ckpt["tau_max"]` *and* against `config.model.tau_max`. |
| age standardization from the checkpoint? | Yes — `resolve_age_standardization`. | Unchanged, plus the same bit-identity check post-load. |
| Fourier buffers / race ordering / `s` from the checkpoint? | **No — never checked.** `load_backbone`'s `strict=False` tolerates an absent buffer, so a rebuilt Fourier band would have loaded silently. | `assert_frozen_constants` (INV-FT-FROZEN). |
| `optim.build_param_groups`, so the age group exists at fine-tune LR? | Yes. `build_param_groups(model, lr_backbone, lr_age, lr_head)` and `Adam(groups)`; the group report goes into `config.json`. | Unchanged. Defaults `1e-5 / 1e-3 / 1e-3` — the age group runs 100× the backbone LR. |
| arm identity | `--arm` was `required=True` and used verbatim. A `kernel` script pointed at the `vanilla` checkpoint would have produced a complete, plausible, meaningless run. | `resolve_arm_from_checkpoint` (INV-FT-ARM). `--arm` is now optional and exists only to make a mismatch loud; the shared-vanilla design must say `--allow_arm_mismatch`. |
| batch-order determinism across arms | **Broken.** `DataLoader(shuffle=True)` used the global RNG, and constructing the age modules consumes an arm-dependent number of draws, so the four arms shuffled differently. | An owned `torch.Generator(seed)`, `train_order_hash`, `eval_order_hash`, and `assert_order_matches_siblings` (INV-FT-ORDER). |
| age bands | `D.band_index` / `D.band_names` with the adult table — youngest band 12–17, which holds ~4% of a PIC cohort. | `--band_table pediatric` threads `PEDIATRIC_AGE_BANDS` through every stratified metric. |
| pre-run declarations | none | `pic_config.json`, written before the first step. |

## 4. Is there an equivalent of `eval_pretrain.py` for fine-tuning?

There was not. There is now: **`model_new/eval_finetune.py`**, implemented and structurally
smoke-tested but **not run on real fine-tuned checkpoints** (none exist yet for PIC). It
mirrors all four structural properties of `eval_pretrain.py` and adds patient-level
bootstrap CIs and paired per-patient deltas, which a 1,280-sequence validation split
requires. See §7.

## 5. Phase B — what the measurements say

### M1 reproduces, and it is the cohort window

| quantity | value (all four tasks) | MIMIC |
|---|---|---|
| τ̃ range under the frozen `τ_max = 6.7238` | `[-1.0000, -0.9603]` | `[-1.0000, 0.9780]` |
| fraction of `[-1, 1]` occupied | **1.99%** | 98.9% |
| Chebyshev Gram condition (no `T₀`) | **6e14 – 7e15** | **15.6** |
| clamp rate | 0.0 | 0.0 |
| within-row τ spread, median | **0.107** | 4.46 |
| rows with spread < 0.1 | **35–36%** | 1.5% |

README §5c quotes 5.7e16 for `heart_malformations`; this pass measures 6.1e14 on 5.27M
pairs. Both are numerically singular — on a matrix this ill-conditioned the reported value
is float noise, not a stable quantity — so **the figure to quote is the occupancy (2.0%),
not the condition number.** That is a reporting change, not a disagreement.

The 35% of attention rows with spread < 0.1 is the number that matters more than the
condition number: softmax ignores a per-row constant, so on a third of all rows the kernel
has nothing to discriminate *whatever* `α` is.

### B4 settles what M1 is

`audit.json → per_task.<task>.B4_unclipped`, on the same cohort subjects from
`data/processed/pic/train_events.parquet`:

| window | events/seq (p50) | span p50 | τ̃ occupancy | Gram cond | spread p50 | rows < 0.1 |
|---|---|---|---|---|---|---|
| 1 d (shipped) | 133 | 0.96 d | 1.99% | 4.4e15 | 0.108 | 34.8% |
| 3 d | 279 | 2.92 d | 5.30% | 7.4e12 | 0.267 | 0.7% |
| 7 d | 468 | 6.87 d | 10.31% | 1.8e10 | 0.543 | 0.2% |
| 30 d | 656 | 13.4 d | 24.75% | 1.1e7 | 1.147 | 0.1% |
| full stay | 677 | 14.1 d | **75.10%** | **2.1e3** | 1.322 | 0.07% |

(`mortality`; the other three are within a percent. Ratios: 37.8×, 37.7×, 35.6×, 37.7×.)

**M1 is a consequence of `OBS_WINDOW_DAYS = 1.0`, not of the frozen `τ_max`.** At the full
stay the same frozen `τ_max` gives 75% occupancy and a condition number of ~2e3 — four
orders of magnitude better and in the same regime as MIMIC's 15.6.

**But the leakage-free direction is empty.** `pre_index_history.fraction_of_admissions_
with_any = 0.0000` for every task: not one sampled admission has a single event before its
own first event. PIC subjects have exactly one admission each (n_sequences == n_subjects in
every split), so *widening the window backwards buys nothing*. The 37× is bought entirely
by extending **forward**, which leaks the outcome for `mortality` and `los_gt7`. The audit
labels that block `forward_sweep_caveat` and does not propose it.

So the option set for `DECISION D1` is narrower than README §5c implies: the data-side fix
exists but requires **re-deriving the cohorts** with a longer observation window and an
outcome defined strictly after it (e.g. "predict from the first 7 days, outcome from day 8
onward"), not merely re-tensorizing.

### B3 — headroom on real PIC batches

`audit.json → per_task.mortality.B3_headroom`. Equal-norm probe (`preflight.headroom`),
6 batches × 8 sequences, each arm's own loaded backbone:

| arm | reindexed into MIMIC vocab | with the PIC BGE table |
|---|---|---|
| vanilla | 0.0141 / 0.0099 | 0.0555 / 0.382 |
| kernel | 0.0083 / 0.0077 | 0.0457 / 0.293 |
| random_constant | 0.0014 / 0.0009 | 0.0957 / 0.573 |
| additive | 0.0178 / 0.0121 | 0.0463 / 0.299 |

(`max|Δlogit|` / `max per logit sd`.) Against README §5's MIMIC-at-init figure of
0.0990 / 1.44 and preflight's stored PIC reference of 0.0059. Under the PIC table the
kernel retains roughly half the absolute authority it has on MIMIC but a fifth of the
relative authority; under reindexing it has almost none, which is a second, independent
reason option (a) is not viable.

## 6. `DECISION D3` — vocabulary and embedding table

`A4_vocab`. PIC has 2,198 codes; **545 (24.8%)** appear in the MIMIC vocabulary, and the
overlap is one family:

| family | PIC codes | in MIMIC |
|---|---|---|
| `PHE_` | 565 | 545 (96.5%) |
| `LAB_` | 821 | 0 |
| `DRUG_` | 638 | 0 |
| `ICD10_` | 119 | 0 |
| `EXAM_` | 36 | 0 |
| `CHART_` | 19 | 0 |

The `PHE_` prefix is a shared phecode namespace; every other PIC family uses institution-
local identifiers that share no string with the MIMIC vocabulary. Neither corpus uses the
`CCS_`/`RXN_`/`DRG_` families the brief asked about on the PIC side (`RXN_`, `DRG_`, `CCS_`
exist only in MIMIC; PIC uses `DRUG_`, `EXAM_`, `ICD10_`).

**Token-level UNK rate under option (a): 0.9899 – 0.9933 across every task and split.**
Reindexing maps 99% of PIC events to `[UNK]`. That is far past the ~20% threshold: **option
(a) is not viable and should be stated plainly rather than measured further.**

Option (b) is the only route, and it has one consequence that must be recorded rather than
discovered: the checkpoint's `embedding_table` buffer is `[30637, 1024]` and the PIC table
is `[2200, 1024]`, so **it cannot be restored from the checkpoint.** `load_backbone`
correctly refuses a partial transfer; `pic_audit._build_from_ckpt` substitutes the PIC table
explicitly and records `embedding_table_substituted_not_restored: true`. INV-FROZEN's
"restored from the checkpoint rather than rebuilt" clause does not hold for the table under
option (b), and the fine-tune path will need the same explicit substitution.

Neither route is implemented. `train_finetune.py` builds `DKMModel` from `--embedding_path`,
so passing `bge_embeddings_pic.pt` gets the right table but will fail `load_backbone` on the
shape mismatch. **This is the one code change PIC fine-tuning still needs**, and it is
deliberately left undone because it *is* D3.

## 7. `eval_finetune.py` — implemented, not run

Structural smoke on the real PIC `heart_malformations` val split (1,280 sequences) with
four synthetic checkpoints built at init — **no optimizer was constructed and no step was
taken**. It exercised: config agreement + rebuild verification, the shared hash-asserted
pass, pediatric band stratification with the thin `adolescent` band (n = 48) correctly
flagged `unreliable`, patient-level and paired bootstrap CIs, the equal-norm probe, the
Gram condition on the PIC lag distribution, generator gradient fractions, parameter drift
from the pretrained backbone, and the pretrain→fine-tune `Δα` change over 0–18 y. All three
output files were written. 52 s wall at `--n_boot 40`; expect ~5–10 min at the default
2,000.

## 8. Other observations worth recording

- **Cohorts are patient-disjoint** across train/val/test for all four tasks (`A2_cohort.
  patient_level_disjoint: true`, zero overlapping subjects). One sequence per subject, so
  the patient-level bootstrap and a row-level one coincide here — the machinery is still
  correct for a corpus where they would not.
- **Positives in the smallest split**: `mortality` 79, `pneumonia` 105,
  `heart_malformations` 223, `los_gt7` 402 (all on `val`). `mortality` at 79 positives
  across six pediatric bands is the binding constraint on band-stratified reporting, and is
  why `PEDIATRIC_MIN_BAND_N = 50` rather than `MIN_BAND_N = 200`.
- **PIC race is 100% `UNKNOWN`** (value 6 of 6, `race_UNKNOWN` → demographic channel **8**,
  ordering read from `config.model.demo_channels` and confirmed to match
  `data.RACE_LABELS`). Every PIC patient therefore receives an identical seven-column
  one-hot: six zeros and a one in the column MIMIC used for its own missing-race patients.
  The race channels carry no PIC signal and are not a confound, but `demo_proj`'s learned
  response to `race_UNKNOWN` is inherited wholesale from MIMIC's missing-race population.
- **Standardized PIC age is −3.82 to −2.74** (median −3.78), matching README's expected
  −3.8 to −2.7. Every PIC patient sits 3–4 sd below the pretraining mean; `demo_proj`'s
  behaviour there is extrapolation, exactly as `Δα(a)` is. Reported, not re-standardized.
- **Sequence length**: median 133–149 events, p90 ~265, max **1,018** across every task and
  split — under `--max_seq_len 1024`, so left-truncation never fires on PIC.
- **Length-1 sequences exist and are not rare**: 26–59 per train split, 3–4 per val, 2–8
  per test (`A3_length_window.per_split.*.n_sequences_length_1`). A one-event sequence has
  no pairwise lag at all, so the kernel contributes literally nothing on those rows;
  `_pad_common` accepts them and `build_pair_mask`'s forced diagonal keeps `softmax`
  finite. INV-NAN covers the gradient path. Worth excluding, or at least reporting
  separately, in any final table.

## 9. What is blocked on what

| item | blocked on | consequence if unblocked |
|---|---|---|
| any PIC fine-tune run | **D3** | the embedding-table substitution in `train_finetune.load_backbone`, ~10 lines, plus a recorded deviation |
| interpreting a null kernel result | **D1** | at 2% occupancy a null result is uninformative about the mechanism; it measures the cohort window |
| the shape of the four-arm table | **D2** | `finetune.sh` takes both designs already (`MODE=matched\|shared`, `CKPT_MAP`), so this is a run-time choice, not a code change |
