# Stage-2 NCH pediatric adaptation — implementation report

Continued longitudinal pretraining on processed NCH SleepBank, initialized from Stage-1 MIMIC. Full NCH training was **not** launched.

## Files changed

New package `stage2_nch/` (Stage-1 code was reused, not rewritten):

- `stage2_nch/config.py` — pediatric \(z_P\), probe ages, age bands, target spec
- `stage2_nch/init_from_stage1.py` — load `adkm_s0`, drop adult \(\beta_A\) / \(\mu,\sigma\), force \(\beta_P=0\)
- `stage2_nch/tensorize.py` — window already-mapped NCH events into Stage-1 forecast shards
- `stage2_nch/dataset.py` — `NCHForecastDataset` + stratification metadata
- `stage2_nch/compatibility.py` — vocab / embedding / head / shard-index report
- `stage2_nch/sign_test.py` — controlled \(\lambda_0\) near vs distant attention
- `stage2_nch/metrics.py` — pos/neg BCE, prevalence baseline, history tertiles, param drift
- `stage2_nch/evaluate.py` — epoch metrics, age/history strata, shuffle / constant-age tests
- `stage2_nch/plots.py` — required run plots
- `stage2_nch/train.py` — matched two-arm trainer
- `stage2_nch/tests/test_sanity.py` — 14 automated sanity tests
- `stage2_nch/tests/run_all.py` — unit tests then tiny-NCH overfit

Derived (not a code remapping):

- `artifacts/nch_stage2/v2/tensorized_forecast/diagnoses_only/{train,val,test}/shard_000.npz`
- `artifacts/nch_stage2/v2/tensorized_forecast/diagnoses_only/tensorize_report.json`

Smoke artifacts: `stage2_nch/run/smoke.json` and `stage2_nch/run/smoke_{no_interaction,age_temporal}_*`.

## NCH dataset sizes / splits

Source: v2 cleaned diagnosis events, cut strictly before each patient’s first pediatric sleep-study `index_time`. Patient JSON splits (seed 42) are unchanged.

| split | JSON patients | written (≥2 encounters) | forecast windows |
| ----- | ------------- | ----------------------- | ---------------- |
| train | 2433          | 2397                    | 137405           |
| val   | 522           | 512                     | 33256            |
| test  | 520           | 511                     | 28147            |

- Split overlap: train∩val = train∩test = val∩test = 0.
- 52 patients have fewer than two pre-index encounters and contribute no windows.
- Last-event ages on train windows: 0.00–17.98 years (median 4.76).
- Age-band window counts (train): `<1` 16892, `1–5` 63983, `6–11` 40158, `12–17` 16372.
- Default `max_seq_len = 1024` (Stage-1). On a 20k train subsample, event-length tertiles are ~129 / 248 / 448; ~11% of windows hit the 1024 cap.
- History bins are **train tertiles of `n_input_events`**, not arbitrary visit counts.

## Vocabulary compatibility

Report: each run’s `compatibility.json`. Result: **passed**.

- Frozen MIMIC vocab \(|V|=30635\); special tokens unchanged (`PAD=0`, `UNK=1`, real = `vocab_index+2`).
- BGE table rows 30637 × 1024 match Stage-1 `embedding_table`.
- Prediction head output is 30635, matching Stage-1. NCH is a **subset** of that space; the full head is retained.
- Shard codes sit in `[5752, 30635]` with `unk_vocab_index = 30635`, consistent with the existing mapping (OOV stored as `|V|`, then `+2` at collate).
- No silent remapping: `mimic_token_id` from `canonical_events_clean.parquet` is used as-is.

OOV under the existing contract (not a new mapper): 109402 / 1063639 pre-index diagnosis events (10.27%) become UNK on input and are **dropped from the multi-hot target**.

## Exact target / horizon

```
input  = diagnosis events with timestamp < start(V_{m+1})  (strict; ties go to the target)
target = multi-hot code set of the next NCH encounter_id, UNK dropped, duplicates collapsed
horizon= next clinical encounter, not a fixed calendar window
index  = only events strictly before the first pediatric sleep-study index_time
loss   = torch.nn.BCEWithLogitsLoss()  (no sigmoid first; no pos_weight)
```

Representation: `diagnoses_only`. Masking is padding-only; the target encounter is outside the input window (same INV-HORIZON contract as Stage-1).

## Stage-1 checkpoint used

`stage1_mimic_pretrain/run/adkm_s0/checkpoint_best.pt`

- arm `age_temporal`, epoch 7, val BCE **0.009130**
- transferred \(\lambda_0 = -1.61681\)
- adult \(\beta_A = -0.08274\) **not used**
- adult \(\mu=63.336\), \(\sigma=16.575\) **not used**

## Exact transferred / reset parameters

Transferred: frozen BGE `embedding_table`, Transformer encoder, pooling, demographic projection, prediction head, \(\lambda_0\).

Reset before any Stage-2 step:

- \(\beta_P = 0\) for **both** arms
- `age_mean = 9`, `age_sd = 9` on the model **and** `AgeTemporalBias` (so demo age and attention age share \(z_P\))
- `no_interaction`: `beta.requires_grad = False`
- `age_temporal`: `beta.requires_grad = True`

Init identity (NCH batch, \(\beta=0\)): `max_abs_logit_diff = 0.0`.

## Pediatric age normalization

Implemented, unclipped:

\[
z_P(a)=\frac{a-9}{9},\qquad a=\texttt{age\_at\_event\_days}/365.25
\]

Endpoints: \(z_P(0)=-1\), \(z_P(9)=0\), \(z_P(18)=+1\).

Pre-index pediatric diagnosis ages are all in \([0, 17.98]\). Tensorize counted `n_age_lt0 = 0`, `n_age_gt18 = 0`, so **clipping is not applied**. If an out-of-range age appeared, shards would clip to \([0,18]\) years (equivalent to \(\mathrm{clip}(z_P,-1,1)\)) and the run config would record `clip_applied=true`.

## Attention equation

Unchanged from Stage-1:

\[
s_{ij}^{(h)}=\frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_h}}-[\lambda_0+\beta_P z_P(a_i)]\,\tau_{ij}
\]

with \(\tau_{ij}=\log(1+|t_i-t_j|/7)\). Production defaults: `d_model=256`, `n_heads=4`, `n_layers=1`. One shared \(\lambda_0\), one shared \(\beta_P\), temporal modulation only in self-attention, age-neutral pooling, no Fourier / Chebyshev / age MLP / per-head parameters.

`no_interaction` is the same equation with \(\beta_P\) frozen at 0. Demographic age remains in the shared demo pathway.

## Sign interpretation of \(\lambda_0\)

Controlled 3-event sequence (\(t=0,1,1000\) days), \(\beta=0\):

| \(\lambda_0\) | mass on distant \(t=0\) | mass on self |
| ------------- | ----------------------- | ------------ |
| 0             | 0.312                   | 0.368        |
| \(+2\)        | \(4.3\times10^{-5}\)    | 0.9999       |
| \(-2\)        | 0.497                   | \(2.9\times10^{-5}\) |

**Positive \(\lambda_0\) is recency** (suppresses long lags). **Negative \(\lambda_0\) is long-range preference.** The transferred Stage-1 value \(\lambda_0\approx-1.617\) is therefore a **long-range prior, not recency**. Do not describe it as recency in Stage-2 writeups.

## Sanity-test results

`python -m stage2_nch.tests.test_sanity` → **14/14 passed** (also confirmed inside `python -m stage2_nch.tests.run_all`).

| # | test | result |
| - | ---- | ------ |
| 1 | Stage-1 checkpoint loads; \(\beta_P=0\); embeddings frozen | PASS |
| 2 | arms identical at \(\beta=0\) | PASS |
| 3 | \(z_P(0,9,18)=(-1,0,+1)\) | PASS |
| 4 | age changes attention logits only when \(\beta\neq0\) | PASS |
| 5 | \(\beta\) gets gradient in `age_temporal` | PASS |
| 6 | \(\beta\) stays exactly 0 in `no_interaction` | PASS |
| 7 | \(\lambda_0\) gets gradient | PASS |
| 8 | padding mass is ~0 | PASS |
| 9 | no future-event leakage (`t < start(V_{m+1})`) | PASS |
| 10 | MIMIC/NCH vocab indices align | PASS |
| 11 | pooling independent of \(\beta\) | PASS |
| 12 | shuffling attention age changes outputs when \(\beta\neq0\) | PASS |
| 13 | no NaN/Inf at pediatric extremes / huge \(\Delta t\) | PASS |
| 14 | patient JSON splits disjoint | PASS |

Plus a synthetic tiny-overfit step inside the unit suite (loss drop > 0.02 over 40 Adam steps).

## Tiny-NCH pilot results

`python -m stage2_nch.tests.run_all` on CPU: 8 examples, `max_seq_len=32`, 2 epochs / 6 steps, batch 4, both arms, seed 0.

| | `no_interaction` | `age_temporal` |
| - | ---------------- | -------------- |
| init val BCE (epoch 0) | 0.004191 | 0.004191 |
| epoch-2 train BCE | 0.00200 | 0.00200 |
| best val BCE | 0.002224 | 0.002224 |
| \(\lambda_0\) (end) | −1.6181 | −1.6181 |
| \(\beta_P\) (end) | **0.0** (\(\|g\beta\|=0\)) | −0.0011 (\(\|g\beta\|>0\)) |
| \(\Delta L_{\text{shuffle}}\) | 0 | \(-1.1\times10^{-8}\) |
| \(R_{\text{bias}}\) | 0.513 | 0.513 |
| wall clock | 4.7 s | 4.5 s |

Required artifacts written per arm: `checkpoint_best_bce.pt`, `checkpoint_best_auprc.pt`, `checkpoint_final.pt`, `config.json`, `seed.json`, `metrics.csv`, `history.json`, `age_tests.json`, `lambda_trajectory.json`, `predictive_metrics.json`, `age_stratified_metrics.json`, `history_stratified_metrics.json`, `prevalence_baseline.json`, plus plots under `plots/` (`loss`, `micro_auprc`, `positive_bce`, `lambda0`, `beta`, `lambda_a`, `age_shuffle`, `bias_vs_content`, `age_groups`, `history_groups`).

This 8-example run is a **pipeline smoke**, not evidence about pediatric age × temporal use. Loss dropped (overfit), arms matched at step 0, and \(\beta_P\) moved only in `age_temporal`.

## Baseline performance

On the same 8-example smoke val slice, the **train-set prevalence predictor** had BCE 0.00140 vs model BCE ~0.00222. That comparison is **not interpretable**: almost all of 30635 classes are never-positive in 8 rows, so unweighted BCE is dominated by negatives. Production runs will compute the prevalence baseline on the full train loader and score full val/test.

## Initial arm comparison

At initialization the arms are functionally identical (\(\beta_P=0\), logit diff 0). After 6 tiny steps they remain essentially tied on BCE/AUPRC; `age_temporal` has a tiny \(\beta_P\) with \(\Delta L_{\text{shuffle}}\approx0\). **Do not conclude anything about the interaction from this smoke.** Full training must still require all five interpretation criteria in the spec (trajectory, \(R_{\text{bias}}\), reproducible \(\Delta L_{\text{shuffle}}>0\), predictive gain vs `no_interaction`, and age/history strata).

## GPU / training configuration

Smoke device: **CPU** (`torch.cuda.is_available() == False` in this session). The trainer selects CUDA when present (`--device cuda`).

Production defaults (matched across arms; do not retune per arm):

```
d_model=256  n_heads=4  n_layers=1  ffn_mult=4  demo_hidden=64
max_seq_len=1024  batch_size=16  epochs=12  patience=4
lr_backbone=1e-4  lr_age=1e-3  lr_head=1e-3  grad_clip=1.0
num_workers=6  seed=0
loss=BCEWithLogitsLoss (unweighted)
early stopping: best val BCE; also save best val micro-AUPRC
```

Launch later (not part of this task):

```bash
python -m stage2_nch.train --arm no_interaction --run_name nint_nch_s0 --seed 0
python -m stage2_nch.train --arm age_temporal --run_name adkm_nch_s0 --seed 0
```

Adult-\(\beta\) transfer is **not** implemented here (coordinate change \(z_A\to z_P\) is a later ablation).

## Discovered incompatibilities / bugs

1. **Index NPZ sequences have no next-encounter target.** Existing `diagnoses_only_sequences.npz` is encoder-only pre-index history. Stage-2 therefore windows `canonical_events_clean.parquet` by `encounter_id` under the same MIMIC token IDs. This is **not** a remapping reprocess.
2. Adult \(\mu/\sigma\) would send a 9-year-old to \(z_A\approx-3.3\). They are discarded; \(z_P\) is used for both attention and demo age.
3. ~10% of NCH diagnosis events are OOV under the frozen MIMIC vocab; handled as UNK per the existing contract.
4. `np.unique(..., return_index=True)` would have sorted encounter IDs and scrambled visit order; tensorize uses first-seen order instead.
5. Matplotlib mathtext rejected `\sqrt d` in the \(R_{\text{bias}}\) ylabel; plots now use ASCII for that label and survive `tight_layout` failures.
6. CUDA was not available for this smoke. That is an environment limit, not a trainer bug. Full NCH training should be run on GPU.

No remaining code blockers for a matched two-arm launch.

READY FOR FULL NCH STAGE-2 TRAINING
