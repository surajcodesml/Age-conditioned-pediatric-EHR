# CEHR-BERT-small capacity-control experiment (S2, 32-target)

## Scientific question

Does CEHR-BERT's stronger counterfactual surface fitting persist when trainable capacity is approximately matched to DTR (~55k)?

## Exact architecture (chosen)

| Hyperparameter | Value |
|---|---|
| `d_model` | 32 |
| `d_ff` | 88 |
| `n_layers` | 3 |
| `n_heads` | 4 |
| `dropout` | 0.1 |
| `time_dim` | 32 |
| `age_dim` | 32 |
| `max_seq_len` | 112 |
| age representation | Time2Vec (unchanged) |
| time representation | Time2Vec (unchanged) |
| projection | concat(code, segment, time, age) → Linear → d_model |

## Parameter count

Counting convention: `baselines.common.capacity_report.count_parameters`.

| Model | Trainable | Frozen | Total | Code/seg embeddings |
|---|---:|---:|---:|---|
| CEHR-BERT-full | ~1,126,304 | 0 | ~1,126,304 | **trainable** |
| CEHR-BERT-small | 55,016 | 0 | 55,016 | **trainable** |
| DTR (age_temporal) | 55,107 | 0 | 55,107 | trainable |

### Capacity search table (time_dim=age_dim=32 fixed)

| hidden_dim | ffn_dim | layers | heads | trainable_params |
|---:|---:|---:|---:|---:|
| 32 | 88 | 3 | 4 | 55016 |
| 36 | 108 | 2 | 4 | 54988 |
| 48 | 64 | 1 | 4 | 55136 |
| 32 | 48 | 4 | 4 | 54720 |
| 40 | 64 | 2 | 4 | 55688 |

Chosen config is closest multi-layer match to DTR (Δ = -91).

## Training config

Matched to full CEHR-BERT synthetic baseline protocol:

- LR = `0.0003`, weight_decay = `0.01`
- max_epochs = `25`, patience = `5`
- grad_clip = `1.0`, batch_size = `32`
- loss = `BCEWithLogitsLoss`
- checkpoint selection = `best_val_bce` (lower val BCE)
- data_seed = `20260922` (patient-disjoint split)
- model seeds = `[0, 1, 2, 3, 4]` (seed 0 = canonical / paper-table seed)

## Seeds and checkpoints

| Seed | Role | Checkpoint |
|---:|---|---|
| 0 | canonical | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/cehrbert_small/seed0/S2/best_checkpoint.pt` |
| 1 | uncertainty | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/cehrbert_small/seed1/S2/best_checkpoint.pt` |
| 2 | uncertainty | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/cehrbert_small/seed2/S2/best_checkpoint.pt` |
| 3 | uncertainty | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/cehrbert_small/seed3/S2/best_checkpoint.pt` |
| 4 | uncertainty | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/cehrbert_small/seed4/S2/best_checkpoint.pt` |

## Primary metrics (32-target)

### Canonical seed 0

| Metric | CEHR-small | CEHR-full | DTR | Temporal-only DTR |
|---|---:|---:|---:|---:|
| AUPRC | 0.546 | 0.582 | 0.577 | 0.569 |
| AUROC | 0.742 | 0.764 | 0.761 | 0.758 |
| BCE | 0.475 | 0.456 | 0.459 | 0.463 |
| CF-RMSE-age | 0.116 | 0.098 | 0.142 | 0.177 |
| CF-RMSE-lag | 0.131 | 0.111 | 0.166 | 0.184 |
| Surface RMSE | 0.162 | 0.131 | 0.201 | 0.236 |

### Multi-seed mean ± SD (CEHR-BERT-small)

| Metric | mean ± SD |
|---|---|
| AUPRC | 0.541 ± 0.013 |
| AUROC | 0.739 ± 0.009 |
| BCE | 0.478 ± 0.006 |
| CF_RMSE_age | 0.129 ± 0.034 |
| CF_RMSE_lag | 0.145 ± 0.009 |
| Surface_RMSE | 0.165 ± 0.012 |

## Comparison table

| Model | Trainable params | AUPRC | AUROC | CF-RMSE-age | CF-RMSE-lag | Surface RMSE |
|---|---:|---:|---:|---:|---:|---:|
| DTR | 55107 | 0.577 | 0.761 | 0.142 | 0.166 | 0.201 |
| Temporal-only DTR | 55106 | 0.569 | 0.758 | 0.177 | 0.184 | 0.236 |
| CEHR-BERT-small (~55k) seed0 | 55016 | 0.546 | 0.742 | 0.116 | 0.131 | 0.162 |
| CEHR-BERT-small multi-seed | 55016 | 0.541 ± 0.013 | 0.739 ± 0.009 | 0.129 ± 0.034 | 0.145 ± 0.009 | 0.165 ± 0.012 |
| CEHR-BERT-full (~1.1M) | 1126304 | 0.582 | 0.764 | 0.098 | 0.111 | 0.131 |

## Deltas (canonical seed 0)

### CEHR-small vs CEHR-full

```
{
  "AUPRC": -0.03592115362433479,
  "AUROC": -0.02112980551871657,
  "CF_RMSE_age": 0.0175188656555694,
  "CF_RMSE_lag": 0.02031846959985141,
  "Surface_RMSE": 0.03136932849811838,
  "Surface_RMSE_rel": 0.23985481027694355
}
```

### CEHR-small vs DTR

```
{
  "AUPRC": -0.030631325195244363,
  "AUROC": -0.018137625910615296,
  "CF_RMSE_age": -0.02644342060799476,
  "CF_RMSE_lag": -0.03448754806316276,
  "Surface_RMSE": -0.03913958619021421,
  "Surface_RMSE_rel": -0.19444032088687246
}
```

## Capacity hypothesis

- Full−DTR Surface advantage: 0.0705
- Small−DTR Surface advantage: 0.0391
- Fraction of full advantage retained by small: 0.56

**Primary reading (Surface RMSE):** CEHR-BERT-small (seed 0) remains better than DTR
(0.162 vs 0.201) but worse than full CEHR-BERT (0.131). It retains about half of the
full model's Surface-RMSE advantage at matched capacity. That is **not** strong enough
to claim that parameter count alone explains CEHR-BERT's surface fit, and **not**
strong enough to claim the advantage is capacity-independent.

Closest pre-registered labels:
- Toward **B** on counterfactuals: a non-trivial Surface / CF-RMSE advantage over DTR
  survives at ~55k params, consistent with CEHR-BERT's less constrained age/time
  representation contributing beyond raw capacity.
- Toward **A** / partial capacity dependence: shrinking from 1.1M → 55k still costs
  ~0.031 Surface RMSE (~24% relative) vs full CEHR-BERT.
- **Not C:** predictive AUPRC/AUROC also drop (below both full CEHR-BERT *and* DTR),
  so this is not a clean “prediction OK / surface broken” dissociation.

**Do not conclude** that parameter count explains CEHR-BERT performance. The matched-capacity
result supports a **mixed** account: capacity matters for the magnitude of the surface
advantage, but representation differences likely remain important because CEHR-small still
beats DTR on CF-age, CF-lag, and Surface RMSE despite slightly worse prediction.

## Secondary analysis: 8 interaction targets

Produced from the **same** 32-target checkpoint by restricting metrics / CF to `mechanism=interaction` labels (ids 0–7). **Not** the primary result.

| Seed | AUPRC | AUROC | BCE | CF-age | CF-lag | Surface |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.797 | 0.895 | 0.338 | 0.168 | 0.176 | 0.229 |
| 1 | 0.737 | 0.862 | 0.381 | 0.275 | 0.202 | 0.244 |
| 2 | 0.798 | 0.896 | 0.340 | 0.169 | 0.217 | 0.233 |
| 3 | 0.791 | 0.892 | 0.344 | 0.143 | 0.192 | 0.202 |
| 4 | 0.789 | 0.891 | 0.347 | 0.179 | 0.206 | 0.252 |

## Artifacts

- Config: `configs/synthetic/cehr_bert_small_s2_32target.yaml`
- Results root: `results/baselines/synthetic/cehrbert_small`
- Per-seed: `result.json`, `cf_report.json`, `cf_surfaces.npz`, checkpoints
- Summary: `results/baselines/synthetic/cehrbert_small/summary.json`
- Figures: `figures/final/capacity_vs_surface_rmse.{svg,png}`, `capacity_vs_auprc.{svg,png}`

## Fairness / instability notes

- Full CEHR-BERT artifacts under `results/baselines/synthetic/cehrbert/` were **not** modified.
- DTR and the synthetic benchmark were **not** modified.
- Training budget and early-stopping criterion match full CEHR-BERT (max_epochs=25, patience=5, best val BCE).
- Only ordinary capacity hyperparameters were reduced; age/time Time2Vec inputs and concat→project architecture were retained with dim=32.
- DTR multi-seed runs were not available; only canonical seed 0 exists for DTR. CEHR-small uses seed 0 for direct comparison plus seeds 1–4 for uncertainty.
