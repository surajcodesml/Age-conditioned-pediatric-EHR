# Stage-1 MIMIC-IV pretraining — implementation report

Minimal age × temporal attention, trained with the existing future-visit
code-prediction objective. New code lives in `stage1_mimic_pretrain/`.
Fourier / Chebyshev DKM machinery is not imported and is not instantiated.

## Files changed (all new)

| Path | Role |
|---|---|
| `stage1_mimic_pretrain/config.py` | Arms, frozen μ/σ, documented transforms |
| `stage1_mimic_pretrain/model.py` | Shared λ0, β; encoder; pooling; head |
| `stage1_mimic_pretrain/metrics.py` | Multi-label metrics, shuffle / constant-age tests |
| `stage1_mimic_pretrain/evaluate.py` | Val/test eval, patient-ID dump |
| `stage1_mimic_pretrain/train.py` | Pretraining loop, checkpoints, epoch logs |
| `stage1_mimic_pretrain/plots.py` | Required run plots |
| `stage1_mimic_pretrain/tests/test_sanity.py` | Shape / gradient / mask / overfit tests |
| `stage1_mimic_pretrain/tests/run_all.py` | Unit tests + 1-shard MIMIC smoke |

Reused, not copied: `model_new.data` (dataset, collate, `lag_to_tau`,
`tau_from_timestamps`, corpus stats), `model_new.encoder.build_pair_mask` /
`build_key_mask`, frozen BGE table, patient-level tensorized splits.

Not used: `model_new.basis`, `model_new.age_encoding`, `DKMModel`.

## Attention equation (implemented)

For every head \(h\) in Transformer self-attention:

\[
s_{ij}^{(h)}
=
\frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_h}}
-
\bigl[\lambda_0+\beta z(a_i)\bigr]\tau_{ij}
\]

- \(\lambda_0,\beta\): **one scalar each**, shared across heads. Not per-head.
- `no_interaction`: \(\beta\) is registered and **frozen at 0**. Demographic age remains.
- `age_temporal`: both trained in self-attention.
- `temporal_only`: CLI alias of `no_interaction`.
- Content Q/K/V remain Linear→GELU as in `model_new.encoder`.
- The bias tensor is `[B, L, L]` and is added with `unsqueeze(1)` so it
  broadcasts over heads without a per-head copy.

Pooling is **identical across arms** by default: \(s_j = q_{\mathrm{base}}\cdot e_j\)
(no \(\lambda_0/\beta\)). The old pooling bias can be restored with
`--pool_temporal_bias` and is off for Stage-1.

Production layout: `n_heads=4`, `d_model=256`, so \(d_h=64\).

## Age normalization

Fixed affine map on **years**, frozen from the MIMIC train event-level corpus
(`model_new.data.corpus_stats`, \(N=405{,}519{,}425\) events):

\[
z(a)=\frac{a-\mu}{\sigma},\quad
a=\frac{\texttt{age\_at\_event\_days}}{365.25},\quad
\mu=63.33601047086648,\quad
\sigma=16.574804662346914
\]

Not trainable. No Fourier features, no age MLP. Adult range is appropriate:
median 58.7 y, min 16.6 y, 99.95% of events ≥ 18.

**Conflict with old DKM (kept, documented):** demographic channel 0 is still
standardized with the same \((\mu,\sigma)\) and concatenated into the head
(legacy R1). Shuffle / constant-age tests override **attention ages** by
default so \(\Delta L\) isolates the \(\lambda(a)\tau\) interaction. A second
constant-age number also overwrites the demographic age channel.

## Temporal distance

Existing convention, not redefined:

\[
\tau_{ij}=\log\bigl(1+|t_i-t_j|/c\bigr),\quad c=7\text{ days}
\]

Implemented by `model_new.data.lag_to_tau` (float64 abs + log1p).
**No** Chebyshev rescaling \(\tilde\tau=2\tau/\tau_{\max}-1\). \(\tau_{\max}\)
is unused.

## Future-event target / horizon (existing)

`TensorizedPretrainDataset` (INV-HORIZON), not changed:

- One sample per (patient, next hadm-visit \(V_{m+1}\)) with ≥1 prior event.
- Input = events with `timestamp < start_time(V_{m+1})` (strict; ties go to the target).
- Target \(y\in\{0,1\}^{C}\), \(C=30635\): unique non-UNK codes in \(V_{m+1}\).
- Truncation keeps the newest 1024 pre-boundary events.
- Horizon is the **next visit**, not a fixed calendar window.
- Masking is **padding-only**, not causal: the target visit is outside the
  window, so bidirectional attention inside it is not leakage (D4).

## Loss

\[
\mathcal L_{\mathrm{pretrain}}=\mathrm{BCEWithLogits}(y,\hat y_{\mathrm{logits}})
\]

`torch.nn.BCEWithLogitsLoss()`, unweighted. Sigmoid is not applied before the
loss. **No `pos_weight`:** the old pipeline never used one; imbalance is
reported and left unweighted.

Smoke-sample imbalance (8 val examples): mean prevalence \(0.0025\),
99.1% of codes never positive in that cap, ~77 positives per example.
That is expected multi-hot future-visit prediction, not a new weighting scheme.

## Model / config (production defaults = `model_new.train`)

| Setting | Default |
|---|---|
| `d_model` | 256 |
| `n_layers` | 1 |
| `n_heads` | **4** (`d_h = 64`) |
| residual / LayerNorm / FFN | on (`ffn_mult=4`) |
| embeddings | frozen BGE, 1024-d, `\|V\|+2` rows |
| demo | 9-d one-hot race, `demo_hidden=64` |
| head | Linear→GELU→Linear, final bias −7.0 |
| optim | Adam, lr backbone \(10^{-4}\) / age \(10^{-3}\) / head \(10^{-3}\) |
| splits | existing patient-level `tensorized_flat/{train,val,test}` |
| seed | recorded in `seed.json` |

Smoke used `d_model=64`, `n_heads=4` (production head count; smaller width for CPU).

```bash
python -m stage1_mimic_pretrain.train --arm no_interaction --run_name nint_s0 --seed 0
python -m stage1_mimic_pretrain.train --arm age_temporal   --run_name adkm_s0 --seed 0
python -m stage1_mimic_pretrain.tests.run_all
```

## Smoke-test results

All 14 unit tests passed (`stage1_mimic_pretrain/run/smoke.json`).

| Test | Result |
|---|---|
| Bias broadcasts identically across 4 heads | PASS |
| \(\beta=0\) matches `no_interaction` logits | PASS (max \(\lvert\Delta\rvert < 10^{-6}\)) |
| Changing age changes logits iff \(\beta\neq 0\) | PASS |
| Changing \(\Delta t\) changes logits | PASS |
| Gradients reach \(\lambda_0\) and \(\beta\) | PASS (`age_temporal`) |
| `no_interaction`: \(\lambda_0\) grad only; \(\beta\) frozen | PASS |
| Shuffled ages change outputs when \(\beta\neq 0\) | PASS |
| Padded keys get zero attention | PASS |
| Extreme \(\Delta t=10^7\) d, ages 0/200: no NaN/Inf | PASS |
| Overfit 4 synthetic sequences, 40 steps | PASS (loss dropped) |
| Macro AUROC skips all-pos / all-neg classes | PASS |
| Shared backbone init across arms | PASS |

Tiny real MIMIC (1 shard, 8 examples, 6 steps, CPU, seed 0):

| | `no_interaction` | `age_temporal` |
|---|---|---|
| train BCE ep1 / ep2 | 0.01041 / 0.01028 | 0.01041 / 0.01028 |
| val BCE (best) | 0.01836 | 0.01836 |
| \(\lambda_0\) | +0.00351 | +0.00351 |
| \(\beta\) | 0 (frozen) | −0.00093 |
| \(\lvert g_{\lambda_0}\rvert\) | \(3.1\times 10^{-7}\) | \(3.1\times 10^{-7}\) |
| \(\lvert g_{\beta}\rvert\) | 0 | \(1.8\times 10^{-7}\) |
| \(\lvert-\lambda\tau\rvert / \lvert q^\top k/\sqrt d\rvert\) | — | 0.0018 |

Gradients **do** reach \(\lambda_0\) and \(\beta\). After 6 steps the temporal
bias is ~0.18% of content-logit magnitude, so a nonzero \(\beta\) is not yet
functionally visible. That is the diagnostic this pipeline is meant to catch.

## Correct-age vs shuffled-age (smoke)

Protocol: permute last-event age across sequences in the batch, broadcast onto
valid positions; **demographics unchanged**; 3 seeded shuffles.

| Arm | \(\beta\) | \(\Delta L_{\mathrm{shuffle}}\) | interpretation |
|---|---|---|---|
| `no_interaction` | 0 | 0 | \(\beta\approx 0\) and \(\Delta L\approx 0\): interaction unused |
| `age_temporal` | −9.3e-4 | 0 | nonzero \(\beta\), negligible \(\Delta L\): param moved, little effect |

Val metrics on this 8-example cap were identical across arms (micro AUROC
~0.80, macro AUROC 0.53 over **268** valid classes out of 30635; 30367 codes
had no positive). This is **not** a full-split comparison. No GPU was
available, so a production-scale temporal-only vs Minimal-DKM run was not
executed.

## Bugs / inconsistencies found in the old stack

1. **Fourier + Chebyshev + age MLP** are far larger than \(\lambda_0+\beta z(a)\tau\).
   Prior work already showed the frozen MIMIC \(\tau_{\max}\) makes the Chebyshev
   basis numerically singular on PIC. This pipeline removes that basis entirely.
2. **Pooling no longer carries \(\lambda(a)\tau\)** on the main path. The
   experimental contrast is only in self-attention. Demographic age (R1) is
   still in the head for both arms, which is why the control arm is named
   `no_interaction` rather than `temporal_only`.
3. **Age still sits in demographics (R1)** on every arm. A shuffle of attention
   ages can report \(\Delta L=0\) while the head still sees age. Shuffle tests
   therefore isolate the interaction on purpose; constant-age can also overwrite
   the demo channel.
4. **No `pos_weight`** despite extreme multi-label imbalance (~0.25% prevalence,
   ~99% of codes never positive in a small cap). Left unweighted, matching the
   existing pretrain loss.
5. **Pre-horizon shards leaked the target** (~59% of windows had
   `target_time ≤ input_end`; `audit/signal/out/horizon_phase0.json`). Current
   `TensorizedPretrainDataset` uses a strict time cut (INV-HORIZON). We did not
   silently redefine the target; we reused that dataset.
6. **Old DKM generator was zero-init**, so \(\Delta\alpha\equiv 0\) at step 0 and
   the first layer of the age MLP had zero gradient. \(\lambda_0,\beta\) here
   receive gradient from step 1 (confirmed).
7. **`recall_per_example` in `model_new.diagnostics`** uses `n_pos.clamp(min=1)`,
   scoring empty targets as recall 0. Ranking here follows `topk_per_example`
   (NaN when \(\lvert\mathrm{true}\rvert=0\)).
8. Causal `torch.tril` was already rejected in `model_new` (D4). We keep
   padding-only, as required by the existing sequence setup.

## Saved outputs (every run)

`checkpoint_best.pt` (min val BCE), `checkpoint_final.pt`, `epoch_*.pt`,
`config.json`, `seed.json`, `{train,val,test}_subject_ids.npy` (unless
`--skip_split_ids`), `metrics.csv`, `history.json`, `lambda_trajectory.json`,
`age_tests.json`, `predictive_metrics.json`, plots under `plots/`.

## What a full run still has to show

Smoke confirms the mechanism is wired, differentiable, and correctly inert
when \(\beta=0\). It does **not** yet show that MIMIC pretraining *uses* the
age × time interaction. That requires the two production-scale arms on the
same patient split and seed, then reading:

- \(\beta\) trajectory and \(\lvert g_\beta\rvert\)
- \(\lambda(a)\) at 20…80 y
- ratio \(\lvert-\lambda\tau\rvert / \lvert q^\top k/\sqrt d\rvert\)
- \(\Delta L_{\mathrm{shuffle}}\) (mean ± std) and the constant-age deltas

If \(\beta\) moves but that ratio stays \(\ll 1\) and \(\Delta L_{\mathrm{shuffle}}\approx 0\),
the interaction is not functionally used — the failure mode this pipeline is
built to detect.
