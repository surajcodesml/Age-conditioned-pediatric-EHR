# `model_new/` — developmental kernel modulation (DKM), clean implementation

**Audience:** anyone extending, reviewing, or reproducing the DKM experiment.
**Status:** implementation complete; tests and preflight run; pretraining not yet run to completion.
**Relationship to other trees:** self-contained. Imports nothing from `model/`, `model_ablation/`
or `finetune/`, and modifies none of them. Those trees remain valid provenance for earlier runs.

---

## 1. What the model does

Patient age in years is mapped through a **fixed** Fourier feature embedding `ψ(a)`, a small MLP
produces additive coefficient deltas `Δα(a)`, and those coefficients parameterise a temporal
kernel that is injected into pre-softmax attention scores:

```
u          = log1p(a)                                 # log-age coordinate
ψ(a)       = [sin(2π u / p_m), cos(2π u / p_m)]_{m=1..M}
Δα(a)      = MLP(ψ(a)) ∈ R^s                          # final layer zero-init
α(a)       = α_base + Δα(a)
τ          = log1p(Δt_days / 7)
τ̃          = 2τ / τ_max − 1                            # clamped to [−1, 1]
log w      = Σ_{k=1..s} α_k(a) T_k(τ̃)                  # Chebyshev, NO constant term
score_ij   ← score_ij + log w_ij                       # direct log-space injection
```

This is FiLM-style additive modulation of a temporal-decay polynomial. It is not ALiBi, not
AdaLN, and not a learned age lookup table.

The kernel is applied at **two sites**: every encoder attention layer (conditioned on the
per-**query** age `a_i`) and the attention-pooling step (conditioned on the age at the last valid
event `a_n`). The two sites share form and nothing else — separate `α_base`, separate generator.

---

## 2. The four arms

`--arm` is the only flag that differs between runs. Everything else — seed, data, schedule,
optimizer settings, masking, backbone code — is shared.

| arm | demo feature (R1) | kernel `Δα` (R2) | concat to `h` (R3) | age into `ψ` |
|---|---|---|---|---|
| `vanilla` | ✓ | ≡ 0, **no params** | — | unused |
| `random_constant` | ✓ | `generator(fixed random vec)` | — | bypassed |
| `additive` | ✓ | ≡ 0, **no params** | `generator(ψ(a_n)) → [B,s]` | real, last event |
| `kernel` | ✓ | `generator(ψ(a_i))` | — | real, per query |

**R1 is present in every arm and is not the experimental variable.** Age stays in the demographic
vector because that is the route age already has, and it is the one DKM has to improve on.
Removing it would make `vanilla` age-blind and the baseline a strawman.

The variable is **where the generator's output is delivered**.

`random_constant`'s generator can only ever produce a single learned constant, which the free
`α_base` absorbs. It should therefore track `vanilla` closely; a large gap between them indicates
a bug, not a finding.

---

## 3. Defects corrected relative to `model/` and `model_ablation/`

| # | Legacy defect | Fix | Where |
|---|---|---|---|
| D1 | Monomial basis `[1, τ, …, τ⁵]`; Gram condition number 2.30e9 measured on 1.10M empirical pairwise lags | Chebyshev `T₁…T₅` on `τ̃`: **15.1** as parameterised (no constant term), 105 if `T₀` is included for a like-for-like comparison | `basis.py` |
| D2 | Constant term `T₀` carried | Dropped. Softmax is invariant to a per-row constant, so `α₀` cannot change any attention weight — within a row the query age, hence `α₀(a)`, is fixed. `s = 5`, indices 1..5 | `basis.py` |
| D3 | Age read from `demographics[..., 0]`; kernel input and demographic feature are the same tensor | `age_years` is its own `[B,L]` tensor. Age **stays** in the demographic vector too | `data.py`, `model.py` |
| D4 | Causal `torch.tril` masking | Padding only. The target visit lies outside the input window, so bidirectional attention within it is not leakage. **Measured** (full split): causal masking leaves 21.4% of attention rows with τ spread < 0.1 — nothing for the kernel to discriminate — against 1.5% under padding-only | `encoder.py` |
| D5 | Pooling used `scores = relevance * w` on a **signed** relevance, so `w ∈ (0,1)` *raised* attention on negative-relevance events | `scores = relevance + log w` — the same log-space injection attention uses | `pooling.py` |
| D6 | `Adam(model.parameters(), lr)`; the age pathway trained at backbone LR | Three groups from **module-declared** parameter sets. No name matching anywhere | `optim.py` |
| D7 | Fourier band `1/12 … 200 y` on **linear** age — a near-orthogonal hash, leaving constant `Δα` as the cheapest solution | Fourier over **log-age**, periods log-spaced in u-units | `age_encoding.py` |
| D8 | `τ_max` hardcoded and recomputable per dataset | Computed once on the pretraining corpus, stored as a persistent buffer in the checkpoint, reused verbatim at fine-tune. Disagreement raises | `basis.py`, `train_finetune.py` |
| D9 | Fine-tune used `return_repr_only=True`, skipping pooling → pooling-site age params gradient-dead | No such path exists. The fine-tune head sits on pooled `h`, exactly as pretraining does | `model.py` |
| D10 | Weibull time-gap head and loss | Removed. Pretraining is code BCE only | `train.py` |
| D11 | `w(t)` statistics printed from three call sites in three formats | Modules never print. `diagnostics.py` owns all output and JSON; a test greps for it | `diagnostics.py` |
| D12 | No assertion that `age_emb_dim` is even | Asserted at construction | `age_encoding.py` |

---

## 4. Layout

```
model_new/
  __init__.py
  arms.py              ArmConfig, resolve_arm, assert_arm_invariants
  basis.py             ChebyshevKernel — the only Chebyshev evaluation
  age_encoding.py      LogAgeFourier, CoefficientGenerator, AgeConditioner, characterize_band
  encoder.py           TimeAwareAttention, EncoderBlock, Encoder, mask helpers
  pooling.py           AttentionPooling
  model.py             DKMModel — one class, arm-gated, task ∈ {pretrain, classification}
  data.py              pretrain dataset + collate + τ_max + corpus stats
  data_finetune.py     fine-tune dataset + collate
  optim.py             build_param_groups
  diagnostics.py       sole owner of all logging and JSON
  train.py             pretraining
  train_finetune.py    fine-tuning
  preflight.py         Phase 10 review checkpoint
  run/                 pretrain.sh, finetune.sh
  tests/               one test per invariant + run_all.py
  INVARIANTS.md
```

Every module except `diagnostics.py` runs as `python -m model_new.<mod>` for its own smoke test.
Those smoke tests route their output *through* `diagnostics`, so the no-print rule holds.

---

## 5. Encoder depth — a resolved ambiguity

Figure 1A of the draft shows a transformer stack (LayerNorm → time-aware attention → Add & Norm →
FFN → Add & Norm, ×N). The legacy code is a single un-normalised attention op with no residual, no
LayerNorm, no FFN and no stacking. **These are not the same model.**

Both are available and recorded in `config.json`:

| flag | default | effect |
|---|---|---|
| `--n_layers` | 1 | kernel applied at **every** layer, same per-query age |
| `--legacy_block` | off | disables residual + LayerNorm + FFN → exactly the legacy encoder |
| `--no_residual` / `--no_layernorm` / `--no_ffn` | off | individual control |

**The default is the standard block (all three on), and preflight justifies it empirically.**
Headroom on real MIMIC batches, now under an **equal-norm** probe (a decaying kernel
`[−2,0,−1,0,0]` vs a growing one `[+2,0,+1,0,0]`, same L2 norm) — the earlier probe compared
a kernel against *no* kernel, which measured presence rather than shape discrimination:

| encoder block | max\|Δlogit\| | max/logit sd | vs legacy |
|---|---|---|---|
| `standard_block` | 0.0990 | 1.44 | **92×** |
| `layernorm_only` | 0.0468 | 0.88 | **43×** |
| `residual_only` | 0.0024 | 0.47 | 2.2× |
| `ffn_only` | 0.0004 | 0.09 | 0.4× |
| `legacy_block` | 0.0011 | 0.24 | 1× |

LayerNorm on the attention input is what gives the kernel authority. Un-normalised 1024-d BGE
embeddings produce QK logits large enough to swamp a bias of order `‖α‖₁`. The draft's concern that
residuals would *dilute* the kernel is not what the data shows at N=1 — `residual_only` beats
legacy, not the reverse.

Within-row τ spread (which decides whether kernel *shape* can matter, since softmax ignores a
per-row constant), from the full-split `corpus_stats`: median **4.46** padding-only vs **3.15**
causal, and the fraction of rows with spread < 0.1 is **1.5%** padding-only vs **21.4%** causal —
the measured form of D4.

Parameter counts, measured (d_model=256, |V|=30635, `demo_dim=9`), arm = `kernel`:

| config | encoder trainable | age | total trainable |
|---|---|---|---|
| `n_layers=1`, standard block | 1,579,909 | 4,864 | 11,519,797 |
| `n_layers=1`, legacy block | 789,637 | 4,864 | 10,729,525 |
| `n_layers=2`, standard block | 2,306,314 | 7,296 | 12,246,202 |

The figure-faithful version costs ~790k parameters over the legacy encoder at N=1, and a further
~726k per additional layer. Against a 9.94M-parameter output head, that is a ~7% increase in total
trainable parameters for a ~90× increase in kernel headroom (equal-norm probe).

---

## 5b. Where the draft and the data disagree

Measured on real MIMIC-IV pretraining batches. Each of these is a place the manuscript should
change rather than the code.

| draft / brief says | measured | consequence |
|---|---|---|
| `demo_dim = 3` (age, sex, race) | race has **cardinality 7**; scalar encoding imposes an arbitrary ordinal (WHITE=0 < BLACK=1 < …), and raw age (median ~56) beside eight 0/1 channels dominates `demo_proj`'s input scale ~50× | default is one-hot, `demo_dim = 9`, **and the age channel is standardized** with frozen corpus constants (Fix C). `--race_encoding scalar` restores 3 |
| day-resolution timestamps with many same-admission codes drive low within-row τ spread | after **float64** differencing (Fix D): timestamp resolution is magnitude-dependent (float32 storage quantises large `t` to ~40 s); only ~3–4% of valid pairs have `Δt = 0`; median within-row spread ~4.6 of a ~6.7 maximum | the low-spread concern does not apply to MIMIC pretraining. It was a property of the PIC CHD 24 h window, not of this corpus |
| Fourier band saturates at a **7.4-month** gap | re-measured by an explicit procedure: **12.3 months** for the legacy linear band, 27.0 months for log-age. The same procedure reproduces the brief's other three band numbers (3.35, 4.06, ~1.0) exactly | the 7.4 figure used an unstated definition of "asymptotic". The relative comparison is unaffected |
| Chebyshev condition number ~46 | on 1.10M **empirical** pairwise lags: monomial 2.30e9, Chebyshev **105** with `T₀`, **15.1** as actually parameterised | quote 15.1, or 105 for a like-for-like comparison with the monomial figure |
| causal masking is a fidelity detail | causal masking leaves **21.4%** of attention rows with τ spread < 0.1, vs 1.5% under padding-only (full split) | D4 is load-bearing, not cosmetic |
| Figure 1A shows a transformer stack; the code is one un-normalised attention op | the difference changes kernel headroom by **~90×** (equal-norm probe) | Figure 1A and the implementation must be reconciled explicitly; `config.json` records which was run |
| the pediatric claim rests on MIMIC pretraining | over the **full** train split (405M events, not a 64-sequence sample): youngest event age **16.6 y**, **189,861** events (0.047%) under 18, all in the 12–17 band; 3.8% at the 89+ censoring value | the pediatric range of `Δα(a)` is near-pure extrapolation after pretraining. The transfer claim rests on fine-tuning reshaping it from PIC |
| — (new, Fix A) | two earlier code paths computed overlapping corpus stats on different samples and disagreed (min age 17.5 vs 22.3; 12–17 band 186 vs 0) | there is now one function, `corpus_stats`, run once over the full split (cached across arms); `τ_max` and every per-event statistic are exact. Age mean/sd for standardization: **63.3 / 16.6 y** |
| **the frozen MIMIC `τ_max` transfers to PIC** | **M1: on PIC fine-tune data the τ̃ distribution occupies only 2.0% of [−1,1] (all near −1), where the Chebyshev Gram condition number is 5.7e16 vs 15.1 on MIMIC** | **the most important finding of this pass — see §5c. The kernel basis is numerically singular on PIC under the frozen `τ_max`, and both are frozen in the checkpoint** |

---

## 5c. M1 — the frozen `τ_max` makes the kernel basis singular on PIC (review-and-decide)

This is the measurement the brief singled out as the one that cannot be revisited after
pretraining, and it comes back **far past the stop threshold**.

`τ_max` is computed on MIMIC (max window span ~5800 days → `τ_max ≈ 6.72`) and frozen into
the checkpoint so every learned coefficient is defined against the same domain (D8 /
INV-TMAX). PIC stays are ICU-length. Measured on the PIC `heart_malformations` fine-tune
data (1.06M pairwise lags):

| quantity | PIC under frozen `τ_max` | MIMIC pretrain |
|---|---|---|
| τ̃ range | [−1.000, −0.960] | [−1.000, 0.980] |
| fraction of [−1, 1] occupied | **2.0%** | ~99% |
| Chebyshev Gram cond (no `T₀`) | **5.7 × 10¹⁶** | 15.1 |

Chebyshev polynomials are near-orthogonal on the whole `[−1, 1]`, not on a 2% sliver near
−1, where `T₁…T₅` collapse toward a common affine shape and the Gram matrix goes singular to
float precision. **Under the current design the fine-tune kernel has almost no usable degrees
of freedom on PIC** — `Δα(a)` cannot produce distinguishable `w(τ|a)` shapes there, which is
exactly the effect the whole experiment is meant to measure.

The brief's instruction was to print and stop if this exceeded 10³; `preflight` does (it is a
review checkpoint, exits cleanly, trains nothing). **This needs a decision before
pretraining**, because both `τ_max` and the basis are frozen in the checkpoint. The tension
is real and not a bug: `τ_max` *must* be shared for coefficient comparability, yet the shared
value is wrong for PIC. Options, none taken automatically:

1. **Rescale the kernel domain per corpus while keeping the coefficients comparable** — e.g.
   define `τ̃` against a `τ_max` chosen so both corpora occupy a reasonable fraction, accepting
   heavier clamping on MIMIC's long tail. Needs a fresh headroom check on both.
2. **A basis orthogonal on the PIC sub-interval** (shifted Chebyshev on `[−1, τ̃_max^PIC]`), at
   the cost of the clean MIMIC conditioning.
3. **Accept that MIMIC and PIC live on different time scales** and revisit whether a single
   frozen temporal kernel can transfer at all — which is itself a publishable negative.

This connects to the existing note that the PIC CHD shards were clipped to a 24 h window: with
all lags under ~1 day, `τ = log1p(days/7) < 0.13`, hence the 2% sliver. The clipping and the
`τ_max` freeze compound.

---

## 6. Invariants

12 invariant IDs, each mapping to exactly one test, all HARD, all CPU, all under a minute.
See `INVARIANTS.md` for statements. Run:

```bash
python -m model_new.tests.run_all      # runs pytest, then prints the ID → test table
```

`run_all.py` additionally checks the mapping is 1:1 in both directions: no invariant without a
test, no test file unmapped, and every tested ID declared in `INVARIANTS.md`.

Two invariants have scope conditions worth knowing:

- **`INV-QUERY`** (perturbing `age_years[:, j]` changes encoder row `j` only) holds for the full
  encoder at `n_layers=1`. At `n_layers ≥ 2` it *cannot* hold for the stack — layer 2's row `i`
  reads layer 1's row `j` as a value — so it is checked on block 0's output there. That is a
  property of stacking, not a bug.
- **`INV-ZERO-A`** covers `vanilla`/`kernel`/`random_constant` only. `additive` has an `s`-wider
  head, so xavier's `fan_in` differs and its logits differ at init by construction; `INV-ZERO-B` is
  the arm-appropriate form. The *shared backbone* is nonetheless bit-identical across all four
  arms — parameters are re-initialised from **per-parameter** generators seeded by
  `(seed, param name)`, so a shape change in one parameter cannot shift the draws for any other.

---

## 7. Artifacts

Written to `model_new/run/<run_name>/`:

- **`config.json`** — written once *before* the first step. Model config, per-group parameter
  counts, optimizer settings, data paths and corpus statistics, `τ_max` and its source, band
  characterization, `primary_endpoint` (declared before the run and not changed afterwards),
  `deviations_from_draft`, environment, and a `figure_1a_note` stating what was actually run.
- **`train.json`** — a JSON array, rewritten atomically (`.tmp` then `os.replace`) after every
  epoch, so a crashed run leaves a valid file. Per epoch: train/val loss; recall@{5,10,20} overall
  and **stratified by age band**; micro-AUROC tagged `diagnostic_only`; `α_base` and `‖α‖₁` per
  site; the `Δα` constant-vs-varying decomposition overall and per band; `Δα(a)` on a dense 0–90
  grid; `w(τ|a)` on a fixed τ grid at fixed ages; parameter drift per group; clamp rate; attention
  entropy and peakedness; LRs; wall clock; step count.
- **`paper_numbers.json`** — the quantities that appear as `[]` placeholders in the draft.
- **`epoch_NNN.pt`** — checkpoint including `tau_max` and the full config.

Figure 3 (`w(τ|a)` curves) and the `Δα(a)` extrapolation figure are generated from `train.json`
with no extra run.

### 7b. Offline evaluation — `eval_pretrain.py`

`train.json`'s `val_loss` is computed on `--val_max_batches` batches (50 by default) during
training, and its recall figures come from the same subset. That is a training monitor, not an
endpoint. `eval_pretrain.py` re-evaluates finished checkpoints offline on the **full** validation
split and writes:

```
model_new/run/eval_pretrain/<arm>/epochs.json   per-epoch metrics + DKM diagnostics
model_new/run/eval_pretrain/selection.json      the three selection rules  (also model_new/run/selection.json)
model_new/run/eval_pretrain/summary.json        cross-arm table at primary_rule
```

Per arm and per saved epoch: validation BCE (element mean over all `(sequence, code)` pairs, no
`pos_weight`), micro-AUPRC, macro-AUPRC over codes with ≥ `--min_pos` validation positives,
recall@10/@20 with an **uncapped** `|true|` denominator, nDCG@20 — each of them pooled and
stratified by `diagnostics.AGE_BANDS` — plus the gradient probe (per optimizer group, no
`optimizer.step`), the per-site nonzero-gradient fraction of the coefficient generators,
`‖Δα(a)‖₂` on both a dense 0–90 grid and the empirical validation age distribution, centered
kernel separation across representative ages, the equal-norm headroom probe from `preflight`, and
the Chebyshev Gram condition number on the validation lag distribution.

Four properties are structural rather than incidental:

- **One deterministic pass, shared by every metric and every arm.** `shuffle=False`, no dropout,
  `torch.no_grad` outside the gradient probe. The batch sequence is hashed and the hash is
  asserted identical for all four arms, so a difference cannot come from a different batch order.
- **The score matrix is never materialised.** 52,227 × 30,635 is 1.6e9 scores. Micro- and
  macro-AUPRC come from fixed-edge histograms whose edges are chosen once, from a pass over every
  checkpoint, and shared by all arms; the estimator is tested against
  `sklearn.average_precision_score` to 1e-3 in `tests/test_auprc_histogram.py`.
- **Configs are compared before anything is measured.** Differences that are a consequence of the
  arm are verified by rebuilding the model from the *shared* constructor kwargs; anything else is
  a hard error unless named in `--allow_config_diff`, which is then recorded in every output file.
- **`--primary_rule` is required, has no default, and is written to disk before any cross-arm
  number is printed.** The three rules (`per_arm_best`, `vanilla_matched`, `kernel_matched`) are
  all computed and all written regardless.

A band with `n` below `--min_band_n` reports `n`, `n_pos`, `n_neg`, `unreliable: true` and a
reason, with every metric `null`. MIMIC-IV hosp has no patients under 18, so every band below
12–17 is empty and 12–17 itself holds 44 of the 52,227 validation sequences.

---

## 8. Optimizer groups

```
age      = ⋃ module.age_parameters()    over modules that declare it
head     = ⋃ module.head_parameters()   over modules that declare it
backbone = every other trainable parameter
```

Asserted pairwise disjoint, union exactly the trainable set, `age` empty iff `vanilla`. Defaults
`1e-4 / 1e-3 / 1e-3`, all settable, all recorded.

`optim.py` contains no string matching on parameter names, and a test enforces that — the original
failure was a parameter that should have been in the age group sitting in the backbone group at a
3,400× smaller learning rate.

**Expect zero age-group gradient at step 0.** The generator's final layer is zero-initialised, so
its first layer receives exactly zero gradient (`∂L/∂W₁ ∝ W₂ᵀ = 0`) and the pathway warms up over
the first few hundred steps. Do not perturb the init to "fix" this. It is also why the acceptance
signal is **parameter drift**, not gradient norm: under Adam's second-moment normalisation a tiny
gradient still produces a full-size step.

---

## 9. Running it

```bash
# Review checkpoint. Computes everything, prints, exits cleanly. Nothing is trained.
python -m model_new.preflight

# Invariants.
python -m model_new.tests.run_all

# Four-arm pretrain. Identical flags apart from --arm and --run_name.
./model_new/run/pretrain.sh
SEEDS="0 1 2" ./model_new/run/pretrain.sh          # extra seeds are a loop, not an edit

# Early liveness check: stop at step 200 and report.
python -m model_new.train --arm kernel --seed 0 --run_name probe \
    --report_at_step 200 --stop_after_report

# Offline evaluation + epoch selection over finished runs. --primary_rule has no default.
python -m model_new.eval_pretrain --runs model_new/run/vanilla_s0 \
    model_new/run/kernel_s0_072420260946 model_new/run/random_constant_s0_072420261750 \
    model_new/run/additive_s0_072520260143 --primary_rule per_arm_best

# Fine-tune all four arms from ONE shared backbone.
CKPT=model_new/run/vanilla_s0/epoch_008.pt ./model_new/run/finetune.sh
```

All of the above assume the `ehr` conda environment.

---

## 10. Anti-duplication

Each of these exists in exactly one place:

| thing | location |
|---|---|
| `log1p(Δt/7)` | `data.lag_to_tau` (torch, float64), called from `DKMModel.forward` on the GPU via `data.tau_from_timestamps`; `data.spans_to_tau` is the numpy twin (shares the constants). The collate ships `timestamps_days`, not `tau` |
| corpus statistics | `data.corpus_stats` — one full-split pass returning a frozen `CorpusStats`; `preflight` and `train` each call it once (INV-STATS-SINGLE) |
| `τ_max` | `ChebyshevKernel.tau_max`, a persistent buffer; the exact value comes from `corpus_stats`; `DKMModel.tau_max` asserts all sites agree |
| age standardization `(mean, sd)` | `DKMModel.age_mean` / `age_sd` buffers, from `corpus_stats`; applied once in `standardize_demo_age` |
| Chebyshev evaluation | `basis.chebyshev_basis`; `diagnostics.gram_condition_numbers` imports it rather than inlining |
| Fourier frequencies | built once per site in `_FourierBase.__init__`, persistent buffers, never rebuilt at load |
| `w(τ)` statistics | computed once per epoch in `train.py`, emitted once by `diagnostics` |
| age-band definitions | `diagnostics.AGE_BANDS`, shared by metrics, by the `Δα` decomposition and by `eval_pretrain` |
| average precision | `diagnostics.average_precision_from_counts`, used by both the micro and the per-code histogram |
| equal-norm headroom probe | `preflight.headroom`; `eval_pretrain` imports it rather than reimplementing it |
| masking | `encoder.build_pair_mask` / `encoder.build_key_mask`, used by encoder and pooling |
| optimizer groups | `optim.build_param_groups`, from declared sets |
| printing | `diagnostics.py` only |

---

## 11. What to check before changing this code

1. **Do not tune hyperparameters per arm**, run one arm more than another, or choose which arm to
   report. The primary endpoint is in `config.json` before the run.
2. **Keep age in the demographic vector.** Removing it turns `vanilla` into a strawman.
3. **Do not re-derive `τ_max`** at fine-tune. It changes the meaning of every learned coefficient.
4. **Do not add a regulariser on `‖α‖₁`.** A kernel that grows to dominate QK is a finding.
5. **Do not enable `--center_delta_alpha` with `random_constant`** — it makes that arm's `Δα`
   exactly zero and collapses the capacity control onto `vanilla`. The code refuses.
6. **Modules must not print.** Return diagnostic tensors; let `diagnostics.py` format them.
7. If you change an invariant, change `INVARIANTS.md`, its test, and `run_all.INVARIANT_TESTS`
   together — `run_all` fails if they disagree.
