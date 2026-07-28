# Age conditioning (DKM) — implementation reference for the manuscript

**Purpose.** Methods-ready description of how age enters the model in `model_new/`.
Use this when writing the paper; it is the single source for equations, tensor shapes,
hyperparameters, and the four-arm design as implemented.

**Scope.** Pretraining on MIMIC-IV and arm-matched fine-tuning (each arm continues from
**its own** pretrained checkpoint). This document does **not** describe the shared-backbone
fine-tune ablation (one vanilla backbone fine-tuned under every arm), which is not part of
the reported experiment.

**Code.** Self-contained under `model_new/`. Primary modules: `age_encoding.py`, `basis.py`,
`encoder.py`, `pooling.py`, `model.py`, `arms.py`. Broader engineering notes live in
`MODEL_NEW_IMPLEMENTATION.md`.

---

## 1. High-level idea

Developmental Kernel Modulation (DKM) makes the **temporal attention kernel** a smooth
function of patient age. Age does **not** replace content attention; it adds a lag-dependent
bias in log space (FiLM-style additive modulation of a temporal-decay polynomial).

Age reaches the network by **two explicit routes**:

| Route | What it is | Present in |
|---|---|---|
| **R1 — demographic feature** | Standardized age (and sex/race) at the last event, projected and concatenated to the pooled representation | **All** arms |
| **R2 — kernel coefficients** | Raw age → Fourier features → MLP → Chebyshev coefficients → `log w` on attention / pooling scores | **Kernel** (and capacity control `random_constant`) |
| **R3 — additive head** | Generator output concatenated to the pooled vector `h` (no kernel modulation) | **Additive** only |

R1 is the baseline route age already has; it is **not** the experimental variable.
Removing it would make `vanilla` age-blind. The experiment varies **where** an age-conditioned
generator’s output is delivered (kernel bias vs head concat vs unused).

---

## 2. Notation and shapes

For a batch of sequences:

| Symbol | Meaning | Typical shape |
|---|---|---|
| `B`, `L` | batch size, padded sequence length | — |
| `x` | frozen code embeddings | `[B, L, d_emb]` (`d_emb = 1024`) |
| `E` | encoder outputs | `[B, L, d]` (`d = 256`) |
| `h` | pooled patient representation | `[B, d]` |
| `a` | age in years (raw) | `[B, L]` as `age_years` |
| `a_i` | age at query position `i` | scalar per row |
| `a_n` | age at last valid event | `[B]` |
| `τ_ij` | pairwise lag feature | `[B, L, L]` |
| `τ_to_now` | lag from each event to last valid | `[B, L]` |
| `s` | Chebyshev degree (no constant) | `5` |
| `M` | Fourier frequency count | `16` → `ψ ∈ R^{2M}` |

Timestamps and ages are shipped in the batch; pairwise `τ` is computed on device from
`timestamps_days` (not materialized in the DataLoader).

---

## 3. Temporal lag feature

A single convention is used everywhere:

```
τ = log1p(|Δt_days| / 7)
```

- **Encoder:** `τ_ij` from every pair of event times within the window (`pairwise_tau`).
- **Pooling:** `τ_to_now` from each event to the last valid event.

Differencing is done in float64, then cast to float32 after `log1p`, to limit cancellation on
large absolute timestamps.

The Chebyshev domain uses a frozen corpus maximum `τ_max` (exact max over the pretraining
split, stored in the checkpoint):

```
τ̃ = clamp(2 · τ / τ_max − 1, −1, 1)
```

Every learned coefficient is defined against this domain; fine-tuning reuses the same
`τ_max` (does not recompute from PIC).

---

## 4. Age → Fourier features → coefficient deltas

### 4.1 Log-age Fourier embedding `ψ(a)`

Age is mapped through a **fixed** (non-learned) Fourier feature map on a **log-age**
coordinate. Periods are log-spaced in that coordinate (not on linear years).

```
u(a)   = log1p(max(a, 0))
p_m    = exp(linspace(log p_max, log p_min, M))     # m = 1..M
ψ(a)   = [ sin(2π u / p_m), cos(2π u / p_m) ]_{m=1..M} ∈ R^{2M}
```

**Defaults:** `M = 16`, `p_min = 0.15`, `p_max = 6.0` → `ψ ∈ R^{32}`.
With `a ∈ [0, 90]`, `u` spans `[0, log1p(90) ≈ 4.51]`. `p_max > 4.51` gives one
sub-cycle component that acts as a slow global coordinate; resolution is denser at young ages
(developmental rate) than a linear-year band.

**Why log-age.** A legacy band with periods `1/12 … 200` years on **linear** age behaves like
a near-orthogonal hash of age: nearby pediatric ages are not closer in embedding space than
distant ages, so the cheapest generator solution is a near-constant `Δα`, which is absorbed by
`α_base` and yields an age-invariant offset. Log-age allocates resolution by developmental
rate.

Frequencies / periods are persistent buffers, restored from checkpoint, never rebuilt at load.

### 4.2 Coefficient generator

```
Δα(a) = MLP(ψ(a)) ∈ R^s
α(a)  = α_base + Δα(a)
```

MLP: `Linear(2M → H) → GELU → Linear(H → s)` with `H = 64`, `s = 5`.

- Final layer **weights zero-initialized** → `Δα ≡ 0` at init → all arms start from the same
  kernel (`α = α_base`).
- Final layer **bias omitted by default** (optional `--gen_final_bias`). A free bias makes a
  constant `Δα` the cheapest direction; a constant is fully absorbed by `α_base`.

Optional centering (`--center_delta_alpha`): subtract the mean of `Δα` over a fixed reference
age grid `[0, 90]` at 0.5 y spacing. **Must not** be used with `random_constant` (that arm’s
`Δα` is constant, so centering zeros it and collapses the control onto `vanilla`).

### 4.3 `AgeConditioner` as one site

Each kernel site owns its own `AgeConditioner` = `(LogAgeFourier, CoefficientGenerator)` plus
its own `α_base`. Encoder layers and pooling **share form only**, not parameters.

```
age_years  →  ψ(a)  →  Δα(a)  →  α = α_base + Δα
```

---

## 5. Chebyshev temporal kernel

### 5.1 Basis (no constant term)

```
log w(τ; α) = Σ_{k=1}^{s} α_k · T_k(τ̃)
```

`T_k` are Chebyshev polynomials of the first kind on `τ̃ ∈ [−1, 1]`. The constant `T_0` is
**never** a free coefficient: softmax is invariant to a per-row additive constant, and within
one attention row the query age (hence any `α_0(a)`) is fixed. Carrying `α_0` only adds a
null direction for the optimizer.

`α` is zero-initialized (`α_base = 0`, `Δα = 0` at init) → `log w ≡ 0` at the start of training;
population decay is learned rather than assumed.

### 5.2 Why Chebyshev (vs monomials)

Legacy kernels used raw monomials `[1, τ, …, τ^5]` on `τ ∈ [0, τ_max]`. On ~1.1M empirical
MIMIC pairwise lags the Gram condition number is ~`2.3×10^9`. Chebyshev on `τ̃` as
parameterized (no `T_0`) measures **~15.1**; with `T_0` included for a like-for-like
comparison, ~105. Quote **15.1** for the implemented basis.

### 5.3 Broadcast conventions

| Site | `τ` | `α` | Conditioning |
|---|---|---|---|
| Encoder attention | `[B, L, L]` | `[B, L, s]` | **Query** age `a_i` for row `i` (broadcast along keys) |
| Attention pooling | `[B, L]` | `[B, s]` | Last-event age `a_n` |

Conditioning on the **key** age would put a different kernel shape in every entry of one
softmax row, making weights incomparable. Query-only conditioning is intentional.

---

## 6. Where the kernel is injected

### 6.1 Encoder: time-aware self-attention

For each encoder layer (default `n_layers = 1`, standard pre-LN block with residual, LayerNorm,
and FFN):

```
Q, K, V     = MLP_q(x), MLP_k(x), MLP_v(x)
content     = Q K^⊤ / √d_head
log w_ij    = Σ_k α_k(a_i) T_k(τ̃_ij)
scores_ij   = content_ij + log_w_ij          # direct log-space injection
attn        = softmax(scores; padding mask only)
E           = attn V   (+ optional out-proj if n_heads > 1)
```

Masking is **padding-only** (bidirectional within the input window). The pretraining target
visit lies **outside** the window, so this is not label leakage. Causal masking collapses
within-row lag diversity (many rows with near-constant `τ`), leaving the kernel little to
discriminate.

The same per-query age conditions the kernel at **every** layer when `n_layers > 1`.

### 6.2 Pooling: single-query attention over the sequence

```
relevance_j = q_base · E_j
log w_j     = Σ_k α_k(a_n) T_k(τ̃_to_now_j)
scores_j    = relevance_j + log_w_j
h           = softmax(scores; valid keys) · E
```

Same log-space injection as the encoder. (Legacy pooling multiplied a signed relevance by
`w ∈ (0,1)`, which *raised* attention on negative-relevance events — corrected here.)

---

## 7. Patient representation → pretraining / fine-tune head

```
demo_last   = demographics at last valid index
              (channel 0 = age, standardized with frozen pretrain mean/sd;
               other channels untouched: sex + race one-hot by default)
u           = [ h ; demo_proj(demo_last) ]          # kernel / vanilla / random_constant
            = [ h ; demo_proj(demo_last) ; g(ψ(a_n)) ]   # additive only, g ∈ R^s

pretrain:   code_logits = Head(u) ∈ R^{|V|}         # Linear → GELU → Linear, bias −7
finetune:   logit       = Head(u) ∈ R                 # same backbone path; new 1-d head
```

**Demographic age standardization.** Raw age (median ~56 on MIMIC) beside 0/1 covariates would
dominate `demo_proj` at init. Channel 0 is standardized with corpus `(mean, sd)` frozen in the
checkpoint (~63.3 / 16.6 y on MIMIC). **`age_years` fed to `ψ` stays raw** — only the
demographic channel is standardized.

For the **kernel** arm, age shapes `h` only through attention/pooling (`R2`). There is no
extra age concat onto `h` (`R3` is additive-only).

---

## 8. Experimental arms

`--arm` is the only intentional difference across runs. Seed, data, schedule, optimizer
settings, masking, and backbone code are shared.

| Arm | R1 demo age | Kernel `Δα` (R2) | Concat to `h` (R3) | Age into `ψ` |
|---|---|---|---|---|
| `vanilla` | yes | ≡ 0, **no age params** | — | unused |
| `random_constant` | yes | generator on a **fixed random vector** (age bypassed) | — | bypassed |
| `additive` | yes | ≡ 0, no kernel age params | `generator(ψ(a_n)) → R^s` | real, last event |
| `kernel` | yes | `generator(ψ(a_i))` at encoder + pool | — | real, per query / `a_n` |

**Interpretation.**

- `vanilla` — content + demographic age only.
- `kernel` — the proposed model: age-dependent temporal kernel.
- `random_constant` — capacity-matched control: same generator size, no real age signal.
  Its constant `Δα` is absorbed by `α_base`; it should track `vanilla`. A large gap is a bug
  signal, not a finding.
- `additive` — age features concatenated at the head (not through the temporal kernel).

**Fine-tuning (reported setup).** Arm-matched: fine-tune each arm from **that arm’s**
pretrained weights. Frozen across pretrain → fine-tune: `τ_max`, age standardization
`(mean, sd)`, Fourier frequency buffers, race encoding / ordering, Chebyshev degree `s`.

---

## 9. Architecture defaults (as run)

| Quantity | Default | Notes |
|---|---|---|
| `d_model` | 256 | |
| `n_layers` | 1 | kernel at every layer if > 1 |
| `n_heads` | 1 | |
| Block | residual + LayerNorm + FFN | `--legacy_block` disables all three |
| Code embeddings | frozen BGE table | not trained |
| `s` | 5 | `T_1…T_5` only |
| Fourier `M`, `(p_min, p_max)` | 16, (0.15, 6.0) | log-age |
| Age MLP hidden | 64 | |
| Demo | age + sex + 7-way race one-hot → `demo_dim = 9` | `--race_encoding scalar` → 3 |
| Demo proj hidden | 64 | |
| Pretrain loss | multilabel BCE over codes | no time-gap / Weibull head |
| Masking | padding only | |
| Kernel injection | additive in log space | `scores ← scores + log w` |

**Optimizer groups** (module-declared membership, not name matching):

| Group | Contents | Default LR |
|---|---|---|
| `age` | all `AgeConditioner` / generator params | `1e-3` |
| `head` | prediction head | `1e-3` |
| `backbone` | everything else trainable | `1e-4` |

Expect ~zero age-group gradient at step 0 (final generator layer zero-init). Warm-up of the
age pathway over early steps is expected; do not “fix” by perturbing that init.

---

## 10. Compact equation block (copy for Methods)

With query age `a_i`, lag feature `τ_ij = log1p(|t_i − t_j|/7)`, and frozen `τ_max`:

```
u_i        = log1p(a_i)
ψ(a_i)     = [sin(2π u_i / p_m), cos(2π u_i / p_m)]_{m=1}^M
Δα(a_i)    = MLP(ψ(a_i)) ∈ R^s
α(a_i)     = α_base + Δα(a_i)
τ̃_ij      = clamp(2 τ_ij / τ_max − 1, −1, 1)
log w_ij   = Σ_{k=1}^s α_k(a_i) T_k(τ̃_ij)
score_ij   = (q_i · k_j)/√d + log w_ij
```

Pooling uses the same kernel form with a single age `a_n` and lags to the present.
The patient vector is `h` from pooling, concatenated with a projection of demographics
(including standardized age), then a two-layer head.

---

## 11. Design choices that belong in the paper (not just the code)

1. **Age-dependent kernel, not age lookup / ALiBi / AdaLN.** Modulation is additive in the
   log-attention scores via a polynomial of lag.
2. **Query-conditioned** encoder kernel; **last-event-conditioned** pooling kernel.
3. **Log-age Fourier** (smooth developmental coordinate) vs linear-age Fourier (near-hash).
4. **Chebyshev without `T_0`** on a frozen `[−1,1]` lag domain.
5. **Zero-init `Δα`** so arms share an identical initial kernel; age pathway learns residual
   shape.
6. **R1 kept in every arm** so the baseline is not age-blind.
7. **`random_constant`** as a capacity control that should track `vanilla`.
8. **Padding-only** attention so within-row lag diversity remains available to the kernel.
9. **Arm-matched fine-tune** with frozen `τ_max`, age standardization, and Fourier buffers so
   coefficients keep the same meaning as at pretraining.

---

## 12. Tensor-flow checklist (kernel arm)

```
codes → frozen embeddings x
timestamps → τ [B,L,L], τ_to_now [B,L]

age_years (raw)
  → LogAgeFourier → ψ
  → CoefficientGenerator → Δα
  → α = α_base + Δα
  → ChebyshevKernel(τ, α) → log w
  → scores = content + log w → E   (each encoder layer)

age_last = age_years[:, last]
  → same pathway → log w on τ_to_now
  → h = softmax(q·E + log w) · E

demographics[last] (age standardized) → demo_proj → concat with h → Head → logits
```

Fourier never sees timestamps; Chebyshev never sees age. They meet only through **`α(a)`**.

---

## 13. File map (for authors chasing a detail)

| Topic | File |
|---|---|
| Arm resolution | `model_new/arms.py` |
| Fourier + generator | `model_new/age_encoding.py` |
| Chebyshev kernel | `model_new/basis.py` |
| Encoder attention + mask | `model_new/encoder.py` |
| Pooling | `model_new/pooling.py` |
| Full forward, demo std, head | `model_new/model.py` |
| `τ` definition, demographics | `model_new/data.py` |
| Optimizer groups | `model_new/optim.py` |

---

*Derived from the `model_new/` implementation and `MODEL_NEW_IMPLEMENTATION.md`, restricted to
the age-conditioning formulation and the four-arm, arm-matched experiment as reported.*
