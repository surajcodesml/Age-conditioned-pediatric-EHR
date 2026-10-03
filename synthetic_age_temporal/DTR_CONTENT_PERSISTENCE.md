# Content-Persistence DTR: `dtr_age_temporal_new` and `dtr_temporal_only`

Both names are the locked Content-Persistence Developmental Temporal Retrieval model in `synthetic_age_temporal/model_dtr.py` (`DevelopmentalTemporalRetrieval`). The `_new` suffix selects this architecture. Unsuffixed `dtr_age_temporal` / `dtr_no_interaction` are the older Minimal-DKM transformer and are a different model.

The two arms are the same module, the same inputs, the same loss, and the same optimizer. They differ in one branch of `forward`:

| Arm | Directory name | `age_temporal` flag | Decay rate |
|---|---|---|---|
| Age × temporal | `dtr_age_temporal_new` | `True` | `λ_m = softplus(θ_m + β z(a*))` |
| Temporal only | `dtr_temporal_only` / `dtr_temporal_only_new` | `False` | `λ_m = softplus(θ_m)` |

`β` is a scalar `nn.Parameter`, initialized at 0. On the temporal-only arm, `β.requires_grad_(False)` and the forward never reads `β`. A nonzero value stored in a temporal-only checkpoint does not change logits.

At `β = 0` the two forwards are the same function, so logits match to numerical noise (`matched_arm_init_check`, atol `1e-6`).

Age still reaches the logit on both arms through the additive head `f_age(z)`. “Temporal only” means the decay rate does not depend on age. It does not mean the model is blind to age.

---

## 1. What the model is asked to do

One patient example is a variable-length list of encounters, each a bag of clinical codes, plus the patient’s age at prediction time. The output is a vector of logits, one per target:

- Synthetic benchmark: `T = 32` multilabel targets (or a subset, if a caller filters `target_idx`).
- MIMIC Stage-1 and NCH Stage-2: `T = n_codes` (vocabulary width, about 30k), next-visit multilabel.

The loss on every `_new` training path is mean `BCEWithLogits` over the `B × T` entries.

---

## 2. Inputs

`forward` reads five tensors. Extra keys are ignored (`**kwargs`).

| Tensor | Shape | Meaning |
|---|---|---|
| `enc_code_ids` | `[B, M, C]` long | Code ids. 0 is padding in the synthetic embedding. |
| `enc_code_mask` | `[B, M, C]` bool | `True` = real code inside the encounter. |
| `enc_tau` | `[B, M]` float | Lag of encounter `m` from prediction time, `τ_m = log(1 + Δt_days / 7)`. |
| `enc_padding_mask` | `[B, M]` bool | `True` = this encounter slot is padding. |
| `age` | `[B]` float | Age in years at prediction time, `a*`. |

`M ≤ 64` encounters, newest kept if the history is longer. `C ≤ 64` codes per encounter on MIMIC/NCH and `C ≤ 32` on the synthetic DTR dataset. Width is dynamic per batch; the caps are upper bounds applied by the collate, not inside `forward`.

### How an encounter is built

**MIMIC / NCH** (`baselines/mimic/encounter_batch.py`): valid tokens are sorted by `timestamps_days`. Consecutive tokens with the same timestamp are one encounter. `Δt` is `t_last − t_encounter`, where `t_last` is the timestamp of the last valid token, so the newest encounter has `τ = 0`. Age is the age on that last valid token. Codes within an encounter keep sort order; overflow past 64 codes is dropped from the end of that chunk.

**Synthetic** (`synthetic_age_temporal/encounters.py`, `dataset_dtr.py`): events are truncated the same way as the event-level benchmark, the query/prediction token is dropped, and events whose lag-from-cutoff agrees to 0.001 day are one encounter. Encounter lag and `τ` are the means of the member events. Order is oldest → newest. The newest encounter has the smallest lag, which is positive whenever the history ends before the cutoff.

`τ` is the only time feature. The encoder never sees age, lag, absolute time, or code type.

### Age standardization

```
z(a*) = (a* − 9) / 9
```

Constants are `AGE_CENTER = 9`, `AGE_SCALE = 9` in `synthetic_age_temporal/config.py`. Then `z(0) = −1`, `z(9) = 0`, `z(18) = 1`. There is no clamp. One `z` is computed per patient and broadcast across encounters. The age of the patient at the time of an old encounter is not an input. `τ` and `a*` are separate; the model does not form `a* − lag`.

---

## 3. Parameters

`d = 64` for every `_new` arm (`CP_D_MODEL` / `DTR_D_MODEL`). Dropout on the CP path is 0, so the `Dropout` module is the identity in train and eval.

### 3.1 Encounter encoder `f_enc`

Synthetic: `nn.Embedding(n_codes, d, padding_idx=0)`, then

```
mean_m = (Σ_c e_{m,c} 1_{mask}) / max(1, Σ_c 1_{mask})
v_m = W2 GELU(W1 mean_m + b1) + b2
```

`W1, W2` are `d × d`. An encounter with no valid codes yields `mean = 0` and `v = MLP(0)`.

MIMIC / NCH replace the embedding after construction (`MIMICCPDTRAdapter._init_frozen_bge_code_table`):

- `code_emb_table` is a persistent buffer, the frozen BGE table, rows `n_codes + 2` (pad and unk). It is not an `nn.Parameter`.
- Ids are clamped to `[0, n_emb − 1]`, then `e = code_proj(table[id])`.
- `code_proj` is `Linear(bge_dim, d, bias=False)`, Xavier uniform.
- The same mean-pool and MLP follow.

Gradient stops at the BGE vectors. It enters `code_proj` and the MLP.

### 3.2 Content relevance

```
k_m = W_k v_m                         # Linear(d, d), no bias
u_m = (q · k_m) / √d                  # q is a free vector in R^d
```

`q` (`content_query`) is initialized `N(0, 0.02²)`. `W_k` (`content_key`) uses the default Kaiming uniform. The module comment writes `qᵀ k_m`; the executed op divides by `√d = 8`. The scale is a constant. It does not change the ranking of encounters. It scales gradients into `q` and `W_k` by `1/8`.

Padded encounters are then overwritten: `u_m ← 0`. That overwrite blocks gradient into `q` and `W_k` from pad slots.

### 3.3 Content persistence

```
θ_content,m = (r · v_m + b_r) · 1_{m valid}
θ_m = θ₀ + θ_content,m
```

`persistence_projection` is `Linear(d, 1, bias=True)`. `r` is its weight (shape `[1, d]`), `b_r` its bias. Both are initialized to 0, so at step 0, `θ_m = θ₀ = 0` for every encounter.

`θ₀` and `b_r` are the same kind of scalar. On a valid encounter,

```
θ_m = θ₀ + b_r + r · v_m
```

They start equal (both 0) and, under the `_new` trainer, receive the same gradient and the same AdamW update, so they stay equal: `θ₀(t) = b_r(t)`. The decay the loss sees is `softplus(2 θ₀ + r · v_m + …)`. The two names are not separately identifiable. Weight decay pulls each toward 0, which pulls the effective baseline `θ₀ + b_r` toward 0 at twice the rate of a single parameter.

`r` starts at 0, so the persistence path into `v` is 0 on the first backward. The gradient into `r` itself is not 0 (see §6.4). The first step can move `r`; later steps feed that change back into `v`.

### 3.4 Developmental scalars

| Name | Shape | Init | Age × temporal | Temporal only |
|---|---|---|---|---|
| `θ₀` | `[1]` | 0 | trained | trained |
| `β` | `[1]` | 0 | trained | frozen, and absent from the graph |

`age_parameters()` returns `(θ₀, β)` filtered by `requires_grad`. Temporal only therefore reports only `θ₀`. The `_new` baseline trainer does not use that split; see §7.

### 3.5 Heads

```
h = Σ_m w_m v_m                         # [B, d], raw sum
history_logit = W_h2 GELU(W_h1 h + b_h1) + b_h2
age_logit = W_a z + b_a                 # Linear(1, T)
logit = history_logit + age_logit + b
```

`b` is `self.bias`, shape `[T]`, initialized at 0.

On MIMIC/NCH construction only, the last history linear (`W_h2`, `b_h2`), the whole age head (`W_a`, `b_a`), and `b` are set to 0. The initial logit is then the zero vector and the initial BCE is `log 2`, regardless of `h` and `z`. Synthetic `_new` leaves the Kaiming init on those layers, so the initial logit is nonzero.

`b_h2`, `b_a`, and `b` are three per-target constants added into the same logit. Their gradients are identical (each receives `∂L/∂logit`). On MIMIC/NCH they start at 0 and share one AdamW group, so they remain equal and the effective bias is `3b`. On synthetic they start at different Kaiming values; the logit depends only on the sum, and weight decay moves each toward 0 from its own value.

`M = Σ_m w_m` is computed and stored. The canonical aggregation is `"raw_additive"`, which does not divide by `M`. `M` is not on the path to the loss. The alternate `"weighted_mean_plus_log_mass"` path exists in code and is not used by any `_new` arm.

---

## 4. Forward, age × temporal

This is `DevelopmentalTemporalRetrieval.forward` with `age_temporal=True`. `hist` is the valid-encounter mask.

```
v_m     = f_enc(C_m)                                          # no age, no τ
k_m     = W_k v_m
u_m     = (q · k_m) / √d          if m valid, else 0
θ_m     = θ₀ + (r · v_m + b_r)    if m valid, else θ₀
z       = (a* − 9) / 9
s_m     = θ_m + β z
λ_m     = softplus(s_m)           if m valid, else 1
g_m     = exp(−λ_m τ_m) · 1_{m valid}
ũ_m     = min(u_m, 20)
w_m     = exp(ũ_m) · g_m · 1_{m valid}
h       = Σ_m w_m v_m
logit   = f_history(h) + f_age(z) + b
```

`softplus(s) = log(1 + exp(s)) > 0`, so `λ_m > 0` and, for `τ_m ≥ 0`, `g_m ∈ (0, 1]` on valid encounters. Pad slots are forced to `λ = 1` only so `exp(−λτ)` stays finite; they are then multiplied by 0.

`CONTENT_SCORE_EXP_CLAMP = 20`. `exp(20) ≈ 4.85×10^8`. The clamp does not renormalize across encounters. For `u ≤ 20`, `w = exp(u) g`. Weights do not sum to 1. Raising one encounter’s weight does not reduce another’s. `h` grows with the number of encounters and with `exp(u)`.

### Where age and lag enter

Age enters in exactly two places:

1. `β z` inside `s_m`, which changes every encounter’s decay. This term exists only on `dtr_age_temporal_new`.
2. `f_age(z)`, an additive shift of the logit that does not depend on the history. This term exists on both arms.

`τ` enters in exactly one place: the product `λ τ` inside the exponential. It is not a feature of the encoder, the query, or the persistence projection.

### Value of the gate at initialization

`θ_m = 0`, `β = 0`, so `λ = softplus(0) = log 2 ≈ 0.693` for every valid encounter, and `g = 2^{−τ}`.

| Lag | `τ = log(1 + days/7)` | `g` at init |
|---|---|---|
| 0 days | 0 | 1 |
| 7 days | 0.693 | 0.618 |
| 30 days | 1.665 | 0.315 |
| 365 days | 3.973 | 0.064 |

On MIMIC the newest encounter has `τ = 0`, so its gate is 1 for any `λ`. Changing `β`, `θ₀`, or `r` cannot change that encounter’s gate. On synthetic the newest lag is whatever remains before the cutoff.

### Cache

After the logit is formed, detached copies of `u, g, w, M, λ, θ_m, θ_content, h, history_logit, age_logit, v` are stored on `self._cache`. Detach happens after the loss path has already used the live tensors. The cache is not an input to the loss. Reading it cannot carry gradient.

`return_parts=True` returns the live tensors. Training does not use that flag.

---

## 5. Forward, temporal only

Same equations as §4, with this branch instead:

```
λ_m = softplus(θ_m)     if m valid, else 1
```

`β` is not an operand. `z` is still computed and still drives `f_age(z)`. Content persistence `r · v_m + b_r` is still inside `θ_m`. The temporal-only arm is not a global-`θ₀`-only gate. The adapter docstring that writes `λ = softplus(θ₀)` is behind the locked forward.

Because `β` is also `requires_grad=False`, it is absent from the optimizer. Either fact alone is enough to keep it at 0. Both are in effect.

---

## 6. Gradient flow

The `_new` loss is

```
L = mean BCEWithLogits(logit~, y)
```

with `y ∈ {0,1}` and `logit~` defined below. For one entry, before the mean,

```
∂ℓ/∂logit_k = σ(logit_k) − y_k
```

Mean reduction divides every gradient by `B · T`.

### 6.1 Logit clamp (MIMIC and NCH only)

`MIMICCPDTRAdapter.training_step` and `predict` clamp logits to `[−20, 20]` before the loss and before metrics. Synthetic `DTRAdapter.training_step` does not clamp.

`clamp` passes gradient where the pre-clamp logit is inside `[−20, 20]`, including the endpoints, and zeros it outside. A saturated target contributes nothing upstream for that example. Inside the interval, `∂L/∂logit` is the usual BCE term.

### 6.2 Split at the sum

```
logit = history_logit + age_logit + b
```

`∂L/∂logit` is copied unchanged into all three branches.

**Bias `b`, age-head bias `b_a`, history-head bias `b_h2`.** Each receives `∂L/∂logit`. No other tensor sits between them and the loss.

**Age head.** `age_logit = W_a z + b_a`. `z` is a constant function of the input age. Gradient updates `W_a` and `b_a` and stops. It does not enter `β`, `θ₀`, `r`, the encoder, the query, or the gate. The age main effect and the age × decay interaction are separate parameters with separate backward paths. Supervision can fit an age shift without ever moving `β`, and can move `β` without that update passing through `f_age`.

**History head.** Standard MLP backward through `Linear → GELU → Linear` produces `∂L/∂h ∈ R^d`. GELU’s derivative is `Φ(x) + x φ(x)` at the pre-activation. From here every history parameter is reached only through `h`.

### 6.3 From `h` into weights and values

```
h = Σ_m w_m v_m
∂L/∂w_m = (∂L/∂h) · v_m          # scalar
∂L/∂v_m ⊇ w_m (∂L/∂h)            # value path; more terms below
```

There is no softmax denominator, so there is no competitive term that would subtract a share of the gradient from the other encounters. An encounter with `w_m = 0` (a pad, or a gate that has underflowed) sends no value-path gradient into its `v_m`.

`w_m = exp(ũ_m) g_m` on valid encounters. Two factors:

```
∂w/∂g = exp(ũ)
∂w/∂ũ = exp(ũ) g = w          when u < 20
∂w/∂ũ = 0                     when u > 20
```

The content-score clamp stops the query/key gradient and does not stop the gate gradient. For the gate, `∂w/∂λ = exp(ũ) ∂g/∂λ`, and `w = exp(ũ) g`, so

```
∂g/∂λ = −τ g
∂w/∂λ = −τ w
```

holds on both sides of the clamp. A saturated content score still teaches `λ`. It stops teaching `q` and `W_k`.

### 6.4 Gate parameters

```
g = exp(−λ τ)
λ = softplus(s)
∂λ/∂s = σ(s) ∈ (0, 1)
```

At initialization `s = 0`, so `σ(s) = 1/2`. Half of the pre-activation gradient reaches `θ_m` and `β`.

**Lag multiplier.** `∂w/∂λ = −τ w`. If `τ_m = 0`, this factor is 0. The encounter still contributes `w_m = exp(ũ_m)` to `h` and still trains the content query through `u`, but it contributes nothing to `β`, `θ₀`, `b_r`, or `r`. On MIMIC that is the newest encounter. Encounters with very large `λτ` have `g ≈ 0` and `w ≈ 0`, so `−τ w ≈ 0` as well: a fully decayed encounter also stops teaching the gate.

**`θ₀` and `b_r` (both arms).** For every valid encounter `∂s/∂θ₀ = ∂s/∂b_r = 1`. Pad slots are multiplied out (`θ_content` by the history mask, `λ` overwritten to the constant 1), so they add nothing.

```
∂L/∂θ₀ = ∂L/∂b_r
       = Σ_{valid m} (∂L/∂h · v_m) (−τ_m w_m) σ(s_m)
```

**Persistence direction `r` (both arms).** `s` depends on `r · v_m`.

```
∂L/∂r = Σ_{valid m} (∂L/∂h · v_m) (−τ_m w_m) σ(s_m) v_mᵀ
```

At step 0, `r = 0`, but `v_m` and `(∂L/∂h · v_m)(−τ w)σ(s)` are generally nonzero, so `r` gets a gradient immediately. The path from `r` back into `v` (below) is zero until `r` leaves zero.

**`β` (age × temporal only).** `s_m = θ_m + β z` and `z` is shared by every encounter of the patient.

```
∂L/∂β = Σ_{valid m} (∂L/∂h · v_m) (−τ_m w_m) σ(s_m) z
```

`z` factors out of a patient’s encounters:

```
∂L/∂β = Σ_patients  z_patient · (Σ_{m in patient} (∂L/∂h · v_m) (−τ_m w_m) σ(s_m))
```

Sign, with the rest of the chain held fixed:

- `z > 0` (`a* > 9`): increasing `β` increases `λ` for every encounter of that patient (faster decay).
- `z < 0` (`a* < 9`): increasing `β` decreases `λ` (slower decay).
- `z = 0` (`a* = 9`): that patient contributes 0 to `∂L/∂β`. Age-9 examples still train `θ₀` and `r`.

`β` is one global scalar. Every target and every encounter writes into the same number. Targets that want different age trends compromise in that sum.

On the temporal-only arm this product is not formed. `β.grad` stays `None`.

### 6.5 Three paths back into `v_m`

`v_m` is used as the retrieved vector, as the input to `W_k`, and as the input to `r`. The total `∂L/∂v_m` is the sum of:

1. **Value path.** `w_m ∂L/∂h`. Present on both arms. Scale is the retrieval weight. Independent of `β` except insofar as `w_m` depends on `λ(β)`.

2. **Content-score path.** Only when `u_m < 20`.

   ```
   ∂u/∂v = W_kᵀ q / √d
   ∂L/∂v ⊇ (∂L/∂w) (∂w/∂u) W_kᵀ q / √d
   ```

   `∂w/∂u = w` below the clamp, and `w` contains `g = exp(−λτ)`. The gate scales how strongly an encounter teaches the query. An old, heavily decayed encounter contributes little gradient to `q` and `W_k`. Age changes this path only by changing `λ` through `β`.

3. **Persistence path.**

   ```
   ∂s/∂v = r
   ∂L/∂v ⊇ (∂L/∂w) (−τ w) σ(s) r
   ```

   Zero at initialization because `r = 0`. Afterwards, content that the gate wants to decay faster is pushed in the `r` direction.

Those three gradients add, then enter the encoder.

### 6.6 Encoder

Mean pool over valid codes, equal weight `1/n_valid`:

```
∂L/∂e_c = (∂L/∂mean) / n_valid     for codes with mask 1
∂L/∂e_c = 0                         for masked codes
```

Then backward through `W2`, GELU, `W1`.

Synthetic embedding: the gradient accumulates into the rows that were looked up. `padding_idx=0` has its gradient zeroed by `nn.Embedding` even if a mask were wrong. Masked positions already contribute 0 from the pool.

MIMIC/NCH: `e = code_proj(BGE[id])`. BGE is a buffer. Gradient updates `code_proj.weight` and stops. Ids are not differentiable. The clamp of ids to the table range is a discrete gather.

### 6.7 What is disconnected

| Tensor / parameter | Why no gradient reaches it |
|---|---|
| `β` on temporal only | Not an input of `forward`, and `requires_grad` is false |
| Frozen BGE rows | Registered buffer, not a parameter |
| Pad encounters | `u`, `θ_content`, `g`, and `w` are multiplied or filled by 0 |
| `τ = 0` encounters, into the gate | `∂g/∂λ = −τ g = 0` |
| `u > 20`, into `q` and `W_k` | Clamp blocks `∂w/∂u`; gate path stays open |
| `f_age` parameters, from the gate | `z` is data; the head does not call into `λ` |
| Gate parameters, from `f_age` | Additive split; `∂L/∂age_logit` stops at `W_a`, `b_a` |
| `_cache` | `.detach()` |
| `M` on the canonical path | Computed, never used to form `logit` |
| Logits outside `[−20, 20]` on MIMIC/NCH | Clamp in the adapter, upstream gradient 0 |

Dropout does not disconnect anything: `p = 0`.

### 6.8 First optimizer step

At `β = 0`, `s_m = θ_m` on both arms, so the forwards match and so do `∂L/∂θ₀`, `∂L/∂r`, `∂L/∂q`, and every encoder and head gradient. `∂L/∂β` is already nonzero on the age × temporal arm whenever some patient has `z ≠ 0` and some encounter has `τ > 0` and `w > 0`. The temporal-only arm has no such term.

After that step, `β` leaves 0 only on `dtr_age_temporal_new`. Gates diverge. From the second step on, gradients of the shared weights diverge too, because `w` and `σ(s)` now depend on `β z`.

Weight decay on `β` is `0` while `β` is `0`, so the first move is entirely the data gradient above.

---

## 7. Training procedure used by the `_new` runs

All three runners call `baselines.common.training.train_neural_baseline`. One AdamW group over every parameter with `requires_grad=True`. No separate learning rate for `β` or `θ₀`. The 10× learning-rate, zero-decay group in `synthetic_age_temporal/train_dtr.py` is a different trainer and is not what wrote `results/baselines/*/dtr_*_new`.

| | Synthetic | MIMIC Stage-1 | NCH Stage-2 |
|---|---|---|---|
| Class | `DTRAdapter` | `MIMICCPDTRAdapter` | same adapter as MIMIC |
| Code features | learned embedding | frozen BGE + linear | frozen BGE + linear |
| `d` | 64 | 64 | 64 |
| Dropout | 0 | 0 | 0 |
| Head init | Kaiming | last layer, age head, and `b` zeroed | same, then checkpoint load |
| Loss | BCE-with-logits | BCE-with-logits on clamped logits | same as MIMIC |
| Optimizer | AdamW, lr `3e-4`, wd `1e-2` | same | same |
| Grad clip | 1.0 | 1.0 | 1.0 |
| Epochs | 25, patience 5, min epochs 12 | 10, patience 3, min epochs 5 | same as MIMIC |
| Batch | 32 | 32 | 128 |
| Selection | best validation BCE | best validation BCE, val capped at 50 batches | same cap |
| Init | independent, `β = 0` | independent, `β = 0` | both arms load MIMIC `dtr_temporal_only_new`, then `β` is filled with 0 |

Early stopping cannot fire before `min_epochs`. The best state is reloaded into the module at the end of training. Checkpoints: `best_checkpoint.pt`, `last_checkpoint.pt`, `checkpoint.pt` (a copy of best).

NCH `apply_shared_cp_init` loads that shared MIMIC temporal-only checkpoint into both arms, sets `β = 0`, freezes `β` on temporal only, and leaves `β` trainable on age × temporal. A fingerprint over the state dict excluding `β` is required to match before training. Stage-2 therefore starts from one set of weights. The first NCH step matches §6.8: shared gradients agree, and only the age × temporal arm receives `∂L/∂β`. Later steps diverge.

`β` on the age × temporal arm is in the single AdamW group, so it is trained at `3e-4` with weight decay `1e-2`, the same as `θ₀` and the encoder.

---

## 8. Component list, in forward order

1. **Code table.** Synthetic embedding, or frozen BGE gather plus `code_proj`. Produces per-code vectors. No age, no lag.
2. **Masked mean.** Bag-of-codes inside the encounter. Pad codes drop out. Empty bags become the zero vector.
3. **Encoder MLP.** `Linear → GELU → Dropout(0) → Linear`. Output `v_m`.
4. **Content key and query.** One global query for all targets. Score `u_m`, scaled by `√d`, zero on pads.
5. **Persistence projection.** `r · v_m + b_r`, zeroed on pads, added to `θ₀`.
6. **Age standardization.** `z = (a* − 9) / 9`, one scalar per patient.
7. **Decay.** `softplus(θ_m + β z)` or `softplus(θ_m)`. Positive by construction.
8. **Temporal gate.** `exp(−λ τ)` on valid encounters, 0 on pads.
9. **Retrieval weight.** `exp(min(u, 20)) · g`. Unnormalized.
10. **Raw sum.** `h = Σ w_m v_m`. Mass is kept. No attention normalization.
11. **History MLP.** Maps `h` to `T` logits.
12. **Age linear.** Maps `z` to `T` logits. Both arms.
13. **Bias.** Added per target.
14. **Optional logit clamp.** MIMIC/NCH adapters only, after `forward` returns.
15. **BCE-with-logits.** Mean over batch and targets.

---

## 9. Diagnostics that do not use the full gate

`_GateFacade.lambda_of(age)` and `DevelopmentalTemporalRetrieval.lambda_of(age)` without a `persistence_offset` evaluate `softplus(θ₀ + β z)` or `softplus(θ₀)`. They omit `r · v + b_r`. `baselines/synthetic/eval_dtr_mechanism_new.py` `surface_metrics` uses that global curve. It is a probe of the scalar developmental parameters, not a replay of `forward`.

The per-encounter decay the loss actually uses is in the forward cache key `"lambda"` (and `"theta_m"`), which includes content persistence. Ablations that call `zero_all_betas_` / `restore_betas_` change the live `β` and therefore the real gate. On the temporal-only arm those ablations are a no-op: `β` is already 0 and is not read.

---

## 10. Equations side by side

**`dtr_age_temporal_new`**

```
v_m = f_enc(C_m)
u_m = (q · W_k v_m) / √d
θ_m = θ₀ + b_r + r · v_m
λ_m(a*) = softplus(θ_m + β (a* − 9) / 9)
g_m = exp(−λ_m(a*) τ_m)
w_m = exp(min(u_m, 20)) g_m
h = Σ_m w_m v_m
logit = f_history(h) + W_a (a* − 9) / 9 + b_a + b
```

Gradient reaches `β` from the loss only along

`logit → f_history → h → w → g → λ → softplus → (β z)`.

**`dtr_temporal_only`**

```
λ_m = softplus(θ_m)
```

with every other line unchanged, `β` held at 0 and removed from the graph. Gradient from the loss still reaches `θ₀`, `b_r`, `r`, `q`, `W_k`, the encoder, `f_history`, and `f_age`. It does not reach `β`.
