# Model Implementation Guide — `model/` and `model_ablation/`

**Audience:** AI agents and collaborators who need an accurate map of the *current* TALE-EHR and age-conditioning code before proposing changes.

**Scope:** This document describes only `/model` and `/model_ablation`. Preprocessing, PIC/MIMIC pipelines, and the older `finetune/` package are mentioned only when they affect these packages.

**Last aligned to code:** 2026-07-22.

---

## 0. One-paragraph project claim

The novelty claim is **continuous developmental-age conditioning of TALE-EHR’s temporal kernel** \(w(\Delta t)\): patient age (in years) is mapped through a fixed Fourier feature embedding \(\phi(a)\), then a small MLP produces additive polynomial-coefficient deltas \(\Delta\alpha(a)\) so

\[
\alpha(a) = \alpha_{\text{base}} + \Delta\alpha(a),\qquad
w(\Delta t, a) = \sigma\!\big(\mathrm{poly}_{\alpha(a)}(\Delta t)\big).
\]

This is FiLM-style additive modulation of the temporal-decay polynomial (not ALiBi, not AdaLN, not a learned age lookup table).

There are **two code trees**:

| Tree | Role | Status |
|---|---|---|
| `model/` | Original research implementation (baseline + age-conditioned pretrain) | Provenance / earlier experiments; still runnable |
| `model_ablation/` | Self-contained four-arm ablation (vanilla / random_constant / additive / kernel) | **Preferred path for new age-conditioning work** |

`model_ablation/` imports **nothing** from `model/` or `finetune/`. Treat `model/` as frozen provenance unless you intentionally change the original path.

---

## 1. Shared conceptual stack (both trees)

### 1.1 Backbone: TALE-EHR (Yu et al. 2025, reimplemented)

Event sequence → frozen BGE code embeddings → time-aware attention → (optional) multi-scale temporal aggregation → predictors.

**Inputs (conceptually):**

- `code_indices` `[B, L]` — vocab indices into frozen BGE table `[N+2, 1024]` (PAD=0, UNK=1, real≥2)
- `timestamps_days` `[B, L]` — continuous days from first event
- `delta_t` `[B, L, L]` — pairwise `log1p(|t_j − t_k| / 7)` (week-scaled log lag)
- `attention_mask` `[B, L]` — True = real event
- Demographics / age — **layout differs between trees** (see §2 vs §3)

**Core modules:**

1. **Time-aware attention** (single head)
   - `q,k,v = MLP(code_emb)` → `[B,L,d_model]`
   - scores = `q kᵀ / √d`
   - inject temporal kernel into pre-softmax scores
   - causal + padding mask → softmax → `E = attn @ v` → `[B,L,d_model]`

2. **Multi-scale temporal aggregation**
   - relevance = `q_base · e`
   - weight by temporal kernel on lag-to-current
   - softmax over events → patient vector `h` `[B,d_model]`

3. **Heads**
   - Pretrain: multi-label next-code logits over vocab; optionally Weibull time-gap params (model/ only)
   - Fine-tune (ablation): binary classifier on last-event `[h_last ; demo_last]`

### 1.2 Temporal kernel injection modes

| Mode | Formula | Notes |
|---|---|---|
| `multiplicative` | `scores = scores * sigmoid(poly)` | Present in `model/` only. Diagnostics showed it anti-focuses when many QK logits are negative. |
| `additive_logspace` | `scores = scores + logsigmoid(poly)` | Default in `model/`; **locked** in `model_ablation/`. Preferred. |

### 1.3 Age feature embedding (both trees)

`FourierAgeEmbedding` (`age_embedding.py` in each tree):

- Input: age in **years**
- `num_frequencies=16` → embedding dim **32** (`sin` ∥ `cos`)
- Periods geometrically spaced from **1 month** (`1/12` y) to **200** y
- Frequencies are a **fixed buffer** (not learned)

`AgeCoefficientGenerator`:

- MLP: `Linear(32 → 64) → GELU → Linear(64 → poly_degree+1)`
- Final layer **zero-initialized** so \(\Delta\alpha = 0\) at init (starts identical to vanilla kernel)
- Modes: `real` | `random_constant` | `none` (semantics differ slightly between trees — see below)

### 1.4 Where age conditioning is applied

| Site | What age is used | What is modulated |
|---|---|---|
| Attention temporal weight | Per-event age (or constant) | Pairwise \(w(\Delta t_{jk}, a)\) |
| Aggregation temporal weight | Age at **last valid event** (current age) | Weights on lag-to-current |

**Critical fine-tune fact (ablation):** classification uses `return_repr_only=True`, which returns per-event `E` and **does not call aggregation**. So at fine-tune time, the **live age pathways are only inside attention** (plus the additive code-embedding path). Aggregation age params exist for pretrain parity but are gradient-dead under the fine-tune graph.

---

## 2. Package `model/` — original implementation

### 2.1 File map

| File | Purpose |
|---|---|
| `tale_ehr.py` | Vanilla `TALEEHR` backbone |
| `tale_ehr_age.py` | Age-conditioned `TALEEHRAge` |
| `time_aware_attention.py` | Vanilla polynomial temporal weight + attention + aggregation |
| `time_aware_attention_age.py` | Age-conditioned polynomial weight + attention + aggregation |
| `age_embedding.py` | `FourierAgeEmbedding`, `AgeCoefficientGenerator` |
| `age_diagnostics.py` | `‖Δα(a)‖` stats by developmental bucket |
| `dataset.py` | Parquet / tensorized pretrain datasets + collate |
| `train.py` | Pretrain loop for both variants |
| `debug_no_time_loss.py`, `debug_weibull.py` | One-off debug scripts |

### 2.2 Vanilla model (`TALEEHR`)

- Frozen BGE `embedding_table` buffer
- `TimeAwareAttention` + `MultiScaleTemporalAggregation`
- `demo_dim=3` by default: demographics channels are **`[age_years, sex, race]`**
- `demo_proj`: Linear(3→64)+GELU; `history_proj = Identity`
- Predictors on `[h ; demo_last]`:
  - `code_predictor` → `num_codes` logits (final bias init −7.0 for sparse positives)
  - `time_params_predictor` → 2 Weibull params `(k, λ)` via softplus in the loss
- `kernel_injection`: `"multiplicative"` or `"additive_logspace"` (default additive)

**Polynomial basis (vanilla / age in this tree):** raw **monomials** `[1, t, t², …, t⁵]` on `t = log1p(Δt/7)`. Degree 5 → 6 coefficients. Init: `α₀=0.5`, rest 0 → `sigmoid(0.5)≈0.62`.

### 2.3 Age-conditioned model (`TALEEHRAge`) — main focus in this tree

**Class:** `model/tale_ehr_age.py::TALEEHRAge`

**Architecture deltas vs vanilla:**

1. Replaces attention/aggregation with age-conditioned versions.
2. Extracts age as `age_years = demographics[..., 0]` (channel 0 of the 3-D demo tensor).
3. Passes `age_years` into both attention and aggregation.
4. Same predictor heads as vanilla (including Weibull head).
5. Trainable param delta vs baseline (smoke-tested): **+5004**.

**`AgeConditionedPolynomialWeight`:**

```
alpha = coefficients + AgeCoefficientGenerator(phi(age))
poly  = sum_k alpha[..., k] * delta_t**k     # monomial basis
w     = sigmoid(poly)
```

- Attention: `age_features = FourierAgeEmbedding(age_years)` with shape `[B,L,32]`; each query position uses its own age to form \(\Delta\alpha\), then poly is evaluated on the `[B,L,L]` lag matrix (broadcast on the last coeff dim).
- Aggregation: uses **current** (last-event) age only → `[B,32]` features.

**`age_conditioning_mode` (in `AgeCoefficientGenerator`):**

| Mode | Behavior |
|---|---|
| `real` | MLP sees real Fourier age features |
| `random_constant` | MLP input replaced by a fixed random buffer `random_constant` (seed 0), expanded to batch shape — capacity-matched, no real age |
| `none` | Returns exact zero \(\Delta\alpha\) (should match vanilla if shared base weights) |

**Important:** in this tree, `random_constant` replaces the **Fourier feature vector** with a random constant vector *inside* the generator. In `model_ablation/`, the control instead feeds a **constant age (7 years)** through the same Fourier path (architecturally identical to kernel). Do not mix these semantics when comparing results.

### 2.4 Pretrain (`model/train.py`)

**Entry flags of interest:**

```bash
python model/train.py \
  --model_variant {baseline,age_conditioned} \
  --age_conditioning_mode {real,random_constant,none} \
  --kernel_injection {multiplicative,additive_logspace} \
  --code_loss {bce,focal} \
  --gamma_loss <float> \          # weight on Weibull NLL
  --no_time_loss \                # optimize code loss only; still log Weibull
  --use_tensorized \
  ...
```

**Loss:**

- Code: BCE-with-logits (default) or focal; optional `pos_weight`
- Time: Weibull NLL on `target_time_gap`, scaled by `gamma_loss` (unless `--no_time_loss`)

**Metrics:** recall@{5,10,20}, optional AUROC.

**Age diagnostics during train:** `age_diagnostics.compute_alpha_delta_stats` every `--age_diag_every` steps when variant is age-conditioned.

### 2.5 Dataset / demographics leak (known issue)

In `model/dataset.py` collate:

- `demographics[:,:,0] = age_years`
- `demographics[:,:,1] = sex`
- `demographics[:,:,2] = race`

So age enters **twice** in `TALEEHRAge`:

1. Explicitly into the temporal kernel via `demographics[...,0]`
2. Implicitly into `demo_proj` / classifier via the same tensor

This **demographics leak** makes it hard to attribute gains to the kernel pathway alone. `model_ablation/` was built specifically to fix this (INV-demo).

### 2.6 Known limitations / diagnostic findings (agent should not re-discover blindly)

From `diagnostics/age_kernel_diagnosis.md` and project docs (summary):

1. Multiplicative injection is a bad operator on this data (anti-focusing).
2. Monomial basis on `log1p(Δt/7)∈[0,≈6.5]` is severely ill-conditioned (Gram cond ~1e9); Chebyshev on `[-1,1]` is ~34.
3. Early age MLP often learned a large **age-invariant offset** rather than true age-varying \(\Delta\alpha\).
4. MIMIC-IV pretrain is adult-dominated → little pediatric age diversity for conditioning to learn from.
5. Aggregation age path can be near gradient-dead under some training regimes.

**Implication:** Prefer reading/changing `model_ablation/` for new age work unless the task is explicitly about the legacy path.

---

## 3. Package `model_ablation/` — four-arm age ablation (preferred)

Self-contained. Single front-end flag `--arm`. Designed so that **only the age pathway differs** across arms; seed, data, optimizer schedule, and non-age architecture stay identical.

### 3.1 File map

| File | Purpose |
|---|---|
| `arms.py` | `--arm` → `ArmConfig` resolution |
| `age_embedding.py` | Fourier embed + kernel MLP + **additive** age MLP |
| `time_aware_attention_age.py` | Chebyshev temporal weight + attention + aggregation |
| `tale_ehr_age.py` | `TALEEHRAblation` (one class, arm-gated) |
| `tale_ehr.py` | Thin `TALEEHR` = ablation with `arm="vanilla"` (shared pretrain) |
| `dataset.py` | Pretrain tensorized dataset; **separate `age_years`** |
| `dataset_finetune.py` | Disease classification dataset/collate; same age split |
| `model_finetune.py` | `TALEEHRAblationClassifier` + legacy demo_proj adapter |
| `train.py` | Shared **vanilla** pretrain (code loss only) |
| `train_finetune.py` | Per-arm fine-tune (CHD / other binary tasks) |
| `verify_arms.py` | Structural invariants + gradient-liveness harness |
| `positive_control.py` | Gate: can forcing kernel shapes move logits? |
| `tensorize_pretrain.py` | Flat mmap shards for spawn-safe pretrain loaders |
| `run_pretrain.sh`, `run_finetune_ablation.sh`, `run_finetune_matrix.sh`, `schedule_finetune.sh` | Launch scripts |
| `README.md` | Short operator notes |

### 3.2 Four arms (the experiment)

Resolved by `arms.resolve_arm(arm) → ArmConfig`:

| `--arm` | Kernel \(\Delta\alpha\) | Additive embed \(\delta\) | Age into Fourier |
|---|---|---|---|
| `vanilla` | identically 0, **no params** | identically 0, **no params** | unused |
| `random_constant` | MLP(\(\phi(\texttt{const})\)) | 0, no params | **constant 7.0 years** |
| `additive` | 0, no params | MLP(\(\phi(\texttt{real age})\)) → added to code emb | real |
| `kernel` | MLP(\(\phi(\texttt{real age})\)) | 0, no params | real |

**Locked for every arm:**

- Kernel injection = `additive_logspace` only (`scores + logsigmoid(poly)`)
- No multiplicative path
- No QK-normalization
- No Weibull / time loss
- Temporal poly = **Chebyshev** degree 5 on  
  \(x = 2\cdot\log1p(\Delta t/7)/6.5 - 1\)  
  (`CHEB_TMAX = 6.5`)

**Design rationale (do not “refactor into strategy objects” without reason):** one class + config keeps invariants assertable in one place (`assert_arm_invariants`, `age_pathway_param_count`) and guarantees `random_constant` is architecturally identical to `kernel` (only the age *input* differs).

### 3.3 Demographics leak fix (INV-demo)

- `age_years`: separate tensor `[B, L]`
- `demographics`: `[B, L, 2]` = `(sex, race)` only
- `demo_dim == 2` hardcoded (`DEMO_DIM = 2`)
- Forward asserts:
  - `demographics.shape[-1] == 2`
  - `"age_years" in batch`
- Age-stratified eval in fine-tune reads `age_years`, not demographics

Legacy checkpoints with `demo_proj` input dim 3 are adapted at load by dropping column 0 (age) in `model_finetune._adapt_legacy_demo_proj`.

### 3.4 Age pathways in detail

#### A. Kernel pathway (arms: `kernel`, `random_constant`)

Modules:

- `FourierAgeEmbedding` inside attention (and aggregation for pretrain)
- `AgeCoefficientGenerator` with `mode ∈ {real, random_constant}` → builds MLP params
- For `vanilla` / `additive`: `mode="none"` → **MLP not constructed** (zero pathway params)

`ChebyshevPolynomialWeight._poly`:

```
alpha_delta = age_coeff_gen(age_features)   # [..., 6]
alpha = base_coefficients + alpha_delta
x = 2 * log_delta_t / 6.5 - 1
poly = sum_k alpha[..., k] * T_k(x)         # Chebyshev recurrence
```

Attention forward expects **precomputed** `age_features` (caller chooses real vs constant age). Aggregation similarly receives current-age features.

#### B. Additive pathway (arm: `additive` only)

`AdditiveAgeEmbedding`:

- Same MLP shape as kernel gen but `out_dim = 1024` (code embedding dim)
- Zero-init final layer
- Applied **before** `mlp_q/k/v`:

```
code_embeddings = embedding_table[indices] + additive_delta * mask
```

This tests whether age helps as a **content** bias on embeddings rather than as a **temporal kernel** modulator. Parameter count is intentionally comparable in spirit to the kernel MLP (capacity-matched ablation family), but dims differ (out 1024 vs 6).

### 3.5 Model class `TALEEHRAblation`

File: `tale_ehr_age.py`

Forward sketch:

1. Validate INV-demo.
2. Additive path: Fourier(real age) → MLP → add to code embeddings (zero if disabled).
3. Kernel path: Fourier(real or const age) → attention with Chebyshev+Δα.
4. If `return_repr_only`: return `{h_repr: E, demo_features: demo_proj(demographics)}` — **no aggregation, no code head**.
5. Else (pretrain): aggregate with current-age kernel features → `code_predictor` → `{code_logits, h}`.

**No `time_params_predictor`** in this package.

`TALEEHR` in `tale_ehr.py` is simply `TALEEHRAblation(arm="vanilla")` so the shared pretrain state_dict keys match every arm’s non-age parameters (`strict=False` load at fine-tune adds age keys fresh).

### 3.6 Fine-tune classifier

`TALEEHRAblationClassifier`:

1. Infer hparams from checkpoint (`num_codes`, `d_model`, `poly_degree`, `demo_hidden`).
2. Build `TALEEHRAblation(arm=...)`.
3. Adapt legacy demo_proj if needed; `load_state_dict(..., strict=False)`.
4. Assert missing/unexpected keys are only age modules or dropped heads.
5. Replace `code_predictor` with `Identity`.
6. Linear classifier: `Linear(d_model + demo_hidden → 1)`.

**Optimizer groups in `train_finetune.py`:** separate LRs:

- backbone (non-age): default `1e-5`
- classifier head: `1e-3`
- age-injection params (kernel and/or additive): `1e-3` (`--lr_age`)

**Eval:** BCE with pos_weight from class imbalance; AUROC + AUPRC; age-stratified bands  
`(<1, 1–5, 6–11, 12–17, 18–25)` years.

Early stop on val AUPRC (patience default 6).

### 3.7 Training protocol (intended experiment)

1. **Shared vanilla pretrain** once:
   ```bash
   python model_ablation/train.py --tensorized_dir data/processed/tensorized_flat
   ```
   Code BCE only; Chebyshev kernel; no age modules active.

2. **Verify arms** (params, invariants, gradient liveness):
   ```bash
   python model_ablation/verify_arms.py --pretrained_ckpt <ckpt> --tensorized_dir <task_dir>
   ```

3. **Positive control** (optional gate): force fast- vs slow-decay Chebyshev coeffs; require `max|Δlogit|` above threshold, else stop — kernel mechanism cannot matter.

4. **Fine-tune all four arms** from the **same** backbone, identical seed/hparams, only `--arm` changes:
   ```bash
   python model_ablation/train_finetune.py --arm {vanilla|random_constant|additive|kernel} \
     --pretrained_ckpt <shared> --tensorized_dir <CHD_or_other>
   ```

Interpretation guide for agents:

| Result pattern | Meaning |
|---|---|
| `kernel` ≫ `random_constant` ≈ `vanilla` | Real age in the kernel helps beyond capacity |
| `kernel` ≈ `random_constant` ≫ `vanilla` | Capacity / extra MLP helps; not real age |
| `additive` ≫ `kernel` | Age as content bias beats temporal modulation (or vice versa) |
| All arms ≈ equal | Age conditioning not load-bearing on this task/data |

### 3.8 Dataset contracts (ablation)

**Pretrain collate** (`dataset.py`): emits `age_years` + `demographics[sex,race]` + `target_codes` (no time-gap target).

**Fine-tune collate** (`dataset_finetune.py`): same age split; emits `labels` for binary disease classification.

Tensorized format: flat concatenated arrays + offsets (mmap-safe). Stale object-dtype shards should fail loudly.

### 3.9 Scripts

- `run_pretrain.sh` — vanilla shared pretrain
- `run_finetune_ablation.sh` — sequential 4-arm CHD run (paths point at PIC heart_malformations)
- `run_finetune_matrix.sh` / `schedule_finetune.sh` — multi-task / scheduled variants

---

## 4. Side-by-side: age conditioning differences agents must respect

| Topic | `model/` | `model_ablation/` |
|---|---|---|
| Preferred for new work? | No (legacy) | **Yes** |
| Age in demographics? | Yes (`demo[:,:,0]`) | **No** (separate `age_years`) |
| `demo_dim` | 3 | **2** |
| Poly basis | Monomials | **Chebyshev** |
| Kernel injection | Configurable (mult / add) | **Locked additive_logspace** |
| Age arms | modes on one model | **4 discrete `--arm`s** |
| Additive embed path | Absent | Present (`additive` arm) |
| `random_constant` meaning | Random Fourier-sized vector inside MLP | Constant **age years=7** through Fourier |
| Age module params when off | MLP still exists (`none` zeros output) | **Params omitted** when pathway unused |
| Time / Weibull loss | Yes (optional disable) | **Never** |
| Fine-tune graph | External `finetune/` package | Built-in `model_finetune.py` |
| Aggregation used at FT? | Depends on finetune wrapper | **No** (`return_repr_only`) |

---

## 5. Mathematical summary (age-conditioned kernel arm)

**Fourier features** (fixed):

\[
\phi(a) = \big[\sin(2\pi f_i a),\;\cos(2\pi f_i a)\big]_{i=1}^{16},\quad
f_i = 1/T_i,\quad T_i \in [1/12,\,200]\ \text{(log-spaced)}.
\]

**Coefficient generator** (trainable; last layer zero-init):

\[
\Delta\alpha(a) = \mathrm{MLP}(\phi(a)) \in \mathbb{R}^{6}.
\]

**Chebyshev temporal poly** (ablation):

\[
x = \frac{2\log(1+\Delta t/7)}{6.5}-1,\quad
\mathrm{poly}(x;a)=\sum_{k=0}^{5}(\alpha_k+\Delta\alpha_k(a))\,T_k(x).
\]

**Attention injection:**

\[
\mathrm{score}_{jk} \leftarrow \mathrm{score}_{jk} + \log\sigma(\mathrm{poly}_{jk}).
\]

**Additive arm alternative:**

\[
\tilde{e}_j = e_j + \mathrm{MLP}_{\mathrm{add}}(\phi(a_j)),\quad
\text{then standard QKV on }\tilde{e}.
\]

---

## 6. What an agent should check before changing code

1. **Which tree?** Default to `model_ablation/` for age-conditioning changes. Do not silently “port” monomial/`demo_dim=3` assumptions into the ablation package.
2. **Fine-tune vs pretrain graph:** if changing age pathways for classification metrics, gradients must hit **attention** (and/or additive embed), not aggregation alone.
3. **Preserve INV-demo:** never re-merge age into `demographics` in the ablation package.
4. **Preserve arm orthogonality:** `random_constant` must stay capacity-matched to `kernel`; `vanilla`/`additive` must keep \(\Delta\alpha\equiv0\) with no kernel-age params; `vanilla`/`kernel` must keep additive delta ≡0 with no additive params.
5. **Keep injection additive-logspace** unless there is a new diagnostic reason to reopen multiplicative.
6. **Data age support:** adult-only MIMIC cannot prove pediatric age conditioning; PIC / pediatric cohorts matter for the scientific claim.
7. **Run gates after structural edits:** `verify_arms.py`, then `positive_control.py`, then the 4-arm fine-tune matrix.

---

## 7. Likely change surfaces (for planning)

These are the natural places future work usually touches:

| Goal | Touch |
|---|---|
| Change Fourier banding / pediatric resolution | `*/age_embedding.py` `FourierAgeEmbedding` |
| Change how age enters the kernel | `AgeCoefficientGenerator`, `ChebyshevPolynomialWeight` / `AgeConditionedPolynomialWeight` |
| New ablation arm | `arms.py` + gates in `TALEEHRAblation` + verify harness |
| Fix / replace temporal basis | `time_aware_attention_age.py` (ablation) or `time_aware_attention*.py` (legacy) |
| Fine-tune task / metrics | `train_finetune.py`, `dataset_finetune.py` |
| Shared backbone training | `model_ablation/train.py` |
| Legacy pretrain experiments | `model/train.py` + `TALEEHRAge` |

---

## 8. Quick command cheat sheet

```bash
# Ablation: shared vanilla pretrain
conda run -n ehr python model_ablation/train.py \
  --tensorized_dir data/processed/tensorized_flat

# Ablation: verify four arms
conda run -n ehr python model_ablation/verify_arms.py \
  --pretrained_ckpt <shared_backbone.pt> \
  --tensorized_dir <tensorized_task_dir>

# Ablation: fine-tune one arm
conda run -n ehr python model_ablation/train_finetune.py \
  --arm kernel \
  --pretrained_ckpt <shared_backbone.pt> \
  --tensorized_dir <tensorized_task_dir>

# Legacy: age-conditioned pretrain
conda run -n ehr python model/train.py \
  --model_variant age_conditioned \
  --age_conditioning_mode real \
  --kernel_injection additive_logspace \
  --use_tensorized \
  --no_time_loss
```

---

## 9. Glossary

| Term | Meaning |
|---|---|
| TALE-EHR | Time-Aware Longitudinal EHR transformer-style model (paper reimplementation) |
| \(w(\Delta t)\) | Temporal decay / emphasis weight from a polynomial + sigmoid |
| \(\Delta\alpha(a)\) | Age-generated additive shift to polynomial coefficients |
| FiLM | Feature-wise linear modulation; here: additive coeff modulation |
| INV-demo | Invariant: age must not enter `demo_proj` |
| CHD | Congenital heart disease / heart_malformations fine-tune cohort |
| BGE | Frozen `BAAI/bge-m3` code-text embeddings |
| `return_repr_only` | Forward mode returning per-event `E` for classification (skips aggregation/code head) |
