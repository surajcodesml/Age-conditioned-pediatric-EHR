# Age-conditioned temporal kernel — diagnostic review

> Part 1 (below) reviews the **multiplicative** checkpoint. The **Addendum** at the bottom adds the
> **additive-logspace** checkpoint, reconciles "additive was worse," and gives the revised bottom line.
> If you only read one thing, read the Addendum's "Revised bottom line."

**Scope of Part 1:** multiplicative injection. Checkpoint under review:
`checkpoints/age_real_202605112156/epoch_012.pt` (`age_conditioned`, `age_conditioning_mode="real"`,
`gamma_loss=500`, `lr=1e-4`, 12 epochs). This checkpoint predates the `kernel_injection` flag
(`config.json` has no such key); the code it trained under hard-coded `scores = scores * w`
(confirmed at `git show 342a901:model/time_aware_attention.py`), i.e. it *is* the multiplicative variant.

**Reproduce:** `python diagnostics/age_kernel_diagnosis.py --sections ABCDE`
(script writes all numbers quoted below; run against `data/processed/tensorized/val`).

---

## TL;DR — what is limiting age-conditioning performance

Ranked by expected impact:

1. **Multiplicative injection is the wrong operator and is actively anti-focusing.** `scores * w`
   with `w∈(0,1)` *promotes* the ~47% of QK logits that are negative (multiplying a negative logit
   by a small `w` moves it toward 0, i.e. up). Measured consequence: pairs the kernel means to kill
   (`w<0.05`, 43% of keys/row) still absorb **28.8%** of attention mass; the additive-logspace kernel
   leaves them **0.4%**. Attention entropy *rises* to 4.32 nats vs 3.87 with no kernel at all — the
   kernel makes attention blurrier, not sharper. This also throttles the age-path gradient ~9× (§D).
   **This alone explains why the age signal can't do useful work.**

2. **Polynomial basis is catastrophically ill-conditioned.** Raw monomials `[1,t,…,t^5]` on the
   empirical `log1p(Δt/7)∈[0,6.5]` give a Gram condition number of **2.6e9**; a Chebyshev basis on
   rescaled `[-1,1]` gives **34** — eight orders of magnitude better. This is why coefficients wander,
   why the curve saturates to `w=0` past ~90 d, and why `Δα` has to reach norm ~5 to move the output.

3. **The FiLM generator learned an age-*invariant* global reshaping, not age conditioning.**
   `‖Δα(a)‖≈5.33` but 95% of it (`‖offset‖=5.32`) is a constant offset identical for every age;
   the age-*varying* part is only `‖·‖≈1.25`. So the "age" MLP is mostly a second, redundant set of
   base coefficients. Behaviorally (§C): replacing every patient's age with a single constant 30 y
   **matches real-age recall** (R@10 0.1214 vs 0.1207); turning the whole temporal kernel off costs
   only ~3% relative (0.1170). The age path is not load-bearing.

4. **Fourier age embedding is grossly over-resolved (aliasing).** Highest frequency has a 1-month
   period, so a 7-day age change moves the embedding by `‖Δφ‖=1.69` out of `‖φ‖=4.0` (42%). The model
   sees spurious high-frequency age structure that is noise for this task.

5. **Data has essentially no age diversity in the target range.** `age_at_event` mean = 63.3 y,
   min = 16.8 y; the training log shows **0** neonate/infant/child samples and only 49 adolescent
   hits vs 1340+ for each adult bucket. Despite the repo name ("pediatric"), this checkpoint trained
   on an all-adult population — there is little pediatric age signal for the module to learn from.

The aggregation-level age path (`temporal_aggregation`) is separately **dead**: its `age_coeff_gen`
gets grad/param ratio 5e-5 (§D) and its `w` is saturated near 1 across the useful range.

---

## Evidence

### A. Checkpoint statics (`section_a`)

```
attn base coeffs: [0.6923, -0.327, -0.0263, 0.0607, 0.0274, -0.0701]
agg  base coeffs: [1.0666, 0.069, 0.1706, 0.1347, 0.0564, -0.0385]
||Dalpha(a)|| over a∈[0,100): mean=5.329  (||base||=0.772 → ratio 6.90)
Dalpha per-coef mean (age-invariant): [1.677, -4.812, -0.201, 1.174, 0.818, -0.494]  ->  ||offset||=5.319
Dalpha per-coef std over ages (age-varying): [0.53, 1.264, 0.157, 0.442, 0.261, 0.175]  ->  mean ||varying||=1.247
Fourier: ||phi(a+7d)-phi(a)||=1.686 ; ||phi(a+10y)-phi(a)||=5.025  (||phi||=4.000)
```

`w(Δt, age)` at the attention level collapses to 0 beyond ~30–90 days for every age, and the age
ordering is non-monotonic (e.g. 7 d: age 1→0.49, age 18→0.25, age 50→0.36). The kernel's dynamic
range across ages (~0.24 at 7 d) is small relative to the QK spread it multiplies.

### B. Real-batch attention forensics (`section_b`, ~14.8 M causal pairs)

```
raw QK scores q·k/sqrt(d): mean=2.258 std=6.155  q01/25/50/75/99=[-5.9,-3.1,0.59,6.98,16.9]  frac<0=0.470
kernel w:                  mean=0.364 std=0.410   frac<0.05=0.474  frac>0.95=0.003
per-row logit range: raw=9.92  after *w = 8.21  (ratio 0.85)   # kernel barely reshapes ranking
attention entropy (nats): multiplicative=4.316  additive-counterfactual=2.951  no-kernel=3.867
suppressed pairs (w<0.05): 43.0% of keys/row; mass they still receive:
        multiplicative=28.8%   additive-counterfactual=0.4%
```

The 28.8% vs 0.4% row is the core failure: multiplicative injection cannot suppress, because ~half the
logits are negative and `x·w → 0⁻` *raises* them above genuinely-relevant strongly-negative pairs.

### C. Inference ablations (`section_c`, 40 val batches, real data)

| condition | recall@5 | recall@10 | recall@20 | code BCE |
|---|---|---|---|---|
| real ages | 0.0648 | **0.1207** | **0.2184** | 0.0106 |
| const age = 63 y (data mean) | 0.0598 | 0.1136 | 0.2074 | 0.0108 |
| const age = 30 y | 0.0618 | **0.1214** | 0.2172 | 0.0111 |
| const age = 5 y | 0.0535 | 0.1047 | 0.1964 | 0.0133 |
| ages shuffled across batch | 0.0544 | 0.1055 | 0.1984 | 0.0119 |
| temporal kernel OFF (`w≡1`) | 0.0615 | 0.1170 | 0.2111 | 0.0115 |

Real, const-30, const-63 and kernel-off all cluster within ~6% relative. A single constant age (30 y)
**reproduces real-age recall** → the learned age map extracts almost no patient-specific value. Only
grossly-wrong ages (5 y, shuffled — both push samples out of the adult training distribution) degrade it.

### D. Gradient flow into the age path (`section_d`, one batch, γ=500)

```
                     multiplicative           additive_logspace
attn.age_coeff_gen   ||g||=0.44 ratio 6.6e-2  ||g||=3.95 ratio 5.9e-1   # ~9x more signal
attn.base_coeffs     ratio 1.3e-1             ratio 1.08
agg.age_coeff_gen    ratio 5.3e-5             ratio 1.9e-4              # dead either way
code_predictor       ||g||=10.9               ||g||=8.8                # dominates the objective
```

Multiplicative injection delivers ~9× weaker gradient to the attention age generator than additive;
the aggregation age generator is starved regardless (~1e-4).

### E. Polynomial conditioning (`section_e`)

```
empirical log1p(Δt/7): min/25/50/75/95/99/max = 0.0/0.10/1.81/4.10/5.73/6.23/6.49
Gram cond, monomial basis (raw log-dt): 2.62e9
Gram cond, Chebyshev basis on [-1,1]:   3.42e1
```

---

## Design-choice verdicts

| component (call) | verdict | why |
|---|---|---|
| **Multiplicative injection** (`AgeConditionedTimeAwareAttention.forward`, `scores * w`) | **Replace** | Cannot suppress negative logits; anti-focusing (§B), 9× weaker age gradient (§D). Switch to additive-logspace `scores + logsigmoid(poly)` (already implemented). |
| **Monomial `PolynomialWeight._poly`** (raw `t^k`) | **Reparameterize** | Gram cond 2.6e9 (§E). Use Chebyshev on `x=2·log1p(Δt/7)/t_max−1`, or replace with a small monotone/bump kernel. Cuts conditioning to ~34. |
| **poly_degree = 5** | **Reduce to 2–3** | The fitted curve is effectively "high near 0, →0 by ~90 d"; degrees 4–5 only add ill-conditioned wobble (base coeffs c4,c5 ≈ ±0.03–0.07, noise-level). |
| **Coefficient additivity** `alpha = base + Δα` | **Keep, but bound Δα** | Additive is fine; the problem is Δα is unbounded and grew to 6.9× the base, 95% of it an age-invariant offset (§A). Bound/regularize Δα and zero-center it so `base` stays the "mean-age" curve. |
| **FiLM generator** (`AgeCoefficientGenerator`, zero-init, hidden 64) | **Keep zero-init; add reg** | Zero-init is correct (starts at baseline). Hidden 64 is not the bottleneck. Add weight decay / an L2 penalty on Δα and consider FiLM (scale+shift) on a *fixed* well-conditioned basis rather than free coefficient deltas. |
| **Fourier age embedding** (`FourierAgeEmbedding`, 16 freq, period 1 mo–200 y) | **Re-band** | 1-month min period aliases (§A: 42% embedding move per 7 d). Raise min period to ~1–2 y for adult data (or make it learnable); for a genuinely pediatric run keep months but only with real pediatric samples. |
| **`log1p(Δt/7)` scaling** | **Keep** | Reasonable; matches collator. Just feed it through a conditioned basis (Chebyshev needs the known `t_max≈6.5`). |
| **Aggregation age path** (`AgeConditionedMultiScaleTemporalAggregation`) | **Fix or drop** | Gradient-starved (§D, 5e-5) and `w` saturated ≈1. Same multiplicative bug on `relevance * w`. Fix with additive injection or remove age conditioning here until the attention path works. |

---

## Prioritized recommendations (ranked, with expected effect)

1. **Switch injection to `additive_logspace` and retrain.** *(largest, cheapest)*
   `scores + logsigmoid(poly)` is already in the code — just set `--kernel_injection additive_logspace`.
   Expected: suppressed-pair leakage 28.8% → ~0.4%, attention entropy 4.3 → 3.0 nats, ~9× stronger
   age-path gradient. This is the precondition for the age signal to matter at all.

2. **Reparameterize the polynomial to a conditioned basis and drop degree to 2–3.**
   Chebyshev on `x=2·log1p(Δt/7)/6.5−1`. Expected: Gram cond 2.6e9 → ~30, stable coefficient learning,
   removes the "saturate to 0 past 90 d" pathology and the need for `‖Δα‖≈5`.

3. **Regularize / bound `Δα` and zero-center it.** Add L2 on `Δα` (or `tanh`-bound per coefficient)
   so `base` remains the mean-age curve and `Δα` is a genuine age *modulation*. Expected: the age-varying
   fraction rises from ~23% of `‖Δα‖` toward ~100%; const-age ablation should then diverge from real-age
   (currently identical, §C) — a direct, measurable success criterion.

4. **Re-band the Fourier embedding** (min period ~1–2 y) or make frequencies learnable. Expected:
   removes sub-year aliasing; smoother, more sample-efficient age response.

5. **Fix or disable the aggregation-level age path.** Apply the same additive injection; if it stays
   grad-starved after (1)–(2), drop age conditioning there and keep it only in `time_aware_attention`.

6. **Confirm the data actually spans the ages you condition on.** This checkpoint saw ~0 pediatric
   patients (mean 63 y, min 16.8 y). No architecture change recovers a signal that isn't in the data —
   either source pediatric records or restate the objective as adult age-conditioning. Validate with a
   held-out per-age-bucket recall breakdown, not just aggregate recall.

**Suggested validation loop after (1)–(3):** rerun `--sections BCD`. Success looks like: suppressed-pair
mass < 2% (B), real-age recall clearly above const-age and kernel-off (C), and attention `age_coeff_gen`
grad/param ratio ≳ 0.3 (D).

---

# Addendum — additive-logspace checkpoint, and reconciling "additive was worse"

**Added:** review of `checkpoints/age_additive_code_only_full_20260623/epoch_010.pt`
(`additive_logspace`, code-only objective `no_time_loss=true`, γ=1, 10 epochs, cuda), run through
the *same* script/sections/batches as the multiplicative review.

**Reproduce:**
`python diagnostics/age_kernel_diagnosis.py --sections ABCDE --ckpt_dir checkpoints/age_additive_code_only_full_20260623 --ckpt_epoch 10 --injection additive_logspace`

## Direct answers to your three questions

**Q1. Additive performed worse — why?** *It didn't, at matched budget.* The comparison you ran is
confounded three ways: epochs (add 10 vs mult 12), objective (add = code-only γ=1; mult = `weibull +
500·code` with the broken Weibull **in** the backward pass), and device (cuda vs cpu). Line them up at
**matched epoch 10, from each run's own val log:**

| | recall@5 | recall@10 | recall@20 |
|---|---|---|---|
| multiplicative, epoch 10 | 0.0642 | 0.1179 | 0.2173 |
| additive, epoch 10 | 0.0640 | **0.1183** | **0.2180** |
| multiplicative, epoch 12 *(what you compared against)* | 0.0682 | 0.1237 | 0.2243 |

Additive is a dead heat — marginally *ahead* on R@10/R@20. The 0.1237 you were comparing to is
multiplicative's **two extra epochs**, which additive never ran. There is no additive regression to explain.

**Q2. My "additive → ~9× stronger age gradient" claim — reconcile.** That number was measured on the
*multiplicative* checkpoint's weights, whose QK logits are negative-heavy (47% < 0). In that regime the
two injection ops really do differ ~9×. On the *additive* checkpoint's own weights the QK geometry is
completely different — **all-positive, mean +8.3, 0% negative** (§B below) — and the injection-op gap
collapses to ~1.15× (attn.age_coeff_gen ‖g‖ 0.285 additive vs 0.249 multiplicative, §D). So the 9× was a
property of that weight regime, not a law; I over-generalized it. What *did* hold mechanically: additive
attention is sharper (entropy 2.9 vs 4.3–5.7 nats) and suppresses much better. It just didn't buy recall.

**Q3. Was the broken Weibull the cause?** *It was a real contaminant of the multiplicative run, but not
the cause of the age failure.* Gradient decomposition on the additive checkpoint (§D):

| objective term | attn.age_coeff_gen ‖g‖ | time_params_predictor ‖g‖ | code_predictor ‖g‖ |
|---|---|---|---|
| code-only (γ=1) | **4.0e-4** | 0 | 2.3e-2 |
| weibull-only | **2.3e-1** | 1.06e+3 | 0 |
| weibull + 500·code | 2.85e-1 | 1.06e+3 | 1.16e+1 |

The broken Weibull dumps ~**570×** more gradient into the age generator than the code task does, plus a
huge 1.06e3 into the shared `time_params_predictor` trunk. In the multiplicative run the objective was
`loss_time + 500·loss_code` with `loss_time≈5.9` and `500·loss_code≈5.5` — i.e. the broken Weibull was
~**half the objective and the dominant driver of the temporal/age parameters**. That is why `‖Δα‖`
ballooned to ~6 there: it was fitting a broken time distribution, not codes. **But** removing Weibull
entirely (the additive code-only run) did **not** raise recall, because the code task on its own barely
trains the age path (4e-4). So: fix/remove the Weibull loss for cleanliness and interpretability — it is
genuinely corrupting the temporal trunk — but do not expect it to unlock age conditioning.

## The unifying finding: QK swamps the bounded kernel (both modes)

Your original hypothesis was right. The temporal kernel is not decisively controlling attention in
*either* injection mode, because the QK dot-product is unbounded and large:

- **Multiplicative ckpt** (§B): QK mean 2.26, std 6.16, 47% negative; `scores*w` promotes negatives →
  w<0.05 pairs keep **28.8%** of attention mass; entropy 4.3 > no-kernel 3.9 (anti-focusing).
- **Additive ckpt** (§B): training pushed QK **all-positive**, mean 8.29, per-row range ~9–15. The
  additive penalty `logsigmoid(poly)` saturates at only ~−4 to −6 nats, too small to override that
  spread, so even correct-sign suppression leaks: w<0.05 pairs (83% of keys) still hold **60%** of mass
  (vs 83% base-rate — so it *is* suppressing, just weakly). The model literally inflated QK to route
  around its own kernel; at the attention level, w for age ≥ 50 is ~0 even at Δt=0d (the adult kernel
  turned itself off and leans entirely on QK).

So neither run lets the kernel's ±5 nats move the ranking against a 9–15-nat QK spread. **This is the
actual lever:** normalize/bound the QK path (unit-norm q,k / cosine attention / a learned temperature)
so the temporal-age kernel is on comparable scale. Until then, age and temporal decay are near-free
parameters w.r.t. recall.

## Ablation shapes are identical across both checkpoints (§C, same 40 batches)

| condition | mult R@10 | mult R@20 | add R@10 | add R@20 |
|---|---|---|---|---|
| real ages | 0.1207 | 0.2184 | 0.1124 | 0.2104 |
| const age = 30y | 0.1214 | 0.2172 | 0.1137 | 0.2102 |
| temporal kernel OFF (w≡1) | 0.1170 | 0.2111 | 0.1134 | 0.2076 |
| ages shuffled | 0.1055 | 0.1984 | 0.1031 | 0.1961 |

(Absolute add<mult here is the epoch-10-vs-12 sampling gap; the *pattern* is what matters.) In **both**
checkpoints, a single constant age matches real ages, and turning the kernel fully off matches or beats
real. Only grossly out-of-distribution ages (5y, shuffled) hurt. The age path and the temporal kernel
are not load-bearing for next-visit code recall — consistent regardless of injection mode.

## Revised bottom line

1. Additive did **not** underperform; it tied multiplicative at matched epochs and is the mechanically
   correct operator. Keep additive.
2. The thing blocking *both* is that the **unbounded QK swamps the bounded kernel** — fix QK scale
   first (highest-value change now; supersedes the old "switch to additive" as the #1 item since you've
   already switched).
3. **Remove/repair the Weibull loss** regardless — in the multiplicative run it was ~half the objective
   and the dominant gradient into the temporal/age trunk (‖g‖ 1.06e3 into `time_params_predictor`,
   570× the code task into the age generator). It corrupted the multiplicative checkpoint's temporal
   params; it is not the reason age doesn't help, but it makes every temporal diagnostic untrustworthy.
4. Then the earlier structural fixes still stand: **Chebyshev basis + lower degree** (Gram cond
   2.6e9 → 34), **bound/zero-center Δα**, **re-band the Fourier embedding**, and above all **get age
   diversity** — this population is all-adult (mean 63y, ~0 pediatric), so there is little age signal to
   condition on no matter how clean the mechanism.
5. Success criterion unchanged and now sharper: after fixing QK scale + removing Weibull, real-age
   recall should *separate* from const-age and kernel-off in §C. Today it does not, in either checkpoint.
