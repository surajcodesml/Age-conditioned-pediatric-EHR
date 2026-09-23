# NCH Stage-2 age×temporal analysis — FINAL

**Status:** complete. Both arms finished (early stop epoch 7). Primary comparison uses `checkpoint_best_auprc.pt` for each arm (both epoch 4).  
**Held-out:** `artifacts/nch_stage2/v2/tensorized_forecast/diagnoses_only/test` — 28 147 windows, 510 patients.  
**Root:** `artifacts/nch_stage2/analysis_age_temporal/`

---

## 1. Verified model contract

\[
s_{ij}^{(h)}=\frac{q_i^{(h)\top}k_j^{(h)}}{\sqrt{d_h}}-\bigl[\lambda_0+\beta\,z_P(a_i)\bigr]\tau_{ij}
\]

| Item | Value |
|---|---|
| \(z_P(a)\) | \((a-9)/9\) (fixed pediatric; adult \(\mu/\sigma\) not reused) |
| \(\tau\) | \(\log(1+\|\Delta t\|/7)\), \(\Delta t\) in days |
| \(K(a,\Delta t)\) | \(-\lambda(a)\,\tau\); relevance \(\mathrm{e}^{K}\) |
| \(\lambda_0,\beta\) | **scalars**, shared across all heads; not in pooling |
| Conditioning age \(a_i\) | **per-query event age** (years) |
| Sign | \(+\lambda\) = recency; \(-\lambda\) = long-range (sign-test verified) |
| age_temporal arm | \(\beta\) trainable (init 0) |
| no_interaction arm | \(\beta\) frozen at 0 |

### Parameters

| Source | \(\lambda_0\) | \(\beta\) |
|---|---:|---:|
| Stage-1 adult | −1.617 | −0.083 |
| Stage-2 init | −1.617 | 0 |
| ADKM primary (ep 4 best AUPRC) | −1.023 | **+0.092** |
| NINT primary (ep 4 best AUPRC) | −1.070 | **0** |
| ADKM final (ep 7) | −1.034 | +0.126 |
| NINT final (ep 7) | −1.108 | 0 |

---

## 2. What the age-conditioned model learned (A–C)

- \(\lambda(a)\) stays **negative** on 0–18y (−1.12 → −0.93): long-range prior retained.
- With \(\beta>0\), \(|\lambda|\) **decreases with age** (less long-range tilt in adolescents vs infants).
- Pediatric FT moves \(\lambda_0\) from −1.62 → ≈−1.02 and learns nonzero \(\beta\); largest \(\Delta K\) vs init at older ages × long lags (\(\Delta K_\min\approx-4.3\) at 18y / 10y lag).
- Figures: `fig_lambda_by_age`, `fig_temporal_curves_selected_ages`, `fig_age_lag_*`, `fig_pediatric_shift_*`.

---

## 3. Predictive behavior (D–F)

### Overall (ADKM, patient bootstrap)

| Metric | Point | 95% CI |
|---|---:|---|
| Recall@5 | 0.391 | [0.362, 0.419] |
| Recall@20 | 0.567 | [0.537, 0.591] |
| Precision@5 | 0.174 | [0.160, 0.188] |
| BCE | 4.44×10⁻⁴ | — |

### By developmental age (Recall@5)

| Band | N pts | Recall@5 |
|---|---:|---:|
| \<1 | 264 | 0.350 |
| 1–5 | 407 | **0.446** |
| 6–11 | 283 | 0.361 |
| 12–17 | 128 | **0.268** |

### Controlled history truncation

Recall@5: 30d 0.368 → 90d 0.384 → **180d 0.392 ≈ full 0.391**. Useful history plateaus by ~180 days; little gain from multi-year context on this task.

Natural available-history and age×horizon heatmaps: `fig_performance_by_*`, `fig_performance_age_history*`.

---

## 4. Does the model use the interaction? (G)

Inference-only counterfactuals on ADKM (Recall@5):

| Perturbation | Δ vs natural |
|---|---:|
| Set \(\beta\leftarrow0\) | **+0.0004** (≈ null) |
| Permute conditioning ages | −0.0012 |
| Constant age \(a=9\) | −0.0103 |

**Consensus:** \(\beta\) is nonzero but **functionally weak** for ranking. Training’s own age test also reported “nonzero β but negligible ΔL_shuffle.”

---

## 5. Arm comparison — FINAL (J)

Same held-out examples; patient-level paired bootstrap; primary ckpts = each arm’s `checkpoint_best_auprc` (epoch 4).

### Validation selection (for context)

| Arm | best val micro-AUPRC | best val BCE |
|---|---:|---:|
| age_temporal (ADKM) | **0.171** (ep 4) | 4.38×10⁻⁴ (ep 3) |
| no_interaction (NINT) | 0.166 (ep 4) | 4.39×10⁻⁴ (ep 3) |

ADKM wins on **val AUPRC**; arms nearly tied on val BCE.

### Held-out overall Δ = ADKM − NINT

| Metric | ADKM | NINT | Δ | 95% CI | CI excludes 0? |
|---|---:|---:|---:|---|---|
| Recall@5 | 0.3915 | 0.3937 | −0.0022 | [−0.0054, +0.0010] | No |
| Recall@10 | 0.4764 | 0.4754 | +0.0010 | [−0.0020, +0.0042] | No |
| Recall@20 | 0.5667 | 0.5643 | +0.0024 | [−0.0011, +0.0055] | No |
| Precision@5 | 0.1739 | 0.1743 | −0.0004 | [−0.0018, +0.0010] | No |
| BCE | 4.438e−4 | 4.440e−4 | −2.8e−7 | spans 0 | No |
| Brier | ~equal | ~equal | ~0 | spans 0 | No |

### By age (Recall@5 Δ)

| Band | Δ | 95% CI | excludes 0? |
|---|---:|---|---|
| \<1 | +0.0040 | [−0.0006, +0.0084] | No |
| 1–5 | −0.0036 | [−0.0093, +0.0019] | No |
| 6–11 | −0.0024 | [−0.0065, +0.0030] | No |
| 12–17 | −0.0029 | [−0.0073, +0.0016] | No |

Secondary note: 12–17 **Recall@20** Δ = +0.0086 [+0.0001, +0.0162] (excludes 0) — small N (128 pts); treat as exploratory, not a primary claim.

### By truncation horizon (Recall@5 Δ)

| Horizon | Δ | 95% CI | excludes 0? |
|---|---:|---|---|
| 30d | +0.0001 | [−0.0022, +0.0024] | No |
| 90d | +0.0021 | [+0.0002, +0.0041] | **Yes (tiny)** |
| 180d | +0.0009 | [−0.0013, +0.0031] | No |
| 1y | −0.0013 | [−0.0032, +0.0010] | No |
| 3y | +0.0009 | [−0.0018, +0.0033] | No |
| full | −0.0022 | [−0.0054, +0.0011] | No |

Age×horizon improvement heatmap: `fig_model_delta_age_history` — no large coherent positive block.

---

## 6. Consensus

1. **Learned mechanism is real but mild.** Stage-2 learns \(\beta\approx0.09\) and an age-sloped \(\lambda(a)\) that stays long-range-preferring; pediatric FT clearly moves the adult temporal prior.
2. **Functional use of age×temporal interaction for ranking is weak.** Disabling \(\beta\) at inference barely changes Recall@5; training diagnostics agree.
3. **Held-out predictive performance of the two arms is essentially equivalent.** Across Recall@K, Precision@5, BCE, and Brier, paired CIs include zero. Point estimates flip sign depending on metric (NINT slightly higher Recall@5; ADKM slightly higher Recall@20).
4. **No robust age- or horizon-specific advantage** for the interaction arm on the primary metric. The 90d horizon and adolescent Recall@20 signals are small, isolated, and not enough to claim a meaningful interaction benefit.
5. **Task geometry:** most useful longitudinal signal is within ~180 days; absolute performance varies by developmental age (best 1–5y, worst 12–17) for **both** arms’ operating regime, not specifically because of \(\beta\).
6. **Paper-facing takeaway:** age-conditioning successfully adapts the temporal kernel in parameter space, but on this NCH diagnoses-only next-encounter task and seed-0 run, it does **not** deliver a supported held-out ranking gain over a no-interaction control with the same backbone and \(\lambda_0\) adaptation. Report parameter/geometry findings (A–C, E2) as positive scientific content; report arm Δ as a null/near-null result with uncertainty.

---

## 7. Artifact index

- Figures (PNG@300 + PDF): `figures/fig_*`
- Tables: `tables/*.csv` (incl. `model_delta_overall_metrics.csv`, `model_delta_by_age_metrics.csv`, `model_delta_by_horizon.csv`)
- Predictions: `predictions/adkm_*.parquet`, `predictions/nint_*.parquet`
- Contract / params: `raw/age_temporal_contract.json`, `raw/temporal_params.json`, `raw/arm_selection_summary.json`
