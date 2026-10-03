# Figure Manifest — DTR ICLR Paper

All figures are saved as PNG (`figures/png/`) and SVG (`figures/svg/`).
Scripts live under `figures/scripts/`. Regenerate with:

```bash
python3 figures/scripts/generate_all_figures.py
```

No numbers were invented; all values come from checked-in experiment artifacts.

---

## Main paper

### `synthetic_mechanism_recovery` (Figure 2)

| | |
|---|---|
| **Files** | `figures/png/synthetic_mechanism_recovery.png`, `figures/svg/synthetic_mechanism_recovery.svg` |
| **Script** | `figures/scripts/synthetic_mechanism_recovery.py` |
| **Shows** | (A) Oracle S2 age×lag relevance \(R(a,\tau)\); (B) Final encounter-level DTR recovered surface; (C) Original Transformer recovered surface (parametric non-factorized baseline); (D) age-shuffle / \(\beta{=}0\) \(\Delta\)BCE across S0–S3 |
| **Data sources** | `synthetic_age_temporal/outputs/runs/dtr/controlled_S2_aggcmp_dtr_age_temporal_raw_additive_m0/metrics.json` (DTR recovery); `synthetic_age_temporal/outputs/runs/controlled/arch_S2_age_temporal_d20260922_m0_interonly/metrics.json` (Transformer); `synthetic_age_temporal/results/final_validation/controlled_metrics.csv` (Panel D); `synthetic_age_temporal/results/final_validation/bootstrap_results.json` (S2 patient-bootstrap SD); surfaces reconstructed via `synthetic_age_temporal/config.py` `relevance` / `tau_from_days` |

### `nch_developmental_behavior` (Figure 3)

| | |
|---|---|
| **Files** | `figures/png/nch_developmental_behavior.png`, `figures/svg/nch_developmental_behavior.svg` |
| **Script** | `figures/scripts/nch_developmental_behavior.py` |
| **Shows** | (A) NCH DTR learned \(K(a,\tau)=-\lambda(a)\tau\) and row-normalized \(e^{K}\); (B) micro-AUPRC by age bin (DTR vs Temporal-only) with bootstrap CIs; (C) Recall@5 by history truncation horizon |
| **Data sources** | `figures/results/raw/nch_age_lag_kernel.npz`; `figures/results/nch_subgroup_performance.csv`; `artifacts/nch_stage2/analysis_age_temporal/tables/adkm_performance_by_truncation.csv`; `artifacts/nch_stage2/analysis_age_temporal/tables/model_delta_by_horizon.csv` |

---

## Appendix

### `synthetic_lambda_curves` (Figure A1)

| | |
|---|---|
| **Files** | `figures/png/synthetic_lambda_curves.png`, `figures/svg/synthetic_lambda_curves.svg` |
| **Script** | `figures/scripts/synthetic_lambda_curves.py` |
| **Shows** | True vs learned \(\lambda(a)\) for S0–S3 (final encounter-level DTR, raw_additive) |
| **Data sources** | `synthetic_age_temporal/outputs/runs/dtr/controlled_S{0,1,3}_dtr_age_temporal_raw_additive_m0/metrics.json`; S2: `.../controlled_S2_aggcmp_dtr_age_temporal_raw_additive_m0/metrics.json` |

### `synthetic_architecture_ladder` (Figure A2)

| | |
|---|---|
| **Files** | `figures/png/synthetic_architecture_ladder.png`, `figures/svg/synthetic_architecture_ladder.svg` |
| **Script** | `figures/scripts/synthetic_architecture_ladder.py` |
| **Shows** | Multi-metric ladder: Original Transformer → M1 → M2 → M3 → Final DTR (\(\Delta\)AUROC/AUPRC, shuffle/\(\beta{=}0\) \(\Delta\)BCE, Surface RMSE, corr(\(\lambda\))) |
| **Data sources** | `synthetic_age_temporal/results/final_validation/architecture_comparison.csv` |

### `additive_vs_softmax` (Figure A3)

| | |
|---|---|
| **Files** | `figures/png/additive_vs_softmax.png`, `figures/svg/additive_vs_softmax.svg` |
| **Script** | `figures/scripts/additive_vs_softmax.py` |
| **Shows** | Grouped bars: Surface RMSE, shuffle/\(\beta{=}0\) \(\Delta\)BCE, AUPRC, AUROC, corr(\(\lambda\)) for additive vs softmax (M1) |
| **Data sources** | `synthetic_age_temporal/results/followup/M1_results.json` |

### `age_decoding_probe` (Figure A4)

| | |
|---|---|
| **Files** | `figures/png/age_decoding_probe.png`, `figures/svg/age_decoding_probe.svg` |
| **Script** | `figures/scripts/age_decoding_probe.py` |
| **Shows** | Age-probe \(R^2\) and age-band accuracy from frozen representations (no-age / temporal-only / DTR × pre-pool / pooled / head-mean) |
| **Data sources** | `synthetic_age_temporal/results/followup/followup_age_probe.json` |

### `background_content_ablation` (Figure A5)

| | |
|---|---|
| **Files** | `figures/png/background_content_ablation.png`, `figures/svg/background_content_ablation.svg` |
| **Script** | `figures/scripts/background_content_ablation.py` |
| **Shows** | Full vs signal-only S2 histories: \(\Delta\)AUPRC/AUROC (DTR − temporal-only) and mechanism \(\Delta\)BCE |
| **Data sources** | `synthetic_age_temporal/results/followup/background_ablation_results.json` |

### `synthetic_multiseed_summary` (Figure A6)

| | |
|---|---|
| **Files** | `figures/png/synthetic_multiseed_summary.png`, `figures/svg/synthetic_multiseed_summary.svg` |
| **Script** | `figures/scripts/synthetic_multiseed_summary.py` |
| **Shows** | \(\hat\beta\), shuffle/\(\beta{=}0\) \(\Delta\)BCE, Surface RMSE across S0–S3 (seed 0); S2 error bars from patient bootstrap SD |
| **Data sources** | `synthetic_age_temporal/results/final_validation/controlled_metrics.csv`; `synthetic_age_temporal/results/final_validation/bootstrap_results.json` |

### `mimic_learning_curves` (Figure A7)

| | |
|---|---|
| **Files** | `figures/png/mimic_learning_curves.png`, `figures/svg/mimic_learning_curves.svg` |
| **Script** | `figures/scripts/mimic_learning_curves.py` |
| **Shows** | Stage-1 MIMIC train/val BCE and val micro-AUPRC for DTR vs Temporal-only |
| **Data sources** | `stage1_mimic_pretrain/run/adkm_s0/history.json`; `stage1_mimic_pretrain/run/nint_s0/history.json` |

### `pic_results_by_age` (Figure A8)

| | |
|---|---|
| **Files** | `figures/png/pic_results_by_age.png`, `figures/svg/pic_results_by_age.svg` |
| **Script** | `figures/scripts/pic_results_by_age.py` |
| **Shows** | PIC age-stratified AUPRC (age-kernel vs vanilla) remapped to paper bins `<1, 1–5, 6–11, 12–17` for four tasks |
| **Data sources** | `results/pic/age_stratified/{mortality,pneumonia,los_gt7,heart_malformations}.csv` |

### `persistence_distribution` (Figure A9)

| | |
|---|---|
| **Files** | `figures/png/persistence_distribution.png`, `figures/svg/persistence_distribution.svg` |
| **Script** | `figures/scripts/persistence_distribution.py` |
| **Shows** | Synthetic S5 learned persistence offsets \(\theta_m\) for acute / intermediate / chronic |
| **Data sources** | `results/paper_synthetic/dtr_mechanism_summary.json` → `S5.learned_persistence_offset` |

### `umap_embeddings` (Figure A10)

| | |
|---|---|
| **Files** | `figures/png/umap_embeddings.png`, `figures/svg/umap_embeddings.svg` |
| **Script** | `figures/scripts/umap_embeddings.py` |
| **Shows** | PIC representation embeddings (Vanilla vs Age-kernel) colored by age band; PCA-2 fallback when `umap-learn` is unavailable |
| **Data sources** | `outputs/umap/umap_embeddings.npz` (`H__Vanilla`, `H__Kernel_age`, `age_band`) |

---

## Shared helpers

| File | Role |
|---|---|
| `figures/scripts/_paper_style.py` | Shared matplotlib style, colors, panel labels, PNG/SVG save |
| `figures/scripts/generate_all_figures.py` | Runs all figure scripts |
