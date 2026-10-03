# Missing / Incomplete Data Report — DTR Paper Figures

Figures that **were generated** despite gaps are noted with workarounds. Items that require new experiments (not just a lightweight analysis script) are marked **requires experiment rerun**.

---

## 1. Multi-seed synthetic replications (affects Figure A6, Figure 2 Panel D error bars)

| Missing | Detail |
|---|---|
| **Expected** | Multi-seed aggregates for \(\hat\beta\), Surface RMSE, age-shuffle \(\Delta\)BCE, \(\beta{=}0\) \(\Delta\)BCE across S0–S3 (and S5) |
| **Found** | Seed `0` only. Pipeline was run with `--skip-multiseed` (documented in `synthetic_age_temporal/report.md` / `results/paper_synthetic/key_findings.md`) |
| **Partial substitute** | `synthetic_age_temporal/results/final_validation/bootstrap_results.json` — **patient bootstrap** CIs on one seed for controlled/full S2 functional metrics only (not \(\hat\beta\) / Surface RMSE; not S0/S1/S3) |
| **Figure status** | **A6 generated** with single-seed points; S2 functional metrics show bootstrap SD. **Fig 2D** same |
| **To complete properly** | **Requires experiment rerun** with multiple model seeds |

---

## 2. CEHR-BERT (and other black-box) age×lag gate surfaces (Figure 2 Panel C)

| Missing | Detail |
|---|---|
| **Expected** | 2D recovered relevance / CF prediction surface for strongest non-DTR baseline (CEHR-BERT has lowest S2 CF Surface RMSE = 0.1308 in `results/paper_synthetic/table_mechanism.md`) |
| **Found** | `results/baselines/synthetic/cehrbert/S2/cf_report.json` — **scalars only** (`Surface_RMSE`, `CF_RMSE_age/lag`); no saved 2D grid |
| **Used instead** | Original Transformer parametric surface from `arch_S2_age_temporal_.../metrics.json` (reconstructible \(\hat\theta_0,\hat\beta\); Surface RMSE = 0.214; fails functional recovery) |
| **Figure status** | **Generated** with label “Best baseline: Original Transformer” |
| **To show CEHR-BERT surface** | Lightweight analysis possible if CF prediction grids are recomputed from the saved CEHR-BERT checkpoint / predictions; **no retraining** if checkpoint + eval code exist. Raw 2D arrays are not currently on disk |

---

## 3. NCH content-dependent persistence \(\theta_m\) (Figure 3 Panel A; Figure A9)

| Missing | Detail |
|---|---|
| **Expected** | Low / median / high persistence heatmaps from content-dependent \(\theta_m\); clinical persistence histogram with code-type annotations |
| **Found** | NCH/MIMIC DTR checkpoints expose only `temporal.lambda0`, `temporal.beta`, `temporal.age_mean`, `temporal.age_sd` — **no \(\theta_m\)** |
| **Used instead** | Fig 3A: single-model \(K(a,\tau)\) + row-normalized \(e^{K}\) from `nch_age_lag_kernel.npz`. Fig A9: **synthetic S5** offsets from `dtr_mechanism_summary.json` |
| **Figure status** | **Generated** with caveat |
| **To complete for clinical models** | **Requires architecture/experiment change** (content-persistence head on NCH) |

---

## 4. NCH history horizon = 730 days (Figure 3 Panel C)

| Missing | Detail |
|---|---|
| **Expected** | Horizons 30, 90, 180, 365, **730**, full |
| **Found** | Truncation table has `30d, 90d, 180d, 1y(365), 3y(1095), full` — **no 730d** |
| **Figure status** | **Generated** with available horizons |
| **To add 730d** | Lightweight analysis: re-run truncation eval from saved predictions/checkpoints (no retraining) |

---

## 5. PIC age bins and arm naming (Figure A8)

| Missing | Detail |
|---|---|
| **Expected** | Paper bins `<1, 1–5, 6–11, 12–17` with DTR vs Temporal-only / baseline + CIs |
| **Found** | PIC bands are neonate/infant/toddler/preschool/school/adolescent; arms are **age-kernel vs vanilla** (not NCH DTR vs temporal-only). AUPRC CIs not stored per cell (AUROC CIs exist) |
| **Used instead** | Remapped bins by pooling bands; AUPRC point estimates; sparse adolescent cells flagged (mortality n_pos=4, pneumonia n_pos=3) |
| **Figure status** | **Generated** with remapping caveat |
| **To match draft exactly** | Lightweight remapping already done; true DTR-vs-control PIC eval would need a new evaluation protocol if those arms were never trained on PIC |

---

## 6. Shared-protocol Previous DTR functional ablations

| Missing | Detail |
|---|---|
| **Expected** | age-shuffle / \(\beta{=}0\) \(\Delta\)BCE for `results/baselines/synthetic/dtr_*` shared-protocol runs |
| **Found** | Not saved (`SYNTHETIC_PAPER_RESULTS.md` caveat). Mechanism-recovery path uses Content-Persistence / final encounter DTR under `synthetic_age_temporal/` |
| **Impact** | Did not block figures; Fig 2D / A6 use final-validation controlled DTR metrics |
| **To add** | Lightweight analysis from checkpoints if ablation hooks exist; else **experiment rerun** |

---

## 7. Persisted 2D surface tensors

| Missing | Detail |
|---|---|
| **Expected** | Saved `.npy`/`.npz` oracle/learned surface grids |
| **Found** | Only \(\beta,\theta_0\) and \(\lambda(a)\) curves; surfaces reconstructed at plot time (same as `plots_final.fig_d_surface`) |
| **Impact** | None for figure generation |

---

## 8. UMAP-learn (Figure A10)

| Missing | Detail |
|---|---|
| **Expected** | UMAP 2D coordinates |
| **Found** | High-dim embeddings in `outputs/umap/umap_embeddings.npz`; pre-rendered `outputs/umap/umap_by_age_band.{png,svg}`; `umap-learn` not installed in this environment |
| **Used instead** | Standardized PCA-2 exploratory plot (labeled as such) |
| **To get true UMAP** | `pip install umap-learn` and re-run `figures/scripts/umap_embeddings.py`, or copy pre-rendered files from `outputs/umap/` |

---

## Summary

| Figure | Status | Blocker severity |
|---|---|---|
| Fig 2 `synthetic_mechanism_recovery` | Generated | Medium (Panel C uses Transformer, not CEHR-BERT surface) |
| Fig 3 `nch_developmental_behavior` | Generated | Low (no \(\theta_m\) mini-heatmaps; no 730d) |
| A1 `synthetic_lambda_curves` | Generated | None |
| A2 `synthetic_architecture_ladder` | Generated | None |
| A3 `additive_vs_softmax` | Generated | None |
| A4 `age_decoding_probe` | Generated | None |
| A5 `background_content_ablation` | Generated | None |
| A6 `synthetic_multiseed_summary` | Generated (single-seed + S2 bootstrap) | High for true multi-seed claim |
| A7 `mimic_learning_curves` | Generated | None |
| A8 `pic_results_by_age` | Generated (remapped / sparse) | Medium |
| A9 `persistence_distribution` | Generated (synthetic S5 only) | High for clinical \(\theta_m\) |
| A10 `umap_embeddings` | Generated (PCA fallback) | Low |
