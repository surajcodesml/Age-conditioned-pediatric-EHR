# Result provenance — Content-Persistence DTR (`*_new`)

Primary synthetic comparison: **32-target S2** protocol (8 interaction + 6 temporal_only + 6 age_only + 6 content_only + 6 null).
The 8-interaction-only evaluation is secondary/appendix and is not used here as the main table.

- Data seed: `20260922`
- Model seed: `0`
- Data root: `synthetic_age_temporal/outputs/data/seed20260922/controlled/{S}`
- Architecture: Content-Persistence DTR (mass-preserving / raw-additive)
- Arms: `dtr_age_temporal_new`, `dtr_temporal_only_new`

## Per-run table

| Scenario | Arm | Checkpoint | Seed | n_targets | splits (tr/va/te) | params | AUPRC | AUROC | CF-age | CF-lag | Surf | β̂ | θ̂₀ |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | dtr_age_temporal_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_age_temporal_new/S0/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55107 | 0.526 | 0.730 | 0.099 | 0.120 | 0.139 | 0.002 | -0.059 |
| S0 | dtr_temporal_only_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_temporal_only_new/S0/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55106 | 0.530 | 0.733 | 0.084 | 0.115 | 0.150 | 0.000 | -0.058 |
| S1 | dtr_age_temporal_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_age_temporal_new/S1/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55107 | 0.539 | 0.742 | 0.133 | 0.120 | 0.170 | 0.027 | -0.079 |
| S1 | dtr_temporal_only_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_temporal_only_new/S1/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55106 | 0.537 | 0.740 | 0.141 | 0.121 | 0.169 | 0.000 | -0.093 |
| S2 | dtr_age_temporal_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_age_temporal_new/S2/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55107 | 0.577 | 0.761 | 0.142 | 0.166 | 0.201 | -0.615 | -0.088 |
| S2 | dtr_temporal_only_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_temporal_only_new/S2/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55106 | 0.569 | 0.758 | 0.177 | 0.184 | 0.236 | 0.000 | -0.109 |
| S3 | dtr_age_temporal_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_age_temporal_new/S3/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55107 | 0.529 | 0.732 | 0.132 | 0.091 | 0.146 | 0.588 | -0.105 |
| S3 | dtr_temporal_only_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_temporal_only_new/S3/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55106 | 0.522 | 0.730 | 0.132 | 0.089 | 0.147 | 0.000 | -0.100 |
| S5 | dtr_age_temporal_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_age_temporal_new/S5/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55107 | 0.587 | 0.772 | 0.254 | 0.219 | 0.183 | -0.480 | -0.074 |
| S5 | dtr_temporal_only_new | `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_temporal_only_new/S5/best_checkpoint.pt` | 0 | 32 | 4906/1070/1031 | 55106 | 0.583 | 0.768 | 0.281 | 0.251 | 0.219 | 0.000 | -0.078 |

## Evaluation artifacts

For each arm/scenario:
- `best_checkpoint.pt` / `checkpoint.pt` / `last_checkpoint.pt`
- `result.json` (train history + test metrics + mechanism fields)
- `cf_report.json` (CF-RMSE-age/lag, Surface RMSE, mechanism class)
- `history.json`

Aggregates:
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_new_mechanism_summary.json`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/dtr_new_results_table.csv`
- `/home/suraj/Git/Age-conditioned-pediatric-EHR/results/baselines/synthetic/cf_summary_S*_new.json`

## Primary S2 table verification (3 d.p.)

- **DTR**: MATCH — artifact AUPRC/AUROC/CF-age/CF-lag/Surf = 0.577/0.761/0.142/0.166/0.201
- **Temporal-only DTR**: MATCH — artifact AUPRC/AUROC/CF-age/CF-lag/Surf = 0.569/0.758/0.177/0.184/0.236
- **Count+LightGBM**: MATCH — artifact AUPRC/AUROC/CF-age/CF-lag/Surf = 0.520/0.729/0.193/0.220/0.207
- **RETAIN**: MATCH — artifact AUPRC/AUROC/CF-age/CF-lag/Surf = 0.435/0.675/0.223/0.193/0.236
- **BEHRT**: MATCH — artifact AUPRC/AUROC/CF-age/CF-lag/Surf = 0.508/0.723/0.189/0.186/0.217
- **CEHR-BERT**: MATCH — artifact AUPRC/AUROC/CF-age/CF-lag/Surf = 0.582/0.764/0.098/0.111/0.131

## Discrepancies (not silently reconciled)

- results/baselines/synthetic/all_results_new.json has null predictive/CF metrics for all scenarios; authoritative numbers live in per-arm result.json / cf_report.json / dtr_new_mechanism_summary.json / dtr_new_results_table.csv.
- S2 dtr_age_temporal_new has Surface_RMSE=0.201 (partial threshold=0.25) but mechanism_classification=NO_MECHANISM_RECOVERY because the classifier also requires S0 CF-RMSE-age ≤ 0.05; observed S0 CF-RMSE-age for this arm is 0.0986.
- S0 itself is still labeled PARTIAL_RECOVERY from Surface RMSE alone: `classify_mechanism` only applies the S0 CF-age gate when classifying *other* scenarios (S1–S3), not when classifying S0.
- No per-run config.json under synthetic dtr_*_new/; training used baselines.synthetic.runner defaults (seed=0, data_seed=20260922, batch_size=32, max_epochs=25, Content-Persistence DTR d_model=64).
- No per-patient prediction tensors or age×lag surface arrays are saved under dtr_*_new/; only scalar CF metrics in cf_report.json / result.json. Surfaces in figures/final/ are regenerated by forward pass from best_checkpoint.pt on the same held-out CF template protocol.
- Legacy Transformer baseline dirs (`dtr_age_temporal` without `_new`) store CF metrics only in `result.json`; β̂ / shuffle / λ_corr for the old Transformer live in `synthetic_age_temporal/outputs/runs/controlled/arch_S2_age_temporal_d20260922_m0_interonly/metrics.json` (used for `parameter_vs_functional_recovery`).

## Notes

- Patient splits are example-level controlled splits from `splits.json` (same sizes across S0–S5 for seed 20260922): {'train': 4906, 'val': 1070, 'test': 1031}.
- Frozen text/code embeddings: none for DTR (code embeddings are trainable).
- NCH leakage-corrected matched `_new` pair is incomplete at provenance time (see `nch_final_analysis.md`).
