# Key findings

Numbers below are copied from saved test metrics and counterfactual reports.
Shared-protocol models predict all 32 targets. Content-Persistence numbers are interaction-label only and are not differences against RETAIN, BEHRT, or CEHR-BERT.

- On the shared protocol, S2 AUPRC best is Previous DTR (age×temporal) at 0.584; second is Previous DTR (temporal-only) at 0.582.
- Previous DTR (age×temporal) S2 AUPRC is 0.584 versus Previous DTR (temporal-only) 0.582 (delta +0.0010).
- S2 Surface RMSE lowest is CEHR-BERT at 0.1308; second is Count + LightGBM at 0.2071. Previous DTR (age×temporal) is 0.2266 (NO_MECHANISM_RECOVERY).
- S3 AUPRC best is CEHR-BERT at 0.534; second is Previous DTR (age×temporal) at 0.532. Previous DTR (age×temporal) AUPRC is 0.532, Surface RMSE 0.1757 (NO_MECHANISM_RECOVERY).
- S5 AUPRC best is CEHR-BERT at 0.601; second is Previous DTR (temporal-only) at 0.598. Previous DTR (age×temporal) mean persistence Surface RMSE is 0.2676; persistence order correct is False.
- Saved beta for Previous DTR (age×temporal): S2 -0.1284 (true -2.5, sign match True); S3 0.0417 (true +2.5, sign match True). Both magnitudes stay near 0 rather than near 2.5.
- Content-Persistence DTR, interaction labels only: S2 AUPRC 0.857 versus temporal-only 0.831 (delta +0.0259); beta_hat -2.3585; gate Surface RMSE 0.1251.
- Content-Persistence DTR S3 beta_hat 2.0651 (sign match True); AUPRC delta versus temporal-only +0.0361.
- Content-Persistence DTR S0/S1 beta_hat 0.0086 / 0.0751. beta=0 ΔBCE 0.000007 / 0.000300.
- S5 Content-Persistence AUPRC 0.817 versus Global-Persistence 0.812 (delta +0.0045). Learned offsets 1.109 / 0.196 / -0.535. Order acute > intermediate > chronic is True.
- S0 false-interaction flag (CF-RMSE age > 0.05) is true for: Previous DTR (age×temporal), Previous DTR (temporal-only), Previous DTR (age-only), Previous DTR (no age), Count + LightGBM, RETAIN, EHR-BERT, Med-BERT, BEHRT, CEHR-BERT.
- No model on the shared protocol has mechanism class FUNCTIONAL_RECOVERY on S2 or S3. Every saved class is NO_MECHANISM_RECOVERY.
- Every shared-protocol S5 run has persistence_order_correct = false.
- Patient-level bootstrap intervals were not computed: bootstrap unavailable from saved artifacts.
- No multi-seed replication is in the saved runs (seed 0 only).
