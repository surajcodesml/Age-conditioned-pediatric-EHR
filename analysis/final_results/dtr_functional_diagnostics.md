# DTR functional mechanism diagnostics (`*_new`)

Held-out patients: controlled test split (n=1031, data_seed=20260922).
Stored ΔBCE_* in result.json were computed on **validation** (see `baselines/synthetic/eval_dtr_mechanism_new.py`); this report also recomputes ΔBCE and mean |Δlogit| on the **test** set.

| Scenario | Arm | β̂ | θ̂₀ | AUPRC | Surf | CF-age | shuffleΔBCE(val/test) | β=0ΔBCE(val/test) | |Δlogit| shuffle | |Δlogit| β=0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | dtr_age_temporal_new | 0.0019 | -0.0590 | 0.526 | 0.139 | 0.099 | 0.0078/0.0071 | 0.0000/-0.0000 | 0.2264 | 0.0006 |
| S0 | dtr_temporal_only_new | 0.0000 | -0.0582 | 0.530 | 0.150 | 0.084 | —/— | —/— | — | — |
| S1 | dtr_age_temporal_new | 0.0265 | -0.0794 | 0.539 | 0.170 | 0.133 | 0.0100/0.0095 | 0.0000/0.0001 | 0.2646 | 0.0084 |
| S1 | dtr_temporal_only_new | 0.0000 | -0.0929 | 0.537 | 0.169 | 0.141 | —/— | —/— | — | — |
| S2 | dtr_age_temporal_new | -0.6151 | -0.0882 | 0.577 | 0.201 | 0.142 | 0.0201/0.0204 | 0.0069/0.0071 | 0.3621 | 0.2101 |
| S2 | dtr_temporal_only_new | 0.0000 | -0.1093 | 0.569 | 0.236 | 0.177 | —/— | —/— | — | — |
| S3 | dtr_age_temporal_new | 0.5880 | -0.1052 | 0.529 | 0.146 | 0.132 | 0.0176/0.0195 | 0.0059/0.0062 | 0.3341 | 0.1765 |
| S3 | dtr_temporal_only_new | 0.0000 | -0.1000 | 0.522 | 0.147 | 0.132 | —/— | —/— | — | — |
| S5 | dtr_age_temporal_new | -0.4796 | -0.0737 | 0.587 | 0.183 | 0.254 | 0.0215/0.0212 | 0.0045/0.0045 | 0.3585 | 0.1595 |
| S5 | dtr_temporal_only_new | 0.0000 | -0.0781 | 0.583 | 0.219 | 0.281 | —/— | —/— | — | — |

## S0 CF-RMSE-age investigation

- (e) **Ruled out as β-pathway false interaction:** β̂≈0.0019, β_true=0, and β=0 ΔBCE≈0. The gate interaction pathway is inactive.
- (a) **Direct age main-effect head:** Oracle age-std on *interaction* targets across CF ages is **0.0**, but the model still has age-std **0.042** on those targets. Zeroing `f_age` *reduces* interaction CF-RMSE-age 0.094→0.065 (while *worsening* age_only 0.087→0.160), so a large share of interaction-target S0 CF error is the additive age head, not β.
- (c) **All-target averaging:** The reported metric averages over 32 targets (8 interaction + 6 age_only + …). Age_only targets have genuine oracle age effects (oracle age-std 0.115).
- (d) **Metric definition:** CF-RMSE-age compares full probability vectors under age counterfactuals; it does not isolate the age×lag gate.
- (b) **Implicit age-in-content:** Secondary. After zeroing `f_age`, residual all-target CF-RMSE-age remains 0.127 (content/history mismatch vs oracle can remain).

### Quantitative decomposition

```json
{
  "beta_hat": 0.0018794552888721228,
  "age_head_weight_l2": 2.573582649230957,
  "oracle_age_std_across_cf_ages": {
    "all": 0.021517653791179286,
    "interaction": 0.0,
    "age_only": 0.11476082021962286
  },
  "model_age_std_across_cf_ages": {
    "all": 0.03968491405248642,
    "interaction": 0.04248993098735809,
    "age_only": 0.07026609778404236
  },
  "cf_rmse_age_by_mechanism_subset": {
    "all": 0.09861166967821862,
    "interaction": 0.09411122084673092,
    "age_only": 0.0869864983717533
  },
  "cf_rmse_age_with_age_head_zeroed": {
    "all": 0.12726191137294182,
    "interaction": 0.06501454318170101,
    "age_only": 0.15961997185059476
  }
}
```

**Conclusion:** Nontrivial S0 CF-RMSE-age is **not** false β-interaction sensitivity. It is driven by (a) the shared age main-effect head (including on interaction targets where the oracle is age-invariant) plus (c)/(d) all-target CF metric aggregation over age_only and other mechanisms. The official all-target CF-RMSE-age number is left unchanged.