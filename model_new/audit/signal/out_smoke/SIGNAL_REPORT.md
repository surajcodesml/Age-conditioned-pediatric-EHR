# SIGNAL REPORT

Do not soften the verdict. Missing tests are marked explicitly.

Present: D9, D7, D1, D4, D8.1, D8.2, D2/D5
Missing: (none)

## Routing table

| Test | Number | Threshold | Route |
|---|---|---|---|
| D9 persistence recall@10 | 0.08315 | ≈ arms (0.134–0.138) | baselines below arms → backbone has headroom over recurrence |
| D9 co-occurrence recall@10 | 0.0005805 | ≈ arms | below arms |
| D7 half-life vs age | CI overlap=False | CIs overlap / separate | signal exists, model failing → multi-head + TTE |
| D1 constant Δrecall@10 (vanilla) | 0.00127 | <1% / >5% | objective timing-blind → build TTE |
| D1 jitter onset | 365 | ≥ ±365d | kernel over-parameterised → drop s to 2–3 |
| D8 recall vs gap | flat=False | flat / steep | horizon already used → TTE unnecessary |
| D4 positive loss mass | 0.7195 | <5% | positive mass adequate for full softmax |
| D2 Δα epoch correlation | 0.9602 | <0.3 | Δα trajectory stable across epochs 3→8 |

## D9 baselines

| Baseline | recall@5 | recall@10 | recall@20 |
|---|---|---|---|
| persistence | 0.04058 | 0.08315 | 0.1603 |
| cooccurrence | 0.0004068 | 0.0005805 | 0.001912 |
| global_prior | 0.05334 | 0.1052 | 0.1812 |

Hand-check match: True

## D7 half-life by age band

| Band | median h (days) | CI lo | CI hi | n |
|---|---|---|---|---|
| 18-35 | 469.7 | 415.5 | 474 | 20 |
| 35-50 | 421.1 | 364.6 | 424.3 | 20 |
| 50-65 | 375.3 | 351.8 | 382.6 | 20 |
| 65-80 | 343.3 | 334.3 | 358.5 | 20 |
| 80+ | 383.4 | 327.2 | 450.2 | 20 |

Monotonicity slope=-25.05 perm_p=0

## D1 timestamp conditions

Vanilla constant Δr@10: 0.00127 [-0.0003019, 0.004237]
Kernel constant Δr@10: 0.0007085 [-0.0011, 0.003511]
Assertions: {"determinism_true_vs_true_repeat": true, "constant_tau_max_zero": true, "shuffle_preserves_multiset": true, "kernel_constant_degrades_at_least_vanilla": true}

| Arm | jitter days | Δr@10 | CI |
|---|---|---|---|
| vanilla | jitter_1 | 0.001072 | [-0.00103, 0.003916] |
| vanilla | jitter_7 | 0.0004612 | [-0.0006944, 0.002228] |
| vanilla | jitter_30 | 0.002584 | [-6.584e-05, 0.006835] |
| vanilla | jitter_180 | 0.004332 | [0.001006, 0.008829] |
| vanilla | jitter_365 | 0.003058 | [0.0002905, 0.007899] |
| kernel | jitter_1 | 0.001069 | [-0.001823, 0.004551] |
| kernel | jitter_7 | 0.0009868 | [-0.001638, 0.004904] |
| kernel | jitter_30 | -0.0003518 | [-0.002701, 0.003071] |
| kernel | jitter_180 | -0.0001928 | [-0.001956, 0.002666] |
| kernel | jitter_365 | -0.00182 | [-0.002948, -0.0006175] |

## D4 loss mass

Positive mass = 0.7195; mean positives/example = 113.8
Grad split: {"head_grad_norm_from_pos": 12850.124938964844, "head_grad_norm_from_neg": 10418.373260498047, "pos_over_neg": 1.2334099208834208}

## D8.1 gap histogram

Signed median=22.82 IQR=[-1.1933965682983398, 205.39309692382812] p10/p90=-920/583.4; frac_negative=0.2852

## D8.2 recall vs gap

| Gap bin | n | recall@10 |
|---|---|---|
| <0 | 73 | 0.1407 |
| 0-1d | 20 | 0.1814 |
| 1-7d | 16 | 0.08952 |
| 7-30d | 29 | 0.1165 |
| 30-90d | 30 | 0.09569 |
| 90-365d | 49 | 0.1184 |
| >365d | 39 | 0.1206 |

## D2 / D5 from train.json

- encoder_layer0: mean consecutive r=0.9589 unidentified=False
- pooling: mean consecutive r=0.9615 unidentified=False

### Selection tables

| Arm | BCE-best ep | r@10 | r10-best ep | r@10 |
|---|---|---|---|---|
| vanilla | 8 | 0.1381 | 9 | 0.1405 |
| random_constant | 8 | 0.1365 | 9 | 0.139 |
| additive | 6 | 0.134 | 10 | 0.1356 |
| kernel | 6 | 0.1339 | 8 | 0.1399 |

Declared primary_endpoint: metric=AUPRC, dataset=PIC heart_malformations, comparison=kernel vs random_constant is the identifying comparison (exactly parameter-matched); vanilla is the floor; additive is an alternative delivery site with a different architecture

## Figures

- `d1_jitter_curve.png`: present
- `d7_halflife_vs_age.png`: present
- `d8_recall_vs_gap.png`: present

## Next action

Objective is timing-blind (|Δrecall@10| under constant-τ < 1%). Build a horizon-conditioned / TTE objective before touching the age mechanism.
