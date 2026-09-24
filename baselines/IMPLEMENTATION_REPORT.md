# Baseline Implementation Fidelity Report

## Overview

This report documents the fidelity of each baseline implementation relative to the
original papers and official repositories. Each model is explicitly labeled as
**faithful**, **adapted**, or **task-matched adaptation**.

> **Important**: None of these runs are exact reproductions of the original papers.
> Vocabulary, objective, and sequence length differ from the published settings.
> All models are labeled `{Name}-task-matched` when trained on our prediction task.

---

## 1. Count + LightGBM

| Field | Value |
|-------|-------|
| **Paper citation** | N/A — strong non-neural baseline |
| **Official repository** | N/A |
| **Architecture** | One-vs-rest LightGBM with count features |
| **Input representation** | Binary code presence, code counts, demographics |
| **Age representation** | Explicit scalar feature (age at prediction time) |
| **Time representation** | None in primary mode |
| **Pretraining objective** | N/A |
| **Downstream objective** | Binary cross-entropy (one-vs-rest) |
| **Parameter count** | ~trees (not neural) |
| **Sequence limit** | Unlimited (bag-of-codes) |
| **Label** | **Non-neural control** |

---

## 2. RETAIN

| Field | Value |
|-------|-------|
| **Paper citation** | Choi et al., "RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism", NeurIPS 2016 |
| **Official repository** | mp2893/RETAIN |
| **Architecture** | Reverse-time GRU_α (scalar attention) + GRU_β (variable attention) |
| **Differences from paper** | Event-level micro-visits vs. encounter-level (matching benchmark format) |
| **Input representation** | Multi-hot code vectors per event |
| **Age representation** | None |
| **Time representation** | None (reverse-time order implicit in GRU) |
| **Pretraining objective** | N/A |
| **Downstream objective** | BCEWithLogitsLoss |
| **Parameter count** | ~66K (d_emb=128, d_rnn=128) |
| **Sequence limit** | Sequential (GRU — unlimited) |
| **Label** | **Faithful architecture, task-matched adaptation** |

---

## 3. Vanilla EHR-BERT

| Field | Value |
|-------|-------|
| **Paper citation** | N/A — architectural control |
| **Official repository** | N/A |
| **Architecture** | Standard BERT: code emb + sinusoidal position + segment + [CLS] → head |
| **Differences from paper** | N/A (control, not a specific published model) |
| **Input representation** | Flattened code tokens with positional/segment IDs |
| **Age representation** | **None** (architectural control) |
| **Time representation** | **None** (architectural control) |
| **Pretraining objective** | MLM (canonical mode) / BCEWithLogits (task-matched) |
| **Downstream objective** | BCEWithLogitsLoss |
| **Parameter count** | ~2.1M (d=256, 4 layers, 4 heads) |
| **Sequence limit** | 512 |
| **Label** | **Architectural control** |

---

## 4. BEHRT

| Field | Value |
|-------|-------|
| **Paper citation** | Li et al., "BEHRT: Transformer for Electronic Health Records", Scientific Reports 2020 |
| **Official repository** | deepmedicine/BEHRT |
| **Architecture** | BERT with code + **learnable age bucket** + position + visit segment embeddings |
| **Differences from paper** | Age buckets extended for pediatric range; vocabulary differs; no future-disease MLM |
| **Input representation** | Flattened diagnosis/code tokens with per-encounter age/position/segment |
| **Age representation** | **Learnable embedding per discretized age bracket** (NOT DTR's continuous function) |
| **Time representation** | None (positional order only) |
| **Pretraining objective** | BEHRT MLM (canonical) / BCEWithLogits (task-matched) |
| **Downstream objective** | BCEWithLogitsLoss |
| **Parameter count** | ~6.1M (d=288, 6 layers, 12 heads) |
| **Sequence limit** | 512 |
| **Label** | **Faithful architecture, task-matched adaptation** |

---

## 5. Med-BERT

| Field | Value |
|-------|-------|
| **Paper citation** | Rasmy et al., "Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction", npj Digital Medicine 2021 |
| **Official repository** | ZhiGroup/Med-BERT |
| **Architecture** | BERT with code + **visit index** + **within-visit serialization** embeddings |
| **Differences from paper** | Vocabulary differs; no prolonged-LOS auxiliary task in task-matched mode |
| **Input representation** | Flattened code tokens with visit/serialization indices |
| **Age representation** | **None** (canonical Med-BERT has no age) |
| **Time representation** | **None** (canonical Med-BERT has no time) |
| **Pretraining objective** | MLM + λ_LOS L_LOS (canonical) / BCEWithLogits (task-matched) |
| **Downstream objective** | BCEWithLogitsLoss |
| **Parameter count** | ~1.6M (d=192, 6 layers, 6 heads) |
| **Sequence limit** | 512 |
| **Label** | **Faithful architecture, task-matched adaptation** |

---

## 6. CEHR-BERT

| Field | Value |
|-------|-------|
| **Paper citation** | Pang et al., "CEHR-BERT: Incorporating temporal information from structured EHR data to improve prediction tasks", ML4H 2021 |
| **Official repository** | cumc-dbmi/cehrbert |
| **Architecture** | BERT with code + segment + **Time2Vec age** + **Time2Vec timestamp** → concat → project |
| **Differences from paper** | VTP omitted (MIMIC visit-type limitation); vocabulary differs |
| **Input representation** | Chronological codes with [VS]/[VE] boundaries and ATT tokens |
| **Age representation** | **Time2Vec** (continuous, sinusoidal) |
| **Time representation** | **Time2Vec** (continuous, sinusoidal) |
| **Pretraining objective** | MLM (existing checkpoint) / BCEWithLogits (task-matched) |
| **Downstream objective** | BCEWithLogitsLoss |
| **Parameter count** | ~0.8M (d=128, 5 layers, 8 heads) |
| **Sequence limit** | 300 |
| **Label** | **Faithful architecture, task-matched adaptation** |
| **Existing checkpoint** | Preserved as `CEHR-BERT-canonical` (MLM-pretrained) |

---

## 7. DTR (Developmental Temporal Retrieval)

| Field | Value |
|-------|-------|
| **Paper citation** | This work |
| **Official repository** | This repository |
| **Architecture** | Transformer with age-conditioned temporal kernel: λ(a) = softplus(θ₀ + β z(a)) |
| **Differences from paper** | Reference implementation |
| **Input representation** | Code tokens with timestamps |
| **Age representation** | **Continuous via developmental gate** (Fourier / softplus-parameterized) |
| **Time representation** | **Explicit τ = log1p(Δt/7) in attention bias** |
| **Pretraining objective** | BCEWithLogits (future-visit prediction) |
| **Downstream objective** | Same |
| **Parameter count** | Varies by arm and d_model |
| **Sequence limit** | Varies (synthetic: 96; MIMIC: 1024) |
| **Label** | **Reference implementation** |

---

## Comparison Summary

| Model | Age | Time | Architecture | d_model | Layers | Heads |
|-------|:---:|:----:|-------------|--------:|-------:|------:|
| Count+LightGBM | ✓ (scalar) | ✗ | Tree ensemble | — | — | — |
| RETAIN | ✗ | ✗ (order only) | Reverse-time GRU | 128 | 2 GRU | — |
| EHR-BERT | ✗ | ✗ | Standard BERT | 256 | 4 | 4 |
| BEHRT | ✓ (bucket) | ✗ | BERT + age emb | 288 | 6 | 12 |
| Med-BERT | ✗ | ✗ | BERT + visit/serial | 192 | 6 | 6 |
| CEHR-BERT | ✓ (Time2Vec) | ✓ (Time2Vec) | BERT + temporal | 128 | 5 | 8 |
| DTR | ✓ (continuous) | ✓ (kernel) | Age×temporal BERT | 256 | 1 | 4 |

---

## Scientific Integrity Notes

1. **No hidden advantages**: Models keep canonical architectures. No age injection into Med-BERT, no time injection into EHR-BERT.
2. **Poor results reported**: If a faithful implementation underperforms, the result is reported, not patched.
3. **No oracle access**: Baselines never receive ground-truth β, λ, mechanism type, or oracle surfaces.
4. **No test-set tuning**: Hyperparameters selected on validation only.
