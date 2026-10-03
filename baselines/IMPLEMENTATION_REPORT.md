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

## 8. MOTOR

| Field | Value |
|-------|-------|
| **Paper citation** | Steinberg et al., "Language models are realistic autoregressive models of patient histories", ICLR 2024; FEMR / MOTOR framework |
| **Official repository** | `som-shahlab/motor_code_release` / FEMR |
| **Architecture** | Pre-RMSNorm Transformer with continuous time-to-event Rotary Position Embedding (RoPE) and linear age projection |
| **Differences from paper** | Official implementation is written in Haiku/JAX with specialized CUDA kernels. Faithfully ported the mathematical continuous RoPE formulation, Pre-RMSNorm, and survival hazard formulation to native PyTorch; task-matched to benchmark multilabel BCE prediction and continuous counterfactuals |
| **Input representation** | Tokenized event sequence with continuous inter-event timestamp offsets (days/hours) and patient age |
| **Age representation** | **Explicit scalar projection** (`Linear(1, d_model)`) added to input embeddings |
| **Time representation** | **Continuous time-to-event RoPE**: $\theta_i = t / b^{2i/d}$, rotating query/key heads according to exact continuous time differences without quantization |
| **Pretraining objective** | Next-event autoregressive modeling / piecewise exponential survival hazard |
| **Downstream objective** | BCEWithLogitsLoss / time-to-event survival loss |
| **Parameter count** | ~3.2M (d=256, 6 layers, 8 heads) |
| **Sequence limit** | 512 |
| **Label** | **Faithful architecture (ported from Haiku/JAX official release), task-matched adaptation** |

---

## 9. TALE-EHR

| Field | Value |
|-------|-------|
| **Paper citation** | Yu et al., "TALE-EHR: Time-Aware Multi-Scale Continuous Time Modeling for Electronic Health Records", arXiv:2507.14847, July 2025 |
| **Official repository** | None publicly released (arXiv preprint) |
| **Architecture** | Continuous time-aware self-attention with polynomial decay weighting $w(\Delta t)$, semantic code embeddings, and multi-scale hierarchical history pooling $\mathbf{h}_t$ |
| **Differences from paper** | Implemented strictly from the mathematical definitions in arXiv:2507.14847 Section 3; adapted to unified MIMIC/NCH vocabulary and counterfactual evaluation |
| **Input representation** | Code tokens with continuous timestamp offsets and visit interval durations |
| **Age representation** | **Explicit scalar age projection** combined with continuous temporal decay |
| **Time representation** | **Order-5 polynomial continuous temporal weighting**: $A_{j,k} = \text{Softmax}(Q_j K_k^T / \sqrt{d}) \cdot \sigma(\sum_{l=0}^5 a_l \|t_j - t_k\|^l)$ |
| **Pretraining objective** | Multi-task next-visit readmission & diagnosis prediction |
| **Downstream objective** | BCEWithLogitsLoss |
| **Parameter count** | ~2.1M (d=256, 4 layers, 4 heads) |
| **Sequence limit** | 512 |
| **Label** | **Faithful architecture (implemented strictly from arXiv:2507.14847 mathematical equations), task-matched adaptation** |

---

## 10. NEST (Nested Event Stream Transformer)

| Field | Value |
|-------|-------|
| **Paper citation** | Sun et al., "NEST: Nested Event Stream Transformer for Sequences of Multisets", arXiv:2602.00520, February 2026 |
| **Official repository** | Unreleased by authors (preprint note: repository to be made public upon acceptance) |
| **Official weights** | Unreleased (trained on proprietary Duke CDM dataset of ~500k patients) |
| **Architecture** | Hierarchical sequence-of-multisets Transformer: Set-Wise Encoder (SWE) for intra-encounter permutation invariance + Cross-Set Encoder (CSE) with RoPE on encounter times + SwiGLU FFN + Pre-LayerNorm |
| **Differences from paper** | Official code/weights unreleased; fully and mathematically specified in Section 3 of the paper. Native PyTorch implementation adhering strictly to equations 1–8 with an explicit feasibility gate |
| **Input representation** | Sequences of multisets (encounters $1 \dots M$, each containing up to $N$ unordered events + $[CLS]_{m,0}$) |
| **Age representation** | **Scalar patient age projection** |
| **Time representation** | **Inter-encounter Continuous RoPE**: Applied strictly at the CSE level across encounter $[CLS]$ tokens based on encounter timestamps; SWE attention is time-unaware within encounters to enforce exact permutation invariance |
| **Pretraining objective** | Masked Set Modeling (MSM) — masking random codes within encounter multisets |
| **Downstream objective** | BCEWithLogitsLoss on final encounter $[CLS]$ representation |
| **Parameter count** | ~2.8M (d=256, 4 layers, 4 heads, d_ff=683 SwiGLU) |
| **Sequence limit** | 32 encounters × 32 codes/encounter (up to 1,024 events) |
| **Label** | **Faithful architecture (mathematically specified from arXiv:2602.00520 Section 3), unreleased official repository/weights gate documented** |

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
| MOTOR | ✓ (scalar) | ✓ (Continuous RoPE) | Pre-RMSNorm Transformer | 256 | 6 | 8 |
| TALE-EHR | ✓ (scalar) | ✓ (Order-5 Poly) | Time-Aware Attention | 256 | 4 | 4 |
| NEST | ✓ (scalar) | ✓ (CSE RoPE) | Interleaved SWE+CSE | 256 | 4 | 4 |
| DTR | ✓ (continuous) | ✓ (kernel) | Age×temporal BERT | 256 | 1 | 4 |

---

## Scientific Integrity Notes

1. **No hidden advantages**: Models keep canonical architectures. No age injection into Med-BERT, no time injection into EHR-BERT.
2. **Poor results reported**: If a faithful implementation underperforms, the result is reported, not patched.
3. **No oracle access**: Baselines never receive ground-truth β, λ, mechanism type, or oracle surfaces.
4. **No test-set tuning**: Hyperparameters selected on validation only.
5. **No architectural mixing**: Baselines retain strictly their published temporal/age mechanisms without grafting DTR's age×temporal mechanism.
