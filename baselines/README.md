# Pediatric EHR Baselines Framework

This directory contains benchmark implementations of state-of-the-art EHR baseline architectures evaluated across three experimental stages:
1. **Synthetic Benchmark (S0–S3, S5)**: Controlled counterfactual evaluation of age conditioning, continuous temporal mechanisms, and heterogeneous persistence.
2. **MIMIC Stage-1 (Next-Visit Pretraining)**: Large-scale multilabel prediction on MIMIC-IV (~30,000 code vocabulary).
3. **NCH Stage-2 (Pediatric Finetuning)**: Transfer and fine-tuning on pediatric electronic health records initialized from Stage-1 checkpoints.

---

## Baseline Architectures

| Baseline | Paper Reference | Temporal Mechanism | Age Conditioning | Official Repository / Fidelity |
| :--- | :--- | :--- | :--- | :--- |
| **Count + LightGBM** | Baseline standard | Bag-of-codes counts | Scalar feature | Non-neural baseline control |
| **RETAIN** | Choi et al., NeurIPS 2016 | Reverse-time dual GRU ($\alpha, \beta$) | None | `mp2893/RETAIN` (Task-matched) |
| **Vanilla EHR-BERT** | Control | Sinusoidal absolute positions | None | Transformer control (Task-matched) |
| **BEHRT** | Li et al., JAMIA 2020 | Discretized age intervals | Age embeddings | `kexindavis/BEHRT` (Task-matched) |
| **Med-BERT** | Rasmy et al., npj DM 2021 | Visit order positional embeddings | None | `Ryan-R/Med-BERT` (Task-matched) |
| **CEHR-BERT** | Pang et al., JBI 2021 | Artificial time-to-next-token intervals | Explicit scalar | `Sep-Pang/CEHR-BERT` (Adapted) |
| **MOTOR** | Steinberg et al., ICLR 2024 | Continuous time-to-event RoPE | Linear age projection | `som-shahlab/motor_code_release` / FEMR (Faithful) |
| **TALE-EHR** | Yu et al., arXiv:2507.14847 | Continuous order-5 polynomial temporal weighting | Linear age projection | Paper mathematical formulation (Faithful) |
| **NEST** | Sun et al., arXiv:2602.00520 | Interleaved SWE + CSE (RoPE on encounters) | Linear age projection | Paper mathematical formulation (Feasibility Gated) |
| **DTR** | Proposed architecture | Content-Persistence continuous decay | Age-conditioned kernel | Core proposed model |

---

## Unified Runner Suite

All baseline training and evaluations across all stages are orchestrated via `baselines.run_suite`:

### 1. Synthetic Benchmark
```bash
# Run all synthetic scenarios (S0–S3, S5) with counterfactual evaluation
python -m baselines.run_suite \
    --models motor,tale_ehr,nest \
    --stage synthetic \
    --config configs/baselines/synthetic.yaml

# Quick smoke test (2 epochs, S0)
python -m baselines.run_suite \
    --models motor,tale_ehr,nest \
    --stage synthetic \
    --smoke
```

### 2. MIMIC Stage-1 Pretraining
```bash
# Pretrain baselines on MIMIC-IV next-visit prediction (bf16, throughput optimized)
python -m baselines.run_suite \
    --models motor,tale_ehr,nest \
    --stage mimic_pretrain \
    --config configs/baselines/mimic.yaml
```

### 3. NCH Stage-2 Pediatric Finetuning
```bash
# Finetune on NCH starting from Stage-1 MIMIC checkpoints
python -m baselines.run_suite \
    --models motor,tale_ehr,nest \
    --stage nch_finetune \
    --config configs/baselines/nch.yaml \
    --use-mimic-checkpoints
```

### 4. Full End-to-End Pipeline
```bash
# Execute synthetic -> mimic_pretrain -> nch_finetune sequentially with checkpoint resumption
python -m baselines.run_suite \
    --models motor,tale_ehr,nest \
    --stage all \
    --config-dir configs/baselines \
    --resume
```

---

## NEST Feasibility Gate

The official code and pretrained weights for **NEST** (Sun et al., Feb 2026, arXiv:2602.00520) have not yet been made public by the authors (the preprint notes that code will be released upon acceptance, and models were trained on proprietary Duke Health data).

- `baselines/nest/model.py` implements the **faithful paper architecture** strictly following Section 3 of arXiv:2602.00520:
  - Sequence of multisets (encounters)
  - Set-Wise Encoder (SWE) enforcing intra-encounter permutation invariance
  - Cross-Set Encoder (CSE) with Continuous RoPE across encounter $[CLS]$ representations
  - SwiGLU feedforward networks and Pre-LayerNorm
  - Masked Set Modeling (MSM) support
- The runner suite includes `check_nest_feasibility()`:
  - When `--strict-feasibility` is passed, unreleased official implementations are cleanly skipped with a diagnostic notice.
  - By default, the paper-faithful architecture is executed seamlessly.

---

## Unit and Contract Tests

All baselines implement the unified `BaselineModel` contract (`predict`, `training_step`, `save_checkpoint`, `load_checkpoint`, `model_card`, `has_age_input`, `has_time_input`).

To run the full unit test suite:
```bash
# Test all existing baselines (55 tests)
pytest baselines/tests/test_all_baselines.py -v

# Test new baselines (MOTOR, TALE-EHR, NEST: 20 tests)
pytest baselines/tests/test_new_baselines.py -v
```
All 75 unit and contract tests verify:
- Output tensor shapes and logits calibration
- Gradient flow and finite loss backward pass
- Overfitting on tiny synthetic batches
- Checkpoint saving and load roundtrip invariance
- Handling of padding, lengths, timestamps, and age tensors
- Permutation invariance within encounter multisets (NEST SWE)
