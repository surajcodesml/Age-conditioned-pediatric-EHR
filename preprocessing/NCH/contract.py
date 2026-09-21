"""Dump the Stage-1 MIMIC pretraining input contract from the live implementation.

Reads code + the already-written ``adkm_s0`` config (read-only). Does not
regenerate the MIMIC vocabulary or touch the running training process.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import paths


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _vocab_prefix_counts(vocab: dict[str, int]) -> dict[str, int]:
    from collections import Counter
    c = Counter(k.split("_", 1)[0] for k in vocab)
    return dict(c.most_common())


def build_contract() -> dict:
    from model_new.data import (
        DAYS_PER_YEAR,
        N_RACE,
        RACE_LABELS,
        WEEK_DAYS,
        demo_layout,
        lag_to_tau,
    )
    from stage1_mimic_pretrain.config import (
        DEFAULT_MAX_SEQ_LEN,
        EMBEDDING_PATH,
        MIMIC_AGE_MEAN_YEARS,
        MIMIC_AGE_STD_YEARS,
        TENSORIZED_DIR,
        VOCAB_PATH,
        age_transform_spec,
        target_spec,
        tau_transform_spec,
    )

    vocab = {str(k): int(v) for k, v in _load_json(paths.VOCAB_PATH).items()}
    descriptions = {}
    if paths.DESCRIPTIONS_PATH.exists():
        descriptions = _load_json(paths.DESCRIPTIONS_PATH)
    run_cfg = _load_json(paths.STAGE1_RUN_CONFIG) if paths.STAGE1_RUN_CONFIG.exists() else {}
    corpus = {}
    if paths.CORPUS_STATS_PATH.exists():
        corpus = _load_json(paths.CORPUS_STATS_PATH)

    demo_dim, demo_channels = demo_layout("one_hot")
    prefixes = _vocab_prefix_counts(vocab)
    v = len(vocab)

    # Confirm lag_to_tau is the documented function.
    import torch
    tau_probe = float(lag_to_tau(torch.tensor([7.0])).item())  # log1p(1) = ln2

    age_from_run = (run_cfg.get("data") or {}).get("age_transform") or {}
    tau_from_run = (run_cfg.get("data") or {}).get("tau_transform") or {}
    data_paths = (run_cfg.get("data") or {}).get("paths") or {}

    contract = {
        "source": {
            "active_run": "stage1_mimic_pretrain.train --arm age_temporal --run_name adkm_s0 --seed 0",
            "active_run_config": str(paths.STAGE1_RUN_CONFIG),
            "git_commit": run_cfg.get("git_commit"),
            "note": "Contract is taken from implementation + the already-written adkm_s0 config. The run directory is not modified.",
        },
        "pipeline": {
            "raw_mimic": "/home/suraj/Data/MIMIC-IV/physionet.org/files/mimiciv/3.1",
            "build_event_table": {
                "file": "preprocessing/build_event_table.py",
                "function": "main",
                "output": str(paths.MIMIC_PROCESSED / "patient_events_full.parquet"),
            },
            "rollup": {
                "file": "preprocessing/rollup_and_describe.py",
                "functions": ["normalize_code", "create_phecode_maps", "create_ccs_maps", "build_rollup"],
                "output": str(paths.MIMIC_PROCESSED / "patient_events_rolled_full.parquet"),
            },
            "splits": {
                "file": "preprocessing/build_splits.py",
                "function": "assign_splits",
                "strategy": "patient-level, quintiles of event count, 70/10/20, seed=42",
            },
            "embeddings_and_vocab": {
                "file": "preprocessing/compute_bge_embeddings.py",
                "function": "main",
                "vocab_path": str(VOCAB_PATH),
                "embedding_path": str(EMBEDDING_PATH),
                "encoder": "BAAI/bge-m3",
                "embedding_dim": 1024,
                "frozen": True,
            },
            "tensorize": {
                "file": "model_new/tensorize_pretrain.py",
                "functions": ["build_subject_payload", "tensorize_bucket", "rebuild_split"],
                "output": str(TENSORIZED_DIR),
            },
            "dataset": {
                "file": "model_new/data.py",
                "class": "TensorizedPretrainDataset",
                "collate": "pretrain_collate / make_collate",
            },
            "model": {
                "file": "stage1_mimic_pretrain/model.py",
                "class": "MinimalDKMModel",
                "train": "stage1_mimic_pretrain/train.py",
            },
        },
        "event_representation": {
            "parquet_schema": [
                "subject_id", "hadm_id", "event_time", "code_id", "code_type",
                "timestamp_days", "log_delta_t", "age_at_event_days", "sex", "race",
            ],
            "model_event_tuple": {
                "code_indices": "[B, L] int64; PAD=0, UNK=1, real = vocab_index + 2",
                "timestamps_days": "[B, L] float64; days from the patient's first event; padded 0.0",
                "age_years": "[B, L] float32; age_at_event_days / 365.25; padded 0.0",
                "demographics": "[B, L, 9] float32; last-event vector is consumed by the head",
                "attention_mask": "[B, L] bool",
                "lengths": "[B] int64",
            },
            "timestamp": {
                "raw": "event_time (absolute, deidentified)",
                "model": "timestamp_days = (event_time - min_event_time_of_patient) / 86400",
                "implementation": "preprocessing/build_event_table.py events_time CTE",
                "storage_dtype": "float32 in shards; collate promotes to float64 for lag arithmetic",
            },
            "age_at_event": {
                "formula": "age_at_event_days = (event_time - dob) / 86400",
                "dob_mimic": "July 1 of (anchor_year - anchor_age); see patients_demo in build_event_table.py",
                "years": "age_years = age_at_event_days / 365.25",
                "clipping": "none in the loader",
                "log1p": False,
            },
            "index_age": {
                "pretrain": "not a single index date; each sample is a next-visit forecast window",
                "pooling_age": "age of the last valid input event (AttentionPooling.last_valid_index)",
                "demographic_age": "same last-event age, then z-scored in MinimalDKMModel.standardize_demo_age",
            },
            "token": {
                "string": "rolled code_id with type prefix (PHE_, CCS_, RXN_, ...)",
                "vocab_id": "code_vocab.json maps token -> [0, V)",
                "model_id": "PAD=0, UNK=1, real=vocab_id+2 (model_new.data._pad_common)",
                "text_embeddings": "frozen BGE-M3 1024-d table; row 0/1 are zeros for PAD/UNK",
            },
            "event_types_included": {
                "diagnosis": "ICD9_/ICD10_ then PheCode PHE_ (unmapped ICD leftover kept)",
                "procedure": "PROC9_/PROC10_ then CCS_ (ICD-9-CM / ICD-10-PCS)",
                "hcpcs": "HCPCS_ then CCS_ via the ICD-10-PCS CCS join (mostly leftover HCPCS_)",
                "medication": "NDC_ then RXN_<RxCUI>",
                "lab": "LAB_<mimic_itemid> passthrough",
                "chart": "CHART_<mimic_itemid> passthrough",
                "drg": "DRG_<code> passthrough",
                "input": "INPUT_<itemid> passthrough",
                "output": "OUTPUT_<itemid> passthrough",
                "icu_procedure": "ICUPROC_<itemid> passthrough",
            },
            "demographics": {
                "sex": "1 if gender=='M' else 0 (build_event_table.patients_demo)",
                "race": "model_new.data.encode_race; 7 buckets " + ",".join(RACE_LABELS),
                "race_encoding_default": "one_hot",
                "demo_dim": demo_dim,
                "demo_channels": list(demo_channels),
                "n_race": N_RACE,
                "note": "patient-level constants broadcast across the sequence; head uses last-event demo",
            },
            "continuous_values": "not used as sequence tokens; labs/charts are itemid presence only",
            "encounter": "hadm_id defines visit blocks for the pretraining target; not an input feature",
        },
        "vocabulary": {
            "path": str(paths.VOCAB_PATH),
            "size": v,
            "id_range": [0, v - 1],
            "token_to_id": "JSON object {token: int}; keys sorted alphabetically at BGE build time",
            "id_to_token": "inverse of code_vocab.json; embedding table code_ids = ['[PAD]','[UNK]'] + sorted tokens",
            "special_tokens": {
                "PAD": {"vocab_index": None, "model_id": 0, "embedding": "zeros"},
                "UNK": {"vocab_index": v, "model_id": 1, "embedding": "zeros",
                        "note": "unk_vocab_index = len(code_vocab) in shards; collate maps that id -> 1"},
            },
            "prefixes": prefixes,
            "code_systems": {
                "ICD-9 diagnosis": "raw ICD9_<code> (no dots, as stored by MIMIC) or rolled PHE_",
                "ICD-10 diagnosis": "raw ICD10_<code> (no dots) or rolled PHE_",
                "ICD-9/10 procedures": "rolled CCS_ (231 categories); leftover PROC10_ exists, PROC9_ none in frozen vocab",
                "CPT/HCPCS": "HCPCS_<code> leftover after CCS join against ICD-10-PCS (low yield)",
                "medication": "RXN_<RxCUI>; 12 leftover NDC_ tokens",
                "labs": "LAB_<MIMIC itemid> — MIMIC-local, not portable",
                "chart": "CHART_<MIMIC itemid> — MIMIC-local, not portable",
                "other": "DRG_, INPUT_, OUTPUT_, ICUPROC_",
            },
            "normalization": {
                "function": "preprocessing.rollup_and_describe.normalize_code",
                "rule": "strip().upper().replace('.', '')",
                "icd_join": "PheWAS maps joined on the normalized ICD body after stripping the ICD9_/ICD10_ prefix",
                "ndc": "strip hyphens/spaces before RxNorm lookup",
                "type_encoded_in_token": True,
            },
            "text_encoder": {
                "name": "BAAI/bge-m3",
                "file": "preprocessing/compute_bge_embeddings.py",
                "frozen": True,
                "dim": 1024,
                "rows": v + 2,
                "descriptions_path": str(paths.DESCRIPTIONS_PATH),
                "n_descriptions": len(descriptions),
            },
            "rare_code_filtering": "none at vocab build; vocab = every code that received a description after rollup",
            "oov": "codes missing from code_vocab.json map to unk_vocab_index = V, then model id 1",
            "padding": "collate left-aligns real events and fills the right with PAD=0; mask False on pad",
        },
        "temporal": {
            "event_order": "subject_id, event_time, code_id (parquet); within a visit: event_time, code_id (tensorize)",
            "equal_timestamps": "stable secondary key code_id (and original index at window selection via lexsort)",
            "lag_definition": "tau_ij = log1p(|t_i - t_j| / c) with c = 7 days",
            "lag_function": "model_new.data.lag_to_tau",
            "lag_probe_tau_of_7_days": tau_probe,
            "units": "days",
            "log1p": True,
            "c_days": WEEK_DAYS,
            "tau_max": {
                "computed": corpus.get("tau_max"),
                "source": corpus.get("tau_max_source"),
                "used_by_stage1_model": False,
                "chebyshev_rescale": False,
                "note": (
                    "tau_max is recorded by corpus_stats for the obsolete DKM Chebyshev path "
                    "(INV-TMAX). stage1_mimic_pretrain.config.tau_transform_spec sets "
                    "tau_max_used=False and chebyshev_rescale=False. The live model consumes "
                    "raw log1p(|dt|/7). NCH must still freeze this MIMIC tau_max for any "
                    "diagnostic of temporal support; it must not be re-estimated."
                ),
            },
            "clipping": "none on tau in Stage-1 attention",
            "normalization_minus1_1": False,
            "mimic_derived_constants": {
                "WEEK_DAYS": WEEK_DAYS,
                "DAYS_PER_YEAR": DAYS_PER_YEAR,
                "tau_max_mimic_train": corpus.get("tau_max"),
                "span_days_max": corpus.get("span_days_max"),
            },
            "max_history": "no calendar window; count truncation only",
            "sequence_truncation": {
                "max_seq_len": DEFAULT_MAX_SEQ_LEN,
                "direction": "keep newest events (drop oldest); select_forecast_input_indices",
            },
            "temporal_masking": "none (no causal mask)",
            "padding_masking": "build_pair_mask / build_key_mask; padded keys get -inf then 0 attention",
            "stored_log_delta_t": {
                "file": "preprocessing/build_event_table.py",
                "formula": "ln(1 + max(dt_consecutive_days, 0))  — NOT the model tau",
                "used_by_model": False,
            },
            "from_active_run": tau_from_run,
        },
        "age": {
            "calculation": "age_at_event_days = (event_time - dob)/86400; years = days/365.25",
            "units": {"days": "age_at_event_days", "years": "age_years"},
            "age_at_event_vs_index": (
                "Every event carries its own age. Attention conditions on query age a_i. "
                "The prediction head uses the last input event's age (and its z-scored demo channel)."
            ),
            "clipping": "none",
            "log1p": False,
            "normalization": {
                "name": "standardize_years",
                "formula": "z(a) = (a - mean) / sd",
                "mean_frozen_config": MIMIC_AGE_MEAN_YEARS,
                "sd_frozen_config": MIMIC_AGE_STD_YEARS,
                "mean_active_run": age_from_run.get("mean"),
                "sd_active_run": age_from_run.get("sd"),
                "source": "model_new.data.corpus_stats event-level moments on MIMIC train",
                "trainable": False,
                "two_sites": {
                    "attention": "AgeTemporalBias.z_of(age_years) using buffers age_mean/age_sd",
                    "demographics": "MinimalDKMModel.standardize_demo_age on channel 0 of last-event demo",
                },
            },
            "how_age_enters_model": [
                "token sequence does not include age",
                "age_years tensor -> lambda(a)=lambda0 + beta * z(a) in self-attention",
                "demographics[...,0]=raw years; last event z-scored and projected (demo_proj) then concatenated onto pooled h",
            ],
            "transform_spec": age_transform_spec(),
        },
        "sequence_construction": {
            "unit": "one sample per (patient, next hadm-visit V_{m+1}) with ≥1 prior event",
            "index_event": "start_time(V_{m+1}) = min timestamps in the target visit (strict past for input)",
            "history_inclusion": "timestamp < target_time; ties go to the target (INV-HORIZON)",
            "max_seq_len": DEFAULT_MAX_SEQ_LEN,
            "min_events": (
                "raw table: ≥5 events and >1 unique timestamp per patient "
                "(build_event_table). tensorize: ≥2 visits. dataset: ≥1 pre-boundary event."
            ),
            "padding": "right pad to batch max length (not max_seq_len if shorter)",
            "truncation_direction": "drop oldest",
            "chronological_order": True,
            "duplicates": "not dropped; identical (time, code) rows remain",
            "splits": {
                "strategy": "patient-level quintile-stratified 70/10/20, seed 42 (build_splits.py)",
                "active_run_counts": (run_cfg.get("data") or {}).get("patient_splits"),
                "leakage_prevention": "no subject_id overlap across train/val/test (asserted in adkm_s0 config)",
            },
        },
        "stage1_objective": {
            "pretrain_only": {
                "target": "multi-hot codes of the next hadm visit; UNK dropped; duplicates collapsed",
                "loss": "BCEWithLogitsLoss, unweighted, no pos_weight",
                "head": "PredictionHead over |V|=30635",
                "spec": target_spec(),
            },
            "encoder_inputs_stage2_must_reproduce": [
                "code_indices with PAD=0 UNK=1 real=id+2 using the frozen MIMIC vocab",
                "timestamps_days in days (origin may be first event of the retained window; only differences matter)",
                "age_years = age_at_event_days/365.25",
                "demographics layout (age_years, sex, 7-d race one-hot)",
                "attention_mask / lengths",
                "WEEK_DAYS=7 lag_to_tau",
                "frozen age mean/sd for z(a)",
                "max_seq_len=1024 newest-event truncation",
                "padding-only mask (not causal)",
            ],
            "not_required_for_stage2_encoder": [
                "future-visit multi-hot target_codes",
                "hadm visit-block construction for forecasting",
                "pretrain prediction head",
            ],
        },
        "active_run_paths": data_paths,
        "reuse_for_nch": {
            "must_reuse": [
                "data/processed/code_vocab.json (do not regenerate)",
                "data/processed/bge_embeddings.pt",
                "lag_to_tau / WEEK_DAYS=7",
                "DAYS_PER_YEAR=365.25",
                "MIMIC_AGE_MEAN_YEARS / MIMIC_AGE_STD_YEARS (or the adkm_s0 recorded mean/sd)",
                "encode_race / encode_sex conventions",
                "PAD=0 UNK=1 offset +2",
                "max_seq_len=1024 newest truncation",
                "PheWAS ICD→PheCode maps under data/processed/mappings/",
            ],
            "must_not_refit_on_nch": [
                "vocabulary",
                "BGE embeddings",
                "age mean/sd",
                "tau_max (even though unused by the live kernel)",
                "WEEK_DAYS",
            ],
        },
    }
    return contract


def contract_to_markdown(c: dict) -> str:
    voc = c["vocabulary"]
    ev = c["event_representation"]
    tmp = c["temporal"]
    age = c["age"]
    lines = [
        "# Stage-1 MIMIC pretraining contract",
        "",
        "Source of truth: implementation (`stage1_mimic_pretrain/`, `model_new/data.py`,",
        "`preprocessing/build_event_table.py`, `preprocessing/rollup_and_describe.py`)",
        "plus the already-written `adkm_s0` config. The MIMIC vocabulary was **not** regenerated.",
        "",
        f"- Active run: `{c['source']['active_run']}`",
        f"- Git commit recorded by the run: `{c['source']['git_commit']}`",
        "",
        "## Event representation",
        "",
        "Model batch (`model_new.data.pretrain_collate`):",
        "",
    ]
    for k, v in ev["model_event_tuple"].items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        f"- Timestamp on the event table: {ev['timestamp']['model']}",
        f"- Age at event: `{age['calculation']}`",
        f"- Token string: `{ev['token']['string']}`",
        f"- Model id: `{ev['token']['model_id']}`",
        "",
        "Modalities in the frozen vocabulary (counts are unique tokens):",
        "",
    ]
    for pref, n in voc["prefixes"].items():
        lines.append(f"- `{pref}_`: {n}")
    lines += [
        "",
        "## Vocabulary",
        "",
        f"- Path: `{voc['path']}`",
        f"- Size `|V|` = **{voc['size']}**",
        f"- Token→ID: JSON `{voc['token_to_id']}`",
        f"- Special: PAD model id **0**, UNK model id **1** (`unk_vocab_index = {voc['size']}` in shards)",
        f"- Frozen text encoder: `{voc['text_encoder']['name']}` dim {voc['text_encoder']['dim']}, "
        f"{voc['text_encoder']['rows']} rows (PAD/UNK zero vectors)",
        f"- Rare-code filter: {voc['rare_code_filtering']}",
        f"- OOV: {voc['oov']}",
        f"- Code-system is encoded in the token prefix: `{voc['normalization']['type_encoded_in_token']}`",
        f"- ICD normalization: `{voc['normalization']['rule']}` (`{voc['normalization']['function']}`)",
        "",
        "## Temporal preprocessing",
        "",
        f"- Order: {tmp['event_order']}",
        f"- Ties: {tmp['equal_timestamps']}",
        f"- **τ** = `{tmp['lag_definition']}` implemented by `{tmp['lag_function']}`",
        f"- `log1p`: {tmp['log1p']}; `c` = {tmp['c_days']} days",
        f"- Chebyshev / `[-1,1]` rescale: **{tmp['normalization_minus1_1']}**",
        f"- `tau_max` used by Stage-1 model: **{tmp['tau_max']['used_by_stage1_model']}**",
        f"- Recorded MIMIC `tau_max` (diagnostic / INV-TMAX): `{tmp['tau_max']['computed']}`",
        f"- Truncation: newest {c['sequence_construction']['max_seq_len']} events",
        f"- Masking: padding-only, not causal",
        "",
        "**NCH must reuse MIMIC `WEEK_DAYS=7` and the recorded `tau_max` for diagnostics;**",
        "it must not estimate a new `c` or `tau_max`. The live kernel does not clip τ.",
        "",
        "## Age",
        "",
        f"- `{age['calculation']}`",
        f"- z(a) mean = `{age['normalization']['mean_active_run']}` (run) / `{age['normalization']['mean_frozen_config']}` (config default)",
        f"- z(a) sd = `{age['normalization']['sd_active_run']}` / `{age['normalization']['sd_frozen_config']}`",
        "- No age clipping, no age `log1p`.",
        "- Age enters attention as λ(a)=λ0+β z(a_i) and the head as z-scored last-event demo channel 0.",
        "",
        "## Sequence construction",
        "",
        f"- {c['sequence_construction']['unit']}",
        f"- History: `{c['sequence_construction']['history_inclusion']}`",
        f"- Splits: {c['sequence_construction']['splits']['strategy']}",
        "",
        "## Stage-1 objective vs Stage-2 encoder inputs",
        "",
        "Pretraining-only: next-visit multi-hot BCE over `|V|`.",
        "",
        "Stage-2 must reproduce the encoder inputs listed in the JSON field",
        "`stage1_objective.encoder_inputs_stage2_must_reproduce`.",
        "",
        "## Traceability",
        "",
    ]
    for name, spec in c["pipeline"].items():
        if isinstance(spec, dict) and "file" in spec:
            fn = spec.get("function") or spec.get("functions") or spec.get("class")
            lines.append(f"- **{name}**: `{spec['file']}` ({fn})")
    lines.append("")
    return "\n".join(lines)


def write_contract(out_dir: Path | None = None) -> dict:
    from model_new.diagnostics import write_json

    out_dir = Path(out_dir) if out_dir is not None else paths.CONTRACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = build_contract()
    write_json(out_dir / "pretraining_contract.json", contract)
    (out_dir / "pretraining_contract.md").write_text(contract_to_markdown(contract), encoding="utf-8")
    return contract
