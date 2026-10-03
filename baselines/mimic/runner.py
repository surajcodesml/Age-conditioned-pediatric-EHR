#!/usr/bin/env python3
"""Stage-1 MIMIC baseline runner — train and evaluate baselines on MIMIC-IV next-visit prediction.

Usage:
    python -m baselines.mimic.runner --models all
    python -m baselines.mimic.runner --models dtr --smoke
    python -m baselines.mimic.runner --models dtr --name-suffix _new
    python -m baselines.mimic.runner --models retain,behrt --seed 1

Order for ``--models all``: DTR arms first, then retain / ehr_bert / behrt /
medbert / cehrbert.

``--seed`` (default 0): seed 0 writes under ``results/baselines/mimic/<name>/``;
seed ≠ 0 under ``results/baselines/mimic/seed_<seed>/<name>/`` so multi-seed
runs never skip or overwrite the canonical seed-0 artifacts.

Legacy DTR (no suffix): Minimal-DKM arms ``age_temporal``, ``no_interaction``
under ``results/baselines/mimic/dtr_<arm>/``.

Corrected Content-Persistence DTR (``--name-suffix _new``): arms
``temporal_only``, ``age_temporal`` under ``dtr_<arm>_new/``; writes
``all_results_new.json``. Never overwrites legacy dirs or ``all_results.json``.

Artifacts per model dir:
  best_checkpoint.pt  last_checkpoint.pt  checkpoint.pt(=best)
  config.json  history.json  result.json
Early stopping (patience=3) cannot fire before MIN_EPOCHS=5.
Smoke runs write under results/baselines/mimic/_smoke/ (never clobber production).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model_new.data import (
    TensorizedPretrainDataset,
    dataloader_worker_init,
    make_collate,
)

from baselines.common.metrics import multilabel_metrics
from baselines.common.training import (
    evaluate_loader, set_seed, get_device, train_neural_baseline,
)
from baselines.common.capacity_report import count_parameters

# Import all baselines
from baselines.lightgbm.model import LightGBMBaseline, build_features_from_batch
from baselines.retain.model import RETAINModel
from baselines.ehr_bert.model import EHRBertModel
from baselines.behrt.model import BEHRTModel
from baselines.medbert.model import MedBERTModel
from baselines.cehrbert_adapter.adapter import CEHRBertAdapter
from baselines.motor.model import MOTORModel
from baselines.tale_ehr.model import TALEEHRModel
from baselines.nest.model import NESTModel
from baselines.mimic.encounter_batch import (
    tokens_to_encounter_batch,
    DEFAULT_MAX_CODES_PER_ENCOUNTER,
    DEFAULT_MAX_ENCOUNTERS,
)

# MIMIC defaults
BATCH_SIZE = 32  # 64 + full-val concat OOMs on ~30k-code heads
MAX_EPOCHS = 10
MAX_SEQ_LEN = 256
MODEL_SEED = 0
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
DROPOUT = 0.1
LR = 3e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 3
MIN_EPOCHS = 5  # early stopping cannot fire before this many epochs
GRAD_CLIP = 1.0
# Cap eval batches: full val/test logit tensors are 6–13+ GB and get the process OOM-killed.
VAL_MAX_BATCHES = 50    # 50 × 32 ≈ 1.6k examples for early stopping
TEST_MAX_BATCHES = 100   # 100 × 32 ≈ 3.2k examples for reported test metrics
NUM_WORKERS = 4

# Throughput-optimized micro-batches (bf16, Radeon AI PRO R9700 30GiB).
# Chosen from train-step sweeps: maximize samples/sec with peak VRAM ≲12GiB
# so long multi-seed runs keep fragmentation headroom. Host RAM is ~30GiB —
# avoid 768+ with num_workers>0 (one-hot |V|≈30k labels dominate prefetch).
THROUGHPUT_BATCH_SIZES: dict[str, int] = {
    "retain": 512,
    "ehr_bert": 512,
    "behrt": 384,
    "medbert": 512,
    "cehrbert": 512,
    "motor": 256,
    "tale_ehr": 256,
    "nest": 32,
    "count_lightgbm": 512,
    "dtr": 32,
}


def configure_dataloader_multiprocessing(num_workers: int) -> None:
    """Avoid CUDA+fork stalls when multi-process loaders are requested.

    Must run before the first DataLoader with workers > 0 iterates (workers
    spawn on first ``iter``). On ROCm, forking after ``model.to(cuda)`` can
    stall minutes on batch 1.
    """
    if num_workers <= 0:
        return
    import torch.multiprocessing as mp

    method = mp.get_start_method(allow_none=True)
    if method != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
            print(
                f"  DataLoader start_method: spawn "
                f"(was {method!r}; avoids CUDA+fork worker stalls)",
                flush=True,
            )
        except RuntimeError as e:
            print(
                f"  WARNING: could not set spawn start method ({e}); "
                f"prefer --num_workers 0 on ROCm",
                flush=True,
            )

# Canonical Content-Persistence width (matches synthetic ``_new`` lock).
CP_D_MODEL = 64
# OOB embedding fix landed; batch 32 is Stage-1 standard and fits easily in 32GB
# VRAM (peak ~2–4GB). No grad accumulation needed.
CP_MICRO_BATCH = 32
CP_GRAD_ACCUM = 1
# Half of the 16 CPU cores — leave headroom for the main process + OS.
CP_NUM_WORKERS = 8

BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "count_lightgbm": {},
    "retain": {"d_emb": 128, "d_rnn": 128, "dropout": DROPOUT},
    "ehr_bert": {"d_model": D_MODEL, "n_layers": 4, "n_heads": 4,
                 "d_ff": D_MODEL * 4, "dropout": DROPOUT, "max_seq_len": MAX_SEQ_LEN},
    "behrt": {"d_model": 288, "n_layers": 6, "n_heads": 12, "dropout": DROPOUT,
              "max_seq_len": MAX_SEQ_LEN},
    "medbert": {"d_model": 192, "n_layers": 6, "n_heads": 6, "dropout": DROPOUT,
                "max_seq_len": MAX_SEQ_LEN},
    "cehrbert": {"d_model": 128, "n_layers": 5, "n_heads": 8, "dropout": DROPOUT,
                 "max_seq_len": MAX_SEQ_LEN},
    "motor": {"d_model": D_MODEL, "n_layers": 6, "n_heads": 8, "dropout": DROPOUT,
              "max_seq_len": MAX_SEQ_LEN},
    "tale_ehr": {"d_model": D_MODEL, "n_layers": 4, "n_heads": 4, "dropout": DROPOUT,
                 "max_seq_len": MAX_SEQ_LEN},
    "nest": {"d_model": D_MODEL, "n_layers": 4, "n_heads": 4, "dropout": DROPOUT,
             "max_encounters": 32, "max_codes_per_encounter": 32, "max_seq_len": MAX_SEQ_LEN},
}

class RenameLoader:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for batch in self.loader:
            if "target_codes" in batch and "labels" not in batch:
                batch["labels"] = batch["target_codes"]
            if "code_indices" in batch and "code_ids" not in batch:
                batch["code_ids"] = batch["code_indices"]
            if "attention_mask" in batch and "padding_mask" not in batch:
                batch["padding_mask"] = ~batch["attention_mask"]
            if "is_query" not in batch:
                batch["is_query"] = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
            if "timestamps_days" in batch and "tau" not in batch:
                batch["tau"] = batch["timestamps_days"].float()
            if "age_years" in batch and "age" not in batch:
                batch["age"] = batch["age_years"].max(dim=1).values.float()
            yield batch
    def __len__(self):
        return len(self.loader)


class EncounterLoader:
    """RenameLoader → Content-Persistence encounter tensors (main-process fallback)."""

    def __init__(self, loader: RenameLoader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            yield tokens_to_encounter_batch(batch)

    def __len__(self):
        return len(self.loader)


def _rename_batch(batch: dict[str, Any]) -> dict[str, Any]:
    if "target_codes" in batch and "labels" not in batch:
        batch["labels"] = batch["target_codes"]
    if "code_indices" in batch and "code_ids" not in batch:
        batch["code_ids"] = batch["code_indices"]
    if "attention_mask" in batch and "padding_mask" not in batch:
        batch["padding_mask"] = ~batch["attention_mask"]
    if "is_query" not in batch and "attention_mask" in batch:
        batch["is_query"] = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
    if "timestamps_days" in batch and "tau" not in batch:
        batch["tau"] = batch["timestamps_days"].float()
    if "age_years" in batch and "age" not in batch:
        batch["age"] = batch["age_years"].max(dim=1).values.float()
    return batch


def make_cp_collate(
    base_collate,
    max_codes_per_encounter: int = DEFAULT_MAX_CODES_PER_ENCOUNTER,
    max_encounters: int = DEFAULT_MAX_ENCOUNTERS,
):
    """Collate that emits encounter tensors inside DataLoader workers."""

    def _collate(examples):
        batch = _rename_batch(base_collate(examples))
        return tokens_to_encounter_batch(
            batch,
            max_codes_per_encounter=max_codes_per_encounter,
            max_encounters=max_encounters,
        )

    return _collate


# Legacy Minimal-DKM ablation arms (unsuffixed dirs).
DTR_ARMS_LEGACY = ("age_temporal", "no_interaction")
# Content-Persistence DTR ablation pair (``_new`` dirs).
DTR_ARMS_CP = ("temporal_only", "age_temporal")
# Default when no suffix (backward compatible).
DTR_ARMS = DTR_ARMS_LEGACY


def dtr_arm_dirname(arm: str, name_suffix: str = "") -> str:
    return f"dtr_{arm}{name_suffix}"


def uses_content_persistence(arm: str, name_suffix: str = "") -> bool:
    """Only suffixed ``_new`` DTR runs use Content-Persistence (legacy untouched)."""
    return bool(name_suffix) and arm in DTR_ARMS_CP


class MIMICDTRAdapter(torch.nn.Module):
    """Minimal-DKM (DTR) wrapper for MIMIC Stage-1 next-visit pretraining."""

    def __init__(
        self,
        n_codes,
        arm: str = "age_temporal",
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ):
        super().__init__()
        from stage1_mimic_pretrain.model import MinimalDKMModel
        from model_new.data import demo_layout
        demo_dim, demo_channels = demo_layout("one_hot")
        self.arm = arm
        self.n_codes = n_codes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.model = MinimalDKMModel(
            num_codes=n_codes,
            arm=arm,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            use_residual=True,
            use_layernorm=True,
            use_ffn=True,
            demo_dim=demo_dim,
            demo_channels=demo_channels,
            race_encoding="one_hot",
            demo_hidden=64,
            embedding_path=REPO_ROOT / "data/processed/bge_embeddings.pt",
        )

    @property
    def model_card(self):
        return {
            "name": f"dtr_{self.arm}",
            "arm": self.arm,
            "architecture": "MinimalDKM",
            "n_codes": self.n_codes,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
        }

    def training_step(self, batch):
        self.model.train()
        out = self.model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["code_logits"], batch["labels"]
        )
        return {"loss": loss}

    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
            return {"logits": out["code_logits"]}

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Training loop already restored best weights into ``self``.
        torch.save(self.state_dict(), path / "checkpoint.pt")
        with (path / "config.json").open("w") as f:
            json.dump(self.model_card, f, indent=2)

    def load_checkpoint(self, path):
        path = Path(path)
        ckpt = path / "checkpoint.pt"
        if not ckpt.exists():
            ckpt = path / "best_checkpoint.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.load_state_dict(state)


class MIMICCPDTRAdapter(torch.nn.Module):
    """Content-Persistence DTR for MIMIC Stage-1 (``_new`` arms).

    temporal_only: β frozen at 0; age_temporal: β learnable (init 0).
    Mass-preserving raw-additive history aggregation.
    """

    def __init__(
        self,
        n_codes: int,
        arm: str = "age_temporal",
        d_model: int = CP_D_MODEL,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Append (do not prepend): synthetic_age_temporal/baselines.py would
        # otherwise shadow the baselines/ package.
        sat = str(REPO_ROOT / "synthetic_age_temporal")
        if sat not in sys.path:
            sys.path.append(sat)
        from model_dtr import DevelopmentalTemporalRetrieval

        if arm not in ("temporal_only", "age_temporal"):
            raise ValueError(
                f"CP DTR arms are temporal_only|age_temporal, got {arm!r}"
            )
        self.arm = arm
        self.n_codes = n_codes
        self.d_model = d_model
        # Encoder vocab must be num_codes+2 (pad/unk specials appear in MIMIC
        # sequences). Prediction head stays at n_codes (= label width).
        self.n_emb_codes = n_codes + 2
        self.model = DevelopmentalTemporalRetrieval(
            n_codes=self.n_emb_codes,
            n_targets=n_codes,
            d_model=d_model,
            age_temporal=(arm == "age_temporal"),
            aggregation="raw_additive",
            dropout=dropout,
            max_codes_per_encounter=64,
        )
        # Stage-1 MIMIC protocol: frozen BGE code table + trainable projection
        # into d_model (same frozen-embedding convention as MinimalDKM).
        self._init_frozen_bge_code_table(n_emb_codes=self.n_emb_codes, d_model=d_model)
        # Zero-init the |V|≈30k prediction layers. Default Linear init yields
        # ±O(10) logits on MIMIC and can trip ROCm kernels; all-zero logits
        # start at BCE≈log(2) and match common multilabel practice.
        last = self.model.history_head[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        torch.nn.init.zeros_(self.model.age_head.weight)
        torch.nn.init.zeros_(self.model.age_head.bias)
        if hasattr(self.model, "bias") and isinstance(self.model.bias, torch.nn.Parameter):
            torch.nn.init.zeros_(self.model.bias)

    def _init_frozen_bge_code_table(self, *, n_emb_codes: int, d_model: int) -> None:
        emb_path = REPO_ROOT / "data/processed/bge_embeddings.pt"
        obj = torch.load(emb_path, map_location="cpu", weights_only=False)
        table = obj["embeddings"] if isinstance(obj, dict) else obj
        table = table.float()
        if table.shape[0] < n_emb_codes:
            raise ValueError(
                f"BGE table rows {table.shape[0]} < n_emb_codes {n_emb_codes} "
                f"(need num_codes+2 for MIMIC specials)"
            )
        # Full MinimalDKM-sized table (vocab + 2 specials). Truncating to
        # n_codes caused ROCm HSA aborts on OOB gathers (ids 30635/30636).
        table = table[:n_emb_codes].contiguous()
        enc = self.model.encounter_encoder
        # Replace trainable Embedding with frozen table + Linear(BGE→d_model).
        del enc.code_emb
        enc.register_buffer("code_emb_table", table, persistent=True)
        enc.n_emb_codes = n_emb_codes
        enc.code_proj = torch.nn.Linear(table.shape[1], d_model, bias=False)
        torch.nn.init.xavier_uniform_(enc.code_proj.weight)

        def _encode(enc_code_ids, enc_code_mask, _enc=enc):
            # Clamp protects against any future OOB; table covers 0..n_emb-1.
            ids = enc_code_ids.clamp(0, _enc.n_emb_codes - 1)
            e = _enc.code_proj(_enc.code_emb_table[ids])
            mask = enc_code_mask.to(e.dtype).unsqueeze(-1)
            summed = (e * mask).sum(dim=2)
            denom = mask.sum(dim=2).clamp(min=1.0)
            return _enc.enc_mlp(summed / denom)

        enc.forward = _encode  # type: ignore[method-assign]

    @property
    def model_card(self):
        card = {
            "name": f"dtr_{self.arm}",
            "arm": self.arm,
            "architecture": "Content-Persistence DTR",
            "n_codes": self.n_codes,
            "d_model": self.d_model,
            "aggregation": "raw_additive",
            "code_embeddings": "frozen_bge_plus_linear_proj",
            "n_emb_codes": self.n_emb_codes,
        }
        if hasattr(self.model, "architecture_config"):
            card.update(self.model.architecture_config())
        return card

    def training_step(self, batch):
        self.model.train()
        logits = self.model(
            enc_code_ids=batch["enc_code_ids"],
            enc_code_mask=batch["enc_code_mask"],
            enc_tau=batch["enc_tau"],
            enc_padding_mask=batch["enc_padding_mask"],
            age=batch["age"],
        )
        # Clamp logits before BCE: overconfident ±inf/large values have aborted
        # ROCm kernels on this box during Stage-1 multilabel training.
        logits = logits.clamp(-20.0, 20.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["labels"]
        )
        return {"loss": loss}

    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=batch["age"],
            )
            return {"logits": logits.clamp(-20.0, 20.0)}

    def save_checkpoint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "checkpoint.pt")
        with (path / "config.json").open("w") as f:
            json.dump(self.model_card, f, indent=2)

    def load_checkpoint(self, path):
        path = Path(path)
        ckpt = path / "checkpoint.pt"
        if not ckpt.exists():
            ckpt = path / "best_checkpoint.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.load_state_dict(state)


def build_model(
    name: str,
    n_codes: int,
    arm: str = "age_temporal",
    *,
    name_suffix: str = "",
) -> Any:
    if name == "count_lightgbm":
        return LightGBMBaseline(n_codes=n_codes, n_targets=n_codes)
    elif name == "retain":
        return RETAINModel(n_codes=n_codes, n_targets=n_codes, **BASELINE_CONFIGS["retain"])
    elif name == "ehr_bert":
        return EHRBertModel(n_codes=n_codes, n_targets=n_codes, **{**BASELINE_CONFIGS["ehr_bert"], "max_seq_len": MAX_SEQ_LEN + 1})
    elif name == "behrt":
        return BEHRTModel(n_codes=n_codes, n_targets=n_codes, **{**BASELINE_CONFIGS["behrt"], "max_seq_len": MAX_SEQ_LEN + 1})
    elif name == "medbert":
        return MedBERTModel(n_codes=n_codes, n_targets=n_codes, **{**BASELINE_CONFIGS["medbert"], "max_seq_len": MAX_SEQ_LEN + 1})
    elif name == "cehrbert":
        return CEHRBertAdapter(n_codes=n_codes, n_targets=n_codes, **{**BASELINE_CONFIGS["cehrbert"], "max_seq_len": MAX_SEQ_LEN + 1})
    elif name == "motor":
        return MOTORModel(n_codes=n_codes, n_targets=n_codes, **{**BASELINE_CONFIGS["motor"], "max_seq_len": MAX_SEQ_LEN + 1})
    elif name == "tale_ehr":
        return TALEEHRModel(n_codes=n_codes, n_targets=n_codes, **{**BASELINE_CONFIGS["tale_ehr"], "max_seq_len": MAX_SEQ_LEN + 1})
    elif name == "nest":
        return NESTModel(n_codes=n_codes, n_targets=n_codes, **{**BASELINE_CONFIGS["nest"], "max_seq_len": MAX_SEQ_LEN + 1})
    elif name == "dtr":
        if uses_content_persistence(arm, name_suffix):
            return MIMICCPDTRAdapter(n_codes=n_codes, arm=arm, d_model=CP_D_MODEL)
        return MIMICDTRAdapter(n_codes=n_codes, arm=arm)
    else:
        raise ValueError(f"Unknown model: {name}")

def train_lightgbm(model, train_loader, val_loader, test_loader, n_codes):
    def extract(loader):
        Xs, Ys = [], []
        for batch in loader:
            X = build_features_from_batch(batch, n_codes)
            Xs.append(X)
            y = batch["labels"].numpy() if isinstance(batch["labels"], torch.Tensor) else batch["labels"]
            Ys.append(y)
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

    X_train, y_train = extract(train_loader)
    X_val, y_val = extract(val_loader)
    X_test, y_test = extract(test_loader)

    t0 = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - t0

    logits_test = model.predict_logits(X_test)
    test_metrics = multilabel_metrics(y_test, logits_test)
    logits_val = model.predict_logits(X_val)
    val_metrics = multilabel_metrics(y_val, logits_val)

    return {"val_metrics": val_metrics, "test_metrics": test_metrics, "train_time_s": train_time}

def run_one_model(
    model_name, output_dir, train_ds, val_ds, test_ds, batch_size, smoke, device,
    arm: str = "age_temporal",
    val_max_batches: int = VAL_MAX_BATCHES,
    test_max_batches: int = TEST_MAX_BATCHES,
    num_workers: int = NUM_WORKERS,
    name_suffix: str = "",
    grad_accum_steps: int = 1,
    seed: int = MODEL_SEED,
    amp: str = "fp32",
):
    run_name = (
        dtr_arm_dirname(arm, name_suffix) if model_name == "dtr" else f"{model_name}{name_suffix}"
    )
    # Hard guard: suffixed runs must never land in legacy dirs.
    if name_suffix and name_suffix not in run_name:
        raise RuntimeError(f"suffix {name_suffix!r} missing from run_name {run_name!r}")
    if name_suffix and run_name in ("dtr_age_temporal", "dtr_no_interaction"):
        raise RuntimeError(f"refusing to write legacy dir for suffixed run: {run_name}")

    cp = model_name == "dtr" and uses_content_persistence(arm, name_suffix)
    print(
        f"\n{'='*60}\nModel: {run_name} | MIMIC Stage-1 | seed={seed}"
        f" | arch={'Content-Persistence' if cp else 'MinimalDKM' if model_name == 'dtr' else model_name}"
        f"\n{'='*60}"
    )
    n_codes = train_ds.dataset.num_codes if hasattr(train_ds, "dataset") else train_ds.num_codes
    max_ep = 2 if smoke else MAX_EPOCHS
    min_ep = max_ep if smoke else MIN_EPOCHS
    # Smoke: evaluate the tiny subset fully.
    v_batches = None if smoke else val_max_batches
    t_batches = None if smoke else test_max_batches
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "result.json").exists() and not smoke:
        print(f"  Already trained, skipping {run_name}")
        with (run_dir / "result.json").open() as f:
            return json.load(f)

    set_seed(seed)
    base_collate = make_collate("one_hot")
    n_workers = 0 if smoke else num_workers
    if model_name == "nest" and batch_size > 32:
        print(f"  Clamping NEST batch_size from {batch_size} to 32 for encounter-multiset VRAM safety")
        batch_size = 32

    # BEHRT seed-0 was OOM-killed during full-bs test concat; finalize used bs=64.
    test_bs = batch_size if smoke else min(int(batch_size), 64)

    def mk_ldr(ds, shuf, *, bsz: int | None = None):
        # CP and NEST: encounter conversion runs inside workers via make_cp_collate so the
        # GPU is not stalled on Python grouping. pin_memory=True for H2D overlap.
        # prefetch_factor=2 keeps RAM bounded with 8 workers on a 32GB host.
        use_encounter = cp or model_name == "nest"
        if use_encounter:
            if model_name == "nest":
                nest_cfg = BASELINE_CONFIGS.get("nest", {})
                use_collate = make_cp_collate(
                    base_collate,
                    max_codes_per_encounter=nest_cfg.get("max_codes_per_encounter", 32),
                    max_encounters=nest_cfg.get("max_encounters", 32),
                )
            else:
                use_collate = make_cp_collate(base_collate)
        else:
            use_collate = base_collate
        use_bs = batch_size if bsz is None else bsz
        kw = dict(
            batch_size=use_bs, shuffle=shuf, collate_fn=use_collate,
            num_workers=n_workers, pin_memory=True,
            worker_init_fn=dataloader_worker_init,
        )
        if n_workers > 0:
            kw["persistent_workers"] = True
            # Large one-hot |V|≈30k labels: keep prefetch low to avoid host RAM spikes.
            kw["prefetch_factor"] = 1 if use_bs >= 256 else 2
        loader = DataLoader(ds, **kw)
        if use_encounter:
            return loader  # already encounter-formatted
        return RenameLoader(loader)

    train_loader = mk_ldr(train_ds, True)
    val_loader = mk_ldr(val_ds, False)
    test_loader = mk_ldr(test_ds, False, bsz=test_bs)

    result = {
        "model": run_name,
        "seed": seed,
        "smoke": smoke,
        "n_codes": n_codes,
        "batch_size": batch_size,
        "test_batch_size": test_bs,
        "val_max_batches": v_batches,
        "test_max_batches": t_batches,
        "name_suffix": name_suffix,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * max(1, grad_accum_steps),
        "amp": amp,
    }
    if model_name == "dtr":
        result["arm"] = arm
        result["architecture"] = (
            "Content-Persistence DTR" if cp else "MinimalDKM"
        )

    model = build_model(model_name, n_codes, arm=arm, name_suffix=name_suffix)
    if model_name == "count_lightgbm":
        lgb_result = train_lightgbm(model, train_loader, val_loader, test_loader, n_codes)
        result.update(lgb_result)
        result["model_card"] = model.model_card
        model.save_checkpoint(run_dir)
    else:
        model.to(get_device(device))
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            alloc = torch.cuda.memory_allocated() / (1024**3)
            print(f"  GPU mem after model load: {alloc:.2f} GiB", flush=True)
        result["param_counts"] = count_parameters(model) if isinstance(model, torch.nn.Module) else {}
        print(
            f"  batch_size={batch_size}  test_batch_size={test_bs}  accum={grad_accum_steps}  "
            f"effective_batch={batch_size * max(1, grad_accum_steps)}  "
            f"num_workers={n_workers}  amp={amp}  "
            f"val_max_batches={v_batches}  test_max_batches={t_batches}  "
            f"n_codes={n_codes}  train_examples={len(train_ds)}  seed={seed}"
        )
        train_result = train_neural_baseline(
            model=model, train_fn=model.training_step, predict_fn=model.predict,
            train_loader=train_loader, val_loader=val_loader,
            lr=LR, weight_decay=WEIGHT_DECAY, max_epochs=max_ep, patience=PATIENCE,
            min_epochs=min_ep, val_max_batches=v_batches,
            grad_clip=GRAD_CLIP, device=device, run_dir=run_dir, seed=seed,
            grad_accum_steps=grad_accum_steps, amp=amp,
        )
        # Drop train/val loaders before test concat to free worker + prefetch RAM.
        del train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"  starting capped test eval (bs={test_bs}, max_batches={t_batches})...",
            flush=True,
        )
        test_metrics = evaluate_loader(
            model, model.predict, test_loader, get_device(device),
            max_batches=t_batches,
            metrics_mode="full",
        )
        result["train"] = train_result
        result["test_metrics"] = test_metrics
        result["model_card"] = model.model_card
        # Persist config.json; weight files already written by train_neural_baseline
        # (best_checkpoint.pt, last_checkpoint.pt, checkpoint.pt=best).
        model.save_checkpoint(run_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Flatten key metrics for quick inspection
    if "test_metrics" in result:
        result["AUROC"] = result["test_metrics"].get("micro_auroc")
        result["AUPRC"] = result["test_metrics"].get("micro_auprc")
        result["BCE"] = result["test_metrics"].get("bce")
    if "train" in result:
        result["best_val_bce"] = result["train"].get("best_val_bce")
        result["epochs_trained"] = result["train"].get("epochs_trained")

    with (run_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)

    # history.json / best+last checkpoints already written by train_neural_baseline
    for fname in ("result.json", "history.json", "best_checkpoint.pt",
                  "last_checkpoint.pt", "checkpoint.pt"):
        if model_name != "count_lightgbm" and not (run_dir / fname).exists():
            if fname != "result.json":
                print(f"  WARNING: missing artifact {fname} in {run_dir}")

    auroc = result.get("test_metrics", {}).get("micro_auroc", "N/A")
    print(f"  Test AUROC={auroc}")
    print(f"  Artifacts → {run_dir}")
    # Free GPU memory before the next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all",
                        help="Comma-separated models or 'all'. "
                             "DTR expands to legacy Minimal-DKM arms "
                             "(age_temporal, no_interaction) unless "
                             "--name-suffix is set (then CP: temporal_only, "
                             "age_temporal).")
    parser.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Append to DTR arm dirs (e.g. '_new' → dtr_age_temporal_new). "
             "Enables Content-Persistence DTR and writes all_results{suffix}.json. "
             "Legacy dirs / all_results.json are never touched.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=MODEL_SEED,
                        help="Model init / training seed. seed=0 writes under "
                             "results/baselines/mimic/<model>/; seed!=0 under "
                             "results/baselines/mimic/seed_<seed>/<model>/.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--throughput_optimized",
        action="store_true",
        help="Per-model train batch sizes from THROUGHPUT_BATCH_SIZES + recommend "
             "pairing with --amp bf16. Overrides --batch_size for non-DTR arms.",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Autocast precision for neural train steps (bf16 ≈3–5× faster on R9700).",
    )
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="DataLoader workers (0 = main process only).")
    parser.add_argument("--val_max_batches", type=int, default=VAL_MAX_BATCHES,
                        help="Cap validation batches (avoids OOM on 30k-code heads).")
    parser.add_argument("--test_max_batches", type=int, default=TEST_MAX_BATCHES,
                        help="Cap test batches for reported metrics.")
    parser.add_argument("--tensorized_dir", default="data/processed/tensorized_flat")
    parser.add_argument("--vocab_path", default="data/processed/code_vocab.json")
    args = parser.parse_args()

    name_suffix = args.name_suffix
    seed = int(args.seed)
    output_dir = REPO_ROOT / "results" / "baselines" / "mimic"
    if args.smoke:
        output_dir = output_dir / "_smoke"
    # Keep seed-0 layout stable; isolate other seeds so result.json skip-guards
    # do not clobber or skip the canonical seed-0 runs.
    elif seed != 0:
        output_dir = output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tdir = REPO_ROOT / args.tensorized_dir
    if args.smoke:
        from stage1_mimic_pretrain.train import maybe_restrict_tensorized
        tdir = maybe_restrict_tensorized(tdir, output_dir / "data_subset", max_shards=1, seed=0)

    train_ds = TensorizedPretrainDataset(tdir / "train", REPO_ROOT / args.vocab_path, max_seq_len=MAX_SEQ_LEN)
    val_ds = TensorizedPretrainDataset(tdir / "val", REPO_ROOT / args.vocab_path, max_seq_len=MAX_SEQ_LEN)
    test_ds = TensorizedPretrainDataset(tdir / "test", REPO_ROOT / args.vocab_path, max_seq_len=MAX_SEQ_LEN)

    # Subsample for smoke testing
    if args.smoke:
        def subset(ds):
            return torch.utils.data.Subset(ds, range(min(64, len(ds))))
        train_ds = subset(train_ds)
        val_ds = subset(val_ds)
        test_ds = subset(test_ds)

    dtr_arms = DTR_ARMS_CP if name_suffix else DTR_ARMS_LEGACY

    # DTR first, then other baselines. DTR expands to ablation arms.
    if args.models == "all":
        model_specs = [("dtr", arm) for arm in dtr_arms] + [
            ("retain", "age_temporal"),
            ("ehr_bert", "age_temporal"),
            ("behrt", "age_temporal"),
            ("medbert", "age_temporal"),
            ("cehrbert", "age_temporal"),
            ("motor", "age_temporal"),
            ("tale_ehr", "age_temporal"),
            ("nest", "age_temporal"),
        ]
    else:
        model_specs = []
        for m in args.models.split(","):
            m = m.strip()
            if m == "dtr":
                for arm in dtr_arms:
                    model_specs.append(("dtr", arm))
            elif m.startswith("dtr_"):
                arm = m.replace("dtr_", "", 1)
                if name_suffix and arm.endswith(name_suffix):
                    arm = arm[: -len(name_suffix)]
                model_specs.append(("dtr", arm))
            else:
                model_specs.append((m, "age_temporal"))

    all_results = {}
    # Content-Persistence ``_new``: batch 32, accum 1 (Stage-1 effective batch).
    # Legacy / non-DTR keep the requested batch_size with accum=1.
    cp_bsz = (8 if args.smoke else CP_MICRO_BATCH) if name_suffix else None
    cp_accum = (1 if args.smoke else CP_GRAD_ACCUM) if name_suffix else 1
    bsz = 8 if args.smoke else args.batch_size
    # Honor CLI --num_workers always (default NUM_WORKERS=4; launch with 8 for CP).
    amp = "fp32" if args.smoke else args.amp
    if args.throughput_optimized and not args.smoke:
        print(
            f"  throughput_optimized=1  amp={amp}  "
            f"per-model batches={ {k: v for k, v in THROUGHPUT_BATCH_SIZES.items() if k != 'dtr'} }",
            flush=True,
        )
    if not args.smoke and args.num_workers > 0:
        configure_dataloader_multiprocessing(args.num_workers)
    elif args.num_workers == 0:
        print(
            "  num_workers=0 (main-process DataLoader; avoids CUDA+fork stalls)",
            flush=True,
        )

    for m, arm in model_specs:
        key = dtr_arm_dirname(arm, name_suffix) if m == "dtr" else f"{m}{name_suffix}"
        use_cp = m == "dtr" and uses_content_persistence(arm, name_suffix)
        if use_cp and cp_bsz is not None:
            run_bsz = cp_bsz
        elif args.throughput_optimized and not args.smoke:
            run_bsz = THROUGHPUT_BATCH_SIZES.get(m, bsz)
        else:
            run_bsz = bsz
        run_accum = cp_accum if use_cp else 1
        run_workers = args.num_workers
        try:
            res = run_one_model(
                m, output_dir, train_ds, val_ds, test_ds,
                run_bsz, args.smoke, args.device, arm=arm,
                val_max_batches=args.val_max_batches,
                test_max_batches=args.test_max_batches,
                num_workers=run_workers,
                name_suffix=name_suffix,
                grad_accum_steps=run_accum,
                seed=seed,
                amp=amp,
            )
            all_results[key] = res
        except Exception as e:
            print(f"ERROR on {key}: {e}")
            import traceback
            traceback.print_exc()
            all_results[key] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_name = f"all_results{name_suffix}.json" if name_suffix else "all_results.json"
    combined_path = output_dir / combined_name
    with combined_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nMIMIC results saved to {combined_path}")
    print(f"seed={seed}  output_dir={output_dir}")
    if name_suffix:
        print(f"DTR arm dirs use suffix {name_suffix!r} (legacy dirs untouched)")

if __name__ == "__main__":
    main()
