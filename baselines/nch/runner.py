#!/usr/bin/env python3
"""Stage-2 NCH baseline runner — finetune MIMIC Stage-1 checkpoints on NCH.

Usage:
    python -m baselines.nch.runner --models all
    python -m baselines.nch.runner --models dtr_age_temporal,retain,ehr_bert
    python -m baselines.nch.runner --models dtr --name-suffix _new
    python -m baselines.nch.runner --models retain,behrt --seed 1 --amp bf16 \\
        --throughput_optimized

Loads Stage-1 weights from results/baselines/mimic/<name>/ (seed 0) or
    results/baselines/mimic/seed_<seed>/<name>/ (seed ≠ 0; aliases:
    retain → retain-backup, dtr → dtr_age_temporal). Writes under
    results/baselines/nch/<name>/ (seed 0) or
    results/baselines/nch/seed_<seed>/<name>/.

Corrected Content-Persistence DTR (``--name-suffix _new``):
  arms temporal_only / age_temporal under ``nch`` dirs
  ``dtr_temporal_only_new``, ``dtr_age_temporal_new``.
  Both arms initialize from the same MIMIC Stage-1 checkpoint
  ``results/baselines/mimic/dtr_temporal_only_new/`` with β=0;
  TO keeps β fixed, AT learns β. Legacy dirs / all_results.json untouched.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stage2_nch.dataset import (
    NCHForecastDataset as TensorizedPretrainDataset,
    make_nch_collate as make_collate,
)
from model_new.data import dataloader_worker_init

from baselines.common.metrics import multilabel_metrics
from baselines.common.training import (
    evaluate_loader, set_seed, get_device, train_neural_baseline,
)
from baselines.common.capacity_report import count_parameters

from baselines.lightgbm.model import LightGBMBaseline, build_features_from_batch
from baselines.retain.model import RETAINModel
from baselines.ehr_bert.model import EHRBertModel
from baselines.behrt.model import BEHRTModel
from baselines.medbert.model import MedBERTModel
from baselines.cehrbert_adapter.adapter import CEHRBertAdapter
from baselines.motor.model import MOTORModel
from baselines.tale_ehr.model import TALEEHRModel
from baselines.nest.model import NESTModel
from baselines.mimic.runner import (  # CP DTR shared with Stage-1
    CP_D_MODEL,
    CP_MICRO_BATCH,
    MIMICCPDTRAdapter,
    THROUGHPUT_BATCH_SIZES,
    make_cp_collate,
)

# NCH Stage-2 defaults
BATCH_SIZE = 128
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
MIN_EPOCHS = 5
GRAD_CLIP = 1.0
VAL_MAX_BATCHES = 50
TEST_MAX_BATCHES = 100
# Default 0: after CUDA init, forking DataLoader workers (PyTorch default
# start method) copies the parent address space (~multi-GB model + ROCm
# context). Empirically that caused 15–37 min silent stalls on the first
# train batch of each new arm, and growing per-epoch first-batch times
# when workers were non-persistent. Main-process loading is fast enough
# here (steady GPU steps dominate). If num_workers > 0, we force spawn +
# persistent_workers (see configure_dataloader_multiprocessing).
NUM_WORKERS = 0

# Legacy Minimal-DKM ablation arms (unsuffixed dirs).
DTR_ARMS_LEGACY = ("age_temporal", "no_interaction")
# Content-Persistence DTR ablation pair (``_new`` dirs).
DTR_ARMS_CP = ("temporal_only", "age_temporal")
DTR_ARMS = DTR_ARMS_LEGACY

# Shared Stage-1 CP init for BOTH NCH ``_new`` arms (not MIMIC age_temporal).
SHARED_CP_STAGE1_DIR = (
    REPO_ROOT / "results" / "baselines" / "mimic" / "dtr_temporal_only_new"
)

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

# Stage-1 checkpoint directory aliases under results/baselines/mimic/
MIMIC_CKPT_ALIASES: dict[str, tuple[str, ...]] = {
    "retain": ("retain", "retain-backup"),
    "dtr_age_temporal": ("dtr_age_temporal", "dtr"),
    "dtr_no_interaction": ("dtr_no_interaction",),
    "ehr_bert": ("ehr_bert",),
    "behrt": ("behrt",),
    "medbert": ("medbert",),
    "cehrbert": ("cehrbert",),
    "motor": ("motor",),
    "tale_ehr": ("tale_ehr",),
    "nest": ("nest",),
}


def dtr_arm_dirname(arm: str, name_suffix: str = "") -> str:
    return f"dtr_{arm}{name_suffix}"


def uses_content_persistence(arm: str, name_suffix: str = "") -> bool:
    """Only suffixed ``_new`` DTR runs use Content-Persistence (legacy untouched)."""
    return bool(name_suffix) and arm in DTR_ARMS_CP


def fingerprint_excluding_beta(model: torch.nn.Module) -> str:
    """SHA256 over state_dict excluding β (arms may differ only in β requires_grad)."""
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        if key.endswith("beta") or key == "beta":
            continue
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def apply_shared_cp_init(model: MIMICCPDTRAdapter, ckpt_dir: Path) -> dict[str, Any]:
    """Load shared MIMIC TO weights, force β=0, set arm-specific β trainability."""
    model.load_checkpoint(ckpt_dir)
    with torch.no_grad():
        model.model.beta.fill_(0.0)
    if model.arm == "temporal_only":
        model.model.beta.requires_grad_(False)
    else:
        model.model.beta.requires_grad_(True)
    return {
        "pretrained_from": str(ckpt_dir.relative_to(REPO_ROOT)),
        "beta_init": float(model.model.beta.detach().cpu()),
        "beta_requires_grad": bool(model.model.beta.requires_grad),
        "theta0_init": float(model.model.theta0.detach().cpu()),
        "pretrained_fingerprint_ex_beta": fingerprint_excluding_beta(model),
        "age_temporal": bool(model.model.age_temporal),
    }


def verify_identical_cp_init(n_codes: int, ckpt_dir: Path, out_path: Path) -> dict[str, Any]:
    """Build both CP arms, load the same ckpt, assert shared weights match before unfreeze."""
    to = MIMICCPDTRAdapter(n_codes=n_codes, arm="temporal_only", d_model=CP_D_MODEL)
    at = MIMICCPDTRAdapter(n_codes=n_codes, arm="age_temporal", d_model=CP_D_MODEL)
    meta_to = apply_shared_cp_init(to, ckpt_dir)
    meta_at = apply_shared_cp_init(at, ckpt_dir)
    fp_to = meta_to["pretrained_fingerprint_ex_beta"]
    fp_at = meta_at["pretrained_fingerprint_ex_beta"]
    match = fp_to == fp_at
    report = {
        "shared_ckpt": str(ckpt_dir.relative_to(REPO_ROOT)),
        "fingerprint_temporal_only": fp_to,
        "fingerprint_age_temporal": fp_at,
        "fingerprints_match": match,
        "beta_to": meta_to["beta_init"],
        "beta_at": meta_at["beta_init"],
        "theta0_to": meta_to["theta0_init"],
        "theta0_at": meta_at["theta0_init"],
        "beta_requires_grad_to": meta_to["beta_requires_grad"],
        "beta_requires_grad_at": meta_at["beta_requires_grad"],
    }
    if not match:
        raise RuntimeError(
            "CP Stage-2 init fingerprints differ between arms before β unfreeze: "
            f"TO={fp_to} AT={fp_at}"
        )
    if abs(meta_to["beta_init"]) > 1e-12 or abs(meta_at["beta_init"]) > 1e-12:
        raise RuntimeError("CP Stage-2 requires β=0 at initialization for both arms")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    del to, at
    return report


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
                batch["is_query"] = torch.zeros_like(
                    batch["attention_mask"], dtype=torch.bool
                )
            if "timestamps_days" in batch and "tau" not in batch:
                batch["tau"] = batch["timestamps_days"].float()
            if "age_years" in batch and "age" not in batch:
                batch["age"] = batch["age_years"].max(dim=1).values.float()
            elif "last_age_years" in batch and "age" not in batch:
                batch["age"] = batch["last_age_years"].float()
            yield batch

    def __len__(self):
        return len(self.loader)

    def shutdown(self) -> None:
        """Drop the underlying iterator so worker processes exit promptly."""
        loader = self.loader
        it = getattr(loader, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
            try:
                loader._iterator = None
            except Exception:
                pass


def configure_dataloader_multiprocessing(num_workers: int) -> None:
    """Avoid CUDA+fork stalls when multi-process loaders are requested.

    Must run before CUDA init / first DataLoader with workers > 0.
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


def safe_shutdown_loader(loader) -> None:
    """Shut down RenameLoader workers if present; no-op for plain DataLoaders."""
    if loader is None:
        return
    fn = getattr(loader, "shutdown", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def release_cuda_between_arms() -> None:
    """Reclaim host/GPU memory so the next arm does not inherit a bloated RSS."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()


class NCHDTRAdapter(torch.nn.Module):
    """Minimal-DKM (DTR) wrapper for NCH Stage-2 finetuning."""

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
        return EHRBertModel(
            n_codes=n_codes, n_targets=n_codes,
            **{**BASELINE_CONFIGS["ehr_bert"], "max_seq_len": MAX_SEQ_LEN + 1},
        )
    elif name == "behrt":
        return BEHRTModel(
            n_codes=n_codes, n_targets=n_codes,
            **{**BASELINE_CONFIGS["behrt"], "max_seq_len": MAX_SEQ_LEN + 1},
        )
    elif name == "medbert":
        return MedBERTModel(
            n_codes=n_codes, n_targets=n_codes,
            **{**BASELINE_CONFIGS["medbert"], "max_seq_len": MAX_SEQ_LEN + 1},
        )
    elif name == "cehrbert":
        return CEHRBertAdapter(
            n_codes=n_codes, n_targets=n_codes,
            **{**BASELINE_CONFIGS["cehrbert"], "max_seq_len": MAX_SEQ_LEN + 1},
        )
    elif name == "motor":
        return MOTORModel(
            n_codes=n_codes, n_targets=n_codes,
            **{**BASELINE_CONFIGS["motor"], "max_seq_len": MAX_SEQ_LEN + 1},
        )
    elif name == "tale_ehr":
        return TALEEHRModel(
            n_codes=n_codes, n_targets=n_codes,
            **{**BASELINE_CONFIGS["tale_ehr"], "max_seq_len": MAX_SEQ_LEN + 1},
        )
    elif name == "nest":
        return NESTModel(
            n_codes=n_codes, n_targets=n_codes,
            **{**BASELINE_CONFIGS["nest"], "max_seq_len": MAX_SEQ_LEN + 1},
        )
    elif name == "dtr":
        if uses_content_persistence(arm, name_suffix):
            return MIMICCPDTRAdapter(n_codes=n_codes, arm=arm, d_model=CP_D_MODEL)
        return NCHDTRAdapter(n_codes=n_codes, arm=arm)
    else:
        raise ValueError(f"Unknown model: {name}")


def resolve_mimic_ckpt_dir(
    run_name: str, *, cp: bool = False, seed: int = 0,
) -> Path | None:
    """Find Stage-1 MIMIC checkpoint dir for a Stage-2 run name.

    Content-Persistence ``_new`` arms always load the shared temporal_only_new
    checkpoint (never MIMIC age_temporal_new).

    Non-CP: seed 0 → ``results/baselines/mimic/<alias>/``;
    seed ≠ 0 → ``results/baselines/mimic/seed_<seed>/<alias>/`` (no fallback
    to seed 0 — wrong init would silently corrupt multi-seed Stage-2).
    """
    if cp:
        if (SHARED_CP_STAGE1_DIR / "best_checkpoint.pt").exists() or (
            SHARED_CP_STAGE1_DIR / "checkpoint.pt"
        ).exists():
            return SHARED_CP_STAGE1_DIR
        return None
    mimic_root = REPO_ROOT / "results" / "baselines" / "mimic"
    if int(seed) != 0:
        mimic_root = mimic_root / f"seed_{int(seed)}"
    aliases = MIMIC_CKPT_ALIASES.get(run_name, (run_name,))
    for alias in aliases:
        cand = mimic_root / alias
        if (cand / "checkpoint.pt").exists() or (cand / "best_checkpoint.pt").exists():
            return cand
    return None


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
    require_pretrained: bool = True,
    name_suffix: str = "",
    seed: int = MODEL_SEED,
    amp: str = "fp32",
):
    run_name = (
        dtr_arm_dirname(arm, name_suffix) if model_name == "dtr" else f"{model_name}{name_suffix}"
    )
    if name_suffix and name_suffix not in run_name:
        raise RuntimeError(f"suffix {name_suffix!r} missing from run_name {run_name!r}")
    if name_suffix and run_name in ("dtr_age_temporal", "dtr_no_interaction"):
        raise RuntimeError(f"refusing to write legacy dir for suffixed run: {run_name}")

    cp = model_name == "dtr" and uses_content_persistence(arm, name_suffix)
    print(
        f"\n{'='*60}\nModel: {run_name} | NCH Stage-2 | seed={seed}"
        f" | arch={'Content-Persistence' if cp else 'MinimalDKM' if model_name == 'dtr' else model_name}"
        f"\n{'='*60}"
    )
    n_codes = train_ds.dataset.num_codes if hasattr(train_ds, "dataset") else train_ds.num_codes
    max_ep = 2 if smoke else MAX_EPOCHS
    min_ep = max_ep if smoke else MIN_EPOCHS
    v_batches = None if smoke else val_max_batches
    t_batches = None if smoke else test_max_batches
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "result.json").exists() and not smoke:
        # Skip only finished non-smoke runs
        try:
            with (run_dir / "result.json").open() as f:
                prev = json.load(f)
            if not prev.get("smoke", False):
                print(f"  Already trained, skipping {run_name}")
                return prev
            print(f"  Found smoke result for {run_name}; re-running full Stage-2")
        except Exception:
            pass

    if model_name == "nest" and batch_size > 32:
        print(f"  Clamping NEST batch_size from {batch_size} to 32 for encounter-multiset VRAM safety")
        batch_size = 32

    set_seed(seed)
    base_collate = make_collate("one_hot")
    use_encounter = cp or model_name == "nest"
    if use_encounter:
        if model_name == "nest":
            nest_cfg = BASELINE_CONFIGS.get("nest", {})
            collate = make_cp_collate(
                base_collate,
                max_codes_per_encounter=nest_cfg.get("max_codes_per_encounter", 32),
                max_encounters=nest_cfg.get("max_encounters", 32),
            )
        else:
            collate = make_cp_collate(base_collate)
    else:
        collate = base_collate
    n_workers = 0 if smoke else num_workers
    # Cap test micro-batch like MIMIC (BEHRT host OOM on full-bs logit concat).
    test_bs = batch_size if smoke else min(int(batch_size), 64)
    if n_workers > 0:
        configure_dataloader_multiprocessing(n_workers)

    def mk_ldr(
        ds, shuf, *, workers: int | None = None, persistent: bool | None = None,
        bsz: int | None = None,
    ):
        # Val/test always workers=0 (main process). Train uses n_workers; when
        # workers > 0 we keep persistent_workers so they are not re-forked
        # every epoch after CUDA is live (that caused 1–5+ min/epoch stalls).
        nw = n_workers if workers is None else workers
        use_bs = batch_size if bsz is None else bsz
        if persistent is None:
            persistent = nw > 0
        kw: dict[str, Any] = dict(
            batch_size=use_bs, shuffle=shuf, collate_fn=collate,
            num_workers=nw, pin_memory=(nw > 0),
        )
        if nw > 0:
            kw["worker_init_fn"] = dataloader_worker_init
            if persistent:
                kw["persistent_workers"] = True
                kw["prefetch_factor"] = 1 if use_bs >= 256 else 2
        loader = DataLoader(ds, **kw)
        if use_encounter:
            return loader  # already encounter-formatted via make_cp_collate
        return RenameLoader(loader)

    train_loader = mk_ldr(train_ds, True)
    # Val on main process only — never fork after CUDA init.
    val_loader = mk_ldr(val_ds, False, workers=0)
    # Test loader built after training.

    result = {
        "model": run_name,
        "seed": seed,
        "smoke": smoke,
        "n_codes": n_codes,
        "batch_size": batch_size,
        "test_batch_size": test_bs,
        "val_max_batches": v_batches,
        "test_max_batches": t_batches,
        "stage": "nch_stage2",
        "name_suffix": name_suffix,
        "amp": amp,
    }
    if model_name == "dtr":
        result["arm"] = arm
        result["architecture"] = (
            "Content-Persistence DTR" if cp else "MinimalDKM"
        )

    model = build_model(model_name, n_codes, arm=arm, name_suffix=name_suffix)

    # Load Stage-1 MIMIC checkpoint (matched seed)
    mimic_ckpt_dir = resolve_mimic_ckpt_dir(run_name, cp=cp, seed=seed)
    if mimic_ckpt_dir is not None and hasattr(model, "load_checkpoint"):
        print(f"  Loading Stage-1 MIMIC checkpoint from {mimic_ckpt_dir}")
        if cp:
            init_meta = apply_shared_cp_init(model, mimic_ckpt_dir)
            result.update(init_meta)
            print(
                f"  CP init: β={init_meta['beta_init']:.6g} "
                f"requires_grad={init_meta['beta_requires_grad']} "
                f"θ0={init_meta['theta0_init']:.6g} "
                f"fp={init_meta['pretrained_fingerprint_ex_beta'][:12]}…",
                flush=True,
            )
        else:
            model.load_checkpoint(mimic_ckpt_dir)
            result["pretrained_from"] = str(mimic_ckpt_dir.relative_to(REPO_ROOT))
    elif model_name != "count_lightgbm":
        msg = (
            f"No MIMIC checkpoint found for {run_name} "
            f"(cp={cp}, seed={seed}, aliases={MIMIC_CKPT_ALIASES.get(run_name)})"
        )
        if require_pretrained and not smoke:
            raise FileNotFoundError(msg)
        print(f"  WARNING: {msg}; training from scratch!")
        result["pretrained_from"] = None

    if model_name == "count_lightgbm":
        test_loader = mk_ldr(test_ds, False, workers=0, bsz=test_bs)
        lgb_result = train_lightgbm(model, train_loader, val_loader, test_loader, n_codes)
        result.update(lgb_result)
        result["model_card"] = model.model_card
        model.save_checkpoint(run_dir)
    else:
        model.to(get_device(device))
        result["param_counts"] = count_parameters(model) if isinstance(model, torch.nn.Module) else {}
        print(
            f"  batch_size={batch_size}  test_batch_size={test_bs}  "
            f"num_workers={n_workers}  amp={amp}  "
            f"val_max_batches={v_batches}  test_max_batches={t_batches}  "
            f"n_codes={n_codes}  train_examples={len(train_ds)}  seed={seed}"
        )
        # Warmup on the *same* loading path as training (workers=0 always here)
        # so kernel init is not charged to logged batch-1 wall time.
        try:
            warm_loader = mk_ldr(train_ds, False, workers=0)
            warm_batch = next(iter(warm_loader))
            warm_batch = {
                k: v.to(get_device(device)) if torch.is_tensor(v) else v
                for k, v in warm_batch.items()
            }
            print("  CUDA warmup (1 train step)...", flush=True)
            tw0 = time.time()
            model.train()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(amp == "bf16" and str(device).startswith("cuda")),
            ):
                loss = model.training_step(warm_batch)["loss"]
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"  CUDA warmup done in {time.time() - tw0:.1f}s", flush=True)
            safe_shutdown_loader(warm_loader)
            del warm_loader, warm_batch, loss
        except Exception as e:
            print(f"  WARNING: CUDA warmup skipped ({e})", flush=True)

        train_result = train_neural_baseline(
            model=model, train_fn=model.training_step, predict_fn=model.predict,
            train_loader=train_loader, val_loader=val_loader,
            lr=LR, weight_decay=WEIGHT_DECAY, max_epochs=max_ep, patience=PATIENCE,
            min_epochs=min_ep, val_max_batches=v_batches,
            grad_clip=GRAD_CLIP, device=device, run_dir=run_dir, seed=seed,
            amp=amp,
        )
        # Shut down train workers before building test / next arm.
        safe_shutdown_loader(train_loader)
        safe_shutdown_loader(val_loader)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"  Running capped test eval (bs={test_bs}, max_batches={t_batches})...",
            flush=True,
        )
        test_loader = mk_ldr(test_ds, False, workers=0, bsz=test_bs)
        test_metrics = evaluate_loader(
            model, model.predict, test_loader, get_device(device),
            max_batches=t_batches,
            metrics_mode="full",  # micro AUROC subsampled internally at 5M cells
        )
        safe_shutdown_loader(test_loader)
        result["train"] = train_result
        result["test_metrics"] = test_metrics
        result["model_card"] = model.model_card
        model.save_checkpoint(run_dir)

    if "test_metrics" in result:
        result["AUROC"] = result["test_metrics"].get("micro_auroc")
        result["AUPRC"] = result["test_metrics"].get("micro_auprc")
        result["BCE"] = result["test_metrics"].get("bce")
    if "train" in result:
        result["best_val_bce"] = result["train"].get("best_val_bce")
        result["epochs_trained"] = result["train"].get("epochs_trained")

    with (run_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)

    auroc = result.get("test_metrics", {}).get("micro_auroc", "N/A")
    print(f"  Test AUROC={auroc}")
    print(f"  Artifacts → {run_dir}")
    del model, train_loader, val_loader
    release_cuda_between_arms()
    return result


def parse_model_specs(models_arg: str, *, name_suffix: str = "") -> list[tuple[str, str]]:
    dtr_arms = DTR_ARMS_CP if name_suffix else DTR_ARMS_LEGACY
    if models_arg == "all":
        return [("dtr", arm) for arm in dtr_arms] + [
            ("retain", "age_temporal"),
            ("ehr_bert", "age_temporal"),
            ("behrt", "age_temporal"),
            ("medbert", "age_temporal"),
            ("cehrbert", "age_temporal"),
            ("motor", "age_temporal"),
            ("tale_ehr", "age_temporal"),
            ("nest", "age_temporal"),
        ]
    specs = []
    for m in models_arg.split(","):
        m = m.strip()
        if not m:
            continue
        if m == "dtr":
            for arm in dtr_arms:
                specs.append(("dtr", arm))
        elif m.startswith("dtr_"):
            arm = m.replace("dtr_", "", 1)
            if name_suffix and arm.endswith(name_suffix):
                arm = arm[: -len(name_suffix)]
            specs.append(("dtr", arm))
        else:
            specs.append((m, "age_temporal"))
    return specs


def main():
    parser = argparse.ArgumentParser(description="NCH Stage-2 baseline finetuning")
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
             "Legacy dirs / all_results.json are never touched. "
             "Both CP arms init from mimic/dtr_temporal_only_new with β=0.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=MODEL_SEED,
        help="Model init / training seed. seed=0 writes under "
             "results/baselines/nch/<model>/ and loads MIMIC seed-0 ckpts; "
             "seed!=0 under results/baselines/nch/seed_<seed>/ and loads "
             "results/baselines/mimic/seed_<seed>/<model>/.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--throughput_optimized",
        action="store_true",
        help="Per-model train batch sizes from MIMIC THROUGHPUT_BATCH_SIZES. "
             "Pair with --amp bf16. Overrides --batch_size for non-DTR arms.",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help="Autocast precision for neural train steps (bf16 recommended).",
    )
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--val_max_batches", type=int, default=VAL_MAX_BATCHES)
    parser.add_argument("--test_max_batches", type=int, default=TEST_MAX_BATCHES)
    parser.add_argument(
        "--tensorized_dir",
        default="artifacts/nch_stage2/v2/tensorized_forecast/diagnoses_only",
    )
    parser.add_argument("--vocab_path", default="data/processed/code_vocab.json")
    parser.add_argument(
        "--allow_scratch", action="store_true",
        help="Allow training without a Stage-1 MIMIC checkpoint.",
    )
    parser.add_argument(
        "--clear_smoke", action="store_true",
        help="Move prior smoke artifacts under results/baselines/nch/_smoke_archive/.",
    )
    args = parser.parse_args()

    name_suffix = args.name_suffix
    seed = int(args.seed)
    output_dir = REPO_ROOT / "results" / "baselines" / "nch"
    if args.smoke:
        output_dir = output_dir / "_smoke"
    elif seed != 0:
        output_dir = output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_smoke and not args.smoke:
        archive = output_dir / "_smoke_archive"
        archive.mkdir(parents=True, exist_ok=True)
        for child in list(output_dir.iterdir()):
            if child.name.startswith("_"):
                continue
            rj = child / "result.json" if child.is_dir() else None
            if rj and rj.exists():
                try:
                    prev = json.loads(rj.read_text())
                except Exception:
                    prev = {}
                if prev.get("smoke"):
                    dest = archive / child.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(child), str(dest))
                    print(f"Archived smoke run → {dest}")

    tdir = REPO_ROOT / args.tensorized_dir
    if not tdir.exists():
        raise FileNotFoundError(f"NCH tensorized dir missing: {tdir}")

    if args.smoke:
        from stage1_mimic_pretrain.train import maybe_restrict_tensorized
        tdir = maybe_restrict_tensorized(
            tdir, output_dir / "data_subset", max_shards=1, seed=0
        )

    train_ds = TensorizedPretrainDataset(
        tdir / "train", REPO_ROOT / args.vocab_path, max_seq_len=MAX_SEQ_LEN
    )
    val_ds = TensorizedPretrainDataset(
        tdir / "val", REPO_ROOT / args.vocab_path, max_seq_len=MAX_SEQ_LEN
    )
    test_ds = TensorizedPretrainDataset(
        tdir / "test", REPO_ROOT / args.vocab_path, max_seq_len=MAX_SEQ_LEN
    )

    if args.smoke:
        def subset(ds):
            return torch.utils.data.Subset(ds, range(min(64, len(ds))))
        train_ds = subset(train_ds)
        val_ds = subset(val_ds)
        test_ds = subset(test_ds)

    model_specs = parse_model_specs(args.models, name_suffix=name_suffix)
    all_results: dict[str, Any] = {}
    # CP ``_new``: same micro-batch as MIMIC Stage-1 CP (32). Legacy keeps CLI bsz.
    bsz = 8 if args.smoke else args.batch_size
    cp_bsz = (8 if args.smoke else CP_MICRO_BATCH) if name_suffix else None
    amp = "fp32" if args.smoke else args.amp
    if args.throughput_optimized and not args.smoke:
        print(
            f"  throughput_optimized=1  amp={amp}  seed={seed}  "
            f"per-model batches={ {k: v for k, v in THROUGHPUT_BATCH_SIZES.items() if k != 'dtr'} }",
            flush=True,
        )

    # Configure spawn before any CUDA work if multi-worker loaders are used.
    if not args.smoke and args.num_workers > 0:
        configure_dataloader_multiprocessing(args.num_workers)
    elif args.num_workers == 0:
        print(
            "  num_workers=0 (main-process DataLoader; avoids CUDA+fork stalls)",
            flush=True,
        )

    # Verify identical shared init before any CP arm trains.
    if name_suffix and any(m == "dtr" for m, _ in model_specs):
        n_codes = (
            train_ds.dataset.num_codes
            if hasattr(train_ds, "dataset")
            else train_ds.num_codes
        )
        verify_path = output_dir / f"dtr{name_suffix}_init_verify.json"
        print("  Verifying identical CP pretrained weights (ex-β) across arms…", flush=True)
        report = verify_identical_cp_init(n_codes, SHARED_CP_STAGE1_DIR, verify_path)
        print(
            f"  Init verify OK: fingerprints_match={report['fingerprints_match']} "
            f"β_to={report['beta_to']} β_at={report['beta_at']} → {verify_path}",
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
        try:
            res = run_one_model(
                m, output_dir, train_ds, val_ds, test_ds,
                run_bsz, args.smoke, args.device, arm=arm,
                val_max_batches=args.val_max_batches,
                test_max_batches=args.test_max_batches,
                num_workers=args.num_workers,
                require_pretrained=not args.allow_scratch,
                name_suffix=name_suffix,
                seed=seed,
                amp=amp,
            )
            all_results[key] = res
        except Exception as e:
            print(f"ERROR on {key}: {e}")
            import traceback
            traceback.print_exc()
            all_results[key] = {"error": str(e)}
        release_cuda_between_arms()

    combined_name = f"all_results{name_suffix}.json" if name_suffix else "all_results.json"
    combined_path = output_dir / combined_name
    with combined_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nNCH Stage-2 results saved to {combined_path}")
    print(f"seed={seed}  output_dir={output_dir}")
    if name_suffix:
        print(f"DTR arm dirs use suffix {name_suffix!r} (legacy dirs untouched)")


if __name__ == "__main__":
    main()
