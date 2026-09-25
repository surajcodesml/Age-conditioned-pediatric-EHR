#!/usr/bin/env python3
"""Stage-1 MIMIC baseline runner — train and evaluate all baselines on MIMIC-IV next-visit prediction.

Usage:
    python -m baselines.mimic.runner --models all
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

# Import all baselines
from baselines.lightgbm.model import LightGBMBaseline, build_features_from_batch
from baselines.retain.model import RETAINModel
from baselines.ehr_bert.model import EHRBertModel
from baselines.behrt.model import BEHRTModel
from baselines.medbert.model import MedBERTModel
from baselines.cehrbert_adapter.adapter import CEHRBertAdapter
from baselines.dtr_adapter.adapter import DTRAdapter

# MIMIC defaults
BATCH_SIZE = 32
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
VAL_MAX_BATCHES = 50
TEST_MAX_BATCHES = 100

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

class DTRAdapter(torch.nn.Module):
    def __init__(self, n_codes, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        from stage1_mimic_pretrain.model import MinimalDKMModel
        from model_new.data import demo_layout
        demo_dim, demo_channels = demo_layout("one_hot")
        self.model = MinimalDKMModel(
            num_codes=n_codes, arm="age_temporal", d_model=d_model, n_layers=n_layers, 
            n_heads=n_heads, use_residual=True, use_layernorm=True, use_ffn=True,
            demo_dim=demo_dim, demo_channels=demo_channels, race_encoding="one_hot", demo_hidden=64,
            embedding_path=REPO_ROOT / "data/processed/bge_embeddings.pt"
        )
    @property
    def model_card(self): return {"name": "dtr_mimic"}
    def training_step(self, batch):
        self.model.train()
        out = self.model(batch)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out["code_logits"], batch["labels"])
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
            json.dump({"name": "dtr_mimic", "n_codes": getattr(self, "n_codes", None)}, f, indent=2, default=str)

    def load_checkpoint(self, path):
        path = Path(path)
        ckpt = path / "checkpoint.pt"
        if not ckpt.exists():
            ckpt = path / "best_checkpoint.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.load_state_dict(state)

def build_model(name: str, n_codes: int) -> Any:
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
    elif name == "dtr":
        return DTRAdapter(n_codes=n_codes)
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

def run_one_model(model_name, output_dir, train_ds, val_ds, test_ds, batch_size, smoke, device):
    print(f"\n{'='*60}\nModel: {model_name} | NCH Stage-2\n{'='*60}")
    n_codes = train_ds.dataset.num_codes if hasattr(train_ds, "dataset") else train_ds.num_codes
    max_ep = 2 if smoke else MAX_EPOCHS
    min_ep = max_ep if smoke else MIN_EPOCHS
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "result.json").exists() and not smoke:
        print(f"  Already trained, skipping {model_name}")
        with (run_dir / "result.json").open() as f:
            return json.load(f)

    set_seed(MODEL_SEED)
    collate = make_collate("one_hot")
    def mk_ldr(ds, shuf):
        return RenameLoader(DataLoader(ds, batch_size=batch_size, shuffle=shuf, 
                                       collate_fn=collate, num_workers=0 if smoke else 4, pin_memory=True, 
                                       worker_init_fn=dataloader_worker_init))
    
    train_loader = mk_ldr(train_ds, True)
    val_loader = mk_ldr(val_ds, False)
    test_loader = mk_ldr(test_ds, False)

    result = {"model": model_name, "seed": MODEL_SEED, "smoke": smoke, "n_codes": n_codes}

    model = build_model(model_name, n_codes)

    # Load Stage-1 MIMIC checkpoint for adaptation
    mimic_ckpt_dir = REPO_ROOT / "results" / "baselines" / "mimic" / model_name
    if (mimic_ckpt_dir / "checkpoint.pt").exists() and hasattr(model, "load_checkpoint"):
        print(f"  Loading Stage-1 MIMIC checkpoint from {mimic_ckpt_dir.name}")
        model.load_checkpoint(mimic_ckpt_dir)
    elif model_name != "count_lightgbm":
        print(f"  WARNING: No MIMIC checkpoint found for {model_name}, training from scratch!")
    if model_name == "count_lightgbm":
        lgb_result = train_lightgbm(model, train_loader, val_loader, test_loader, n_codes)
        result.update(lgb_result)
        result["model_card"] = model.model_card
        model.save_checkpoint(run_dir)
    else:
        model.to(get_device(device))
        result["param_counts"] = count_parameters(model) if isinstance(model, torch.nn.Module) else {}
        train_result = train_neural_baseline(
            model=model, train_fn=model.training_step, predict_fn=model.predict,
            train_loader=train_loader, val_loader=val_loader,
            lr=LR, weight_decay=WEIGHT_DECAY, max_epochs=max_ep, patience=PATIENCE,
            min_epochs=min_ep,
            val_max_batches=None if smoke else VAL_MAX_BATCHES,
            grad_clip=GRAD_CLIP, device=device, run_dir=run_dir, seed=MODEL_SEED
        )
        test_metrics = evaluate_loader(
            model, model.predict, test_loader, get_device(device),
            max_batches=None if smoke else TEST_MAX_BATCHES,
        )        result["train"] = train_result
        result["test_metrics"] = test_metrics
        result["model_card"] = model.model_card
        model.save_checkpoint(run_dir)

    with (run_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)
    
    auroc = result.get("test_metrics", {}).get("micro_auroc", "N/A")
    print(f"  Test AUROC={auroc}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tensorized_dir", default="artifacts/nch_stage2/v2/tensorized_forecast/diagnoses_only")
    parser.add_argument("--vocab_path", default="data/processed/code_vocab.json")
    args = parser.parse_args()

    output_dir = REPO_ROOT / "results" / "baselines" / "nch"
    output_dir.mkdir(parents=True, exist_ok=True)

    tdir = REPO_ROOT / args.tensorized_dir
    if args.smoke:
        from stage1_mimic_pretrain.train import maybe_restrict_tensorized
        tdir = maybe_restrict_tensorized(tdir, output_dir, max_shards=1, seed=0)

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
    
    models = ["retain", "ehr_bert", "behrt", "medbert", "cehrbert", "dtr"] if args.models == "all" else args.models.split(",")
    all_results = {}

    for m in models:
        try:
            res = run_one_model(m, output_dir, train_ds, val_ds, test_ds, 8 if args.smoke else BATCH_SIZE, args.smoke, args.device)
            all_results[m] = res
        except Exception as e:
            print(f"ERROR on {m}: {e}")
            import traceback
            traceback.print_exc()

    with (output_dir / "all_results.json").open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
