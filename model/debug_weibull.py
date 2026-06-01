#!/usr/bin/env python3
"""Run a fixed-length Weibull diagnostic training pass."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

try:
    from model.dataset import TensorizedEHRDataset, ehr_collate
    from model.tale_ehr import TALEEHR
    from model.train import _dataloader_worker_init, pretrain
except ModuleNotFoundError:
    from dataset import TensorizedEHRDataset, ehr_collate
    from tale_ehr import TALEEHR
    from train import _dataloader_worker_init, pretrain


class _TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _subset_steps(ds, batch_size: int, n_steps: int):
    n_needed = batch_size * n_steps
    if len(ds) < n_needed:
        raise RuntimeError(f"Dataset too small: need {n_needed}, have {len(ds)}")
    return Subset(ds, list(range(n_needed)))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "processed"
    vocab_path = data_dir / "code_vocab.json"
    emb_path = data_dir / "bge_embeddings.pt"
    save_dir = repo_root / "checkpoints" / "debug_weibull"
    save_dir.mkdir(parents=True, exist_ok=True)

    console_path = save_dir / "console.log"
    console_fh = open(console_path, "w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, console_fh)
    sys.stderr = _TeeStream(orig_stderr, console_fh)
    try:
        print(f"[run] checkpoints -> {save_dir}", flush=True)
        print(f"[run] console log -> {console_path}", flush=True)

        with vocab_path.open("r", encoding="utf-8") as f:
            code_vocab = json.load(f)
        num_codes = len(code_vocab)

        tensorized_root = data_dir / "tensorized"
        train_base = TensorizedEHRDataset(tensorized_root / "train", vocab_path)
        val_base = TensorizedEHRDataset(tensorized_root / "val", vocab_path)

        batch_size = 16
        train_ds = _subset_steps(train_base, batch_size=batch_size, n_steps=2000)
        val_ds = _subset_steps(val_base, batch_size=batch_size, n_steps=200)

        loader_kw = dict(
            batch_size=batch_size,
            num_workers=0,
            collate_fn=ehr_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            worker_init_fn=_dataloader_worker_init,
        )
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

        model = TALEEHR(
            embedding_path=emb_path,
            num_codes=num_codes,
            d_model=256,
            poly_degree=5,
        )
        model._variant_tag = "baseline"
        model._age_mode_tag = None

        pretrain(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            lr=1e-4,
            gamma_loss=1.0,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            save_dir=save_dir,
            dry_run=False,
            code_loss_name="bce",
            bce_pos_weight=0.0,
            resume_from=None,
            model_variant="baseline",
            age_conditioning_mode=None,
            age_diag_every=200,
            w_curve_every=200,
        )
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        console_fh.close()
