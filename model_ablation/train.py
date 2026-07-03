#!/usr/bin/env python3
"""SHARED vanilla pretrain for the ablation (code loss only; no time/Weibull loss).

Produces the single backbone that all four fine-tune arms load. Age conditioning
is NOT present here (vanilla); it is introduced only at fine-tune, symmetrically
for the two age arms.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_ablation.dataset import TensorizedEHRDataset, _dataloader_worker_init, ehr_collate
from model_ablation.tale_ehr import TALEEHR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared vanilla pretrain (ablation).")
    p.add_argument("--tensorized_dir", type=Path, default=REPO_ROOT / "data/processed/tensorized")
    p.add_argument("--embedding_path", type=Path, default=REPO_ROOT / "data/processed/bge_embeddings.pt")
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--poly_degree", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--save_dir", type=Path, default=REPO_ROOT / "checkpoints/ablation_pretrain")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import json
    num_codes = len(json.load(open(args.vocab_path)))
    device = torch.device(args.device)

    train_ds = TensorizedEHRDataset(args.tensorized_dir / "train", args.vocab_path)
    val_ds = TensorizedEHRDataset(args.tensorized_dir / "val", args.vocab_path)
    kw = dict(batch_size=args.batch_size, collate_fn=ehr_collate, num_workers=args.num_workers,
              pin_memory=(args.device == "cuda"), persistent_workers=(args.num_workers > 0),
              prefetch_factor=(4 if args.num_workers > 0 else None), worker_init_fn=_dataloader_worker_init)
    if args.num_workers > 0:
        kw["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    model = TALEEHR(embedding_path=args.embedding_path, num_codes=num_codes,
                    d_model=args.d_model, poly_degree=args.poly_degree).to(device)
    model._variant_tag = "vanilla"
    opt = Adam(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            if args.dry_run and step > 3:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
            with ctx:
                out = model(batch)
                loss = F.binary_cross_entropy_with_logits(out["code_logits"], batch["target_codes"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        print(f"Epoch {epoch:03d} | train_code_bce={np.mean(losses):.6f}", flush=True)
        torch.save({"epoch": epoch, "model_variant": "vanilla",
                    "model_state_dict": model.state_dict()}, args.save_dir / f"epoch_{epoch:03d}.pt")
        if args.dry_run:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
