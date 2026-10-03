#!/usr/bin/env python3
"""Steady-state samp/s: seed0 recipe vs throughput_optimized+bf16."""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model_new.data import (  # noqa: E402
    TensorizedPretrainDataset,
    dataloader_worker_init,
    make_collate,
)
from baselines.mimic.runner import (  # noqa: E402
    MAX_SEQ_LEN,
    THROUGHPUT_BATCH_SIZES,
    RenameLoader,
    build_model,
    configure_dataloader_multiprocessing,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    tdir = REPO / "data/processed/tensorized_flat"
    vocab = REPO / "data/processed/code_vocab.json"
    ds = TensorizedPretrainDataset(tdir / "train", vocab, max_seq_len=MAX_SEQ_LEN)
    n_codes = ds.num_codes
    collate = make_collate("one_hot")
    if args.num_workers > 0:
        configure_dataloader_multiprocessing(args.num_workers)
    device = torch.device("cuda")

    cfgs = [
        ("seed0", "behrt", 128, "fp32"),
        ("opt", "behrt", THROUGHPUT_BATCH_SIZES["behrt"], "bf16"),
        ("seed0", "cehrbert", 128, "fp32"),
        ("opt", "cehrbert", THROUGHPUT_BATCH_SIZES["cehrbert"], "bf16"),
        ("seed0", "retain", 128, "fp32"),
        ("opt", "retain", THROUGHPUT_BATCH_SIZES["retain"], "bf16"),
        ("seed0", "ehr_bert", 128, "fp32"),
        ("opt", "ehr_bert", THROUGHPUT_BATCH_SIZES["ehr_bert"], "bf16"),
        ("seed0", "medbert", 128, "fp32"),
        ("opt", "medbert", THROUGHPUT_BATCH_SIZES["medbert"], "bf16"),
    ]

    for label, name, bsz, amp in cfgs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(name, n_codes).to(device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=3e-4
        )
        kw: dict = dict(
            batch_size=bsz,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=dataloader_worker_init,
        )
        if args.num_workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 1 if bsz >= 256 else 2
        loader = RenameLoader(DataLoader(ds, **kw))
        it = iter(loader)
        model.train()
        use_amp = amp == "bf16"
        try:
            for i in range(args.warm + args.steps):
                batch = next(it)
                batch = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                if i == args.warm:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                ):
                    loss = model.training_step(batch)["loss"]
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            sps = (args.steps * bsz) / dt
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"{label:5s} {name:10s} bs={bsz:4d} amp={amp:4s}  "
                f"{sps:7.1f} samp/s  {args.steps / dt:5.2f} batch/s  "
                f"peak={peak:5.2f}GiB  steady_s={dt:5.1f}",
                flush=True,
            )
        finally:
            del model, opt, loader, it
            gc.collect()
            torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
