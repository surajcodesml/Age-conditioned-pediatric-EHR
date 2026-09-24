import os
import sys
import time
from pathlib import Path
import torch
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from stage1_mimic_pretrain.train import maybe_restrict_tensorized
from model_new.data import TensorizedPretrainDataset

tdir = Path("data/processed/tensorized_flat")
out_dir = Path("results/baselines/mimic")
vocab = Path("data/processed/code_vocab.json")

print("Restricting tensorized...", flush=True)
t0 = time.time()
tdir_sub = maybe_restrict_tensorized(tdir, out_dir, max_shards=1, seed=0)
print(f"Done in {time.time()-t0:.2f}s. Sub dir: {tdir_sub}", flush=True)

print("Initializing dataset...", flush=True)
t0 = time.time()
ds = TensorizedPretrainDataset(tdir_sub / "train", vocab, max_seq_len=1024)
print(f"Done in {time.time()-t0:.2f}s. Size: {len(ds)}", flush=True)

print("Getting first item...", flush=True)
t0 = time.time()
item = ds[0]
print(f"Done in {time.time()-t0:.2f}s.", flush=True)
for k, v in item.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: {v}")
