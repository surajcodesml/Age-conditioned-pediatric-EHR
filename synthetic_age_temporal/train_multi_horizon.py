import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from config import DATA_SEED, AGE_CENTER, AGE_SCALE
from dataset import build_vocab
from dataset_dtr import DTRDataset, collate_dtr
from model_dtr import build_dtr, count_parameters
from evaluate import classification_metrics

HORIZONS = [30, 90, 180, 365]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _move(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

class MultiHorizonDTR(nn.Module):
    def __init__(self, base_dtr, num_horizons: int, n_targets: int, d_model: int, aggregation: str):
        super().__init__()
        self.base = base_dtr
        self.num_horizons = num_horizons
        
        hist_in = (d_model * self.base.multi_query_K) + (self.base.multi_query_K if aggregation == "weighted_mean_plus_log_mass" else 0)
        
        self.f_history_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hist_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_targets),
            ) for _ in range(num_horizons)
        ])
        self.f_age_list = nn.ModuleList([nn.Linear(1, n_targets) for _ in range(num_horizons)])
        self.bias_list = nn.ParameterList([nn.Parameter(torch.zeros(n_targets)) for _ in range(num_horizons)])

    def forward(self, enc_code_ids, enc_code_mask, enc_tau, enc_padding_mask, age, **kwargs):
        v = self.base.encode_encounters(enc_code_ids, enc_code_mask)
        hist = ~enc_padding_mask
        
        k = self.base.W_k(v)
        val = self.base.W_v(v)
        u = torch.matmul(k, self.base.q.T) / np.sqrt(k.size(-1))
        u = u.masked_fill(~hist.unsqueeze(-1), 0.0)

        theta = self.base.W_r(v).squeeze(-1) if self.base.content_persistence else None
        g = self.base.gate.gate(age, enc_tau, theta) * hist.to(enc_tau.dtype)

        w = torch.exp(u.clamp(max=20.0)) * g.unsqueeze(-1)
        w = w * hist.unsqueeze(-1).to(w.dtype)

        M = w.sum(dim=1)
        weighted = torch.einsum("bmk,bmd->bkd", w, val)

        if self.base.aggregation == "raw_additive":
            h_hist = weighted.reshape(weighted.size(0), -1)
        else:
            h_bar = weighted / (M.unsqueeze(-1) + 1e-6)
            log_mass = torch.log1p(M)
            h_hist = torch.cat([h_bar.reshape(weighted.size(0), -1), log_mass], dim=-1)

        z = ((age - AGE_CENTER) / AGE_SCALE).unsqueeze(-1)
        
        logits = []
        for i in range(self.num_horizons):
            h_logit = self.f_history_list[i](h_hist)
            a_logit = self.f_age_list[i](z)
            logits.append(h_logit + a_logit + self.bias_list[i])
            
        self.base._cache = {
            "u": u.detach(), "g": g.detach(), "w": w.detach(), "M": M.detach(),
            "lambda": self.base.gate.lambda_of(age).detach(), "h_hist": h_hist.detach()
        }
            
        return torch.stack(logits, dim=1) # [B, num_horizons, num_targets]

    def age_parameters(self):
        return self.base.gate.age_parameters()

    def zero_all_betas_(self):
        return self.base.zero_all_betas_()

    def restore_betas_(self, saved):
        self.base.restore_betas_(saved)

def load_forecast_data(scenario_dir, batch_size, target_idx=None, num_workers=0):
    import pandas as pd
    examples = pd.read_parquet(scenario_dir / "examples.parquet")
    forecast_data = np.load(scenario_dir / "labels_forecast.npz")
    labels = forecast_data["Y"] # [N, 4, num_targets]
    
    with (scenario_dir / "meta.json").open() as f:
        meta = json.load(f)
    with (scenario_dir / "target_specs.json").open() as f:
        specs = json.load(f)
        
    vocab = build_vocab(examples)
    n_targets = labels.shape[2] if target_idx is None else len(target_idx)
    if target_idx is not None:
        labels = labels[:, :, target_idx]
        
    loaders = {}
    
    from torch.utils.data import DataLoader
    for split in ("train", "val", "test"):
        ds = DTRDataset(
            examples,
            labels,
            split,
            vocab,
            target_idx=None,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_dtr,
        )
    return loaders["train"], loaders["val"], loaders["test"], vocab, {
        "n_codes": len(vocab), "n_targets": n_targets
    }

def evaluate_multi(model, loader, device, single_horizon_idx=None):
    model.eval()
    ys, logits = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            out = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=batch["age"],
            )
            # out is [B, H, T], labels is [B, H, T]
            ys.append(batch["labels"].cpu().numpy())
            logits.append(out.cpu().numpy())
            
    y = np.concatenate(ys)
    p = np.concatenate(logits)
    
    res = {}
    if single_horizon_idx is not None:
        # Evaluate only the trained horizon vs the target horizon
        res["primary"] = classification_metrics(y[:, single_horizon_idx], p)
        for h_idx in range(len(HORIZONS)):
            res[f"horizon_{HORIZONS[h_idx]}"] = classification_metrics(y[:, h_idx], p)
    else:
        for h_idx in range(len(HORIZONS)):
            res[f"horizon_{HORIZONS[h_idx]}"] = classification_metrics(y[:, h_idx], p[:, h_idx])
            
    return res

def train_forecast():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--single-horizon", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    
    scenario_dir = args.scenario_dir
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with (scenario_dir / "target_specs.json").open() as f:
        specs = json.load(f)
    target_idx = [i for i, s in enumerate(specs) if s["mechanism"] == "interaction"]
    
    train_loader, val_loader, test_loader, vocab, info = load_forecast_data(scenario_dir, 64, target_idx)
    
    base = build_dtr(
        age_temporal=True,
        n_codes=info["n_codes"],
        n_targets=info["n_targets"],
        d_model=64,
        aggregation="weighted_mean_plus_log_mass",
        content_persistence=False,
        multi_query_K=1,
    )
    
    if args.single_horizon:
        # train only on horizon 180 (index 2)
        model = base.to(dev)
    else:
        model = MultiHorizonDTR(base, 4, info["n_targets"], 64, "weighted_mean_plus_log_mass").to(dev)
        
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = BCEWithLogitsLoss()
    
    best = float("inf")
    patience = 10
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = _move(batch, dev)
            opt.zero_grad()
            out = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=batch["age"],
            )
            y = batch["labels"]
            if args.single_horizon:
                loss = loss_fn(out, y[:, 2]) # 180 days
            else:
                loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * y.size(0)
            n += y.size(0)
            
        val = evaluate_multi(model, val_loader, dev, single_horizon_idx=2 if args.single_horizon else None)
        if args.single_horizon:
            val_bce = val["primary"]["bce"]
            val_auroc = val["primary"]["micro_auroc"]
        else:
            val_bce = np.mean([val[f"horizon_{h}"]["bce"] for h in HORIZONS])
            val_auroc = np.mean([val[f"horizon_{h}"]["micro_auroc"] for h in HORIZONS])
            
        print(f"Ep {epoch}: loss={total/n:.4f} val_bce={val_bce:.4f} val_auroc={val_auroc:.4f}")
        
        if val_bce < best - 1e-5:
            best = val_bce
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 10
        else:
            patience -= 1
            if patience <= 0:
                break
                
    model.load_state_dict(best_state)
    test = evaluate_multi(model, test_loader, dev, single_horizon_idx=2 if args.single_horizon else None)
    
    # Calculate recovery (optional, reuse logic from train_dtr)
    from train_dtr import recovery, ablations
    
    if args.single_horizon:
        # Wrapper to make model return correctly shaped output for ablations
        class Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.gate = m.gate
            def forward(self, *a, **k):
                return self.m(*a, **k)
            def zero_all_betas_(self): return self.m.zero_all_betas_()
            def restore_betas_(self, s): return self.m.restore_betas_(s)
        abl_model = Wrap(model)
        # Monkey patch batch to only have single horizon
        class SingleHorizonLoader:
            def __init__(self, loader): self.loader = loader
            def __iter__(self):
                for b in self.loader:
                    b = dict(b)
                    b["labels"] = b["labels"][:, 2]
                    yield b
        abl = ablations(abl_model, SingleHorizonLoader(test_loader), dev)
    else:
        # we need custom ablation logic for multi horizon or just average it
        abl = {}
        
    rec = recovery(model if args.single_horizon else model.base, -2.5, 0.0)
    
    result = {
        "test": test,
        "ablations": abl,
        "recovery": rec,
        "beta_hat": rec["beta_hat"],
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    train_forecast()
