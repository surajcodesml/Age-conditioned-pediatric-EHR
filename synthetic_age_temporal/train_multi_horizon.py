"""Experimental multi-horizon trainer for DTR.

NOTE: Multi-horizon supervision is currently experimental and is not part of
the locked canonical architecture because naive joint training with content
persistence inverted beta on the synthetic S2 benchmark.

Do NOT enable this as the default/canonical trainer. Prefer train_dtr.py.
Do NOT add horizon-specific persistence offsets (delta_h) here without a
separate controlled experiment.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

from config import DATA_SEED, AGE_CENTER, AGE_SCALE
from dataset import build_vocab
from dataset_dtr import DTRDataset, collate_dtr
from model_dtr import CONTENT_SCORE_EXP_CLAMP, build_dtr, count_parameters
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
    """Experimental wrapper: shared Content-Persistence encoder + per-horizon heads."""

    def __init__(self, base_dtr, num_horizons: int, n_targets: int, d_model: int, aggregation: str):
        super().__init__()
        self.base = base_dtr
        self.num_horizons = num_horizons
        self.aggregation = aggregation

        hist_in = d_model + (1 if aggregation == "weighted_mean_plus_log_mass" else 0)

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
        hist_f = hist.to(v.dtype)

        k = self.base.content_key(v)
        scale = np.sqrt(k.size(-1))
        u = torch.einsum("bmd,d->bm", k, self.base.content_query) / scale
        u = u.masked_fill(~hist, 0.0)

        theta_content = self.base.persistence_offset(v) * hist_f
        theta_m = self.base.theta0 + theta_content
        z = ((age - AGE_CENTER) / AGE_SCALE)
        if self.base.age_temporal:
            lam = F.softplus(theta_m + self.base.beta * z.unsqueeze(-1))
        else:
            lam = F.softplus(theta_m)
        g = torch.exp(-lam * enc_tau) * hist_f

        w = torch.exp(u.clamp(max=CONTENT_SCORE_EXP_CLAMP)) * g
        w = w * hist_f

        M = w.sum(dim=1, keepdim=True)
        weighted = (w.unsqueeze(-1) * v).sum(dim=1)

        if self.aggregation == "raw_additive":
            h_hist = weighted
        else:
            h_bar = weighted / (M + 1e-6)
            log_mass = torch.log1p(M)
            h_hist = torch.cat([h_bar, log_mass], dim=-1)

        z1 = z.unsqueeze(-1)
        logits = []
        for i in range(self.num_horizons):
            h_logit = self.f_history_list[i](h_hist)
            a_logit = self.f_age_list[i](z1)
            logits.append(h_logit + a_logit + self.bias_list[i])

        self.base._cache = {
            "u": u.detach(), "g": g.detach(), "w": w.detach(), "M": M.detach(),
            "lambda": lam.detach(), "h_hist": h_hist.detach(),
            "theta_content": theta_content.detach(),
        }
        return torch.stack(logits, dim=1)  # [B, num_horizons, num_targets]

    def age_parameters(self):
        return self.base.age_parameters()

    def zero_all_betas_(self):
        return self.base.zero_all_betas_()

    def restore_betas_(self, saved):
        return self.base.restore_betas_(saved)

    @property
    def beta(self):
        return self.base.beta

    @property
    def theta0(self):
        return self.base.theta0

    @property
    def gate(self):
        return self.base.gate


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


def _flatten_ht(y: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse [N, H, T] → [N*H, T] for classification_metrics."""
    if y.ndim == 2:
        return y, logits
    n, h, t = y.shape
    return y.reshape(n * h, t), logits.reshape(n * h, t)


@torch.no_grad()
def ablations_multi(model, loader, device) -> dict[str, Any]:
    """Age-shuffle and β=0 ablations for multi-horizon logits [B, H, T]."""
    model.eval()

    def _forward_batches(batches, ages=None, beta0=False):
        saved = None
        if beta0:
            saved = model.zero_all_betas_()
        ys, logits = [], []
        for i, batch in enumerate(batches):
            age = batch["age"] if ages is None else ages[i]
            out = model(
                enc_code_ids=batch["enc_code_ids"],
                enc_code_mask=batch["enc_code_mask"],
                enc_tau=batch["enc_tau"],
                enc_padding_mask=batch["enc_padding_mask"],
                age=age,
            )
            ys.append(batch["labels"].cpu().numpy())
            logits.append(out.cpu().numpy())
        if saved is not None:
            model.restore_betas_(saved)
        return classification_metrics(
            *_flatten_ht(np.concatenate(ys), np.concatenate(logits))
        )

    batches = [_move(batch, device) for batch in loader]
    base = _forward_batches(batches)

    flat = np.concatenate([b["age"].cpu().numpy() for b in batches])
    shuf = np.random.default_rng(0).permutation(flat)
    ptr = 0
    ages_shuf = []
    for batch in batches:
        bsz = batch["age"].size(0)
        ages_shuf.append(
            torch.tensor(shuf[ptr : ptr + bsz], dtype=torch.float32, device=device)
        )
        ptr += bsz

    sh = _forward_batches(batches, ages=ages_shuf)
    b0 = _forward_batches(batches, beta0=True)

    return {
        "normal": base,
        "shuffle_age": sh,
        "beta0": b0,
        "delta_bce_shuffle_age": sh["bce"] - base["bce"],
        "delta_bce_beta0": b0["bce"] - base["bce"],
        "delta_auroc_shuffle": base["micro_auroc"] - sh["micro_auroc"],
        "delta_auroc_beta0": base["micro_auroc"] - b0["micro_auroc"],
        "functional_shuffle": bool(sh["bce"] - base["bce"] > 1e-4),
        "functional_beta0": bool(b0["bce"] - base["bce"] > 1e-4),
    }


def train_forecast():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--single-horizon", action="store_true")
    ap.add_argument("--content-persistence", action=argparse.BooleanOptionalAction, default=True)
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
    
    # Experimental multi-horizon path — not part of locked canonical architecture.
    base = build_dtr(
        age_temporal=True,
        n_codes=info["n_codes"],
        n_targets=info["n_targets"],
        d_model=64,
        aggregation="raw_additive",
        content_persistence=args.content_persistence,
        multi_query_K=1,
    )
    
    if args.single_horizon:
        # train only on horizon 180 (index 2)
        model = base.to(dev)
    else:
        model = MultiHorizonDTR(base, 4, info["n_targets"], 64, "raw_additive").to(dev)
        
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = BCEWithLogitsLoss()
    
    best = float("inf")
    patience = 10
    best_state = None
    
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
                
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
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
        abl = ablations_multi(model, test_loader, dev)
        
    rec = recovery(model if args.single_horizon else model.base, -2.5, 0.0)

    mean_auprc = None
    if not args.single_horizon:
        mean_auprc = float(np.mean([test[f"horizon_{h}"]["micro_auprc"] for h in HORIZONS]))
    
    result = {
        "test": test,
        "ablations": abl,
        "recovery": rec,
        "beta_hat": rec["beta_hat"],
        "n_params": count_parameters(model),
        "content_persistence": bool(args.content_persistence),
        "multi_horizon": not args.single_horizon,
        "multi_horizon_mean_auprc": mean_auprc,
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)
    torch.save({"model": best_state, "config": result}, run_dir / "checkpoint.pt")
    print(
        f"Done. mean_auprc={mean_auprc} beta_hat={rec['beta_hat']:.4f} "
        f"sign_match={rec['sign_match']} "
        f"ΔBCE_shuffle={abl.get('delta_bce_shuffle_age')} "
        f"ΔBCE_β0={abl.get('delta_bce_beta0')}"
    )

if __name__ == "__main__":
    train_forecast()
