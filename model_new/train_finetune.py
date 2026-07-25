#!/usr/bin/env python3
"""Fine-tuning: binary disease classification from the shared pretrained backbone.

Two legacy defects are structurally impossible here.

D8 / INV-TMAX -- ``tau_max`` is read from the checkpoint and reused **bit-for-bit**. It is
never re-derived from the fine-tune corpus, and an explicit ``--tau_max`` that disagrees
with the checkpoint raises. Re-deriving it would silently change the meaning of every
learned coefficient.

D9 -- there is no ``return_repr_only``. The classification head sits on the pooled ``h``,
exactly as pretraining does, so the pooling-site age parameters carry gradient.

Every arm runs this file with identical flags apart from ``--arm`` and ``--run_name``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as _dt
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_new import diagnostics as D
from model_new.arms import ARMS
from model_new.data import dataloader_worker_init, demo_layout
from model_new.data_finetune import (
    TensorizedFinetuneDataset, check_tau_max, make_finetune_collate,
)
from model_new.model import DKMModel
from model_new.optim import build_param_groups
from model_new.train import _git, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_root", type=Path, default=Path("model_new/run_finetune"))
    p.add_argument("--pretrained_ckpt", type=Path, required=True)
    p.add_argument("--tensorized_dir", type=Path, required=True,
                   help="directory containing train/ val/ test/ shard_*.npz")
    p.add_argument("--embedding_path", type=Path,
                   default=REPO_ROOT / "data/processed/bge_embeddings.pt")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--tau_max", type=float, default=None,
                   help="must equal the checkpoint value; provided only so a mismatch is loud")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_age", type=float, default=1e-3)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=6)
    return p


def resolve_tau_max(ckpt: dict, override: float | None) -> float:
    """INV-TMAX. The checkpoint is the single source of truth."""
    if "tau_max" not in ckpt:
        raise AssertionError(
            "[INV-TMAX] checkpoint has no tau_max. It must be stored at pretrain time and "
            "reused verbatim; recomputing it from the fine-tune corpus would change the "
            "meaning of every learned coefficient."
        )
    ckpt_tau = float(ckpt["tau_max"])
    if override is not None and float(override) != ckpt_tau:
        raise AssertionError(
            f"[INV-TMAX] --tau_max={float(override)!r} disagrees with the checkpoint's "
            f"{ckpt_tau!r}. Refusing to run: tau_max must match bit-for-bit."
        )
    return ckpt_tau


def resolve_age_standardization(ckpt: dict, override: tuple[float, float] | None
                                ) -> tuple[float, float]:
    """INV-AGESTD. The demographic-age standardization constants come from the checkpoint and
    are reused verbatim, exactly like tau_max.

    Re-deriving (mean, sd) from the fine-tune corpus would put a PIC child near 0 -- PIC's own
    mean -- instead of near -3, its true position relative to the adult pretraining corpus,
    silently redefining what the demographic age feature means to demo_proj.
    """
    cfg = ckpt.get("config", {}).get("model", {})
    std = cfg.get("age_standardization")
    if std is None:
        raise AssertionError(
            "[INV-AGESTD] checkpoint has no age_standardization constants. They must be "
            "stored at pretrain time and reused verbatim."
        )
    mean, sd = float(std["mean"]), float(std["sd"])
    if override is not None and (float(override[0]) != mean or float(override[1]) != sd):
        raise AssertionError(
            f"[INV-AGESTD] override (mean={override[0]}, sd={override[1]}) disagrees with the "
            f"checkpoint's (mean={mean}, sd={sd}). Refusing to run: constants must match."
        )
    return mean, sd


def load_backbone(model: DKMModel, state: dict, arm: str) -> dict:
    """Load the shared pretrain backbone. Missing/unexpected keys must be age modules or the
    head, never backbone weights.

    The pretrain head predicts |V| codes and the fine-tune head predicts one logit, so its
    keys are dropped before loading: ``strict=False`` tolerates *absent* keys but still
    raises on a shape mismatch for a key that is present.
    """
    dropped = sorted(k for k in state if k.startswith("head."))
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    incompatible = model.load_state_dict(state, strict=False)
    age_ids = {id(p) for p in model.age_parameters()}
    age_keys = {name for name, p in model.named_parameters() if id(p) in age_ids}
    age_prefixes = tuple(sorted({k.rsplit(".", 2)[0] for k in age_keys})) or ("__none__",)

    def ok(key: str) -> bool:
        return key.startswith("head.") or key.startswith(age_prefixes) or ".age." in key

    bad_missing = [k for k in incompatible.missing_keys if not ok(k)]
    bad_unexpected = [k for k in incompatible.unexpected_keys if not ok(k)]
    if bad_missing or bad_unexpected:
        raise AssertionError(
            f"[HARD] backbone weights did not transfer for arm={arm}. "
            f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}"
        )
    return {"missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "dropped_pretrain_head_keys": dropped}


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> dict:
    was_training = model.training
    model.eval()
    logits, labels, ages, losses = [], [], [], []
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        with ctx:
            out = model(batch)
            loss = F.binary_cross_entropy_with_logits(out["logits"].float(), batch["labels"])
        losses.append(float(loss))
        logits.append(out["logits"].float().cpu())
        labels.append(batch["labels"].cpu())
        ages.append(out["age_last"].cpu())
    if was_training:
        model.train()
    y = torch.cat(labels).numpy()
    p = torch.sigmoid(torch.cat(logits)).numpy()
    a = torch.cat(ages)
    res = {"loss": float(np.mean(losses)) if losses else float("nan"), "n": int(y.size)}
    res.update(_binary_metrics(y, p))
    idx = D.band_index(a)
    res["by_band"] = {}
    for i, name in enumerate(D.band_names()):
        sel = idx == i
        entry = {"n": int(sel.sum()), "pos": int(y[sel].sum()) if sel.any() else 0}
        if sel.sum() > 0:
            entry.update(_binary_metrics(y[sel], p[sel]))
        res["by_band"][name] = entry
    return res


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out: dict[str, float] = {"prevalence": float(y.mean()) if y.size else float("nan")}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if 0 < y.sum() < y.size:
            out["auroc"] = float(roc_auc_score(y, p))
            out["auprc"] = float(average_precision_score(y, p))
    except Exception:
        pass
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)
    tau_max = resolve_tau_max(ckpt, args.tau_max)
    age_mean, age_sd = resolve_age_standardization(ckpt, None)
    pre_cfg = ckpt.get("config", {}).get("model", {})
    race_encoding = pre_cfg.get("race_encoding", "one_hot")
    demo_dim, demo_channels = demo_layout(race_encoding)
    if pre_cfg.get("demo_dim") not in (None, demo_dim):
        raise AssertionError(
            f"[HARD] checkpoint demo_dim={pre_cfg.get('demo_dim')} but this race_encoding "
            f"gives {demo_dim}; the demographic layout must be identical across pretrain and "
            f"fine-tune and across arms.")

    collate = make_finetune_collate(race_encoding)
    loader_kw = dict(num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                     worker_init_fn=dataloader_worker_init,
                     persistent_workers=args.num_workers > 0)
    splits = {name: TensorizedFinetuneDataset(args.tensorized_dir / name,
                                              max_seq_len=args.max_seq_len)
              for name in ("train", "val", "test")
              if (args.tensorized_dir / name).exists()}
    if "train" not in splits:
        raise FileNotFoundError(f"no train/ split under {args.tensorized_dir}")
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **loader_kw)
    eval_loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False, **loader_kw)
                    for k, v in splits.items() if k != "train"}

    model = DKMModel(
        num_codes=pre_cfg.get("head_out", ckpt.get("config", {}).get("data", {})
                              .get("vocab_size")),
        embedding_path=args.embedding_path, arm=args.arm, seed=args.seed,
        d_model=pre_cfg.get("d_model", 256), n_layers=pre_cfg.get("n_layers", 1),
        n_heads=pre_cfg.get("n_heads", 1), use_residual=pre_cfg.get("use_residual", True),
        use_layernorm=pre_cfg.get("use_layernorm", True), use_ffn=pre_cfg.get("use_ffn", True),
        s=pre_cfg.get("s", 5), tau_max=tau_max,
        age_M=pre_cfg.get("fourier", {}).get("M", 16),
        age_p_min=pre_cfg.get("fourier", {}).get("p_min", 0.15),
        age_p_max=pre_cfg.get("fourier", {}).get("p_max", 6.0),
        age_hidden=pre_cfg.get("age_hidden", 64),
        gen_final_bias=pre_cfg.get("gen_final_bias", False),
        center_delta_alpha=pre_cfg.get("center_delta_alpha", False),
        demo_dim=demo_dim, demo_channels=demo_channels, race_encoding=race_encoding,
        demo_hidden=pre_cfg.get("demo_hidden", 64), age_mean=age_mean, age_sd=age_sd,
        task="classification",
    )
    load_info = load_backbone(model, ckpt["model_state_dict"], args.arm)
    model.to(device)

    if abs(model.tau_max - tau_max) > 0:
        raise AssertionError(f"[INV-TMAX] model tau_max={model.tau_max} != checkpoint {tau_max}")
    if float(model.age_mean) != age_mean or float(model.age_sd) != age_sd:
        raise AssertionError(
            f"[INV-AGESTD] model age std ({float(model.age_mean)}, {float(model.age_sd)}) != "
            f"checkpoint ({age_mean}, {age_sd})")

    groups, group_report = build_param_groups(model, args.lr_backbone, args.lr_age, args.lr_head)
    opt = torch.optim.Adam(groups)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    theta0 = D.snapshot_parameters(groups)

    domain = check_tau_max(splits["train"], tau_max, n_samples=min(2000, len(splits["train"])),
                           seed=args.seed)
    config = {
        "run_id": args.run_name, "arm": args.arm, "seed": args.seed,
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pretrained_ckpt": str(args.pretrained_ckpt),
        "model": model.config_dict(),
        "params": model.parameter_report(),
        "optim": {**group_report, "batch_size": args.batch_size, "epochs": args.epochs,
                  "optimizer": "Adam"},
        "data": {"tensorized_dir": str(args.tensorized_dir),
                 "split_sizes": {k: len(v) for k, v in splits.items()},
                 "race_encoding": race_encoding},
        "tau_max": tau_max,
        "tau_max_source": ckpt.get("tau_max_source"),
        "tau_domain_check": domain,
        "state_dict_load": load_info,
    }
    D.write_json(run_dir / "config.json", config)
    D.print_config_summary(config)
    D.print_kv("INV-TMAX / fine-tune tau domain  [MEASURE]", domain)

    records: list[dict] = []
    best, bad_epochs = -np.inf, 0
    t_start = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        model.reset_clamp_stats()
        running, n_batches = 0.0, 0
        diag_batch = None
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            if diag_batch is None:
                diag_batch = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                              for k, v in batch.items()}
            ctx = (torch.amp.autocast(device_type="cuda", enabled=True) if use_amp
                   else nullcontext())
            with ctx:
                out = model(batch)
                loss = F.binary_cross_entropy_with_logits(out["logits"].float(), batch["labels"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], args.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss)
            n_batches += 1

        ages_flat = diag_batch["age_years"][diag_batch["attention_mask"]].detach()
        record = {
            "epoch": epoch, "wall_clock_s": time.time() - t_start,
            "train_loss": running / max(1, n_batches),
            "eval": {k: evaluate(model, ld, device, use_amp) for k, ld in eval_loaders.items()},
            "alpha": D.alpha_diagnostics(model, ages_flat),
            "delta_alpha_grid": D.delta_alpha_grid(model),
            "w_curves": D.w_curves(model),
            "param_drift": D.parameter_drift(groups, theta0),
            "clamp_rate": D.clamp_rates(model),
        }
        records.append(record)
        D.append_train_json(run_dir / "train.json", records)
        D.print_finetune_epoch(record)

        score = record["eval"].get("val", {}).get("auprc", -np.inf)
        if score > best:
            best, bad_epochs = score, 0
            torch.save({"epoch": epoch, "arm": args.arm, "seed": args.seed,
                        "model_state_dict": model.state_dict(), "tau_max": tau_max,
                        "config": config}, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                D.print_block("early stop", [f"no val AUPRC improvement in {args.patience} "
                                             f"epochs; best={best:.6f}"])
                break

    D.print_block("done", [f"run_dir: {run_dir}", f"best val AUPRC: {best:.6f}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
