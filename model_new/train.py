#!/usr/bin/env python3
"""Pretraining. Code BCE only (D10: no Weibull time-gap head, no time loss).

Every arm runs this file with identical flags apart from ``--arm`` and ``--run_name``. Seed,
data, schedule, optimizer settings and masking are shared; nothing is tuned per arm and the
primary endpoint is written into ``config.json`` before the first step.

    python -m model_new.train --arm vanilla         --seed 0 --run_name van_s0
    python -m model_new.train --arm kernel          --seed 0 --run_name ker_s0
    python -m model_new.train --arm random_constant --seed 0 --run_name rnd_s0
    python -m model_new.train --arm additive        --seed 0 --run_name add_s0
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as _dt
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_new import diagnostics as D
from model_new.age_encoding import LinearAgeFourier, LogAgeFourier, characterize_band
from model_new.arms import ARMS
from model_new.data import (
    TensorizedPretrainDataset, corpus_stats_cached, demo_layout,
    dataloader_worker_init, make_collate, sample_empirical_taus,
)
from model_new.model import DKMModel
from model_new.optim import build_param_groups

REPO_ROOT = Path(__file__).resolve().parents[1]

DEVIATIONS_FROM_DRAFT = [
    {"item": "direct log-space injection",
     "reason": "score += sum_k alpha_k T_k(tau_tilde), per draft Eq. 3; the legacy code used "
               "logsigmoid(poly). Only one is implemented. ||alpha||_1 is monitored but not "
               "regularised: a kernel that grows to dominate QK is a finding, not a bug."},
    {"item": "Fourier over log-age",
     "reason": "the linear band saturates at a 7.4-month gap and is a near-orthogonal hash of "
               "age, leaving a constant Delta-alpha as the cheapest solution (D7)."},
    {"item": "attention pooling",
     "reason": "per the draft body; Figure 1A draws mean pooling."},
    {"item": "padding-only masking",
     "reason": "the draft says 'masked at padding'; the legacy code used causal tril. The "
               "pretraining target visit is outside the input window, so bidirectional "
               "attention within the window is not leakage (D4)."},
    {"item": "additive arm concatenates to h",
     "reason": "per the draft; not the older internal guide's 1024-dim delta on code "
               "embeddings."},
    {"item": "no fabricated parameter matching",
     "reason": "kernel and random_constant are exactly matched by construction and that is the "
               "identifying comparison. additive has a different architecture and a different "
               "count, reported honestly rather than padded. The draft's 'parameter-matched' "
               "language applies to the kernel/random_constant pair and should be corrected "
               "for additive."},
    {"item": "demo_dim = 9, not 3",
     "reason": "race has cardinality 7 in this pipeline and the legacy scalar encoding imposes "
               "an arbitrary ordinal on a categorical. One-hot gives demo_dim = 2 + n_race. "
               "Identical across arms. --race_encoding scalar restores demo_dim = 3."},
    {"item": "age remains a demographic feature in every arm",
     "reason": "it is the route age already has and the one DKM must improve on; removing it "
               "would make vanilla age-blind and the baseline a strawman (D3)."},
    {"item": "demographic age channel is standardized (Fix C)",
     "reason": "raw age (median ~56) beside eight 0/1 channels dominates demo_proj's input "
               "scale ~50x, making R1 -- the route DKM must beat -- a poorly-scaled function "
               "of age. Constants are frozen from the pretrain corpus and reused at fine-tune "
               "(INV-AGESTD). age_years fed to psi stays raw."},
    {"item": "float64 lag arithmetic (Fix D)",
     "reason": "at span_days ~5800 the float32 ulp is ~40s and differencing two large nearby "
               "timestamps loses precision exactly where tau is most sensitive. Differencing "
               "and log1p run in float64, cast to float32 after."},
    {"item": "exact tau_max over the full split (Fix A)",
     "reason": "tau_max is frozen into the checkpoint and defines every coefficient, so a "
               "sampled maximum the full split exceeds would clamp silently for the whole "
               "run. It is now the exact full-split maximum with one ulp of float32 headroom."},
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_root", type=Path, default=Path("model_new/run"))

    p.add_argument("--tensorized_dir", type=Path,
                   default=REPO_ROOT / "data/processed/tensorized_flat")
    p.add_argument("--embedding_path", type=Path,
                   default=REPO_ROOT / "data/processed/bge_embeddings.pt")
    p.add_argument("--vocab_path", type=Path, default=REPO_ROOT / "data/processed/code_vocab.json")
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--race_encoding", choices=("one_hot", "scalar"), default="one_hot")

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--legacy_block", action="store_true",
                   help="disable residual, LayerNorm and FFN -> the legacy single-attention "
                        "encoder. Figure 1A does NOT depict this.")
    p.add_argument("--no_residual", action="store_true")
    p.add_argument("--no_layernorm", action="store_true")
    p.add_argument("--no_ffn", action="store_true")
    p.add_argument("--s", type=int, default=5)
    p.add_argument("--tau_max", type=float, default=None,
                   help="override; by default the EXACT max over the full pretrain corpus (D8).")
    p.add_argument("--stats_sample_windows", type=int, default=4000,
                   help="windows sampled for the O(L^2) pairwise corpus statistics; tau_max, "
                        "age std and all per-event stats are exact over the full split "
                        "regardless.")
    p.add_argument("--age_M", type=int, default=16)
    p.add_argument("--age_p_min", type=float, default=0.15)
    p.add_argument("--age_p_max", type=float, default=6.0)
    p.add_argument("--age_hidden", type=int, default=64)
    p.add_argument("--demo_hidden", type=int, default=64)
    p.add_argument("--gen_final_bias", action="store_true")
    p.add_argument("--center_delta_alpha", action="store_true")

    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_age", type=float, default=1e-3)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--val_max_batches", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=0, help="0 = no cap")
    p.add_argument("--report_at_step", type=int, default=200,
                   help="print the age-pathway liveness report at this step")
    p.add_argument("--stop_after_report", action="store_true",
                   help="exit cleanly right after the --report_at_step report")

    p.add_argument("--endpoint_dataset", type=str, default="PIC heart_malformations")
    p.add_argument("--endpoint_task", type=str,
                   default="binary classification: congenital heart malformation")
    p.add_argument("--endpoint_metric", type=str, default="AUPRC")
    return p


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_block_flags(args) -> tuple[bool, bool, bool]:
    if args.legacy_block:
        return False, False, False
    return not args.no_residual, not args.no_layernorm, not args.no_ffn


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, max_batches: int) -> dict:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    per_example: dict[int, list[torch.Tensor]] = {}
    ages: list[torch.Tensor] = []
    auroc_logits, auroc_targets = [], []
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        with ctx:
            out = model(batch)
            loss = F.binary_cross_entropy_with_logits(out["code_logits"].float(),
                                                      batch["target_codes"])
        losses.append(float(loss))
        for k, v in D.recall_per_example(out["code_logits"], batch["target_codes"]).items():
            per_example.setdefault(k, []).append(v)
        ages.append(out["age_last"].detach().cpu())
        if len(auroc_logits) < 4:  # micro-AUROC on a small cap; it is diagnostic_only anyway
            auroc_logits.append(out["code_logits"].detach().float().cpu())
            auroc_targets.append(batch["target_codes"].detach().cpu())
    if was_training:
        model.train()

    agg = D.aggregate_recall({k: torch.cat(v) for k, v in per_example.items()},
                             torch.cat(ages))
    agg["loss"] = float(np.mean(losses)) if losses else float("nan")
    agg["micro_auroc"] = D.micro_auroc_diagnostic_only(
        torch.cat(auroc_logits), torch.cat(auroc_targets)) if auroc_logits else None
    return agg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    use_residual, use_layernorm, use_ffn = resolve_block_flags(args)

    run_dir = Path(args.run_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    # -- data ---------------------------------------------------------------- #
    train_ds = TensorizedPretrainDataset(args.tensorized_dir / "train", args.vocab_path,
                                         max_seq_len=args.max_seq_len)
    val_ds = TensorizedPretrainDataset(args.tensorized_dir / "val", args.vocab_path,
                                       max_seq_len=args.max_seq_len)
    demo_dim, demo_channels = demo_layout(args.race_encoding)
    collate = make_collate(args.race_encoding)
    # The collate now ships timestamps (8 KB/sample), not tau (4.19 MB/sample), so it is no
    # longer the bottleneck: pin + persistent + prefetch are cheap and num_workers can be low.
    loader_kw = dict(num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
                     worker_init_fn=dataloader_worker_init,
                     persistent_workers=args.num_workers > 0)
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # -- corpus statistics: ONE full-split pass, the single source of truth (Fix A) --- #
    # This gives the exact tau_max frozen into the checkpoint (D8) and the age
    # standardization constants (Fix C), so both are corpus-exact rather than sampled.
    # The pass is O(events) and takes a few minutes on the full split; it depends only on
    # the corpus (not the arm), so it is cached and reused across all four arms and seeds.
    stats = corpus_stats_cached(
        train_ds, args.tensorized_dir / "train", split="train",
        sample_windows=args.stats_sample_windows, seed=args.seed,
        max_seq_len=args.max_seq_len)
    if args.tau_max is not None:
        tau_max = float(args.tau_max)
        tau_source = f"--tau_max override ({args.tau_max})"
    else:
        tau_max = stats.tau_max
        tau_source = stats.tau_max_source
    age_mean, age_sd = stats.event_age_mean, stats.event_age_sd

    # -- model --------------------------------------------------------------- #
    model = DKMModel(
        num_codes=train_ds.num_codes, embedding_path=args.embedding_path, arm=args.arm,
        seed=args.seed, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        use_residual=use_residual, use_layernorm=use_layernorm, use_ffn=use_ffn, s=args.s,
        tau_max=tau_max, age_M=args.age_M, age_p_min=args.age_p_min, age_p_max=args.age_p_max,
        age_hidden=args.age_hidden, gen_final_bias=args.gen_final_bias,
        center_delta_alpha=args.center_delta_alpha, demo_dim=demo_dim,
        demo_channels=demo_channels, race_encoding=args.race_encoding,
        demo_hidden=args.demo_hidden, age_mean=age_mean, age_sd=age_sd,
    ).to(device)
    # HARD (Fix A): tau_max is the EXACT full-split maximum with float32 headroom, so the
    # window that defined it must not clamp. Verify on that maximum directly.
    if args.tau_max is None:
        model.reset_clamp_stats()
        probe_site = model.kernel_sites()[0][1].kernel
        probe_site.rescale(torch.tensor([stats.span_days_max_tau], dtype=torch.float32,
                                        device=probe_site.tau_max.device))
        if probe_site.clamp_fraction != 0.0:
            raise AssertionError(
                f"[HARD] the exact corpus tau_max={tau_max!r} clamps its own defining window "
                f"(span_tau={stats.span_days_max_tau!r}); float32 headroom is insufficient.")
        model.reset_clamp_stats()

    groups, group_report = build_param_groups(model, args.lr_backbone, args.lr_age, args.lr_head)
    opt = torch.optim.Adam(groups)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    theta0 = D.snapshot_parameters(groups)

    # -- artifacts written BEFORE the first step ----------------------------- #
    band = characterize_band(LogAgeFourier(M=args.age_M, p_min=args.age_p_min,
                                           p_max=args.age_p_max), M=args.age_M)
    legacy_band = characterize_band(LinearAgeFourier(M=args.age_M), M=args.age_M)
    params = model.parameter_report()
    config = {
        "run_id": args.run_name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": args.seed,
        "arm": args.arm,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": model.config_dict(),
        "params": params,
        "optim": {**group_report, "batch_size": args.batch_size, "epochs": args.epochs,
                  "optimizer": "Adam", "grad_clip": args.grad_clip, "amp": use_amp},
        "data": {
            "paths": {"tensorized_dir": str(args.tensorized_dir),
                      "embedding_path": str(args.embedding_path),
                      "vocab_path": str(args.vocab_path)},
            "split_sizes": {"train": len(train_ds), "val": len(val_ds)},
            "vocab_size": train_ds.num_codes,
            "max_seq_len": args.max_seq_len,
            "tau_max": tau_max,
            "tau_max_source": tau_source,
            "corpus_stats": stats.to_json(),
        },
        "band_characterization": {"log_age": band, "legacy_linear": legacy_band},
        "primary_endpoint": {
            "dataset": args.endpoint_dataset,
            "task": args.endpoint_task,
            "metric": args.endpoint_metric,
            "comparison": ("kernel vs random_constant is the identifying comparison (exactly "
                           "parameter-matched); vanilla is the floor; additive is an "
                           "alternative delivery site with a different architecture"),
            "age_bands": D.band_names(),
            "declared_before_run": True,
        },
        "deviations_from_draft": DEVIATIONS_FROM_DRAFT,
        "env": {"torch": torch.__version__, "python": sys.version.split()[0],
                "device": str(device), "hip": getattr(torch.version, "hip", None),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "figure_1a_note": (
            "Figure 1A depicts a transformer encoder stack (LN -> attention -> Add&Norm -> FFN "
            "-> Add&Norm). This run used n_layers=%d, residual=%s, layernorm=%s, ffn=%s."
            % (args.n_layers, use_residual, use_layernorm, use_ffn)),
    }
    D.write_json(run_dir / "config.json", config)

    # Condition numbers must be computed on the EMPIRICAL lag distribution, not a uniform
    # sweep of [0, tau_max] -- a uniform sample flatters both bases.
    tau_sample = sample_empirical_taus(train_ds, n_examples=400, seed=args.seed)
    D.write_json(run_dir / "paper_numbers.json", {
        "arm": args.arm,
        "age_pathway_params": params["age"],
        "age_share_of_encoder": params["age_share_of_encoder"],
        "params_by_group": {k: params[k] for k in ("backbone", "age", "head", "total_trainable")},
        "vocab_size": train_ds.num_codes,
        "tau_max": tau_max,
        "tau_max_source": tau_source,
        "age_standardization": {"mean": age_mean, "sd": age_sd},
        "condition_numbers": {**D.gram_condition_numbers(tau_sample, args.s, tau_max),
                              "distribution": "empirical within-window pairwise lags"},
        "age_band_n_pretrain_events": stats.event_age_band_counts,
        "youngest_event_age": stats.event_age_min,
        "events_under_18": {"count": stats.event_age_under_18_count,
                            "fraction": stats.event_age_under_18_fraction},
        "band_characterization": {"log_age": band, "legacy_linear": legacy_band},
    })

    D.print_config_summary(config)
    D.print_band_characterization(band, legacy_band)

    # -- train --------------------------------------------------------------- #
    records: list[dict] = []
    train_json = run_dir / "train.json"
    step = 0
    step_losses: list[float] = []
    t_start = time.time()
    diag_batch = None
    model.train()

    for epoch in range(1, args.epochs + 1):
        model.reset_clamp_stats()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            if diag_batch is None:
                diag_batch = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                              for k, v in batch.items()}
            ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
            with ctx:
                out = model(batch)
                loss = F.binary_cross_entropy_with_logits(out["code_logits"].float(),
                                                          batch["target_codes"])
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
            step += 1
            step_losses.append(float(loss))

            if step == args.report_at_step:
                _step_report(model, groups, theta0, diag_batch, step_losses, step, args.arm)
                if args.stop_after_report:
                    D.print_block("stopping after the step report as requested",
                                  [f"run_dir: {run_dir}"])
                    return 0
            if args.max_steps and step >= args.max_steps:
                break

        # The epoch-end record runs even when --max_steps cut the epoch short, so a short
        # run still exercises evaluation, the diagnostics, train.json and the checkpoint.
        val = evaluate(model, val_loader, device, use_amp, args.val_max_batches)
        ages_flat = diag_batch["age_years"][diag_batch["attention_mask"]].detach()
        record = {
            "epoch": epoch,
            "step": step,
            "wall_clock_s": time.time() - t_start,
            "train_loss": running / max(1, n_batches),
            "val_loss": val["loss"],
            "micro_auroc": val["micro_auroc"],
            "micro_auroc_tag": "diagnostic_only",
            **{f"recall@{k}": val[f"recall@{k}"] for k in (5, 10, 20)},
            "recall_by_band": val["by_band"],
            "alpha": D.alpha_diagnostics(model, ages_flat),
            "delta_alpha_grid": D.delta_alpha_grid(model),
            "w_curves": D.w_curves(model),
            "param_drift": D.parameter_drift(groups, theta0),
            "clamp_rate": D.clamp_rates(model),
            "alpha_l1": D.alpha_l1(model),
            "attention": D.attention_stats(model, diag_batch),
            "lr": {g["name"]: g["lr"] for g in opt.param_groups},
        }
        records.append(record)
        D.append_train_json(train_json, records)
        D.print_epoch(record)

        torch.save({
            "epoch": epoch, "arm": args.arm, "seed": args.seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "tau_max": tau_max,            # the single source of truth at fine-tune (D8)
            "tau_max_source": tau_source,
            "age_standardization": {"mean": age_mean, "sd": age_sd},  # frozen (Fix C, INV-AGESTD)
            "config": config,
        }, run_dir / f"epoch_{epoch:03d}.pt")

        if args.max_steps and step >= args.max_steps:
            D.print_block("stopping: --max_steps reached", [f"steps={step}"])
            break

    D.print_block("done", [f"run_dir : {run_dir}",
                           f"epochs  : {len(records)}",
                           f"steps   : {step}",
                           f"wall    : {time.time() - t_start:.1f}s"])
    return 0


def _step_report(model, groups, theta0, batch, step_losses: list[float], step: int,
                 arm: str) -> None:
    """The Phase 11 early check: is the age pathway alive, and is the loss descending?"""
    ages = batch["age_years"][batch["attention_mask"]].detach()
    alpha = D.alpha_diagnostics(model, ages)
    drift = D.parameter_drift(groups, theta0)
    w = max(1, len(step_losses) // 4)
    first, last = float(np.mean(step_losses[:w])), float(np.mean(step_losses[-w:]))
    lines = [
        f"arm={arm}  step={step}",
        f"train loss  first {w} steps = {first:.6f}   last {w} steps = {last:.6f}   "
        f"delta = {last - first:+.6f}",
        f"            step 1 = {step_losses[0]:.6f}   min = {min(step_losses):.6f}   "
        f"descending = {last < first}",
        "",
        "param drift ||theta_t - theta_0|| / ||theta_0||:",
    ]
    for name, v in drift.items():
        lines.append(f"    {name:<10}: {'n/a (empty group)' if v is None else f'{v:.6e}'}")
    if arm != "vanilla" and (drift.get("age") in (None, 0.0)):
        lines.append("    ^ AGE DRIFT IS ZERO -- the age pathway is not training. Investigate")
        lines.append("      before committing to a multi-day run.")
    lines.append("")
    for site, st in alpha.items():
        lines.append(f"[{site}] Delta-alpha decomposition over valid events:")
        lines.append(f"    ||d_alpha|| mean={st['delta_alpha_norm_mean']:.6e} "
                     f"std={st['delta_alpha_norm_std']:.6e}")
        lines.append(f"    constant={st['constant_component']:.6e}   "
                     f"varying={st['varying_component']:.6e}")
        lines.append(f"    alpha_base={np.round(st['alpha_base'], 5).tolist()}  "
                     f"clamp={st['clamp_fraction']:.3e}")
    lines += [
        "",
        "Reading this: the varying component is what says the mechanism is alive. A growing",
        "constant component with a flat varying component means the MLP learned an offset,",
        "which alpha_base absorbs. Gradient norm is NOT the signal here -- the generator's",
        "zero-init final layer gives its first layer zero gradient at step 0, and Adam's",
        "second-moment normalisation turns a tiny gradient into a full-size step anyway.",
    ]
    D.print_block(f"step {step} age-pathway liveness report  [MEASURE]", lines)


if __name__ == "__main__":
    raise SystemExit(main())
