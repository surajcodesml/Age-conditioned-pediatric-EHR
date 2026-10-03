"""Runner for the predefined architecture ladder.

Independent seeds can run concurrently. Two jobs is the measured peak on the
R9700 for these small models; more jobs contend and slow each step down.
Batch size, learning rate, epoch budget, and seeds are not changed.

Usage (ehr environment):
    python -m ladder.run_ladder --mode smoke
    python -m ladder.run_ladder --mode full --jobs 2
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any

import torch

from baselines.common.training import get_device, set_seed

from ladder import ARTIFACT_ROOT
from ladder.cached_data import cached_dtr_loaders
from ladder.artifacts import (
    file_sha256,
    git_commit,
    run_directory,
    stage_a_directory,
    utc_now,
    write_config_yaml,
    write_json,
)
from ladder.config import experiment_configs
from ladder.evaluation.aggregate import aggregate
from ladder.evaluation.metrics import compute_run_metrics
from ladder.inference.predict import run_inference
from ladder.models.factory import build_model
from ladder.report import write_report
from ladder.training.loop import (
    finalize_checkpoint,
    flatten_history,
    fork_matched_arms,
    forward_logits,
    max_state_diff,
    train_joint_temporal_group,
    train_standard,
    train_temporal_only_group,
)
from ladder.visualization.figures import make_all_figures

ARMS = ("age_temporal", "temporal_only")


def _manifest(
    cfg: dict[str, Any],
    *,
    scenario: str,
    arm: str,
    seed: int,
    checkpoint: Path,
    n_trainable: int,
) -> dict[str, Any]:
    keys = (
        "lr", "weight_decay", "max_epochs", "patience", "min_epochs", "grad_clip",
        "batch_size", "d_model", "dropout", "lambda_init", "staged", "stage_b_epochs",
        "temporal_lr_mult", "temporal_weight_decay", "aggregation", "n_channels",
        "n_components", "knots", "data_seed", "seeds",
    )
    return {
        "experiment_id": cfg["experiment_id"],
        "architecture": cfg["architecture"],
        "variant": arm,
        "arm": arm,
        "dataset": cfg["dataset"],
        "scenario": scenario,
        "seed": int(seed),
        "split": "patient_split:data_seed=20260922:train=0.70,val=0.15,test=0.15",
        "git_commit": git_commit(),
        "config_path": cfg.get("config_path"),
        "config_hash": cfg.get("config_hash"),
        "checkpoint_path": str(checkpoint),
        "timestamp": utc_now(),
        "training_settings": {key: cfg.get(key) for key in keys},
        "n_trainable": int(n_trainable),
        "hypothesis": cfg.get("hypothesis", "").strip(),
    }


def _complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in (
        "checkpoint_best.pt", "predictions.parquet", "metrics.json", "mechanism_metrics.json",
    ))


def _infer_and_metrics(cfg, run_dir: Path, *, scenario: str, device: torch.device, limit_batches: int | None) -> None:
    """Score an existing checkpoint. Does not rewrite checkpoint_best.pt."""
    checkpoint = run_dir / "checkpoint_best.pt"
    if not (run_dir / "predictions.parquet").exists():
        run_inference(
            checkpoint=checkpoint,
            scenario=scenario,
            run_dir=run_dir,
            data_seed=int(cfg["data_seed"]),
            device=device,
            shuffle_seed=int(cfg["age_shuffle_seed"]),
            batch_size=int(cfg["batch_size"]),
            max_test_batches=limit_batches,
        )
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest["predictions_sha256"] = file_sha256(run_dir / "predictions.parquet")
        manifest["predictions_timestamp"] = utc_now()
        write_json(manifest_path, manifest)
    if not (run_dir / "metrics.json").exists():
        compute_run_metrics(run_dir, n_boot=20 if limit_batches else 200)


def _finish_run(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    run_dir: Path,
    *,
    scenario: str,
    arm: str,
    seed: int,
    n_codes: int,
    n_targets: int,
    history_rows: list[dict[str, Any]],
    device: torch.device,
    limit_batches: int | None,
) -> None:
    age_temporal = arm == "age_temporal"
    checkpoint = finalize_checkpoint(
        model, run_dir, cfg,
        n_codes=n_codes, n_targets=n_targets, age_temporal=age_temporal,
        history_rows=history_rows,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    manifest = _manifest(cfg, scenario=scenario, arm=arm, seed=seed, checkpoint=checkpoint, n_trainable=n_trainable)
    write_json(run_dir / "manifest.json", manifest)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _infer_and_metrics(cfg, run_dir, scenario=scenario, device=device, limit_batches=limit_batches)


def _train_arm(
    cfg: dict[str, Any],
    *,
    scenario: str,
    arm: str,
    seed: int,
    n_codes: int,
    n_targets: int,
    train_loader,
    val_loader,
    root: Path,
    device: torch.device,
    limit_batches: int | None,
    staged_state: dict[str, torch.Tensor] | None,
    stage_a_history: list[dict[str, Any]] | None,
) -> None:
    run_dir = run_directory(root, cfg["experiment_id"], scenario, arm, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    if _complete(run_dir):
        print(f"skip complete {run_dir}", flush=True)
        return
    age_temporal = arm == "age_temporal"
    if (run_dir / "checkpoint_best.pt").exists():
        _infer_and_metrics(cfg, run_dir, scenario=scenario, device=device, limit_batches=limit_batches)
        return

    dev = str(device)
    if staged_state is None:
        set_seed(int(seed))
        model = build_model(cfg, n_codes, n_targets, age_temporal=age_temporal).to(device)
        result = train_standard(
            model, train_loader, val_loader,
            cfg=cfg, run_dir=run_dir, seed=int(seed), device=dev, limit_batches=limit_batches,
        )
        history_rows = flatten_history([("train", result)])
    else:
        model = build_model(cfg, n_codes, n_targets, age_temporal=age_temporal)
        model.load_state_dict(staged_state)
        model.configure_arm_()
        with torch.no_grad():
            model.beta_param().zero_()
        model.to(device)
        stage_b = train_temporal_only_group(
            model, train_loader, val_loader,
            cfg=cfg, run_dir=run_dir, seed=int(seed), device=dev,
            epochs=int(cfg["stage_b_epochs"]), limit_batches=limit_batches,
        )
        stage_c = train_joint_temporal_group(
            model, train_loader, val_loader,
            cfg=cfg, run_dir=run_dir, seed=int(seed), device=dev, limit_batches=limit_batches,
        )
        history_rows = list(stage_a_history or [])
        history_rows.extend(flatten_history([("B", stage_b), ("C", stage_c)]))
    _finish_run(
        model, cfg, run_dir,
        scenario=scenario, arm=arm, seed=seed,
        n_codes=n_codes, n_targets=n_targets,
        history_rows=history_rows, device=device, limit_batches=limit_batches,
    )


def _stage_a(
    cfg: dict[str, Any],
    *,
    scenario: str,
    seed: int,
    n_codes: int,
    n_targets: int,
    train_loader,
    val_loader,
    root: Path,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    run_dir = stage_a_directory(root, cfg["experiment_id"], scenario, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint_best.pt"
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        history = []
        if (run_dir / "history.json").exists():
            history = json.loads((run_dir / "history.json").read_text())
        return ckpt["state_dict"], history
    set_seed(int(seed))
    model = build_model(cfg, n_codes, n_targets, age_temporal=False).to(device)
    result = train_standard(
        model, train_loader, val_loader,
        cfg=cfg, run_dir=run_dir, seed=int(seed), device=str(device), limit_batches=limit_batches,
    )
    history = flatten_history([("A", result)])
    finalize_checkpoint(
        model, run_dir, cfg,
        n_codes=n_codes, n_targets=n_targets, age_temporal=False, history_rows=history,
    )
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    return state, history


def _execute_seed(
    cfg: dict[str, Any],
    *,
    scenario: str,
    seed: int,
    root: Path,
    device: torch.device,
    limit_batches: int | None,
) -> list[dict[str, str]]:
    """Train both arms for one seed. Returns failure records."""
    failures: list[dict[str, str]] = []
    root = Path(root)
    if all(
        _complete(run_directory(root, cfg["experiment_id"], scenario, arm, seed))
        for arm in ARMS
    ):
        print(f"skip complete {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
        return failures
    print(f"loading {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
    train_loader, val_loader, _test, _vocab, info = cached_dtr_loaders(
        scenario,
        data_seed=int(cfg["data_seed"]),
        batch_size=int(cfg["batch_size"]),
    )
    n_codes = int(info["n_codes"])
    n_targets = int(info["n_targets"])
    try:
        if cfg.get("staged"):
            state, history = _stage_a(
                cfg, scenario=scenario, seed=seed,
                n_codes=n_codes, n_targets=n_targets,
                train_loader=train_loader, val_loader=val_loader,
                root=root, device=device, limit_batches=limit_batches,
            )
            at, to = fork_matched_arms(state, cfg, n_codes, n_targets)
            diff = max_state_diff(at, to)
            if diff > 0.0:
                raise RuntimeError(f"E02 fork state diff {diff}")
            batch = next(iter(val_loader))
            at.to(device)
            to.to(device)
            with torch.no_grad():
                left = forward_logits(at, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
                right = forward_logits(to, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            gap = float((left - right).abs().max().item())
            if gap > 1e-5:
                raise RuntimeError(f"E02 fork logit gap {gap}")
            write_json(
                stage_a_directory(root, cfg["experiment_id"], scenario, seed) / "fork_check.json",
                {"state_diff": diff, "logit_gap_before_stage_b": gap},
            )
            del at, to
            for arm in ARMS:
                print(f"train {cfg['experiment_id']} {scenario} {arm} seed {seed}", flush=True)
                _train_arm(
                    cfg, scenario=scenario, arm=arm, seed=seed,
                    n_codes=n_codes, n_targets=n_targets,
                    train_loader=train_loader, val_loader=val_loader,
                    root=root, device=device, limit_batches=limit_batches,
                    staged_state=state, stage_a_history=history,
                )
        else:
            for arm in ARMS:
                try:
                    print(f"train {cfg['experiment_id']} {scenario} {arm} seed {seed}", flush=True)
                    _train_arm(
                        cfg, scenario=scenario, arm=arm, seed=seed,
                        n_codes=n_codes, n_targets=n_targets,
                        train_loader=train_loader, val_loader=val_loader,
                        root=root, device=device, limit_batches=limit_batches,
                        staged_state=None, stage_a_history=None,
                    )
                except Exception as exc:
                    message = traceback.format_exc()
                    print(message, flush=True)
                    failures.append({
                        "experiment": cfg["experiment_id"],
                        "scenario": scenario,
                        "arm": arm,
                        "seed": str(seed),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": message,
                    })
    except Exception as exc:
        message = traceback.format_exc()
        print(message, flush=True)
        failures.append({
            "experiment": cfg["experiment_id"],
            "scenario": scenario,
            "arm": "both" if cfg.get("staged") else "see traceback",
            "seed": str(seed),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": message,
        })
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return failures


def _seed_job(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Process-pool entry. Spawned workers each keep their own tensor cache."""
    device = get_device(str(payload["device"]))
    return _execute_seed(
        payload["cfg"],
        scenario=str(payload["scenario"]),
        seed=int(payload["seed"]),
        root=Path(payload["root"]),
        device=device,
        limit_batches=payload["limit_batches"],
    )


def run_configs(
    configs: list[dict[str, Any]],
    *,
    root: Path,
    device: torch.device,
    limit_batches: int | None,
    scenarios: list[str] | None,
    seeds: list[int] | None,
    jobs: int = 1,
) -> list[dict[str, str]]:
    pending: list[dict[str, Any]] = []
    for cfg in configs:
        use_scenarios = scenarios or list(cfg["scenarios"])
        use_seeds = seeds or [int(s) for s in cfg["seeds"]]
        print(f"\n=== {cfg['experiment_id']} scenarios={use_scenarios} seeds={use_seeds} ===", flush=True)
        for scenario in use_scenarios:
            for seed in use_seeds:
                if all(
                    _complete(run_directory(root, cfg["experiment_id"], scenario, arm, seed))
                    for arm in ARMS
                ):
                    print(f"skip complete {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
                    continue
                pending.append({
                    "cfg": cfg,
                    "scenario": scenario,
                    "seed": int(seed),
                    "root": str(root),
                    "device": str(device),
                    "limit_batches": limit_batches,
                })
    print(f"pending seeds: {len(pending)}  jobs: {jobs}", flush=True)
    failures: list[dict[str, str]] = []
    if not pending:
        return failures
    if jobs <= 1:
        for payload in pending:
            failures.extend(_seed_job(payload))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(jobs)) as pool:
            for batch in pool.imap_unordered(_seed_job, pending):
                failures.extend(batch)
    if failures:
        write_json(root / "failures.json", failures)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthea DTR architecture ladder")
    parser.add_argument("--mode", choices=["smoke", "full", "aggregate"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Concurrent seeds on one GPU. 2 is the measured peak for these models.",
    )
    args = parser.parse_args()
    device = get_device(args.device)
    configs = experiment_configs()
    if args.mode == "smoke":
        root = args.artifact_root or (ARTIFACT_ROOT / "_smoke")
        for cfg in configs:
            cfg["max_epochs"] = 1
            cfg["min_epochs"] = 1
            cfg["patience"] = 1
            cfg["stage_b_epochs"] = 1
        failures = run_configs(
            configs, root=root, device=device, limit_batches=2,
            scenarios=["S2"], seeds=[0], jobs=1,
        )
    elif args.mode == "aggregate":
        root = args.artifact_root or ARTIFACT_ROOT
        aggregate(root)
        make_all_figures(root)
        path = write_report(root)
        print(f"wrote {path}", flush=True)
        return
    else:
        root = args.artifact_root or ARTIFACT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        write_config_yaml(root / "protocol_seeds.yaml", {
            "seeds": configs[0]["seeds"],
            "source": "existing cehrbert_small seed list; fixed before ladder runs",
            "git_commit": git_commit(),
            "timestamp": utc_now(),
            "jobs": int(args.jobs),
            "data": "cached encounter tensors, split-max padding, masked; shuffle from trainer seed",
            "unchanged": ["batch_size", "lr", "weight_decay", "max_epochs", "patience", "min_epochs", "seeds"],
        })
        failures = run_configs(
            configs, root=root, device=device, limit_batches=None,
            scenarios=None, seeds=None, jobs=int(args.jobs),
        )
    print("aggregating", flush=True)
    aggregate(root)
    make_all_figures(root)
    report_path = root / "smoke_report.md" if args.mode == "smoke" else None
    path = write_report(root, report_path=report_path)
    print(f"wrote {path}", flush=True)
    if failures:
        print(f"{len(failures)} failures recorded", flush=True)


if __name__ == "__main__":
    main()
