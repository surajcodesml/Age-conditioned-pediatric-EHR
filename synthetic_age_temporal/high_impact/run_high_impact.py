"""Run D00–D02. Detach-friendly. Seeds run two at a time.

Usage (ehr environment):
    python -m high_impact.run_high_impact --mode smoke
    python -m high_impact.run_high_impact --mode full --jobs 2
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
from baselines.synthetic.data_adapter import scenario_dir
from ladder.artifacts import (
    file_sha256,
    git_commit,
    run_directory,
    stage_a_directory,
    utc_now,
    write_config_yaml,
    write_json,
)
from ladder.cached_data import cached_dtr_loaders
from ladder.training.loop import (
    finalize_checkpoint,
    flatten_history,
    forward_logits,
    max_state_diff,
    train_joint_temporal_group,
    train_standard,
    train_temporal_only_group,
)

from high_impact import ARTIFACT_ROOT
from high_impact.aggregate import aggregate
from high_impact.config import experiment_configs
from high_impact.figures import make_all_figures
from high_impact.inference import run_inference
from high_impact.metrics import compute_run_metrics
from high_impact.models import build_high_impact
from high_impact.report import write_report

ARMS = ("age_temporal", "temporal_only")


def _complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in (
        "checkpoint_best.pt", "predictions.parquet", "metrics.json", "mechanism_metrics.json",
    ))


def _oracle_params(scenario: str, data_seed: int) -> tuple[float, float]:
    meta = json.loads((scenario_dir(scenario, data_seed) / "meta.json").read_text())
    return float(meta["theta0"]), float(meta["beta_true"])


def _build(cfg, n_codes, n_targets, *, age_temporal, scenario, data_seed):
    theta0, beta = _oracle_params(scenario, data_seed)
    model = build_high_impact(
        cfg, n_codes, n_targets, age_temporal=age_temporal,
        oracle_theta0=theta0, oracle_beta=beta,
    )
    model.set_oracle_(theta0, beta)
    return model, theta0, beta


def _manifest(cfg, *, scenario, arm, seed, checkpoint, n_trainable, oracle_theta0, oracle_beta):
    return {
        "experiment_id": cfg["experiment_id"],
        "architecture": cfg["variant"],
        "variant": cfg["variant"],
        "arm": arm,
        "dataset": cfg["dataset"],
        "scenario": scenario,
        "seed": int(seed),
        "git_commit": git_commit(),
        "config_hash": cfg.get("config_hash"),
        "checkpoint": str(checkpoint),
        "timestamp": utc_now(),
        "n_trainable": int(n_trainable),
        "oracle_theta0": oracle_theta0,
        "oracle_beta": oracle_beta,
        "training": {
            "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "batch_size": cfg["batch_size"],
            "max_epochs": cfg["max_epochs"], "patience": cfg["patience"], "min_epochs": cfg["min_epochs"],
            "staged": bool(cfg.get("staged")),
        },
    }


def _infer_and_metrics(cfg, run_dir, *, scenario, device, limit_batches):
    if not (run_dir / "predictions.parquet").exists():
        run_inference(
            checkpoint=run_dir / "checkpoint_best.pt",
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
        write_json(manifest_path, manifest)
    if not (run_dir / "metrics.json").exists():
        compute_run_metrics(run_dir, n_boot=20 if limit_batches else 200)


def _save_ckpt(model, run_dir, cfg, *, n_codes, n_targets, age_temporal, history_rows, oracle_theta0, oracle_beta):
    path = run_dir / "checkpoint_best.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": cfg,
            "n_codes": int(n_codes),
            "n_targets": int(n_targets),
            "age_temporal": bool(age_temporal),
            "architecture": getattr(model, "architecture", cfg.get("variant")),
            "oracle_theta0": float(oracle_theta0),
            "oracle_beta": float(oracle_beta),
        },
        path,
    )
    write_json(run_dir / "history.json", history_rows)
    write_config_yaml(run_dir / "config.yaml", cfg)
    return path


def _finish(model, cfg, run_dir, *, scenario, arm, seed, n_codes, n_targets, history_rows, device, limit_batches, oracle_theta0, oracle_beta):
    checkpoint = _save_ckpt(
        model, run_dir, cfg, n_codes=n_codes, n_targets=n_targets,
        age_temporal=(arm == "age_temporal"), history_rows=history_rows,
        oracle_theta0=oracle_theta0, oracle_beta=oracle_beta,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(run_dir / "manifest.json", _manifest(
        cfg, scenario=scenario, arm=arm, seed=seed, checkpoint=checkpoint,
        n_trainable=n_trainable, oracle_theta0=oracle_theta0, oracle_beta=oracle_beta,
    ))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _infer_and_metrics(cfg, run_dir, scenario=scenario, device=device, limit_batches=limit_batches)


def _train_arm(cfg, *, scenario, arm, seed, n_codes, n_targets, train_loader, val_loader, root, device, limit_batches, staged_state, stage_a_history, oracle_theta0, oracle_beta):
    run_dir = run_directory(root, cfg["experiment_id"], scenario, arm, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    if _complete(run_dir):
        print(f"skip complete {run_dir}", flush=True)
        return
    if (run_dir / "checkpoint_best.pt").exists():
        _infer_and_metrics(cfg, run_dir, scenario=scenario, device=device, limit_batches=limit_batches)
        return
    age_temporal = arm == "age_temporal"
    dev = str(device)
    if staged_state is None:
        set_seed(int(seed))
        model, theta0, beta = _build(cfg, n_codes, n_targets, age_temporal=age_temporal, scenario=scenario, data_seed=int(cfg["data_seed"]))
        model.to(device)
        result = train_standard(model, train_loader, val_loader, cfg=cfg, run_dir=run_dir, seed=int(seed), device=dev, limit_batches=limit_batches)
        history_rows = flatten_history([("train", result)])
    else:
        model, theta0, beta = _build(cfg, n_codes, n_targets, age_temporal=age_temporal, scenario=scenario, data_seed=int(cfg["data_seed"]))
        model.load_state_dict(staged_state)
        model.set_oracle_(oracle_theta0, oracle_beta)
        model.configure_arm_()
        if model.variant != "oracle_gate":
            with torch.no_grad():
                model._active_beta().zero_()
                if model.variant == "multihead_dev":
                    model.delta.zero_()
        model.to(device)
        history_rows = list(stage_a_history or [])
        temporal = model.temporal_parameters()
        if temporal:
            stage_b = train_temporal_only_group(
                model, train_loader, val_loader, cfg=cfg, run_dir=run_dir, seed=int(seed),
                device=dev, epochs=int(cfg["stage_b_epochs"]), limit_batches=limit_batches,
            )
            history_rows.extend(flatten_history([("B", stage_b)]))
        else:
            write_json(run_dir / "stage_b_skipped.json", {"reason": "no trainable temporal parameters"})
        stage_c = train_joint_temporal_group(
            model, train_loader, val_loader, cfg=cfg, run_dir=run_dir, seed=int(seed),
            device=dev, limit_batches=limit_batches,
        )
        history_rows.extend(flatten_history([("C", stage_c)]))
    _finish(
        model, cfg, run_dir, scenario=scenario, arm=arm, seed=seed,
        n_codes=n_codes, n_targets=n_targets, history_rows=history_rows,
        device=device, limit_batches=limit_batches, oracle_theta0=theta0, oracle_beta=beta,
    )


def _stage_a(cfg, *, scenario, seed, n_codes, n_targets, train_loader, val_loader, root, device, limit_batches):
    run_dir = stage_a_directory(root, cfg["experiment_id"], scenario, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint_best.pt"
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        history = json.loads((run_dir / "history.json").read_text()) if (run_dir / "history.json").exists() else []
        return ckpt["state_dict"], history, float(ckpt["oracle_theta0"]), float(ckpt["oracle_beta"])
    set_seed(int(seed))
    # Stage A always trains under temporal_only settings / oracle beta=0 path.
    model, theta0, beta = _build(cfg, n_codes, n_targets, age_temporal=False, scenario=scenario, data_seed=int(cfg["data_seed"]))
    model.to(device)
    result = train_standard(
        model, train_loader, val_loader, cfg=cfg, run_dir=run_dir, seed=int(seed),
        device=str(device), limit_batches=limit_batches,
    )
    history = flatten_history([("A", result)])
    _save_ckpt(
        model, run_dir, cfg, n_codes=n_codes, n_targets=n_targets, age_temporal=False,
        history_rows=history, oracle_theta0=theta0, oracle_beta=beta,
    )
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    return state, history, theta0, beta


def _execute_seed(cfg, *, scenario, seed, root, device, limit_batches):
    failures = []
    root = Path(root)
    if all(_complete(run_directory(root, cfg["experiment_id"], scenario, arm, seed)) for arm in ARMS):
        print(f"skip complete {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
        return failures
    print(f"loading {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
    train_loader, val_loader, _test, _vocab, info = cached_dtr_loaders(
        scenario, data_seed=int(cfg["data_seed"]), batch_size=int(cfg["batch_size"]),
    )
    n_codes, n_targets = int(info["n_codes"]), int(info["n_targets"])
    try:
        state, history, theta0, beta = _stage_a(
            cfg, scenario=scenario, seed=seed, n_codes=n_codes, n_targets=n_targets,
            train_loader=train_loader, val_loader=val_loader, root=root, device=device,
            limit_batches=limit_batches,
        )
        # After Stage A, arms must match with developmental slope inactive.
        # Oracle-gate injects beta_true when age_temporal=True, so zero it for the check.
        left, _, _ = _build(cfg, n_codes, n_targets, age_temporal=True, scenario=scenario, data_seed=int(cfg["data_seed"]))
        right, _, _ = _build(cfg, n_codes, n_targets, age_temporal=False, scenario=scenario, data_seed=int(cfg["data_seed"]))
        left.load_state_dict(state)
        right.load_state_dict(state)
        left.configure_arm_()
        right.configure_arm_()
        if left.variant == "oracle_gate":
            left.zero_all_betas_()
        diff = max_state_diff(left, right)
        if diff > 0:
            raise RuntimeError(f"fork state diff {diff}")
        batch = next(iter(val_loader))
        left.to(device)
        right.to(device)
        with torch.no_grad():
            a = forward_logits(left, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            b = forward_logits(right, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
        gap = float((a - b).abs().max().item())
        if gap > 1e-5:
            raise RuntimeError(f"fork logit gap {gap}")
        write_json(
            stage_a_directory(root, cfg["experiment_id"], scenario, seed) / "fork_check.json",
            {"state_diff": diff, "logit_gap_before_stage_b": gap},
        )
        del left, right
        for arm in ARMS:
            print(f"train {cfg['experiment_id']} {scenario} {arm} seed {seed}", flush=True)
            try:
                _train_arm(
                    cfg, scenario=scenario, arm=arm, seed=seed, n_codes=n_codes, n_targets=n_targets,
                    train_loader=train_loader, val_loader=val_loader, root=root, device=device,
                    limit_batches=limit_batches, staged_state=state, stage_a_history=history,
                    oracle_theta0=theta0, oracle_beta=beta,
                )
            except Exception as exc:
                message = traceback.format_exc()
                print(message, flush=True)
                failures.append({
                    "experiment": cfg["experiment_id"], "scenario": scenario, "arm": arm,
                    "seed": str(seed), "error": f"{type(exc).__name__}: {exc}", "traceback": message,
                })
    except Exception as exc:
        message = traceback.format_exc()
        print(message, flush=True)
        failures.append({
            "experiment": cfg["experiment_id"], "scenario": scenario, "arm": "staged",
            "seed": str(seed), "error": f"{type(exc).__name__}: {exc}", "traceback": message,
        })
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return failures


def _seed_job(payload):
    return _execute_seed(
        payload["cfg"], scenario=str(payload["scenario"]), seed=int(payload["seed"]),
        root=Path(payload["root"]), device=get_device(str(payload["device"])),
        limit_batches=payload["limit_batches"],
    )


def run_configs(configs, *, root, device, limit_batches, scenarios, seeds, jobs):
    pending = []
    for cfg in configs:
        use_scenarios = scenarios or list(cfg["scenarios"])
        use_seeds = seeds or [int(s) for s in cfg["seeds"]]
        print(f"\n=== {cfg['experiment_id']} scenarios={use_scenarios} seeds={use_seeds} ===", flush=True)
        for scenario in use_scenarios:
            for seed in use_seeds:
                if all(_complete(run_directory(root, cfg["experiment_id"], scenario, arm, seed)) for arm in ARMS):
                    print(f"skip complete {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
                    continue
                pending.append({
                    "cfg": cfg, "scenario": scenario, "seed": int(seed),
                    "root": str(root), "device": str(device), "limit_batches": limit_batches,
                })
    print(f"pending seeds: {len(pending)}  jobs: {jobs}", flush=True)
    failures = []
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
    parser = argparse.ArgumentParser(description="High-impact DTR follow-up")
    parser.add_argument("--mode", choices=["smoke", "full", "aggregate"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    device = get_device(args.device)
    configs = experiment_configs()
    if args.mode == "aggregate":
        root = args.artifact_root or ARTIFACT_ROOT
        aggregate(root)
        make_all_figures(root)
        print(f"wrote {write_report(root)}", flush=True)
        return
    if args.mode == "smoke":
        root = args.artifact_root or (ARTIFACT_ROOT / "_smoke")
        for cfg in configs:
            cfg["max_epochs"] = cfg["min_epochs"] = cfg["patience"] = cfg["stage_b_epochs"] = 1
        failures = run_configs(configs, root=root, device=device, limit_batches=2, scenarios=["S2"], seeds=[0], jobs=1)
        report_path = root / "smoke_report.md"
    else:
        root = args.artifact_root or ARTIFACT_ROOT
        root.mkdir(parents=True, exist_ok=True)
        write_config_yaml(root / "protocol_seeds.yaml", {
            "seeds": configs[0]["seeds"],
            "baseline": "C01_staged_current",
            "git_commit": git_commit(),
            "timestamp": utc_now(),
            "jobs": int(args.jobs),
        })
        failures = run_configs(configs, root=root, device=device, limit_batches=None, scenarios=None, seeds=None, jobs=int(args.jobs))
        report_path = None
    print("aggregating", flush=True)
    aggregate(root)
    make_all_figures(root)
    print(f"wrote {write_report(root, report_path=report_path)}", flush=True)
    if failures:
        print(f"{len(failures)} failures recorded", flush=True)


if __name__ == "__main__":
    main()
