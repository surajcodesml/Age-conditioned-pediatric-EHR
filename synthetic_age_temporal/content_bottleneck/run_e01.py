"""Train E01_target_conditioned_retrieval with C01 staged protocol."""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import traceback
from pathlib import Path

import torch
import yaml

from baselines.common.training import get_device, set_seed
from content_bottleneck import ARTIFACT_ROOT, CONFIG_PATH
from content_bottleneck.inference_e01 import run_inference
from content_bottleneck.models_e01 import build_e01
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
from atomic.metrics import compute_run_metrics

ARMS = ("age_temporal", "temporal_only")


def load_cfg() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return cfg


def _complete(run_dir: Path) -> bool:
    return all((run_dir / name).exists() for name in (
        "checkpoint_best.pt", "predictions.parquet", "metrics.json", "mechanism_metrics.json",
    ))


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
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["predictions_sha256"] = file_sha256(run_dir / "predictions.parquet")
            write_json(manifest_path, manifest)
    if not (run_dir / "metrics.json").exists():
        compute_run_metrics(run_dir, n_boot=20 if limit_batches else 200)
        # Append content recovery into mechanism metrics if present.
        rec_path = run_dir / "content_recovery.json"
        mech_path = run_dir / "mechanism_metrics.json"
        if rec_path.exists() and mech_path.exists():
            mech = json.loads(mech_path.read_text())
            mech["content_recovery"] = json.loads(rec_path.read_text())
            write_json(mech_path, mech)


def _finish(model, cfg, run_dir, *, scenario, arm, seed, n_codes, n_targets, history_rows, device, limit_batches):
    checkpoint = finalize_checkpoint(
        model, run_dir, cfg, n_codes=n_codes, n_targets=n_targets,
        age_temporal=(arm == "age_temporal"), history_rows=history_rows,
    )
    write_json(run_dir / "manifest.json", {
        "experiment_id": cfg["experiment_id"],
        "architecture": "target_conditioned",
        "arm": arm,
        "scenario": scenario,
        "seed": int(seed),
        "git_commit": git_commit(),
        "checkpoint": str(checkpoint),
        "timestamp": utc_now(),
        "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training": {"staged": True, "lr": cfg["lr"], "max_epochs": cfg["max_epochs"]},
    })
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _infer_and_metrics(cfg, run_dir, scenario=scenario, device=device, limit_batches=limit_batches)


def _stage_a(cfg, *, scenario, seed, n_codes, n_targets, train_loader, val_loader, root, device, limit_batches):
    run_dir = stage_a_directory(root, cfg["experiment_id"], scenario, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint_best.pt"
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        history = json.loads((run_dir / "history.json").read_text()) if (run_dir / "history.json").exists() else []
        return ckpt["state_dict"], history
    set_seed(int(seed))
    model = build_e01(cfg, n_codes, n_targets, age_temporal=False).to(device)
    result = train_standard(
        model, train_loader, val_loader, cfg=cfg, run_dir=run_dir, seed=int(seed),
        device=str(device), limit_batches=limit_batches,
    )
    history = flatten_history([("A", result)])
    finalize_checkpoint(model, run_dir, cfg, n_codes=n_codes, n_targets=n_targets, age_temporal=False, history_rows=history)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    return state, history


def _train_arm(cfg, *, scenario, arm, seed, n_codes, n_targets, train_loader, val_loader, root, device, limit_batches, staged_state, stage_a_history):
    run_dir = run_directory(root, cfg["experiment_id"], scenario, arm, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    if _complete(run_dir):
        print(f"skip complete {run_dir}", flush=True)
        return
    if (run_dir / "checkpoint_best.pt").exists():
        _infer_and_metrics(cfg, run_dir, scenario=scenario, device=device, limit_batches=limit_batches)
        return
    age_temporal = arm == "age_temporal"
    model = build_e01(cfg, n_codes, n_targets, age_temporal=age_temporal)
    model.load_state_dict(staged_state)
    model.configure_arm_()
    with torch.no_grad():
        model._active_beta().zero_()
    model.to(device)
    history_rows = list(stage_a_history or [])
    stage_b = train_temporal_only_group(
        model, train_loader, val_loader, cfg=cfg, run_dir=run_dir, seed=int(seed),
        device=str(device), epochs=int(cfg["stage_b_epochs"]), limit_batches=limit_batches,
    )
    stage_c = train_joint_temporal_group(
        model, train_loader, val_loader, cfg=cfg, run_dir=run_dir, seed=int(seed),
        device=str(device), limit_batches=limit_batches,
    )
    history_rows.extend(flatten_history([("B", stage_b), ("C", stage_c)]))
    _finish(
        model, cfg, run_dir, scenario=scenario, arm=arm, seed=seed,
        n_codes=n_codes, n_targets=n_targets, history_rows=history_rows,
        device=device, limit_batches=limit_batches,
    )


def _execute_seed(cfg, *, scenario, seed, root, device, limit_batches):
    failures = []
    root = Path(root)
    if all(_complete(run_directory(root, cfg["experiment_id"], scenario, arm, seed)) for arm in ARMS):
        print(f"skip complete {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
        return failures
    print(f"loading {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
    train_loader, val_loader, _t, _v, info = cached_dtr_loaders(
        scenario, data_seed=int(cfg["data_seed"]), batch_size=int(cfg["batch_size"]),
    )
    n_codes, n_targets = int(info["n_codes"]), int(info["n_targets"])
    try:
        state, history = _stage_a(
            cfg, scenario=scenario, seed=seed, n_codes=n_codes, n_targets=n_targets,
            train_loader=train_loader, val_loader=val_loader, root=root, device=device,
            limit_batches=limit_batches,
        )
        left = build_e01(cfg, n_codes, n_targets, age_temporal=True)
        right = build_e01(cfg, n_codes, n_targets, age_temporal=False)
        left.load_state_dict(state)
        right.load_state_dict(state)
        left.configure_arm_()
        right.configure_arm_()
        if max_state_diff(left, right) > 0:
            raise RuntimeError("E01 fork state diff")
        batch = next(iter(val_loader))
        left.to(device)
        right.to(device)
        with torch.no_grad():
            a = forward_logits(left, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            b = forward_logits(right, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
        if float((a - b).abs().max()) > 1e-5:
            raise RuntimeError("E01 fork logit gap")
        write_json(stage_a_directory(root, cfg["experiment_id"], scenario, seed) / "fork_check.json", {"ok": True})
        del left, right
        for arm in ARMS:
            print(f"train {cfg['experiment_id']} {scenario} {arm} seed {seed}", flush=True)
            try:
                _train_arm(
                    cfg, scenario=scenario, arm=arm, seed=seed, n_codes=n_codes, n_targets=n_targets,
                    train_loader=train_loader, val_loader=val_loader, root=root, device=device,
                    limit_batches=limit_batches, staged_state=state, stage_a_history=history,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_cfg()
    device = get_device(args.device)
    root = args.artifact_root or ARTIFACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        root = root / "_smoke"
        cfg["max_epochs"] = cfg["min_epochs"] = cfg["patience"] = cfg["stage_b_epochs"] = 1
        scenarios, seeds, limit, jobs = ["S2"], [0], 2, 1
    else:
        scenarios = list(cfg["scenarios"])
        seeds = [int(s) for s in cfg["seeds"]]
        limit, jobs = None, int(args.jobs)
        write_config_yaml(root / "protocol_seeds.yaml", {
            "seeds": seeds, "baseline": "C01_staged_current",
            "git_commit": git_commit(), "timestamp": utc_now(),
        })
    pending = []
    for scenario in scenarios:
        for seed in seeds:
            if all(_complete(run_directory(root, cfg["experiment_id"], scenario, arm, seed)) for arm in ARMS):
                print(f"skip complete {cfg['experiment_id']} {scenario} seed {seed}", flush=True)
                continue
            pending.append({
                "cfg": cfg, "scenario": scenario, "seed": seed,
                "root": str(root), "device": str(device), "limit_batches": limit,
            })
    print(f"pending seeds: {len(pending)} jobs: {jobs}", flush=True)
    failures = []
    if jobs <= 1:
        for payload in pending:
            failures.extend(_seed_job(payload))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=jobs) as pool:
            for batch in pool.imap_unordered(_seed_job, pending):
                failures.extend(batch)
    if failures:
        write_json(root / "failures.json", failures)
        print(f"{len(failures)} failures", flush=True)
    print("E01 training complete", flush=True)


if __name__ == "__main__":
    main()
