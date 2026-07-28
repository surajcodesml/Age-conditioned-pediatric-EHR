"""Shared loaders, checkpoint resolution, and batch utilities for the age audit.

Reuses ``eval_pretrain`` reconstruction helpers and ``diagnostics`` I/O exclusively.
Modules in this package never print and never write files directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_new.data import TensorizedPretrainDataset, make_collate, dataloader_worker_init
from model_new.eval_pretrain import (
    EXPECTED_TAU_MAX,
    TAU_MAX_TOL,
    BatchOrderHash,
    build_model,
    check_configs,
    make_val_loader,
    model_kwargs_from_config,
)
from model_new.model import DKMModel
from model_new.train import set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "model_new" / "run"
DEFAULT_OUT_DIR = REPO_ROOT / "model_new" / "audit"

ARMS = ("vanilla", "random_constant", "additive", "kernel")


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(run_root: Path) -> dict[str, Path]:
    """Map arm -> run directory from ``selection.json`` when present, else by glob.

    Only seed-0 pretraining runs at the top of ``model_new/run/`` are in scope.
    Fine-tune trees under ``MIMIC/`` / ``pic/`` are ignored.
    """
    run_root = Path(run_root)
    sel = run_root / "selection.json"
    if sel.exists():
        blob = read_json(sel)
        runs: dict[str, Path] = {}
        for arm, rel in blob["runs"].items():
            p = Path(rel)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if not p.is_dir():
                raise FileNotFoundError(f"[HARD] selection.json points to missing run: {p}")
            runs[arm] = p
        return runs

    found: dict[str, Path] = {}
    for arm in ARMS:
        matches = sorted(run_root.glob(f"{arm}_s*"))
        if (run_root / f"{arm}_s0").is_dir() and (run_root / f"{arm}_s0") not in matches:
            matches = [run_root / f"{arm}_s0"] + matches
        uniq: list[Path] = []
        for m in matches:
            if m.is_dir() and m not in uniq:
                uniq.append(m)
        if not uniq:
            raise FileNotFoundError(f"[HARD] no run directory for arm={arm!r} under {run_root}")
        with_ckpt = [m for m in uniq if any(m.glob("epoch_*.pt"))]
        if len(with_ckpt) != 1:
            raise AssertionError(
                f"[HARD] expected exactly one checkpointed run for arm={arm!r}, "
                f"got {[str(x) for x in uniq]}; refuse to pick silently")
        found[arm] = with_ckpt[0]
    return found


def list_epochs(run_dir: Path) -> list[int]:
    epochs = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("epoch_*.pt"))
    if not epochs:
        raise FileNotFoundError(f"[HARD] no epoch_*.pt under {run_dir}")
    return epochs


def checkpoint_path(run_dir: Path, epoch: int) -> Path:
    path = run_dir / f"epoch_{epoch:03d}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"[HARD] missing checkpoint {path}; refusing to substitute another epoch or seed")
    return path


def select_best_epoch(run_dir: Path) -> tuple[int, float]:
    """Per-arm best by validation BCE in ``train.json`` (``per_arm_best``)."""
    train = read_json(run_dir / "train.json")
    if not isinstance(train, list) or not train:
        raise AssertionError(f"[HARD] {run_dir}/train.json is empty or not a list")
    best = min(train, key=lambda e: float(e["val_loss"]))
    epoch = int(best["epoch"])
    checkpoint_path(run_dir, epoch)  # fail loudly if missing
    return epoch, float(best["val_loss"])


def load_checkpoint(model: DKMModel, path: Path, *, arm: str, epoch: int,
                    device: torch.device) -> dict:
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if ckpt.get("arm") != arm:
        raise AssertionError(
            f"[HARD] {path} holds arm={ckpt.get('arm')!r}, expected {arm!r}")
    if int(ckpt.get("epoch", -1)) != int(epoch):
        raise AssertionError(
            f"[HARD] {path} holds epoch={ckpt.get('epoch')!r}, expected {epoch}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    got = float(model.tau_max)
    if abs(got - EXPECTED_TAU_MAX) > TAU_MAX_TOL:
        raise AssertionError(
            f"[INV-TMAX] {path}: tau_max={got!r} != frozen {EXPECTED_TAU_MAX!r}")
    return {
        "path": str(path),
        "epoch": int(epoch),
        "tau_max": got,
        "tau_max_source": ckpt.get("tau_max_source"),
        "age_standardization": ckpt.get("age_standardization"),
        "seed": int(ckpt.get("seed", -1)),
    }


def backbone_init_hash(shared: dict, arm: str) -> str:
    """Hash of shared non-age, non-head parameters at construction.

    The additive arm's head is wider, so hashing every trainable tensor would make
    ``additive`` look like a different init even when the encoder/demo backbone is
    bit-identical (INV-ZERO-A). Head weights are therefore excluded; they are checked
    separately via ``check_configs`` rebuild.
    """
    m = build_model(shared, arm)
    age_ids = {id(p) for p in m.age_parameters()}
    head_ids = {id(p) for p in m.head_parameters()}
    h = hashlib.blake2b(digest_size=16)
    for name, p in sorted(m.named_parameters()):
        if id(p) in age_ids or id(p) in head_ids or not p.requires_grad:
            continue
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    del m
    return h.hexdigest()


def fourier_buffer_digest(model: DKMModel) -> dict[str, str]:
    out: dict[str, str] = {}
    for site_name, site in model.kernel_sites():
        freq = site.age.fourier.frequencies.detach().cpu().numpy()
        periods = site.age.fourier.periods.detach().cpu().numpy()
        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(freq).tobytes())
        h.update(np.ascontiguousarray(periods).tobytes())
        out[site_name] = h.hexdigest()
    return out


def patient_ids_from_dataset(ds: TensorizedPretrainDataset, n_rows: int | None = None
                             ) -> np.ndarray:
    """Stable patient keys from ``(shard_id, pos)``; one id per validation window."""
    ids = np.array([int(s) * 10_000_000 + int(p) for s, p, _ in ds._index], dtype=np.int64)
    if n_rows is not None:
        ids = ids[: int(n_rows)]
    return ids


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def age_last_of(batch: dict) -> torch.Tensor:
    lengths = batch["lengths"]
    rows = torch.arange(lengths.shape[0], device=lengths.device)
    return batch["age_years"][rows, lengths - 1].float()


def build_shared_context(run_dirs: dict[str, Path], *, seed: int, device: torch.device,
                         batch_size: int, num_workers: int, max_val_batches: int | None,
                         allow_config_diff: set[str] | None = None) -> dict:
    """Load configs, assert shared constructor kwargs, build the shared val loader."""
    set_seed(seed)
    configs = {arm: read_json(run_dirs[arm] / "config.json") for arm in ARMS}
    for arm, cfg in configs.items():
        if cfg.get("arm") != arm:
            raise AssertionError(
                f"[HARD] {run_dirs[arm]}/config.json arm={cfg.get('arm')!r} != {arm!r}")
    order = list(ARMS)
    cfg_report = check_configs(configs, order, allow=allow_config_diff or set())
    shared = model_kwargs_from_config(configs[order[0]])

    tensorized = REPO_ROOT / shared["tensorized_dir"] / "val"
    vocab = REPO_ROOT / shared["vocab_path"]
    ds = TensorizedPretrainDataset(tensorized, vocab, max_seq_len=shared["max_seq_len"])
    loader = make_val_loader(ds, batch_size, num_workers, shared["race_encoding"])

    hasher = BatchOrderHash()
    for i, batch in enumerate(loader, 1):
        if max_val_batches and i > max_val_batches:
            break
        hasher.update(batch)

    loader = make_val_loader(ds, batch_size, num_workers, shared["race_encoding"])
    patient_ids = patient_ids_from_dataset(ds, hasher.n_rows)

    selected = {}
    for arm in order:
        ep, vl = select_best_epoch(run_dirs[arm])
        selected[arm] = {
            "epoch": ep,
            "val_loss_train_json": vl,
            "checkpoint": str(checkpoint_path(run_dirs[arm], ep)),
        }

    return {
        "seed": int(seed),
        "device": device,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_val_batches": max_val_batches,
        "run_dirs": {a: str(run_dirs[a]) for a in order},
        "configs": configs,
        "shared": shared,
        "config_check": cfg_report,
        "dataset": ds,
        "loader": loader,
        "batch_order_hash": hasher.hexdigest,
        "n_batches": hasher.n_batches,
        "n_examples": hasher.n_rows,
        "patient_ids": patient_ids,
        "selected": selected,
        "order": order,
    }


def iter_batches(loader: DataLoader, max_batches: int | None):
    for i, batch in enumerate(loader, 1):
        if max_batches and i > max_batches:
            break
        yield batch


def generator_final_weight(site) -> torch.Tensor | None:
    """W2 of ``Linear(H -> s)``, the zero-init final layer; None if mode=none."""
    gen = site.age.generator
    if gen.mlp is None:
        return None
    return gen.mlp[-1].weight


def generator_first_weight(site) -> torch.Tensor | None:
    gen = site.age.generator
    if gen.mlp is None:
        return None
    return gen.mlp[0].weight
