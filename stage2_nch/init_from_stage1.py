"""Load a Stage-1 MIMIC checkpoint into a Stage-2 model.

Transfers backbone, head, frozen embeddings, and λ0. Discards adult β_A and
adult age μ/σ. Both arms start at β_P = 0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from stage1_mimic_pretrain.model import MinimalDKMModel
from stage2_nch.config import (
    PEDIATRIC_AGE_CENTER_YEARS,
    PEDIATRIC_AGE_SCALE_YEARS,
    STAGE1_BEST_CKPT,
    resolve_arm,
)

SKIP_KEYS = frozenset({
    "age_mean",
    "age_sd",
    "temporal.age_mean",
    "temporal.age_sd",
    "temporal.beta",
})


def load_stage1_blob(path: Path | str = STAGE1_BEST_CKPT) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint missing: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def transfer_stage1_state(model: MinimalDKMModel, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Copy compatible tensors, keep pediatric μ/σ, force β_P = 0."""
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    shape_mismatch = []
    for k, v in state.items():
        if k in SKIP_KEYS:
            skipped.append(k)
            continue
        if k not in model_sd:
            skipped.append(k)
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            shape_mismatch.append({"key": k, "ckpt": list(v.shape), "model": list(model_sd[k].shape)})
            continue
        filtered[k] = v
    incompatible, unexpected = model.load_state_dict(filtered, strict=False)
    with torch.no_grad():
        model.temporal.beta.zero_()
        model.age_mean.fill_(PEDIATRIC_AGE_CENTER_YEARS)
        model.age_sd.fill_(PEDIATRIC_AGE_SCALE_YEARS)
        model.temporal.age_mean.fill_(PEDIATRIC_AGE_CENTER_YEARS)
        model.temporal.age_sd.fill_(PEDIATRIC_AGE_SCALE_YEARS)
    if model.arm == "no_interaction":
        model.temporal.beta.requires_grad_(False)
    else:
        model.temporal.beta.requires_grad_(True)
    return {
        "n_loaded": len(filtered),
        "skipped": skipped,
        "shape_mismatch": shape_mismatch,
        "missing_after_load": list(incompatible),
        "unexpected_after_load": list(unexpected),
        "lambda0": float(model.temporal.lambda0.detach().cpu()),
        "beta": float(model.temporal.beta.detach().cpu()),
        "age_mean": float(model.age_mean.detach().cpu()),
        "age_sd": float(model.age_sd.detach().cpu()),
        "beta_requires_grad": bool(model.temporal.beta.requires_grad),
        "lambda0_requires_grad": bool(model.temporal.lambda0.requires_grad),
        "embedding_frozen": not bool(model.embedding_table.requires_grad),
    }


def build_stage2_model(
    *,
    num_codes: int,
    arm: str,
    embedding_path: str | Path | None = None,
    embedding_table: torch.Tensor | None = None,
    stage1_ckpt: Path | str | None = STAGE1_BEST_CKPT,
    seed: int = 0,
    d_model: int = 256,
    n_layers: int = 1,
    n_heads: int = 4,
    use_residual: bool = True,
    use_layernorm: bool = True,
    use_ffn: bool = True,
    ffn_mult: int = 4,
    demo_dim: int = 9,
    demo_channels: tuple[str, ...] = (),
    race_encoding: str = "one_hot",
    demo_hidden: int = 64,
    pool_temporal_bias: bool = False,
) -> tuple[MinimalDKMModel, dict[str, Any]]:
    arm = resolve_arm(arm)
    model = MinimalDKMModel(
        num_codes=num_codes,
        embedding_path=embedding_path,
        embedding_table=embedding_table,
        arm=arm,
        seed=seed,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
        use_ffn=use_ffn,
        ffn_mult=ffn_mult,
        demo_dim=demo_dim,
        demo_channels=demo_channels,
        race_encoding=race_encoding,
        demo_hidden=demo_hidden,
        age_mean=PEDIATRIC_AGE_CENTER_YEARS,
        age_sd=PEDIATRIC_AGE_SCALE_YEARS,
        pool_temporal_bias=pool_temporal_bias,
    )
    report: dict[str, Any] = {
        "stage1_ckpt": None,
        "transferred": None,
        "reset": ["temporal.beta -> 0", "age_mean/age_sd -> 9/9 (z_P)"],
        "kept": ["embedding_table", "encoder", "pooling", "demo_proj", "head", "temporal.lambda0"],
        "not_used": ["adult beta_A", "MIMIC age mu/sigma"],
    }
    if stage1_ckpt is not None:
        blob = load_stage1_blob(stage1_ckpt)
        sd = blob["model_state_dict"]
        report["stage1_ckpt"] = str(Path(stage1_ckpt))
        report["stage1_kind"] = blob.get("kind")
        report["stage1_epoch"] = blob.get("epoch")
        report["stage1_val_bce"] = blob.get("val_bce")
        report["stage1_arm"] = blob.get("arm")
        report["transferred"] = transfer_stage1_state(model, sd)
    return model, report


@torch.no_grad()
def logits_max_abs_diff(a: MinimalDKMModel, b: MinimalDKMModel, batch: dict) -> float:
    a.eval()
    b.eval()
    la = a(batch)["code_logits"]
    lb = b(batch)["code_logits"]
    return float((la - lb).abs().max().cpu())
