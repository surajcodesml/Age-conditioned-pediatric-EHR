"""Extract frozen C01 encounter representations and content scores."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from atomic.inference import load_trained_model
from content_bottleneck import C01_ARTIFACT_ROOT
from content_bottleneck.generator_content import (
    encounter_signal_membership,
    load_weight_matrix,
    n_background_codes,
    oracle_content_pre_decay,
    signal_id_map,
)
from ladder.cached_data import cached_dtr_loaders
from ladder.artifacts import run_directory


def c01_checkpoint(scenario: str, seed: int, arm: str = "age_temporal") -> Path:
    return (
        run_directory(C01_ARTIFACT_ROOT, "C01_staged_current", scenario, arm, seed)
        / "checkpoint_best.pt"
    )


def pre_encoder_pool(model, enc_code_ids: torch.Tensor, enc_code_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pooled code embeddings before the encounter MLP."""
    enc = model.base.encounter_encoder
    e = enc.code_emb(enc_code_ids)
    mask = enc_code_mask.to(e.dtype).unsqueeze(-1)
    summed = (e * mask).sum(dim=2)
    denom = mask.sum(dim=2).clamp(min=1.0)
    return summed / denom


@torch.no_grad()
def extract_encounter_table(
    *,
    scenario: str,
    seed: int,
    data_seed: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    ckpt_path = c01_checkpoint(scenario, seed)
    model, _ckpt = load_trained_model(ckpt_path, device)
    _, _, test_loader, vocab, info = cached_dtr_loaders(
        scenario, data_seed=data_seed, batch_size=32,
    )
    specs = info["specs"]
    W, signal_names, mechs = load_weight_matrix(specs)
    sig_map = signal_id_map(vocab)
    n_signals = len(signal_names)

    rows: dict[str, list] = {
        "example_id": [], "encounter_idx": [], "seed": [], "scenario": [],
        "age": [], "tau": [], "n_background": [], "is_background_only": [],
        "n_codes": [], "u": [], "signal_membership": [], "w_true": [],
        "oracle_pre_decay": [], "v": [], "pre_pool": [], "primary_signal": [],
    }

    for bi, batch in enumerate(test_loader):
        if max_batches is not None and bi >= int(max_batches):
            break
        moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        parts = model(
            enc_code_ids=moved["enc_code_ids"],
            enc_code_mask=moved["enc_code_mask"],
            enc_tau=moved["enc_tau"],
            enc_padding_mask=moved["enc_padding_mask"],
            age=moved["age"],
            return_parts=True,
        )
        v = parts["v"].detach().cpu().numpy()
        u = parts["u"].detach().cpu().numpy()
        pre = pre_encoder_pool(model, moved["enc_code_ids"], moved["enc_code_mask"]).cpu().numpy()
        pad = batch["enc_padding_mask"].numpy()
        code_ids = batch["enc_code_ids"].numpy()
        code_mask = batch["enc_code_mask"].numpy()
        tau = batch["enc_tau"].numpy()
        ages = batch["age"].numpy()
        example_ids = batch["example_ids"].numpy()

        bsz, n_enc = pad.shape
        for b in range(bsz):
            for m in range(n_enc):
                if bool(pad[b, m]):
                    continue
                mem = encounter_signal_membership(code_ids[b, m], code_mask[b, m], sig_map, n_signals)
                n_bg = n_background_codes(code_ids[b, m], code_mask[b, m], sig_map)
                n_codes = int(code_mask[b, m].sum())
                is_bg = bool(np.all(mem == 0))
                primary = -1
                if not is_bg:
                    primary = int(np.argmax(mem))
                # w_true for primary signal column; zeros if background
                w_true = W[:, primary] if primary >= 0 else np.zeros(W.shape[0], dtype=np.float64)
                pre_decay = oracle_content_pre_decay(mem, specs, W)
                rows["example_id"].append(int(example_ids[b]))
                rows["encounter_idx"].append(int(m))
                rows["seed"].append(int(seed))
                rows["scenario"].append(scenario)
                rows["age"].append(float(ages[b]))
                rows["tau"].append(float(tau[b, m]))
                rows["n_background"].append(int(n_bg))
                rows["is_background_only"].append(int(is_bg))
                rows["n_codes"].append(int(n_codes))
                rows["u"].append(float(u[b, m]))
                rows["signal_membership"].append(mem.astype(np.float32))
                rows["w_true"].append(w_true.astype(np.float32))
                rows["oracle_pre_decay"].append(pre_decay.astype(np.float32))
                rows["v"].append(v[b, m].astype(np.float32))
                rows["pre_pool"].append(pre[b, m].astype(np.float32))
                rows["primary_signal"].append(int(primary))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out: dict[str, Any] = {
        "signal_names": np.asarray(signal_names),
        "mechanisms": np.asarray(mechs),
        "W": W.astype(np.float64),
    }
    for key, values in rows.items():
        if key in ("signal_membership", "w_true", "oracle_pre_decay", "v", "pre_pool"):
            out[key] = np.stack(values, axis=0)
        elif key == "scenario":
            out[key] = np.asarray(values)
        else:
            out[key] = np.asarray(values)
    return out
