#!/usr/bin/env python3
"""Fine-tuning model wrapper for TALE-EHR binary classification."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.tale_ehr import TALEEHR


def _infer_backbone_hparams(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    num_codes = int(state_dict["code_predictor.4.weight"].shape[0])
    d_model = int(state_dict["time_aware_attention.mlp_q.0.weight"].shape[0])
    poly_degree = int(state_dict["time_aware_attention.temporal_weight.coefficients"].shape[0] - 1)
    demo_hidden = int(state_dict["demo_proj.0.weight"].shape[0])
    demo_dim = int(state_dict["demo_proj.0.weight"].shape[1])
    return {
        "num_codes": num_codes,
        "d_model": d_model,
        "poly_degree": poly_degree,
        "demo_hidden": demo_hidden,
        "demo_dim": demo_dim,
    }


class TALEEHRClassifier(nn.Module):
    """TALE-EHR with a classification head for binary disease prediction."""

    def __init__(self, pretrained_ckpt_path: str | Path, freeze_backbone: bool = False):
        super().__init__()
        ckpt_path = Path(pretrained_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing 'model_state_dict': {ckpt_path}")
        state_dict = ckpt["model_state_dict"]
        hparams = _infer_backbone_hparams(state_dict)

        repo_root = Path(__file__).resolve().parents[1]
        embedding_path = repo_root / "data" / "processed" / "bge_embeddings.pt"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {embedding_path}")

        self.backbone = TALEEHR(
            embedding_path=embedding_path,
            num_codes=hparams["num_codes"],
            d_model=hparams["d_model"],
            poly_degree=hparams["poly_degree"],
            demo_dim=hparams["demo_dim"],
            demo_hidden=hparams["demo_hidden"],
        )
        load_result = self.backbone.load_state_dict(state_dict, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in pretrained load: {load_result.unexpected_keys}")

        # Discard pretraining heads: classification uses last event repr + demo feature.
        self.backbone.code_predictor = nn.Identity()
        self.backbone.intensity_predictor = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(self.backbone.d_model + self.backbone.demo_hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.backbone(batch, return_repr_only=True)
        h_repr = out["h_repr"]  # [B, L, d_model]
        demo_features = out["demo_features"]  # [B, L, demo_hidden]

        b = h_repr.shape[0]
        lengths = batch["attention_mask"].sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        ridx = torch.arange(b, device=h_repr.device)

        h_last = h_repr[ridx, last_idx]
        d_last = demo_features[ridx, last_idx]
        logits = self.classifier(torch.cat([h_last, d_last], dim=-1)).squeeze(-1)
        return logits
