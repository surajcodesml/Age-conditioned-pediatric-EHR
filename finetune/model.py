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
from model.tale_ehr_age import TALEEHRAge


def _infer_backbone_hparams(
    state_dict: dict[str, torch.Tensor],
    age_conditioning_mode: str | None = None,
) -> dict[str, int | str]:
    age_variant = any(("age_emb" in k) or ("age_coeff_gen" in k) for k in state_dict.keys())
    num_codes = int(state_dict["code_predictor.4.weight"].shape[0])
    d_model = int(state_dict["time_aware_attention.mlp_q.0.weight"].shape[0])
    poly_degree = int(state_dict["time_aware_attention.temporal_weight.coefficients"].shape[0] - 1)
    demo_hidden = int(state_dict["demo_proj.0.weight"].shape[0])
    demo_dim = int(state_dict["demo_proj.0.weight"].shape[1])
    base = {
        "num_codes": num_codes,
        "d_model": d_model,
        "poly_degree": poly_degree,
        "demo_hidden": demo_hidden,
        "demo_dim": demo_dim,
        "variant": "age_conditioned" if age_variant else "vanilla",
    }
    if not age_variant:
        return base

    # FourierAgeEmbedding uses num_frequencies where age_emb_dim = 2 * num_frequencies.
    num_freq = int(state_dict["time_aware_attention.age_emb.frequencies"].shape[0])
    age_emb_dim = int(2 * num_freq)
    age_hidden_dim = int(state_dict["time_aware_attention.temporal_weight.age_coeff_gen.mlp.0.weight"].shape[0])
    mode = age_conditioning_mode if age_conditioning_mode in {"real", "random_constant", "none"} else "real"
    return {
        **base,
        "age_emb_dim": age_emb_dim,
        "age_hidden_dim": age_hidden_dim,
        "age_conditioning_mode": mode,
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
        hparams = _infer_backbone_hparams(
            state_dict,
            age_conditioning_mode=ckpt.get("age_conditioning_mode"),
        )
        print(f"[finetune/model.py] detected backbone variant: {hparams['variant']}")

        repo_root = Path(__file__).resolve().parents[1]
        embedding_path = repo_root / "data" / "processed" / "bge_embeddings.pt"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {embedding_path}")

        if hparams["variant"] == "age_conditioned":
            self.backbone = TALEEHRAge(
                embedding_path=embedding_path,
                num_codes=hparams["num_codes"],
                d_model=hparams["d_model"],
                poly_degree=hparams["poly_degree"],
                demo_dim=hparams["demo_dim"],
                demo_hidden=hparams["demo_hidden"],
                age_conditioning_mode=hparams["age_conditioning_mode"],
                age_emb_dim=hparams["age_emb_dim"],
                age_hidden_dim=hparams["age_hidden_dim"],
            )
        else:
            self.backbone = TALEEHR(
                embedding_path=embedding_path,
                num_codes=hparams["num_codes"],
                d_model=hparams["d_model"],
                poly_degree=hparams["poly_degree"],
                demo_dim=hparams["demo_dim"],
                demo_hidden=hparams["demo_hidden"],
            )
        load_result = self.backbone.load_state_dict(state_dict, strict=False)
        # Allow legacy pretraining-head keys that no longer exist in the current model
        # (e.g. intensity_predictor from the Poisson TPP era, superseded by time_params_predictor).
        _known_legacy_prefixes = ("intensity_predictor.", "time_params_predictor.")
        truly_unexpected = [
            k for k in load_result.unexpected_keys
            if not any(k.startswith(p) for p in _known_legacy_prefixes)
        ]
        if truly_unexpected:
            raise RuntimeError(f"Unexpected keys in pretrained load: {truly_unexpected}")

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
