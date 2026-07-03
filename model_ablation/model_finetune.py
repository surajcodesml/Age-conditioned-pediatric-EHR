#!/usr/bin/env python3
"""Fine-tune classifier wrapper for the four-arm ablation.

Every arm loads the ONE shared vanilla backbone; the age pathway for the arm is
introduced here at fine-tune time (symmetrically for the two age arms). The
backbone's non-age keys are byte-identical across arms, so loading is clean.

Handling the frozen-checkpoint demo leak: a legacy vanilla pretrain saved a
``demo_proj`` with 3 input columns [age, sex, race]. We slice off column 0 (age)
so the fine-tune ``demo_proj`` (demo_dim=2) inherits only the sex/race projection
and age is structurally excluded. A backbone pretrained with THIS package already
has demo_dim=2 and loads directly.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_ablation.tale_ehr_age import TALEEHRAblation

EMBEDDING_PATH = REPO_ROOT / "data" / "processed" / "bge_embeddings.pt"


def _infer_backbone_hparams(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    num_codes = int(state_dict["code_predictor.4.weight"].shape[0])
    d_model = int(state_dict["time_aware_attention.mlp_q.0.weight"].shape[0])
    poly_degree = int(state_dict["time_aware_attention.temporal_weight.coefficients"].shape[0] - 1)
    demo_hidden = int(state_dict["demo_proj.0.weight"].shape[0])
    demo_in = int(state_dict["demo_proj.0.weight"].shape[1])
    return {"num_codes": num_codes, "d_model": d_model, "poly_degree": poly_degree,
            "demo_hidden": demo_hidden, "demo_in": demo_in}


def _adapt_legacy_demo_proj(state_dict: dict[str, torch.Tensor], demo_in: int) -> dict[str, torch.Tensor]:
    """If the pretrain used demo_dim=3 [age, sex, race], drop the age column so the
    fine-tune demo_proj (demo_dim=2) inherits only sex/race."""
    if demo_in == 2:
        return state_dict
    if demo_in != 3:
        raise ValueError(f"Unexpected pretrain demo_dim={demo_in}; expected 2 or 3.")
    sd = dict(state_dict)
    w = sd["demo_proj.0.weight"]              # [hidden, 3] = [age, sex, race]
    sd["demo_proj.0.weight"] = w[:, 1:].contiguous()   # -> [hidden, 2] = [sex, race]
    print("[finetune] adapted legacy demo_proj: dropped age input column (3 -> 2).")
    return sd


class TALEEHRAblationClassifier(nn.Module):
    def __init__(self, arm: str, pretrained_ckpt_path: str | Path, freeze_backbone: bool = False,
                 embedding_path: str | Path = EMBEDDING_PATH) -> None:
        super().__init__()
        self.arm = arm
        ckpt_path = Path(pretrained_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        hp = _infer_backbone_hparams(state_dict)

        self.backbone = TALEEHRAblation(
            embedding_path=embedding_path,
            num_codes=hp["num_codes"],
            arm=arm,
            d_model=hp["d_model"],
            poly_degree=hp["poly_degree"],
            demo_hidden=hp["demo_hidden"],
        )
        state_dict = _adapt_legacy_demo_proj(state_dict, hp["demo_in"])

        # Non-age backbone keys must all be present; age keys / dropped heads may differ.
        load = self.backbone.load_state_dict(state_dict, strict=False)
        age_key = lambda k: ("age_coeff_gen" in k) or ("additive_age_emb" in k) or ("age_emb" in k) or ("additive_fourier" in k)
        head_key = lambda k: k.startswith("code_predictor") or k.startswith("time_params_predictor") or k.startswith("intensity_predictor")
        bad_missing = [k for k in load.missing_keys if not age_key(k)]
        bad_unexpected = [k for k in load.unexpected_keys if not (age_key(k) or head_key(k) or "embedding_table" in k)]
        if bad_missing:
            raise RuntimeError(f"Missing non-age backbone keys on load: {bad_missing}")
        if bad_unexpected:
            raise RuntimeError(f"Unexpected keys on load: {bad_unexpected}")

        # Classification uses the last-event representation only.
        self.backbone.code_predictor = nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(self.backbone.d_model + self.backbone.demo_hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.backbone(batch, return_repr_only=True)
        h_repr = out["h_repr"]              # [B, L, d_model]
        demo_features = out["demo_features"]  # [B, L, demo_hidden]
        b = h_repr.shape[0]
        lengths = batch["attention_mask"].sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        ridx = torch.arange(b, device=h_repr.device)
        h_last = h_repr[ridx, last_idx]
        d_last = demo_features[ridx, last_idx]
        return self.classifier(torch.cat([h_last, d_last], dim=-1)).squeeze(-1)
