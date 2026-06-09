#!/usr/bin/env python3
"""PIC fine-tuning model wrapper for TALE-EHR binary classification.

The stock ``finetune/model.py`` bakes the MIMIC vocabulary into the backbone:
it reads the MIMIC ``bge_embeddings.pt`` table and the checkpoint's persistent
``embedding_table`` buffer ([30637, 1024]). PIC has its own 2,200-row BGE table
and its own 2,198-code vocabulary, so the MIMIC table cannot be used directly --
PIC ``bge_codes`` index rows 2..2199, which point at MIMIC codes in the MIMIC
table.

This wrapper performs *transfer* of the pretrained backbone onto PIC:
  - the TALE-EHR temporal/attention/demographic modules operate on 1024-d BGE
    vectors and ``d_model`` hidden states, so they are vocabulary-agnostic and
    are warm-started from the pretrained checkpoint;
  - the code embedding table is swapped to PIC's BGE table (loaded from file by
    the backbone constructor); the checkpoint's MIMIC ``embedding_table`` buffer
    and the discarded pretraining heads are stripped before loading.

Nothing in the parent ``finetune/`` package is modified.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.model import _infer_backbone_hparams  # reused, read-only
from model.tale_ehr import TALEEHR
from model.tale_ehr_age import TALEEHRAge

# State-dict prefixes that must NOT be transferred from the MIMIC checkpoint:
#   - embedding_table : MIMIC [30637,1024] buffer; PIC keeps its own table
#   - *_predictor     : pretraining heads, discarded for classification
_STRIP_PREFIXES = (
    "embedding_table",
    "code_predictor.",
    "intensity_predictor.",
    "time_params_predictor.",
)

# Backbone module prefixes that MUST be warm-started (transferred) intact.
_REQUIRED_TRANSFER_PREFIXES = (
    "time_aware_attention.",
    "temporal_aggregation.",
    "demo_proj.",
)


def _num_codes_from_vocab(vocab_path: Path) -> int:
    with Path(vocab_path).open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return int(len(vocab))


class PICTALEEHRClassifier(nn.Module):
    """TALE-EHR backbone warm-started from a MIMIC checkpoint, retargeted to PIC."""

    def __init__(
        self,
        pretrained_ckpt_path: str | Path,
        pic_embedding_path: str | Path,
        pic_num_codes: int,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        ckpt_path = Path(pretrained_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing pretrained checkpoint: {ckpt_path}")
        emb_path = Path(pic_embedding_path)
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing PIC embedding file: {emb_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing 'model_state_dict': {ckpt_path}")
        state_dict = ckpt["model_state_dict"]

        # Infer architecture from the checkpoint, then override num_codes for PIC.
        # num_codes only sizes the (discarded) code-prediction head; the embedding
        # table check requires PIC_table_rows >= pic_num_codes + 2.
        hparams = _infer_backbone_hparams(
            state_dict, age_conditioning_mode=ckpt.get("age_conditioning_mode")
        )
        variant = str(hparams["variant"])
        print(
            f"[finetune/PIC/model_pic.py] backbone variant={variant} "
            f"d_model={hparams['d_model']} poly_degree={hparams['poly_degree']} "
            f"pic_num_codes={pic_num_codes}",
            flush=True,
        )

        if variant == "age_conditioned":
            self.backbone = TALEEHRAge(
                embedding_path=emb_path,
                num_codes=int(pic_num_codes),
                d_model=int(hparams["d_model"]),
                poly_degree=int(hparams["poly_degree"]),
                demo_dim=int(hparams["demo_dim"]),
                demo_hidden=int(hparams["demo_hidden"]),
                age_conditioning_mode=str(hparams["age_conditioning_mode"]),
                age_emb_dim=int(hparams["age_emb_dim"]),
                age_hidden_dim=int(hparams["age_hidden_dim"]),
            )
        else:
            self.backbone = TALEEHR(
                embedding_path=emb_path,
                num_codes=int(pic_num_codes),
                d_model=int(hparams["d_model"]),
                poly_degree=int(hparams["poly_degree"]),
                demo_dim=int(hparams["demo_dim"]),
                demo_hidden=int(hparams["demo_hidden"]),
            )

        transfer_sd = {
            k: v
            for k, v in state_dict.items()
            if not any(k.startswith(p) for p in _STRIP_PREFIXES)
        }
        load_result = self.backbone.load_state_dict(transfer_sd, strict=False)

        # Any required backbone module that failed to transfer is a hard error:
        # shapes for these modules are vocab-independent and must match exactly.
        missing_required = [
            k
            for k in load_result.missing_keys
            if any(k.startswith(p) for p in _REQUIRED_TRANSFER_PREFIXES)
        ]
        if missing_required:
            raise RuntimeError(
                f"Pretrained backbone failed to transfer required keys: {missing_required}"
            )
        unexpected = list(load_result.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when transferring backbone: {unexpected}")

        n_transferred = sum(
            1
            for k in transfer_sd
            if any(k.startswith(p) for p in _REQUIRED_TRANSFER_PREFIXES)
        )
        print(
            f"[finetune/PIC/model_pic.py] warm-started {n_transferred} backbone tensors; "
            f"PIC embedding_table rows={int(self.backbone.embedding_table.shape[0])}",
            flush=True,
        )

        # Discard pretraining heads: classification uses last event repr + demo feature.
        self.backbone.code_predictor = nn.Identity()
        if hasattr(self.backbone, "intensity_predictor"):
            self.backbone.intensity_predictor = nn.Identity()
        if hasattr(self.backbone, "time_params_predictor"):
            self.backbone.time_params_predictor = nn.Identity()

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


__all__ = ["PICTALEEHRClassifier", "_num_codes_from_vocab"]
