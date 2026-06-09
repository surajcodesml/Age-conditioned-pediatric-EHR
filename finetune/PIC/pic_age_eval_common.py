"""Shared constants and loaders for PIC age-stratified evaluation / kernel viz."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.model import _infer_backbone_hparams
from finetune.PIC.model_pic import PICTALEEHRClassifier, _num_codes_from_vocab

TASKS = ("pneumonia", "heart_malformations", "los_gt7", "mortality")
BACKBONES = ("vanilla", "age")

PIC_CKPT_ROOT = REPO_ROOT / "checkpoints" / "finetune" / "PIC"
PIC_TENSOR_ROOT = REPO_ROOT / "data" / "tensorized" / "pic"
PIC_VOCAB = REPO_ROOT / "data" / "processed" / "pic" / "code_vocab_pic.json"
PIC_EMB = REPO_ROOT / "data" / "processed" / "pic" / "bge_embeddings_pic.pt"

AGE_PRETRAIN = REPO_ROOT / "checkpoints" / "age_real_202605112156" / "epoch_012.pt"
VANILLA_PRETRAIN = REPO_ROOT / "checkpoints" / "run_20260427_152603" / "best_pretrain.pt"

REPRESENTATIVE_AGES_YR = (0.05, 0.5, 2.0, 5.0, 10.0, 16.0)
MIN_BAND_N = 40


@dataclass(frozen=True)
class DevBand:
    name: str
    lo_yr: float
    hi_yr: float

    @property
    def center_yr(self) -> float:
        return 0.5 * (self.lo_yr + self.hi_yr)


DEV_BANDS: tuple[DevBand, ...] = (
    DevBand("neonate", 0.0, 1.0 / 12.0),
    DevBand("infant", 1.0 / 12.0, 1.0),
    DevBand("toddler", 1.0, 3.0),
    DevBand("preschool", 3.0, 6.0),
    DevBand("school", 6.0, 12.0),
    DevBand("adolescent", 12.0, 18.0),
)


def assign_band(age_years: np.ndarray) -> np.ndarray:
    """Return band name per sample; empty string if age >= 18 or invalid."""
    age = np.clip(np.asarray(age_years, dtype=np.float64), 0.0, None)
    out = np.full(age.shape, "", dtype=object)
    for band in DEV_BANDS:
        mask = (age >= band.lo_yr) & (age < band.hi_yr)
        out[mask] = band.name
    return out


def detect_variant_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Detect vanilla vs age-conditioned from finetuned or pretrain state dict keys."""
    keys = list(state_dict.keys())
    if any("code_predictor.4.weight" in k for k in keys):
        return str(_infer_backbone_hparams(state_dict)["variant"])
    age_variant = any(("age_emb" in k) or ("age_coeff_gen" in k) for k in keys)
    return "age_conditioned" if age_variant else "vanilla"


def load_finetuned_classifier(
    run_dir: Path,
    backbone: str,
    device: torch.device,
) -> PICTALEEHRClassifier:
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_path}")

    pretrained = AGE_PRETRAIN if backbone == "age" else VANILLA_PRETRAIN
    model = PICTALEEHRClassifier(
        pretrained_ckpt_path=pretrained,
        pic_embedding_path=PIC_EMB,
        pic_num_codes=_num_codes_from_vocab(PIC_VOCAB),
        freeze_backbone=True,
    )
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    variant = detect_variant_from_state_dict(ckpt["model_state_dict"])
    expected = "age_conditioned" if backbone == "age" else "vanilla"
    if variant != expected:
        raise RuntimeError(f"{run_dir}: expected {expected}, got {variant}")
    return model.to(device).eval()


def alpha_deviation_l2(
    temporal_weight: nn.Module,
    age_emb: nn.Module,
    age_yr: float,
) -> float:
    """||alpha(a) - base coefficients||_2 for age-conditioned polynomial weights."""
    with torch.no_grad():
        age_t = torch.tensor([[float(age_yr)]], dtype=torch.float32)
        age_feat = age_emb(age_t.clamp(min=0.0))
        delta = temporal_weight.age_coeff_gen(age_feat).squeeze(0)
        base = temporal_weight.coefficients
        return float(torch.norm(base + delta - base, p=2).item())


def kernel_deviation_at_band_center(
    backbone: nn.Module,
    band: DevBand,
    module: str = "attention",
) -> float:
    """L2 norm of age_coeff_gen output at band center (attention path by default)."""
    if module == "attention":
        tw = backbone.time_aware_attention.temporal_weight
        age_emb = backbone.time_aware_attention.age_emb
    else:
        tw = backbone.temporal_aggregation.temporal_weight
        age_emb = backbone.temporal_aggregation.age_emb
    with torch.no_grad():
        age_t = torch.tensor([[band.center_yr]], dtype=torch.float32)
        age_feat = age_emb(age_t.clamp(min=0.0))
        delta = tw.age_coeff_gen(age_feat).squeeze(0)
        return float(torch.norm(delta, p=2).item())
