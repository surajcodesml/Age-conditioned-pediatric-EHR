#!/usr/bin/env python3
"""Pre-training verification harness for the four-arm ablation.

(a) Per-arm dry_run: instantiate each arm's classifier, print age-pathway param
    counts, and assert invariants 1-4. Also verify the Fourier embedding resolves
    the 0-2y range.
(b) One-batch age-pathway gradient readout (additive, kernel) under
    ``return_repr_only=True`` -> classification loss. Confirms the AgeCoefficientGenerator
    (kernel) and the additive-MLP (additive) receive NON-trivial gradient in the
    fine-tune graph (the aggregation path is unused there). If either is
    gradient-dead, the harness FAILS and prints STOP.

Usage:
  python model_ablation/verify_arms.py \
      --pretrained_ckpt checkpoints/run_20260427_152603/best_pretrain.pt \
      --tensorized_dir data/finetune/heart_failure_tensorized

Notes:
  * "CHD" maps to the heart_malformations cohort, which is not built yet; any
    tensorized disease cohort works for this structural/gradient check. Pass the
    real CHD tensorized dir once built.
  * If --pretrained_ckpt is omitted, arms are built from a random backbone
    (param-count + invariant + gradient checks are all valid without trained weights).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_ablation.age_embedding import FourierAgeEmbedding
from model_ablation.arms import ARMS, KERNEL_INJECTION, resolve_arm
from model_ablation.dataset_finetune import TensorizedDiseaseClassificationDataset, disease_collate
from model_ablation.model_finetune import EMBEDDING_PATH, TALEEHRAblationClassifier
from model_ablation.tale_ehr_age import TALEEHRAblation

VOCAB = REPO_ROOT / "data/processed/code_vocab.json"


def _num_codes() -> int:
    import json
    return len(json.load(open(VOCAB)))


def build_classifier(arm: str, pretrained_ckpt: Path | None, num_codes: int) -> nn.Module:
    if pretrained_ckpt is not None:
        return TALEEHRAblationClassifier(arm, pretrained_ckpt, freeze_backbone=False)

    # From-scratch fallback: wrap a random backbone in the same classifier head.
    class _Scratch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = TALEEHRAblation(EMBEDDING_PATH, num_codes=num_codes, arm=arm)
            self.backbone.code_predictor = nn.Identity()
            self.classifier = nn.Linear(self.backbone.d_model + self.backbone.demo_hidden, 1)

        def forward(self, batch):
            out = self.backbone(batch, return_repr_only=True)
            b = out["h_repr"].shape[0]
            lengths = batch["attention_mask"].sum(dim=1).long()
            last = (lengths - 1).clamp(min=0)
            ridx = torch.arange(b)
            h = out["h_repr"][ridx, last]
            d = out["demo_features"][ridx, last]
            return self.classifier(torch.cat([h, d], dim=-1)).squeeze(-1)

    return _Scratch()


def get_batch(tensorized_dir: Path, batch_size: int) -> dict:
    ds = TensorizedDiseaseClassificationDataset(tensorized_dir / "train", max_seq_len=512)
    items = [ds[i] for i in range(min(batch_size, len(ds)))]
    return disease_collate(items)


def fourier_resolution_check() -> None:
    print("\n[Fourier 0-2y resolution] (age fed in YEARS; min period 1 month)")
    emb = FourierAgeEmbedding(num_frequencies=16)
    ages = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])  # 0mo,3mo,6mo,1y,18mo,2y
    phi = emb(ages)
    labels = ["0", "3mo", "6mo", "1y", "18mo", "2y"]
    print("  pairwise ||phi(a_i)-phi(a_j)|| across the 0-2y grid (||phi||=%.2f):" % float(phi[0].norm()))
    min_d = float("inf")
    for i in range(len(ages)):
        for j in range(i + 1, len(ages)):
            d = float((phi[i] - phi[j]).norm())
            min_d = min(min_d, d)
    for i in range(len(ages) - 1):
        d = float((phi[i] - phi[i + 1]).norm())
        print(f"    {labels[i]:>4} -> {labels[i+1]:<4} : {d:.3f}")
    assert min_d > 0.1, f"Fourier collapses infants: min pairwise distance {min_d:.4f} too small"
    print(f"  min pairwise distance over grid = {min_d:.3f} (>0.1 OK -> infants are resolved, not one bucket)")


def section_a(pretrained_ckpt: Path | None, num_codes: int) -> dict[str, int]:
    print("=" * 78)
    print("(a) PER-ARM DRY RUN: param counts + invariants")
    print(f"    kernel_injection (locked, all arms) = {KERNEL_INJECTION}")
    print("=" * 78)
    age_years = torch.rand(4, 12) * 90.0  # spans pediatric..adult
    demographics2 = torch.zeros(4, 12, 2)  # sex, race only
    pathway_counts: dict[str, int] = {}
    for arm in ARMS:
        cfg = resolve_arm(arm)
        clf = build_classifier(arm, pretrained_ckpt, num_codes)
        backbone = clf.backbone
        kcount = backbone.time_aware_attention.temporal_weight.age_coeff_gen.num_pathway_params()
        acount = backbone.additive_age_emb.num_pathway_params()
        total = backbone.age_pathway_param_count()
        pathway_counts[arm] = total
        # Invariant 1: demo_proj consumes 2 inputs (sex/race), age excluded.
        demo_in = backbone.demo_proj[0].in_features
        assert demo_in == 2, f"[INV-1] arm={arm}: demo_proj in_features={demo_in}, expected 2 (sex/race only)"
        assert backbone.demo_dim == 2
        # Invariants 2 & 3: Delta-alpha / additive-delta exactly zero where required.
        backbone.assert_arm_invariants(age_years)
        print(f"  arm={arm:<16} age_source={cfg.age_source:<9} mode={cfg.age_conditioning_mode:<15} "
              f"additive_embed={str(cfg.additive_embed):<5} | kernel_params={kcount:<6} "
              f"additive_params={acount:<6} total_age_params={total}")
    # Invariant 4: additive age-MLP params >= kernel age-pathway params.
    add_only = build_classifier("additive", pretrained_ckpt, num_codes).backbone.additive_age_emb.num_pathway_params()
    ker_only = build_classifier("kernel", pretrained_ckpt, num_codes).backbone.time_aware_attention.temporal_weight.age_coeff_gen.num_pathway_params()
    print(f"\n  [INV-4] additive-MLP params ({add_only}) >= kernel age-pathway params ({ker_only}): ", end="")
    assert add_only >= ker_only, f"[INV-4] additive {add_only} < kernel {ker_only}"
    print("OK")
    # Invariant: vanilla has zero age params.
    assert pathway_counts["vanilla"] == 0, "[INV] vanilla must have zero age params"
    # random_constant architecturally identical to kernel.
    assert pathway_counts["random_constant"] == pathway_counts["kernel"], \
        "[INV] random_constant must be architecturally identical to kernel"
    print("  [INV] vanilla age params == 0; random_constant params == kernel params: OK")
    fourier_resolution_check()
    print("\n  invariants 1-4 PASSED for all arms.")
    return pathway_counts


def section_b(pretrained_ckpt: Path | None, tensorized_dir: Path, num_codes: int) -> None:
    print("\n" + "=" * 78)
    print("(b) ONE-BATCH AGE-PATHWAY GRADIENT (fine-tune graph, return_repr_only)")
    print("=" * 78)
    batch = get_batch(tensorized_dir, batch_size=16)
    criterion = nn.BCEWithLogitsLoss()
    dead = []
    for arm in ("additive", "kernel"):
        torch.manual_seed(0)
        clf = build_classifier(arm, pretrained_ckpt, num_codes)
        clf.train()
        clf.zero_grad(set_to_none=True)
        logits = clf(batch)
        loss = criterion(logits, batch["labels"])
        loss.backward()
        backbone = clf.backbone
        if arm == "kernel":
            mod = backbone.time_aware_attention.temporal_weight.age_coeff_gen
            name = "AgeCoefficientGenerator (kernel)"
        else:
            mod = backbone.additive_age_emb
            name = "AdditiveAgeEmbedding (additive)"
        gnorm = float(np.sqrt(sum(float(p.grad.norm()) ** 2 for p in mod.parameters() if p.grad is not None)))
        pcount = sum(1 for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        alive = gnorm > 1e-9 and pcount > 0
        print(f"  arm={arm:<9} {name:<34} ||grad||={gnorm:.3e}  params_with_grad={pcount}  "
              f"{'ALIVE' if alive else 'DEAD'}")
        if not alive:
            dead.append(arm)
    if dead:
        print("\n  *** STOP: age pathway gradient-DEAD in fine-tuning for: "
              f"{dead}. Do NOT proceed to the ablation. ***")
        raise SystemExit(2)
    print("\n  both age pathways receive non-trivial gradient in the fine-tune graph. OK to proceed.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", type=Path, default=None)
    ap.add_argument("--tensorized_dir", type=Path, default=REPO_ROOT / "data/finetune/heart_failure_tensorized")
    args = ap.parse_args()
    ckpt = args.pretrained_ckpt if (args.pretrained_ckpt and args.pretrained_ckpt.exists()) else None
    if args.pretrained_ckpt and ckpt is None:
        print(f"[warn] pretrained_ckpt {args.pretrained_ckpt} not found; using random backbone.")
    num_codes = _num_codes()
    print(f"[setup] num_codes={num_codes} backbone={'pretrained ' + str(ckpt) if ckpt else 'RANDOM (from scratch)'}")
    print(f"[setup] batch source (CHD stand-in) = {args.tensorized_dir}")
    section_a(ckpt, num_codes)
    section_b(ckpt, args.tensorized_dir, num_codes)
    print("\nVERIFICATION PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
