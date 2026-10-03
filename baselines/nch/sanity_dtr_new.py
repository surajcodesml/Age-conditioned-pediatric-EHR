#!/usr/bin/env python3
"""Quick forward/gradient sanity checks for NCH Content-Persistence DTR ``_new``.

Checks (must all pass before full Stage-2 training):
  1. temporal_only λ is age-invariant
  2. age_temporal λ changes with age when β ≠ 0
  3. β receives gradients on age_temporal; unused on temporal_only
  4. evidence mass M age dependence matches arm
  5. history weights are raw-additive (no softmax)
  6. shared MIMIC TO init → identical ex-β fingerprints; β=0 on both arms
  7. pediatric z uses center/scale 9 (z(0)=-1, z(18)=1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.nch.runner import (  # noqa: E402
    SHARED_CP_STAGE1_DIR,
    apply_shared_cp_init,
    fingerprint_excluding_beta,
)
from baselines.mimic.runner import MIMICCPDTRAdapter  # noqa: E402


def _fake_batch(n_codes: int = 512, B: int = 4, M: int = 8, C: int = 4, device="cpu"):
    enc_code_ids = torch.randint(1, min(n_codes, 200), (B, M, C), device=device)
    enc_code_mask = torch.ones(B, M, C, dtype=torch.bool, device=device)
    enc_tau = torch.linspace(0.1, 2.0, M, device=device).unsqueeze(0).expand(B, M).contiguous()
    enc_padding_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    age = torch.tensor([1.0, 5.0, 10.0, 15.0], device=device)[:B]
    labels = torch.zeros(B, n_codes, device=device)
    labels[:, 1:5] = 1.0
    return {
        "enc_code_ids": enc_code_ids,
        "enc_code_mask": enc_code_mask,
        "enc_tau": enc_tau,
        "enc_padding_mask": enc_padding_mask,
        "age": age,
        "labels": labels,
    }


def _mass_and_lambda(model: MIMICCPDTRAdapter, batch: dict):
    out = model.model(
        enc_code_ids=batch["enc_code_ids"],
        enc_code_mask=batch["enc_code_mask"],
        enc_tau=batch["enc_tau"],
        enc_padding_mask=batch["enc_padding_mask"],
        age=batch["age"],
        return_parts=True,
    )
    return out["w"], out["M"].squeeze(-1), out["lambda"]


def _build_light(arm: str, n_codes: int = 512) -> MIMICCPDTRAdapter:
    sat = str(REPO_ROOT / "synthetic_age_temporal")
    if sat not in sys.path:
        sys.path.append(sat)
    from model_dtr import DevelopmentalTemporalRetrieval

    m = MIMICCPDTRAdapter.__new__(MIMICCPDTRAdapter)
    torch.nn.Module.__init__(m)
    m.arm = arm
    m.n_codes = n_codes
    m.d_model = 64
    m.model = DevelopmentalTemporalRetrieval(
        n_codes=n_codes,
        n_targets=n_codes,
        d_model=64,
        age_temporal=(arm == "age_temporal"),
        aggregation="raw_additive",
        dropout=0.0,
        max_codes_per_encounter=64,
    )
    enc = m.model.encounter_encoder
    del enc.code_emb
    enc.register_buffer("code_emb_table", torch.randn(n_codes, 128), persistent=True)
    enc.code_proj = torch.nn.Linear(128, 64, bias=False)

    def _encode(enc_code_ids, enc_code_mask, _enc=enc):
        e = _enc.code_proj(_enc.code_emb_table[enc_code_ids])
        mask = enc_code_mask.to(e.dtype).unsqueeze(-1)
        summed = (e * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp(min=1.0)
        return _enc.enc_mlp(summed / denom)

    enc.forward = _encode  # type: ignore[method-assign]
    return m


def main() -> int:
    device = torch.device("cpu")
    n_codes = 512
    to = _build_light("temporal_only", n_codes)
    at = _build_light("age_temporal", n_codes)
    with torch.no_grad():
        at.model.beta.fill_(-1.0)

    batch = _fake_batch(n_codes=n_codes, device=device)
    checks: list[tuple[str, bool, str]] = []

    ages = [torch.full((4,), a, device=device) for a in (1.0, 9.0, 18.0)]
    lams_to = []
    for a in ages:
        _, _, lam = _mass_and_lambda(to, {**batch, "age": a})
        lams_to.append(lam.detach())
    to_ok = all(torch.allclose(lams_to[0], x, atol=1e-6) for x in lams_to[1:])
    checks.append(("TO λ age-invariant", to_ok, f"λ rows equal across ages={to_ok}"))

    lams_at = []
    for a in ages:
        _, _, lam = _mass_and_lambda(at, {**batch, "age": a})
        lams_at.append(lam.detach())
    at_lam_diff = (lams_at[0] - lams_at[-1]).abs().mean().item()
    checks.append(("AT λ age-sensitive", at_lam_diff > 1e-4, f"mean|Δλ|={at_lam_diff:.6f}"))

    at.zero_grad(set_to_none=True)
    logits = at.model(
        enc_code_ids=batch["enc_code_ids"],
        enc_code_mask=batch["enc_code_mask"],
        enc_tau=batch["enc_tau"],
        enc_padding_mask=batch["enc_padding_mask"],
        age=batch["age"],
    )
    F.binary_cross_entropy_with_logits(logits, batch["labels"]).backward()
    beta_grad = at.model.beta.grad
    checks.append(
        (
            "AT β receives grad",
            beta_grad is not None and float(beta_grad.abs()) > 0,
            f"β.grad={None if beta_grad is None else float(beta_grad):.6e}",
        )
    )

    to.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(
        to.model(
            enc_code_ids=batch["enc_code_ids"],
            enc_code_mask=batch["enc_code_mask"],
            enc_tau=batch["enc_tau"],
            enc_padding_mask=batch["enc_padding_mask"],
            age=batch["age"],
        ),
        batch["labels"],
    ).backward()
    to_beta_grad = to.model.beta.grad
    checks.append(
        (
            "TO β unused in graph",
            to_beta_grad is None or float(to_beta_grad.abs()) == 0.0,
            "β.grad=None" if to_beta_grad is None else f"β.grad={float(to_beta_grad):.6e}",
        )
    )

    Ms_to, Ms_at = [], []
    for a in ages:
        _, M_to, _ = _mass_and_lambda(to, {**batch, "age": a})
        _, M_at, _ = _mass_and_lambda(at, {**batch, "age": a})
        Ms_to.append(M_to.detach())
        Ms_at.append(M_at.detach())
    dM_to = (Ms_to[0] - Ms_to[-1]).abs().mean().item()
    dM_at = (Ms_at[0] - Ms_at[-1]).abs().mean().item()
    checks.append(("TO mass age-invariant", dM_to < 1e-5, f"mean|ΔM|={dM_to:.6e}"))
    checks.append(("AT mass age-sensitive", dM_at > 1e-5, f"mean|ΔM|={dM_at:.6e}"))

    w, M, _ = _mass_and_lambda(at, batch)
    row_sums = w.sum(dim=1)
    checks.append(
        (
            "no softmax (Σw ≠ 1)",
            not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3),
            f"Σw mean={row_sums.mean().item():.4f}",
        )
    )
    checks.append(
        (
            "M == Σw",
            torch.allclose(row_sums, M, atol=1e-5),
            f"max|M-Σw|={(M - row_sums).abs().max().item():.2e}",
        )
    )

    # Pediatric z convention
    sat = str(REPO_ROOT / "synthetic_age_temporal")
    if sat not in sys.path:
        sys.path.append(sat)
    from config import AGE_CENTER, AGE_SCALE  # noqa: E402
    from model_dtr import z_of_age  # noqa: E402

    z0 = float(z_of_age(torch.tensor([0.0])))
    z18 = float(z_of_age(torch.tensor([18.0])))
    checks.append(
        (
            "pediatric z=(a-9)/9",
            abs(AGE_CENTER - 9.0) < 1e-9
            and abs(AGE_SCALE - 9.0) < 1e-9
            and abs(z0 + 1.0) < 1e-6
            and abs(z18 - 1.0) < 1e-6,
            f"center={AGE_CENTER} scale={AGE_SCALE} z(0)={z0:.4f} z(18)={z18:.4f}",
        )
    )

    # Shared MIMIC TO init fingerprints (full BGE adapter)
    if SHARED_CP_STAGE1_DIR.exists():
        n_full = 30635
        to_full = MIMICCPDTRAdapter(n_codes=n_full, arm="temporal_only", d_model=64)
        at_full = MIMICCPDTRAdapter(n_codes=n_full, arm="age_temporal", d_model=64)
        meta_to = apply_shared_cp_init(to_full, SHARED_CP_STAGE1_DIR)
        meta_at = apply_shared_cp_init(at_full, SHARED_CP_STAGE1_DIR)
        fp_match = (
            meta_to["pretrained_fingerprint_ex_beta"]
            == meta_at["pretrained_fingerprint_ex_beta"]
        )
        checks.append(
            (
                "shared TO init fingerprints match",
                fp_match,
                f"fp={meta_to['pretrained_fingerprint_ex_beta'][:16]}…",
            )
        )
        checks.append(
            (
                "β=0 after shared init",
                abs(meta_to["beta_init"]) < 1e-12 and abs(meta_at["beta_init"]) < 1e-12,
                f"β_to={meta_to['beta_init']} β_at={meta_at['beta_init']}",
            )
        )
        checks.append(
            (
                "AT β learnable / TO β frozen",
                (not meta_to["beta_requires_grad"]) and meta_at["beta_requires_grad"],
                f"req_to={meta_to['beta_requires_grad']} req_at={meta_at['beta_requires_grad']}",
            )
        )
        del to_full, at_full
    else:
        checks.append(("shared TO ckpt present", False, f"missing {SHARED_CP_STAGE1_DIR}"))

    print("NCH CP DTR Stage-2 sanity checks")
    print("-" * 60)
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{status}] {name}: {detail}")
    print("-" * 60)
    print("ALL PASS" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
