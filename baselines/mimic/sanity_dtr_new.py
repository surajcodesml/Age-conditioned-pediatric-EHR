#!/usr/bin/env python3
"""Quick forward/gradient sanity checks for MIMIC Content-Persistence DTR ``_new``.

Checks (must all pass before full training):
  1. temporal_only λ is age-invariant
  2. age_temporal λ changes with age when β ≠ 0
  3. β receives gradients on age_temporal
  4. evidence mass M changes with age when β ≠ 0 (AT) / not when β=0 (TO)
  5. history weights are raw-additive (no softmax: Σw is not forced to 1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from baselines.mimic.runner import MIMICCPDTRAdapter  # noqa: E402


def _fake_batch(n_codes: int = 512, B: int = 4, M: int = 8, C: int = 4, device="cpu"):
    enc_code_ids = torch.randint(1, min(n_codes, 200), (B, M, C), device=device)
    enc_code_mask = torch.ones(B, M, C, dtype=torch.bool, device=device)
    enc_tau = torch.linspace(0.1, 2.0, M, device=device).unsqueeze(0).expand(B, M).contiguous()
    enc_padding_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    age = torch.tensor([2.0, 8.0, 14.0, 20.0], device=device)[:B]
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


def _mass_and_lambda(model: MIMICCPDTRAdapter, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model.model(
        enc_code_ids=batch["enc_code_ids"],
        enc_code_mask=batch["enc_code_mask"],
        enc_tau=batch["enc_tau"],
        enc_padding_mask=batch["enc_padding_mask"],
        age=batch["age"],
        return_parts=True,
    )
    w = out["w"]
    M = out["M"].squeeze(-1)
    lam = out["lambda"]
    return w, M, lam


def main() -> int:
    device = torch.device("cpu")
    n_codes = 512  # small head for sanity; wiring identical to full MIMIC
    # Temporarily monkeypatch BGE path by building then swapping table size.
    # Full BGE load is slow; use a random frozen table of matching API.
    to = MIMICCPDTRAdapter.__new__(MIMICCPDTRAdapter)
    torch.nn.Module.__init__(to)
    sat = str(REPO_ROOT / "synthetic_age_temporal")
    if sat not in sys.path:
        sys.path.insert(0, sat)
    from model_dtr import DevelopmentalTemporalRetrieval

    def build(arm: str) -> MIMICCPDTRAdapter:
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
        # Lightweight stand-in for frozen BGE+proj used in the real MIMIC adapter.
        enc = m.model.encounter_encoder
        del enc.code_emb
        enc.register_buffer(
            "code_emb_table", torch.randn(n_codes, 128), persistent=True
        )
        enc.code_proj = torch.nn.Linear(128, 64, bias=False)
        def _encode(enc_code_ids, enc_code_mask, _enc=enc):
            e = _enc.code_proj(_enc.code_emb_table[enc_code_ids])
            mask = enc_code_mask.to(e.dtype).unsqueeze(-1)
            summed = (e * mask).sum(dim=2)
            denom = mask.sum(dim=2).clamp(min=1.0)
            return _enc.enc_mlp(summed / denom)
        enc.forward = _encode  # type: ignore[method-assign]
        # Keep random head init here so β is in the loss graph for the probe.
        # Production MIMIC training still zero-inits the 30k-way heads.
        return m

    to = build("temporal_only")
    at = build("age_temporal")
    # Force nonzero β for the age-sensitivity probe.
    with torch.no_grad():
        at.model.beta.fill_(-1.0)

    batch = _fake_batch(n_codes=n_codes, device=device)
    checks: list[tuple[str, bool, str]] = []

    # 1. TO λ age-invariant
    ages = [torch.full((4,), a, device=device) for a in (2.0, 10.0, 18.0)]
    lams_to = []
    for a in ages:
        b = {**batch, "age": a}
        _, _, lam = _mass_and_lambda(to, b)
        lams_to.append(lam.detach())
    to_ok = all(torch.allclose(lams_to[0], x, atol=1e-6) for x in lams_to[1:])
    checks.append(("TO λ age-invariant", to_ok, f"λ rows equal across ages={to_ok}"))

    # 2. AT λ changes with age when β≠0
    lams_at = []
    for a in ages:
        b = {**batch, "age": a}
        _, _, lam = _mass_and_lambda(at, b)
        lams_at.append(lam.detach())
    at_lam_diff = (lams_at[0] - lams_at[-1]).abs().mean().item()
    at_ok = at_lam_diff > 1e-4
    checks.append(("AT λ age-sensitive", at_ok, f"mean|Δλ|={at_lam_diff:.6f}"))

    # 3. β gradient on AT
    at.zero_grad(set_to_none=True)
    logits = at.model(
        enc_code_ids=batch["enc_code_ids"],
        enc_code_mask=batch["enc_code_mask"],
        enc_tau=batch["enc_tau"],
        enc_padding_mask=batch["enc_padding_mask"],
        age=batch["age"],
    )
    loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
    loss.backward()
    beta_grad = at.model.beta.grad
    beta_ok = beta_grad is not None and float(beta_grad.abs()) > 0
    checks.append(("AT β receives grad", beta_ok, f"β.grad={None if beta_grad is None else float(beta_grad):.6e}"))

    # TO β frozen / unused (parameter exists but age_temporal=False)
    to.zero_grad(set_to_none=True)
    logits_to = to.model(
        enc_code_ids=batch["enc_code_ids"],
        enc_code_mask=batch["enc_code_mask"],
        enc_tau=batch["enc_tau"],
        enc_padding_mask=batch["enc_padding_mask"],
        age=batch["age"],
    )
    F.binary_cross_entropy_with_logits(logits_to, batch["labels"]).backward()
    to_beta_grad = to.model.beta.grad
    # β may get None grad if unused in graph
    to_beta_detail = (
        "β.grad=None"
        if to_beta_grad is None
        else f"β.grad={float(to_beta_grad):.6e}"
    )
    checks.append(
        (
            "TO β unused in graph",
            to_beta_grad is None or float(to_beta_grad.abs()) == 0.0,
            to_beta_detail,
        )
    )

    # 4. Mass M age dependence
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

    # 5. No softmax: row sums of w need not be 1; vary across examples
    w, M, _ = _mass_and_lambda(at, batch)
    row_sums = w.sum(dim=1)
    not_softmax = not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    mass_matches = torch.allclose(row_sums, M, atol=1e-5)
    checks.append(("no softmax (Σw ≠ 1)", not_softmax, f"Σw mean={row_sums.mean().item():.4f}"))
    checks.append(("M == Σw", mass_matches, f"max|M-Σw|={(M-row_sums).abs().max().item():.2e}"))

    print("MIMIC CP DTR sanity checks")
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
