"""Convert MIMIC token-level batches into Content-Persistence encounter batches.

Grouping rule (matches synthetic DTR):
  - Co-timed tokens (same ``timestamps_days``) form one encounter.
  - Encounter τ = lag_to_tau(t_last − t_enc) using the Stage-1 /7 log1p convention.
  - Age = age at the last valid token (prediction time).
  - Encounters ordered oldest → newest; if truncated, keep the newest.

Performance notes:
  - Uses NumPy on CPU (fast for DataLoader workers).
  - Dynamic per-batch padding (not fixed 64×64) to cut wasted GEMM work.
  - Caps still apply as upper bounds.
"""
from __future__ import annotations

import numpy as np
import torch

from model_new.data import lag_to_tau

# Cap codes per encounter (rare MIMIC days exceed this; take first codes).
DEFAULT_MAX_CODES_PER_ENCOUNTER = 64
# Cap encounters per example (keep newest).
DEFAULT_MAX_ENCOUNTERS = 64


def tokens_to_encounter_batch(
    batch: dict[str, torch.Tensor],
    *,
    max_codes_per_encounter: int = DEFAULT_MAX_CODES_PER_ENCOUNTER,
    max_encounters: int = DEFAULT_MAX_ENCOUNTERS,
) -> dict[str, torch.Tensor]:
    """Map a MIMIC RenameLoader batch → enc_* tensors for DevelopmentalTemporalRetrieval.

    Required input keys: ``code_ids``/``code_indices``, ``timestamps_days``/``tau``,
    ``attention_mask``/``padding_mask``, ``age_years``/``age``, ``labels``.
    """
    code_ids_t = batch.get("code_ids", batch.get("code_indices"))
    timestamps_t = batch.get("timestamps_days", batch.get("tau"))
    if "attention_mask" in batch:
        attn_t = batch["attention_mask"].bool()
    else:
        attn_t = ~batch["padding_mask"].bool()
    age_years_t = batch.get("age_years")
    age_scalar_t = batch.get("age")
    labels = batch["labels"]

    # Work on CPU numpy — DataLoader workers should stay off-GPU.
    code_ids = code_ids_t.detach().cpu().numpy()
    timestamps = timestamps_t.detach().cpu().numpy().astype(np.float64, copy=False)
    attn = attn_t.detach().cpu().numpy().astype(bool, copy=False)
    if age_years_t is not None:
        age_years = age_years_t.detach().cpu().numpy()
    else:
        age_years = None
        age_scalar = age_scalar_t.detach().cpu().numpy().astype(np.float64, copy=False)

    B = code_ids.shape[0]
    per_ex: list[tuple[list[np.ndarray], np.ndarray, float]] = []
    max_m = 1
    max_c = 1

    for b in range(B):
        valid = attn[b]
        codes_b = code_ids[b][valid]
        ts_b = timestamps[b][valid]
        if codes_b.size == 0:
            age_b = float(age_scalar[b]) if age_years is None else 0.0
            per_ex.append(([np.zeros(1, dtype=np.int64)], np.zeros(1, dtype=np.float64), age_b))
            continue
        if age_years is not None:
            age_b = float(age_years[b][valid][-1])
        else:
            age_b = float(age_scalar[b])
        t_last = float(ts_b[-1])

        # Group consecutive equal timestamps after sorting by time.
        order = np.argsort(ts_b, kind="mergesort")
        ts_s = ts_b[order]
        codes_s = codes_b[order]
        # Boundaries where timestamp changes
        change = np.empty(ts_s.size, dtype=bool)
        change[0] = True
        change[1:] = ts_s[1:] != ts_s[:-1]
        starts = np.flatnonzero(change)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = ts_s.size

        enc_codes: list[np.ndarray] = []
        enc_tau_days: list[float] = []
        for s, e in zip(starts, ends):
            chunk = codes_s[s:e][:max_codes_per_encounter]
            enc_codes.append(chunk.astype(np.int64, copy=False))
            enc_tau_days.append(t_last - float(ts_s[s]))
        if max_encounters is not None and len(enc_codes) > max_encounters:
            enc_codes = enc_codes[-max_encounters:]
            enc_tau_days = enc_tau_days[-max_encounters:]
        tau_arr = np.asarray(enc_tau_days, dtype=np.float64)
        per_ex.append((enc_codes, tau_arr, age_b))
        max_m = max(max_m, len(enc_codes))
        max_c = max(max_c, max((c.size for c in enc_codes), default=1))

    max_m = min(max_m, max_encounters) if max_encounters is not None else max_m
    max_c = min(max_c, max_codes_per_encounter)

    enc_code_ids = torch.zeros(B, max_m, max_c, dtype=torch.long)
    enc_code_mask = torch.zeros(B, max_m, max_c, dtype=torch.bool)
    enc_tau = torch.zeros(B, max_m, dtype=torch.float32)
    enc_padding_mask = torch.ones(B, max_m, dtype=torch.bool)  # True=pad
    age = torch.zeros(B, dtype=torch.float32)

    for b, (enc_codes, enc_tau_days, age_b) in enumerate(per_ex):
        age[b] = age_b
        if not enc_codes:
            continue
        tau_vals = lag_to_tau(torch.from_numpy(enc_tau_days.astype(np.float32)))
        for m, codes in enumerate(enc_codes):
            n = int(codes.size)
            enc_code_ids[b, m, :n] = torch.from_numpy(codes[:n])
            enc_code_mask[b, m, :n] = True
            enc_tau[b, m] = tau_vals[m]
            enc_padding_mask[b, m] = False

    return {
        "enc_code_ids": enc_code_ids,
        "enc_code_mask": enc_code_mask,
        "enc_tau": enc_tau,
        "enc_padding_mask": enc_padding_mask,
        "age": age,
        "labels": labels.detach().cpu() if torch.is_tensor(labels) else labels,
    }
