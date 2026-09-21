"""Build Stage-2 sequences with Stage-1-compatible UNK retention (not silent drop)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.NCH.v2 import paths as P
from preprocessing.NCH.v2.cleanup import HISTORY_SLACK_DAYS

AGE_BANDS = (
    ("<1", 0.0, 1.0),
    ("1-5", 1.0, 6.0),
    ("6-11", 6.0, 12.0),
    ("12-17", 12.0, 18.0),
)


def _band(age_y: float) -> str:
    if age_y != age_y:
        return "missing"
    for name, lo, hi in AGE_BANDS:
        if lo <= age_y < hi:
            return name
    return ">=18"


def build_indexed_sequences(
    events: pd.DataFrame,
    indexes: pd.DataFrame,
    vocab: dict[str, int],
    *,
    modalities: tuple[str, ...],
    representation: str,
    retain_oov_as_unk: bool = True,
    max_seq_len: int = P.MAX_SEQ_LEN,
    token_col: str = "mimic_token",
    token_id_col: str = "mimic_token_id",
) -> tuple[list[dict], pd.DataFrame, dict]:
    """Build sequences for each index row.

    Stage-1 contract: OOV codes map to unk_vocab_index = |V|, then model id 1.
    When retain_oov_as_unk=True, OOV clinical events are KEPT as UNK (not dropped).
    Measurements are never sequence tokens.
    """
    unk = len(vocab)
    mods = set(modalities)
    ev = events[events["event_type"].isin(mods)].copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev = ev.dropna(subset=["event_time", "patient_id"])

    mid = ev[token_id_col] if token_id_col in ev.columns else pd.Series(pd.NA, index=ev.index)
    has_id = mid.notna()
    tok = ev[token_col] if token_col in ev.columns else pd.Series(pd.NA, index=ev.index)
    tok_str = tok.fillna("").astype(str)
    in_vocab = tok_str.isin(vocab) & ~has_id
    raw = ev["raw_code"].fillna("").astype(str).str.strip() if "raw_code" in ev.columns else pd.Series("", index=ev.index)
    clinical = ev["event_type"].isin(["diagnosis", "procedure", "medication", "drg"])
    not_redacted = ~raw.str.upper().isin({"", "REDACTED", "NI", "UN", "UNKNOWN", "NULL", "NA", "N/A"})
    keep_unk = retain_oov_as_unk & clinical & not_redacted & ~has_id & ~in_vocab

    code_id = pd.Series(np.full(len(ev), np.nan, dtype=np.float64), index=ev.index)
    if has_id.any():
        code_id.loc[has_id] = mid.loc[has_id].astype(float)
    if in_vocab.any():
        code_id.loc[in_vocab] = tok_str.loc[in_vocab].map(vocab).astype(float)
    code_id.loc[keep_unk] = float(unk)
    ev = ev.assign(code_id=code_id, is_unk=(code_id == unk))
    ev = ev[ev["code_id"].notna() & (ev["code_id"] >= 0)].copy()
    ev["code_id"] = ev["code_id"].astype(np.int64)

    samples: list[dict] = []
    meta_rows: list[dict] = []
    indexes = indexes.copy()
    indexes["index_time"] = pd.to_datetime(indexes["index_time"], errors="coerce")
    by_pat = {pid: g for pid, g in ev.groupby("patient_id", sort=False)}

    n_empty = 0
    n_hist_impossible = 0
    for r in indexes.itertuples(index=False):
        if pd.isna(r.index_time):
            continue
        g = by_pat.get(int(r.patient_id))
        if g is None:
            n_empty += 1
            continue
        sub = g[g["event_time"] < r.index_time].sort_values(
            ["event_time", "event_type", "raw_code"], kind="mergesort"
        )
        if sub.empty:
            n_empty += 1
            continue

        index_age = float(r.index_age_days) if pd.notna(r.index_age_days) else np.nan
        if index_age == index_age:
            ages = sub["age_at_event_days"].to_numpy(dtype=np.float64)
            ok = np.isnan(ages) | (ages <= index_age + HISTORY_SLACK_DAYS)
            times = sub["event_time"]
            dur = (r.index_time - times).dt.total_seconds().to_numpy() / 86400.0
            ok &= dur <= index_age + HISTORY_SLACK_DAYS
            if not ok.all():
                n_hist_impossible += int((~ok).sum())
                sub = sub.loc[ok]
            if sub.empty:
                n_empty += 1
                continue

        times = pd.to_datetime(sub["event_time"])
        t0 = times.iloc[0]
        ts_days = ((times - t0).dt.total_seconds().to_numpy(dtype=np.float64)) / 86400.0
        ages = sub["age_at_event_days"].to_numpy(dtype=np.float64)
        codes = sub["code_id"].to_numpy(dtype=np.int64)
        unk_flags = codes == unk
        n_before = int(codes.shape[0])
        hist_start = times.iloc[0]
        duration = (r.index_time - hist_start).total_seconds() / 86400.0
        earliest_age = float(np.nanmin(ages)) if np.isfinite(ages).any() else float("nan")

        if n_before > max_seq_len:
            codes = codes[-max_seq_len:]
            ts_days = ts_days[-max_seq_len:]
            ages = ages[-max_seq_len:]
            unk_flags = unk_flags[-max_seq_len:]
            retained_start = times.iloc[n_before - max_seq_len]
            retained_duration = (r.index_time - retained_start).total_seconds() / 86400.0
            earliest_age_ret = float(np.nanmin(ages)) if np.isfinite(ages).any() else float("nan")
        else:
            retained_duration = duration
            earliest_age_ret = earliest_age

        n_after = int(codes.shape[0])
        idx_age_y = index_age / P.DAYS_PER_YEAR if index_age == index_age else float("nan")
        sample = {
            "patient_id": int(r.patient_id),
            "sleep_study_id": int(r.sleep_study_id),
            "encounter_id": int(r.study_enc_id) if pd.notna(getattr(r, "study_enc_id", np.nan)) else -1,
            "index_time": r.index_time,
            "index_age_days": index_age,
            "sex": int(r.sex) if pd.notna(r.sex) else 0,
            "race": int(r.race) if pd.notna(r.race) else 6,
            "code_indices": codes,
            "timestamps_days": ts_days.astype(np.float32),
            "age_days": np.nan_to_num(ages, nan=0.0).astype(np.float32),
            "n_history_events": n_before,
            "history_duration": float(duration),
            "retained_duration": float(retained_duration),
            "sequence_length_before_truncation": n_before,
            "sequence_length_after_truncation": n_after,
            "n_unk": int(unk_flags.sum()),
            "frac_unk": float(unk_flags.mean()) if n_after else 0.0,
            "max_event_time": times.iloc[-1],
            "earliest_age_before": earliest_age,
            "earliest_age_after": earliest_age_ret,
            "age_band": _band(idx_age_y),
        }
        samples.append(sample)
        meta_rows.append({
            "patient_id": sample["patient_id"],
            "sleep_study_id": sample["sleep_study_id"],
            "index_time": str(r.index_time),
            "index_age_days": index_age,
            "index_age_years": idx_age_y,
            "age_band": sample["age_band"],
            "n_history_events": n_before,
            "sequence_length_before_truncation": n_before,
            "sequence_length_after_truncation": n_after,
            "history_duration": float(duration),
            "retained_duration": float(retained_duration),
            "frac_history_retained": float(retained_duration / duration) if duration > 0 else 1.0,
            "earliest_age_before": earliest_age,
            "earliest_age_after": earliest_age_ret,
            "n_unk": sample["n_unk"],
            "frac_unk": sample["frac_unk"],
            "sex": sample["sex"],
            "race": sample["race"],
            "representation": representation,
            "modalities": ",".join(modalities),
        })

    meta = pd.DataFrame(meta_rows)
    summary = {
        "representation": representation,
        "modalities": list(modalities),
        "retain_oov_as_unk": retain_oov_as_unk,
        "unk_vocab_index": unk,
        "n_samples": len(samples),
        "n_patients": int(meta["patient_id"].nunique()) if len(meta) else 0,
        "n_empty_indexes": n_empty,
        "n_impossible_history_events_dropped": n_hist_impossible,
        "unk_event_total": int(sum(s["n_unk"] for s in samples)),
        "unk_event_pct": (
            100.0 * sum(s["n_unk"] for s in samples)
            / max(1, sum(s["sequence_length_after_truncation"] for s in samples))
        ),
        "pct_sequences_with_unk": (
            100.0 * sum(1 for s in samples if s["n_unk"] > 0) / max(1, len(samples))
        ),
        "truncation_pct": (
            100.0 * sum(1 for s in samples if s["sequence_length_before_truncation"] > max_seq_len)
            / max(1, len(samples))
        ),
        "events_total_after_truncation": int(sum(s["sequence_length_after_truncation"] for s in samples)),
        "events_per_patient_mean": (
            float(np.mean([s["sequence_length_after_truncation"] for s in samples])) if samples else 0.0
        ),
    }
    return samples, meta, summary


def write_sequence_npz(path: Path, samples: list[dict], unk: int) -> None:
    if not samples:
        return
    seq_len = np.asarray([s["code_indices"].shape[0] for s in samples], dtype=np.int64)
    offsets = np.zeros(len(samples) + 1, dtype=np.int64)
    np.cumsum(seq_len, out=offsets[1:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        subject_id=np.asarray([s["patient_id"] for s in samples], dtype=np.int64),
        hadm_id=np.asarray([s["encounter_id"] for s in samples], dtype=np.int64),
        sleep_study_id=np.asarray([s["sleep_study_id"] for s in samples], dtype=np.int64),
        label=np.full(len(samples), np.nan, dtype=np.float32),
        sex=np.asarray([s["sex"] for s in samples], dtype=np.int8),
        race=np.asarray([s["race"] for s in samples], dtype=np.int16),
        n_events_in_window=np.asarray([s["n_history_events"] for s in samples], dtype=np.int64),
        unk_vocab_index=np.asarray([unk], dtype=np.int64),
        offsets=offsets,
        code_indices=np.concatenate([s["code_indices"] for s in samples]).astype(np.int64),
        timestamps_days=np.concatenate([s["timestamps_days"] for s in samples]).astype(np.float32),
        age_days=np.concatenate([s["age_days"] for s in samples]).astype(np.float32),
        index_age_days=np.asarray([s["index_age_days"] for s in samples], dtype=np.float32),
        history_duration_days=np.asarray([s["history_duration"] for s in samples], dtype=np.float32),
        seq_len_before=np.asarray([s["sequence_length_before_truncation"] for s in samples], dtype=np.int64),
        seq_len_after=np.asarray([s["sequence_length_after_truncation"] for s in samples], dtype=np.int64),
        n_unk=np.asarray([s["n_unk"] for s in samples], dtype=np.int64),
    )


def assert_no_index_encounter_leakage(
    indexes: pd.DataFrame, clean_events: pd.DataFrame
) -> dict:
    """Assert no linked PSG-encounter event remains in the clean event table."""
    sleep = indexes.dropna(subset=["study_enc_id"]).copy()
    sleep["study_enc_id"] = sleep["study_enc_id"].astype("int64")
    sleep["patient_id"] = sleep["patient_id"].astype("int64")
    keys = set(zip(sleep["patient_id"], sleep["study_enc_id"]))
    ev = clean_events.dropna(subset=["encounter_id"]).copy()
    if ev.empty or not keys:
        return {"psg_encounter_events_in_clean_table": 0, "passed": True}
    hits = [
        (int(p), int(e)) in keys
        for p, e in zip(ev["patient_id"].astype("int64"), ev["encounter_id"].astype("int64"))
    ]
    n = int(sum(hits))
    return {"psg_encounter_events_in_clean_table": n, "passed": n == 0}
