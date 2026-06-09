"""PIC (Paediatric Intensive Care, v1.1.0) preprocessing pipeline.

Mirrors the MIMIC-IV preprocessing pipeline in the parent ``preprocessing/`` package
so that downstream ``dataset.py`` / ``tensorize.py`` / model code run unchanged.

Shared helpers (PheCode maps, BGE-m3 embedding routine, patient-level split logic)
are imported from the parent package rather than reimplemented; see ``_shared.py``.
"""
