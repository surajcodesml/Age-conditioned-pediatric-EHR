"""NCH Sleep DataBank preprocessing that conforms to the Stage-1 MIMIC contract.

MIMIC defines the representation. These modules map NCH into that contract; they
do not rebuild the MIMIC vocabulary, embeddings, or temporal/age constants.
"""
from __future__ import annotations

__all__ = ["paths"]
