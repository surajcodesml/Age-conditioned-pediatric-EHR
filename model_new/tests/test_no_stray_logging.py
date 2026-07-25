#!/usr/bin/env python3
"""The grep half of INV-LOG, kept separate so the failure names file and line.

D11: the legacy tree emitted the same w(t) statistics from three call sites in three
formats. Modules here return diagnostic tensors; diagnostics.py owns all formatting.
"""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ALLOWED = {"diagnostics.py"}
PRINT_RE = re.compile(r"(?<![\w.])print\s*\(")

# The tests directory is excluded: these two files necessarily contain the literal strings
# they search for, and the invariant is about the library, not about its checker.
SKIP_DIRS = {"__pycache__", "tests"}


def _library_files():
    return [p for p in sorted(PKG.rglob("*.py")) if not (SKIP_DIRS & set(p.parts))]


def test_grep_print():
    hits = []
    for path in _library_files():
        if path.name in ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if PRINT_RE.search(line):
                hits.append(f"  {path.relative_to(PKG.parent)}:{i}\n      {line.strip()}")
    assert not hits, (
        "grep -rn 'print(' model_new/ must match only diagnostics.py.\n"
        "Offending lines:\n" + "\n".join(hits))


def test_no_module_level_debug_sample():
    hits = []
    for path in _library_files():
        text = path.read_text()
        for needle in ("debug_sample", "DEBUG_SAMPLE"):
            if needle in text:
                hits.append(f"{path.relative_to(PKG.parent)}: {needle}")
    assert not hits, f"module-level debug hooks found: {hits}"
