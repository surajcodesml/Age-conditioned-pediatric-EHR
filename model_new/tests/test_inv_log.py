#!/usr/bin/env python3
"""INV-LOG -- no print outside diagnostics.py, and no imports from the legacy trees."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
FORBIDDEN_PACKAGES = ("model.", "model_ablation", "finetune")


# tests/ is excluded from the print scan: the checker files necessarily contain the literal
# string they search for. The import scan still covers everything.
def _py_files(include_tests: bool = True):
    skip = {"__pycache__"} | (set() if include_tests else {"tests"})
    return sorted(p for p in PKG.rglob("*.py") if not (skip & set(p.parts)))


def test_no_print_outside_diagnostics():
    offenders = []
    for path in _py_files(include_tests=False):
        if path.name == "diagnostics.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "print(" not in line:
                continue
            if any(tok in line for tok in ("D.print", "diagnostics.print", "print_block",
                                           "print_kv", "pprint", ".print_")):
                continue
            offenders.append(f"{path.relative_to(PKG.parent)}:{i}: {stripped}")
    assert not offenders, "print() outside diagnostics.py:\n" + "\n".join(offenders)


def test_no_imports_from_legacy_trees():
    offenders = []
    for path in _py_files():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            s = line.strip()
            if not (s.startswith("import ") or s.startswith("from ")):
                continue
            for pkg in FORBIDDEN_PACKAGES:
                if pkg in s and "model_new" not in s:
                    offenders.append(f"{path.relative_to(PKG.parent)}:{i}: {s}")
    assert not offenders, "imports from a legacy tree:\n" + "\n".join(offenders)


def test_diagnostics_owns_the_json_writers():
    """One module owns all output: D11."""
    offenders = []
    for path in _py_files():
        if path.name in ("diagnostics.py",) or path.parent.name == "tests":
            continue
        text = path.read_text()
        if "json.dump" in text:
            offenders.append(str(path.relative_to(PKG.parent)))
    assert not offenders, f"json.dump outside diagnostics.py: {offenders}"
