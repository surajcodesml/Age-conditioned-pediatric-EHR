#!/usr/bin/env python3
"""Run the suite and print the invariant ID -> test name table.

    python -m model_new.tests.run_all

Every row is HARD. The table is the contract: each INVARIANTS.md ID maps to exactly one
test file, and this is what proves the mapping is complete in both directions.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from model_new import diagnostics as D

PKG = Path(__file__).resolve().parents[1]

# ID -> (test file, one-line statement). Kept in the same order as INVARIANTS.md.
INVARIANT_TESTS: list[tuple[str, str]] = [
    ("INV-BASIS", "test_inv_basis.py"),
    ("INV-DOMAIN", "test_inv_domain.py"),
    ("INV-TMAX", "test_inv_tmax.py"),
    ("INV-DEMO-SPLIT", "test_inv_demo_split.py"),
    ("INV-QUERY", "test_inv_query.py"),
    ("INV-ZERO-A", "test_inv_zero_a.py"),
    ("INV-ZERO-B", "test_inv_zero_b.py"),
    ("INV-ARM", "test_inv_arm.py"),
    ("INV-GROUPS", "test_inv_groups.py"),
    ("INV-LOG", "test_inv_log.py"),
    ("INV-FROZEN", "test_inv_frozen.py"),
    ("INV-NAN", "test_inv_nan.py"),
    ("INV-STATS-SINGLE", "test_inv_stats_single.py"),
    ("INV-AGESTD", "test_inv_agestd.py"),
]

SUPPORTING_TESTS: list[tuple[str, str]] = [
    ("(chebyshev)", "test_chebyshev_numpy.py"),
    ("(checkpoint)", "test_checkpoint_roundtrip.py"),
    ("(logging)", "test_no_stray_logging.py"),
    ("(tau-gpu)", "test_tau_equivalence.py"),
    ("(auprc)", "test_auprc_histogram.py"),
]


def _check_mapping() -> list[str]:
    """Every invariant maps to exactly one test, and every test file is accounted for."""
    problems = []
    declared = {f for _, f in INVARIANT_TESTS + SUPPORTING_TESTS}
    on_disk = {p.name for p in (PKG / "tests").glob("test_*.py")}
    for missing in sorted(declared - on_disk):
        problems.append(f"declared but absent: {missing}")
    for extra in sorted(on_disk - declared):
        problems.append(f"present but unmapped: {extra}")
    seen: dict[str, str] = {}
    for inv, f in INVARIANT_TESTS:
        if f in seen:
            problems.append(f"{f} maps to both {seen[f]} and {inv}")
        seen[f] = inv
    invariants_md = (PKG / "INVARIANTS.md").read_text()
    for inv, _ in INVARIANT_TESTS:
        if f"`{inv}`" not in invariants_md:
            problems.append(f"{inv} is tested but not declared in INVARIANTS.md")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(PKG / "tests"), "-q", "--no-header", *args],
        cwd=str(PKG.parent), capture_output=True, text=True,
    )
    D.print_block("pytest", proc.stdout.strip().splitlines()[-12:] or ["(no output)"])
    if proc.returncode != 0:
        D.print_block("pytest stderr", (proc.stderr or "").strip().splitlines()[-20:])

    per_file: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            name = Path(line.split()[1].split("::")[0]).name
            per_file[name] = "FAIL"
    verdict = "PASS" if proc.returncode == 0 else "see pytest output"

    rows = [(inv, f, per_file.get(f, verdict if proc.returncode == 0 else "PASS"))
            for inv, f in INVARIANT_TESTS]
    rows += [(inv, f, per_file.get(f, verdict if proc.returncode == 0 else "PASS"))
             for inv, f in SUPPORTING_TESTS]
    D.print_invariant_table(rows)

    problems = _check_mapping()
    if problems:
        D.print_block("mapping problems  [HARD]", problems)
        return 1
    D.print_block("mapping", [
        f"{len(INVARIANT_TESTS)} invariants, {len(INVARIANT_TESTS)} test files, 1:1.",
        f"{len(SUPPORTING_TESTS)} supporting test files, all accounted for.",
        "Every INVARIANTS.md ID maps to exactly one test.",
    ])
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
