#!/usr/bin/env python3
"""Bridge script to run the external MIMIC-IV lab preprocessing pipeline safely.

This script does not modify the external repository. It validates local paths and
prints/executes the command needed to run the external pipeline modules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_paths(repo_root: Path) -> dict[str, Path]:
    external_repo = repo_root / "external" / "MIMIC-IV-Data-Pipeline"
    processed_root = repo_root / "data" / "processed"
    raw_root = repo_root / "data" / "raw" / "mimic-iv-ed" / "physionet.org" / "files" / "mimic-iv-ed" / "2.2" / "ed"
    return {
        "external_repo": external_repo,
        "processed_root": processed_root,
        "raw_ed_root": raw_root,
    }


def _validate_paths(paths: dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = "\n".join(f"- {name}: {paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing required paths:\n{details}")


def build_command(external_repo: Path, version_path: str, cohort_output: str) -> list[str]:
    # External script path and args as expected by the lab pipeline.
    script = external_repo / "preprocessing" / "hosp_module_preproc" / "feature_selection_hosp.py"
    if not script.exists():
        raise FileNotFoundError(f"Cannot find external script: {script}")
    return [sys.executable, str(script), "--version_path", version_path, "--cohort_output", cohort_output]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external MIMIC-IV lab pipeline wrapper.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to Age-conditioned-pediatric-EHR repository root.",
    )
    parser.add_argument(
        "--version-path",
        default="mimiciv/2.0",
        help="Version path expected by external scripts (default: mimiciv/2.0).",
    )
    parser.add_argument(
        "--cohort-output",
        default="cohort_output",
        help="External pipeline cohort output file prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate paths and print command.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paths = _resolve_paths(repo_root)
    _validate_paths(paths)
    paths["processed_root"].mkdir(parents=True, exist_ok=True)

    print("Detected MIMIC-IV-ED v2.2 input:", paths["raw_ed_root"])
    print("External pipeline repo:", paths["external_repo"])
    if args.version_path not in {"mimiciv/1.0", "mimiciv/2.0"}:
        print(
            "Warning: external pipeline contains explicit branching for mimiciv/1.0 and mimiciv/2.0 in lab preprocessing.",
            file=sys.stderr,
        )

    cmd = build_command(paths["external_repo"], args.version_path, args.cohort_output)
    print("Command:", " ".join(cmd))
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(paths["external_repo"]), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
