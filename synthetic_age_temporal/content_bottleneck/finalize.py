"""End-to-end: audit → E01 → aggregate → report."""
from __future__ import annotations

import argparse
from pathlib import Path

from content_bottleneck import ARTIFACT_ROOT
from content_bottleneck.aggregate import aggregate
from content_bottleneck.figures import make_all_figures
from content_bottleneck.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["aggregate"], default="aggregate")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.artifact_root or ARTIFACT_ROOT
    aggregate(root)
    make_all_figures(root)
    print(f"wrote {write_report(root)}", flush=True)


if __name__ == "__main__":
    main()
