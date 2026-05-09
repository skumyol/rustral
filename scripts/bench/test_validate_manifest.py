#!/usr/bin/env python3
"""
CLI smoke test for `validate_manifest.py`: validates at least one curated NLP aggregate.

Run from repo root:
  python3 scripts/bench/test_validate_manifest.py

Used from CI `rust` job so manifest validation is exercised without waiting on bench-cpu.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    vm = REPO_ROOT / "scripts" / "bench" / "validate_manifest.py"
    candidate = REPO_ROOT / "benchmarks" / "runs" / "v0.1.0" / "nlp" / "sst2.json"
    if not candidate.exists():
        print(f"skip: missing fixture {candidate.relative_to(REPO_ROOT)}")
        return 0
    r = subprocess.run(
        [sys.executable, str(vm), str(candidate)],
        cwd=str(REPO_ROOT),
    )
    return int(r.returncode)


if __name__ == "__main__":
    sys.exit(main())
