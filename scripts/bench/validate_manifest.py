#!/usr/bin/env python3
"""
Validate curated NLP manifests under benchmarks/runs/ against benchmarks/manifest_schema.json.

Usage:
  python3 scripts/bench/validate_manifest.py
  python3 scripts/bench/validate_manifest.py benchmarks/runs/v0.1.0/nlp/sst2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "manifest_schema.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise SystemExit(f"missing schema: {SCHEMA_PATH}")
    return _load_json(SCHEMA_PATH)


def _validate_one(path: Path, schema: Dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "jsonschema is required. Install with: python -m pip install jsonschema\n"
            f"import error: {e}"
        )

    obj = _load_json(path)
    jsonschema.validate(instance=obj, schema=schema)


def _default_targets() -> List[Path]:
    runs_root = REPO_ROOT / "benchmarks" / "runs"
    if not runs_root.exists():
        return []
    return sorted(runs_root.glob("v*/nlp/*.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="files to validate (default: all v*/nlp/*.json)")
    args = ap.parse_args()

    schema = _load_schema()
    targets = [Path(p) for p in args.paths] if args.paths else _default_targets()
    targets = [p if p.is_absolute() else (REPO_ROOT / p) for p in targets]

    if not targets:
        raise SystemExit("no manifests found")

    for p in targets:
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        _validate_one(p, schema)

    print(f"ok: validated {len(targets)} manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

