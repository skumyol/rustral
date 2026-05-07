#!/usr/bin/env python3
"""
Validate benchmark JSON files against `benchmarks/schema_v2.json`.

Accepts either:
  - a per-suite document (top-level keys: suite, schema_version, machine, samples)
  - a combined harness document (top-level keys: tool, suites)
    -> in this case each entry of `suites[]` is validated against the schema.

Exits non-zero on any validation failure or invalid input.

Dependencies: `jsonschema` (pip-install if missing). For environments without
network access the validator falls back to a small handwritten checker that
covers the same required keys, so this script never blocks on a missing pip
install in CI.

Usage:
    python3 scripts/bench/validate_schema.py benchmarks/results/*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "schema_v2.json"


def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def _try_jsonschema():
    try:
        import jsonschema  # type: ignore

        return jsonschema
    except Exception:
        return None


REQUIRED_SUITE_KEYS = {"suite", "schema_version", "machine", "samples"}
REQUIRED_MACHINE_KEYS = {"os", "arch", "hostname"}
REQUIRED_SAMPLE_KEYS = {"name", "backend", "device", "dtype", "params", "runs_ms"}


def _fallback_validate_suite(doc: Dict[str, Any], path: str) -> List[str]:
    errs: List[str] = []
    missing = REQUIRED_SUITE_KEYS - set(doc.keys())
    if missing:
        errs.append(f"{path}: suite document missing keys: {sorted(missing)}")
    sv = doc.get("schema_version", "")
    if not isinstance(sv, str) or not sv.startswith("2."):
        errs.append(f"{path}: schema_version must be 2.x.y, got {sv!r}")
    machine = doc.get("machine") or {}
    if not isinstance(machine, dict):
        errs.append(f"{path}: machine must be an object")
    else:
        m_missing = REQUIRED_MACHINE_KEYS - set(machine.keys())
        if m_missing:
            errs.append(f"{path}: machine missing keys: {sorted(m_missing)}")
    samples = doc.get("samples") or []
    if not isinstance(samples, list):
        errs.append(f"{path}: samples must be an array")
    else:
        for i, s in enumerate(samples):
            if not isinstance(s, dict):
                errs.append(f"{path}: samples[{i}] is not an object")
                continue
            s_missing = REQUIRED_SAMPLE_KEYS - set(s.keys())
            if s_missing:
                errs.append(f"{path}: samples[{i}] missing keys: {sorted(s_missing)}")
            rm = s.get("runs_ms")
            if not isinstance(rm, list) or not rm:
                errs.append(f"{path}: samples[{i}].runs_ms must be a non-empty array")
    return errs


def iter_suite_documents(doc: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (label, suite_doc) pairs from either a single-suite or combined document."""
    if "suites" in doc and isinstance(doc.get("suites"), list):
        for i, s in enumerate(doc["suites"]):
            yield f"suites[{i}]", s
    else:
        yield "<root>", doc


def validate_file(path: Path) -> List[str]:
    try:
        doc = json.loads(path.read_text())
    except Exception as e:
        return [f"{path}: failed to parse JSON: {e}"]

    errs: List[str] = []
    schema = load_schema()
    js = _try_jsonschema()
    for label, suite_doc in iter_suite_documents(doc):
        prefix = f"{path}::{label}"
        if js is not None:
            try:
                js.validate(instance=suite_doc, schema=schema)
            except Exception as e:  # pragma: no cover - error path
                errs.append(f"{prefix}: jsonschema validation failed: {e}")
        else:
            errs.extend(_fallback_validate_suite(suite_doc, prefix))
    return errs


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: validate_schema.py <json> [<json> ...]", file=sys.stderr)
        return 2
    failed = 0
    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            failed += 1
            continue
        errs = validate_file(path)
        if errs:
            failed += 1
            for e in errs:
                print(e, file=sys.stderr)
        else:
            print(f"OK: {path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
