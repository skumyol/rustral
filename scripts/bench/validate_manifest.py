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
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "manifest_schema.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise SystemExit(f"missing schema: {SCHEMA_PATH}")
    return _load_json(SCHEMA_PATH)


def _try_jsonschema():
    try:
        import jsonschema  # type: ignore

        return jsonschema
    except Exception:
        return None


def _fallback_validate_one(obj: Dict[str, Any], path: Path) -> List[str]:
    """
    Handwritten validator used when `jsonschema` is not installed.

    This is intentionally minimal: it enforces the keys this repo relies on, and
    the SST-2 diagnostics fields that power parity debugging.
    """
    errs: List[str] = []

    def req_keys(o: Any, keys: List[str], where: str) -> None:
        if not isinstance(o, dict):
            errs.append(f"{path}: {where} must be an object")
            return
        missing = [k for k in keys if k not in o]
        if missing:
            errs.append(f"{path}: {where} missing keys: {missing}")

    req_keys(obj, ["schema_version", "created_at", "version", "task", "metric", "aggregate", "runs"], "<root>")
    if "runs" in obj and isinstance(obj["runs"], list):
        for i, r in enumerate(obj["runs"]):
            req_keys(r, ["seed", "manifest"], f"runs[{i}]")
            manifest = r.get("manifest")
            if not isinstance(manifest, dict):
                continue
            if obj.get("task") == "sst2_classifier":
                # Allow PyTorch manifests (they won't have diagnostics).
                if manifest.get("framework") == "pytorch":
                    continue
                diag = manifest.get("diagnostics")
                if not isinstance(diag, dict):
                    errs.append(f"{path}: runs[{i}].manifest.diagnostics is required for sst2_classifier")
                    continue
                req_keys(
                    diag,
                    [
                        "train_label_counts",
                        "dev_label_counts",
                        "dev_confusion_matrix",
                        "dev_predicted_counts",
                        "dev_positive_prob_hist",
                    ],
                    f"runs[{i}].manifest.diagnostics",
                )
    else:
        errs.append(f"{path}: runs must be a non-empty array")

    return errs


def _validate_one(path: Path, schema: Dict[str, Any]) -> None:
    obj = _load_json(path)
    js = _try_jsonschema()
    if js is not None:
        js.validate(instance=obj, schema=schema)
        return
    errs = _fallback_validate_one(obj, path)
    if errs:
        raise SystemExit("\n".join(errs))


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

