#!/usr/bin/env python3
"""
Compare two schema-v2 harness JSON files (per-suite or combined `tool` + `suites`).

Matches samples by (suite, name, params). Reports mean_ms ratio current/baseline.
Use --fail-on-regression with --max-slowdown to exit 1 when current is slower than
baseline * max_slowdown (e.g. 1.15 allows up to 15% slower).

Does not run benchmarks; intended for local or opt-in CI (default PR CI stays timing-agnostic).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

def _read(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _iter_samples(doc: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if "suites" in doc and isinstance(doc["suites"], list):
        for block in doc["suites"]:
            suite = str(block.get("suite", "?"))
            for s in block.get("samples") or []:
                if isinstance(s, dict):
                    yield suite, s
    else:
        suite = str(doc.get("suite", "?"))
        for s in doc.get("samples") or []:
            if isinstance(s, dict):
                yield suite, s


def _key(suite: str, sample: Dict[str, Any]) -> Tuple[str, str, Tuple[Tuple[str, str], ...]]:
    name = str(sample.get("name", ""))
    params = sample.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    pt = tuple(sorted((str(k), str(v)) for k, v in params.items()))
    return (suite, name, pt)


def _index(doc: Dict[str, Any]) -> Dict[Tuple[str, str, Tuple[Tuple[str, str], ...]], Dict[str, Any]]:
    out: Dict[Tuple[str, str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
    for suite, s in _iter_samples(doc):
        out[_key(suite, s)] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True, help="reference harness JSON")
    ap.add_argument("--current", type=Path, required=True, help="candidate harness JSON")
    ap.add_argument(
        "--max-slowdown",
        type=float,
        default=1.15,
        help="allowed current_mean / baseline_mean (default 1.15)",
    )
    ap.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any matched sample exceeds --max-slowdown",
    )
    ap.add_argument(
        "--require-matching-suites",
        action="store_true",
        help="require every baseline key to exist in current",
    )
    args = ap.parse_args()

    base = _read(args.baseline)
    cur = _read(args.current)
    bi = _index(base)
    ci = _index(cur)

    missing_in_current: List[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = []
    extra_in_current: List[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = []
    regressions = 0
    compared = 0
    rows: List[str] = []

    for k in sorted(bi.keys()):
        if k not in ci:
            missing_in_current.append(k)
            continue
        bs = bi[k]
        cs = ci[k]
        bm = bs.get("mean_ms")
        cm = cs.get("mean_ms")
        if not isinstance(bm, (int, float)) or not isinstance(cm, (int, float)):
            rows.append(f"skip {k[0]}/{k[1]}: non-numeric mean")
            continue
        if float(bm) <= 0:
            rows.append(f"skip {k[0]}/{k[1]}: baseline mean_ms <= 0")
            continue
        compared += 1
        ratio = float(cm) / float(bm)
        flag = " **REGRESSION**" if ratio > args.max_slowdown else ""
        if ratio > args.max_slowdown:
            regressions += 1
        rows.append(
            f"{k[0]:20} {k[1]:30} ratio={ratio:.4f} (baseline {bm:.4f} ms -> current {cm:.4f} ms){flag}"
        )

    for k in sorted(ci.keys()):
        if k not in bi:
            extra_in_current.append(k)

    print(f"baseline: {args.baseline}")
    print(f"current:  {args.current}")
    print(f"compared: {compared} samples with numeric mean_ms; regressions: {regressions}")
    print()
    for line in rows:
        print(line)

    if missing_in_current:
        print()
        print(f"Missing in current ({len(missing_in_current)}):")
        for k in missing_in_current[:50]:
            print(f"  {k[0]} {k[1]} {k[2]}")
        if len(missing_in_current) > 50:
            print(f"  ... and {len(missing_in_current) - 50} more")

    if extra_in_current:
        print()
        print(f"Extra in current (not in baseline) ({len(extra_in_current)}):")
        for k in extra_in_current[:20]:
            print(f"  {k[0]} {k[1]}")
        if len(extra_in_current) > 20:
            print(f"  ... and {len(extra_in_current) - 20} more")

    if args.require_matching_suites and missing_in_current:
        print("\nFAIL: --require-matching-suites and baseline keys missing in current", file=sys.stderr)
        return 1

    if args.fail_on_regression and regressions > 0:
        print(f"\nFAIL: {regressions} sample(s) slower than {args.max_slowdown}x baseline", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
