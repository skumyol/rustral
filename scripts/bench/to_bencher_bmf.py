#!/usr/bin/env python3
"""
Convert a Rustral schema-v2 combined harness JSON into Bencher Metric Format (BMF).

Bencher.dev's `bencher run --adapter json` accepts a flat object keyed by benchmark name
where each value carries one or more metric kinds. We keep it simple: every Rustral sample
becomes a BMF entry under `<suite>/<backend>/<device>/<name>` whose `latency` metric is
the per-run mean wall-time in nanoseconds, with `latency.lower_value` and
`latency.upper_value` set from `min_ms` and `max_ms`.

This mapping is intentionally shallow — BMF supports throughput / count metrics too, but
the canonical Rustral measurement is wall time, so the BMF view stays a pure latency
dashboard.

Usage:
    python3 scripts/bench/to_bencher_bmf.py <input_combined.json> <output_bmf.json>
    cat <input>.json | python3 scripts/bench/to_bencher_bmf.py - <output>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def ms_to_ns(ms: float) -> float:
    return float(ms) * 1.0e6


def to_bmf(combined: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    suites = combined.get("suites", []) if isinstance(combined, dict) else []
    if not isinstance(suites, list):
        return out
    for suite in suites:
        suite_name = suite.get("suite", "?")
        for sample in suite.get("samples", []):
            name = sample.get("name", "?")
            backend = sample.get("backend", "?")
            device = sample.get("device", "?")
            label = f"{suite_name}/{backend}/{device}/{name}"
            mean_ns = ms_to_ns(sample.get("mean_ms", 0.0))
            min_ns = ms_to_ns(sample.get("min_ms", 0.0))
            max_ns = ms_to_ns(sample.get("max_ms", 0.0))
            entry: Dict[str, Any] = {
                "latency": {
                    "value": mean_ns,
                    "lower_value": min_ns,
                    "upper_value": max_ns,
                }
            }
            # Surface model_params as an integer metric when present so dashboards can
            # filter by "training-step" vs "operator" workloads.
            model_params = sample.get("model_params")
            if isinstance(model_params, (int, float)) and model_params:
                entry["model_params"] = {"value": float(model_params)}
            out[label] = entry
    return out


def main(argv):
    if len(argv) != 3:
        print("usage: to_bencher_bmf.py <input_combined.json> <output_bmf.json>", file=sys.stderr)
        return 2
    src, dst = argv[1], argv[2]
    if src == "-":
        combined = json.load(sys.stdin)
    else:
        combined = json.loads(Path(src).read_text())
    bmf = to_bmf(combined)
    Path(dst).write_text(json.dumps(bmf, indent=2) + "\n")
    print(f"wrote {dst} ({len(bmf)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
