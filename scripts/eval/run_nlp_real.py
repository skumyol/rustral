#!/usr/bin/env python3
"""
Run real-data NLP evals (3 seeds) and curate manifests.

Writes:
  benchmarks/runs/v0.1.0/nlp/sst2.json
  benchmarks/runs/v0.1.0/nlp/wikitext2.json

Each curated JSON embeds the raw per-seed manifests (as emitted by the Rust examples)
and adds mean/std aggregates for the headline metric.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: List[str], env: Dict[str, str], cwd: Path) -> None:
    p = subprocess.run(cmd, env=env, cwd=str(cwd))
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (float(statistics.mean(xs)), float(statistics.stdev(xs)))


@dataclass(frozen=True)
class RunResult:
    seed: int
    out_dir: Path
    manifest: Dict[str, Any]


def _example_env(cache_dir: Path | None) -> Dict[str, str]:
    env = dict(os.environ)
    # Force online mode: if user has these set globally, remove them for this script.
    env.pop("RUSTRAL_DATASET_OFFLINE", None)
    env.pop("RUSTRAL_DATASET_SKIP_CHECKSUM", None)
    if cache_dir is not None:
        env["RUSTRAL_CACHE_DIR"] = str(cache_dir)
    # Make output deterministic and avoid surprise GPU behavior.
    env.setdefault("CANDLE_FORCE_CPU", "1")
    return env


def _run_sst2(seed: int, out_dir: Path, cache_dir: Path | None) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _example_env(cache_dir)
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "rustral-runtime",
        "--features",
        "training",
        "--example",
        "sst2_classifier",
        "--",
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    _run(cmd, env=env, cwd=REPO_ROOT)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"sst2 run did not produce {manifest_path}")
    return RunResult(seed=seed, out_dir=out_dir, manifest=_read_json(manifest_path))


def _run_wikitext2(seed: int, out_dir: Path, cache_dir: Path | None, train_tokens: int) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _example_env(cache_dir)
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "rustral-runtime",
        "--features",
        "training",
        "--example",
        "wikitext2_lm",
        "--",
        "--seed",
        str(seed),
        "--train-tokens",
        str(train_tokens),
        "--out-dir",
        str(out_dir),
    ]
    _run(cmd, env=env, cwd=REPO_ROOT)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"wikitext2 run did not produce {manifest_path}")
    return RunResult(seed=seed, out_dir=out_dir, manifest=_read_json(manifest_path))


def _curate_sst2(runs: List[RunResult], out_path: Path) -> None:
    accs = [float(r.manifest["dev_accuracy"]) for r in runs]
    mean, std = _mean_std(accs)
    obj: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": _now_iso(),
        "version": VERSION,
        "task": "sst2_classifier",
        "metric": "dev_accuracy",
        "aggregate": {"mean": mean, "std": std, "n": len(accs)},
        "runs": [
            {
                "seed": r.seed,
                "out_dir": str(r.out_dir.relative_to(REPO_ROOT)),
                "manifest": r.manifest,
            }
            for r in runs
        ],
    }
    _write_json(out_path, obj)


def _curate_wikitext2(runs: List[RunResult], out_path: Path) -> None:
    ppls = [float(r.manifest["dev_perplexity"]) for r in runs]
    mean, std = _mean_std(ppls)
    obj: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": _now_iso(),
        "version": VERSION,
        "task": "wikitext2_word_lm",
        "metric": "dev_perplexity",
        "aggregate": {"mean": mean, "std": std, "n": len(ppls)},
        "runs": [
            {
                "seed": r.seed,
                "out_dir": str(r.out_dir.relative_to(REPO_ROOT)),
                "manifest": r.manifest,
            }
            for r in runs
        ],
    }
    _write_json(out_path, obj)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated list (default: 0,1,2)")
    ap.add_argument(
        "--out-root",
        default="out/nlp_real",
        help="where to store per-seed run artifacts (default: out/nlp_real)",
    )
    ap.add_argument(
        "--cache-dir",
        default="",
        help="optional RUSTRAL_CACHE_DIR override (default: unset)",
    )
    ap.add_argument(
        "--wikitext-train-tokens",
        type=int,
        default=50_000,
        help="train token cap for wikitext2_lm (default: 50000)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete --out-root before running",
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("no seeds provided")

    out_root = (REPO_ROOT / args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None

    print("Real-data NLP runs (rustral)")
    print("===========================")
    print(f"repo_root            : {REPO_ROOT}")
    print(f"out_root             : {out_root}")
    print(f"cache_dir            : {cache_dir if cache_dir else '(default)'}")
    print(f"seeds                : {seeds}")
    print(f"wikitext_train_tokens: {args.wikitext_train_tokens}")
    print()

    sst2_runs: List[RunResult] = []
    wikitext_runs: List[RunResult] = []

    for seed in seeds:
        print(f"[sst2] seed={seed}")
        sst2_runs.append(_run_sst2(seed, out_root / "sst2" / f"seed_{seed}", cache_dir))
        print(f"[wikitext2] seed={seed}")
        wikitext_runs.append(
            _run_wikitext2(
                seed,
                out_root / "wikitext2" / f"seed_{seed}",
                cache_dir,
                train_tokens=args.wikitext_train_tokens,
            )
        )

    curated_dir = REPO_ROOT / "benchmarks" / "runs" / f"v{VERSION}" / "nlp"
    _curate_sst2(sst2_runs, curated_dir / "sst2.json")
    _curate_wikitext2(wikitext_runs, curated_dir / "wikitext2.json")

    print()
    print("Wrote curated manifests:")
    print(f"  {curated_dir / 'sst2.json'}")
    print(f"  {curated_dir / 'wikitext2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

