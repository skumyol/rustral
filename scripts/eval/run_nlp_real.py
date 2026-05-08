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
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"

def _default_cache_dir() -> Path:
    env = os.environ.get("RUSTRAL_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    home = os.environ.get("HOME", "")
    if home:
        return (Path(home) / ".cache" / "rustral").resolve()
    return (REPO_ROOT / ".cache" / "rustral").resolve()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: List[str], env: Dict[str, str], cwd: Path) -> None:
    p = subprocess.run(cmd, env=env, cwd=str(cwd))
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}")

def _build_runtime_examples(env: Dict[str, str], cargo_features: str) -> None:
    cmd = [
        "cargo",
        "build",
        "--release",
        "-p",
        "rustral-runtime",
        "--features",
        cargo_features,
        "--examples",
    ]
    _run(cmd, env=env, cwd=REPO_ROOT)


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


def _example_env(cache_dir: Path | None, device: str) -> Dict[str, str]:
    env = dict(os.environ)
    # Force online mode: if user has these set globally, remove them for this script.
    env.pop("RUSTRAL_DATASET_OFFLINE", None)
    env.pop("RUSTRAL_DATASET_SKIP_CHECKSUM", None)
    if cache_dir is not None:
        env["RUSTRAL_CACHE_DIR"] = str(cache_dir)
    # Default to CPU for determinism, but allow CUDA runs explicitly.
    if device == "cpu":
        env.setdefault("CANDLE_FORCE_CPU", "1")
    return env


def _ensure_wikitext2_splits(cache_dir: Path) -> None:
    """
    Ensure cache_dir/datasets/wikitext-2/{train,valid,test}.txt exist.

    We do this in Python so local environments don't need the `unzip` CLI that the Rust
    fetch path shells out to.
    """
    out_dir = cache_dir / "datasets" / "wikitext-2"
    train_txt = out_dir / "train.txt"
    valid_txt = out_dir / "valid.txt"
    test_txt = out_dir / "test.txt"
    if train_txt.exists() and valid_txt.exists() and test_txt.exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "wikitext-2-raw-v1.zip"
    url = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
    if not zip_path.exists():
        print(f"[wikitext2] downloading {url} -> {zip_path}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:  # nosec - pinned public dataset URL
            zip_path.write_bytes(resp.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Mirror matches Rust loader expectations.
    candidates = [out_dir / "wikitext-2-raw", out_dir / "wikitext-2-raw-v1"]
    base = None
    for c in candidates:
        if (c / "wiki.train.raw").exists():
            base = c
            break
    if base is None:
        raise RuntimeError("wikitext2 zip did not contain expected wiki.train.raw")

    (out_dir / "train.txt").write_text((base / "wiki.train.raw").read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "valid.txt").write_text((base / "wiki.valid.raw").read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "test.txt").write_text((base / "wiki.test.raw").read_text(encoding="utf-8"), encoding="utf-8")


def _run_sst2(
    seed: int,
    out_dir: Path,
    cache_dir: Path | None,
    device: str,
    extra_args: Optional[List[str]] = None,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _example_env(cache_dir, device)
    exe = REPO_ROOT / "target" / "release" / "examples" / "sst2_classifier"
    cmd = [
        str(exe),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    _run(cmd, env=env, cwd=REPO_ROOT)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"sst2 run did not produce {manifest_path}")
    return RunResult(seed=seed, out_dir=out_dir, manifest=_read_json(manifest_path))


def _run_wikitext2(
    seed: int,
    out_dir: Path,
    cache_dir: Path | None,
    device: str,
    train_tokens: int,
    train_windows: int,
    eval_windows: int,
    extra_args: Optional[List[str]] = None,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always run wikitext2 backed by a local extracted zip so environments without
    # the `unzip` CLI can still run this benchmark.
    cache_dir_resolved = cache_dir if cache_dir is not None else _default_cache_dir()
    env = _example_env(cache_dir_resolved, device)
    _ensure_wikitext2_splits(cache_dir_resolved)
    env["RUSTRAL_DATASET_OFFLINE"] = "1"
    env["RUSTRAL_DATASET_SKIP_CHECKSUM"] = "1"
    exe = REPO_ROOT / "target" / "release" / "examples" / "wikitext2_lm"
    cmd = [
        str(exe),
        "--seed",
        str(seed),
        "--train-tokens",
        str(train_tokens),
        "--train-windows",
        str(train_windows),
        "--eval-windows",
        str(eval_windows),
        "--out-dir",
        str(out_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
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
        "--skip-sst2",
        action="store_true",
        help="skip SST-2 runs/curation (useful if dependencies are missing locally)",
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="device for Rust examples (default: cpu)")
    ap.add_argument(
        "--skip-wikitext2",
        action="store_true",
        help="skip WikiText-2 runs/curation (useful if dependencies are missing locally)",
    )
    ap.add_argument(
        "--cache-dir",
        default="",
        help="optional RUSTRAL_CACHE_DIR override (default: unset)",
    )
    ap.add_argument(
        "--wikitext-train-tokens",
        type=int,
        default=8_000,
        help="train token cap for wikitext2_lm (0 = no cap; default: 8000)",
    )
    ap.add_argument(
        "--wikitext-train-windows",
        type=int,
        default=400,
        help="cap number of training windows for wikitext2_lm (0 = all, default: 400)",
    )
    ap.add_argument(
        "--wikitext-eval-windows",
        type=int,
        default=800,
        help="cap number of validation windows for perplexity (0 = all, default: 800)",
    )
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="fast preset: tiny transformer + minimal data (for CI / local iteration)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="delete --out-root before running",
    )
    args = ap.parse_args()

    sst2_extra: Optional[List[str]] = None
    wiki_extra: Optional[List[str]] = None
    if args.benchmark:
        # Small model + small data; finishes in minutes on CPU.
        args.wikitext_train_tokens = 4_000
        args.wikitext_train_windows = 150
        args.wikitext_eval_windows = 300
        sst2_extra = [
            "--quick",
            "--num-layers",
            "1",
            "--epochs",
            "1",
            "--seq-len",
            "16",
            "--d-model",
            "32",
            "--num-heads",
            "2",
            "--ffn-dim",
            "64",
        ]
        wiki_extra = [
            "--quick",
            "--block-size",
            "16",
            "--d-model",
            "32",
            "--num-heads",
            "2",
            "--ffn-dim",
            "64",
            "--num-layers",
            "1",
        ]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("no seeds provided")

    out_root = (REPO_ROOT / args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    env = _example_env(cache_dir, args.device)
    cargo_features = "training,cuda" if args.device == "cuda" else "training"
    _build_runtime_examples(env, cargo_features)

    print("Real-data NLP runs (rustral)")
    print("===========================")
    print(f"repo_root            : {REPO_ROOT}")
    print(f"out_root             : {out_root}")
    print(f"cache_dir            : {cache_dir if cache_dir else '(default)'}")
    print(f"seeds                : {seeds}")
    print(f"benchmark            : {args.benchmark}")
    print(f"device               : {args.device}")
    print(f"wikitext_train_tokens: {args.wikitext_train_tokens}")
    print(f"wikitext_train_windows: {args.wikitext_train_windows}")
    print(f"wikitext_eval_windows: {args.wikitext_eval_windows}")
    print()

    sst2_runs: List[RunResult] = []
    wikitext_runs: List[RunResult] = []

    for seed in seeds:
        if not args.skip_sst2:
            print(f"[sst2] seed={seed}")
            sst2_runs.append(
                _run_sst2(seed, out_root / "sst2" / f"seed_{seed}", cache_dir, args.device, sst2_extra)
            )
        if not args.skip_wikitext2:
            print(f"[wikitext2] seed={seed}")
            wikitext_runs.append(
                _run_wikitext2(
                    seed,
                    out_root / "wikitext2" / f"seed_{seed}",
                    cache_dir,
                    args.device,
                    train_tokens=args.wikitext_train_tokens,
                    train_windows=args.wikitext_train_windows,
                    eval_windows=args.wikitext_eval_windows,
                    extra_args=wiki_extra,
                )
            )

    curated_dir = REPO_ROOT / "benchmarks" / "runs" / f"v{VERSION}" / "nlp"
    if not args.skip_sst2:
        _curate_sst2(sst2_runs, curated_dir / "sst2.json")
    if not args.skip_wikitext2:
        _curate_wikitext2(wikitext_runs, curated_dir / "wikitext2.json")

    print()
    print("Wrote curated manifests:")
    if not args.skip_sst2:
        print(f"  {curated_dir / 'sst2.json'}")
    if not args.skip_wikitext2:
        print(f"  {curated_dir / 'wikitext2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

