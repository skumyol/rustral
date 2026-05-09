#!/usr/bin/env python3
"""
Unified benchmark harness for Rustral.

Runs one or more workload "suites" with a controlled number of repeats, captures the
shared JSON schema documented in benchmarks/SCHEMA.md, then writes:

    benchmarks/results/<timestamp>.json   (combined raw results)
    benchmarks/results/summary.md         (aggregated Markdown summary)

Suites:
    rustral       -> Rustral via the ndarray-cpu backend (always available)
    candle        -> Candle-direct (Rust, always available)
    pytorch       -> PyTorch CPU (requires `torch` in the active Python environment)
    pytorch-cuda  -> PyTorch CUDA (requires `torch` with GPU; skipped if unavailable)
    rustral-cuda  -> Rustral CUDA backend (requires CUDA toolchain)
    rustral-metal -> Rustral Metal backend (Apple GPUs)
    jax           -> JAX CPU (requires `jax`)
    jax-gpu       -> JAX GPU (requires `jax` + GPU backend)
    tensorflow    -> TensorFlow CPU (requires `tensorflow`)
    tensorflow-gpu-> TensorFlow GPU (requires `tensorflow` + GPU)
    onnxruntime   -> ONNX Runtime CPU (requires `onnx` + `onnxruntime`)
    onnxruntime-cuda -> ONNX Runtime CUDA EP (requires `onnx` + `onnxruntime-gpu`)

Usage:
    python3 scripts/bench/run_all.py
    python3 scripts/bench/run_all.py --repeats 10
    python3 scripts/bench/run_all.py --suite rustral --suite candle
    python3 scripts/bench/run_all.py --suite pytorch
    python3 scripts/bench/run_all.py --suite pytorch --suite pytorch-cuda
    python3 scripts/bench/run_all.py --suite jax --suite tensorflow --suite onnxruntime
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> str:
    """Run a command and return stdout as text. Stderr is forwarded for visibility."""
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return result.stdout


def build_rust_bin(bin_name: str) -> Path:
    """Build a release binary in the bench crate and return its path."""
    print(f"[harness] building release binary: {bin_name}", flush=True)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "rustral-bench", "--bin", bin_name],
        cwd=REPO_ROOT,
        check=True,
    )
    return REPO_ROOT / "target" / "release" / bin_name


def run_suite_rustral(repeats: int, warmup: int) -> dict:
    bin_path = build_rust_bin("rustral_workloads")
    out = run_cmd([str(bin_path), "--repeats", str(repeats), "--warmup", str(warmup)])
    return json.loads(out)


def run_suite_candle(repeats: int, warmup: int) -> dict:
    bin_path = build_rust_bin("candle_workloads")
    out = run_cmd([str(bin_path), "--repeats", str(repeats), "--warmup", str(warmup)])
    return json.loads(out)


def run_suite_pytorch(repeats: int, warmup: int) -> dict:
    return _run_suite_pytorch_device(repeats, warmup, "cpu")


def run_suite_pytorch_cuda(repeats: int, warmup: int) -> dict:
    return _run_suite_pytorch_device(repeats, warmup, "cuda")


def _run_suite_pytorch_device(repeats: int, warmup: int, device: str) -> dict:
    script = REPO_ROOT / "benchmarks" / "pytorch" / "baselines.py"
    if not script.exists():
        raise FileNotFoundError(f"pytorch baseline script missing: {script}")
    out = run_cmd(
        [
            sys.executable,
            str(script),
            "--repeats",
            str(repeats),
            "--warmup",
            str(warmup),
            "--device",
            device,
        ],
    )
    return json.loads(out)


def _build_with_features(bin_name: str, features: List[str]) -> Path:
    """Build a release binary with the listed cargo features, returning its path."""
    feat = ",".join(features)
    print(f"[harness] building release binary: {bin_name} (--features {feat})", flush=True)
    args = [
        "cargo",
        "build",
        "--release",
        "-p",
        "rustral-bench",
        "--features",
        feat,
        "--bin",
        bin_name,
    ]
    subprocess.run(args, cwd=REPO_ROOT, check=True)
    return REPO_ROOT / "target" / "release" / bin_name


def run_suite_rustral_cuda(repeats: int, warmup: int) -> dict:
    bin_path = _build_with_features("rustral_workloads_cuda", ["cuda"])
    out = run_cmd([str(bin_path), "--repeats", str(repeats), "--warmup", str(warmup)])
    return json.loads(out)


def run_suite_rustral_metal(repeats: int, warmup: int) -> dict:
    bin_path = _build_with_features("rustral_workloads_metal", ["metal"])
    out = run_cmd([str(bin_path), "--repeats", str(repeats), "--warmup", str(warmup)])
    return json.loads(out)


def _run_py_baseline(script_rel: str, repeats: int, warmup: int, *extra: str) -> dict:
    script = REPO_ROOT / script_rel
    if not script.exists():
        raise FileNotFoundError(f"baseline script missing: {script}")
    out = run_cmd([sys.executable, str(script), "--repeats", str(repeats), "--warmup", str(warmup), *extra])
    return json.loads(out)


def run_suite_jax(repeats: int, warmup: int) -> dict:
    return _run_py_baseline("benchmarks/jax_baselines.py", repeats, warmup, "--device", "cpu")


def run_suite_jax_gpu(repeats: int, warmup: int) -> dict:
    return _run_py_baseline("benchmarks/jax_baselines.py", repeats, warmup, "--device", "gpu")


def run_suite_tensorflow(repeats: int, warmup: int) -> dict:
    return _run_py_baseline("benchmarks/tensorflow_baselines.py", repeats, warmup, "--device", "cpu")


def run_suite_tensorflow_gpu(repeats: int, warmup: int) -> dict:
    return _run_py_baseline("benchmarks/tensorflow_baselines.py", repeats, warmup, "--device", "gpu")


def run_suite_onnxruntime(repeats: int, warmup: int) -> dict:
    return _run_py_baseline("benchmarks/onnxruntime_baselines.py", repeats, warmup, "--device", "cpu")


def run_suite_onnxruntime_cuda(repeats: int, warmup: int) -> dict:
    return _run_py_baseline("benchmarks/onnxruntime_baselines.py", repeats, warmup, "--device", "cuda")


SUITES = {
    "rustral": run_suite_rustral,
    "candle": run_suite_candle,
    "pytorch": run_suite_pytorch,
    "pytorch-cuda": run_suite_pytorch_cuda,
    "rustral-cuda": run_suite_rustral_cuda,
    "rustral-metal": run_suite_rustral_metal,
    "jax": run_suite_jax,
    "jax-gpu": run_suite_jax_gpu,
    "tensorflow": run_suite_tensorflow,
    "tensorflow-gpu": run_suite_tensorflow_gpu,
    "onnxruntime": run_suite_onnxruntime,
    "onnxruntime-cuda": run_suite_onnxruntime_cuda,
}


def write_summary(combined: dict, summary_path: Path, source_json: Path) -> None:
    """Render a Markdown summary with one row per (workload, suite/backend)."""
    lines: List[str] = []
    lines.append("# Benchmark summary\n")
    lines.append(
        f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S %Z')} from "
        f"`{source_json.parent.name}/{source_json.name}`.\n"
    )
    lines.append("")
    lines.append("| Workload | Suite | Backend | Mean (ms) | Std (ms) | p50 (ms) | Min (ms) | Max (ms) | Runs | Params |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")

    rows: List[tuple] = []
    for suite_block in combined["suites"]:
        suite_name = suite_block.get("suite", "?")
        for s in suite_block.get("samples", []):
            rows.append(
                (
                    s.get("name", "?"),
                    suite_name,
                    s.get("backend", "?"),
                    s.get("mean_ms", 0.0),
                    s.get("std_ms", 0.0),
                    s.get("p50_ms", 0.0),
                    s.get("min_ms", 0.0),
                    s.get("max_ms", 0.0),
                    len(s.get("runs_ms", [])),
                    "; ".join(f"{k}={v}" for k, v in s.get("params", {}).items()),
                )
            )

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    for name, suite, backend, mean, std, p50, mn, mx, runs, params in rows:
        lines.append(
            f"| {name} | {suite} | {backend} | {mean:.3f} | {std:.3f} | {p50:.3f} | {mn:.3f} | {mx:.3f} | {runs} | {params} |"
        )

    summary_path.write_text("\n".join(lines) + "\n")


def detect_git_sha() -> str:
    if not shutil.which("git"):
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per workload (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per workload (default: 1)")
    parser.add_argument(
        "--suite",
        action="append",
        choices=list(SUITES.keys()),
        help="Suite(s) to run; repeatable. Defaults to rustral + candle.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path; default: benchmarks/results/<timestamp>.json",
    )
    args = parser.parse_args()

    suites = args.suite or ["rustral", "candle"]
    print(f"[harness] suites = {', '.join(suites)}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    combined: dict = {
        "tool": "rustral-bench-harness",
        "git_sha": detect_git_sha(),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suites": [],
    }

    for s in suites:
        runner = SUITES[s]
        print(f"[harness] running suite: {s}", flush=True)
        try:
            combined["suites"].append(runner(args.repeats, args.warmup))
        except FileNotFoundError as e:
            print(f"[harness] skipping suite {s}: {e}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            # Exit code 2 is reserved for "missing optional dependency" (e.g. torch not installed);
            # report it as a skipped suite so callers can still get partial results.
            if e.returncode == 2:
                print(
                    f"[harness] skipping suite {s} (missing optional dependency): {e.stderr.strip()}",
                    file=sys.stderr,
                )
                continue
            print(f"[harness] suite {s} failed: {e}", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(1)

    out_path = (args.out or RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2) + "\n")
    try:
        rel = out_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = out_path
    print(f"[harness] wrote {rel}", flush=True)

    summary_path = RESULTS_DIR / "summary.md"
    write_summary(combined, summary_path, out_path)
    try:
        rel = summary_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = summary_path
    print(f"[harness] wrote {rel}", flush=True)


if __name__ == "__main__":
    main()
