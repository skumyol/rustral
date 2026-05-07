#!/usr/bin/env python3
"""
PyTorch CPU baseline workloads (schema v2.0.0).

Mirrors the workload set used by the Rust harness binaries (`rustral_workloads`,
`candle_workloads`) so that PyTorch numbers can be joined into the same JSON
schema documented in `benchmarks/SCHEMA.md`.

Output: a single JSON document on stdout with `suite = "pytorch"`.

Coverage parity:
  - matmul: yes
  - attention: yes (manual QK^T softmax V)
  - conv2d:  yes (`torch.nn.functional.conv2d`)
  - lstm_forward / mlp_train_step / optimizer_step: skipped here. The PyTorch
    baseline focuses on the operator surface that all three suites can compare
    apples-to-apples; training-step parity for PyTorch lives in P3.

Usage (typically invoked by `scripts/bench/run_all.py`):

    python3 benchmarks/pytorch/baselines.py --repeats 5 --warmup 1

Requires `torch` to be importable in the active environment. If torch is not
installed this script exits with a non-zero status (2) and a clear error message
so the orchestrator can record it as a skipped suite.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import sys
import time
from typing import Callable, Dict, List

SCHEMA_VERSION = "2.0.0"
BACKEND = "pytorch-cpu"
DEVICE = "cpu"
DTYPE = "f32"


def time_runs(fn: Callable[[], None], warmup: int, repeats: int) -> List[float]:
    for _ in range(warmup):
        fn()
    out: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def stats(runs: List[float]) -> Dict[str, float]:
    if not runs:
        return {"mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0}
    mean = statistics.mean(runs)
    std = statistics.stdev(runs) if len(runs) > 1 else 0.0
    return {
        "mean_ms": mean,
        "std_ms": std,
        "min_ms": min(runs),
        "max_ms": max(runs),
        "p50_ms": statistics.median(runs),
    }


def make_sample(
    name: str,
    params: Dict[str, str],
    runs: List[float],
    *,
    model_params: int | None = None,
) -> Dict:
    return {
        "name": name,
        "backend": BACKEND,
        "device": DEVICE,
        "dtype": DTYPE,
        "model_params": model_params,
        "params": params,
        "runs_ms": runs,
        **stats(runs),
    }


def detect_features() -> List[str]:
    raw = os.environ.get("RUSTRAL_BENCH_FEATURES", "")
    return [f.strip() for f in raw.split(",") if f.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"PyTorch baseline requires torch: {e}", file=sys.stderr)
        sys.exit(2)

    torch.set_num_threads(max(1, torch.get_num_threads()))

    samples: List[Dict] = []

    # Matmul workloads
    for m, k, n in [(128, 128, 128), (256, 256, 256), (512, 512, 512)]:
        a = torch.ones((m, k), dtype=torch.float32)
        b = torch.ones((k, n), dtype=torch.float32)
        runs = time_runs(lambda: a @ b, args.warmup, args.repeats)
        samples.append(make_sample("matmul", {"m": str(m), "k": str(k), "n": str(n)}, runs))

    # Attention-style workloads
    for tag, d_model, heads, seq_len in [("small", 64, 4, 32), ("medium", 256, 8, 128)]:
        q = torch.ones((1, seq_len, d_model), dtype=torch.float32)
        k = torch.ones((1, seq_len, d_model), dtype=torch.float32)
        v = torch.ones((1, seq_len, d_model), dtype=torch.float32)

        def step():
            scores = q @ k.transpose(1, 2)
            scaled = scores / (d_model ** 0.5)
            probs = torch.softmax(scaled, dim=-1)
            return probs @ v

        runs = time_runs(step, args.warmup, args.repeats)
        samples.append(
            make_sample(
                f"attention.{tag}",
                {"d_model": str(d_model), "heads": str(heads), "seq_len": str(seq_len)},
                runs,
            )
        )

    # Conv2d workloads (parity with rustral_workloads)
    conv_configs = [
        ("small", (1, 1, 28, 28), (6, 1, 5, 5)),
        ("medium", (4, 16, 32, 32), (16, 16, 3, 3)),
        ("large", (8, 64, 64, 64), (64, 64, 3, 3)),
    ]
    for tag, input_shape, filter_shape in conv_configs:
        # Note: PyTorch conv2d expects [out, in/groups, kH, kW]; Rustral uses [out, in, kH, kW].
        # When in_channels matches in/groups (groups=1 default) the shapes coincide.
        x = torch.ones(input_shape, dtype=torch.float32)
        w = torch.ones(filter_shape, dtype=torch.float32)
        runs = time_runs(lambda: F.conv2d(x, w), args.warmup, args.repeats)
        samples.append(
            make_sample(
                f"conv2d.{tag}",
                {
                    "batch": str(input_shape[0]),
                    "in_channels": str(input_shape[1]),
                    "h": str(input_shape[2]),
                    "w": str(input_shape[3]),
                    "out_channels": str(filter_shape[0]),
                    "kernel_h": str(filter_shape[2]),
                    "kernel_w": str(filter_shape[3]),
                },
                runs,
            )
        )

    out = {
        "suite": "pytorch",
        "schema_version": SCHEMA_VERSION,
        "machine": {
            "os": platform.system().lower(),
            "arch": platform.machine().lower(),
            "hostname": socket.gethostname(),
            "rustc": "n/a",
            "commit": os.environ.get("GIT_SHA", "unknown"),
            "features": detect_features(),
            "torch_version": torch.__version__,
            "python": platform.python_version(),
        },
        "samples": samples,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
