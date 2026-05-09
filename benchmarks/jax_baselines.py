#!/usr/bin/env python3
"""
JAX baseline workloads (schema v2.0.0): CPU and optional GPU.

Mirrors the operator-only parity subset of the unified harness:
  - matmul
  - attention.{small,medium} (manual QK^T softmax V)
  - conv2d.{small,medium,large}

Usage (typically invoked by scripts/bench/run_all.py):
  python3 benchmarks/jax_baselines.py --repeats 5 --warmup 1
  python3 benchmarks/jax_baselines.py --device gpu --repeats 5 --warmup 1

If JAX is not installed, exits 2 so callers can treat it as a skipped optional suite.
If --device gpu is requested but no GPU backend is available, exits 2 as well.
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
from pathlib import Path
from typing import Callable, Dict, List

_BENCH_ROOT = Path(__file__).resolve().parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
import stats_harness  # noqa: E402

SCHEMA_VERSION = "2.0.0"
DTYPE = "f32"


def _time_runs(fn: Callable[[], None], warmup: int, repeats: int) -> List[float]:
    for _ in range(warmup):
        fn()
    out: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _stats(runs: List[float]) -> Dict[str, float]:
    if not runs:
        return {"mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0}
    mean = statistics.mean(runs)
    std = statistics.stdev(runs) if len(runs) > 1 else 0.0
    return {
        "mean_ms": float(mean),
        "std_ms": float(std),
        "min_ms": float(min(runs)),
        "max_ms": float(max(runs)),
        "p50_ms": float(statistics.median(runs)),
    }


def _make_sample(name: str, backend: str, device: str, params: Dict[str, str], runs: List[float]) -> Dict:
    d: Dict = {
        "name": name,
        "backend": backend,
        "device": device,
        "dtype": DTYPE,
        "model_params": None,
        "params": params,
        "runs_ms": runs,
        **_stats(runs),
    }
    stats_harness.enrich_sample_stats(d)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = ap.parse_args()

    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"JAX baseline requires jax: {e}", file=sys.stderr)
        sys.exit(2)

    # Select device
    if args.device == "gpu":
        devs = [d for d in jax.devices() if d.platform == "gpu"]
        if not devs:
            print("JAX GPU baseline skipped: no GPU devices", file=sys.stderr)
            sys.exit(2)
        device = devs[0]
        suite = "jax-gpu"
        backend = "jax-gpu"
        device_str = "gpu:0"
    else:
        device = jax.devices("cpu")[0]
        suite = "jax"
        backend = "jax-cpu"
        device_str = "cpu"

    def _block(x):
        return x.block_until_ready()

    samples: List[Dict] = []

    # Matmul
    for m, k, n in [(128, 128, 128), (256, 256, 256), (512, 512, 512)]:
        a = jax.device_put(jnp.ones((m, k), dtype=jnp.float32), device)
        b = jax.device_put(jnp.ones((k, n), dtype=jnp.float32), device)
        fn = jax.jit(lambda x, y: x @ y)
        _block(fn(a, b))

        def step():
            _block(fn(a, b))

        runs = _time_runs(step, args.warmup, args.repeats)
        samples.append(_make_sample("matmul", backend, device_str, {"m": str(m), "k": str(k), "n": str(n)}, runs))

    # Attention-style
    for tag, d_model, heads, seq_len in [("small", 64, 4, 32), ("medium", 256, 8, 128)]:
        q = jax.device_put(jnp.ones((1, seq_len, d_model), dtype=jnp.float32), device)
        k_t = jax.device_put(jnp.ones((1, seq_len, d_model), dtype=jnp.float32), device)
        v = jax.device_put(jnp.ones((1, seq_len, d_model), dtype=jnp.float32), device)

        def attn(q, k_t, v):
            scores = q @ jnp.swapaxes(k_t, 1, 2)
            scaled = scores / (d_model**0.5)
            probs = jax.nn.softmax(scaled, axis=-1)
            return probs @ v

        fn = jax.jit(attn)
        _block(fn(q, k_t, v))

        def step():
            _block(fn(q, k_t, v))

        runs = _time_runs(step, args.warmup, args.repeats)
        samples.append(
            _make_sample(
                f"attention.{tag}",
                backend,
                device_str,
                {"d_model": str(d_model), "heads": str(heads), "seq_len": str(seq_len)},
                runs,
            )
        )

    # Conv2d (NHWC for JAX default; weights HWIO)
    conv_configs = [
        ("small", (1, 28, 28, 1), (5, 5, 1, 6)),
        ("medium", (4, 32, 32, 16), (3, 3, 16, 16)),
        ("large", (8, 64, 64, 64), (3, 3, 64, 64)),
    ]
    for tag, x_shape, w_shape in conv_configs:
        x = jax.device_put(jnp.ones(x_shape, dtype=jnp.float32), device)
        w = jax.device_put(jnp.ones(w_shape, dtype=jnp.float32), device)

        def conv(x, w):
            return jax.lax.conv_general_dilated(
                x,
                w,
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

        fn = jax.jit(conv)
        _block(fn(x, w))

        def step():
            _block(fn(x, w))

        runs = _time_runs(step, args.warmup, args.repeats)
        samples.append(
            _make_sample(
                f"conv2d.{tag}",
                backend,
                device_str,
                {
                    "batch": str(x_shape[0]),
                    "h": str(x_shape[1]),
                    "w": str(x_shape[2]),
                    "in_channels": str(x_shape[3]),
                    "kernel_h": str(w_shape[0]),
                    "kernel_w": str(w_shape[1]),
                    "out_channels": str(w_shape[3]),
                },
                runs,
            )
        )

    machine: Dict[str, object] = {
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "hostname": socket.gethostname(),
        "rustc": "n/a",
        "commit": os.environ.get("GIT_SHA", "unknown"),
        "features": [],
        "jax_version": getattr(jax, "__version__", "unknown"),
        "python": platform.python_version(),
    }
    if args.device == "gpu":
        machine["gpu_device"] = str(device)

    out = {"suite": suite, "schema_version": SCHEMA_VERSION, "machine": machine, "samples": samples}
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

