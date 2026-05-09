#!/usr/bin/env python3
"""
TensorFlow baseline workloads (schema v2.0.0): CPU and optional GPU.

Operator-only parity subset:
  - matmul
  - attention.{small,medium} (manual QK^T softmax V)
  - conv2d.{small,medium,large}

Usage:
  python3 benchmarks/tensorflow_baselines.py --repeats 5 --warmup 1
  python3 benchmarks/tensorflow_baselines.py --device gpu --repeats 5 --warmup 1

If tensorflow is not installed, exits 2 (optional suite semantics).
If --device gpu is requested but no GPU is available, exits 2.

Note: TF execution may be asynchronous on GPU; we call `tf.experimental.sync_devices()`
when available, otherwise fall back to a minimal `x.numpy()` barrier (last resort).
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
        import tensorflow as tf  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"TensorFlow baseline requires tensorflow: {e}", file=sys.stderr)
        sys.exit(2)

    if args.device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("TensorFlow GPU baseline skipped: no GPU devices", file=sys.stderr)
            sys.exit(2)
        suite = "tensorflow-gpu"
        backend = "tensorflow-gpu"
        device_str = "gpu:0"
        tf_device = "/GPU:0"
    else:
        suite = "tensorflow"
        backend = "tensorflow-cpu"
        device_str = "cpu"
        tf_device = "/CPU:0"

    def _sync(x=None) -> None:
        # Prefer an explicit sync if available.
        sync = getattr(tf.experimental, "sync_devices", None)
        if callable(sync):
            sync()
            return
        if x is not None:
            # Last resort: force a host-visible value.
            _ = x.numpy()

    samples: List[Dict] = []

    with tf.device(tf_device):
        # Matmul
        for m, k, n in [(128, 128, 128), (256, 256, 256), (512, 512, 512)]:
            a = tf.ones((m, k), dtype=tf.float32)
            b = tf.ones((k, n), dtype=tf.float32)

            @tf.function(jit_compile=False)
            def fn():
                return tf.matmul(a, b)

            _sync(fn())

            def step():
                _sync(fn())

            runs = _time_runs(step, args.warmup, args.repeats)
            samples.append(_make_sample("matmul", backend, device_str, {"m": str(m), "k": str(k), "n": str(n)}, runs))

        # Attention-style
        for tag, d_model, heads, seq_len in [("small", 64, 4, 32), ("medium", 256, 8, 128)]:
            q = tf.ones((1, seq_len, d_model), dtype=tf.float32)
            k_t = tf.ones((1, seq_len, d_model), dtype=tf.float32)
            v = tf.ones((1, seq_len, d_model), dtype=tf.float32)

            @tf.function(jit_compile=False)
            def attn():
                scores = tf.matmul(q, k_t, transpose_b=True)
                scaled = scores / (d_model**0.5)
                probs = tf.nn.softmax(scaled, axis=-1)
                return tf.matmul(probs, v)

            _sync(attn())

            def step():
                _sync(attn())

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

        # Conv2d (NCHW input for parity with other suites; TF uses NHWC by default)
        conv_configs = [
            ("small", (1, 28, 28, 1), (5, 5, 1, 6)),
            ("medium", (4, 32, 32, 16), (3, 3, 16, 16)),
            ("large", (8, 64, 64, 64), (3, 3, 64, 64)),
        ]
        for tag, x_shape, w_shape in conv_configs:
            x = tf.ones(x_shape, dtype=tf.float32)
            w = tf.ones(w_shape, dtype=tf.float32)

            @tf.function(jit_compile=False)
            def conv():
                return tf.nn.conv2d(x, w, strides=1, padding="VALID", data_format="NHWC")

            _sync(conv())

            def step():
                _sync(conv())

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
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "python": platform.python_version(),
    }
    if args.device == "gpu":
        try:
            machine["gpu_device"] = gpus[0].name  # type: ignore[name-defined]
        except Exception:
            pass

    out = {"suite": suite, "schema_version": SCHEMA_VERSION, "machine": machine, "samples": samples}
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

