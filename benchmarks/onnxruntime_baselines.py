#!/usr/bin/env python3
"""
ONNX Runtime baseline workloads (schema v2.0.0): CPU and optional CUDA provider.

This baseline measures inference execution of small ONNX graphs for:
  - matmul
  - attention.{small,medium} (QK^T softmax V)
  - conv2d.{small,medium,large}

Usage:
  python3 benchmarks/onnxruntime_baselines.py --repeats 5 --warmup 1
  python3 benchmarks/onnxruntime_baselines.py --device cuda --repeats 5 --warmup 1

Dependencies (optional): onnx, onnxruntime.
If missing, exits 2 so the orchestrator can skip the suite.
If --device cuda is requested but CUDAExecutionProvider isn't available, exits 2.
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

import numpy as np

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
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()

    try:
        import onnx  # type: ignore
        from onnx import helper, numpy_helper, TensorProto  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"ONNX Runtime baseline requires onnx: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"ONNX Runtime baseline requires onnxruntime: {e}", file=sys.stderr)
        sys.exit(2)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 0
    sess_opts.inter_op_num_threads = 0

    if args.device == "cuda":
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            print(f"ONNX Runtime CUDA baseline skipped: CUDAExecutionProvider not available ({providers})", file=sys.stderr)
            sys.exit(2)
        suite = "onnxruntime-cuda"
        backend = "onnxruntime-cuda"
        device_str = "cuda:0"
        provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        suite = "onnxruntime"
        backend = "onnxruntime-cpu"
        device_str = "cpu"
        provider_list = ["CPUExecutionProvider"]

    def _session(model) -> "ort.InferenceSession":
        b = model.SerializeToString()
        return ort.InferenceSession(b, sess_options=sess_opts, providers=provider_list)

    samples: List[Dict] = []

    # Matmul: Y = A x B
    for m, k, n in [(128, 128, 128), (256, 256, 256), (512, 512, 512)]:
        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
        b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])
        node = helper.make_node("MatMul", ["A", "B"], ["Y"])
        graph = helper.make_graph([node], "matmul", [a, b], [y])
        model = helper.make_model(graph, producer_name="rustral-ort")
        onnx.checker.check_model(model)
        sess = _session(model)
        A = np.ones((m, k), dtype=np.float32)
        B = np.ones((k, n), dtype=np.float32)

        def step():
            sess.run(None, {"A": A, "B": B})

        runs = _time_runs(step, args.warmup, args.repeats)
        samples.append(_make_sample("matmul", backend, device_str, {"m": str(m), "k": str(k), "n": str(n)}, runs))

    # Attention: scores = Q K^T / sqrt(d); probs = softmax(scores); out = probs V
    for tag, d_model, heads, seq_len in [("small", 64, 4, 32), ("medium", 256, 8, 128)]:
        q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, seq_len, d_model])
        k = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, seq_len, d_model])
        v = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, seq_len, d_model])
        out = helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, seq_len, d_model])

        kt = helper.make_node("Transpose", ["K"], ["KT"], perm=[0, 2, 1])
        mm1 = helper.make_node("MatMul", ["Q", "KT"], ["S"])
        scale = numpy_helper.from_array(np.array([1.0 / (d_model**0.5)], dtype=np.float32), name="scale")
        mul = helper.make_node("Mul", ["S", "scale"], ["SS"])
        sm = helper.make_node("Softmax", ["SS"], ["P"], axis=-1)
        mm2 = helper.make_node("MatMul", ["P", "V"], ["O"])
        graph = helper.make_graph([kt, mm1, mul, sm, mm2], "attn", [q, k, v], [out], initializer=[scale])
        model = helper.make_model(graph, producer_name="rustral-ort")
        onnx.checker.check_model(model)
        sess = _session(model)
        Q = np.ones((1, seq_len, d_model), dtype=np.float32)
        K = np.ones((1, seq_len, d_model), dtype=np.float32)
        V = np.ones((1, seq_len, d_model), dtype=np.float32)

        def step():
            sess.run(None, {"Q": Q, "K": K, "V": V})

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

    # Conv2d: use NCHW input for parity; weights OIHW.
    #
    # ONNX checker requires output type information. Rather than compute output shapes here,
    # we declare Y as a FLOAT tensor with unknown shape (no dims specified).
    conv_configs = [
        ("small", (1, 1, 28, 28), (6, 1, 5, 5)),
        ("medium", (4, 16, 32, 32), (16, 16, 3, 3)),
        ("large", (8, 64, 64, 64), (64, 64, 3, 3)),
    ]
    for tag, x_shape, w_shape in conv_configs:
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape))
        w = helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_shape))
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [])
        node = helper.make_node("Conv", ["X", "W"], ["Y"], strides=[1, 1], pads=[0, 0, 0, 0])
        graph = helper.make_graph([node], "conv", [x, w], [y])
        model = helper.make_model(graph, producer_name="rustral-ort")
        onnx.checker.check_model(model)
        sess = _session(model)
        X = np.ones(x_shape, dtype=np.float32)
        W = np.ones(w_shape, dtype=np.float32)

        def step():
            sess.run(None, {"X": X, "W": W})

        runs = _time_runs(step, args.warmup, args.repeats)
        samples.append(
            _make_sample(
                f"conv2d.{tag}",
                backend,
                device_str,
                {
                    "batch": str(x_shape[0]),
                    "in_channels": str(x_shape[1]),
                    "h": str(x_shape[2]),
                    "w": str(x_shape[3]),
                    "out_channels": str(w_shape[0]),
                    "kernel_h": str(w_shape[2]),
                    "kernel_w": str(w_shape[3]),
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
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
        "python": platform.python_version(),
    }
    if args.device == "cuda":
        machine["providers"] = ",".join(provider_list)

    out_doc = {"suite": suite, "schema_version": SCHEMA_VERSION, "machine": machine, "samples": samples}
    json.dump(out_doc, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

