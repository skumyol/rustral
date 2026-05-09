# Architecture

Rustral is a Rust deep-learning toolkit organized as a workspace of focused crates. This document is a discoverability landing page; the deeper architectural sketch lives in [`docs/architecture.md`](docs/architecture.md), and the canonical roadmap and invariants live in [`docs/master-plan.md`](docs/master-plan.md).

## Top-level layering

```text
Backend          -> tensor + device operations (small trait)
ForwardCtx       -> backend, mode, run id, ShapePolicy, optional OperationProfiler
Parameter        -> owned by modules, never by a global registry
Module           -> typed input/output contract for layers/models
NamedParameters  -> stable, hierarchical parameter names
Tape (autodiff)  -> reverse-mode autodiff with explicit watching
Optimizer        -> applies gradients to parameters
Runtime          -> training/inference orchestration (TapeTrainer, InferencePool)
```

The design optimizes for *legible* deep-learning systems: backend selection, training mode, parameter ownership, and graph lifetime are all visible at API boundaries.

## Crate map

| Crate | Role |
|-------|------|
| [`rustral-core`](crates/core) | `Backend`, `TensorOps`, `FusionOps`, `Module`, `Parameter`, `NamedParameters`, `ForwardCtx`, `ShapePolicy`, `BackendCapabilities`, fusion/numerics/profiling helpers (`FusionOptimizer`, `TensorPool` / `PoolStrategy`, …). |
| [`rustral-ndarray-backend`](crates/ndarray-backend) | Reference CPU backend (correctness-first). |
| [`rustral-candle-backend`](crates/candle-backend) | Optimized CPU/GPU backend via [Candle](https://github.com/huggingface/candle). |
| [`rustral-wgpu-backend`](crates/wgpu-backend) | Experimental cross-platform GPU backend. |
| [`rustral-autodiff`](crates/autodiff) | Reverse-mode `Tape`, common ops, losses. |
| [`rustral-nn`](crates/nn) | Layers and small models (Linear, Embedding, LayerNorm, Conv2d, LSTM, Transformer, etc.). |
| [`rustral-optim`](crates/optim) | Optimizers (SGD, Adam) over `Parameter` slices. |
| [`rustral-data`](crates/data) | `Dataset` / `DataLoader` (in-memory, streaming, mmap). |
| [`rustral-io`](crates/io) | Safetensors save/load, typed state dicts. |
| [`rustral-runtime`](crates/runtime) | High-level training (`TapeTrainer::fit*`), inference pool, model I/O glue. |
| [`rustral-symbolic`](crates/symbolic) | Vocabulary helpers used by embedding/LM examples. |
| [`rustral-distributed`](crates/distributed) | Single-process simulation surface for distributed training. |
| [`rustral-metrics`](crates/metrics) | Lightweight metrics writers (JSONL/TensorBoard-style). |
| [`rustral-autotuner`](crates/autotuner) | Kernel config search + persistent cache; `enabled` / `ci_mode` presets for CI (see section below). |
| [`rustral-bench`](crates/bench) | Criterion microbenches (matmul, conv2d, lstm, attention). |
| [`rustral-hf`](crates/hf) | HuggingFace integration helpers. |
| [`rustral-model-zoo`](crates/model-zoo) | Curated `registry.json` + docs for checkpoint workflows and HF tensor-name pitfalls. |
| [`rustral-onnx-export`](crates/onnx-export) | Experimental ONNX writer (single Linear as `MatMul`+`Add`; vendored `onnx.proto` + `protoc-bin-vendored`). |
| [`rustral-inference-server`](crates/inference-server) | Axum HTTP service for JSON inference (MVP architecture), `/metrics`, Docker. |

## Where to look next

- **Rustdoc**: `cargo doc --workspace --no-deps --open` (per-crate API docs).
- **Examples**: see [`examples/README.md`](examples/README.md) (XOR through MoE) and [`crates/runtime/examples/`](crates/runtime/examples).
- **Serving / export**: [`crates/inference-server/README.md`](crates/inference-server/README.md), [`crates/inference-server/DEPLOYMENT.md`](crates/inference-server/DEPLOYMENT.md), [`docs/export-onnx-torchscript.md`](docs/export-onnx-torchscript.md), [`docs/wasm-wgpu-inference.md`](docs/wasm-wgpu-inference.md), [`docs/mobile-deployment.md`](docs/mobile-deployment.md).
- **Performance**: see [`BENCHMARKS.md`](BENCHMARKS.md).
- **Roadmap**: [`docs/master-plan.md`](docs/master-plan.md) (public feature and deployment tracks). Maintainer-only detail may live in a local `IMPROVEMENT_PLAN.md` at the repo root (gitignored).
- **Security**: [`SECURITY.md`](SECURITY.md) (points to `docs/SECURITY.md`).

## Backend capabilities vs runtime behavior

[`BackendCapabilities`](crates/core/src/backend.rs) reports hardware and layout hints (FP16/BF16, tensor cores, `optimal_batch_size`, conv layout preference, etc.). **Backends fill these fields; some are consumed today with incremental adoption.**

| Field / area | Typical use now | Notes |
|--------------|-----------------|--------|
| `optimal_batch_size` | [`BackendCapabilities::clamp_batch_size`](crates/core/src/backend.rs) | Soft upper hint for dataloaders or examples; not enforced globally. |
| `recommended_training_dtype`, mixed precision flags | [`BackendCapabilities::recommends_mixed_precision`](crates/core/src/backend.rs), [`BackendCapabilities::recommended_dtype_for_operation`](crates/core/src/backend.rs) | Provides concrete BackendCapabilities-driven decisions for dtype selection; incremental adoption pattern. |
| `preferred_conv_layout`, packed layouts | Mostly informational | Conv layers do not yet auto-transpose from these hints alone. |
| `ForwardCtx::shape_policy` | [`ShapePolicy`](crates/core/src/shape_policy.rs) | Documents static vs dynamic shape expectations for future graph capture / pooling. Default is `DynamicUnbounded`. |
| `ForwardCtx::profiler` | Optional [`OperationProfiler`](crates/core/src/operation_profiler.rs) | Attach with `with_profiler` or `set_profiler` for per-run timing. |

## Autotuner presets (`rustral-autotuner`)

- **`TunerConfig::default()`** — Full search within `max_tuning_time` (development).
- **`TunerConfig::fast()`** — Shorter random search; still writes cache when `cache_results` is true.
- **`TunerConfig::ci_safe()`** — Sets `ci_mode`: bounded iterations and time; pair with `enabled` as needed.
- **`TunerConfig::disabled()`** — Sets `enabled = false`: `AutoTuner::tune` returns a default kernel config without searching (CI-friendly).
- **Cache hits** — When `use_cache` is true, a hit re-benchmarks the cached config so `TuningResult` times are populated (not placeholder zeros).

## Key invariants

- No hidden mutable model state; pass `ForwardCtx` and backends explicitly.
- Parameters are owned by modules; optimizers and checkpointers traverse via `NamedParameters`.
- Save/load is **strict** by default: missing keys, extra keys, shape mismatch, and dtype mismatch all fail loudly (see [`crates/runtime/src/model_io.rs`](crates/runtime/src/model_io.rs)).
- Determinism: `TapeTrainer` shuffles via `seed ^ (epoch * constant)`; CPU runs are bitwise reproducible across runs given identical seeds (see [`crates/runtime/examples/emnlp_char_lm.rs`](crates/runtime/examples/emnlp_char_lm.rs)).
