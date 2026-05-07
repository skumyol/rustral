# Architecture

Rustral is a Rust deep-learning toolkit organized as a workspace of focused crates. This document is a discoverability landing page; the deeper architectural sketch lives in [`docs/architecture.md`](docs/architecture.md), and the canonical roadmap and invariants live in [`docs/master-plan.md`](docs/master-plan.md).

## Top-level layering

```text
Backend          -> tensor + device operations (small trait)
ForwardCtx       -> carries backend, mode, run state explicitly
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
| [`rustral-core`](crates/core) | `Backend`, `TensorOps`, `Module`, `Parameter`, `NamedParameters`, `ForwardCtx`. |
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
| [`rustral-autotuner`](crates/autotuner) | Persistent kernel-config cache (per-machine). |
| [`rustral-bench`](crates/bench) | Criterion microbenches (matmul, conv2d, lstm, attention). |
| [`rustral-hf`](crates/hf) | HuggingFace integration helpers. |

## Where to look next

- **Rustdoc**: `cargo doc --workspace --no-deps --open` (per-crate API docs).
- **Examples**: see [`examples/README.md`](examples/README.md) (XOR through MoE) and [`crates/runtime/examples/`](crates/runtime/examples).
- **Performance**: see [`BENCHMARKS.md`](BENCHMARKS.md).
- **Roadmap**: [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) (living plan-vs-codebase status table).
- **Security**: [`SECURITY.md`](SECURITY.md) (points to `docs/SECURITY.md`).

## Key invariants

- No hidden mutable model state; pass `ForwardCtx` and backends explicitly.
- Parameters are owned by modules; optimizers and checkpointers traverse via `NamedParameters`.
- Save/load is **strict** by default: missing keys, extra keys, shape mismatch, and dtype mismatch all fail loudly (see [`crates/runtime/src/model_io.rs`](crates/runtime/src/model_io.rs)).
- Determinism: `TapeTrainer` shuffles via `seed ^ (epoch * constant)`; CPU runs are bitwise reproducible across runs given identical seeds (see [`examples/emnlp_char_lm.rs`](crates/runtime/examples/emnlp_char_lm.rs)).
