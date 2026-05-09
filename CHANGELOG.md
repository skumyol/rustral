# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `rustral-core`: `ShapePolicy`, `ForwardCtx` optional profiler attachment, `BackendCapabilities::clamp_batch_size`, `TensorPool` `PoolStrategy` and `begin_step` for arena-style training clears, `FusionOptimizer::apply_*` entry points, `BackendCapabilities::recommends_mixed_precision`, `BackendCapabilities::recommended_dtype_for_operation`, `OperationType` enum for operation classification.
- `rustral-autodiff`: `Tape::gelu` with backward matching the tanh GELU approximation on `TensorOps`.
- `rustral-nn`: `FusionHelper` delegates to `FusionOptimizer`; tape transformer FFN uses GELU; eager `TransformerDecoderLayer` FFN uses GELU; eager `TransformerEncoderLayer` and `TransformerEncoderBlock` FFN use GELU; MoE `Expert` FFN uses GELU; `tape_feedforward_matches_eager_linears_and_gelu` integration test.
- `rustral-runtime`: `TapeTrainer` optional `tensor_pool` field with `with_tensor_pool` builder method; `clamp_batch_size` helper in SST-2 example.
- `rustral-autotuner`: `AutoTuner::tune` early return when `!config.enabled`; `TuningSession::run` CI mode iteration limit enforcement; `benchmark_cached_config` and `cached_result` methods for realistic cache hit metrics; enhanced `TunerConfig` documentation with detailed `ci_mode` behavior.
- Docs: `ARCHITECTURE.md` capabilities/autotuner tables; root README optimization section expanded; `EVALUATION.md` topology lines updated for GELU FFN; `docs/concepts.md` / `docs/api-signatures.md` synced with `ForwardCtx` and backend surface.

### Changed

- NLP real-data orchestrator (`scripts/eval/run_nlp_real.py`): added `--benchmark` (tiny model + small data) and reduced default WikiText-2 caps for faster local runs; SST-2 / WikiText-2 examples accept CLI overrides for `--seq-len`, `--d-model`, `--num-heads`, `--ffn-dim`, and (WikiText-2) `--block-size` / `--num-layers`.
- PyTorch NLP baselines: `--benchmark` flag for the same fast preset; WikiText-2 train token cap `0` means no cap (matches Rustral).
- `nlp-real` GitHub workflow uses `--benchmark` so CI finishes in reasonable time.

## [0.1.0] - 2026-05-05

### Added

- Workspace of `rustral-*` crates: core tensors/backends, NN layers, autodiff, optimizers, data loaders, I/O (Safetensors), Candle and experimental WebGPU backends, distributed-style APIs, metrics, HF helpers, benchmarks, and runtime helpers.
- Examples workspace under `examples/` (XOR, MNIST, vision/NLP samples).
- CI: `fmt`, `clippy -D warnings`, `doc`, tests (excluding flaky `rustral-wgpu-backend` by default), examples build.
- Documentation: architecture concepts, security guidelines, master roadmap (honest scope).

### Notes

- **`rustral-wgpu-backend`** is experimental; some platforms abort during test process teardown. CI runs it with `continue-on-error`.
- **Distributed training** APIs are suitable for learning and single-process simulation; do not assume full multi-node NCCL/MPI production coverage without additional work.

[0.1.0]: https://github.com/skumyol/rustral/releases/tag/v0.1.0
