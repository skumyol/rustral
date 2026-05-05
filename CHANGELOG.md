# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
