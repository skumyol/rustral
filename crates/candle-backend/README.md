# rustral-candle-backend

High-performance `candle-core` backend (CPU, optional CUDA / Metal).

Implements `FusionOps` for sequence-level fused linear + bias (+ ReLU/GELU) used by `rustral_core::FusionOptimizer` / `rustral_nn::FusionHelper` (true single-kernel fusion remains future work).

Rustral is a Rust workspace for research and learning; see the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and status by backend.
