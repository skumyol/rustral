# rustral-candle-backend

High-performance `candle-core` backend (CPU, optional CUDA / Metal).

Implements `FusionOps` for sequence-level fused linear + bias (+ ReLU/GELU) used by `rustral_core::FusionOptimizer` / `rustral_nn::FusionHelper` (true single-kernel fusion remains future work).

Rustral is a Rust workspace for research and learning; see the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and status by backend.

## CUDA (NVIDIA)

```bash
./scripts/check_cuda_env.sh
cargo test -p rustral-candle-backend --features cuda
# Real device (matmul on GPU 0):
RUSTRAL_TEST_GPU=1 cargo test -p rustral-candle-backend --features cuda --test cuda_smoke
```

Workspace helper: [`scripts/run_gpu_tests.sh`](../../scripts/run_gpu_tests.sh) (Linux + NVIDIA; sets `RUSTRAL_TEST_GPU=1`).
