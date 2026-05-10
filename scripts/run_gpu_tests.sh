#!/usr/bin/env bash
# Run NVIDIA CUDA tests locally (Linux). Metal workloads are macOS-only and skipped here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script targets Linux + NVIDIA CUDA. On macOS use Metal builds instead." >&2
  exit 1
fi

echo "=== CUDA toolkit check ==="
./scripts/check_cuda_env.sh

echo ""
echo "=== nvidia-smi ==="
nvidia-smi || true

export RUSTRAL_TEST_GPU="${RUSTRAL_TEST_GPU:-1}"

echo ""
echo "=== rustral-candle-backend (unit + CUDA smoke) ==="
cargo test -p rustral-candle-backend --features cuda -- --nocapture

echo ""
echo "=== rustral-runtime (training + cuda) ==="
cargo test -p rustral-runtime --features training,cuda -- --nocapture

echo ""
echo "=== rustral-bench (CUDA workload binary integration test) ==="
cargo test -p rustral-bench --features cuda --test workload_bins -- --nocapture

echo ""
echo "=== Optional: root system_tests GPU matmul perf (heavy) ==="
if [[ "${RUSTRAL_RUN_GPU_PERF:-}" == "1" ]]; then
  cargo test --test system_tests --features cuda run_all_system_tests -- --nocapture
else
  echo "Skip system_tests full suite (set RUSTRAL_RUN_GPU_PERF=1 to include; slow + CPU perf gates)."
fi

echo ""
echo "run_gpu_tests.sh: OK"
