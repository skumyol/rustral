#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUSTRAL_RUN_GPU_PERF=1

echo "Running GPU performance tests (CUDA)..."
echo "This is opt-in and may take a while."

./scripts/check_cuda_env.sh

cargo test --test system_tests --features cuda -- --nocapture

