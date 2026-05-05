#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUSTRAL_RUN_EXAMPLE_PERF=1

echo "Running system performance tests (including examples)..."
echo "Tip: unset RUSTRAL_RUN_EXAMPLE_PERF to skip example binaries."

cargo test --test system_tests -- --nocapture

