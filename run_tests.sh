#!/usr/bin/env bash
set -euo pipefail

echo "=== MNR Test Suite ==="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

run_test() {
    local crate=$1
    local extra_args=${2:-}
    echo ""
    echo "--- Testing $crate ---"
    if cargo test -p "$crate" $extra_args -- --nocapture; then
        echo -e "${GREEN}PASS${NC}: $crate"
    else
        echo -e "${RED}FAIL${NC}: $crate"
        FAILED=1
    fi
}

# Format check
echo "--- Format Check ---"
if cargo fmt -- --check; then
    echo -e "${GREEN}PASS${NC}: Formatting"
else
    echo -e "${YELLOW}WARN${NC}: Formatting issues (run 'cargo fmt' to fix)"
fi

# Clippy linting
echo ""
echo "--- Clippy ---"
if cargo clippy --workspace -- -D warnings; then
    echo -e "${GREEN}PASS${NC}: Clippy"
else
    echo -e "${YELLOW}WARN${NC}: Clippy warnings"
fi

# Run tests for each crate
run_test "mnr-core"
run_test "mnr-ndarray-backend"
run_test "mnr-symbolic"
run_test "mnr-nn"
run_test "mnr-runtime"

# Run integration tests / examples
echo ""
echo "--- Examples ---"
if cargo run --example linear_readout; then
    echo -e "${GREEN}PASS${NC}: linear_readout example"
else
    echo -e "${YELLOW}WARN${NC}: linear_readout example failed or not configured"
fi

# Summary
echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}=== All tests passed ===${NC}"
    exit 0
else
    echo -e "${RED}=== Some tests failed ===${NC}"
    exit 1
fi
