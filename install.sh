#!/usr/bin/env bash
set -euo pipefail

echo "=== MNR (Modular Neural Runtime) Setup ==="

# Check Rust
echo "Checking Rust toolchain..."
if ! command -v rustc &> /dev/null; then
    echo "ERROR: Rust not found. Please install Rust: https://rustup.rs/"
    exit 1
fi

RUST_VERSION=$(rustc --version | awk '{print $2}')
echo "  Found Rust $RUST_VERSION"

REQUIRED_VERSION="1.75"
# Simple version check (not robust for all cases but sufficient here)
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$RUST_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "WARNING: Rust version $RUST_VERSION < $REQUIRED_VERSION. Please update."
fi

# Check cargo
if ! command -v cargo &> /dev/null; then
    echo "ERROR: cargo not found."
    exit 1
fi
echo "  Found cargo $(cargo --version | awk '{print $2}')"

# Install additional tools if missing
echo "Checking additional tools..."

if ! cargo clippy --version &> /dev/null; then
    echo "  Installing clippy..."
    rustup component add clippy || echo "  WARNING: Could not install clippy"
else
    echo "  clippy OK"
fi

if ! cargo fmt --version &> /dev/null; then
    echo "  Installing rustfmt..."
    rustup component add rustfmt || echo "  WARNING: Could not install rustfmt"
else
    echo "  rustfmt OK"
fi

# Build workspace
echo ""
echo "Building workspace..."
cargo build --workspace

echo ""
echo "Setup complete. Run ./run_tests.sh to execute tests."
