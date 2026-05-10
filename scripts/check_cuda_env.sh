#!/usr/bin/env bash
set -euo pipefail

# candle-core 0.10 + RTX-class GPUs: 12.0+ toolkits work in practice; raise if a crate requires newer.
want_major=12
want_minor=0

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. CUDA builds require CUDA toolkit >= ${want_major}.${want_minor}." >&2
  exit 1
fi

ver_line="$(nvcc --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1 \2/p' | tail -n 1)"
if [[ -z "${ver_line}" ]]; then
  echo "Could not parse nvcc version. Output was:" >&2
  nvcc --version >&2 || true
  exit 1
fi

read -r have_major have_minor <<<"${ver_line}"

if (( have_major < want_major )) || { (( have_major == want_major )) && (( have_minor < want_minor )); }; then
  echo "CUDA toolkit too old: nvcc reports ${have_major}.${have_minor}, need >= ${want_major}.${want_minor}." >&2
  echo "Either upgrade CUDA toolkit or build CPU-only (omit --features cuda)." >&2
  exit 1
fi

echo "CUDA OK: nvcc ${have_major}.${have_minor} (>= ${want_major}.${want_minor})"

