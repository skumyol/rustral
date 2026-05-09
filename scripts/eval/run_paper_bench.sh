#!/usr/bin/env bash
# Paper-profile NLP benchmark: Rust examples with --paper, curated under
# benchmarks/runs/v0.2.0/nlp/. Optional PyTorch parity via --pytorch.
#
# Usage:
#   ./scripts/eval/run_paper_bench.sh --clean
#   ./scripts/eval/run_paper_bench.sh --clean --pytorch          # needs torch
#   ./scripts/eval/run_paper_bench.sh --device cuda --clean      # Rust CUDA build
#
# Extra args are forwarded to scripts/eval/run_nlp_real.py (e.g. --seeds 0 --skip-wikitext2).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=../bench/lib_rustral_python.sh
source "${ROOT}/scripts/bench/lib_rustral_python.sh"
_rustral_bench_resolve_python "$ROOT"
export RUSTRAL_PYTHON
exec "${RUSTRAL_PYTHON}" scripts/eval/run_nlp_real.py --paper "$@"
