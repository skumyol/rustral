#!/usr/bin/env bash
# One-shot: full micro-benchmark queue + NLP paper profile + PyTorch NLP parity + comparative Markdown.
# Logs: benchmarks/queue_logs/latest/run.log
# Report: benchmarks/reports/comparative_queue_<stamp>.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RUN_NLP_PAPER=1
export RUN_NLP_PYTORCH="${RUN_NLP_PYTORCH:-1}"
export RUN_PYTORCH_CUDA="${RUN_PYTORCH_CUDA:-1}"
exec ./scripts/bench/queue_all_benchmarks.sh "$@"
