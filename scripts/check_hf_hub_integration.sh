#!/usr/bin/env bash
# Run Hugging Face Hub integration tests: real downloads, weight load, and inference.
# Used by git pre-commit (see scripts/git-hooks/pre-commit) and can be run manually before pushing.
#
# Environment:
#   RUSTRAL_TEST_HF_NETWORK=1  (set automatically by this script)
#   SKIP_HF_PRECOMMIT=1        skip all steps (e.g. offline; not recommended before merging HF changes)

set -euo pipefail

if [[ "${SKIP_HF_PRECOMMIT:-}" == "1" ]]; then
  echo "check_hf_hub_integration: SKIP_HF_PRECOMMIT=1 — skipping Hub tests"
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUSTRAL_TEST_HF_NETWORK=1

echo "=== HF Hub integration: download (rustral-hf) + load + inference (rustral-llm) ==="
echo "    (set SKIP_HF_PRECOMMIT=1 to skip — use only when offline)"
echo ""

echo "--- rustral-hf: real model download smoke ---"
cargo test -p rustral-hf test_download_real_model_smoke -- --nocapture

echo ""
echo "--- rustral-llm: tiny-random-gpt2 safetensors → meta → Gpt2Decoder + greedy generate ---"
cargo test -p rustral-llm --test hf_gpt2_real_load_smoke -- --nocapture

echo ""
echo "--- rustral-llm (hf-tokenizers): snapshot, tokenizer encode, generate ---"
cargo test -p rustral-llm --features hf-tokenizers --test hf_smoke -- --nocapture

echo ""
echo "check_hf_hub_integration: OK"
