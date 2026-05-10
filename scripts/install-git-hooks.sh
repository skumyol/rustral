#!/usr/bin/env bash
# Point this repository at scripts/git-hooks so pre-commit runs check_hf_hub_integration.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git config core.hooksPath scripts/git-hooks
echo "core.hooksPath set to scripts/git-hooks (pre-commit runs check_hf_hub_integration.sh)"
echo "To disable hook path:  git config --unset core.hooksPath"
