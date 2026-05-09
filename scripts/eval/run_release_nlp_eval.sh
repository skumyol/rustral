#!/usr/bin/env bash
# Run full pre-release NLP evaluation (paper profile: 3 seeds, shared vocabs, curated JSON).
#
# This is NOT the fast CI preset (--benchmark). Expect CPU runs on the order of tens of minutes
# to hours depending on hardware. Downloads SST-2 and WikiText-2 on first use.
#
# Usage (from repo root):
#   ./scripts/eval/run_release_nlp_eval.sh 0.2.0
#   ./scripts/eval/run_release_nlp_eval.sh 0.2.0 --pytorch   # also writes *_pytorch.json (needs torch)
#
# Outputs:
#   benchmarks/runs/v<version>/nlp/sst2.json
#   benchmarks/runs/v<version>/nlp/wikitext2.json
#   optional: sst2_pytorch.json, wikitext2_pytorch.json
#
# After a successful run: commit the nlp/*.json files, refresh benchmarks/runs/INDEX.md if needed,
# and update benchmarks/runs/v<version>/manifest.json (date, hardware, git_sha).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || -z "${1:-}" ]]; then
  sed -n '1,20p' "$0"
  exit 0
fi

VERSION="$1"
shift || true
PYTORCH=()
for a in "$@"; do
  if [[ "$a" == "--pytorch" ]]; then
    PYTORCH=(--pytorch)
  else
    echo "unknown arg: $a (only --pytorch supported)" >&2
    exit 2
  fi
done

# Strip leading "v" if caller passed v0.2.0
VERSION="${VERSION#v}"

OUT="out/release_nlp_v${VERSION}"

echo "==> Full release NLP eval (paper profile)"
echo "    curated_version=$VERSION  out_root=$OUT  seeds=0,1,2"
echo

python3 scripts/eval/run_nlp_real.py \
  --paper \
  --clean \
  --curated-version "$VERSION" \
  --seeds 0,1,2 \
  --out-root "$OUT" \
  "${PYTORCH[@]}"

echo
echo "==> Validating curated manifests"
python3 scripts/bench/validate_manifest.py \
  "benchmarks/runs/v${VERSION}/nlp/sst2.json" \
  "benchmarks/runs/v${VERSION}/nlp/wikitext2.json"

echo
echo "OK. Next steps for maintainers:"
echo "  - git add benchmarks/runs/v${VERSION}/nlp/"
echo "  - Update benchmarks/runs/v${VERSION}/manifest.json (date, hardware, git_sha) and INDEX.md"
echo "  - git commit -m \"bench: NLP release snapshot v${VERSION}\""
