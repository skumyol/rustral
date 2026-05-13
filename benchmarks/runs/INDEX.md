# Per-release benchmark snapshots

This directory holds curated benchmark snapshots taken at tagged releases.
Each subdirectory `<version>/` contains:

- `manifest.json` — top-level run metadata (release version, git SHA, hardware, harness flags, capture date)
- `<suite>.json`  — one file per suite (`rustral.json`, `candle.json`, `pytorch.json`, ...) conforming to schema v2.0.0
- `summary.md`    — Markdown summary regenerated from the per-suite JSONs

The contents are intentionally small: curated summaries and manifests only, not raw logs or
Criterion intermediate outputs. The full unified harness JSON written to `benchmarks/results/`
is git-ignored — only what lands here is the long-term reproducibility artifact.

## How a snapshot is captured

See [RELEASE_CHECKLIST.md](../../RELEASE_CHECKLIST.md), section "Capture benchmark snapshot".
The canonical command set is:

```bash
python3 scripts/bench/run_all.py --suite rustral --suite candle --repeats 10 --warmup 2 \
    --out benchmarks/results/release-${VERSION}.json
python3 scripts/bench/validate_schema.py benchmarks/results/release-${VERSION}.json
mkdir -p benchmarks/runs/${VERSION}
# (manual step: split per-suite, copy summary.md, write manifest.json)
```

## Index

| Version | Date | Hardware | Suites | Notes |
|---|---|---|---|---|
| v0.1.0 | ? | ? | nlp | Found nlp snapshots: sst2.json, sst2_pytorch.json, wikitext2.json, wikitext2_pytorch.json Files under nlp/: sst2.json, sst2_pytorch.json, wikitext2.json, wikitext2_pytorch.json |
