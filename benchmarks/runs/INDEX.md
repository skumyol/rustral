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
| v0.1.0 | 2026-05-10 | ? | nlp, llm | NLP parity under nlp/ (SST-2, WikiText-2). LLM: `llm/gpt2_generate_example.json` (D3-style curated example manifests for `llm_generate`; regenerate with `rustral-llm generate --features hf-tokenizers --out-dir …`). |
| v0.2.0 | 2026-05-09 | ? | nlp | Paper-profile NLP parity (SST-2 / WikiText-2). Populate nlp/*.json after running Rustral + PyTorch baselines with --paper. |
