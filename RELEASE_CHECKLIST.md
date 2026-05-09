# Release checklist

This is a lightweight checklist for tagging a public release.

## Quality gates

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --exclude rustral-wgpu-backend`
- `cargo test -p rustral-runtime --features training`
- `cargo test -p rustral-nn --features autodiff,tape`
- `cargo test -p rustral-model-zoo` and `cargo test -p rustral-onnx-export` (registry + ONNX encode roundtrip)
- `cargo build -p rustral-inference-server` (HTTP MVP)
- `cargo package --workspace --allow-dirty` (understand any warnings; new crates may be `publish = false` until explicitly released)

## Artifact sanity

- README quickstart compiles and runs as written
- At least one end-to-end demo exists (train → save → load → infer)
- Model save/load is strict on missing/extra keys, shape mismatch, dtype mismatch
- Limitations are stated clearly in README

## Security

- `cargo audit` (install via `cargo install cargo-audit`)
- Review `docs/SECURITY.md`

## Capture benchmark snapshot

Capture a curated benchmark snapshot for the new tag. Snapshots live under
`benchmarks/runs/<version>/` and are the long-term reproducibility artifact
academic readers cite.

```bash
export VERSION="v0.1.1"   # match the new git tag
mkdir -p benchmarks/runs/${VERSION}

# Run the harness with extra repeats for a publishable snapshot.
python3 scripts/bench/run_all.py \
    --suite rustral --suite candle \
    --repeats 10 --warmup 2 \
    --out benchmarks/results/release-${VERSION}.json

# Validate against schema v2.0.0.
python3 scripts/bench/validate_schema.py benchmarks/results/release-${VERSION}.json

# Move the curated artifacts into the snapshot directory:
#   - copy benchmarks/results/summary.md to benchmarks/runs/${VERSION}/summary.md
#   - split per-suite JSON into benchmarks/runs/${VERSION}/<suite>.json
#   - hand-write benchmarks/runs/${VERSION}/manifest.json with required keys:
#       version, git_sha, date, hardware, suites (array), notes (string, optional)

# Regenerate benchmarks/runs/INDEX.md so the new snapshot appears in the table.
python3 scripts/bench/regen_index.py
```

## Nightly NLP gate (real data)

Before tagging, verify the real-data NLP gate runs successfully:

- Trigger [`nlp-real.yml`](.github/workflows/nlp-real.yml) via `workflow_dispatch` (or confirm the most recent scheduled run is green).
- Confirm curated manifests validate against the schema:

```bash
python3 -m pip install jsonschema
python3 scripts/bench/validate_manifest.py
```

Do NOT commit raw logs or Criterion intermediate outputs to `benchmarks/runs/`.
Curated JSON summaries, manifest, and `summary.md` only.

## Versioning & notes

- Update `CHANGELOG.md` with an “Unreleased” section → new version section
- Update `benchmarks/runs/INDEX.md` to add a row for the new snapshot
- Tag release in git

