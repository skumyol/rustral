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

## Full NLP evaluation (required before tagging)

CI’s [`nlp-real.yml`](.github/workflows/nlp-real.yml) job uses the **fast** `--benchmark` preset so every run finishes on a free runner. **That is not sufficient for a release.** Before you tag, you must capture **paper-profile** results (three seeds, full SST-2 train set in the example, 200k WikiText-2 train tokens, larger model — see [`EVALUATION.md`](EVALUATION.md)) and **commit** the curated JSON under `benchmarks/runs/v<version>/nlp/`.

From the repo root (allow plenty of wall time on CPU; first run downloads corpora):

```bash
python3 -m pip install --upgrade pip jsonschema
./scripts/eval/run_release_nlp_eval.sh 0.2.0
# Optional PyTorch parity JSON (requires `torch` in the active Python):
# ./scripts/eval/run_release_nlp_eval.sh 0.2.0 --pytorch
```

Equivalent manual invocation:

```bash
python3 scripts/eval/run_nlp_real.py --paper --clean --curated-version 0.2.0 --seeds 0,1,2 --out-root out/paper_bench_release
python3 scripts/bench/validate_manifest.py benchmarks/runs/v0.2.0/nlp/sst2.json benchmarks/runs/v0.2.0/nlp/wikitext2.json
```

Then:

- Update `benchmarks/runs/v<version>/manifest.json` with **date**, **hardware** (CPU model / GPU if used), and **`git_sha`** for the commit you are about to tag.
- Ensure [`benchmarks/runs/INDEX.md`](benchmarks/runs/INDEX.md) describes the snapshot.
- `git add benchmarks/runs/v<version>/` and include the NLP JSON in the release PR or tag commit.

Maintainers can also trigger a **paper** preset on [`nlp-real.yml`](.github/workflows/nlp-real.yml) via **workflow_dispatch**; artifacts are uploaded for review but **do not** land in git automatically — copy or re-run locally and commit.

## Nightly NLP gate (real data, fast preset)

For regression signal between releases:

- Scheduled / default `workflow_dispatch` runs use `--benchmark` (single seed, tiny model).
- Validate with:

```bash
python3 -m pip install jsonschema
python3 scripts/bench/validate_manifest.py benchmarks/runs/v0.1.0/nlp/sst2.json benchmarks/runs/v0.1.0/nlp/wikitext2.json
```

Do NOT commit raw logs or Criterion intermediate outputs to `benchmarks/runs/`.
Curated JSON summaries, manifest, and `summary.md` only.

## Versioning & notes

- Update `CHANGELOG.md` with an “Unreleased” section → new version section
- Update `benchmarks/runs/INDEX.md` to add a row for the new snapshot
- Tag release in git

