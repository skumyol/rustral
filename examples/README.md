# Rustral Examples Gallery

The runnable examples for Rustral live under `crates/runtime/examples/`. They depend on the trainer, dataset cache, and manifest writer from `rustral-runtime`, so they ship inside that crate instead of in this top-level workspace.

This `examples/` directory is a thin pointer crate: it builds a placeholder binary so `cargo build --manifest-path examples/Cargo.toml --workspace` stays green, and that's all. If you're looking for working code to read, jump straight to `crates/runtime/examples/`.

## What lives where

```text
crates/runtime/examples/
├── emnlp_char_lm.rs       # tiny char-level LM, determinism check, save/load/infer
├── sst2_classifier.rs     # SST-2 sentiment classifier, writes a reproducibility manifest
├── wikitext2_lm.rs        # WikiText-2 word-level LM, dev perplexity + manifest
└── tape_train_demo.rs     # smallest end-to-end tape training loop (XOR-style)
```

Each example writes a `manifest.json` next to its outputs with the git SHA, seed, hyperparameters, and final metric. See [`EVALUATION.md`](../EVALUATION.md) for the methodology and pinned dataset hashes. For optimization hooks (fusion, capabilities, autotuner, profiling), see the root [`README.md`](../README.md) and [`ARCHITECTURE.md`](../ARCHITECTURE.md).

## Running an example

```bash
# Smallest end-to-end tape training loop
cargo run -p rustral-runtime --features training --example tape_train_demo

# Char-level LM with determinism check
cargo run -p rustral-runtime --features training --example emnlp_char_lm
cargo run -p rustral-runtime --features training --example emnlp_char_lm -- --determinism-check

# SST-2 sentiment classifier (real corpus). Use --quick for a smoke run.
cargo run --release -p rustral-runtime --features training --example sst2_classifier
cargo run --release -p rustral-runtime --features training --example sst2_classifier -- --quick

# WikiText-2 word-level LM (real corpus). Use --quick for a smoke run.
cargo run --release -p rustral-runtime --features training --example wikitext2_lm
cargo run --release -p rustral-runtime --features training --example wikitext2_lm -- --quick
```

For CI / offline use (datasets must already be in the cache directory):

```bash
RUSTRAL_DATASET_OFFLINE=1 \
    cargo run --release -p rustral-runtime --features training --example sst2_classifier -- --quick
```

Set `RUSTRAL_CACHE_DIR=<path>` to override the default cache location (`~/.cache/rustral`). The pinned upstream URLs and SHA-256 values for the SST-2 and WikiText-2 mirrors are listed in [`EVALUATION.md`](../EVALUATION.md).

## Writing your own example

Drop a new `.rs` file under `crates/runtime/examples/` if it needs the trainer or dataset fetch (the typical case). Use the existing examples as a template:

- `tape_train_demo.rs` is the smallest end-to-end starting point.
- `sst2_classifier.rs` shows the dataset cache, tokenizer, trainer, and manifest writer wired together.

Run as `cargo run -p rustral-runtime --features training --example <name>`.
