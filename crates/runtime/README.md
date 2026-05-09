# rustral-runtime

Training loops, inference pools, model I/O, and runnable training examples.

Rustral is a Rust workspace for research and learning. See the [repository README](https://github.com/skumyol/rustral#readme) for install steps, examples, and backend status.

## Training Feature

Enable autodiff + optim + checkpoint helpers:

```toml
rustral-runtime = { path = "../crates/runtime", features = ["training"] }
```

Optional live terminal dashboard during `TapeTrainer` runs (pulls in `rustral-tui`):

```toml
rustral-runtime = { path = "../crates/runtime", features = ["training", "tui"] }
```

The `training` feature powers:

- `TapeTrainer` and `SupervisedTapeModel`.
- Model-level `save_model` / `load_model` helpers through stable parameter names.
- `emnlp_char_lm`, a tiny char-level LM with determinism checks.
- `sst2_classifier`, a real-corpus SST-2 classifier that writes `manifest.json`.
- `wikitext2_lm`, a real-corpus WikiText-2 word LM that writes `manifest.json`.

Quick smoke runs:

```bash
cargo run --release -p rustral-runtime --features training --example sst2_classifier -- --quick
cargo run --release -p rustral-runtime --features training --example wikitext2_lm -- --quick
```

See [`EVALUATION.md`](../../EVALUATION.md) for the methodology and offline dataset mode.
