# rustral-runtime

Training loops, inference pools, and parallel execution helpers.

Rustral is a Rust workspace for research and learning; see the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and status by backend.

## Serious training (`training` feature)

Enable autodiff + optim + checkpoint helpers:

```toml
rustral-runtime = { path = "../crates/runtime", features = ["training"] }
```

See `train_synthetic_classification` and `SeriousTrainingConfig` in `src/serious_training.rs`. The `basics/serious_train` binary under `examples/` runs the same loop on **Candle** (CPU by default; `--features cuda` passes through to `rustral-candle-backend`).
