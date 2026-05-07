# rustral-data

Datasets, batching, tokenization, and `DataLoader`-style iteration.

The default crate stays small. Real-corpus dataset fetching is behind the `fetch` feature:

```toml
rustral-data = { path = "../crates/data", features = ["fetch"] }
```

With `fetch` enabled, the crate provides:

- `fetch_url`, a content-addressed HTTP cache with SHA-256 checks.
- `WordLevelTokenizer`, a small in-tree tokenizer used by the SST-2 and WikiText-2 examples.
- Built-in loaders for SST-2 and WikiText-2 raw data.
- Offline mode through `RUSTRAL_DATASET_OFFLINE=1`.

The runtime NLP examples use this crate to produce reproducible runs without making CI depend on network access. See [`EVALUATION.md`](../../EVALUATION.md).
