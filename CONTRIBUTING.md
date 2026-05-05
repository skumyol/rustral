# Contributing to Rustral

Thanks for your interest. Rustral is a learning- and research-oriented Rust ML workspace: keep public APIs explicit, test behavior, and avoid hidden global state.

## Quick checks (match CI)

From the repository root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
cargo test --workspace --exclude rustral-wgpu-backend
cargo build --manifest-path examples/Cargo.toml --workspace
```

Or run `./run_tests.sh` for a slightly looser local script.

## Design principles

- No hidden mutable model state; pass `ForwardCtx` and backends explicitly.
- Prefer small traits; keep backend-specific code inside backend crates.
- Parallelism should be visible at API boundaries (runtime / data loaders).
- Add tests for shape errors and boundary cases when touching ops or layers.

## Pull requests

1. Open an issue for larger changes (API breaks, new backends).
2. Keep commits focused; reference issues when relevant.
3. Update `CHANGELOG.md` under **Unreleased** for user-visible changes.

## Adding a backend

1. Implement `rustral_core::Backend` + `TensorOps` for your tensors.
2. Put the crate under `crates/<name>-backend` with tests against `rustral-core`.
3. Document feature flags (e.g. CUDA) in the crate README.

## Publishing crates (maintainers)

See [`scripts/publish_crates_io.sh`](scripts/publish_crates_io.sh) for the dependency order and `cargo publish --dry-run` checks. Publishing requires a [crates.io](https://crates.io) token and owner permissions.

## Code of conduct

Please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report problems to maintainers via GitHub.
