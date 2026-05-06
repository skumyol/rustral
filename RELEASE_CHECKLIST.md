# Release checklist

This is a lightweight checklist for tagging a public release.

## Quality gates

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --exclude rustral-wgpu-backend`
- `cargo test -p rustral-runtime --features training`
- `cargo test -p rustral-nn --features autodiff,tape`
- `cargo package --workspace --allow-dirty` (understand any warnings)

## Artifact sanity

- README quickstart compiles and runs as written
- At least one end-to-end demo exists (train → save → load → infer)
- Model save/load is strict on missing/extra keys, shape mismatch, dtype mismatch
- Limitations are stated clearly in README

## Security

- `cargo audit` (install via `cargo install cargo-audit`)
- Review `docs/SECURITY.md`

## Versioning & notes

- Update `CHANGELOG.md` with an “Unreleased” section → new version section
- Tag release in git

