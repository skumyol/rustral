# Contributing to Rustral

Thanks for your interest. Rustral is a learning- and research-oriented Rust ML workspace: keep public APIs explicit, test behavior, and avoid hidden global state.

## Quick checks (match CI)

From the repository root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
cargo test --workspace --exclude rustral-wgpu-backend
cargo test -p rustral-runtime --features training
cargo test -p rustral-nn --features autodiff tape
cargo test -p rustral-model-zoo
cargo test -p rustral-onnx-export
cargo build -p rustral-inference-server
cargo build --manifest-path examples/Cargo.toml --workspace
```

Or run `./run_tests.sh` for a slightly looser local script. Benchmark harness usage and baselines are documented in [`BENCHMARKS.md`](BENCHMARKS.md).

### Hugging Face Hub — run before you commit

Changes that touch `rustral-hf`, GPT-2 HF loaders, or Hub snapshots should be validated against the real Hub (download, merged Safetensors, partial weight load, and at least one greedy decode). Offline CI stays fast by default (`RUSTRAL_TEST_HF_NETWORK` unset); locally you should run the full integration script **before committing**:

```bash
./scripts/check_hf_hub_integration.sh
```

That sets `RUSTRAL_TEST_HF_NETWORK=1` and runs:

1. **`rustral-hf`** — download smoke (`tiny-random-bert`).
2. **`rustral-llm`** — [`hf_gpt2_real_load_smoke`](crates/llm/tests/hf_gpt2_real_load_smoke.rs): snapshot → meta → `Gpt2Decoder::from_hf_meta` → `generate_greedy`.
3. **`rustral-llm`** with **`hf-tokenizers`** — [`hf_smoke`](crates/llm/tests/hf_smoke.rs): snapshot, tokenizer encode, generate.

To **enforce** the same checks on every commit (requires network when you commit):

```bash
./scripts/install-git-hooks.sh   # once per clone: git config core.hooksPath scripts/git-hooks
```

If you must commit offline, use **`SKIP_HF_PRECOMMIT=1 git commit …`** only as an exception; run `./scripts/check_hf_hub_integration.sh` before pushing.

## Install troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `error: package 'rustral-...' does not contain this feature: training` | You forgot `--features training` (required for the runtime trainer + model I/O). |
| Build fails with linker errors involving `wgpu` | Skip the experimental backend with `--exclude rustral-wgpu-backend` or skip its tests; CI runs it with `continue-on-error`. |
| `cargo doc` warnings about broken intra-doc links | Run `cargo doc --workspace --no-deps -p <crate>` to isolate; intra-doc links must point to items reachable from that crate. |
| Slow examples | Use `--release`. The default `dev` profile is 10–50x slower for tensor workloads. |
| Tests pass locally but a Candle-backed test fails | Candle picks GPU when available; force CPU with `CANDLE_FORCE_CPU=1`. |
| Need CUDA for the benchmark harness | The PyTorch baseline is optional. Use `--suite rustral --suite candle` to skip the Python suite entirely. |

## Determinism expectations

CPU runs in Rustral are designed to be **bitwise reproducible** when:

- The same backend is selected (the reference `CpuBackend` from `rustral-ndarray-backend`).
- The same seed is passed to the trainer/optimizer/initializers.
- Shuffling uses the trainer’s seed-mixing scheme (`seed ^ (epoch * constant)`).

The EMNLP demo demonstrates this end-to-end. Reproduce the 3-run determinism evidence with:

```bash
cargo run -p rustral-runtime --features training --example emnlp_char_lm -- --determinism-check
```

This runs the same training 3 times and asserts identical logits, validation loss, and generated text. The smoke test for the same property is at [`crates/runtime/tests/emnlp_demo_smoke.rs`](crates/runtime/tests/emnlp_demo_smoke.rs).

GPU determinism is best-effort and depends on the underlying kernels; document any tolerance bands when reporting GPU numbers.

## Design principles

- No hidden mutable model state; pass `ForwardCtx` and backends explicitly (see [`ARCHITECTURE.md`](ARCHITECTURE.md) for optional profiler, `ShapePolicy`, and capability hints).
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
