# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- [`CONTRIBUTING.md`](CONTRIBUTING.md): Hugging Face Hub **pre-commit** workflow — [`scripts/check_hf_hub_integration.sh`](scripts/check_hf_hub_integration.sh), [`scripts/install-git-hooks.sh`](scripts/install-git-hooks.sh), and `SKIP_HF_PRECOMMIT` for offline commits.
- **Pre-release NLP**: [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) and [`EVALUATION.md`](EVALUATION.md) now require a **paper-profile** three-seed run via [`scripts/eval/run_release_nlp_eval.sh`](scripts/eval/run_release_nlp_eval.sh) before tagging; [`nlp-real.yml`](.github/workflows/nlp-real.yml) adds optional `workflow_dispatch` preset **paper** (slow) while scheduled runs stay on the fast **benchmark** preset.
- README **Examples gallery**: commands now match [`crates/runtime/examples/`](crates/runtime/examples/), [`crates/nn/examples/`](crates/nn/examples/), and the top-level [`examples/`](examples/) pointer crate (see [`examples/README.md`](examples/README.md)).
- [`docs/api-signatures.md`](docs/api-signatures.md): full `TensorOps` method inventory aligned with `rustral-core` ([`Backend`](crates/core/src/backend.rs) / `BackendCapabilities` were already listed).
- [`docs/concepts.md`](docs/concepts.md): distributed snippets use crate-root imports (`ZeroOptimizer` / `Zero2Optimizer`, `PipelineParallel` / `PipelineConfig`); `DataParallelTrainer` and `ProcessGroup` remain valid at the crate root.
- **“Paper mode”** in scripts (e.g. `run_nlp_real.py --paper`, benchmark harness presets) means a heavier evaluation or benchmark configuration, not an academic PDF shipped in this repository. External references and methodology live in docs such as [`EVALUATION.md`](EVALUATION.md) and [`crates/nn/TRANSFORMERS.md`](crates/nn/TRANSFORMERS.md).

### Added

- `rustral-llm`: **`generate` CLI metrics** — JSON includes **`hub_snapshot_ms`**, **`model_init_ms`**, **`tokenizer_load_ms`**, **`first_token_ms`** (first greedy step), **`decode_wall_ms`**, **`tokens_per_sec`** (`max_new_tokens` / decode seconds); **`Gpt2Decoder::generate_greedy_timed`** + **`GreedyDecodeTiming`** for the same breakdown.
- `rustral-llm`: **`CausalLm`** trait (`crates/llm/src/causal_lm.rs`) with greedy **`generate_greedy(ctx, …)`** using an explicit **`ForwardCtx`**; **`Gpt2Decoder`** implements it for **`CpuBackend`** and exposes **`backend()`** for building matching contexts. Convenience **`Gpt2Decoder::generate_greedy`** still allocates an inference context.
- `rustral-hf`: **`scan_local_model_dir`** discovers `config.json`, tokenizer files, single or **sharded** SafeTensors (`model.safetensors.index.json` + shards), and optional `model.gguf` under a local path with no Hub API; **`HubModelSnapshot::require_config_json`** for callers that require config; **`snapshot_model_at`** pins Hub downloads to an optional revision (branch/tag/SHA); **`snapshot_model`** delegates to `snapshot_model_at(..., None)` (default `main`).
- `rustral-llm`: re-exports **`HubModelSnapshot`** / **`HubModelFiles`** from `rustral-hf` (removes the duplicated snapshot struct).
- `rustral-io`: **`load_meta_state_dict_from_paths`** merges multiple SafeTensors files into one **`MetaStateDict`**; **`load_meta_state_dict_from_hub_index`** parses `model.safetensors.index.json` and loads shards from a snapshot root; **`IoError::FileRead`** / **`IndexJson`** for filesystem and index parse failures.
- `rustral-llm`: GPT-2 **`hf_weights`** — map HF Safetensors (`transformer.*` / `gpt2.*`) into `TransformerDecoder` **`NamedParameters`** for embeddings, layer norms, FFN, `ln_f`, `lm_head` (F32); **`load_hf_gpt2_weights_into_decoder`** / **`Gpt2Decoder::load_hf_weights_from_meta`** / **`Gpt2Decoder::from_hf_meta`**; HF **attention** fused weights remain unloaded until a conversion pass exists (documented in-module).
- `rustral-inference-server`: Axum HTTP MVP for Safetensors artifacts (`/health`, `/ready`, `/v1/metadata`, `/v1/infer`, Prometheus `/metrics`), graceful shutdown (SIGINT/SIGTERM on Unix), [`DEPLOYMENT.md`](crates/inference-server/DEPLOYMENT.md), `Dockerfile`, and `docker-compose.yml`.
- `rustral-model-zoo`: embedded `registry.json`, HF key vs `NamedParameters` notes, and `registry()` parser for tooling/tests.
- `rustral-onnx-export`: `export_linear_f32` ONNX builder (opset 17), vendored `onnx.proto`, `protoc-bin-vendored` for reproducible builds without system `protoc`.
- `rustral-runtime` example `save_linear_artifact` writes a disk artifact for the inference server.
- Docs: [`docs/export-onnx-torchscript.md`](docs/export-onnx-torchscript.md), [`docs/wasm-wgpu-inference.md`](docs/wasm-wgpu-inference.md), [`docs/mobile-deployment.md`](docs/mobile-deployment.md); root README deployment section; [`docs/master-plan.md`](docs/master-plan.md) Track H updates.
- `rustral-core`: `ShapePolicy`, `ForwardCtx` optional profiler attachment, `TensorPool` `PoolStrategy` and `begin_step` for arena-style training clears, `FusionOptimizer::apply_*` entry points; **`Backend`** extensions `attention_ops`, `quantization_ops`, `as_any`; **`BackendCapabilities`** fields (`supports_fast_fp16_tensor_cores`, `preferred_conv_layout`, layout flags) plus `clamp_batch_size`, `recommends_mixed_precision`, `recommended_dtype_for_operation`, and **`OperationType`**; **`TensorOps::dropout_with_seed`** default hook for reproducible dropout; numerics / golden micro-block test helpers under `crates/core/tests/golden*`; **`OperationProfiler::new_ci_safe`** and CI-oriented snapshot export when `TapeTrainer` runs with `RUSTRAL_PROFILE=1` and `CI` or `RUSTRAL_PROFILE_CI` (optional `RUSTRAL_PROFILE_EXPORT_JSON`, `RUSTRAL_PROFILE_SNAPSHOT_LIMIT`).
- `rustral-autodiff`: `Tape::gelu` with backward matching the tanh GELU approximation on `TensorOps`; fused linear+bias+activation tape ops; **`TapeModule`** for `LinearReLU` / `LinearGELU`, **`SelfAttention`** / **`MultiHeadAttention`** (feature `autodiff`).
- `rustral-nn`: `FusionHelper` delegates to `FusionOptimizer`; tape transformer FFN uses GELU; eager `TransformerDecoderLayer` FFN uses GELU; eager `TransformerEncoderLayer` and `TransformerEncoderBlock` FFN use GELU; MoE `Expert` FFN uses GELU; `tape_feedforward_matches_eager_linears_and_gelu` integration test.
- `rustral-runtime`: `TapeTrainer` optional `tensor_pool` field with `with_tensor_pool` builder method; **`TrainingReport.throughput`** / **`ThroughputStats`** (examples/sec, batches/sec); **`Trainer`** high-level facade (`high_level_trainer`, builder over `TapeTrainer`); `clamp_batch_size` helper in SST-2 example.
- `rustral-optim`: **`Optimizer::step_named_parameters`** visitor path for **`Sgd`** and **`Adam`** (avoids collecting parameters into a flat slice each step); default trait fallback for other optimizers.
- `rustral-autotuner`: `AutoTuner::tune` early return when `!config.enabled`; `TuningSession::run` CI mode iteration limit enforcement; `benchmark_cached_config` and `cached_result` methods for realistic cache hit metrics; enhanced `TunerConfig` documentation with detailed `ci_mode` behavior.
- `rustral-ndarray-backend`: SIMD optimizations for CPU element-wise operations using `wide` crate (f32x4 vectors): `add`, `mul`, `div`, `exp`, `log`, `sqrt`, `relu`, `softmax`. All implementations include scalar fallback for remainder elements. Optional **parallel reductions** for large tensors when `RUSTRAL_PAR_REDUCE=1` or `RUSTRAL_PARALLEL_REDUCTIONS=1` (see `ARCHITECTURE.md` / `docs/master-plan.md` Track O9b).
- Docs: `ARCHITECTURE.md` capabilities/autotuner tables; root README optimization section expanded; `EVALUATION.md` topology lines updated for GELU FFN; `docs/concepts.md` / `docs/api-signatures.md` synced with `ForwardCtx`, full **`Backend`** / **`BackendCapabilities`** surface, and `TensorOps::dropout_with_seed`.

### Changed

- `rustral-llm` **`hf_weights`**: GPT-2 FFN `mlp.c_fc` / `mlp.c_proj` weights accept Hugging Face tensor layouts that are transposed relative to `Linear` storage `[out_dim, in_dim]` and transpose when needed (matches real Hub checkpoints such as `tiny-random-gpt2`).
- NLP real-data orchestrator (`scripts/eval/run_nlp_real.py`): added `--benchmark` (tiny model + small data) and reduced default WikiText-2 caps for faster local runs; SST-2 / WikiText-2 examples accept CLI overrides for `--seq-len`, `--d-model`, `--num-heads`, `--ffn-dim`, and (WikiText-2) `--block-size` / `--num-layers`.
- PyTorch NLP baselines: `--benchmark` flag for the same fast preset; WikiText-2 train token cap `0` means no cap (matches Rustral).
- `nlp-real` GitHub workflow uses `--benchmark` so CI finishes in reasonable time.

## [0.1.0] - 2026-05-05

### Added

- Workspace of `rustral-*` crates: core tensors/backends, NN layers, autodiff, optimizers, data loaders, I/O (Safetensors), Candle and experimental WebGPU backends, distributed-style APIs, metrics, HF helpers, benchmarks, and runtime helpers.
- Examples entry point under `examples/` (pointer crate; runnable demos in `crates/runtime/examples/` and `crates/nn/examples/`).
- CI: `fmt`, `clippy -D warnings`, `doc`, tests (excluding flaky `rustral-wgpu-backend` by default), examples build.
- Documentation: architecture concepts, security guidelines, master roadmap (honest scope).

### Notes

- **`rustral-wgpu-backend`** is experimental; some platforms abort during test process teardown. CI runs it with `continue-on-error`.
- **Distributed training** APIs are suitable for learning and single-process simulation; do not assume full multi-node NCCL/MPI production coverage without additional work.

[0.1.0]: https://github.com/skumyol/rustral/releases/tag/v0.1.0
