# Rustral Concrete Software Development Plan

This is the execution plan for making Rustral credible for **EMNLP**, **JOSS**, and **MLSys**. It focuses on code, tests, examples, benchmarks, and measurable evidence.

Status date: **May 9, 2026** (optimization + doc sync pass).

## Venue Gates

| Venue | Submission reality | What Rustral must prove |
|-------|--------------------|--------------------------|
| EMNLP 2026 main track | ARR deadline: May 25, 2026. This is extremely tight. | Only attempt if a narrow NLP systems/resource paper can be completed with real metrics in under 3 weeks. |
| EMNLP 2026 industry track | Deadline: June 16, 2026. More realistic if framed as a practical NLP systems artifact. | End-to-end NLP demo, reproducible evaluation, clear limitations, anonymized artifact. |
| JOSS | Rolling. Submit only when software is usable, documented, tested, and archived. | Stable release, docs, tests, example workflows, statement of need, DOI. |
| MLSys 2027 | Best topical fit. Expected late-2026 submission window, verify official dates when posted. | Systems contribution with benchmark evidence, baselines, ablations, reproducible scripts. |

## Definition Of Ready

Rustral is not publication-ready until these are true:

- `cargo test --workspace --exclude rustral-wgpu-backend` passes.
- `cargo test -p rustral-nn --features autodiff tape` passes.
- At least one real model trains through a high-level trainer without manual parameter arrays.
- Whole-model `save_model` / `load_model` works through stable parameter names.
- README has one compiling train/save/load/infer workflow.
- Benchmarks can be regenerated from scripts.
- Limitations are documented honestly.

## Plan vs Codebase Status (living)

This section is a quick “where we are” mapping from this plan to the current codebase. Update it whenever tasks land.

Legend: **Done** = merged/implemented with tests; **In progress** = implemented but missing plan exit criteria; **Not started** = only described in plan.

### Track A: Core Usability

- **A1 Fix Current Build Breaks**: **Done**
  - **Implemented**:
    - Missing import fixes landed in:
      - `crates/nn/tests/tape_embedding.rs` (imports `GradExtFromStore`)
      - `crates/nn/tests/tape_layer_norm.rs` (imports `GradExtFromStore`)
    - CI-ish command list / local test runner:
      - `run_tests.sh`
      - README “Run the Test Suite” section
  - **Still needed**: validate full workspace commands on all supported platforms.

- **A2 Finish Named Parameter Plumbing**: **In progress**
  - **Implemented**:
    - Trainer can operate via `NamedParameters` without caller-managed parameter slices:
      - `crates/runtime/src/tape_trainer.rs` (`TapeTrainer::train_model`)
    - Model-level persistence uses stable parameter names as keys:
      - `crates/runtime/src/model_io.rs` (`save_model` / `load_model`)
    - Stable-name tests for key layers/models:
      - `crates/nn/tests/named_parameters_stability.rs`
    - Helper utilities:
      - `crates/core/src/module.rs` (`collect_named_parameters`, `collect_named_parameter_ids`)
  - **Still needed (per plan)**:
    - Mutable visitor adapter for optimizers (avoid parameter cloning during `Optimizer::step`).

- **A3 High-Level Trainer API**: **In progress**
  - **Implemented**:
    - Minimal model-driven trainer exists:
      - `crates/runtime/src/tape_trainer.rs` (`train_model`)
    - Supervised high-level training API:
      - `crates/runtime/src/tape_trainer.rs` (`SupervisedTapeModel`, `fit`, `fit_classification`, `TrainingReport`)
    - Demo:
      - `crates/runtime/examples/tape_train_demo.rs` (MSE regression)
      - `crates/runtime/examples/tape_xor_classification.rs` (classification via `fit_classification`)
  - **Still needed (per plan)**:
    - North-star `Trainer::classification(...).fit(...)` style API.
    - Expand `TrainingReport` (throughput, samples/sec, richer metrics).

### Track B: Autodiff And Tensor Correctness

- **B1 Loss Correctness**: **Done**
  - **Implemented**:
    - Tape losses exist:
      - `crates/autodiff/src/lib.rs` (`mse_loss`, `cross_entropy_loss`)
    - Row-wise `[batch, classes]` cross-entropy with correct scaling and upstream gradient handling.
    - Finite-difference gradient checks:
      - `crates/autodiff/tests/finite_difference.rs`

- **B2 Axis-Aware TensorOps**: **Done**
  - **Implemented**:
    - Axis-aware ops added to `rustral-core` `TensorOps`:
      - `crates/core/src/backend.rs`
    - Implemented for backends:
      - `crates/ndarray-backend/src/lib.rs`
      - `crates/candle-backend/src/lib.rs`

### Track C: Persistence And Artifact Quality

- **C1 Whole-Model State Dict**: **Done**
  - **Implemented**:
    - Save/load via named parameters:
      - `crates/runtime/src/model_io.rs`
    - Roundtrip + backend parity tests:
      - `crates/runtime/tests/model_io_roundtrip.rs`
      - `crates/runtime/tests/backend_correctness.rs`
    - Typed safetensors state dict support (shape + dtype validation):
      - `crates/io/src/lib.rs` (`save_state_dict_typed`, `load_state_dict_typed`)
    - Strict negative tests:
      - `crates/runtime/tests/model_io_negative.rs` (missing/extra/shape/dtype mismatch)
  - **Still needed (per plan)**:
    - Optional path-based `save_model(path, ...)` / `load_model(path, ...)` if that becomes the public API (current API is bytes).

- **C2 Release Hygiene**: **Done**
  - **Implemented**:
    - Plan is not ignored; it can be committed/published (no `.gitignore` rule for `IMPROVEMENT_PLAN.md`).
    - Release/security docs:
      - `SECURITY.md` (points to `docs/SECURITY.md`)
      - `CHANGELOG.md`
      - `RELEASE_CHECKLIST.md`
    - `.gitignore` tightened to ignore generated diagram download directory.

### Track D: EMNLP-Ready NLP Demo

- **D1 Minimal NLP Model**: **Done**
  - **Implemented**:
    - One-command char-level next-token LM demo (in-repo TinyShakespeare excerpt; no downloads):
      - `crates/runtime/examples/emnlp_char_lm.rs`
    - Reuses Rustral's training spine end-to-end:
      - `Embedding` + `reshape` + `Linear` model implementing `SupervisedTapeModel`
      - `TapeTrainer::fit_classification` for training
      - strict `save_model` / `load_model` roundtrip with logit equality assertion
      - greedy deterministic generation from a fixed prompt
    - Examples gallery entry:
      - `examples/README.md` lists the demo with both default and `--determinism-check` invocations.

- **D2 EMNLP Paper Evidence**: **Done**
  - **Implemented**:
    - Per-epoch metrics (train loss, train accuracy), validation loss + accuracy, samples/sec.
    - 3-run determinism mode (`--determinism-check`) writes a structured JSON report and asserts bitwise equality across runs on CPU.
    - Smoke-test gating in CI:
      - `crates/runtime/tests/emnlp_demo_smoke.rs` exercises the same spine (3-run determinism + save/load roundtrip) at a tiny budget.

- **D3 Real-Corpus NLP Evidence (Phase 2 P1)**: **Done**
  - **Implemented**:
    - Dataset fetching with content-addressed cache + offline mode + checksum override:
      - `crates/data/src/fetch.rs` (`ureq` + `sha2` + `flate2`, gated behind the `fetch` feature)
    - In-tree word-level tokenizer adapter:
      - `crates/data/src/tokenizer.rs` (`WordLevelTokenizer`, deferred dependency on the heavy `tokenizers` crate)
    - Built-in dataset loaders:
      - `crates/data/src/datasets/sst2.rs`
      - `crates/data/src/datasets/wikitext2.rs`
    - End-to-end examples with reproducibility manifests (`manifest.json` per run including git SHA, dataset hash, hyperparameters, throughput, dev metric):
      - `crates/runtime/examples/sst2_classifier.rs`
      - `crates/runtime/examples/wikitext2_lm.rs`
    - Smoke tests (offline-flagged, `#[ignore]`-gated, exercise CI without network):
      - `crates/runtime/tests/sst2_smoke.rs`
      - `crates/runtime/tests/wikitext2_lm_smoke.rs`
    - Methodology document: [EVALUATION.md](EVALUATION.md) covers tokenization, splits, seeds, hyperparameters, hardware, units, perplexity definition, and one-command repro.
    - Stronger tape-trained transformer baselines + causal masking:
      - `crates/nn/src/tape_transformer.rs` (position-wise FFN: **Linear → GELU → Linear** via `Tape::gelu`)
      - `crates/nn/tests/tape_transformer_block.rs`
      - `crates/nn/tests/tape_ffn_eager_parity.rs` (tape FFN vs eager `Module` forward)
      - `crates/autodiff/tests/finite_difference.rs` (`finite_difference_gelu_matches_tape`); `crates/autodiff/src/lib.rs` (`test_gelu_and_backward`)
    - Eager decoder FFN aligned with tape: **GELU** in `TransformerDecoderLayer` (`crates/nn/src/transformer.rs`)
    - Curated real-data multi-seed snapshots (v0.1.0):
      - `benchmarks/runs/v0.1.0/nlp/sst2.json`
      - `benchmarks/runs/v0.1.0/nlp/wikitext2.json`
      - `benchmarks/runs/v0.1.0/nlp/sst2_pytorch.json`
      - `benchmarks/runs/v0.1.0/nlp/wikitext2_pytorch.json`
      - `benchmarks/runs/v0.1.0/manifest.json` (snapshot metadata; see `benchmarks/runs/INDEX.md`)
    - Nightly / manual real-data gate + schema validation:
      - `.github/workflows/nlp-real.yml`
      - `benchmarks/manifest_schema.json`
      - `scripts/bench/validate_manifest.py`
  - **Still needed (follow-up phase)**:
    - Larger LM profile (deeper tape-integrated transformer) once `MultiHeadAttention`/`TransformerEncoderLayer` ship `TapeModule` integration.
    - WikiText-103 / OpenWebText scale runs (deferred until WikiText-2 is fully published in a release snapshot).

### Track E: JOSS-Ready Software

- **E1 Documentation**: **Done** (ongoing small updates expected)
  - **Implemented**:
    - Root-level discoverability docs:
      - `ARCHITECTURE.md` (crate map + invariants + **BackendCapabilities vs runtime**, **autotuner presets**, pointers to deeper docs)
      - `BENCHMARKS.md` (canonical perf entry point; autotuner pointer)
    - Updated examples + contributing docs:
      - `examples/README.md` includes the EMNLP demo and its determinism flag; links to optimization overview
      - `CONTRIBUTING.md` adds install troubleshooting, expanded test commands, determinism expectations, and `ForwardCtx`/capabilities pointers
    - **Optimization / API sync (May 2026)**:
      - `README.md` (extended `ForwardCtx`, fusion, `PoolStrategy`, numerics)
      - `EVALUATION.md` (topology diagrams: GELU tape FFN; note on older ReLU wording)
      - `docs/concepts.md`, `docs/api-signatures.md`, `docs/master-plan.md`, `docs/architecture.md`
      - Crate READMEs: `crates/core`, `crates/nn`, `crates/autodiff`, `crates/autotuner`, `crates/candle-backend`, `crates/runtime`
      - `CHANGELOG.md` **Unreleased** entries for the above
    - Rustdoc hygiene pass: `cargo doc --workspace --no-deps` is now warning-free (fixed broken intra-doc links and unclosed HTML tags across `core`, `nn`, `autodiff`, `candle-backend`, `distributed`).

### Track F: MLSys-Ready Metrics

- **F Benchmark harness**: **Done**
  - **Implemented**:
    - Unified runner with multi-run variance:
      - `scripts/bench/run_all.py` orchestrates suites, captures per-run timings, writes `benchmarks/results/<timestamp>.json` and a regenerable `benchmarks/results/summary.md`.
    - Shared JSON schema across suites:
      - `benchmarks/SCHEMA.md`
      - Rust harness library: `crates/bench/src/lib.rs` (timing + JSON serialization)
    - Rustral suite (Rust):
      - `crates/bench/src/bin/rustral_workloads.rs` (matmul, attention.small, attention.medium)
    - Candle-direct baseline (Rust):
      - `crates/bench/src/bin/candle_workloads.rs` (same workloads, no Rustral abstraction layer)
    - PyTorch baseline (Python, optional):
      - `benchmarks/pytorch/baselines.py` (auto-skipped by the orchestrator if `torch` is missing)
    - Generated bench artifacts ignored:
      - `.gitignore` covers `benchmarks/results/`.

- **F2 Phase 2 Bench expansion (P0+P2+P3+P4)**: **Done**
  - **Implemented (P0 Foundation, schema v2)**:
    - Schema v2.0.0 with `device`, `dtype`, `model_params`, `schema_version`, `commit`, `hostname`, `rustc`, `features`:
      - `benchmarks/schema_v2.json`, `benchmarks/SCHEMA.md`, `crates/bench/src/lib.rs` (`Sample` + `samples_to_json`)
    - Validator: `scripts/bench/validate_schema.py` (jsonschema-based; falls back to a hand-written validator).
    - New JSON-harness workloads beyond matmul/attention: `conv2d.{small,medium,large}`, `mlp_train_step`, `optimizer_step.{sgd,adam}` (10M default, `--profile heavy` for 100M).
    - Per-release snapshot directory with index regenerator:
      - `benchmarks/runs/INDEX.md`, `scripts/bench/regen_index.py` (`--check` mode for CI).
    - Release procedure: `RELEASE_CHECKLIST.md` "Capture benchmark snapshot" section.
  - **Implemented (P2 CI artifacts)**:
    - `bench-cpu` job in `.github/workflows/ci.yml`: builds release bins, runs the harness, validates against schema_v2.json (blocking), uploads JSON artifact for 14 days, verifies INDEX.md is in sync via `regen_index.py --check`.
    - NLP smoke tests in CI (`nlp-smoke` job) for SST-2 and WikiText-2 examples.
  - **Implemented (P3 Backend matrix)**:
    - CUDA workload binary: `crates/bench/src/bin/rustral_workloads_cuda.rs` (gated by `--features cuda`).
    - Metal workload binary: `crates/bench/src/bin/rustral_workloads_metal.rs` (gated by `--features metal`); workspace-level `metal` feature forwarded through `crates/candle-backend/Cargo.toml` and root `Cargo.toml`.
    - Optional GPU CI workflow: `.github/workflows/bench-gpu.yml` (label-gated `bench-gpu`, self-hosted runner).
    - PyTorch parity (matmul, attention, conv2d) in `benchmarks/pytorch/baselines.py` against schema v2.
    - Bencher.dev integration: `scripts/bench/to_bencher_bmf.py` plus an opt-in `bencher run` step in the `bench-cpu` job (gated on `BENCHER_API_TOKEN` + `BENCHER_PROJECT`).
  - **Implemented (P4 Wider workloads)**:
    - Transformer encoder forward bench (2L/d128/h4/seq128).
    - Decoder prefill + per-token decode (no-cache baseline).
    - KV cache prefill vs decode micro-benchmark (`KVCache::append`).
    - Save / load throughput on a synthetic ~50M f32-param model via `rustral-runtime::model_io`.
    - GitHub Pages dashboard: `scripts/bench/render_site.py` + `.github/workflows/pages.yml` rendering per-version snapshots from `benchmarks/runs/`.
  - **Still needed (tracked, deferred)**:
    - LSTM `lstm_lm_train_step` JSON-harness coverage (gated on tape-integrated LSTM). `lstm_forward` is now implemented following weight layout fix.
    - Tape-integrated full encoder train step bench (fwd + bwd + optimizer.step) once a canonical tape-trained encoder stack is selected for the harness.
    - First full unified-harness per-release snapshot under `benchmarks/runs/<version>/` (e.g. `rustral.json` / `candle.json` from schema v2). Current snapshots are NLP-only (`suites: ["nlp"]`).

### Track O: Device-agnostic optimizations & observability

Legend: reflects the **architecture checklist** and related code landed **May 2026**; see also [`CHANGELOG.md`](CHANGELOG.md) **Unreleased**.

- **O1 Fusion policy (single surface)**: **Done**
  - `FusionOptimizer::apply_matmul_bias_*` in `crates/core/src/fusion.rs`
  - `FusionHelper` delegates from `crates/nn/src/fusion_helper.rs`
- **O2 Tape GELU + transformer FFN parity**: **Done** (see Track D3 bullets for tests)
- **O3 `ForwardCtx` extensions**: **Done**
  - `ShapePolicy`: `crates/core/src/shape_policy.rs`, wired in `crates/core/src/context.rs`
  - Optional `OperationProfiler`: `with_profiler` / `set_profiler` / `profiler()` on `ForwardCtx`
- **O4 `TensorPool` strategies**: **Done**
  - `PoolStrategy`, `begin_step`, `with_strategy` / `with_limits_and_strategy` in `crates/core/src/tensor_pool.rs`
  - **`TapeTrainer`** calls `pool.begin_step()` after `optimizer.step` when a pool is configured (`crates/runtime/src/tape_trainer.rs`)
- **O5 `BackendCapabilities` consumption (initial)**: **Done**
  - `clamp_batch_size` in `crates/core/src/backend.rs` (other capability fields remain mostly advisory; see `ARCHITECTURE.md`)
- **O6 Autotuner CI / opt-out behavior**: **Done**
  - `TunerConfig::enabled`, `ci_mode`, cache re-benchmark on hit: `crates/autotuner/src/tuner.rs` (+ crate / `ARCHITECTURE.md` docs)
- **O7 Fused ops on `Tape`**: **Done**
  - Tape-level fused linear+activation ops: `crates/autodiff/src/lib.rs` (`fused_linear_bias_{relu,gelu}_tape`)
  - `TapeModule` for `LinearReLU` / `LinearGELU`: `crates/nn/src/tape.rs`
  - Grad tests: `crates/nn/tests/tape_linear_activation.rs`
- **O8 TapeModule for attention**: **Done**
  - `TapeModule` for `SelfAttention` / `MultiHeadAttention`: `crates/nn/src/attention.rs` (feature-gated behind `autodiff`)

## Architecture: principles vs implementation (checklist)

Actionable items to keep README and `ARCHITECTURE.md` claims aligned with the code (fusion, tape, autotuner, capabilities). Check boxes as work lands; optional note in parentheses for PR or commit. **Verified against repo: May 9, 2026.**

### Fusion and hot paths

- [x] **Single fusion entry surface** — `FusionOptimizer::apply_*` + `FusionHelper` delegating (`crates/core/src/fusion.rs`, `crates/nn/src/fusion_helper.rs`).
- [x] **Tape FFN matches eager semantics** — `TapeFeedForward` is `Linear -> GELU -> Linear` (`crates/nn/src/tape_transformer.rs`).
- [x] **Fused linear+activation on tape (optional)** — `Tape::{fused_linear_bias_relu_tape,fused_linear_bias_gelu_tape}` + `TapeModule` for `LinearReLU` / `LinearGELU`.

### Transformer blocks

- [x] **Eager decoder FFN activation** — `TransformerDecoderLayer` FFN uses GELU (`crates/nn/src/transformer.rs`).
- [x] **Tape vs eager parity smoke** — `crates/nn/tests/tape_ffn_eager_parity.rs`.

### Capabilities and runtime

- [x] **Consume `BackendCapabilities`** — `BackendCapabilities::clamp_batch_size` (`crates/core/src/backend.rs`).
- [x] **Document advisory vs active flags** — `ARCHITECTURE.md` + README.

### Memory and shapes (optimization roadmap)

- [x] **Shape policy** — `ShapePolicy` on `ForwardCtx` (`crates/core/src/shape_policy.rs`, `context.rs`).
- [x] **Training vs inference pooling** — `PoolStrategy` + `TensorPool::begin_step` (`crates/core/src/tensor_pool.rs`); **`TapeTrainer` invokes `begin_step` after each optimizer step** when `with_tensor_pool` is used (`crates/runtime/src/tape_trainer.rs`).

### Observability

- [x] **`ForwardCtx` and profiling** — `with_profiler` / `set_profiler` / `profiler()` on `ForwardCtx`.

### Autotuner

- [x] **Document CI and opt-out presets** — `rustral-autotuner` crate rustdoc + `ARCHITECTURE.md`.

## Track A: Core Usability

Goal: make the wrapper language useful to users, not just framework authors.

### A1. Fix Current Build Breaks

Tasks:

- Import `GradExtFromStore` in `crates/nn/tests/tape_embedding.rs`.
- Import `GradExtFromStore` in `crates/nn/tests/tape_layer_norm.rs`.
- Run and fix:
  - `cargo test -p rustral-nn --features autodiff tape`
  - `cargo test -p rustral-runtime --features training`
  - `cargo test --workspace --exclude rustral-wgpu-backend`

Exit criteria:

- All focused tests pass locally.
- CI command list is documented in README or CONTRIBUTING.

### A2. Finish Named Parameter Plumbing

Current state:

- `NamedParameters` exists.
- Several layers implement it.
- Trainer and persistence do not yet use it.

Tasks:

- Add tests for stable names on:
  - `Linear`
  - `Embedding`
  - `LayerNorm`
  - `Sequential2`
  - `TransformerEncoder`
  - `TransformerDecoder`
- Add helper utilities:
  - `collect_named_parameters(&model) -> Vec<(String, ParameterRef)>`
  - `collect_named_parameter_ids(&model) -> HashMap<ParameterId, String>`
  - mutable visitor adapter for optimizers
- Remove `#[allow(dead_code)]` from `NamedParameters` once used.

Exit criteria:

- Nested model parameter names are deterministic across runs.
- Names are used by trainer logs and checkpoint keys.

### A3. High-Level Trainer API

Current problem:

- `TapeTrainer` still asks the user for `&mut [Parameter<B>]` and a tape closure.

Target API:

```rust
let mut trainer = Trainer::classification(Adam::new(1e-3))
    .epochs(10)
    .batch_size(64);

trainer.fit(&mut model, train_data, valid_data, &backend)?;
```

Tasks:

- Introduce a supervised trait:

```rust
pub trait SupervisedTapeModel<B: Backend, X, Y>: NamedParameters<B> {
    fn forward_tape(
        &self,
        input: X,
        tape: &mut Tape<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>;

    fn loss_tape(
        &self,
        logits: TensorId,
        target: Y,
        tape: &mut Tape<B>,
        ctx: &mut ForwardCtx<B>,
    ) -> Result<TensorId>;
}
```

- Make `TapeTrainer::fit`:
  - create tape
  - call model forward
  - call loss
  - backprop
  - collect gradients using named parameters
  - call optimizer
  - record metrics
- Support batch size, epochs, seed, shuffle on/off.
- Return `TrainingReport`.

Exit criteria:

- XOR or MLP classification trains without manual tape code in the example.
- Trainer report includes loss history, accuracy if labels are class ids, elapsed time, samples/sec.

## Track B: Autodiff And Tensor Correctness

Goal: make training results trustworthy.

### B1. Loss Correctness

Current problem:

- Tape cross entropy is simplified.
- Log-softmax backward is simplified.

Tasks:

- Implement correct row-wise `cross_entropy_tape` for `[batch, classes]`.
- Scale gradients by batch size.
- Respect upstream `grad_out`.
- Add MSE tape loss.
- Add finite-difference gradient checks for:
  - matmul
  - linear
  - embedding dense gradient
  - layer norm
  - cross entropy

Exit criteria:

- Gradient checks pass within tolerance on CPU.
- Cross entropy matches a small hand-computed example.

### B2. Axis-Aware TensorOps

Tasks:

- Add required core ops:
  - `softmax_dim`
  - `log_softmax_dim`
  - `sum_dim`
  - `mean_dim`
  - `var_dim`
  - `broadcast_to`
- Add optional extension ops:
  - `gelu`
  - `masked_fill`
  - `scatter_add_rows`
- Implement first for ndarray backend.
- Implement Candle backend next.
- Keep `wgpu` as experimental and allowed to return unsupported errors.

Exit criteria:

- Transformer attention no longer relies on whole-tensor softmax.
- Error messages include operation, expected rank/axis, actual shape.

## Track C: Persistence And Artifact Quality

Goal: model artifacts are inspectable, reproducible, and useful.

### C1. Whole-Model State Dict

Tasks:

- Add `StateDict` type:
  - key
  - shape
  - dtype
  - parameter id
  - tensor data
- Add:
  - `state_dict(&model, &backend)`
  - `load_state_dict(&mut model, state, &backend)`
  - `save_model(path, &model, &backend)`
  - `load_model(path, &mut model, &backend)`
- Use `NamedParameters` for keys.
- Use `tensor_to_vec` for serialization.

Exit criteria:

- CPU train/save/load/infer round trip returns same logits within tolerance.
- Missing key, extra key, shape mismatch, and dtype mismatch tests exist.

### C2. Release Hygiene

Tasks:

- Decide whether `IMPROVEMENT_PLAN.md` should stay ignored. If it should be published, remove it from `.gitignore`.
- Remove generated zip exports from docs unless needed.
- Add root `SECURITY.md` or link clearly to `docs/SECURITY.md`.
- Add `CHANGELOG.md` unreleased section if not already current.
- Add release checklist.

Exit criteria:

- `cargo package --workspace --allow-dirty` issues are understood or fixed.
- Public release can be tagged without private/local artifacts.

## Track D: EMNLP-Ready NLP Demo

Goal: create an NLP-centered artifact credible enough for EMNLP industry/demo/workshop framing.

Recommended EMNLP framing:

> Rust-native explicit-context training and inference for small NLP models, with reproducible save/load and backend swapping.

Do not frame as beating PyTorch. Frame as reliability, deployability, and inspectability for NLP systems.

### D1. Minimal NLP Model

Build one polished demo:

- Dataset: Tiny Shakespeare, WikiText-2 subset, AG News subset, or SST-2 subset.
- Model options:
  - character language model
  - small Transformer encoder classifier
  - small GPT-style decoder
- Backends:
  - ndarray correctness path
  - Candle practical path

Minimum demo command:

```bash
cargo run -p rustral-nn --example emnlp_text_classifier --features autodiff
```

or:

```bash
cargo run -p rustral-nn --example emnlp_tiny_lm --features autodiff
```

Metrics:

- training loss curve
- validation accuracy or perplexity
- train samples/sec
- inference latency p50/p95
- checkpoint size
- save/load equality check

Exit criteria:

- One command downloads/prepares data or uses a tiny checked-in sample.
- One command trains.
- One command evaluates.
- One command saves/loads and runs inference.

## Track E: Standalone LLM Loader And Runner

Goal: create an independent LLM module that can work as a standalone CLI like a small `llama.cpp`-style runner, while also integrating with Rustral backends, tensors, state dicts, and examples.

Current state:

- `rustral-hf` can download `model.safetensors` and return a flat `HashMap<String, Vec<f32>>`.
- It cannot currently download sharded LLM weights, read `config.json`, load tokenizers, map Hugging Face parameter names into Rustral model structs, run RoPE/KV-cache decoding, or execute quantized GGUF weights.
- Therefore, Rustral cannot currently load a Hub LLM such as Llama/Mistral/Qwen and run it like `llama.cpp`.

Recommended crate:

```text
crates/llm
```

Package name:

```text
rustral-llm
```

Design requirement:

- The crate must work standalone as a CLI/library.
- The crate must also use Rustral core abstractions where possible:
  - `Backend`
  - `TensorOps`
  - `ForwardCtx`
  - `rustral-io`
  - `rustral-hf`
  - `rustral-nn` layers where useful

### E1. Minimal Non-Quantized Hub Runner

Target first model:

- `hf-internal-testing/tiny-random-gpt2`
- or another tiny CausalLM with SafeTensors and tokenizer files.

Tasks:

- Add Hugging Face repo snapshot support:
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `generation_config.json`
  - `model.safetensors`
  - `model-00001-of-000xx.safetensors`
  - `model.safetensors.index.json`
- Add state dict loader that preserves:
  - tensor name
  - shape
  - dtype
  - shard source
- Add tokenizer support through the `tokenizers` crate.
- Implement one architecture first:
  - GPT-2 style decoder-only transformer
  - or Llama-style decoder-only transformer if RoPE is ready
- Add greedy generation.
- Add CLI:

```bash
cargo run -p rustral-llm -- generate \
  --model hf-internal-testing/tiny-random-gpt2 \
  --prompt "Rust is" \
  --max-new-tokens 32 \
  --backend ndarray
```

Exit criteria:

- Downloads from HF Hub.
- Loads config/tokenizer/weights.
- Generates text deterministically with greedy decoding.
- Runs on CPU reference backend.
- Has one integration test using a tiny model or local fixture.

### E2. Rustral Integration Layer

Tasks:

- Expose a reusable API:

```rust
let model = LlmModel::from_hub("...", &backend)?;
let output = model.generate("Rust is", GenerationConfig::default(), &backend)?;
```

- Provide:
  - `LlmConfig`
  - `GenerationConfig`
  - `TokenizerHandle`
  - `CausalLm` trait
  - `WeightMapper` trait
- Make the inference path use `ForwardCtx`.
- Add model metadata reporting:
  - parameter count
  - dtype
  - max sequence length
  - vocab size
  - backend

Exit criteria:

- CLI and library use the same code path.
- Rustral examples can call the LLM module without shelling out.
- README has a minimal LLM quickstart.

### E3. Practical Llama-Class Path

Do not start here. Do this after the tiny runner works.

Tasks:

- Add RoPE.
- Add RMSNorm.
- Add SiLU/SwiGLU.
- Add grouped-query attention support.
- Add KV cache.
- Add sharded SafeTensors streaming load.
- Add memory-mapped or chunked loading where possible.
- Add Candle backend path for practical speed.

Target models:

- TinyLlama variants only if memory and architecture support are ready.
- Small Qwen/Gemma/Mistral-style models only after architecture mapping is verified.

Exit criteria:

- Can load a small real decoder-only Hugging Face model in SafeTensors.
- Can run prompt prefill plus token-by-token decode.
- Reports tokens/sec and memory usage.

### E4. Quantized Standalone Path

This is what makes it more like `llama.cpp`, but it is a separate milestone.

Options:

- Support GGUF directly.
- Or define a Rustral quantized SafeTensors format first.

Tasks for GGUF path:

- Parse GGUF metadata.
- Load quantized tensors.
- Implement dequantized matmul kernels or bridge through Candle where possible.
- Support common quantization families only after exact format semantics are confirmed.

Tasks for Rustral quantized path:

- Extend `rustral-nn::quantization`.
- Add model export/import for quantized linear layers.
- Benchmark accuracy and memory tradeoffs.

Exit criteria:

- A quantized tiny model runs from CLI.
- Metrics include model size, peak memory, tokens/sec, and output parity against f32 tiny model.

### E5. Metrics For LLM Module

Required metrics:

- model load time
- first-token latency
- tokens/sec after prefill
- peak memory
- checkpoint size
- generated output determinism under greedy decoding
- CPU backend vs Candle backend delta
- Rustral f32 vs external reference output delta on tiny model

This module can become the strongest EMNLP/JOSS demo if scoped honestly: not "we beat llama.cpp," but "Rustral can load, inspect, and run small Hub language models through an explicit Rust-native stack."

## Track E+: Detailed Hugging Face / llama.cpp Integration Plan

Goal: support Hugging Face model repositories and llama.cpp-style local model workflows without breaking Rustral's design principles:

- no network or file-format logic in `rustral-core`
- no hidden global model state
- no backend-specific assumptions in model definitions
- explicit `ForwardCtx` for execution mode and run state
- standalone tools should reuse the same library APIs used by examples/tests
- unsupported model families and quantization formats must fail explicitly

### E+1. Crate Boundaries

Add or refine crates as follows:

| Crate | Responsibility | Must not do |
|-------|----------------|-------------|
| `rustral-core` | Traits: `Backend`, `TensorOps`, `Module`, `ForwardCtx`, parameter traversal | Network, tokenizers, Hugging Face, GGUF, model-family assumptions |
| `rustral-io` | SafeTensors, state dicts, tensor metadata, local model artifact I/O | Hub API calls, tokenizer logic, generation policies |
| `rustral-hf` | Hugging Face repo access, snapshot/file download, Hub metadata | Model execution, architecture mapping, backend ops |
| `rustral-llm` | LLM configs, tokenizers, architecture mapping, generation, CLI | Low-level tensor backend implementations |
| `rustral-gguf` | Optional GGUF parser/metadata/quant tensor reader | General model execution or Hugging Face network logic |
| `rustral-candle-backend` | Practical CPU/CUDA tensor execution | Tokenizer/config parsing, Hub logic |

Recommended package additions:

```text
crates/llm      -> rustral-llm
crates/gguf     -> rustral-gguf (optional, after SafeTensors path works)
```

`rustral-llm` may depend on:

- `rustral-core`
- `rustral-io`
- `rustral-hf`
- `rustral-nn`
- `serde`
- `serde_json`
- `tokenizers`
- `thiserror`

`rustral-gguf` should be optional and behind a feature:

```toml
[features]
gguf = ["dep:rustral-gguf"]
```

### E+2. Hugging Face Repository Support

Current `rustral-hf::download_state_dict` is too small for LLMs. Replace or extend it with a repo snapshot API.

Target API:

```rust
pub struct HubModelSnapshot {
    pub model_id: String,
    pub revision: Option<String>,
    pub root: PathBuf,
    pub files: HubModelFiles,
}

pub struct HubModelFiles {
    pub config_json: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub tokenizer_config_json: Option<PathBuf>,
    pub generation_config_json: Option<PathBuf>,
    pub safetensors_index_json: Option<PathBuf>,
    pub safetensors_files: Vec<PathBuf>,
    pub gguf_files: Vec<PathBuf>,
}

pub fn snapshot_model(model_id: &str, revision: Option<&str>) -> Result<HubModelSnapshot, HfError>;
```

Supported file patterns:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `generation_config.json`
- `model.safetensors`
- `model-00001-of-000xx.safetensors`
- `model.safetensors.index.json`
- `*.gguf` only if `gguf` feature is enabled

Design constraints:

- Downloads are explicit. No model code should fetch files during `forward`.
- Revision should be user-selectable for reproducibility.
- Cache directory should be discoverable and logged.
- Missing files should produce structured errors with model id and file name.

### E+3. Tensor Metadata And State Dict

For LLMs, a flat `HashMap<String, Vec<f32>>` is insufficient.

Add a metadata-preserving state dict:

```rust
pub enum TensorDType {
    F32,
    F16,
    BF16,
    I8,
    U8,
    Quantized(String),
}

pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
    pub data: TensorStorage,
    pub source_file: Option<PathBuf>,
}

pub enum TensorStorage {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
    Mmap { path: PathBuf, offset: u64, len: u64 },
}

pub struct StateDict {
    pub tensors: HashMap<String, TensorEntry>,
    pub metadata: HashMap<String, String>,
}
```

Required capabilities:

- read single SafeTensors
- read sharded SafeTensors via index JSON
- preserve dtype and shape
- detect missing/unexpected keys
- support lazy loading later without changing public API

Do not convert all tensors to `Vec<f32>` by default for large LLMs. That destroys memory behavior and makes quantized/f16 models impossible to represent honestly.

### E+4. Model Family Abstraction

Do not build one generic "any LLM" loader first. Use explicit model-family adapters.

Target traits:

```rust
pub trait CausalLm<B: Backend> {
    fn prefill(&mut self, input_ids: &[u32], ctx: &mut ForwardCtx<B>) -> Result<Logits<B>>;
    fn decode_next(&mut self, token_id: u32, ctx: &mut ForwardCtx<B>) -> Result<Logits<B>>;
    fn reset_kv_cache(&mut self);
}

pub trait ModelFamily<B: Backend> {
    type Config;
    type Model: CausalLm<B>;

    fn family_name(&self) -> &'static str;
    fn parse_config(&self, json: &serde_json::Value) -> Result<Self::Config>;
    fn map_weights(&self, state: &StateDict, backend: &B) -> Result<Self::Model>;
}
```

Initial families:

1. `Gpt2Family`
   - easiest small HF path
   - no RoPE
   - good for first tokenizer/config/weights/generation pipeline
2. `LlamaFamily`
   - RoPE
   - RMSNorm
   - SwiGLU
   - GQA variants later
3. `QwenFamily` or `MistralFamily`
   - only after Llama works

Family detection:

- read `config.json`
- inspect `model_type`
- reject unsupported values clearly:

```text
unsupported model_type "qwen2"; supported: gpt2, llama
```

### E+5. Tokenizer Layer

Tokenization belongs in `rustral-llm`, not `rustral-core`.

API:

```rust
pub struct TokenizerHandle {
    inner: tokenizers::Tokenizer,
}

impl TokenizerHandle {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self>;
    pub fn encode(&self, text: &str) -> Result<Vec<u32>>;
    pub fn decode(&self, ids: &[u32]) -> Result<String>;
}
```

Rules:

- tokenization is explicit before model forward
- generation loop owns decode/encode
- model layers only see token ids/tensors

### E+6. Generation API

Generation should be a high-level utility in `rustral-llm`, not in model layers.

API:

```rust
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub stop_tokens: Vec<u32>,
    pub seed: u64,
}

pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub timings: GenerationTimings,
}

pub fn generate<B: Backend, M: CausalLm<B>>(
    model: &mut M,
    tokenizer: &TokenizerHandle,
    prompt: &str,
    config: &GenerationConfig,
    backend: &B,
) -> Result<GenerationOutput>;
```

Phases:

1. greedy only
2. temperature
3. top-k
4. top-p
5. repetition penalty
6. streaming callback

Execution rules:

- `ForwardCtx::new(backend, Mode::Inference)` is created by generation unless caller provides one.
- KV cache is owned by model instance and reset explicitly.
- No hidden global RNG. Sampling uses `GenerationConfig.seed`.

### E+7. CLI Design

Binary:

```text
rustral-llm
```

Commands:

```bash
rustral-llm inspect --model meta-llama/... --revision ...
rustral-llm generate --model hf-internal-testing/tiny-random-gpt2 --prompt "Rust is"
rustral-llm chat --model ... --system "..." --prompt "..."
rustral-llm convert --from safetensors --to rustral-state --model ...
rustral-llm bench --model ... --prompt-file prompts.txt
```

Minimum first CLI:

```bash
cargo run -p rustral-llm -- generate \
  --model hf-internal-testing/tiny-random-gpt2 \
  --prompt "Rust is" \
  --max-new-tokens 32 \
  --backend ndarray
```

CLI output should include:

- model id
- revision
- architecture
- backend
- parameter count
- load time
- prompt tokens
- generated tokens
- first-token latency
- tokens/sec
- peak memory if available

### E+8. llama.cpp / GGUF Compatibility

Do not mix GGUF into the SafeTensors path. Treat it as a separate adapter.

GGUF support phases:

1. inspect metadata only
2. load unquantized/f16 tensors if present
3. load common quantized tensor blocks
4. execute quantized matmul
5. compare against llama.cpp outputs on tiny prompts

API:

```rust
pub struct GgufModelFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
}

pub fn inspect_gguf(path: impl AsRef<Path>) -> Result<GgufModelFile>;
pub fn load_gguf_state_dict(path: impl AsRef<Path>) -> Result<StateDict>;
```

Compatibility stance:

- Rustral should not depend on llama.cpp internals.
- Rustral may use llama.cpp as an external baseline for metrics.
- GGUF parsing should live in `rustral-gguf`.
- Quantized execution should be optional and clearly marked experimental until benchmarked.

Do not claim llama.cpp parity until:

- same prompt/tokenizer produces comparable token ids
- logits are checked on a tiny model
- generated outputs match under greedy decoding
- tokens/sec and memory are reported

### E+9. Backend Integration

First backend order:

1. ndarray CPU
   - correctness reference
   - tiny models only
2. Candle CPU
   - practical local inference
3. Candle CUDA
   - optional feature
4. `wgpu`
   - experimental only

Backend requirements for useful LLM inference:

- matmul
- reshape/view
- transpose
- softmax over last dim
- elementwise add/mul/div
- layer norm/RMSNorm helpers
- RoPE helper or enough ops to implement RoPE
- efficient gather for embeddings
- KV cache-friendly concat/slice or explicit cache tensors

Missing ops should fail with structured unsupported errors, not panics.

### E+10. Weight Mapping

Weight mapping must be explicit per family.

Example:

```rust
pub struct WeightMapRule {
    pub hf_key: &'static str,
    pub rustral_key: &'static str,
    pub expected_shape: ShapeSpec,
    pub transform: WeightTransform,
}

pub enum WeightTransform {
    Identity,
    Transpose2D,
    SplitQkv,
    MergeQkv,
    PermuteForRope,
}
```

Rules:

- log every missing and unexpected key
- provide strict and permissive loading modes
- strict mode is default for tests and publication metrics
- permissive mode is allowed for partial loading experiments

### E+11. Testing Strategy

Test levels:

1. unit tests
   - config parsing
   - tokenizer encode/decode
   - SafeTensors index parsing
   - weight-name mapping
   - RoPE numeric checks
   - RMSNorm numeric checks
2. tiny fixture tests
   - local tiny model files checked into `tests/fixtures` if license permits
   - otherwise generated synthetic model fixture
3. network ignored tests
   - download tiny HF model
   - run one greedy generation
4. baseline comparison tests
   - compare against known logits from Python/Candle/Transformers for tiny model
   - compare against llama.cpp only for GGUF path

Publication-grade tests:

- deterministic generation with seed
- CPU vs Candle output tolerance
- save/load equality
- unsupported model family error
- sharded SafeTensors fixture

### E+12. Metrics

LLM metrics schema:

```json
{
  "model_id": "hf-internal-testing/tiny-random-gpt2",
  "revision": "main",
  "backend": "candle_cpu",
  "dtype": "f32",
  "parameter_count": 123456,
  "prompt_tokens": 12,
  "generated_tokens": 32,
  "load_time_ms": 120.0,
  "first_token_latency_ms": 18.2,
  "tokens_per_second": 55.4,
  "peak_memory_mb": 210.0,
  "output_sha256": "..."
}
```

Required comparisons:

- Rustral ndarray vs Rustral Candle
- Rustral tiny model vs Transformers/Candle reference logits
- Rustral GGUF path vs llama.cpp only after GGUF support exists

### E+13. Implementation Order

1. `rustral-hf` snapshot API
2. metadata-preserving `StateDict`
3. tokenizer wrapper
4. GPT-2 tiny config parser
5. GPT-2 weight mapper
6. greedy generation on ndarray
7. CLI `generate`
8. Candle backend path
9. metrics JSON output
10. Llama config parser
11. RMSNorm/RoPE/SwiGLU
12. Llama SafeTensors path
13. KV cache
14. GGUF inspect-only
15. GGUF load/execute

Stop after step 9 if the goal is an EMNLP/JOSS demo. Continue through step 13 for MLSys. Continue through step 15 only if quantized local inference becomes central to the paper.

### D2. EMNLP Paper Evidence

Must collect:

- usability: lines of user code for train/save/load/infer
- reproducibility: deterministic seed run variance over 3 runs
- correctness: CPU vs Candle output differences
- deployability: single Rust binary inference demo
- limitation: not comparable to full PyTorch ecosystem

EMNLP deadline decision:

- By **May 15, 2026**, decide if EMNLP main/ARR is viable.
- If no end-to-end NLP demo and initial metrics by May 15, do not submit main track.
- By **June 1, 2026**, decide if industry track is viable.
- If demo and reproducibility are not clean by June 1, skip EMNLP 2026 and target arXiv/JOSS/MLSys.

## Track F: JOSS-Ready Software

Goal: pass a software review, not win a systems benchmark.

### F1. Documentation

Required docs:

- README quickstart
- `ARCHITECTURE.md`
- `BENCHMARKS.md`
- `examples/README.md`
- API docs through rustdoc
- CONTRIBUTING
- SECURITY
- installation troubleshooting

JOSS paper:

- `paper/paper.md`
- `paper/paper.bib`

JOSS paper sections:

- Summary
- Statement of need
- Research use cases
- Related software
- Acknowledgements
- References

Exit criteria:

- A new user can install, run tests, run one example, and find API docs in under 30 minutes.
- Software has a release archive DOI.
- Tests and examples are documented.

## Track G: MLSys-Ready Metrics

Goal: produce a systems paper with measurable claims.

### G1. Benchmark Matrix

| Benchmark | Rustral backend | Baselines | Metrics |
|-----------|-----------------|-----------|---------|
| MLP classification | ndarray, Candle | PyTorch, Candle direct, Burn if feasible | throughput, latency, memory, accuracy |
| CNN small image task | ndarray, Candle | PyTorch, Candle direct | throughput, memory, accuracy |
| Tiny Transformer LM/classifier | ndarray, Candle | PyTorch, Candle direct | tokens/sec, perplexity/accuracy, memory |
| Checkpoint round trip | ndarray, Candle | PyTorch state dict, Candle save if available | save time, load time, size, equality |
| Backend swap | ndarray to Candle | N/A | changed LOC, output delta, runtime delta |
| Debuggability | Rustral APIs | qualitative baseline docs/code | parameter-name coverage, error quality cases |

### G2. Benchmark Harness

Tasks:

- Create `scripts/bench/run_core_benchmarks.sh`.
- Create `benchmarks/` crate or tool.
- Emit JSON:

```json
{
  "benchmark": "mlp_train",
  "backend": "candle_cpu",
  "model": "mlp_784_256_10",
  "samples_per_sec": 1234.5,
  "peak_memory_mb": 512.0,
  "loss_final": 0.21,
  "accuracy": 0.94,
  "git_sha": "..."
}
```

- Add `BENCHMARKS.md` generated from JSON.
- Record hardware and software versions.

Exit criteria:

- One command regenerates all benchmark tables.
- Results include variance over at least 3 runs.
- Claims in paper map directly to benchmark tables.

### G3. Novelty Measurements

A systems paper needs more than speed.

Measure:

- backend swap friction:
  - LOC changed
  - modules rewritten
  - output delta
- explicit context value:
  - tests showing concurrent train/inference contexts do not interfere
  - dropout mode determined only by `ForwardCtx`
- named parameter value:
  - checkpoint key stability test
  - nested model parameter inspection
  - error report includes path to offending parameter
- abstraction overhead:
  - direct Candle vs Rustral-on-Candle for same MLP/Transformer where possible

Exit criteria:

- At least 3 novelty claims have quantitative or executable evidence.

## 6-Week Critical Path

### Week 1

- Fix failing autodiff tests.
- Remove/resolve ignored plan and generated asset confusion.
- Finish named-parameter tests.
- Add trainer design doc or issue.

### Week 2

- Implement trainer over `NamedParameters`.
- Add MLP classification example.
- Add MSE and correct cross entropy tape losses.
- Add CPU gradient checks.

### Week 3

- Implement model-level state dict and `save_model` / `load_model`.
- Add train/save/load/infer example.
- Update README to use real compiling example.

### Week 4

- Build EMNLP NLP demo.
- Add evaluation script.
- Add deterministic 3-run metrics.
- Decide EMNLP main/industry viability.

### Week 5

- Add benchmark harness.
- Add PyTorch/Candle direct baselines for MLP and tiny NLP task.
- Generate first `BENCHMARKS.md`.

### Week 6

- Write `ARCHITECTURE.md`.
- Draft JOSS `paper.md`.
- Draft arXiv technical report outline.
- Tag first release candidate if tests/docs/benchmarks are clean.

## Issue Backlog

### P0

- Fix `rustral-nn --features autodiff tape` tests.
- Trainer uses `NamedParameters`.
- Whole-model save/load.
- Correct cross entropy gradients.
- End-to-end train/save/load/infer MLP.

### P1

- EMNLP NLP demo.
- Axis-aware softmax/reductions.
- Benchmark harness.
- `ARCHITECTURE.md`.
- `BENCHMARKS.md`.

### P2

- Candle direct baseline.
- PyTorch baseline.
- Burn baseline if feasible.
- Debuggability metrics.
- JOSS paper and DOI.

### P3

- `wgpu` benchmark only after stability improves.
- Distributed benchmarks only after local training story is strong.

## Go / No-Go Rules

### EMNLP 2026 Main

Go only if by May 15:

- NLP demo works.
- metrics exist.
- paper claim is narrow and defensible.
- artifact can be anonymized.

Otherwise no-go.

### EMNLP 2026 Industry

Go only if by June 1:

- NLP demo is polished.
- save/load/inference story works.
- evaluation is reproducible.
- limitations are explicit.

Otherwise no-go.

### JOSS

Go only if:

- release exists
- DOI exists
- tests pass
- docs are complete
- research use case is clear

### MLSys

Go only if:

- benchmark matrix is complete
- baselines are credible
- novelty claims are measured
- scripts reproduce tables

## Non-Goals Before First Publication

- Full PyTorch parity.
- Production multi-node distributed training.
- Stable `wgpu` claims.
- JIT/compiler story.
- Large pretrained model ecosystem.

Rustral should publish as a focused, explicit, Rust-native ML systems framework with reproducible code and honest metrics.
