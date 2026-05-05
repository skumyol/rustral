# Rustral

> **A Rust neural network framework for learning, research, and experiments**

<p align="center">
  <img src="docs/assets/rustral-logo-full.svg" alt="Rustral logo" width="720" />
</p>

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

## What is Rustral?

**Rustral** is a deep learning toolkit in Rust for building and training neural networks—from small tutorials to transformers and mixture-of-experts experiments.

Rustral is not trying to be a Python framework with Rust syntax. It is built around a different set of engineering bets:

- **Explicit execution state.** A `ForwardCtx` carries backend, mode, and run state through the call graph. Model code does not depend on hidden global tensor registries or implicit training modes.
- **Backend-independent model definitions.** Layers are written against small backend traits, so the same model can run on the reference CPU backend, Candle, or experimental `wgpu` without rewriting the layer stack.
- **Typed model boundaries.** The `Module` trait makes input and output contracts visible in Rust types. This is useful when experiments grow from one-off notebooks into reusable systems.
- **Inspectable internals.** Autodiff, optimizers, layers, distributed training hooks, and runtime orchestration live in focused crates. The framework is meant to be read, modified, and audited.
- **Rust-native deployment paths.** Trained logic can be compiled into ordinary Rust binaries and services, which reduces the gap between experiment code and production code.

The intended user experience is simple: define a model with normal Rust structs, choose a backend once, run forward passes through an explicit context, then keep the same model shape as you move from tutorials to training loops, checkpointing, and inference.

### Who is it for?

- **Students** learning how networks work under the hood
- **Researchers** who want reproducible, typed model code
- **Engineers** experimenting with Rust-native ML pipelines

---

## Design Philosophy

Rustral favors explicit structure over framework magic. That choice creates more visible code than a notebook-first API, but it also gives senior engineers and ML researchers stronger control over correctness, reproducibility, and system behavior.

### The Core Tradeoff

Most deep learning frameworks optimize for interactive velocity: global tensor state, dynamic graph construction, runtime shape failures, and large native backends hidden behind Python objects. That model is productive, but it makes many production and research questions harder to reason about:

- Which backend owns this tensor?
- Which parameters are trainable in this pass?
- Where does graph lifetime begin and end?
- What does a model need to run outside the training process?
- Can this layer be reused without inheriting hidden runtime state?

Rustral makes those boundaries explicit. The result is a framework where model code is more portable across backends, easier to test at crate boundaries, and easier to integrate into Rust systems that already care about memory safety, concurrency, deployment, and reproducible builds.

### Ease Of Use, Not Hidden State

Rustral treats ease of use as a systems problem: the common path should be short, but the important state should still be visible when something goes wrong.

| User need | Rustral design choice |
|-----------|-----------------------|
| Build a layer without boilerplate | Builders such as `LinearBuilder` initialize backend-owned parameters for you. |
| Run the same model on another backend | Model code depends on `Backend` and `TensorOps`, not a concrete tensor library. |
| Know whether a pass is training or inference | `ForwardCtx` carries `Mode::Train` or `Mode::Inference` explicitly. |
| Compose larger models from smaller pieces | `Module` gives each layer a typed forward contract. |
| Inspect, test, or replace internals | Autodiff, optimizers, runtime, data, and backends are separate crates. |
| Move toward deployment | Rust binaries can embed the same model logic used in experiments. |

The design is intentionally not "magic-free at any cost." The goal is to put convenience at the edges through builders, trainers, examples, and checkpoint helpers while keeping ownership, backend selection, and execution mode clear in the core API.

### Practical Advantages

| Advantage | Why it matters |
|-----------|----------------|
| Explicit `ForwardCtx` | Training/inference mode, backend access, and run state are visible at every forward pass. |
| Small backend contract | New execution engines can be added without changing high-level model definitions. |
| Reference CPU backend | Correctness can be tested on a simple backend before optimizing for CUDA or cross-platform GPU paths. |
| Focused crates | Researchers can study or replace autodiff, optimizers, layers, runtime, data, or distributed pieces independently. |
| Rust ownership model | Parameter ownership, borrowing, and lifetimes are checked by the compiler instead of convention. |
| Single-binary integration | Inference and training utilities can be embedded into normal Rust services, CLIs, and pipelines. |

### What Rustral Optimizes For

- Clear extension points for new layers, backends, optimizers, and training loops.
- Reproducible experiments that can move from research code into systems code.
- Debuggable internals for people who want to understand or modify the framework.
- Correctness-first CPU execution with optional faster backend paths.
- A workspace architecture where each subsystem can be tested and evolved independently.

Rustral is currently best viewed as a research and systems framework: useful for learning, experiments, backend work, and Rust-native ML infrastructure. If you need the largest pretrained model ecosystem today, Python frameworks remain the practical default. If you want a framework whose execution model is explicit and whose internals are approachable, Rustral is designed for that.

The current improvement roadmap is tracked in [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md).

## Quick Start (5 Minutes)

### 1. Installation

```bash
# Install Rust (if you haven't)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone Rustral
git clone https://github.com/skumyol/rustral.git
cd rustral

# Build everything (takes ~5 min on modern hardware)
cargo build --workspace
```

### 2. Run the Test Suite

```bash
./run_tests.sh
```

This runs formatting checks, clippy lints, the full workspace build, and tests for all **16** workspace crates (`rustral-wgpu-backend` may warn on some GPU/driver stacks).

### 3. Run Your First Example: XOR

```bash
cd examples && cargo run --bin xor
```

**What it does:** A 2-layer network learns that `0 XOR 0 = 0`, `0 XOR 1 = 1`, `1 XOR 0 = 1`, `1 XOR 1 = 0`.

**Why this matters:** XOR isn't linearly separable—you need at least one hidden layer. This proves the network actually learns something non-trivial.

### 4. Run a Transformer Example

```bash
cargo run -p rustral-nn --example linear_readout
cargo run -p rustral-nn --example transformer_bert_encoder
```

---

## Architecture

Rustral is organized as a workspace of focused crates. Each crate handles one piece of the puzzle:

<p align="center">
  <img src="docs/assets/system_diagram_1.png" alt="Rustral system diagram" width="920" />
</p>

```mermaid
flowchart TD
    app["Examples and applications"]
    nn["rustral-nn<br/>layers and model composition"]
    core["rustral-core<br/>Backend, TensorOps, Module, ForwardCtx, Parameter"]
    autodiff["rustral-autodiff<br/>tape and gradients"]
    optim["rustral-optim<br/>SGD, Adam, AdamW, schedules"]
    runtime["rustral-runtime<br/>trainers and inference orchestration"]
    dist["rustral-distributed<br/>DP, TP, PP, ZeRO, FSDP-style APIs"]
    data["rustral-data<br/>batches, shuffling, transforms"]
    io["rustral-io<br/>SafeTensors save/load"]
    metrics["rustral-metrics<br/>loss, accuracy, throughput"]
    autotune["rustral-autotuner<br/>backend tuning"]
    symbolic["rustral-symbolic<br/>vocabularies and labels"]
    hf["rustral-hf<br/>Hugging Face helpers"]
    ndarray["rustral-ndarray-backend<br/>reference CPU"]
    candle["rustral-candle-backend<br/>CPU/CUDA via Candle"]
    wgpu["rustral-wgpu-backend<br/>experimental Vulkan/Metal/DX12"]

    app --> nn
    nn --> core

    runtime --> nn
    runtime --> autodiff
    runtime --> optim
    runtime --> io
    runtime --> metrics
    dist --> core

    nn --> data
    nn --> symbolic
    hf --> io
    autotune --> wgpu

    core --> ndarray
    core --> candle
    core --> wgpu
```

```text
examples and applications
  -> rustral-nn
     neural network layers: Linear, Conv2d, LSTM, Transformer, Attention, MoE
  -> rustral-core
     TensorOps, Backend, Parameter, Module, ForwardCtx
  -> training and runtime crates
     rustral-autodiff, rustral-optim, rustral-runtime, rustral-distributed
  -> backend crates
     rustral-ndarray-backend, rustral-candle-backend, rustral-wgpu-backend
```

| Layer | Crates | Responsibility |
|-------|--------|----------------|
| Model surface | `examples`, `rustral-nn` | User-facing layers, model composition, transformer and vision examples |
| Core contracts | `rustral-core` | Tensor operations, backend trait, parameters, module trait, explicit forward context |
| Training stack | `rustral-autodiff`, `rustral-optim`, `rustral-runtime` | Gradients, optimizers, mixed precision hooks, trainer and inference orchestration |
| Scaling stack | `rustral-distributed` | Data, tensor, pipeline, sequence, context, ZeRO, and FSDP-style parallelism APIs |
| Execution backends | `rustral-ndarray-backend`, `rustral-candle-backend`, `rustral-wgpu-backend` | Reference CPU, optimized CPU/CUDA, and experimental cross-platform GPU execution |
| Supporting crates | `rustral-data`, `rustral-io`, `rustral-metrics`, `rustral-autotuner`, `rustral-symbolic`, `rustral-hf` | Data loading, checkpoints, metrics, backend tuning, vocabularies, and Hugging Face helpers |

### Crate-by-Crate Breakdown

| Crate | What It Does |
|-------|-------------|
| `rustral_core` | **Tensors, backends, and the `Module` trait**—the foundation everything builds on |
| `rustral_nn` | **Neural network layers**: Linear, Conv2d, LSTM, Transformer, Attention, MoE, Flash Attention |
| `rustral_autodiff` | **Automatic differentiation**—computes gradients via reverse-mode autodiff |
| `rustral_optim` | **Optimizers**: SGD, Adam, AdamW + learning rate schedules + mixed precision |
| `rustral_distributed` | **Parallel training APIs** (ZeRO/FSDP-style sharding, threading/MPI hooks—see docs for current scope) |
| `rustral_data` | **Data loading**: batching, shuffling, transforms |
| `rustral_io` | **Save/load models** in SafeTensors format |
| `rustral_wgpu_backend` | **Experimental GPU** via WebGPU (Vulkan/Metal/DX12)—see **Backends** |
| `rustral_candle_backend` | **Optimized CPU/CUDA backend** using candle-core (up to ~20x faster than ndarray on CPU) |
| `rustral_ndarray_backend` | **Reference CPU backend** for correctness testing |
| `rustral_metrics` | **Track training**: loss curves, accuracy, throughput |
| `rustral_autotuner` | **Automatically find the fastest GPU settings** |
| `rustral_symbolic` | **Vocabulary and label dictionaries** for NLP tasks |
| `rustral_bench` | **Criterion benchmarks** for core operations |
| `rustral_runtime` | **Training/inference orchestration** (trainers, pools) |
| `rustral_hf` | **Optional Hugging Face Hub** helpers for weights |

### Backends

Rustral is designed to be backend-agnostic. You can swap backends without changing your model code:

```rust
use rustral_ndarray_backend::CpuBackend;      // Reference CPU backend
use rustral_candle_backend::CandleBackend;    // Optimized CPU/CUDA
use rustral_wgpu_backend::WgpuBackend;        // Cross-platform GPU

// All three work with the same model code
let backend = CpuBackend::default();
// let backend = CandleBackend::cpu();
// let backend = WgpuBackend::new_sync()?;
```

| Backend | Hardware | Best For | Status |
|---------|----------|----------|--------|
| `rustral-ndarray-backend` | CPU | Reference/testing, correctness baselines | Stable |
| `rustral-candle-backend` | CPU, CUDA (optional) | Production training, large models | Stable |
| `rustral-wgpu-backend` | GPU (Vulkan/Metal/DX12) | Experiments / inference prototyping | **Experimental** |

\* **Experimental:** uses WGSL compute for matmul, element-wise ops, etc. On some Linux + NVIDIA stacks, full test binaries have aborted during teardown (allocator/driver interaction with `wgpu` 0.19). CI runs these tests as non-blocking; use Candle for reliable GPU-ish throughput until `wgpu` is upgraded.

---

## Core Concepts

### 1. Tensors

A **tensor** is a multi-dimensional array of numbers:

- **0D**: Scalar → `5.0`
- **1D**: Vector → `[0.1, 0.5, 0.9]`
- **2D**: Matrix → `[[1, 2], [3, 4]]`
- **3D+**: Stack of matrices → batch of images `[batch, channels, height, width]`

```rust
let data = vec![1.0, 2.0, 3.0,   // Sample 1
                4.0, 5.0, 6.0];  // Sample 2
let tensor = backend.tensor_from_vec(data, &[2, 3]).unwrap();
```

### 2. The Module Trait

Every layer implements `Module`, a simple contract:

```rust
pub trait Module<B: Backend> {
    type Input;
    type Output;
    fn forward(&self, input: Self::Input, ctx: &mut ForwardCtx<B>) -> Result<Self::Output>;
}
```

This means you can swap a `Linear` for a `Conv2d` and the rest of your code doesn't change.

### 3. Forward Context

Unlike PyTorch's hidden global state, rustral makes everything explicit:

```rust
let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
let output = layer.forward(input, &mut ctx)?;
```

You can run two models simultaneously without interference.

### 4. Bulk Tensor Access

For GPU backends, reading data element-by-element is slow (each call may round-trip to the GPU). Use `tensor_to_vec` for bulk reads:

```rust
// Slow on GPU: one roundtrip per element
for i in 0..n {
    let val = ops.tensor_element(&tensor, i)?;
}

// Fast on GPU: single roundtrip for all elements
let values = ops.tensor_to_vec(&tensor)?;
```

This is especially important for layers like `Conv2d` and normalization that need to access many values.

---

## Examples Gallery

Examples live in the `examples/` directory, which is a separate workspace with its own `Cargo.toml`.

### Basics

| Example | Run Command |
|---------|-------------|
| XOR | `cd examples && cargo run --bin xor` |
| MNIST | `cd examples && cargo run --bin mnist` |
| Linear Regression | `cd examples && cargo run --bin train_demo` |
| Character RNN | `cd examples && cargo run --bin char_rnn` |

### Vision

| Example | Run Command |
|---------|-------------|
| Building Blocks | `cd examples && cargo run --example building_blocks` |
| ResNet | `cd examples && cargo run --example resnet_image_classification` |
| Diffusion | `cd examples && cargo run --example diffusion_model` |

### NLP

| Example | Run Command |
|---------|-------------|
| BERT | `cd examples && cargo run --example bert_fine_tuning` |
| GPT | `cd examples && cargo run --example gpt_training` |

### Advanced

| Example | Run Command |
|---------|-------------|
| MoE | `cd examples && cargo run --example moe_training` |
| Custom Layer | `cd examples && cargo run --example custom_layer` |

### In-Package Examples

Some crates also provide standalone examples:

| Example | Run Command |
|---------|-------------|
| Linear Readout | `cargo run -p rustral-nn --example linear_readout` |
| BERT Encoder | `cargo run -p rustral-nn --example transformer_bert_encoder` |
| GPT Decoder | `cargo run -p rustral-nn --example transformer_gpt_decoder` |
| Candle Benchmark | `cargo run -p rustral-candle-backend --example benchmark` |

---

## Building a Model

The model surface is ordinary Rust: store layers as fields, initialize them from a backend, and thread a `ForwardCtx` through the forward pass. That keeps the easy path readable while still making backend ownership and execution mode explicit.

```rust
use rustral_core::{Backend, ForwardCtx, Module, Mode};
use rustral_nn::{Linear, LinearBuilder, Conv2d, Conv2dConfig, max_pool2d};

struct MyModel<B: Backend> {
    conv1: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> MyModel<B> {
    fn new(backend: &B) -> Self {
        Self {
            conv1: Conv2dConfig::new(32, 3, 3).build(backend),
            fc1: LinearBuilder::new(32 * 7 * 7, 128).build(backend),
            fc2: LinearBuilder::new(128, 10).build(backend),
        }
    }

    fn forward(&self, image: &B::Tensor, ctx: &mut ForwardCtx<B>) -> B::Tensor {
        let ops = ctx.backend().ops();
        let x = self.conv1.forward(image.clone(), ctx).unwrap();
        let x = ops.relu(&x).unwrap();
        let x = max_pool2d(&x, 2, 2, 2, 2, true, ops).unwrap();
        let x = ops.reshape(&x, &[1, 32 * 7 * 7]).unwrap();
        let x = self.fc1.forward(x, ctx).unwrap();
        let x = ops.relu(&x).unwrap();
        self.fc2.forward(x, ctx).unwrap()
    }
}
```

---

## Advanced Features

### Multi-GPU / distributed-style training

v0.1.0 focuses on **correct, inspectable building blocks**. `ProcessGroup::new_threaded` and the data-parallel / ZeRO-style trainers run as **single-process simulations** (threads) for learning and tests—not a drop-in replacement for multi-node PyTorch yet. Optional `mpi` / `nccl` **feature flags** are hooks for future work; they do not provide a full production cluster stack out of the box.

```rust
use rustral_distributed::{DataParallelTrainer, ProcessGroup};
use rustral_optim::Adam;

let pg = ProcessGroup::new_threaded(8, rank)?;
let trainer = DataParallelTrainer::new(pg, Adam::new(0.001));
```

### Mixed Precision Training

```rust
use rustral_optim::{MixedPrecisionOptimizer, DType};

let optimizer = MixedPrecisionOptimizer::new(Adam::new(0.001))
    .with_dtype(DType::Float16)
    .with_loss_scale(1024.0);
```

### Candle Backend (Recommended for Production)

The candle backend uses the [candle-core](https://github.com/huggingface/candle) library for highly optimized CPU (and optional CUDA) execution:

```rust
use rustral_candle_backend::CandleBackend;

let backend = CandleBackend::cpu();
// Or with CUDA (requires nvcc / CUDA toolkit):
// let backend = CandleBackend::cuda(0)?;
```

On large linear layers, candle can be **up to ~20x faster** than the ndarray backend on CPU.

### Flash Attention

```rust
use rustral_nn::{FlashAttention, SelfAttentionConfig};

let config = SelfAttentionConfig::new(768, 12);
let flash_attn = FlashAttention::new(&backend, config, 42)?;
```

### Mixture of Experts (MoE)

```rust
use rustral_nn::{ExpertLayer, MoEConfig};

let config = MoEConfig::new(512, 64, 2048, 2);
let moe = ExpertLayer::new(&backend, config, 42)?;
```

---

## Development Status

- **Library tests:** ~**639** passed across workspace crates with `cargo test --workspace --exclude rustral-wgpu-backend` (exact count changes as tests land).
- **`rustral-wgpu-backend`:** run separately; CI executes it with `continue-on-error` because some platforms abort during teardown.
- **Examples:** built in a nested workspace under `examples/` (`cargo build --manifest-path examples/Cargo.toml --workspace`).

Run formatting, clippy, and tests locally:

```bash
./run_tests.sh
# stricter CI parity:
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings \
  && cargo test --workspace --exclude rustral-wgpu-backend
```

---

## Platform Notes

### Linux + NVIDIA GPU

- **Candle CPU**: Works out of the box, highly optimized.
- **Candle CUDA**: Requires `nvcc` and the CUDA toolkit. Enable with `--features cuda`.
- **wgpu**: Uses Vulkan. Some test binaries may segfault on process exit due to a known wgpu 0.19 + NVIDIA driver cleanup bug. Individual tests pass. Upgrading to wgpu 0.20+ would resolve this.

### macOS

- **Candle CPU**: Works out of the box.
- **Candle Metal**: Supported by candle-core (check upstream for feature flags).
- **wgpu**: Uses Metal natively.

### Windows

- **Candle CPU**: Works out of the box.
- **wgpu**: Uses DirectX 12 natively.

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| [`docs/architecture.md`](docs/architecture.md) | System design and crate relationships |
| [`docs/api-signatures.md`](docs/api-signatures.md) | Public function signatures |
| [`docs/master-plan.md`](docs/master-plan.md) | Feature roadmap |
| [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) | Concrete engineering plan for closing current abstraction and usability gaps |
| [`docs/SECURITY.md`](docs/SECURITY.md) | Security guidelines |
| API Docs (rustdoc) | `cargo doc --open` |

### Rust API docs (rustdoc)

Rust documentation is generated using **rustdoc** (via `cargo doc`), which converts `///` comments in source code into HTML.

- **Generate and open (includes deps)**: `cargo doc --open`
- **Document private items**: `cargo doc --document-private-items`
- **All features**: `cargo doc --all-features`

#### Writing documentation

- **Outer doc comments (`///`)**: placed above items (functions, structs, etc.)
- **Inner doc comments (`//!`)**: placed inside a file/module to document the crate or module
- **Markdown**: use Markdown in comments for formatting, including code blocks
- **Examples**: include `/// Examples` blocks; rustdoc can run them as doctests

---

## Contributing

We welcome contributors at all levels! Good first issues:

1. **Add an example** showing a technique not yet covered
2. **Improve documentation**—explain a concept in simpler terms
3. **Add tests** for edge cases
4. **Fix compiler warnings**—good for learning the codebase

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## License

Rustral is dual-licensed under:

- **[MIT](LICENSE-MIT)** — short permissive license
- **[Apache License 2.0](LICENSE-APACHE)** — same intent, different legal wording

You may use either license.

---

## Acknowledgments

Rustral draws ideas from the Rust ML ecosystem and aims to stay **transparent and hackable** alongside crates like Candle and Burn.

**Key design influences:**
- PyTorch's eager execution and intuitive API
- JAX's functional purity and composability
- Rust's ownership model for memory safety
- Hugging Face Candle's efficient CPU/CUDA kernels
