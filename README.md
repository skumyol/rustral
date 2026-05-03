# Modular Neural Runtime

## Design Principles

- Explicit state beats hidden global state.
- Module boundaries should be small, typed, and testable.
- Parameter ownership belongs to model structs, not global registries.
- Training and inference parallelism should be visible in the API.
- Serialization and configuration should be deterministic.
- Backends are swappable; modules should not depend on one tensor library.

## Workspace

```text
crates/core             Backend traits, tensors, parameters, module contracts
crates/ndarray-backend  Small CPU reference backend for tests and examples
crates/wgpu-backend     GPU compute backend with WGSL shaders
crates/autodiff         Reverse-mode automatic differentiation with tape
crates/optim            Optimizers (SGD, Adam) with checkpointing
crates/nn               Linear, embedding, readout, MLP, transformers
crates/distributed      Multi-GPU training: data parallel, tensor parallel, ZeRO
crates/data             Dataset and DataLoader abstractions
crates/io               Serialization (safetensors format)
crates/symbolic         Vocabulary and label dictionaries
crates/bench            Benchmarks (matmul, conv2d, attention)
examples/               Minimal examples (XOR, MNIST, RNN, training demo)
docs/                   Architecture notes and improvement roadmap
legacy/                 Place for historical source snapshots
```

## Quick Example

```rust
use mnr_core::{Module, ForwardCtx, Mode};
use mnr_ndarray_backend::CpuBackend;
use mnr_nn::{Linear, LinearConfig};

let backend = CpuBackend::default();
let mut ctx = ForwardCtx::new(&backend, Mode::Inference);
let weight = backend.normal_parameter("linear.weight", &[2, 4], 7, 0.1).unwrap();
let bias = backend.normal_parameter("linear.bias", &[2], 8, 0.0).unwrap();
let layer = Linear::<CpuBackend>::from_parameters(
    LinearConfig { in_dim: 4, out_dim: 2, bias: true },
    weight,
    Some(bias),
);
let x = backend.tensor_from_vec(vec![1.0, 0.0, 0.5, -1.0], &[4]).unwrap();
let y = layer.forward(x, &mut ctx).unwrap();
```

## Multi-GPU Training Example

```rust
use mnr_distributed::{DataParallelTrainer, ProcessGroup, ZeroOptimizer};
use mnr_optim::Adam;

// Create process group for 8 GPUs
let pg = ProcessGroup::new_threaded(8, rank)?;

// ZeRO optimizer: 8x memory reduction
let adam = Adam::new(0.001);
let optimizer = ZeroOptimizer::new(adam, pg.clone(), total_params);

// Data parallel trainer
let mut trainer = DataParallelTrainer::new(pg, optimizer);

// Train: gradients are automatically synchronized across GPUs
let loss = trainer.step(&mut params, &batch, &mut loss_fn, &mut ctx)?;
```

## Tensor Parallelism Example

For layers that don't fit on a single GPU, use tensor parallelism:

```rust
use mnr_distributed::TensorParallelLinear;

// Column-parallel: split output dimension across GPUs
// Each GPU handles 1/8 of the output features
let linear = TensorParallelLinear::column_parallel(
    4096,    // in_features
    32768,   // out_features (too large for one GPU)
    &pg,     // process group with 8 GPUs
    &backend
)?;

// Row-parallel: split input dimension
let linear = TensorParallelLinear::row_parallel(
    32768,   // in_features
    4096,    // out_features
    &pg,
    &backend
)?;
```

## Legacy Mapping

See [`docs/legacy-mapping.md`](docs/legacy-mapping.md). The uploaded C++ wrapper inspired these concepts:

- expression/tensor helpers → `Backend`, `TensorOps`, `ForwardCtx`
- dense layer → `mnr_nn::Linear`
- dictionary and labels → `mnr_symbolic::Vocabulary`
- readout model → `mnr_nn::Readout`
- training utility → `mnr_runtime::ParallelTrainer`

## Status

**Production-Ready Infrastructure:** The framework now supports end-to-end training with distributed multi-GPU capabilities.

- ✅ **Core:** Backend traits, tensors, autodiff with correct gradients
- ✅ **NN Modules:** Transformers, CNNs, RNNs, normalization, attention
- ✅ **GPU:** WGSL compute shaders for element-wise ops and matmul
- ✅ **Optimizers:** SGD, Adam with state serialization
- ✅ **Distributed:** Data parallelism, tensor parallelism, ZeRO sharding
- ✅ **Checkpointing:** Distributed save/resume across multiple nodes

The reference CPU backend (`mnr-ndarray-backend`) is suitable for testing and small models. For production training, use the GPU backend (`mnr-wgpu-backend`) or plug in your own backend implementing the `Backend` trait.

See [`docs/improvement-plan.md`](docs/improvement-plan.md) for the complete roadmap and [`docs/distributed_training.md`](docs/distributed_training.md) for distributed training details.

## Testing

Run all tests:
```bash
cargo test --workspace
```

Run specific crate tests:
```bash
cargo test -p mnr-core
cargo test -p mnr-autodiff
cargo test -p mnr-distributed  # Distributed training tests
cargo test -p mnr-wgpu-backend # GPU backend tests
```

Run benchmarks:
```bash
cargo bench -p mnr-bench
```

## API Inventory

Every public function signature is documented in the source with rustdoc comments and summarized in [`docs/api-signatures.md`](docs/api-signatures.md).
