# Distributed Training Infrastructure

This document describes the distributed training capabilities implemented in MNR (Modular Neural Runtime).

## Overview

The distributed training system provides:

1. **Data Parallelism** - Split batches across multiple GPUs, sync gradients
2. **Tensor Parallelism** - Split individual layers across GPUs
3. **Pipeline Parallelism** - Split model into stages across GPUs
4. **ZeRO Sharding** - Shard optimizer states and gradients for memory efficiency
5. **Distributed Checkpointing** - Save/resume training across multiple nodes
6. **Gradient Checkpointing** - Trade compute for memory

## Quick Start

### Data Parallel Training

```rust
use mnr_distributed::{DataParallelTrainer, ProcessGroup};
use mnr_optim::Adam;

// Create process group (single node, multi-GPU)
let pg = ProcessGroup::new_threaded(world_size, rank)?;

// Create optimizer and trainer
let optimizer = Adam::new(0.001);
let mut trainer = DataParallelTrainer::new(pg, optimizer);

// Train
let loss = trainer.step(&mut params, &batch, &mut loss_fn, &mut ctx)?;
```

### ZeRO Memory Optimization

```rust
use mnr_distributed::ZeroOptimizer;

// Wrap Adam with ZeRO-1 (shard optimizer states)
let adam = Adam::new(0.001);
let zero = ZeroOptimizer::new(adam, process_group, total_params);

// ZeRO-2 (also shard gradients)
let zero2 = Zero2Optimizer::new(adam, process_group, total_params);
```

### Tensor Parallelism

```rust
use mnr_distributed::TensorParallelLinear;

// Column-parallel: split output dimension
let linear = TensorParallelLinear::column_parallel(
    in_features, out_features, &process_group, &backend
)?;

// Row-parallel: split input dimension
let linear = TensorParallelLinear::row_parallel(
    in_features, out_features, &process_group, &backend
)?;
```

### Distributed Checkpointing

```rust
use mnr_distributed::DistributedCheckpointManager;

// Create manager (keeps last 3 checkpoints)
let mut manager = DistributedCheckpointManager::new(
    process_group, "/path/to/checkpoints", 3
)?;

// Save checkpoint (each rank saves its shard)
manager.save(epoch, step, &params, Some(&optimizer_checkpoint))?;

// Load checkpoint
let (loaded_step, opt_checkpoint) = manager.load(epoch, &mut params)?;
```

## Architecture

### Process Groups

```
ProcessGroup
├── rank: usize          - This process's rank
├── world_size: usize    - Total number of processes
└── CommunicationBackend
    ├── SingleProcess    - For testing/debugging
    ├── Threaded         - Shared memory for single-node multi-GPU
    └── Mpi              - Multi-node via MPI (optional feature)
```

### Data Parallelism

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  GPU 0  │    │  GPU 1  │    │  GPU 2  │    │  GPU 3  │
│ Batch 0 │    │ Batch 1 │    │ Batch 2 │    │ Batch 3 │
│ Forward │    │ Forward │    │ Forward │    │ Forward │
│  Grad   │    │  Grad   │    │  Grad   │    │  Grad   │
└────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
     └─────────────────┬──────────────────┘
                       │
                ┌────────────┐
                │ All-Reduce │
                │   (NCCL)   │
                └────────────┘
                       │
              ┌────────┴────────┐
         ┌────┴────┐       ┌────┴────┐
         │ Optimizer Step │       │ Optimizer Step │
         └─────────┘       └─────────┘
```

### Tensor Parallelism

**Column Parallel:**
```
Input [batch, in_features] → Replicated on all GPUs
                              ↓
Weight Shard [in_features, out_features/N] on GPU i
                              ↓
Output Partial [batch, out_features/N]
                              ↓
                    All-Gather → Full Output
```

**Row Parallel:**
```
Input [batch, in_features] → Split: [batch, in_features/N] on GPU i
                              ↓
Weight Shard [out_features, in_features/N] on GPU i
                              ↓
Output Partial [batch, out_features]
                              ↓
                    All-Reduce-Sum → Full Output
```

### ZeRO Sharding

**Memory Breakdown for 7B Parameter Model (f32):**

| Component | Standard | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----------|----------|--------|--------|--------|
| Parameters | 28GB | 28GB | 28GB | 3.5GB |
| Gradients | 28GB | 28GB | 3.5GB | 3.5GB |
| Optimizer States | 56GB | 7GB | 7GB | 7GB |
| **Total per GPU** | **112GB** | **63GB** | **38.5GB** | **14GB** |
| **Memory Saved** | - | ~44% | ~66% | ~88% |

*With 8 GPUs, assumes Adam optimizer (2x parameter size for moments)*

### Gradient Checkpointing

```
Standard Training:
  Layer 1 → Layer 2 → Layer 3 → Layer 4 → Loss
     ↓         ↓         ↓         ↓
  Store     Store     Store     Store   (all activations)

With Checkpointing:
  Layer 1 → Layer 2 → Layer 3 → Layer 4 → Loss
     ↓                                    (store only)
  Recompute ← Recompute ← Recompute ← Backward
     ↑         ↑         ↑
  (recompute during backward)
```

**Trade-off:** 1 extra forward pass → ~50% memory reduction

## Modules

### `tensor_parallel.rs`

- `TensorParallelLinear` - Column/row parallel linear layers
- `PipelineStage` - Pipeline parallelism stage
- `PipelineParallelTrainer` - Multi-stage pipeline trainer
- `AllGatherOp`, `AllReduceOp` - Communication primitives

### `zero.rs`

- `ZeroOptimizer` - ZeRO-1 (optimizer state sharding)
- `Zero2Optimizer` - ZeRO-2 (gradients + optimizer state)
- `ZeRoMemoryStats` - Memory usage calculator

### `checkpoint.rs`

- `DistributedCheckpointManager` - Multi-node checkpointing
- `AsyncCheckpointWriter` - Non-blocking checkpoint writes
- `CheckpointMetadata` - Checkpoint versioning info

### `lib.rs`

- `ProcessGroup` - Process communication group
- `DataParallelTrainer` - Main data parallel trainer
- `GradientAccumulator` - Micro-batch gradient accumulation

## Testing

Run distributed tests:

```bash
cargo test -p mnr-distributed
```

Key tests:
- `test_data_parallel_single_process` - Basic DP functionality
- `test_data_parallel_threaded` - Multi-threaded simulation
- `test_zero_memory_stats` - Memory savings calculation
- `test_tensor_parallel_linear_creation` - Tensor parallel layers
- `test_checkpoint_save_load` - Checkpoint round-trip
- `test_memory_savings_comparison` - ZeRO memory benefits

## Configuration Examples

### Single-Node Multi-GPU

```rust
let world_size = 4; // 4 GPUs
for rank in 0..world_size {
    let pg = ProcessGroup::new_threaded(world_size, rank)?;
    // Spawn training thread per GPU
}
```

### Multi-Node (MPI)

```rust
// Requires mpi feature: cargo build --features mpi
let pg = ProcessGroup::from_mpi_communicator(communicator)?;
```

### Mixed Parallelism

```rust
// Data + Tensor parallel for large models
let dp_pg = ProcessGroup::new_threaded(2, rank / 4)?;  // 2 nodes
let tp_pg = ProcessGroup::new_threaded(4, rank % 4)?;  // 4 GPUs per node
```

## Future Work

- [ ] Full NCCL integration for efficient all-reduce
- [ ] ZeRO-Infinity support for CPU-offloaded optimizer states
- [ ] Fully Sharded Data Parallel (FSDP)
- [ ] Pipeline parallelism with automatic stage splitting
- [ ] Communication compression (FP16 gradients, 1-bit Adam)
- [ ] Fault tolerance and elastic training
