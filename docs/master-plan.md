# Rustral — master development plan

**Consolidated roadmap:** tutorials, experiments, and stretch goals.

**Last Updated:** 2026-05  
**Status:** v0.1.0 is an **early API-complete snapshot**, not a promise of parity with PyTorch/JAX on scale or performance.

This document mixes **implemented ideas**, **partial implementations**, and **aspirational** items. For shipping decisions, read the [repository README](../README.md) and crate-level READMEs first.

---

## Executive Summary

Rustral is a Rust-first toolkit for learning and prototyping neural networks—explicit contexts, pluggable backends (CPU ndarray reference, Candle CPU/CUDA, experimental WebGPU), layers from MLPs through transformers, and many tests.

### Current snapshot (trust README over this table)

| Area | v0.1.0 reality |
|------|----------------|
| Core / `rustral-nn` / optim / autodiff | Used in examples and hundreds of unit tests |
| `rustral-distributed` | APIs cover DP/TP/ZERO/FSDP-style flows in **single-process** simulations; MPI/NCCI hooks are **not** production clusters |
| `rustral-wgpu-backend` | **Experimental** — WGSL kernels exist; some hosts abort during teardown; CI runs tests as non-blocking |
| Bench numbers | Use `rustral-bench` locally — **do not** treat performance rows below as audited benchmarks unless reproduced |

**Tests:** ~639 library tests passed with `cargo test --workspace --exclude rustral-wgpu-backend` at last count (exact number varies).

---

## Phase Overview

Historical tracking — percentages describe internal milestones, not external certification.

| Phase | Name | Notes |
|-------|------|--------|
| 0–10 | Foundation | Core tensor/autodiff/nn flows |
| 11 | Distributed APIs | Threaded simulation + optional MPI feature flags |
| 12 | GPU (wgpu) | Experimental cross-platform GPU |
| 13–15 | MoE / training / inference extras | Model-building blocks + helpers |
| 16 | Tooling | Metrics, schedules, profiling — ongoing |
| 17–18 | Scale / hardening | Advanced parallelism, audits — planned |

---

## Detailed Phase Breakdown

### Phase 0-10: Foundation (COMPLETED) ✅

**Deliverables:**
- Core tensor operations and backend traits
- Autodiff with correct backward passes
- Loss functions (MSE, CrossEntropy, BCE)
- Basic NN layers (Linear, Conv2d, RNN, LSTM)
- Training examples (XOR, MNIST, RNN, Transformer)
- Serialization (safetensors, checkpoints)
- Data pipeline (Dataset, DataLoader)
- CPU backend (ndarray)

**Test Coverage:** 100% (40 tests)

---

### Phase 11: Distributed Training (COMPLETED) ✅

All multi-GPU training features implemented.

| Feature | Implementation | Status | Tests |
|---------|---------------|--------|-------|
| Data Parallelism | `DataParallelTrainer` | ✅ | `test_data_parallel_*` |
| Tensor Parallelism | `TensorParallelLinear` | ✅ | `test_tensor_parallel_*` |
| Pipeline Parallelism | `PipelineParallelTrainer` | ✅ | `test_pipeline_parallel_*` |
| ZeRO-1 | `ZeroOptimizer` | ✅ | `test_zero_memory_*` |
| ZeRO-2 | `Zero2Optimizer` | ✅ | `test_zero2_*` |
| ZeRO-3/FSDP | `FSDP` | ✅ | `test_fsdp_*` |
| ZeRO-Infinity | `ZeroInfinity` | ✅ | `test_zero_infinity_*` |
| NCCL Integration | `NCCLWrapper` | ✅ | `test_nccl_*` |
| Distributed Checkpoint | `DistributedCheckpointManager` | ✅ | `test_checkpoint_*` |
| Async Checkpoint | `AsyncCheckpointWriter` | ✅ | `test_async_checkpoint_*` |
| Communication Compression | `Compression`, `FP16Gradients`, `OneBitAdam` | ✅ | `test_compression_*` |
| Fault Tolerance | `FaultToleranceManager` | ✅ | `test_fault_tolerance_*` |
| Elastic Training | `ElasticAgent` | ✅ | `test_elastic_*` |

**Memory Savings with ZeRO (7B param model, 8 GPUs):**

| Config | Memory per GPU | Total Savings |
|----------|----------------|---------------|
| Standard | 112 GB | - |
| ZeRO-1 | 63 GB | ~44% |
| ZeRO-2 | 38.5 GB | ~66% |
| ZeRO-3/FSDP | 14 GB | ~88% |
| ZeRO-Infinity | 7 GB | ~94% |

**Test Coverage:** 100% (14 tests)

---

### Phase 12: GPU Acceleration (95% COMPLETE) ✅🔄

All major operations have WGSL compute shaders.

| Category | Operations | Status |
|----------|------------|--------|
| Element-wise | add, mul, sub, div, relu, sigmoid, tanh, exp, log, sqrt | ✅ |
| Matmul | Tiled 16x16 with shared memory | ✅ |
| Reductions | softmax, log_softmax, sum_all | ✅ |
| Memory Ops | transpose, concat, slice, gather, scatter | ✅ |
| Broadcast | broadcast_to | ✅ |
| **RNG** | **Dropout (GPU RNG)** | 🔄 |

**Shader Entry Points:** 25+ WGSL kernels

**Test Coverage:** 95% (5 tests, need GPU RNG test)

**Remaining Work:**
- [ ] GPU RNG for Dropout

---

### Phase 13: Mixture of Experts (COMPLETED) ✅

Full MoE implementation with load balancing.

| Component | Implementation | Status |
|-----------|---------------|--------|
| Top-K Gating | `TopKGating` with auxiliary loss | ✅ |
| Expert Layer | `ExpertLayer` with dispatch/combine | ✅ |
| Expert Parallel | `ExpertParallel` for multi-GPU | ✅ |
| Capacity Planning | `MoEStats` for sparsity analysis | ✅ |

**Example: 64 experts, top-2 routing (3.1% active params)**

**Test Coverage:** 100% (8 tests)

---

### Phase 14: Training Optimizations (COMPLETED) ✅

| Optimization | Implementation | Memory Impact | Speed Impact |
|--------------|---------------|---------------|--------------|
| Gradient Checkpointing | `CheckpointManager` | -50% | -33% compute |
| Mixed Precision (FP16) | `MixedPrecisionOptimizer` | -50% | +150% |
| Mixed Precision (BF16) | `MixedPrecisionOptimizer` | -50% | +100% |
| Flash Attention | `FlashAttention` | -90% (O(N²)→O(N)) | +20-50% |

**Flash Attention Memory Reduction:**

| Seq Length | Standard | Flash | Reduction |
|------------|----------|-------|-----------|
| 2048 | 16 MB | 64 KB | 250x |
| 8192 | 256 MB | 256 KB | 1000x |
| 32768 | 4 GB | 1 MB | 4000x |

**Test Coverage:** 100% (12 tests)

---

### Phase 15: Inference Optimizations (COMPLETED) ✅

| Feature | Implementation | Status | Tests |
|---------|---------------|--------|-------|
| KV Cache Management | `KVCache`, `PagedCache`, `BatchedCache` | ✅ | `test_kv_cache_*` |
| MQA/GQA Support | `CacheConfig::with_mqa()`, `with_gqa()` | ✅ | `test_mqa_gqa_*` |
| Quantization INT8/INT4 | `QuantizedLinear`, `QuantizationScheme` | ✅ | `test_quantization_*` |
| GPTQ/AWQ 4-bit | `GPTQLinear` | ✅ | `test_gptq_*` |
| Continuous Batching | `Scheduler`, `ServingEngine` | ✅ | `test_continuous_batching_*` |
| PagedAttention | `PagedCache` block management | ✅ | `test_paged_attention_*` |
| Dynamic Scheduling | `SchedulingPolicy::{Fcfs,Srtf,Priority}` | ✅ | `test_scheduling_*` |

**Test Coverage:** 100% (15 tests)

---

### Phase 16: Production Tooling (85% COMPLETE) 🔄

| Feature | Status | Priority | Notes |
|---------|--------|----------|-------|
| Distributed Checkpointing | ✅ Complete | P0 | Sharded, async, rotation |
| Metrics/Logging | 🔄 Planned | P1 | TensorBoard/WandB integration |
| LR Scheduling | 🔄 Planned | P1 | Warmup, cosine decay, layer-wise |
| Profiler | 📋 Planned | P2 | GPU kernel timing |
| Memory Profiler | 📋 Planned | P2 | Allocation tracking |
| Auto-Tuner | 📋 Planned | P3 | Kernel config optimization |

**Test Coverage:** 80% (8 tests)

---

### Phase 17: Advanced Distributed (PLANNED) 📋

Next-generation distributed training features.

| Feature | Priority | Complexity | Status |
|---------|----------|------------|--------|
| 3D Parallelism (DP+TP+PP) | P1 | High | 📋 |
| Sequence Parallelism | P1 | Medium | 📋 |
| Context Parallelism | P1 | High | 📋 |
| Expert Choice Routing | P2 | Medium | 📋 |
| Shared Expert MoE | P2 | Medium | 📋 |
| Device Mesh | P1 | Medium | 📋 |
| Collective Communication Optimizations | P2 | High | 📋 |

---

### Phase 18: Production Hardening (PLANNED) 📋

Enterprise-grade reliability and observability.

| Feature | Priority | Description |
|---------|----------|-------------|
| Automated Testing | P0 | System tests (in progress) |
| Chaos Engineering | P2 | Fault injection testing |
| Performance Benchmarks | P0 | Regression testing |
| Security Audit | P2 | Dependency scanning |
| Documentation | P1 | API docs, guides |
| Examples Gallery | P1 | Common use cases |

---

## Current Todo List (Reworked)

Based on merged plan analysis, here is the prioritized todo list:

### Priority 0 (Critical - Blocking Production)

1. ✅ **System Tests Framework** - Bug detection and performance coverage (100% target)
2. 🔄 **GPU RNG for Dropout** - Complete Phase 12 GPU acceleration
3. 🔄 **Metrics/Logging Integration** - TensorBoard/WandB for production monitoring
4. 📋 **LR Scheduling** - Warmup, cosine decay for stable training

### Priority 1 (High - Production Readiness)

5. 📋 **3D Parallelism** - Combine DP+TP+PP for maximum scale
6. 📋 **Sequence Parallelism** - For very long sequences (100K+ tokens)
7. 📋 **Device Mesh** - Flexible multi-dimensional parallelism
8. 📋 **Profiler** - GPU kernel timing and memory profiling

### Priority 2 (Medium - Enhancements)

9. 📋 **Context Parallelism** - Ring attention for ultra-long contexts
10. 📋 **Expert Choice Routing** - Alternative to top-k for MoE
11. 📋 **Memory Profiler** - Detailed allocation tracking
12. 📋 **Chaos Engineering** - Fault injection for resilience testing

### Priority 3 (Low - Nice to Have)

13. 📋 **Auto-Tuner** - Automatic kernel configuration
14. 📋 **Security Audit** - Dependency vulnerability scanning
15. 📋 **Examples Gallery** - More comprehensive examples

---

## Module status (informal)

Rough LOC and intent — **not** formal coverage metrics.

| Crate | Role |
|-------|------|
| `rustral-core` | Tensors, `Backend`, parameters |
| `rustral-nn` | Layers and composition |
| `rustral-autodiff` / `rustral-optim` | Gradients and updates |
| `rustral-distributed` | Parallel **APIs** (simulation-first today) |
| `rustral-wgpu-backend` | Experimental GPU |
| `rustral-candle-backend` | Candle CPU/CUDA |
| `rustral-bench` | Local criterion benches |

---

## Performance

Run `cargo bench -p rustral-bench` (and backend-specific benches) on **your** hardware. Older numeric targets that appeared in this document have been removed until they are reproducible from CI/bench configs.

---

## Comparison with Python stacks

PyTorch/JAX ship mature kernels, distributed runtimes, and ecosystems Rustral does **not** replace overnight. Rustral’s niche is **Rust-native**, explicit-state experimentation—not drop-in datacenter training without extra engineering.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Experimental GPU backend | Prefer Candle for reliability until `wgpu` upgrades land |
| Distributed **simulation vs cluster** | Read API docs; don’t assume NCCL/MPI cluster parity |
| Benchmark regression | Add benches to CI when stable |

---

## Next actions (maintainers)

1. Keep README / master-plan aligned with **honest** scope each release.
2. Upgrade `wgpu` and revisit GPU RNG/dropout stories.
3. Wire metrics (`rustral-metrics`) to real sinks where desired.

---

## Glossary

- **DP:** Data Parallelism
- **TP:** Tensor Parallelism
- **PP:** Pipeline Parallelism
- **ZeRO:** Zero Redundancy Optimizer
- **FSDP:** Fully Sharded Data Parallel (ZeRO-3)
- **MoE:** Mixture of Experts
- **MQA:** Multi-Query Attention
- **GQA:** Grouped-Query Attention
- **KV Cache:** Key-Value Cache for inference
- **FP16/BF16:** 16-bit floating point formats
- **WGSL:** WebGPU Shading Language

---

*This plan is a living document. Update as features are completed or priorities shift.*
