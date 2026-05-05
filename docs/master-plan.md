# Rustral docs — consolidated master plan

This is the **single source of truth** for Rustral’s architecture direction, current status, and the next implementation steps. Other documents in `docs/` should be treated as **supporting references** and should link back here rather than restating roadmaps.

**Last updated:** 2026-05

---

## Where we are (reality, not aspiration)

Rustral is a Rust-first neural network workspace with:

- **Explicit execution**: `ForwardCtx` + `Mode`, no hidden global state.
- **Backend abstraction**: `Backend` + `TensorOps` (CPU reference backend; Candle CPU/CUDA; wgpu experimental).
- **Training utilities**: tape-based autodiff, optimizers, and a runtime trainer that exercises end-to-end training + checkpoint I/O.
- **Distributed APIs**: a `ProcessGroup` abstraction and higher-level DP/TP/ZeRO-style components (correctness-first; performance backend collectives are future work).

Important caveat: a lot of the “big” distributed and model-parallel APIs exist as **library surfaces** and tests, but they are not yet a production multi-node system.

---

## Canonical design pattern (keep this invariant)

The core layering stays:

```text
rustral-core
  Backend + TensorOps (portable surface)
  ForwardCtx (explicit mode/run-id)
  Parameter (owned by modules)
  Module / Trainable (composition + param exposure)

rustral-autodiff / rustral-optim / rustral-io
  gradients, optimizers, safetensors state dict

rustral-runtime
  orchestration and trainers (batching, stepping, checkpointing)

rustral-distributed
  ProcessGroup + collectives + higher-level parallelism wrappers
```

Backends (ndarray, Candle, wgpu) implement `TensorOps`; orchestration lives above them.

---

## GPU support policy (builds everywhere)

- **CPU-only** builds and tests must remain the default and should work on Linux/macOS/Windows.
- **CUDA builds** are opt-in (`--features cuda`) and require a CUDA toolkit.
- To avoid carrying source patches, CUDA builds require:
  - **CUDA toolkit ≥ 12.2** (see `scripts/check_cuda_env.sh`).

---

## What we shipped in the “serious training enablement” work

### Runtime training (single host)

- `rustral-runtime` gained a `training` feature and an end-to-end trainer utility (`train_synthetic_classification`) that uses:
  - `Tape` autodiff
  - `Adam` optimizer
  - checkpoint round-trip via a safetensors state dict

This validates the plumbing (forward → loss → backward → step → save/load) for CPU and for CUDA when enabled.

### Distributed core (correctness-first)

- `ProcessGroup::new_mpi()` behind the `mpi` feature.
- Collectives added/normalized: `barrier`, `broadcast_f32`, `all_reduce_mean_f32`, `all_gather_f32`.
- Tensor parallel column path now performs an all-gather instead of returning a local shard silently.
- `DistributedCheckpointManager::load` now restores parameters (no placeholder loop).

---

## Next implementation plan (the “make examples trainable” arc)

This is the next big step to match the design and usability goals:

1. **Tape-aware module execution**
   - Add a `TapeModule`-style trait (or equivalent) so `rustral-nn` layers can be executed while recording ops into a `Tape` without rewriting every example.
2. **Migrate key layers**
   - Implement tape-forward for `Linear`, `Embedding`, `LayerNorm`, and the most common activation/loss paths.
3. **Generic trainer**
   - Promote a single “real” trainer in `rustral-runtime` that can train a tape-aware model on CPU and CUDA backends without per-example training loops.
4. **Efficient checkpointing**
   - Prefer `tensor_to_vec` (bulk reads) over per-element reads; checkpointing must not accidentally serialize GPU scalars one-by-one.
5. **Performance suite**
   - Keep `tests/system_tests` as the umbrella suite.
   - Run example binaries and GPU perf **opt-in** via env vars:
     - `RUSTRAL_RUN_EXAMPLE_PERF=1`
     - `RUSTRAL_RUN_GPU_PERF=1` + `--features cuda`

---

## Document index (what lives where)

- **This file** (`docs/master-plan.md`): roadmap + current status + invariants.
- **`docs/api-signatures.md`**: API inventory (should remain mostly “what exists”, not “what we want”).
- **`docs/architecture.md`**: short architecture sketch; should link here for roadmap.
- **`docs/backend-roadmap.md`**: backend-specific notes (Burn/Candle/tch/wgpu), not a second roadmap.
- **`docs/concepts.md`**: tutorial/guide; should avoid repeating roadmap claims.
- **`docs/WGPU_UPGRADE.md`**: wgpu upgrade procedure (experimental backend).
- **`docs/SECURITY.md`**: security guidelines and disclosure process.
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
