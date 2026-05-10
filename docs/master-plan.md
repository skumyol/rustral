# Rustral docs: consolidated master plan

This is the **single source of truth** for Rustral’s architecture direction, current status, and the next implementation steps. Other documents in `docs/` should be treated as **supporting references** and should link back here rather than restating roadmaps.

**Last updated:** 2026-05

---

## Where we are (reality, not aspiration)

Rustral is a Rust-first neural network workspace with:

- **Explicit execution**: `ForwardCtx` + `Mode` (+ run id, optional `ShapePolicy` and `OperationProfiler`), no hidden global state.
- **Backend abstraction**: `Backend` + `TensorOps` (CPU reference backend; Candle CPU/CUDA/Metal; wgpu experimental).
- **Training utilities**: tape-based autodiff, optimizers, and a runtime trainer that exercises end-to-end training + checkpoint I/O.
- **Real evaluation artifacts**: SST-2 and WikiText-2 examples write reproducibility manifests and can run in offline CI smoke tests.
- **Benchmark evidence**: schema-v2 JSON harness, CPU CI artifacts, optional CUDA/Metal suites, bencher.dev upload, and a Pages dashboard for release snapshots.
- **Distributed APIs**: a `ProcessGroup` abstraction and higher-level DP/TP/ZeRO-style components (correctness-first; performance backend collectives are future work).
- **Deployment ecosystem (Track H)**: HTTP inference MVP (`rustral-inference-server`), curated registry metadata (`rustral-model-zoo`), experimental ONNX Linear export (`rustral-onnx-export`), plus docs for nginx/Docker, wasm/mobile scope, and TorchScript bridging.

Important caveat: a lot of the “big” distributed and model-parallel APIs exist as **library surfaces** and tests, but they are not yet a production multi-node system.

---

## Canonical design pattern (keep this invariant)

The core layering stays:

```text
rustral-core
  Backend + TensorOps (portable surface)
  ForwardCtx (explicit mode/run-id; optional shape hints + profiler)
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
- **Metal builds** are opt-in (`--features metal`) and are mainly for local macOS / Apple Silicon benchmarking.
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

## What shipped in the publishable evidence work

### Real-corpus NLP

- `crates/runtime/examples/sst2_classifier.rs` trains a small SST-2 classifier and reports dev accuracy.
- `crates/runtime/examples/wikitext2_lm.rs` trains a small word-level WikiText-2 LM and reports dev perplexity.
- Both examples write `manifest.json` with git SHA, seed, dataset hash, hyperparameters, throughput, and metric output.
- `EVALUATION.md` explains tokenization, splits, metrics, online/offline data fetch, and smoke test commands.
- Curated real-data snapshots (3 seeds) live under `benchmarks/runs/v0.1.0/nlp/`:
  - `sst2.json`, `wikitext2.json` (Rustral)
  - `sst2_pytorch.json`, `wikitext2_pytorch.json` (PyTorch parity baselines)
- A nightly / manual real-data gate runs via `.github/workflows/nlp-real.yml` and validates manifests against `benchmarks/manifest_schema.json`.

### Benchmark and CI evidence

- `crates/bench/src/lib.rs` emits schema-v2 samples with backend, device, dtype, model parameter count, machine metadata, and raw runs.
- `scripts/bench/run_all.py` runs the CPU suites and optional PyTorch / CUDA / Metal suites.
- `scripts/bench/validate_schema.py` validates benchmark JSON in CI.
- `.github/workflows/ci.yml` uploads CPU benchmark artifacts and runs offline NLP smoke tests.
- `.github/workflows/bench-gpu.yml` runs CUDA benchmarks only when requested.
- `scripts/bench/to_bencher_bmf.py` converts results for bencher.dev trend tracking.
- `scripts/bench/render_site.py` renders the Pages dashboard from `benchmarks/runs/<version>/`.

Known gaps are explicit, not hidden: transformer train-step still needs tape support for attention layers, and LSTM workload promotion waits on the `LstmCell` weight-layout fix.

---

## Next implementation plan

The next big step is to turn the current forward and simple-train benchmarks into fuller model-level train-step coverage:

1. **Tape-aware module execution**
   - Add a `TapeModule`-style trait (or equivalent) so `rustral-nn` layers can be executed while recording ops into a `Tape` without rewriting every example.
2. **Migrate key layers**
   - Implement tape-forward for `Linear`, `Embedding`, `LayerNorm`, and the most common activation/loss paths.
3. **Generic trainer**
   - Promote a single “real” trainer in `rustral-runtime` that can train a tape-aware model on CPU and CUDA backends without per-example training loops.
4. **Efficient checkpointing**
   - Prefer `tensor_to_vec` (bulk reads) over per-element reads; checkpointing must not accidentally serialize GPU scalars one-by-one.
5. **Performance suite**
   - Keep schema-v2 JSON as the public benchmark format.
   - Capture the first release snapshot under `benchmarks/runs/<version>/`.
   - Keep GPU perf opt-in through `.github/workflows/bench-gpu.yml`.

---

## Document index (what lives where)

- **This file** (`docs/master-plan.md`): roadmap + current status + invariants.
- **`docs/api-signatures.md`**: API inventory (should remain mostly “what exists”, not “what we want”).
- **`docs/architecture.md`**: short architecture sketch; should link here for roadmap.
- **`docs/backend-roadmap.md`**: backend-specific notes (Burn/Candle/tch/wgpu), not a second roadmap.
- **`docs/concepts.md`**: tutorial/guide; should avoid repeating roadmap claims.
- **`docs/WGPU_UPGRADE.md`**: wgpu upgrade procedure (experimental backend).
- **`docs/export-onnx-torchscript.md`**: ONNX spike + TorchScript / PyTorch bridge stance.
- **`docs/wasm-wgpu-inference.md`**: browser/WebGPU scope (incl. wasm `getrandom` caveat).
- **`docs/mobile-deployment.md`**: iOS/Android integration paths and non-goals.
- **`crates/inference-server/DEPLOYMENT.md`**: reverse proxy, probes, metrics, Docker.
- **`docs/SECURITY.md`**: security guidelines and disclosure process.
- **`EVALUATION.md`**: SST-2 and WikiText-2 methodology.
- **`BENCHMARKS.md`**: benchmark harness, schema v2, backend matrix, snapshots, bencher.dev, and Pages dashboard.

---

## Performance

For local numbers, run:

```bash
python3 scripts/bench/run_all.py --suite rustral --suite candle --repeats 5 --warmup 1
```

For microbench work, `cargo bench -p rustral-bench` still exists. For publishable numbers, use the unified harness and commit only curated release snapshots under `benchmarks/runs/<version>/`.

---

## Comparison with Python stacks

PyTorch/JAX ship mature kernels, distributed runtimes, and ecosystems Rustral does **not** replace overnight. Rustral’s niche is **Rust-native**, explicit-state experimentation, not drop-in datacenter training without extra engineering.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Experimental GPU backend | Prefer Candle for reliability until `wgpu` upgrades land |
| Distributed **simulation vs cluster** | Read API docs; don’t assume NCCL/MPI cluster parity |
| Benchmark regression | CI uploads schema-checked CPU artifacts; bencher.dev upload is opt-in |
| Real-corpus metric drift | Cite `manifest.json` and rerun from `EVALUATION.md` commands |

---

## LLM v3 bite progress

Maintainer-only detail may live in local `IMPROVEMENT_PLAN.md` (gitignored). This table is committed for visibility.

| Bite | Deliverable | Status |
|------|-------------|--------|
| 01 | `rustral-hf::scan_local_model_dir` | Done |
| 02 | `snapshot_model_at` Hub revision pinning | Done |
| 03 | `rustral-io` sharded `MetaStateDict` load (`load_meta_state_dict_from_paths`, `load_meta_state_dict_from_hub_index`) | Done |
| 04 | `rustral-llm` re-export `HubModelSnapshot` / `HubModelFiles` | Done |
| 05 | GPT-2 HF → `NamedParameters` mapping (embed/LN/FFN/`lm_head`) | Done |
| 06 | `Gpt2Decoder::load_hf_weights_from_meta` / `from_hf_meta` | Done |

---

## Next actions (maintainers)

1. Capture the first release benchmark snapshot under `benchmarks/runs/<version>/`.
2. Add tape support for attention layers so transformer train-step benchmarks become real train-step benchmarks.
3. Fix the `LstmCell` weight layout so LSTM workloads can enter the JSON harness.
4. Upgrade `wgpu` and revisit GPU RNG/dropout stories.
5. Wire metrics (`rustral-metrics`) to real sinks where desired.
6. **LLM vertical (v3):** next — GPT-2 **attention** weight conversion (HF `c_attn` / `c_proj` ↔ `SelfAttention`); then **`CausalLm`**, metrics JSON, Candle parity (**07**–**09** in `IMPROVEMENT_PLAN.md` V3 table).

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
