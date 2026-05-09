# rustral-bench

Benchmark tools for Rustral.

This crate has two jobs:

- Criterion microbenches for focused kernel work.
- Schema-v2 JSON workload binaries for backend comparison, CI artifacts, release snapshots, and trend tracking.

Run the main CPU harness from the repo root:

```bash
python3 scripts/bench/run_all.py --suite rustral --suite candle --repeats 5 --warmup 1
```

Add PyTorch (CPU) and optional PyTorch CUDA in one JSON (CUDA is skipped if unavailable):

```bash
python3 scripts/bench/run_all.py --suite pytorch --suite pytorch-cuda --repeats 3 --warmup 1 --out benchmarks/results/pt.json
```

Full local queue (CPU + optional GPU flags): `scripts/bench/queue_all_benchmarks.sh` (see header comments for `RUN_PYTORCH_CUDA`, `RUN_CUDA_BENCH`, `RUN_METAL_BENCH`).

Run the Rustral workload binary directly:

```bash
cargo run --release -p rustral-bench --bin rustral_workloads -- --repeats 5 --warmup 1
```

GPU binaries are opt-in:

```bash
cargo run --release -p rustral-bench --features cuda --bin rustral_workloads_cuda -- --repeats 5 --warmup 2
cargo run --release -p rustral-bench --features metal --bin rustral_workloads_metal -- --repeats 5 --warmup 2
```

Current JSON workloads include matmul, attention, conv2d, MLP train-step, optimizer step, transformer encoder forward, decoder prefill/decode, KV cache append, and model save/load throughput.

See [`BENCHMARKS.md`](../../BENCHMARKS.md) for the full workflow and schema details.
