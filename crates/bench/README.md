# rustral-bench

Benchmark tools for Rustral.

This crate has two jobs:

- Criterion microbenches for focused kernel work.
- Schema-v2 JSON workload binaries for backend comparison, CI artifacts, release snapshots, and trend tracking.

Run the main CPU harness from the repo root:

```bash
python3 scripts/bench/run_all.py --suite rustral --suite candle --repeats 5 --warmup 1
```

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
