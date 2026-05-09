# Benchmarks

This page is the main entry point for Rustral performance work. If you want to run numbers locally, compare backends, check CI artifacts, or publish a release snapshot, start here.

The harness emits **schema v2.0.0** JSON. See [`benchmarks/SCHEMA.md`](benchmarks/SCHEMA.md) and [`benchmarks/schema_v2.json`](benchmarks/schema_v2.json) for the shape. CI uses `scripts/bench/validate_schema.py` to catch broken output before anyone trusts the numbers.

## Surfaces

There are three ways to measure performance:

1. **Criterion microbenches**: fine-grained per-op benches in [`crates/bench/benches/`](crates/bench/benches).
2. **System perf tests**: opt-in performance suites in [`tests/performance/`](tests/performance), env-gated for GPU/example perf.
3. **Backend examples**: backend-specific timing snippets, e.g. [`crates/candle-backend/examples/benchmark.rs`](crates/candle-backend/examples/benchmark.rs).

Each one has a job. For publishable numbers, use the **unified harness** below. It gives you raw timings, summary tables, machine metadata, and schema validation in one place.

Kernel autotuning (optional, `rustral-autotuner`) is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md) (`enabled`, `ci_mode`, cache behavior).

## Unified harness

Run the harness from the repo root:

```bash
# Run every workload with default repeats and write benchmarks/results/<timestamp>.json
python3 scripts/bench/run_all.py

# Override repeats (default: 5), pick a subset, point to a custom output file
python3 scripts/bench/run_all.py --repeats 10 --suite rustral --suite candle
```

Outputs:

- `benchmarks/results/*.json`: structured results with per-workload distributions (`mean_ms`, `p50_ms`, `std_ms`), backend, device, dtype, optional model parameter count, machine metadata, and raw timings.
- `benchmarks/results/summary.md`: regenerable Markdown table aggregating the latest JSON.

`benchmarks/results/` is git-ignored (see [`.gitignore`](.gitignore)); commit only specific runs you intend to publish.

## Per-release snapshots (publishable)

Curated snapshots taken at every tagged release live under [`benchmarks/runs/`](benchmarks/runs/). Each `benchmarks/runs/<version>/` directory contains:

- `manifest.json`: release version, git SHA, capture date, hardware summary, harness flags.
- `<suite>.json`: one schema-v2 file per suite.
- `summary.md`: Markdown table.

This is the directory academic readers should cite for reproducibility. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the release capture steps.

## NLP results (v0.1.0)

Real-data NLP results are curated separately from the schema-v2 performance harness. They live under:

- `benchmarks/runs/v0.1.0/nlp/sst2.json` (SST-2 dev accuracy, 3 seeds)
- `benchmarks/runs/v0.1.0/nlp/wikitext2.json` (WikiText-2 dev perplexity, 3 seeds)

PyTorch parity baselines (same architecture and the same `vocab.txt`) live next to them:

- `benchmarks/runs/v0.1.0/nlp/sst2_pytorch.json`
- `benchmarks/runs/v0.1.0/nlp/wikitext2_pytorch.json`

For **fast** regeneration (small model + small data, suitable for CI and local timing), use `python3 scripts/eval/run_nlp_real.py --benchmark` and the PyTorch scripts with `--benchmark`. Manifests record the exact widths and caps used.

See [`EVALUATION.md`](EVALUATION.md) for methodology, dataset pins, and how to read the curated JSON.

## Schema validation

```bash
python3 scripts/bench/validate_schema.py benchmarks/results/<file>.json
```

This validates each suite document against [`benchmarks/schema_v2.json`](benchmarks/schema_v2.json). Shared CI machines are noisy, so raw performance changes do not fail PRs. Broken JSON does.

## Baselines

The harness can run side-by-side baselines so claims like "Rustral vs Candle-direct vs PyTorch" are easy to check:

```bash
# Rust-only (Rustral + Candle direct)
python3 scripts/bench/run_all.py --suite rustral --suite candle

# Add the Python (PyTorch) baseline (requires a venv with torch installed)
python3 scripts/bench/run_all.py --suite rustral --suite candle --suite pytorch
```

Schema is shared across suites so you can join results in a spreadsheet or plot. See [`benchmarks/SCHEMA.md`](benchmarks/SCHEMA.md) for the JSON shape.

## Variance reporting

Every workload is repeated `--repeats` times. The JSON records:

- `runs_ms`: every individual measurement.
- `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `p50_ms`

This lets reviewers see the noise, not just one cherry-picked timing.

## GPU benchmarks (CUDA / Metal)

Two extra binaries cover the GPU matrix:

```bash
# CUDA (Linux + NVIDIA)
cargo run --release -p rustral-bench --features cuda --bin rustral_workloads_cuda -- \
    --repeats 5 --warmup 2

# Metal (macOS / Apple Silicon)
cargo run --release -p rustral-bench --features metal --bin rustral_workloads_metal -- \
    --repeats 5 --warmup 2
```

Both binaries call `device.synchronize()` around every timed call. That means wall-clock timings measure device execution, not just host queue submission. The timed region does not pull tensors back to the host.

The harness orchestrator wires GPU suites alongside CPU suites:

```bash
python3 scripts/bench/run_all.py --suite rustral --suite candle --suite rustral-metal
```

GPU CI is opt-in through the `bench-gpu` PR label or `workflow_dispatch`: see [`.github/workflows/bench-gpu.yml`](.github/workflows/bench-gpu.yml). The CUDA job expects a self-hosted runner with the `gpu-cuda` label.

## Bencher.dev (continuous trend tracking)

Once a bencher.dev project exists and the repo has `BENCHER_API_TOKEN` plus `BENCHER_PROJECT`, the `bench-cpu` CI job also:

1. Convert the schema-v2 JSON to Bencher Metric Format via [`scripts/bench/to_bencher_bmf.py`](scripts/bench/to_bencher_bmf.py).
2. Invoke `bencher run` to upload the BMF document.

Without those settings, the step is skipped. Every benchmark sample becomes one Bencher entry with a `latency` metric in nanoseconds (`mean`, `min`, `max`). We keep this view latency-only so the dashboard has one simple meaning.

## Workload coverage (schema v2)

Every Rust binary tags each sample with `device` (e.g. `cpu`, `cuda:0`, `metal:0`), `dtype` (`f32`/`f16`/`bf16`), and an optional `model_params` count. See `benchmarks/SCHEMA.md` for the full sample shape.

| Workload | Rustral (rustral_workloads) | Candle direct (candle_workloads) | PyTorch (baselines.py) |
|---|---|---|---|
| `matmul` | yes | yes | yes |
| `attention.{small,medium}` | yes | yes | yes |
| `conv2d.{small,medium,large}` | yes | yes | yes |
| `lstm_forward.{small,medium,large}` | blocked by `LstmCell` weight-layout fix | skipped (would need `candle-nn`) | skipped (kept to operator parity for now) |
| `mlp_train_step` | yes (Adam, no host probes in hot loop) | skipped (training APIs) | skipped (kept to operator parity for now) |
| `optimizer_step.{sgd,adam}` | yes (10M params default; `--profile heavy` for 100M) | n/a | n/a |
| `transformer_encoder.forward` | yes (forward only, see note below) | n/a | n/a |
| `decoder.{prefill,decode_step.no_cache}` | yes (no KV cache; baseline for K3) | n/a | n/a |
| `kv_cache.{prefill,decode_step}` | yes (`KVCache::append` micro) | n/a | n/a |
| `model_io.{save,load}` | yes (~50M f32 params, ~200 MB) | n/a | n/a |
| `lstm_lm_train_step` | tracked but skipped (gated on `LstmCell` weight-layout fix, same as `lstm_forward`) | n/a | n/a |

For GPU workloads, the timed path stays on-device. Any loss or accuracy probe belongs outside the hot loop.

> Note on `transformer_encoder.forward`: this is forward-only for now. A full encoder train step needs tape support for `MultiHeadAttention` and `TransformerEncoderLayer`. Today only `Linear`, `Embedding`, and `LayerNorm` implement `TapeModule`. The forward benchmark still gives us a useful trend line for encoder cost. The `mlp_train_step` workload already covers forward, backward, and optimizer cost on a simple model.

> Note on `lstm_forward` / `lstm_lm_train_step`: the existing Criterion bench panics with a shape mismatch. `LstmCell` stores `wx` as `[input_dim, 4*hidden_dim]`, but the CPU `linear` op expects `[out, in]`. That weight-layout fix is separate work. Once it lands, the LSTM workloads can move into the JSON harness.

> Note on `decoder.decode_step.no_cache`: this is the no-cache baseline. It runs a full-context forward pass for a decoded token. `kv_cache.decode_step` measures the cache append path separately. Wiring that cache into decoder forward is future work.

## Pages dashboard (per-release snapshots)

[`scripts/bench/render_site.py`](scripts/bench/render_site.py) walks `benchmarks/runs/<version>/` and emits a static HTML dashboard under `docs/site/`:

- `index.html`: landing page with a per-version dropdown.
- `<version>.html`: one page per snapshot, with a sortable sample table and the embedded `manifest.json`.

The dashboard is regenerated and deployed to GitHub Pages on every push that touches `benchmarks/runs/` or the renderer: see [`.github/workflows/pages.yml`](.github/workflows/pages.yml). If there are no snapshots yet, the renderer still writes a valid landing page.

To preview locally:

```bash
python3 scripts/bench/render_site.py
open docs/site/index.html
```

The generated `docs/site/` directory is git-ignored.

## Running specific surfaces directly

If you need to drop down to the underlying surface:

### Criterion microbenches

```bash
cargo bench -p rustral-bench --bench matmul
cargo bench -p rustral-bench --bench attention
```

Criterion writes its standard reports under `target/criterion/`.

### System perf tests

```bash
# CPU example perf (longer)
RUSTRAL_RUN_EXAMPLE_PERF=1 ./scripts/perf_examples.sh

# GPU perf (requires CUDA)
./scripts/perf_gpu.sh
```

### Backend example

```bash
cargo run --release -p rustral-candle-backend --example benchmark
```

## Operation fusion (not implemented)

Fused kernels (e.g. matmul + bias + activation in one pass) are not part of the shared `TensorOps` API yet. The practical plan is to profile hotspots first (`scripts/bench/run_all.py`, Criterion benches), then add **optional** fused entry points on backends that benefit (GPU), keeping unfused paths for correctness and for CPU.

## Reproducibility checklist

Before publishing numbers, capture and report:

- Hardware: CPU model + RAM + GPU model (if applicable)
- OS + kernel
- Rust toolchain (`rustc --version`)
- Git SHA
- Feature flags (`--features` used)
- Repeats and warmup iterations
- Any environment toggles (CPU governor, isolated cores, etc.)

The harness records most of this automatically; the rest belongs in your run notes.
