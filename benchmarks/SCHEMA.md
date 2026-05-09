# Benchmark JSON schema (v2.0.0)

Every suite (Rustral CPU/CUDA/Metal, Candle-direct, PyTorch CPU/CUDA) emits a JSON
document with the same shape so the orchestrator (`scripts/bench/run_all.py`) can
join them into a single combined results file and a Markdown summary table.

The canonical machine-readable schema lives at
[`benchmarks/schema_v2.json`](schema_v2.json) and is enforced by
`scripts/bench/validate_schema.py`.

## Suite document

```json
{
  "suite": "rustral | candle | pytorch | pytorch-cuda | rustral-cuda | rustral-metal",
  "schema_version": "2.0.0",
  "machine": {
    "os": "macos | linux | windows",
    "arch": "x86_64 | aarch64",
    "hostname": "...",
    "rustc": "rustc 1.78.0 ...",        // rust suites
    "commit": "abcdef...",                // git SHA
    "features": ["cuda", "metal"],       // active cargo features
    "torch_version": "2.4.0",            // pytorch suites
    "python": "3.11.7",                  // pytorch suites
    "cuda_device_name": "NVIDIA ..."     // pytorch-cuda only, optional
  },
  "samples": [ Sample, ... ]
}
```

## Sample document

```json
{
  "name": "matmul",                                   // workload name (stable across suites)
  "backend": "ndarray-cpu | candle-cpu | pytorch-cpu | pytorch-cuda | ...",
  "device": "cpu | cuda:0 | metal:0 | wgpu:0",       // logical device
  "dtype": "f32 | f16 | bf16",                       // element dtype
  "model_params": 1234567,                            // optional; null when not applicable
  "params": { "m": "128", "k": "128", "n": "128" },  // workload-specific params (string values)
  "runs_ms": [0.45, 0.46, 0.44, 0.45, 0.47],
  "mean_ms": 0.454, "std_ms": 0.011,
  "min_ms": 0.44,  "max_ms": 0.47, "p50_ms": 0.45,
  "ci95_low_ms": 0.44,
  "ci95_high_ms": 0.47,
  "outlier_run_indices": [],
  "stats_note": "t_interval_on_ms;iqr_outliers"
}
```

Optional fields (emitters may omit any of these):

- **`ci95_low_ms` / `ci95_high_ms`**: 95% two-sided confidence interval for the mean latency, using Student’s \(t\) on `runs_ms` with \(n-1\) degrees of freedom. Omitted when `len(runs_ms) < 2`. Assumes run times are roughly normal (standard caveat for micro-benchmarks).
- **`outlier_run_indices`**: 0-based indices into `runs_ms` flagged by Tukey’s rule (fences at \(Q1 - 1.5\\,\mathrm{IQR}\) and \(Q3 + 1.5\\,\mathrm{IQR}\)). Raw `runs_ms` is unchanged; flags are advisory.
- **`stats_note`**: Short fixed tag describing methods, e.g. `t_interval_on_ms;iqr_outliers`.

## Combined harness output

`scripts/bench/run_all.py` writes:

```json
{
  "tool": "rustral-bench-harness",
  "git_sha": "...",
  "repeats": 5,
  "warmup": 1,
  "timestamp": "2026-05-06T10:00:00",
  "suites": [ SuiteDocument, ... ]
}
```

## Migration from v1

- `schema_version` is now required at the suite level (set to `"2.0.0"`).
- `device`, `dtype` are required on every sample. v1 documents implicitly assumed `cpu`/`f32`.
- `model_params` is required (may be `null`) so model-level workloads can report parameter counts unambiguously.
- `machine.hostname`, `machine.commit`, `machine.features` are required on Rust-emitting suites; PyTorch additionally reports `torch_version` and `python`.
- v1 documents under `benchmarks/results/` will fail `validate_schema.py`. Either delete them or move them to `benchmarks/runs/legacy-v1/` for archival.

## Conventions

- **`name`** is shared across suites (`matmul`, `attention.small`, `conv2d.medium`, ...). Joining on `name` lets you compare backends in a spreadsheet/plot directly.
- **`runs_ms`** is the raw measurement vector. Aggregates (`mean_ms`, `std_ms`, `p50_ms`, `min_ms`, `max_ms`) are precomputed for convenience but `runs_ms` is the source of truth.
- **`params`** carries workload-specific parameters as strings to keep the schema simple.
- All times are wall-clock milliseconds.
- `model_params` is the parameter count of the model under test (not the optimizer state). For micro-benchmarks of single ops it is `null`.
- When present, **`ci95_*`** and **`outlier_run_indices`** support statistical reporting and review; they do not replace raw `runs_ms` for custom analysis.
