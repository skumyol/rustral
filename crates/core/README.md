# rustral-core

Foundation crate: `Backend`, `TensorOps`, `Module`, `Parameter`, `NamedParameters`, and **`ForwardCtx`**.

Also hosts cross-cutting **optimization and observability** pieces:

- **Fusion** — `FusionOps`, `FusionOptimizer` / `apply_*`, pattern matching for fused op sequences.
- **Numerics** — `NumericsConfig`, `FusionTestHarness` for fused vs unfused checks.
- **Profiling** — `OperationProfiler`, `ProfilingHooks` (attach optionally to `ForwardCtx`).
- **Capabilities** — `BackendCapabilities`, `clamp_batch_size`, `recommends_mixed_precision`, `recommended_dtype_for_operation`, `OperationType` (incremental adoption pattern; see root [`ARCHITECTURE.md`](../../ARCHITECTURE.md)).
- **Shapes / pooling** — `ShapePolicy`, `TensorPool`, `PoolStrategy`, `begin_step`.

See the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and backend status.
