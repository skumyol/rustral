# Architecture

The runtime follows a layered design. This file is intentionally short; the canonical roadmap and invariants live in [`docs/master-plan.md`](master-plan.md).

```text
Backend
  owns tensor/device operations

ForwardCtx
  carries backend, mode, run id, ShapePolicy, and optional OperationProfiler

Parameter
  owned by modules, never by a global registry

Module
  typed input/output contract

Trainable
  exposes parameters for optimizers/checkpointing

Runtime
  owns parallel training/inference orchestration
```

## Why explicit state?

The legacy C++ wrapper reduces boilerplate by hiding graph lifetime and parameter state. That makes tiny examples pleasant, but it creates hidden coupling between object lifetimes, graph renewal, caching, and training. Rust gives us a better option: make lifetimes and ownership visible at module boundaries.

## Backend abstraction

`rustral-core::Backend` is intentionally small. It lets the architecture support a simple CPU reference backend today and a production backend later. A future Burn or Candle backend should implement the same `TensorOps` contract or a richer extension trait.

## Parallelism

Training uses data-parallel map/reduce at the batch level:

```text
batch examples
  -> parallel loss_and_update(example)
  -> merge_updates
  -> apply_update
```

Inference uses a bounded worker pool with backpressure:

```text
request
  -> bounded queue
  -> worker threads
  -> reply channel
```

This keeps parallelism explicit and testable rather than hiding it inside model code.
