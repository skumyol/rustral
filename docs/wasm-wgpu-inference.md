# WebAssembly and WebGPU inference (Rustral)

This document scopes **Track H Phase 4a**: browser-side inference using Rust + WebGPU, relative to the experimental [`rustral-wgpu-backend`](https://github.com/skumyol/rustral/tree/main/crates/wgpu-backend).

## Current state

- The **WGPU backend** targets native Vulkan/Metal/DX12 via `pollster` for synchronous GPU submission.
- **`rustral-core`** is largely platform-agnostic and is a good candidate for `wasm32-unknown-unknown` builds **if** you avoid APIs that assume native threads or file I/O in hot paths.
- A plain `cargo check -p rustral-core --target wasm32-unknown-unknown` may fail today on transitive **`getrandom`** (needs the `js` feature for wasm). Fixing wasm for core is a small dependency-graph / feature-unification task, not a fundamental blocker.
- **Full `rustral-wgpu-backend` on wasm** requires:
  - `wgpu` with the WebGPU backend (browser or `wasm-bindgen` + `web-sys`).
  - Replacing or wrapping **blocking** `pollster::block_on` patterns with an **async** surface (`wasm-bindgen-futures`), because browsers do not allow long synchronous GPU waits on the main thread the same way as native.
  - Careful **memory / stack** limits and **shader** feature validation (f16, subgroup sizes) that differ from Vulkan.

## Recommended milestones

1. **`cargo check -p rustral-core --target wasm32-unknown-unknown`** — fix any `std` or crate gates; keep optional heavy crates out of the wasm dependency graph.
2. **Minimal tensor demo** — one matmul or elementwise op through `wgpu` in a `wasm-pack` template without the full Rustral stack, to validate WGSL and buffer lifecycle in the browser.
3. **Integrate Rustral `Module` forward** — only after step 2; start with a tiny `Linear` and ndarray CPU fallback for correctness comparisons.

## Limitations (honest)

- **Training** on wasm is impractical for real models; scope wasm to **inference**.
- **Downloaded weights** (e.g. Hugging Face) need a browser fetch + cache story; keep I/O explicit and async.
- **Safetensors parsing** in the browser is fine with `rustral-io`, but large models stress WASM memory; consider **quantized** or **chunked** weights later.

## Related docs

- [`mobile-deployment.md`](mobile-deployment.md) for native iOS/Android scope.
- [`export-onnx-torchscript.md`](export-onnx-torchscript.md) if you export to run in ONNX Runtime Web instead of in-process WGPU.
