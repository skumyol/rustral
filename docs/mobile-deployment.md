# Native mobile deployment (iOS / Android)

Track H **Phase 4b** scopes how Rustral relates to on-device inference on phones, without promising a turnkey App Store pipeline.

## Practical paths

### 1. Server-side inference (simplest)

Run [`rustral-inference-server`](../crates/inference-server/) or your own Rust service in the cloud; mobile apps call HTTP/gRPC. This avoids FFI, codegen, and GPU fragmentation on device.

### 2. On-device CPU (ndarray backend)

- Build **`rustral-ndarray-backend`** + your model as a **static library** (`cdylib` or `staticlib`) and call from **Swift** / **Kotlin** via C ABI.
- Expect **moderate throughput**; suitable for tiny models or preprocessing.
- You own **threading** (one executor per core), **battery**, and **binary size** (strip LTO, split debug info).

### 3. On-device GPU

- **WGPU** on Android/iOS is evolving; Metal on iOS and Vulkan on some Android devices can work with `wgpu`, but **shader and timing** behavior must be validated per device class.
- **Core ML / NNAPI / TFLite** generally require **export** from Rustral weights (e.g. ONNX → converter → mobile runtime). See [`export-onnx-torchscript.md`](export-onnx-torchscript.md).

## Milestones we do *not* claim yet

- Official CocoaPods / Maven packages.
- Bitcode / Play compliance bundles for ML models.
- Hardware-accelerated attention kernels on all flagship devices.

## Recommendation

For production mobile ML, assume **export to a mobile runtime** or **server inference** until a concrete device matrix and benchmark story exists for `rustral-wgpu-backend` on phones.
