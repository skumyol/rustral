# rustral-onnx-export

Experimental **ONNX** export for a tiny, Rustral-friendly subset of operators.

## Supported

- Single **Linear** layer as ONNX `MatMul` + `Add`, float32 weights, **symbolic batch** dimension.

## API

See [`export_linear_f32`](src/lib.rs) and rustdoc.

## Build note

The crate vendors **`onnx.proto`** (ONNX project, Apache-2.0) under `proto/` and uses **`protoc-bin-vendored`** so a system `protoc` install is not required.

## License

The generated code and this crate follow the workspace license; the ONNX `.proto` file retains its original SPDX header.
