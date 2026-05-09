# Export: ONNX and TorchScript

Track H **Phase 5** documents how to leave Rustral’s Rust graph for ecosystem runtimes.

## ONNX (in-repo spike)

The [`rustral-onnx-export`](https://github.com/skumyol/rustral/tree/main/crates/onnx-export) crate can emit a minimal **Linear** (`MatMul` + `Add`) as a standard `.onnx` file (opset 17). Use it when:

- You need a **deployment artifact** for ONNX Runtime, TensorRT, mobile converters, etc.
- Your graph matches the **supported subset** (today: single linear layer with static inner dimensions and symbolic batch).

Example (from your own binary or test):

```rust
use rustral_onnx_export::export_linear_f32;
use std::fs::write;

let w = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2 x 2
let b = vec![0.0_f32, 0.0];
let bytes = export_linear_f32(2, 2, &w, &b).unwrap();
write("linear.onnx", bytes).unwrap();
```

**Scope:** Full transformers and arbitrary `rustral-nn` graphs are **not** exported automatically. Expanding op coverage is incremental work (attention, layer norm, etc.).

### Validating

Use [ONNX Runtime](https://onnxruntime.ai/), `onnx-checker` (Python), or your target stack to load and run the file.

## TorchScript

TorchScript is a **PyTorch** artifact format. Rustral does **not** emit TorchScript bytecode directly.

**Practical bridge:**

1. Train or convert weights to a format PyTorch can load (e.g. rebuild the same architecture in PyTorch and load weights converted from Safetensors / NumPy).
2. `torch.jit.trace` or `torch.jit.script` the PyTorch module.
3. Deploy the `.pt` file with PyTorch Mobile, LibTorch, or your serving stack.

For Hugging Face–centric workflows, exporting **Safetensors** from Rustral and loading in Transformers (with a matching `state_dict` map) is often simpler than chasing TorchScript.

## When to prefer Safetensors + Rustral server

If your deploy target can run a Rust binary, **`save_model` / `load_model`** artifacts plus [`rustral-inference-server`](../crates/inference-server/) may be simpler than ONNX for small services. Choose ONNX when downstream **requires** the format.
