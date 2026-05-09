# rustral-autodiff

Reverse-mode autodiff and gradient tapes (`Tape`).

Common recorded ops include matmul, transpose, ReLU, **GELU** (for transformer FFNs), softmax, layer norm, losses, and parameter watching. Gradients for GELU match the same tanh approximation as `TensorOps::gelu` on each backend.

Rustral is a Rust workspace for research and learning; see the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and status by backend.
