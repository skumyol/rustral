# Backend Roadmap

The reference CPU backend is deliberately small. For real workloads, add production backends behind the same architectural boundary.

## Burn backend

Best for Rust-native training. Implement a crate such as `rustral-burn-backend` that maps `Backend::Tensor` to Burn tensors and delegates autodiff/optimization to Burn.

## Candle backend

Best for lightweight inference and Hugging Face-style model loading. Implement a crate such as `rustral-candle-backend` for LLM/NPC inference and adapter-driven expert modules.

## tch-rs backend

Best if PyTorch compatibility matters more than Rust-native design. Useful for migration, but less ideal as the long-term architectural center.

## Optimizer boundary

Do not bake an optimizer into `Module`. Keep optimization in runtime/training crates so modules stay reusable in inference and simulation.
