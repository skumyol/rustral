# rustral-nn

Layers: convolutions, RNNs, transformers, attention, MoE blocks, and serving-oriented helpers.

**Fusion:** `LinearReLU` and `LinearGELU` use [`FusionHelper`](src/fusion_helper.rs), which delegates to `rustral_core::FusionOptimizer::apply_*` so fused vs unfused policy lives in one place.

**Tape training:** feature `autodiff` + `tape` — transformer FFN blocks use **`Tape::gelu`** (GELU), aligned with standard transformer practice and with eager decoder FFN activations in this repo.

See the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and backend status.
