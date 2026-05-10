# rustral-nn

Layers: convolutions, RNNs, transformers, **LLaMA-shaped** [`LlamaDecoder`](src/llama.rs) (RMSNorm, RoPE, SwiGLU, **GQA** attention, optional **`LlamaDecodeCache`** for incremental decode), attention, MoE blocks, and serving-oriented helpers (`ServingEngine`, continuous batching, `kv_cache` tensor helpers—**library** primitives; for an HTTP process see [`rustral-inference-server`](../inference-server/README.md)).

**Fusion:** `LinearReLU` and `LinearGELU` use [`FusionHelper`](src/fusion_helper.rs), which delegates to `rustral_core::FusionOptimizer::apply_*` so fused vs unfused policy lives in one place.

**Tape training:** feature `autodiff` + `tape` — transformer FFN blocks use **`Tape::gelu`** (GELU), aligned with standard transformer practice and with eager decoder FFN activations in this repo.

See the [repository README](https://github.com/skumyol/rustral#readme) for install, examples, and backend status.
