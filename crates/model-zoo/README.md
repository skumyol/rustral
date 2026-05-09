# Rustral model zoo (registry)

This crate ships a small **registry** ([`registry.json`](registry.json)) describing checkpoints and workflows that are intended to work with Rustral’s tooling.

It is **not** a weight hosting service. For Hugging Face models, weights stay on the Hub; for local demos, artifacts are produced by examples (for example `tape_train_demo`).

## Hugging Face name mapping

`rustral_hf::download_state_dict` returns a map from **HF tensor names** to flat `f32` data. `rustral_runtime::load_model` expects keys to match **`NamedParameters` names** from your Rust module (for example `lin.weight`, `lin.bias` for a nested `Linear`).

Common pitfalls:

1. **Prefix mismatch** — HF uses `bert.encoder.layer.0...`; your module might use `encoder.layers.0...`. Rename in code or build an explicit remapping table before calling `load_state_dict` / `load_model`.
2. **Layout** — Some HF weights need transpose vs Rustral `Linear` storage. Verify with a tiny layer before loading full models.
3. **Sharded / non-f32** — Large Hub models use sharded safetensors and mixed dtypes. The current `download_state_dict` helper targets a single `model.safetensors` and f32-oriented paths; LLM-scale loading belongs in the planned `rustral-llm` track (see `docs/master-plan.md` Track E).

## Programmatic access

```rust
use rustral_model_zoo::registry;

let reg = registry()?;
for e in reg.entries {
    println!("{} — {}", e.id, e.status);
}
```
