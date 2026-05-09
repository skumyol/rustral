//! Model-level save/load helpers.
//!
//! These helpers intentionally operate through stable parameter names so the same
//! names power training, checkpointing, and debugging.

use std::collections::HashMap;

use rustral_core::{Backend, NamedParameters, Parameter};
use rustral_io::{load_state_dict_typed, save_state_dict_typed, StateTensor};

/// Serialize a model to a Safetensors byte buffer.
///
/// This stores each parameter keyed by its stable name, preserving tensor shape.
pub fn save_model<B: Backend, M: NamedParameters<B>>(model: &M, backend: &B) -> anyhow::Result<Vec<u8>>
where
    B::Tensor: Clone,
{
    let ops = backend.ops();
    let mut dict: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut ids: HashMap<String, u64> = HashMap::new();
    model.visit_parameters(&mut |name, p: &Parameter<B>| {
        if let Ok(v) = ops.tensor_to_vec(p.tensor()) {
            let shape = ops.shape(p.tensor());
            dict.insert(name.to_string(), (v, shape));
            ids.insert(name.to_string(), p.id().get());
        }
    });

    let mut meta = HashMap::new();
    meta.insert("rustral.parameter_ids".to_string(), serde_json::to_string(&ids)?);
    Ok(save_state_dict_typed(&dict, Some(meta))?)
}

/// Load parameters from a Safetensors byte buffer into an existing model.
///
/// Shapes are taken from the model's existing parameter tensors.
pub fn load_model<B: Backend, M: NamedParameters<B>>(
    model: &mut M,
    backend: &B,
    bytes: &[u8],
) -> anyhow::Result<()>
where
    B::Tensor: Clone,
{
    let ops = backend.ops();
    let loaded = load_state_dict_typed(bytes)?;
    let mut model_meta: Vec<(String, Vec<usize>)> = Vec::new();
    model.visit_parameters(&mut |name, p| {
        model_meta.push((name.to_string(), ops.shape(p.tensor())));
    });

    // Strict checks: missing keys, extra keys, and shape/length mismatch.
    let mut missing: Vec<String> = Vec::new();
    for (k, _shape) in &model_meta {
        if !loaded.contains_key(k) {
            missing.push(k.clone());
        }
    }
    if !missing.is_empty() {
        anyhow::bail!("load_model: missing keys: {}", missing.join(", "));
    }

    let mut extra: Vec<String> = Vec::new();
    for k in loaded.keys() {
        if !model_meta.iter().any(|(mk, _)| mk == k) {
            extra.push(k.clone());
        }
    }
    if !extra.is_empty() {
        extra.sort();
        anyhow::bail!("load_model: unexpected extra keys: {}", extra.join(", "));
    }

    // Materialize all tensors first so we can fail fast with a real error.
    let mut materialized: HashMap<String, B::Tensor> = HashMap::with_capacity(model_meta.len());
    for (name, shape) in &model_meta {
        let StateTensor { data: values, shape: stored_shape, .. } = loaded
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("load_model: missing key after precheck: {name}"))?;
        if stored_shape != shape {
            anyhow::bail!(
                "load_model: shape mismatch for '{name}': expected {:?}, got {:?}",
                shape,
                stored_shape
            );
        }
        let expected_elems: usize = shape.iter().product();
        if values.len() != expected_elems {
            anyhow::bail!(
                "load_model: shape mismatch for '{name}': expected {expected_elems} elements for shape {:?}, got {}",
                shape,
                values.len()
            );
        }
        let t = ops
            .tensor_from_vec(values.clone(), shape)
            .map_err(|e| anyhow::anyhow!("load_model: failed to materialize tensor for '{name}': {e:?}"))?;
        materialized.insert(name.clone(), t);
    }

    model.visit_parameters_mut(&mut |name, p: &mut Parameter<B>| {
        let t = materialized.get(name).expect("materialized tensor missing for model key");
        *p = p.clone().with_tensor(t.clone());
    });
    Ok(())
}
