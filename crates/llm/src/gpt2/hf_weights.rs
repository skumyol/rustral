//! Hugging Face GPT-2 Safetensors → Rustral `TransformerDecoder` weight loading.
//!
//! # Supported tensors (shape-compatible paths)
//!
//! - `token_embedding.embed` ← `{prefix}.wte.weight`
//! - `lm_head.weight` ← `lm_head.weight`
//! - `layers.{i}.norm1.*`, `norm2.*` ← `h.{i}.ln_1`, `ln_2`
//! - `layers.{i}.ff_linear1.*`, `ff_linear2.*` ← `mlp.c_fc`, `mlp.c_proj` (HF Conv1D storage matches our `[out, in]` Linear weights)
//! - `final_norm.*` ← `{prefix}.ln_f.*`
//!
//! # Not loaded (architecture mismatch)
//!
//! HF merges Q/K/V into `attn.c_attn` and uses a full `c_proj`; our [`rustral_nn::SelfAttention`]
//! uses separate `[d_model, head_dim]` projections. Loading HF attention weights requires a dedicated conversion pass (future work).
//! Those parameters stay at their initialization values; see [`Gpt2WeightLoadReport::skipped_attention_parameters`].

use std::collections::HashMap;

use rustral_core::{Backend, NamedParameters, Parameter};
use rustral_io::{MetaStateDict, TensorEntry};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::TransformerDecoder;
use safetensors::Dtype;

use super::HfGpt2Config;
use crate::LlmError;

/// Result of [`load_hf_gpt2_weights_into_decoder`].
#[derive(Debug, Clone, Default)]
pub struct Gpt2WeightLoadReport {
    /// Rustral parameter names updated from the checkpoint.
    pub loaded_rustral_keys: Vec<String>,
    /// Model parameter names left unchanged (HF attention tensors not mapped).
    pub skipped_attention_parameters: Vec<String>,
    /// HF keys present in the checkpoint that this loader does not consume (includes attention shards).
    pub unmapped_hf_keys: Vec<String>,
}

/// Detect `transformer` vs `gpt2` root prefix used in Hugging Face `state_dict` keys.
pub fn detect_gpt2_state_dict_prefix(meta: &MetaStateDict) -> Result<&'static str, LlmError> {
    let has_transformer = meta.tensors.keys().any(|k| k.starts_with("transformer.wte.") || k.starts_with("transformer.h."));
    let has_gpt2 = meta.tensors.keys().any(|k| k.starts_with("gpt2.wte.") || k.starts_with("gpt2.h."));
    match (has_transformer, has_gpt2) {
        (true, false) => Ok("transformer"),
        (false, true) => Ok("gpt2"),
        (true, true) => Err(LlmError::InvalidArg(
            "checkpoint has both transformer.* and gpt2.* roots; ambiguous prefix".to_string(),
        )),
        (false, false) => Err(LlmError::InvalidArg(
            "could not detect GPT-2 root (expected transformer.* or gpt2.* tensor names)".to_string(),
        )),
    }
}

fn tensor_entry_f32(entry: &TensorEntry) -> Result<Vec<f32>, LlmError> {
    if entry.dtype != Dtype::F32 {
        return Err(LlmError::UnsupportedCheckpointDtype {
            name: entry.name.clone(),
            dtype: format!("{:?}", entry.dtype),
        });
    }
    let n: usize = entry.shape.iter().product();
    if entry.data.len() != n * 4 {
        return Err(LlmError::InvalidArg(format!(
            "tensor '{}': expected {} f32 bytes for shape {:?}, got {}",
            entry.name,
            n * 4,
            entry.shape,
            entry.data.len()
        )));
    }
    Ok(entry
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Build `(rustral_name -> data, shape)` for every parameter we can take verbatim from HF GPT-2 dumps.
pub fn build_gpt2_flat_map(
    meta: &MetaStateDict,
    cfg: &HfGpt2Config,
    prefix: &str,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, LlmError> {
    let mut out: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    let mut take = |rustral: &str, hf: &str| -> Result<(), LlmError> {
        let Some(entry) = meta.tensors.get(hf) else {
            return Ok(());
        };
        let data = tensor_entry_f32(entry)?;
        out.insert(rustral.to_string(), (data, entry.shape.clone()));
        Ok(())
    };

    take("token_embedding.embed", &format!("{prefix}.wte.weight"))?;
    take("lm_head.weight", "lm_head.weight")?;

    if cfg.n_positions > 0 {
        // Learned position embeddings exist in HF but our decoder uses sinusoidal `PositionalEncoding` (no weights).
        let _ = meta.tensors.get(&format!("{prefix}.wpe.weight"));
    }

    for i in 0..cfg.n_layer {
        let h = format!("{prefix}.h.{i}");
        take(&format!("layers.{i}.norm1.weight"), &format!("{h}.ln_1.weight"))?;
        take(&format!("layers.{i}.norm1.bias"), &format!("{h}.ln_1.bias"))?;
        take(&format!("layers.{i}.norm2.weight"), &format!("{h}.ln_2.weight"))?;
        take(&format!("layers.{i}.norm2.bias"), &format!("{h}.ln_2.bias"))?;
        take(&format!("layers.{i}.ff_linear1.weight"), &format!("{h}.mlp.c_fc.weight"))?;
        take(&format!("layers.{i}.ff_linear1.bias"), &format!("{h}.mlp.c_fc.bias"))?;
        take(&format!("layers.{i}.ff_linear2.weight"), &format!("{h}.mlp.c_proj.weight"))?;
        take(&format!("layers.{i}.ff_linear2.bias"), &format!("{h}.mlp.c_proj.bias"))?;
        // Attention keys deliberately omitted — see module docs.
    }

    take("final_norm.weight", &format!("{prefix}.ln_f.weight"))?;
    take("final_norm.bias", &format!("{prefix}.ln_f.bias"))?;

    if !out.contains_key("token_embedding.embed") {
        return Err(LlmError::MissingFile(format!("{prefix}.wte.weight")));
    }

    Ok(out)
}

fn collect_model_param_shapes<B: Backend>(model: &TransformerDecoder<B>, backend: &B) -> HashMap<String, Vec<usize>>
where
    B::Tensor: Clone,
{
    let ops = backend.ops();
    let mut m = HashMap::new();
    model.visit_parameters(&mut |name, p: &Parameter<B>| {
        m.insert(name.to_string(), ops.shape(p.tensor()));
    });
    m
}

fn attention_related_rustral_keys(n_layer: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(n_layer * 4);
    for i in 0..n_layer {
        for part in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            v.push(format!("layers.{i}.self_attn.{part}"));
        }
    }
    v
}

/// Load compatible GPT-2 tensors from `meta` into `model`. Attention projections are skipped; see report.
pub fn load_hf_gpt2_weights_into_decoder(
    model: &mut TransformerDecoder<CpuBackend>,
    backend: &CpuBackend,
    meta: &MetaStateDict,
    cfg: &HfGpt2Config,
) -> Result<Gpt2WeightLoadReport, LlmError> {
    let prefix = detect_gpt2_state_dict_prefix(meta)?;
    let flat = build_gpt2_flat_map(meta, cfg, prefix)?;

    let model_shapes = collect_model_param_shapes(model, backend);
    let ops = backend.ops();

    for (name, (data, shape)) in &flat {
        let expected = model_shapes.get(name).ok_or_else(|| {
            LlmError::InvalidArg(format!("checkpoint has unexpected parameter name '{name}' for this model"))
        })?;
        if expected != shape {
            return Err(LlmError::Gpt2ShapeMismatch {
                name: name.clone(),
                expected: expected.clone(),
                got: shape.clone(),
            });
        }
        let expected_elems: usize = shape.iter().product();
        if data.len() != expected_elems {
            return Err(LlmError::InvalidArg(format!(
                "checkpoint '{name}': len {} does not match shape {:?}",
                data.len(),
                shape
            )));
        }
    }

    let mut materialized: HashMap<String, <CpuBackend as Backend>::Tensor> = HashMap::new();
    for (name, (data, shape)) in &flat {
        let t = ops.tensor_from_vec(data.clone(), shape).map_err(|e| {
            LlmError::InvalidArg(format!("tensor_from_vec failed for '{name}': {e:?}"))
        })?;
        materialized.insert(name.clone(), t);
    }

    let mut loaded = Vec::new();
    model.visit_parameters_mut(&mut |name, p: &mut Parameter<CpuBackend>| {
        if let Some(t) = materialized.get(name) {
            *p = p.clone().with_tensor(t.clone());
            loaded.push(name.to_string());
        }
    });
    loaded.sort();

    let skipped_attention_parameters = attention_related_rustral_keys(cfg.n_layer);

    let mut unmapped_hf_keys: Vec<String> = meta
        .tensors
        .keys()
        .filter(|k| {
            let k = k.as_str();
            k.contains(".attn.") || k.contains(".wpe.") || k.contains("inv_freq")
        })
        .cloned()
        .collect();
    unmapped_hf_keys.sort();

    Ok(Gpt2WeightLoadReport {
        loaded_rustral_keys: loaded,
        skipped_attention_parameters,
        unmapped_hf_keys,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_nn::TransformerDecoderConfig;

    fn entry(name: &str, shape: Vec<usize>, data: Vec<f32>) -> TensorEntry {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        TensorEntry {
            name: name.to_string(),
            shape,
            dtype: Dtype::F32,
            data: bytes,
        }
    }

    #[test]
    fn loads_embedding_ln_ff_lm_head_small_fixture() {
        let d = 8usize;
        let ff = 16usize;
        let vocab = 11usize;
        let n_layer = 1usize;
        let n_head = 2usize;

        let mut tensors = HashMap::new();
        let p = "transformer";
        tensors.insert(format!("{p}.wte.weight"), entry(&format!("{p}.wte.weight"), vec![vocab, d], vec![0.1; vocab * d]));

        tensors.insert("lm_head.weight".to_string(), entry("lm_head.weight", vec![vocab, d], vec![0.2; vocab * d]));

        let h = format!("{p}.h.0");
        tensors.insert(format!("{h}.ln_1.weight"), entry(&format!("{h}.ln_1.weight"), vec![d], vec![1.0; d]));
        tensors.insert(format!("{h}.ln_1.bias"), entry(&format!("{h}.ln_1.bias"), vec![d], vec![0.0; d]));
        tensors.insert(format!("{h}.ln_2.weight"), entry(&format!("{h}.ln_2.weight"), vec![d], vec![1.0; d]));
        tensors.insert(format!("{h}.ln_2.bias"), entry(&format!("{h}.ln_2.bias"), vec![d], vec![0.0; d]));

        tensors.insert(
            format!("{h}.mlp.c_fc.weight"),
            entry(&format!("{h}.mlp.c_fc.weight"), vec![ff, d], vec![0.01; ff * d]),
        );
        tensors.insert(format!("{h}.mlp.c_fc.bias"), entry(&format!("{h}.mlp.c_fc.bias"), vec![ff], vec![0.0; ff]));
        tensors.insert(
            format!("{h}.mlp.c_proj.weight"),
            entry(&format!("{h}.mlp.c_proj.weight"), vec![d, ff], vec![0.01; d * ff]),
        );
        tensors.insert(format!("{h}.mlp.c_proj.bias"), entry(&format!("{h}.mlp.c_proj.bias"), vec![d], vec![0.0; d]));

        tensors.insert(format!("{p}.ln_f.weight"), entry(&format!("{p}.ln_f.weight"), vec![d], vec![1.0; d]));
        tensors.insert(format!("{p}.ln_f.bias"), entry(&format!("{p}.ln_f.bias"), vec![d], vec![0.0; d]));

        // Dummy attention tensors (ignored by loader but present in real checkpoints)
        tensors.insert(
            format!("{h}.attn.c_attn.weight"),
            entry(&format!("{h}.attn.c_attn.weight"), vec![3 * d, d], vec![0.001; 3 * d * d]),
        );
        tensors.insert(
            format!("{h}.attn.c_attn.bias"),
            entry(&format!("{h}.attn.c_attn.bias"), vec![3 * d], vec![0.0; 3 * d]),
        );
        tensors.insert(
            format!("{h}.attn.c_proj.weight"),
            entry(&format!("{h}.attn.c_proj.weight"), vec![d, d], vec![0.001; d * d]),
        );
        tensors.insert(
            format!("{h}.attn.c_proj.bias"),
            entry(&format!("{h}.attn.c_proj.bias"), vec![d], vec![0.0; d]),
        );

        let meta = MetaStateDict { tensors };

        let cfg = HfGpt2Config {
            vocab_size: vocab,
            n_positions: 32,
            n_embd: d,
            n_layer,
            n_head,
            resid_pdrop: Some(0.0),
        };

        let backend = CpuBackend::default();
        let mut dec_cfg = TransformerDecoderConfig::new(d, n_head, n_layer, ff).with_max_seq_len(32);
        dec_cfg.dropout = 0.0;
        let mut model = TransformerDecoder::new(&backend, dec_cfg, vocab, 42).expect("decoder");

        let report = load_hf_gpt2_weights_into_decoder(&mut model, &backend, &meta, &cfg).expect("load");
        assert!(report.loaded_rustral_keys.contains(&"token_embedding.embed".to_string()));
        assert!(report.loaded_rustral_keys.contains(&"layers.0.ff_linear1.weight".to_string()));
        assert_eq!(report.skipped_attention_parameters.len(), 4);
    }
}
