//! Hugging Face Llama safetensors → Rustral [`rustral_nn::LlamaDecoder`] weight loading.
//!
//! Maps **`model.*`** keys from [`LlamaForCausalLM`](https://huggingface.co/docs/transformers/model_doc/llama)-style
//! checkpoints onto our [`NamedParameters`] tree (`token_embedding.embed`, `layers.{i}.*`, `norm.weight`, `lm_head.weight`).
//! Weight matrices accept HF **`[out_features, in_features]`** or the transposed layout and normalize to
//! **`[out_dim, in_dim]`** for [`rustral_nn::Linear`].
//!
//! # Not loaded / ignored for parity reporting
//!
//! - Rotary embedding caches (`rotary_emb`, `inv_freq`) — RoPE is computed in-graph from `rope_theta`.
//! - Quantized checkpoints (non‑F32 dtypes).

use std::collections::{HashMap, HashSet};

use rustral_core::{Backend, NamedParameters, Parameter};
use rustral_io::MetaStateDict;
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::LlamaDecoder;

use super::HfLlamaConfig;
use crate::checkpoint_f32::{insert_hf_tensor_if_present, tensor_entry_f32_rustral_matrix};
use crate::LlmError;

/// Result of [`load_hf_llama_weights_into_decoder`].
#[derive(Debug, Clone, Default)]
pub struct LlamaWeightLoadReport {
    /// Rustral parameter names updated from the checkpoint.
    pub loaded_rustral_keys: Vec<String>,
    /// Parameter paths still missing after load (checkpoint incomplete vs model).
    pub skipped_parameters: Vec<String>,
    /// HF tensor keys present but not mapped into this decoder (includes rotary stubs, extras).
    pub unmapped_hf_keys: Vec<String>,
}

/// Detect HF state dict root (`model` vs bare keys).
pub fn detect_llama_state_dict_root(meta: &MetaStateDict) -> Result<&'static str, LlmError> {
    if meta.tensors.keys().any(|k| k.starts_with("model.embed_tokens.") || k.starts_with("model.layers.")) {
        return Ok("model");
    }
    if meta.tensors.keys().any(|k| k.starts_with("embed_tokens.") && k.contains(".weight"))
        || meta.tensors.keys().any(|k| k.starts_with("layers.") && k.contains(".self_attn."))
    {
        return Ok("");
    }
    Err(LlmError::InvalidArg(
        "could not detect Llama state dict root (expected model.embed_tokens / model.layers.*)".to_string(),
    ))
}

fn hf_join(root: &str, tail: &str) -> String {
    if root.is_empty() {
        tail.to_string()
    } else {
        format!("{root}.{tail}")
    }
}

/// Build `(rustral_name -> data, shape)` from HF Llama tensors; records **consumed** HF keys for reporting.
pub fn build_llama_flat_map(
    meta: &MetaStateDict,
    cfg: &HfLlamaConfig,
    root: &str,
) -> Result<(HashMap<String, (Vec<f32>, Vec<usize>)>, HashSet<String>), LlmError> {
    cfg.validate_supported()?;

    let mut out: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut consumed: HashSet<String> = HashSet::new();

    let d = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let n_layer = cfg.num_hidden_layers;
    let head_dim = d / cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let kv_dim = n_kv * head_dim;

    let embed_candidates = [hf_join(root, "embed_tokens.weight"), "embed_tokens.weight".to_string()];
    for cand in embed_candidates.iter() {
        if let Some(e) = meta.tensors.get(cand.as_str()) {
            consumed.insert(cand.clone());
            let (data, sh) = tensor_entry_f32_rustral_matrix(e, [cfg.vocab_size, d])?;
            out.insert("token_embedding.embed".to_string(), (data, sh));
            break;
        }
    }

    insert_or_track(&mut out, &mut consumed, meta, "norm.weight", &hf_join(root, "norm.weight"))?;

    let lm_candidates = [hf_join(root, "lm_head.weight"), "lm_head.weight".to_string()];
    for cand in lm_candidates.iter() {
        if meta.tensors.contains_key(cand.as_str()) {
            insert_or_track(&mut out, &mut consumed, meta, "lm_head.weight", cand.as_str())?;
            break;
        }
    }

    for i in 0..n_layer {
        let layer = hf_join(root, &format!("layers.{i}"));

        insert_or_track(
            &mut out,
            &mut consumed,
            meta,
            &format!("layers.{i}.input_layernorm.weight"),
            &format!("{layer}.input_layernorm.weight"),
        )?;
        insert_or_track(
            &mut out,
            &mut consumed,
            meta,
            &format!("layers.{i}.post_attention_layernorm.weight"),
            &format!("{layer}.post_attention_layernorm.weight"),
        )?;

        for (rust_proj, hf_name, expected_rows) in [
            ("q_proj", "q_proj", d),
            ("k_proj", "k_proj", kv_dim),
            ("v_proj", "v_proj", kv_dim),
            ("o_proj", "o_proj", d),
        ] {
            let hf_k = format!("{layer}.self_attn.{hf_name}.weight");
            if let Some(e) = meta.tensors.get(&hf_k) {
                consumed.insert(hf_k);
                let (data, sh) = tensor_entry_f32_rustral_matrix(e, [expected_rows, d])?;
                out.insert(format!("layers.{i}.self_attn.{rust_proj}.weight"), (data, sh));
            }
        }

        for (rust_name, hf_suffix, rustral_shape) in [
            ("gate_proj", "gate_proj", [inter, d]),
            ("up_proj", "up_proj", [inter, d]),
            ("down_proj", "down_proj", [d, inter]),
        ] {
            let hf_k = format!("{layer}.mlp.{hf_suffix}.weight");
            if let Some(e) = meta.tensors.get(&hf_k) {
                consumed.insert(hf_k);
                let (data, sh) = tensor_entry_f32_rustral_matrix(e, rustral_shape)?;
                out.insert(format!("layers.{i}.mlp.{rust_name}.weight"), (data, sh));
            }
        }
    }

    if !out.contains_key("token_embedding.embed") {
        return Err(LlmError::MissingFile(hf_join(root, "embed_tokens.weight")));
    }

    Ok((out, consumed))
}

fn insert_or_track(
    out: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    consumed: &mut HashSet<String>,
    meta: &MetaStateDict,
    rustral: &str,
    hf: &str,
) -> Result<(), LlmError> {
    if !meta.tensors.contains_key(hf) {
        return Ok(());
    }
    consumed.insert(hf.to_string());
    insert_hf_tensor_if_present(out, meta, rustral, hf)
}

fn collect_llama_param_shapes<B: Backend>(
    model: &LlamaDecoder<B>,
    backend: &B,
) -> HashMap<String, Vec<usize>> {
    let ops = backend.ops();
    let mut m = HashMap::new();
    model.visit_parameters(&mut |name, p: &Parameter<B>| {
        m.insert(name.to_string(), ops.shape(p.tensor()));
    });
    m
}

/// Apply a pre-built Llama `(data, shape)` map (from [`build_llama_flat_map`]) onto any decoder
/// instance. Used for tests (e.g. Candle parity) and advanced loaders.
pub fn apply_llama_flat_map_to_decoder<B: Backend>(
    model: &mut LlamaDecoder<B>,
    backend: &B,
    flat: &HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<(), LlmError>
where
    B::Tensor: Clone,
{
    let model_shapes = collect_llama_param_shapes(model, backend);
    let ops = backend.ops();

    for (name, (data, shape)) in flat {
        let expected = model_shapes.get(name).ok_or_else(|| {
            LlmError::InvalidArg(format!(
                "checkpoint has unexpected parameter name '{name}' for this Llama model"
            ))
        })?;
        if expected != shape {
            return Err(LlmError::CheckpointShapeMismatch {
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

    let mut materialized: HashMap<String, B::Tensor> = HashMap::new();
    for (name, (data, shape)) in flat {
        let t = ops
            .tensor_from_vec(data.clone(), shape)
            .map_err(|e| LlmError::InvalidArg(format!("tensor_from_vec failed for '{name}': {e:?}")))?;
        materialized.insert(name.clone(), t);
    }

    model.visit_parameters_mut(&mut |name, p: &mut Parameter<B>| {
        if let Some(t) = materialized.get(name) {
            *p = p.clone().with_tensor(t.clone());
        }
    });

    Ok(())
}

fn all_expected_parameter_names(n_layer: usize) -> Vec<String> {
    let mut v = Vec::new();
    v.push("token_embedding.embed".to_string());
    v.push("norm.weight".to_string());
    v.push("lm_head.weight".to_string());
    for i in 0..n_layer {
        v.push(format!("layers.{i}.input_layernorm.weight"));
        v.push(format!("layers.{i}.post_attention_layernorm.weight"));
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            v.push(format!("layers.{i}.self_attn.{proj}.weight"));
        }
        for m in ["gate_proj", "up_proj", "down_proj"] {
            v.push(format!("layers.{i}.mlp.{m}.weight"));
        }
    }
    v
}

/// Load compatible Llama F32 tensors from `meta` into `model`.
pub fn load_hf_llama_weights_into_decoder(
    model: &mut LlamaDecoder<CpuBackend>,
    backend: &CpuBackend,
    meta: &MetaStateDict,
    cfg: &HfLlamaConfig,
) -> Result<LlamaWeightLoadReport, LlmError> {
    let root = detect_llama_state_dict_root(meta)?;
    let (flat, consumed_hf) = build_llama_flat_map(meta, cfg, root)?;

    apply_llama_flat_map_to_decoder(model, backend, &flat)?;

    let mut loaded: Vec<String> = flat.keys().cloned().collect();
    loaded.sort();

    let loaded_set: HashSet<String> = loaded.iter().cloned().collect();
    let mut skipped_parameters: Vec<String> = all_expected_parameter_names(cfg.num_hidden_layers)
        .into_iter()
        .filter(|k| !loaded_set.contains(k))
        .collect();
    skipped_parameters.sort();

    let mut unmapped_hf_keys: Vec<String> = meta
        .tensors
        .keys()
        .filter(|k| {
            if consumed_hf.contains(k.as_str()) {
                return false;
            }
            let k = k.as_str();
            if k.starts_with("lm_head.") {
                return true;
            }
            if root.is_empty() {
                return k.starts_with("layers.") || k.starts_with("embed_tokens.") || k.starts_with("norm.");
            }
            k.starts_with(&format!("{root}."))
        })
        .cloned()
        .collect();
    unmapped_hf_keys.sort();

    Ok(LlamaWeightLoadReport { loaded_rustral_keys: loaded, skipped_parameters, unmapped_hf_keys })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustral_io::{MetaStateDict, TensorEntry};
    use safetensors::Dtype;

    fn entry(name: &str, shape: Vec<usize>, data: Vec<f32>) -> TensorEntry {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        TensorEntry { name: name.to_string(), shape, dtype: Dtype::F32, data: bytes }
    }

    #[test]
    fn loads_tiny_llama_fixture() {
        let d = 16usize;
        let inter = 32usize;
        let vocab = 20usize;
        let n_layer = 1usize;
        let n_head = 4usize;

        let mut tensors = HashMap::new();
        let root = "model";
        tensors.insert(
            format!("{root}.embed_tokens.weight"),
            entry("embed", vec![vocab, d], (0..vocab * d).map(|i| (i as f32) * 1e-4).collect()),
        );
        tensors.insert(format!("{root}.norm.weight"), entry("norm", vec![d], vec![1.0f32; d]));
        tensors
            .insert("lm_head.weight".to_string(), entry("lm_head", vec![vocab, d], vec![0.01f32; vocab * d]));

        let h = format!("{root}.layers.0");
        tensors.insert(format!("{h}.input_layernorm.weight"), entry("iln", vec![d], vec![1.0f32; d]));
        tensors
            .insert(format!("{h}.post_attention_layernorm.weight"), entry("paln", vec![d], vec![1.0f32; d]));
        for (pn, sz) in [("q_proj", [d, d]), ("k_proj", [d, d]), ("v_proj", [d, d]), ("o_proj", [d, d])] {
            tensors
                .insert(format!("{h}.self_attn.{pn}.weight"), entry(&pn, sz.to_vec(), vec![0.001f32; d * d]));
        }
        tensors.insert(
            format!("{h}.mlp.gate_proj.weight"),
            entry("gate", vec![inter, d], vec![0.002f32; inter * d]),
        );
        tensors.insert(
            format!("{h}.mlp.up_proj.weight"),
            entry("up", vec![inter, d], vec![0.002f32; inter * d]),
        );
        tensors.insert(
            format!("{h}.mlp.down_proj.weight"),
            entry("down", vec![d, inter], vec![0.002f32; d * inter]),
        );

        tensors.insert(
            format!("{root}.layers.0.self_attn.rotary_emb.inv_freq"),
            entry("inv", vec![d / 2], vec![1.0f32; d / 2]),
        );

        let meta = MetaStateDict { tensors };

        let cfg = HfLlamaConfig {
            vocab_size: vocab,
            hidden_size: d,
            intermediate_size: inter,
            num_hidden_layers: n_layer,
            num_attention_heads: n_head,
            num_key_value_heads: None,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: Some(64),
        };

        let backend = CpuBackend::default();
        let dec_cfg = cfg.to_decoder_config();
        let mut model = LlamaDecoder::new(&backend, dec_cfg, vocab, 42).expect("decoder");

        let report = load_hf_llama_weights_into_decoder(&mut model, &backend, &meta, &cfg).expect("load");
        assert!(report.loaded_rustral_keys.iter().any(|k| k.contains("self_attn.q_proj.weight")));
        assert!(report.skipped_parameters.is_empty());
        assert!(
            report.unmapped_hf_keys.iter().any(|k| k.contains("rotary_emb")),
            "expected rotary inv_freq listed as unmapped: {:?}",
            report.unmapped_hf_keys
        );

        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let logits = model.forward(vec![1usize, 2], &mut ctx).expect("forward");
        assert_eq!(backend.ops().shape(&logits), &[1, 2, vocab]);
    }

    #[test]
    fn gqa_kv_proj_shapes_use_kv_dim() {
        let d = 64usize;
        let n_head = 8usize;
        let n_kv = 4usize;
        let head_dim = d / n_head;
        let kv_dim = n_kv * head_dim;

        let mut tensors = HashMap::new();
        let root = "model";
        tensors.insert(
            format!("{root}.embed_tokens.weight"),
            entry("embed", vec![100, d], vec![0.001f32; 100 * d]),
        );
        tensors.insert(format!("{root}.norm.weight"), entry("norm", vec![d], vec![1.0f32; d]));
        tensors.insert("lm_head.weight".to_string(), entry("lm_head", vec![100, d], vec![0.01f32; 100 * d]));

        let h = format!("{root}.layers.0");
        tensors.insert(format!("{h}.input_layernorm.weight"), entry("iln", vec![d], vec![1.0f32; d]));
        tensors
            .insert(format!("{h}.post_attention_layernorm.weight"), entry("paln", vec![d], vec![1.0f32; d]));
        tensors.insert(format!("{h}.self_attn.q_proj.weight"), entry("q", vec![d, d], vec![0.001f32; d * d]));
        tensors.insert(
            format!("{h}.self_attn.k_proj.weight"),
            entry("k", vec![kv_dim, d], vec![0.001f32; kv_dim * d]),
        );
        tensors.insert(
            format!("{h}.self_attn.v_proj.weight"),
            entry("v", vec![kv_dim, d], vec![0.001f32; kv_dim * d]),
        );
        tensors.insert(format!("{h}.self_attn.o_proj.weight"), entry("o", vec![d, d], vec![0.001f32; d * d]));
        for (name, sz) in
            [("gate_proj", vec![128, d]), ("up_proj", vec![128, d]), ("down_proj", vec![d, 128])]
        {
            let el = sz.iter().product::<usize>();
            tensors.insert(format!("{h}.mlp.{name}.weight"), entry(name, sz, vec![0.002f32; el]));
        }

        let meta = MetaStateDict { tensors };

        let cfg = HfLlamaConfig {
            vocab_size: 100,
            hidden_size: d,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: n_head,
            num_key_value_heads: Some(n_kv),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: Some(128),
        };

        let backend = CpuBackend::default();
        let dec_cfg = cfg.to_decoder_config();
        let mut model = LlamaDecoder::new(&backend, dec_cfg, cfg.vocab_size, 42).expect("decoder");

        let report = load_hf_llama_weights_into_decoder(&mut model, &backend, &meta, &cfg).expect("load");
        assert!(report.skipped_parameters.is_empty());

        let mut ctx = rustral_core::ForwardCtx::new(&backend, rustral_core::Mode::Inference);
        let logits = model.forward(vec![1usize, 2, 3], &mut ctx).expect("forward");
        assert_eq!(backend.ops().shape(&logits), &[1, 3, cfg.vocab_size]);
    }
}
