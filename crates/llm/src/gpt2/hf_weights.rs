//! Hugging Face GPT-2 Safetensors → Rustral `TransformerDecoder` weight loading.
//!
//! # Supported tensors (shape-compatible paths)
//!
//! - `token_embedding.embed` ← `{prefix}.wte.weight`
//! - `lm_head.weight` ← `lm_head.weight`
//! - `layers.{i}.norm1.*`, `norm2.*` ← `h.{i}.ln_1`, `ln_2`
//! - `layers.{i}.ff_linear1.*`, `ff_linear2.*` ← `mlp.c_fc`, `mlp.c_proj` (weights are accepted as `[out, in]` or the HF-transposed layout and normalized to our `[out_dim, in_dim]` matrices)
//! - `layers.{i}.self_attn.{q,k,v,out}_proj.*` ← `attn.c_attn` (split into three `[d_model, d_model]` blocks) and `attn.c_proj`
//! - `final_norm.*` ← `{prefix}.ln_f.*`
//!
//! # Attention layout
//!
//! Hugging Face `Conv1D` stores
//! `c_attn.weight` as **[nf, nx]** (often `[3 * d_model, d_model]`) with Q/K/V stacked as **row blocks**.
//! Some checkpoints use the transposed **`[d_model, 3 * d_model]`** layout; both are accepted.
//! [`rustral_nn::SelfAttention`] uses four [`rustral_nn::Linear`] maps (`q_proj`, `k_proj`, `v_proj`, `out_proj`), each
//! **`[out_dim, in_dim]`** matching the CPU `TensorOps::linear` convention (`rustral_core`).
//!
//! # Still not loaded
//!
//! - Learned positional embeddings (`wpe`) — our decoder uses sinusoidal positional encoding in `rustral_nn`.

use std::collections::{HashMap, HashSet};

use rustral_core::{Backend, NamedParameters, Parameter};
use rustral_io::{MetaStateDict, TensorEntry};
use rustral_ndarray_backend::CpuBackend;
use rustral_nn::TransformerDecoder;

use super::HfGpt2Config;
use crate::checkpoint_f32::{
    insert_hf_tensor_if_present, tensor_entry_f32, tensor_entry_f32_rustral_matrix, vec_f32_to_tensor_entry,
};
use crate::LlmError;

/// Result of [`load_hf_gpt2_weights_into_decoder`].
#[derive(Debug, Clone, Default)]
pub struct Gpt2WeightLoadReport {
    /// Rustral parameter names updated from the checkpoint.
    pub loaded_rustral_keys: Vec<String>,
    /// `self_attn.*` parameter paths still missing after load (empty when full HF attention tensors were mapped).
    pub skipped_attention_parameters: Vec<String>,
    /// HF keys present in the checkpoint that this loader does not map into the decoder (e.g. `wpe`, unused `attn.*` shards).
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

/// Normalize a `[d_model, d_model]` slice from HF into [`rustral_nn::Linear`] storage `[out_dim, in_dim]`.
fn hf_square_block_to_linear_weight(
    block: Vec<f32>,
    d_model: usize,
    debug_tensor_name: &str,
) -> Result<(Vec<f32>, Vec<usize>), LlmError> {
    let sh = [d_model, d_model];
    let entry = vec_f32_to_tensor_entry(debug_tensor_name.to_string(), sh.to_vec(), block);
    tensor_entry_f32_rustral_matrix(&entry, sh)
}

/// Split HF `attn.c_attn.weight` into Q/K/V matrices (each `[d_model, d_model]` in HF row-major storage).
///
/// Accepted layouts: **`[3*d_model, d_model]`** (Q/K/V row blocks) or **`[d_model, 3*d_model]`** (column blocks).
fn split_gpt2_c_attn_weight(entry: &TensorEntry, d_model: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), LlmError> {
    let data = tensor_entry_f32(entry)?;
    let shape = entry.shape.as_slice();
    let nf = 3 * d_model;
    let nx = d_model;

    let block = |rows: std::ops::Range<usize>, cols: std::ops::Range<usize>, stride_cols: usize| -> Vec<f32> {
        let nrows = rows.end - rows.start;
        let ncols = cols.end - cols.start;
        let mut out = vec![0f32; nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                let r = rows.start + i;
                let c = cols.start + j;
                out[i * ncols + j] = data[r * stride_cols + c];
            }
        }
        out
    };

    if shape == [nf, nx] {
        let q = block(0..d_model, 0..nx, nx);
        let k = block(d_model..2 * d_model, 0..nx, nx);
        let v = block(2 * d_model..nf, 0..nx, nx);
        return Ok((q, k, v));
    }
    if shape == [nx, nf] {
        let q = block(0..nx, 0..d_model, nf);
        let k = block(0..nx, d_model..2 * d_model, nf);
        let v = block(0..nx, 2 * d_model..nf, nf);
        return Ok((q, k, v));
    }

    Err(LlmError::InvalidArg(format!(
        "tensor '{}': c_attn.weight expected {:?} or {:?}, got {:?}",
        entry.name,
        [nf, nx],
        [nx, nf],
        shape
    )))
}

/// Split HF `attn.c_attn.bias` of length `3 * d_model` into Q/K/V bias vectors.
fn split_gpt2_c_attn_bias(data: &[f32], d_model: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), LlmError> {
    let n = 3 * d_model;
    if data.len() != n {
        return Err(LlmError::InvalidArg(format!(
            "expected c_attn.bias length {}, got {}",
            n,
            data.len()
        )));
    }
    Ok((
        data[0..d_model].to_vec(),
        data[d_model..2 * d_model].to_vec(),
        data[2 * d_model..n].to_vec(),
    ))
}

/// Build `(rustral_name -> data, shape)` for every parameter we can take verbatim from HF GPT-2 dumps.
pub fn build_gpt2_flat_map(
    meta: &MetaStateDict,
    cfg: &HfGpt2Config,
    prefix: &str,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, LlmError> {
    let mut out: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

    insert_hf_tensor_if_present(&mut out, meta, "token_embedding.embed", &format!("{prefix}.wte.weight"))?;
    insert_hf_tensor_if_present(&mut out, meta, "lm_head.weight", "lm_head.weight")?;

    if cfg.n_positions > 0 {
        // Learned position embeddings exist in HF but our decoder uses sinusoidal `PositionalEncoding` (no weights).
        let _ = meta.tensors.get(&format!("{prefix}.wpe.weight"));
    }

    let d_model = cfg.n_embd;
    let ff_dim = cfg.n_embd.saturating_mul(4);

    for i in 0..cfg.n_layer {
        let h = format!("{prefix}.h.{i}");
        insert_hf_tensor_if_present(&mut out, meta, &format!("layers.{i}.norm1.weight"), &format!("{h}.ln_1.weight"))?;
        insert_hf_tensor_if_present(&mut out, meta, &format!("layers.{i}.norm1.bias"), &format!("{h}.ln_1.bias"))?;
        insert_hf_tensor_if_present(&mut out, meta, &format!("layers.{i}.norm2.weight"), &format!("{h}.ln_2.weight"))?;
        insert_hf_tensor_if_present(&mut out, meta, &format!("layers.{i}.norm2.bias"), &format!("{h}.ln_2.bias"))?;

        let k_fc_w = format!("{h}.mlp.c_fc.weight");
        if let Some(e) = meta.tensors.get(&k_fc_w) {
            let (data, sh) = tensor_entry_f32_rustral_matrix(e, [ff_dim, d_model])?;
            out.insert(format!("layers.{i}.ff_linear1.weight"), (data, sh));
        }
        insert_hf_tensor_if_present(&mut out, meta, &format!("layers.{i}.ff_linear1.bias"), &format!("{h}.mlp.c_fc.bias"))?;

        let k_proj_w = format!("{h}.mlp.c_proj.weight");
        if let Some(e) = meta.tensors.get(&k_proj_w) {
            let (data, sh) = tensor_entry_f32_rustral_matrix(e, [d_model, ff_dim])?;
            out.insert(format!("layers.{i}.ff_linear2.weight"), (data, sh));
        }
        insert_hf_tensor_if_present(&mut out, meta, &format!("layers.{i}.ff_linear2.bias"), &format!("{h}.mlp.c_proj.bias"))?;

        let k_c_attn_w = format!("{h}.attn.c_attn.weight");
        let k_c_attn_b = format!("{h}.attn.c_attn.bias");
        match (meta.tensors.get(&k_c_attn_w), meta.tensors.get(&k_c_attn_b)) {
            (Some(we), Some(be)) => {
                let (qw, kw, vw) = split_gpt2_c_attn_weight(we, d_model)?;
                let bias_f = tensor_entry_f32(be)?;
                let (qb, kb, vb) = split_gpt2_c_attn_bias(&bias_f, d_model)?;
                let (qw, sh_q) = hf_square_block_to_linear_weight(qw, d_model, &k_c_attn_w)?;
                let (kw, sh_k) = hf_square_block_to_linear_weight(kw, d_model, &k_c_attn_w)?;
                let (vw, sh_v) = hf_square_block_to_linear_weight(vw, d_model, &k_c_attn_w)?;
                out.insert(format!("layers.{i}.self_attn.q_proj.weight"), (qw, sh_q));
                out.insert(format!("layers.{i}.self_attn.k_proj.weight"), (kw, sh_k));
                out.insert(format!("layers.{i}.self_attn.v_proj.weight"), (vw, sh_v));
                out.insert(format!("layers.{i}.self_attn.q_proj.bias"), (qb, vec![d_model]));
                out.insert(format!("layers.{i}.self_attn.k_proj.bias"), (kb, vec![d_model]));
                out.insert(format!("layers.{i}.self_attn.v_proj.bias"), (vb, vec![d_model]));
            }
            (None, None) => {}
            _ => {
                return Err(LlmError::InvalidArg(format!(
                    "{h}: attn.c_attn.weight and attn.c_attn.bias must both be present or both absent"
                )));
            }
        }

        let k_cproj_w = format!("{h}.attn.c_proj.weight");
        let k_cproj_b = format!("{h}.attn.c_proj.bias");
        match (meta.tensors.get(&k_cproj_w), meta.tensors.get(&k_cproj_b)) {
            (Some(we), Some(be)) => {
                let (data, sh) = tensor_entry_f32_rustral_matrix(we, [d_model, d_model])?;
                let bdata = tensor_entry_f32(be)?;
                if bdata.len() != d_model {
                    return Err(LlmError::InvalidArg(format!(
                        "tensor '{}': expected bias len {}, got {}",
                        be.name,
                        d_model,
                        bdata.len()
                    )));
                }
                out.insert(format!("layers.{i}.self_attn.out_proj.weight"), (data, sh));
                out.insert(format!("layers.{i}.self_attn.out_proj.bias"), (bdata, vec![d_model]));
            }
            (None, None) => {}
            _ => {
                return Err(LlmError::InvalidArg(format!(
                    "{h}: attn.c_proj.weight and attn.c_proj.bias must both be present or both absent"
                )));
            }
        }
    }

    insert_hf_tensor_if_present(&mut out, meta, "final_norm.weight", &format!("{prefix}.ln_f.weight"))?;
    insert_hf_tensor_if_present(&mut out, meta, "final_norm.bias", &format!("{prefix}.ln_f.bias"))?;

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

fn attention_linear_parameter_names(n_layer: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(n_layer * 8);
    for i in 0..n_layer {
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            v.push(format!("layers.{i}.self_attn.{proj}.weight"));
            v.push(format!("layers.{i}.self_attn.{proj}.bias"));
        }
    }
    v
}

/// Load compatible GPT-2 tensors from `meta` into `model`.
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

    let loaded_set: HashSet<String> = loaded.iter().cloned().collect();
    let mut skipped_attention_parameters: Vec<String> = attention_linear_parameter_names(cfg.n_layer)
        .into_iter()
        .filter(|k| !loaded_set.contains(k))
        .collect();
    skipped_attention_parameters.sort();

    let mut unmapped_hf_keys: Vec<String> = meta
        .tensors
        .keys()
        .filter(|k| {
            let k = k.as_str();
            if k.contains(".attn.c_attn.") || k.contains(".attn.c_proj.") {
                return false;
            }
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
    use safetensors::Dtype;

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
        let ff = d.saturating_mul(4);
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

        // HF-style shapes `[d_model, ff_dim]` / `[ff_dim, d_model]` (transposed vs our Linear storage).
        tensors.insert(
            format!("{h}.mlp.c_fc.weight"),
            entry(&format!("{h}.mlp.c_fc.weight"), vec![d, ff], vec![0.01; d * ff]),
        );
        tensors.insert(format!("{h}.mlp.c_fc.bias"), entry(&format!("{h}.mlp.c_fc.bias"), vec![ff], vec![0.0; ff]));
        tensors.insert(
            format!("{h}.mlp.c_proj.weight"),
            entry(&format!("{h}.mlp.c_proj.weight"), vec![ff, d], vec![0.01; ff * d]),
        );
        tensors.insert(format!("{h}.mlp.c_proj.bias"), entry(&format!("{h}.mlp.c_proj.bias"), vec![d], vec![0.0; d]));

        tensors.insert(format!("{p}.ln_f.weight"), entry(&format!("{p}.ln_f.weight"), vec![d], vec![1.0; d]));
        tensors.insert(format!("{p}.ln_f.bias"), entry(&format!("{p}.ln_f.bias"), vec![d], vec![0.0; d]));

        // HF `Conv1D` c_attn: row-stacked Q/K/V (`[3*d, d]`, most common) or transposed (`[d, 3*d]`)
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
        let mut dec_cfg = TransformerDecoderConfig::new(d, n_head, n_layer, ff).with_max_seq_len(32); // ff = 4 * d (GPT-2 MLP)
        dec_cfg.dropout = 0.0;
        let mut model = TransformerDecoder::new(&backend, dec_cfg, vocab, 42).expect("decoder");

        let report = load_hf_gpt2_weights_into_decoder(&mut model, &backend, &meta, &cfg).expect("load");
        assert!(report.loaded_rustral_keys.contains(&"token_embedding.embed".to_string()));
        assert!(report.loaded_rustral_keys.contains(&"layers.0.ff_linear1.weight".to_string()));
        assert!(
            report.skipped_attention_parameters.is_empty(),
            "expected attention weights loaded; skipped: {:?}",
            report.skipped_attention_parameters
        );
        assert!(report.loaded_rustral_keys.iter().any(|k| k.ends_with("self_attn.q_proj.weight")));
        assert!(report.loaded_rustral_keys.iter().any(|k| k.ends_with("self_attn.out_proj.bias")));
    }

    #[test]
    fn loads_c_attn_d_by_3d_fixture() {
        let d = 8usize;
        let ff = d.saturating_mul(4);
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
            entry(&format!("{h}.mlp.c_fc.weight"), vec![d, ff], vec![0.01; d * ff]),
        );
        tensors.insert(format!("{h}.mlp.c_fc.bias"), entry(&format!("{h}.mlp.c_fc.bias"), vec![ff], vec![0.0; ff]));
        tensors.insert(
            format!("{h}.mlp.c_proj.weight"),
            entry(&format!("{h}.mlp.c_proj.weight"), vec![ff, d], vec![0.01; ff * d]),
        );
        tensors.insert(format!("{h}.mlp.c_proj.bias"), entry(&format!("{h}.mlp.c_proj.bias"), vec![d], vec![0.0; d]));

        tensors.insert(format!("{p}.ln_f.weight"), entry(&format!("{p}.ln_f.weight"), vec![d], vec![1.0; d]));
        tensors.insert(format!("{p}.ln_f.bias"), entry(&format!("{p}.ln_f.bias"), vec![d], vec![0.0; d]));

        tensors.insert(
            format!("{h}.attn.c_attn.weight"),
            entry(&format!("{h}.attn.c_attn.weight"), vec![d, 3 * d], vec![0.001; d * 3 * d]),
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
        assert!(report.skipped_attention_parameters.is_empty());
        assert!(report.loaded_rustral_keys.iter().any(|k| k.ends_with("self_attn.v_proj.weight")));
    }
}
