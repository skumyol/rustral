//! Integration tests: **real on-disk** `config.json` under `tests/fixtures/` plus synthetic HF-shaped tensors.
//!
//! Builds a full [`MetaStateDict`] matching the fixture dimensions and verifies
//! [`LlamaCausalLm::from_hf_meta`] + forward pass + causal prefix/full logits parity.

use std::collections::HashMap;
use std::path::PathBuf;

use rustral_core::Backend;
use rustral_io::{MetaStateDict, TensorEntry};
use rustral_llm::{HfLlamaConfig, LlamaCausalLm};
use safetensors::Dtype;

fn fixture_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tiny_llama")
        .join("config.json")
}

fn tensor_entry(name: &str, shape: Vec<usize>, data: Vec<f32>) -> TensorEntry {
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

/// Synthetic Llama checkpoint tensors aligned with `cfg` (HF `model.*` naming).
fn synthetic_meta_state_dict(cfg: &HfLlamaConfig) -> MetaStateDict {
    let d = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let n_layer = cfg.num_hidden_layers;

    let mut tensors = HashMap::new();
    let root = "model";

    tensors.insert(
        format!("{root}.embed_tokens.weight"),
        tensor_entry(
            "embed_tokens.weight",
            vec![vocab, d],
            (0..vocab * d).map(|i| (i as f32) * 1e-5).collect(),
        ),
    );
    tensors.insert(
        format!("{root}.norm.weight"),
        tensor_entry("norm.weight", vec![d], vec![1.0f32; d]),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        tensor_entry("lm_head.weight", vec![vocab, d], vec![0.01f32; vocab * d]),
    );

    for layer in 0..n_layer {
        let h = format!("{root}.layers.{layer}");
        tensors.insert(
            format!("{h}.input_layernorm.weight"),
            tensor_entry("iln", vec![d], vec![1.0f32; d]),
        );
        tensors.insert(
            format!("{h}.post_attention_layernorm.weight"),
            tensor_entry("paln", vec![d], vec![1.0f32; d]),
        );
        for pn in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.insert(
                format!("{h}.self_attn.{pn}.weight"),
                tensor_entry(pn, vec![d, d], vec![0.001f32; d * d]),
            );
        }
        tensors.insert(
            format!("{h}.mlp.gate_proj.weight"),
            tensor_entry("gate", vec![inter, d], vec![0.002f32; inter * d]),
        );
        tensors.insert(
            format!("{h}.mlp.up_proj.weight"),
            tensor_entry("up", vec![inter, d], vec![0.002f32; inter * d]),
        );
        tensors.insert(
            format!("{h}.mlp.down_proj.weight"),
            tensor_entry("down", vec![d, inter], vec![0.002f32; d * inter]),
        );
        tensors.insert(
            format!("{h}.self_attn.rotary_emb.inv_freq"),
            tensor_entry("inv_freq", vec![d / 2], vec![1.0f32; d / 2]),
        );
    }

    MetaStateDict { tensors }
}

#[test]
fn tiny_llama_config_json_loads_from_disk() {
    let cfg = HfLlamaConfig::from_json_file(fixture_config_path()).expect("fixture config.json");
    assert_eq!(cfg.vocab_size, 64);
    assert_eq!(cfg.hidden_size, 32);
    assert_eq!(cfg.intermediate_size, 48);
    assert_eq!(cfg.num_hidden_layers, 1);
    assert_eq!(cfg.num_attention_heads, 4);
    assert_eq!(cfg.max_position_embeddings, Some(128));
}

#[test]
fn llama_load_fixture_meta_forward_and_causal_parity() {
    let cfg = HfLlamaConfig::from_json_file(fixture_config_path()).expect("fixture");
    let meta = synthetic_meta_state_dict(&cfg);

    let (lm, report) = LlamaCausalLm::from_hf_meta(&cfg, &meta, 42).expect("from_hf_meta");
    assert!(
        report.skipped_parameters.is_empty(),
        "skipped: {:?}",
        report.skipped_parameters
    );
    assert!(
        report.loaded_rustral_keys.contains(&"token_embedding.embed".to_string()),
        "{:?}",
        report.loaded_rustral_keys
    );

    let backend = lm.backend();
    let ops = backend.ops();
    let vocab = cfg.vocab_size;
    let tokens: Vec<usize> = vec![3, 11, 7, 2, 19];

    let mut ctx = rustral_core::ForwardCtx::new(backend, rustral_core::Mode::Inference);
    let full_logits = lm.model().forward(tokens.clone(), &mut ctx).expect("forward full");
    assert_eq!(ops.shape(&full_logits), &[1, tokens.len(), vocab]);

    let full_flat = ops.tensor_to_vec(&full_logits).expect("full vec");
    let sl = vocab;

    for s in 0..tokens.len() {
        let prefix = tokens[..=s].to_vec();
        let mut ctx_p = rustral_core::ForwardCtx::new(backend, rustral_core::Mode::Inference);
        let step_logits = lm.model().forward(prefix, &mut ctx_p).expect("forward prefix");
        let step_flat = ops.tensor_to_vec(&step_logits).expect("step vec");
        let seq_len = ops.shape(&step_logits)[1];
        let offset_last = (seq_len - 1) * sl;
        for i in 0..sl {
            let a = step_flat[offset_last + i];
            let b = full_flat[s * sl + i];
            assert!(
                (a - b).abs() < 5e-5,
                "causal mismatch pos={s} dim={i}: prefix_last={a} full_row={b}"
            );
        }
    }
}
