//! Full forward logits vs Hugging Face Transformers reference (`transformers_golden_logits.json`).
//!
//! Regenerate the golden file with:
//! `scripts/llm/gen_tiny_llama_golden_transformers.py` (requires torch + transformers).

use std::collections::HashMap;
use std::path::PathBuf;

use rustral_core::{Backend, ForwardCtx, Mode};
use rustral_io::{MetaStateDict, TensorEntry};
use rustral_llm::{HfLlamaConfig, LlamaCausalLm};
use safetensors::Dtype;
use serde::Deserialize;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_llama")
}

fn tensor_entry(name: &str, shape: Vec<usize>, data: Vec<f32>) -> TensorEntry {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    TensorEntry { name: name.to_string(), shape, dtype: Dtype::F32, data: bytes }
}

fn synthetic_meta_state_dict(cfg: &HfLlamaConfig) -> MetaStateDict {
    let d = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let n_layer = cfg.num_hidden_layers;
    let head_dim = d / cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
    let kv_dim = n_kv * head_dim;

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
    tensors.insert(format!("{root}.norm.weight"), tensor_entry("norm.weight", vec![d], vec![1.0f32; d]));
    tensors.insert(
        "lm_head.weight".to_string(),
        tensor_entry("lm_head.weight", vec![vocab, d], vec![0.01f32; vocab * d]),
    );

    for layer in 0..n_layer {
        let h = format!("{root}.layers.{layer}");
        tensors.insert(format!("{h}.input_layernorm.weight"), tensor_entry("iln", vec![d], vec![1.0f32; d]));
        tensors.insert(
            format!("{h}.post_attention_layernorm.weight"),
            tensor_entry("paln", vec![d], vec![1.0f32; d]),
        );
        for (pn, rows) in [("q_proj", d), ("k_proj", kv_dim), ("v_proj", kv_dim), ("o_proj", d)] {
            tensors.insert(
                format!("{h}.self_attn.{pn}.weight"),
                tensor_entry(pn, vec![rows, d], vec![0.001f32; rows * d]),
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

#[derive(Deserialize)]
struct GoldenFile {
    input_ids: Vec<usize>,
    logits: Vec<f32>,
}

#[test]
fn llama_forward_matches_transformers_golden_logits() {
    let cfg = HfLlamaConfig::from_json_file(fixture_dir().join("config.json")).expect("fixture config");
    let meta = synthetic_meta_state_dict(&cfg);
    let (lm, _report) = LlamaCausalLm::from_hf_meta(&cfg, &meta, 42).expect("from_hf_meta");

    let golden_path = fixture_dir().join("transformers_golden_logits.json");
    let golden_raw = std::fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e} (regenerate with scripts/llm/gen_tiny_llama_golden_transformers.py)",
            golden_path.display()
        )
    });
    let golden: GoldenFile = serde_json::from_str(&golden_raw).expect("parse golden json");

    let backend = lm.backend();
    let ops = backend.ops();
    let mut ctx = ForwardCtx::new(backend, Mode::Inference);
    let logits_t = lm.model().forward(golden.input_ids.clone(), &mut ctx).expect("forward");
    let rust_logits = ops.tensor_to_vec(&logits_t).expect("vec");

    assert_eq!(rust_logits.len(), golden.logits.len(), "logits len");
    let mut max_abs = 0f32;
    for (i, (&a, &b)) in rust_logits.iter().zip(golden.logits.iter()).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(d < 2e-3, "logits differ at {i}: rust={a} golden={b} diff={d} (max_abs so far {max_abs})");
    }
}
